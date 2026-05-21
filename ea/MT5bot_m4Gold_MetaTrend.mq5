//+------------------------------------------------------------------+
//| MT5bot_m4Gold_MetaTrend.mq5  v1.20                                |
//|                                                                    |
//| MetaTrend - GOLD-only slow trend + ML meta-gate + (optional)       |
//| 7-quantile distributional forecast.                                |
//|                                                                    |
//| EXIT ENGINE (v1.20 - addresses the "no goal per trade" critique)   |
//|   - Initial SL    : InpSlAtr ATR below/above entry (wide)          |
//|   - Breakeven move: at +InpBreakevenAtr ATR profit, ratchet SL to  |
//|                     entry. Removes downside on trades that worked  |
//|                     even a little. Defaults ON.                    |
//|   - Partial close : at +InpPartialAtr ATR profit, close            |
//|                     InpPartialPct of the position. Defaults OFF -  |
//|                     opt in.                                        |
//|   - ATR trailing  : starts at +InpTrailStartAtr ATR, trails at     |
//|                     InpTrailAtr behind. Lets winners run.          |
//|   - Optional TP   : if InpTpAtr > 0, hard TP at that multiple.     |
//|                     Defaults 0 (off).                              |
//|   - Trend flip    : exit on EMA cross direction reversal.          |
//|   - Timeout       : exit after InpMaxHoldBars (24h default).       |
//|                                                                    |
//| QUANTILE-AWARE DECISION (v1.20)                                    |
//|   When 7 quantile ONNX heads load successfully (auto-detected      |
//|   from M4GOLD_QUANTILE_*_GOLD.onnx in Common Files), entries are   |
//|   filtered further:                                                |
//|     - q50 sign must agree with the primary EMA trend               |
//|     - q10 left-tail veto: refuse trade if q10 < -InpQ10VetoAtr*ATR |
//|     - Kelly fractional sizing: lot = base * clip(mu/sigma^2, ..,)  |
//|     - Dynamic initial SL placed at q05 + InpSlBufferAtr*ATR        |
//|   If quantile heads are absent, the EA falls back to v1.10 logic   |
//|   (pure binary gate). Zero disruption to existing deployment.      |
//+------------------------------------------------------------------+
#property copyright "MT5bot_m4Gold - MetaTrend v1.20"
#property version   "1.20"
#property strict

#include <Trade/Trade.mqh>
#include "includes/MetaGate.mqh"

input long   InpMagic           = 49200;
input double InpBaseLot         = 0.01;
input bool   InpRespectDeploy   = true;
input bool   InpVerboseLog      = true;
// --- risk / exits --------------------------------------------------
input double InpSlAtr           = 3.0;     // initial stop, ATR multiples
input double InpTpAtr           = 0.0;     // hard TP in ATR (0 = off)
input bool   InpUseBreakeven    = true;    // move SL to entry at +BreakevenAtr
input double InpBreakevenAtr    = 1.0;     // breakeven trigger, ATR multiples
input double InpBreakevenBuffer = 0.05;    // extra ATR buffer above entry on BE
input bool   InpUsePartialClose = false;   // close part of stack at +PartialAtr
input double InpPartialAtr      = 1.5;     // partial-close trigger, ATR mult
input double InpPartialPct      = 0.5;     // fraction to close (0..1)
input bool   InpUseTrailing     = true;
input double InpTrailStartAtr   = 2.0;     // begin trailing after +N ATR
input double InpTrailAtr        = 3.0;     // trail distance behind price
input int    InpMaxHoldBars     = 288;     // ~24h
input bool   InpExitOnFlip      = true;
// --- pyramiding ----------------------------------------------------
input int    InpMaxStack        = 3;
input double InpStackStepAtr    = 1.0;
// --- quantile gate (v1.20) -----------------------------------------
// Default OFF: A/B-tested 2026-05-21, the quantile-driven dynamic SL widens
// stops on losing trades and burns 14.5% of the validated edge. The 7 ONNX
// heads stay in Common Files as research artifacts; flip this on only to
// experiment after the dynamic-SL path is reworked.
input bool   InpUseQuantiles    = false;   // auto-disables if ONNX missing
input double InpQ10VetoAtr      = 2.5;     // veto if q10 < -X * ATR (tuned-down)
input bool   InpUseQ50Filter    = false;   // require q50 sign agrees w/ primary
input double InpKellyFraction   = 0.25;    // quarter-Kelly
input double InpKellyMin        = 0.5;     // lot multiplier floor
input double InpKellyMax        = 2.0;     // lot multiplier ceiling (tuned-down)
input double InpSlBufferAtr     = 0.5;     // extra buffer beyond q05 SL

CTrade   trade;
datetime g_last_m5         = 0;
double   g_atr14           = 0.0;
double   g_last_pact       = 0.0;
double   g_last_q[7];                       // last quantile vector
bool     g_last_q_valid    = false;
ulong    g_partialed_tkts[64];              // tickets that have been partial-closed
int      g_n_partialed     = 0;

bool _AlreadyPartialed(ulong t)
{
   for(int i = 0; i < g_n_partialed; i++) if(g_partialed_tkts[i] == t) return true;
   return false;
}
void _MarkPartialed(ulong t)
{
   if(g_n_partialed < 64) g_partialed_tkts[g_n_partialed++] = t;
}

//+------------------------------------------------------------------+
int OnInit()
{
   string s = _Symbol;
   if(StringFind(s, "GOLD") < 0 && StringFind(s, "XAU") < 0)
      PrintFormat("[MetaTrend] WARNING: attached to %s - GOLD-only.", s);
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetTypeFillingBySymbol(s);

   if(!MG_Init())
   {
      Print("[MetaTrend] meta-gate failed to load - EA idle.");
      return INIT_SUCCEEDED;
   }
   if(InpRespectDeploy && !g_mg_deploy)
      Print("[MetaTrend] EA idle - spec deploy=false. Set "
            "InpRespectDeploy=false to demo an ungated model.");
   else
      PrintFormat("[MetaTrend] LIVE v1.20 - version=%s  EMA %d/%d  "
                  "maxStack=%d  breakeven=%s@%.1fATR  partial=%s@%.1fATR x%.0f%%  "
                  "tp=%.1fATR  quantiles=%s",
                  g_mg_version, MG_EMA_FAST, MG_EMA_SLOW, InpMaxStack,
                  InpUseBreakeven ? "ON" : "off", InpBreakevenAtr,
                  InpUsePartialClose ? "ON" : "off", InpPartialAtr,
                  InpPartialPct * 100.0, InpTpAtr,
                  (InpUseQuantiles && g_mg_q_ready) ? "ON" : "off");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   MG_Release();
   PrintFormat("[MetaTrend] deinit reason=%d", reason);
}

//+------------------------------------------------------------------+
//| Position helpers - scoped to this EA's magic.                     |
//+------------------------------------------------------------------+
int _PosDir()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      return (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
   }
   return 0;
}

int _PosBars()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      return iBarShift(_Symbol, PERIOD_M5,
                       (datetime)PositionGetInteger(POSITION_TIME));
   }
   return 0;
}

void _CloseAll()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      trade.PositionClose(t);
   }
   g_n_partialed = 0;
}

int _PosCount()
{
   int n = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      n++;
   }
   return n;
}

double _AggProfitAtr()
{
   if(g_atr14 <= 0) return 0.0;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sum = 0; int n = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      bool is_buy = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
      double op = PositionGetDouble(POSITION_PRICE_OPEN);
      double mv = is_buy ? (bid - op) : (op - ask);
      sum += mv / g_atr14; n++;
   }
   return (n > 0) ? sum / n : 0.0;
}

//+------------------------------------------------------------------+
//| Quantile-aware lot sizing. Returns the lot multiplier for the     |
//| Kelly-fractional bet. Falls back to 1.0 if quantiles are off.     |
//+------------------------------------------------------------------+
double _KellyLotMult(int dir)
{
   if(!InpUseQuantiles || !g_last_q_valid) return 1.0;
   double mu    = g_last_q[3];                     // q50 (log-return)
   double sigma = MathMax(1e-9, (g_last_q[5] - g_last_q[1]) / 2.56);
   double kelly = (mu / (sigma * sigma + 1e-12)) * InpKellyFraction;
   double mult  = MathAbs(kelly);
   if(mult < InpKellyMin) mult = InpKellyMin;
   if(mult > InpKellyMax) mult = InpKellyMax;
   return mult;
}

//+------------------------------------------------------------------+
//| Open one unit in `dir`. Uses quantile q05 for dynamic SL when     |
//| available, falls back to ATR-multiple SL otherwise. Sets TP only  |
//| if InpTpAtr > 0.                                                  |
//+------------------------------------------------------------------+
void _OpenUnit(int dir, double pact)
{
   double lot = InpBaseLot * _KellyLotMult(dir);
   double minlot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lotstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(lot < minlot) lot = minlot;
   if(lotstep > 0) lot = MathFloor(lot / lotstep) * lotstep;
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double atr_sl = InpSlAtr * g_atr14;

   // dynamic SL: place just beyond q05 of forecast distribution
   double sl_dist_atr = InpSlAtr;
   if(InpUseQuantiles && g_last_q_valid)
   {
      // convert q05 (log-return) to price-points: |price * (exp(q05)-1)|
      double price = (dir > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                : SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double q05_pts = MathAbs(price * (MathExp(g_last_q[0]) - 1.0));
      // long: SL = q05_pts below entry  (q05 is the "bad 5%" downside)
      // short: similar - q05 of forward returns when going short is the
      //                upside-against-us; we use the same magnitude.
      double dyn_sl = q05_pts + InpSlBufferAtr * g_atr14;
      // sanity floor/ceiling vs ATR
      double atr_min = 1.0 * g_atr14;       // never tighter than 1 ATR
      double atr_max = 5.0 * g_atr14;       // never wider than 5 ATR
      if(dyn_sl < atr_min) dyn_sl = atr_min;
      if(dyn_sl > atr_max) dyn_sl = atr_max;
      atr_sl = dyn_sl;
      sl_dist_atr = atr_sl / g_atr14;
   }
   double atr_tp = InpTpAtr * g_atr14;

   if(dir > 0)
   {
      double px = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = NormalizeDouble(px - atr_sl, digits);
      double tp = (InpTpAtr > 0) ? NormalizeDouble(px + atr_tp, digits) : 0.0;
      bool ok = trade.Buy(lot, _Symbol, px, sl, tp, "MetaTrend");
      PrintFormat("[MetaTrend] %s BUY lot=%.2f x%.2f sl=%.2fATR tp=%.1fATR P(act)=%.3f rc=%d",
                  ok ? "FIRED" : "FAILED", lot, _KellyLotMult(dir), sl_dist_atr,
                  InpTpAtr, pact, trade.ResultRetcode());
   }
   else
   {
      double px = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = NormalizeDouble(px + atr_sl, digits);
      double tp = (InpTpAtr > 0) ? NormalizeDouble(px - atr_tp, digits) : 0.0;
      bool ok = trade.Sell(lot, _Symbol, px, sl, tp, "MetaTrend");
      PrintFormat("[MetaTrend] %s SELL lot=%.2f x%.2f sl=%.2fATR tp=%.1fATR P(act)=%.3f rc=%d",
                  ok ? "FIRED" : "FAILED", lot, _KellyLotMult(dir), sl_dist_atr,
                  InpTpAtr, pact, trade.ResultRetcode());
   }
}

//+------------------------------------------------------------------+
//| ATR(14) on closed M5 bars.                                        |
//+------------------------------------------------------------------+
double _Atr14()
{
   MqlRates r[];
   if(CopyRates(_Symbol, PERIOD_M5, 1, 16, r) < 16) return 0.0;
   double s = 0;
   for(int k = 1; k < ArraySize(r); k++)
      s += MathMax(r[k].high-r[k].low,
           MathMax(MathAbs(r[k].high-r[k-1].close),
                   MathAbs(r[k].low-r[k-1].close)));
   return s / (ArraySize(r) - 1);
}

//+------------------------------------------------------------------+
//| Per-tick position management: breakeven move, partial close,     |
//| then the ATR trail.                                               |
//+------------------------------------------------------------------+
void _ManagePositions()
{
   if(g_atr14 <= 0) return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      bool is_buy = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
      double op  = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl  = PositionGetDouble(POSITION_SL);
      double tp  = PositionGetDouble(POSITION_TP);
      double cur = is_buy ? bid : ask;
      double prof_atr = (is_buy ? (cur - op) : (op - cur)) / g_atr14;

      // === breakeven move - ratchet SL to entry once we've earned it ===
      if(InpUseBreakeven && prof_atr >= InpBreakevenAtr)
      {
         double be = is_buy ? (op + InpBreakevenBuffer * g_atr14)
                             : (op - InpBreakevenBuffer * g_atr14);
         be = NormalizeDouble(be, digits);
         if((is_buy  && (sl == 0 || be > sl)) ||
            (!is_buy && (sl == 0 || be < sl)))
         {
            if(trade.PositionModify(t, be, tp))
            {
               sl = be;
               if(InpVerboseLog)
                  PrintFormat("[MetaTrend] breakeven moved  ticket=%I64u prof=%.2fATR sl->%.2f",
                              t, prof_atr, be);
            }
         }
      }

      // === partial close - lock in some cash on trades that worked ===
      if(InpUsePartialClose && prof_atr >= InpPartialAtr && !_AlreadyPartialed(t))
      {
         double vol = PositionGetDouble(POSITION_VOLUME);
         double minlot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
         double step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
         double cut = vol * InpPartialPct;
         if(step > 0) cut = MathFloor(cut / step) * step;
         if(cut >= minlot && cut < vol)
         {
            if(trade.PositionClosePartial(t, cut))
            {
               _MarkPartialed(t);
               if(InpVerboseLog)
                  PrintFormat("[MetaTrend] partial close  ticket=%I64u cut=%.2f of %.2f prof=%.2fATR",
                              t, cut, vol, prof_atr);
            }
         }
      }

      // === ATR trail - the original profit-runner mechanism ===
      if(InpUseTrailing && prof_atr >= InpTrailStartAtr)
      {
         double tr = is_buy ? (cur - InpTrailAtr * g_atr14)
                             : (cur + InpTrailAtr * g_atr14);
         tr = NormalizeDouble(tr, digits);
         if((is_buy  && (sl == 0 || tr > sl)) ||
            (!is_buy && (sl == 0 || tr < sl)))
            trade.PositionModify(t, tr, tp);
      }
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   if(!g_mg_ready) return;
   if(InpRespectDeploy && !g_mg_deploy) return;

   _ManagePositions();                              // every tick

   datetime cur = iTime(_Symbol, PERIOD_M5, 0);
   if(cur == g_last_m5) return;                     // decisions per new bar
   g_last_m5 = cur;

   g_atr14 = _Atr14();
   if(g_atr14 <= 0) return;

   int prim = MG_Primary();                         // +1 / -1
   int pos  = _PosDir();

   // === exits ======================================================
   if(pos != 0)
   {
      if(InpExitOnFlip && prim != 0 && prim != pos)
      {
         _CloseAll();
         if(InpVerboseLog) Print("[MetaTrend] trend flipped - closed stack");
         pos = 0;
      }
      else if(InpMaxHoldBars > 0 && _PosBars() >= InpMaxHoldBars)
      {
         _CloseAll();
         if(InpVerboseLog) Print("[MetaTrend] timeout - closed stack");
         pos = 0;
      }
   }

   if(prim == 0) return;

   // === fresh quantile forecast for this bar =======================
   g_last_q_valid = false;
   if(InpUseQuantiles && g_mg_q_ready)
   {
      double q[];
      if(MG_QuantileForecast(q) && ArraySize(q) == 7)
      {
         for(int i = 0; i < 7; i++) g_last_q[i] = q[i];
         g_last_q_valid = true;
      }
   }

   // === meta-gate P(act) ==========================================
   double pact = MG_ActProb();
   g_last_pact = pact;
   if(pact < 0)
   {
      if(InpVerboseLog) Print("[MetaTrend] meta-gate run failed - no action");
      return;
   }

   // === quantile-aware filters ====================================
   bool tail_veto = false, q50_disagree = false;
   if(g_last_q_valid)
   {
      // tail veto: q10 (log-return) implies points = price * (exp(q10) - 1)
      double price = (prim > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                 : SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double q10_pts = price * (MathExp(g_last_q[1]) - 1.0);   // signed
      if(prim > 0 && q10_pts < -InpQ10VetoAtr * g_atr14) tail_veto = true;
      if(prim < 0)
      {
         // for shorts the "bad tail" is the upper tail q90 going against us
         double q90_pts = price * (MathExp(g_last_q[5]) - 1.0);
         if(q90_pts > InpQ10VetoAtr * g_atr14) tail_veto = true;
      }
      // q50 sign must agree with primary - optional, default off
      if(InpUseQ50Filter &&
         ((g_last_q[3] > 0 && prim < 0) || (g_last_q[3] < 0 && prim > 0)))
         q50_disagree = true;
   }

   if(InpVerboseLog)
   {
      if(g_last_q_valid)
         PrintFormat("[MetaTrend] trend=%d  P(act)=%.3f  thr=%.2f  units=%d  "
                     "q05=%.5f q50=%.5f q95=%.5f  veto=%s q50disagree=%s",
                     prim, pact, g_mg_actthr, _PosCount(),
                     g_last_q[0], g_last_q[3], g_last_q[6],
                     tail_veto ? "Y" : "N", q50_disagree ? "Y" : "N");
      else
         PrintFormat("[MetaTrend] trend=%d  P(act)=%.3f  thr=%.2f  units=%d",
                     prim, pact, g_mg_actthr, _PosCount());
   }

   if(pact < g_mg_actthr) return;                   // meta-gate vetoes
   if(tail_veto)
   {
      if(InpVerboseLog) Print("[MetaTrend] q10 tail veto - skipping entry");
      return;
   }
   if(q50_disagree)
   {
      if(InpVerboseLog) Print("[MetaTrend] q50 disagrees with primary - skipping");
      return;
   }

   int cnt = _PosCount();
   if(cnt == 0)
   {
      _OpenUnit(prim, pact);
   }
   else if(prim == pos && InpMaxStack > 1 && cnt < InpMaxStack)
   {
      if(_AggProfitAtr() >= InpStackStepAtr * cnt)
      {
         _OpenUnit(prim, pact);
         if(InpVerboseLog)
            PrintFormat("[MetaTrend] pyramid - unit %d added", cnt + 1);
      }
   }
}
