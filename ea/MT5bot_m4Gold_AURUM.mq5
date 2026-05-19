//+------------------------------------------------------------------+
//| MT5bot_m4Gold_AURUM.mq5 — AURUM v2 AI EA (GOLD-only).             |
//|                                                                    |
//| Trades GOLD from the AURUM ONNX stack (main net + meta gate +      |
//| conformal). v1.10 execution rebuild — see the notes below.         |
//|                                                                    |
//| WHY v1.10. The v1.00 EA closed the open position on EVERY bar the  |
//| conformal set was not a singleton (~80% of bars). That cut every   |
//| winner after ~2 bars, turned a 2:1 TP:SL design into a 0.87:1      |
//| realised ratio, and bled spread on 598 round-trips. Fixes:         |
//|   * a position is HELD once open — exits are SL / TP / trailing /  |
//|     timeout / a confident OPPOSITE signal. Loss of confidence is   |
//|     NOT an exit; it only blocks NEW entries.                       |
//|   * SL floored at InpMinSlAtr ATR and TP at InpMinRR x SL so a     |
//|     noise-tight stop can't get picked off by spread.               |
//|   * break-even + ATR trailing stop lock in open profit while       |
//|     letting trends run.                                            |
//|   * optional pyramiding — add units only into an ALREADY-PROFITABLE|
//|     position, capped at InpMaxStack.                               |
//|   * optional regime filter on entries.                             |
//+------------------------------------------------------------------+
#property copyright "MT5bot_m4Gold — AURUM v2"
#property version   "1.13"
#property strict

#include <Trade/Trade.mqh>
#include "includes/AurumAgent.mqh"

input long   InpMagic          = 49100;
input double InpBaseLot        = 0.01;   // base lot, scaled by AURUM lot_mult
input bool   InpRespectDeploy  = true;   // honour spec deploy=false
input bool   InpVerboseLog     = true;
// --- risk / exits --------------------------------------------------
input double InpMinSlAtr       = 1.5;    // floor on SL distance (ATR mult)
input double InpMinRR          = 2.2;    // TP >= InpMinRR * SL distance
input bool   InpUseBreakeven   = true;   // move SL to entry once in profit
input double InpBreakevenAtr   = 1.5;    // ...after price moves +1.5 ATR
input bool   InpUseTrailing    = true;   // loose ATR trailing stop
input double InpTrailStartAtr  = 2.5;    // begin trailing only well in profit
input double InpTrailAtr       = 3.0;    // trail loosely — catch reversals only
input int    InpMaxHoldBars    = 72;     // force-close after N M5 bars (0=off)
input bool   InpExitOnReverse  = true;   // close on a confident opposite call
// --- pyramiding ----------------------------------------------------
input int    InpMaxStack       = 2;      // max stacked units (1 = no stacking)
input double InpStackStepAtr   = 1.5;    // add a unit per +1.5 ATR of profit
// --- regime filter (which regime classes may OPEN a trade) ---------
//   0 trend-up   1 trend-down   2 range   3 high-vol
input string InpTradeRegimes   = "0,1,2,3";
// --- entry-quality filters (cut the buy-the-top / sell-the-bottom losers)
input bool   InpUseQuantAgree  = true;   // require the model's q50 to back the call
input bool   InpUseOverextGuard= true;   // skip entries chasing a range extreme
input int    InpRangeLookback  = 60;     // M5 bars used to measure the range
input double InpMaxEntryPct    = 0.85;   // skip longs above this %, shorts below 1-this

CTrade   trade;
datetime g_last_m5 = 0;
bool     g_deploy_ok = true;
double   g_atr = 0.0;           // ATR(14) on M5, refreshed each new bar

//+------------------------------------------------------------------+
int OnInit()
{
   string s = _Symbol;
   if(StringFind(s, "GOLD") < 0 && StringFind(s, "XAU") < 0)
      PrintFormat("[AURUM-EA] WARNING: attached to %s — AURUM is GOLD-only.", s);

   trade.SetExpertMagicNumber(InpMagic);
   trade.SetTypeFillingBySymbol(s);

   if(!AURUM_Init())
   {
      Print("[AURUM-EA] AURUM bundle failed to load — EA idle.");
      return INIT_SUCCEEDED;
   }
   g_deploy_ok = g_aurum_deploy;
   if(InpRespectDeploy && !g_deploy_ok)
   {
      if(!g_aurum_spec_ok)
         Print("[AURUM-EA] EA idle — spec file wrong/missing (see *** above).");
      else
         Print("[AURUM-EA] EA idle — spec deploy=false. Set "
               "InpRespectDeploy=false to demo an ungated model.");
   }
   else
      PrintFormat("[AURUM-EA] LIVE v1.13 — deploy=%s, closed-bar sync, "
                  "exits=SL/TP/trail, entry gate ON, maxStack=%d",
                  g_deploy_ok ? "true" : "false", InpMaxStack);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   AURUM_Release();
   PrintFormat("[AURUM-EA] deinit reason=%d", reason);
}

//+------------------------------------------------------------------+
//| Position bookkeeping (all helpers scoped to this EA's magic).     |
//+------------------------------------------------------------------+
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

// Net direction of open positions: +1 long / -1 short / 0 none-or-mixed.
int _PosNetDir()
{
   int dir = 0; bool seen = false;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      int d = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
      if(!seen) { dir = d; seen = true; }
      else if(dir != d) return 0;          // mixed (should not happen)
   }
   return dir;
}

// Bars since the OLDEST open position was opened.
int _OldestBars()
{
   datetime oldest = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      datetime ot = (datetime)PositionGetInteger(POSITION_TIME);
      if(oldest == 0 || ot < oldest) oldest = ot;
   }
   if(oldest == 0) return 0;
   return iBarShift(_Symbol, PERIOD_M5, oldest);
}

// Aggregate open profit expressed in ATR multiples (per 1 lot-ish proxy:
// use price distance of the average entry vs current price).
double _AggProfitAtr()
{
   if(g_atr <= 0) return 0.0;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sum = 0; int n = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      double op = PositionGetDouble(POSITION_PRICE_OPEN);
      double mv = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                  ? (bid - op) : (op - ask);
      sum += mv / g_atr; n++;
   }
   return (n > 0) ? sum / n : 0.0;
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
}

//+------------------------------------------------------------------+
//| Break-even + ATR trailing — ratchet SL, never loosen it.          |
//+------------------------------------------------------------------+
void _ManageOpen()
{
   if(g_atr <= 0) return;
   if(!InpUseBreakeven && !InpUseTrailing) return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double spread = ask - bid;

   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      bool   is_buy = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
      double op  = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl  = PositionGetDouble(POSITION_SL);
      double tp  = PositionGetDouble(POSITION_TP);
      double cur = is_buy ? bid : ask;
      double prof_atr = (is_buy ? (cur - op) : (op - cur)) / g_atr;
      double new_sl = sl;

      if(InpUseBreakeven && prof_atr >= InpBreakevenAtr)
      {
         double be = is_buy ? (op + spread) : (op - spread);
         if(is_buy)  new_sl = MathMax(new_sl, be);
         else        new_sl = (new_sl == 0) ? be : MathMin(new_sl, be);
      }
      if(InpUseTrailing && prof_atr >= InpTrailStartAtr)
      {
         double tr = is_buy ? (cur - InpTrailAtr * g_atr)
                            : (cur + InpTrailAtr * g_atr);
         if(is_buy)  new_sl = MathMax(new_sl, tr);
         else        new_sl = (new_sl == 0) ? tr : MathMin(new_sl, tr);
      }
      new_sl = NormalizeDouble(new_sl, digits);
      // Only modify on a meaningful tightening in our favour.
      if(new_sl > 0 && MathAbs(new_sl - sl) > spread &&
         ((is_buy && new_sl > sl) || (!is_buy && (sl == 0 || new_sl < sl))))
         trade.PositionModify(t, new_sl, tp);
   }
}

//+------------------------------------------------------------------+
double _AtrM5()
{
   MqlRates r[];
   // shift 1 — exclude the forming bar so ATR matches the closed-bar
   // basis the model and the SL/TP sizing were calibrated on.
   if(CopyRates(_Symbol, PERIOD_M5, 1, 20, r) < 16) return 0.0;
   double s = 0; int n = 0;
   for(int k = 1; k < ArraySize(r); k++)
   {
      double tr = MathMax(r[k].high - r[k].low,
                  MathMax(MathAbs(r[k].high - r[k-1].close),
                          MathAbs(r[k].low  - r[k-1].close)));
      s += tr; n++;
   }
   return (n > 0) ? s / n : 0.0;
}

bool _RegimeAllowed(int regime)
{
   return (StringFind(InpTradeRegimes, IntegerToString(regime)) >= 0);
}

//+------------------------------------------------------------------+
//| Entry-quality gate — rejects the trades that became the biggest   |
//| losers: entering AT a local extreme right as the move exhausts    |
//| (buying the top of a range, selling the bottom).                  |
//+------------------------------------------------------------------+
bool _EntryRejected(const AurumDecision &d)
{
   // (A) Head agreement — the model's median forward-return prediction
   //     (q50) must point the SAME way as the direction head. When the
   //     two heads disagree the signal is internally contradictory; at
   //     a turning point that is exactly what happens.
   if(InpUseQuantAgree)
   {
      if(d.direction > 0 && d.q50 <= 0.0)
      { if(InpVerboseLog) Print("[AURUM] entry skipped — q50<=0 vs long"); return true; }
      if(d.direction < 0 && d.q50 >= 0.0)
      { if(InpVerboseLog) Print("[AURUM] entry skipped — q50>=0 vs short"); return true; }
   }

   // (B) Overextension — do not BUY at the top of the recent range or
   //     SELL at the bottom, UNLESS the regime says it is a genuine
   //     trend in that direction (a breakout, which is allowed).
   if(InpUseOverextGuard)
   {
      MqlRates r[];
      int need = InpRangeLookback + 2;
      if(CopyRates(_Symbol, PERIOD_M5, 0, need, r) >= InpRangeLookback)
      {
         int n = ArraySize(r);
         double hi = -DBL_MAX, lo = DBL_MAX;
         for(int i = n - InpRangeLookback; i < n; i++)
         {
            if(r[i].high > hi) hi = r[i].high;
            if(r[i].low  < lo) lo = r[i].low;
         }
         double span = hi - lo;
         if(span > 0.0)
         {
            double px  = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double pos = (px - lo) / span;          // 0 = low, 1 = high
            if(d.direction > 0 && pos > InpMaxEntryPct && d.regime != 0)
            {
               if(InpVerboseLog)
                  PrintFormat("[AURUM] entry skipped — long at %.0f%% of "
                              "range (chasing the top)", 100.0 * pos);
               return true;
            }
            if(d.direction < 0 && pos < (1.0 - InpMaxEntryPct) && d.regime != 1)
            {
               if(InpVerboseLog)
                  PrintFormat("[AURUM] entry skipped — short at %.0f%% of "
                              "range (selling the bottom)", 100.0 * pos);
               return true;
            }
         }
      }
   }
   return false;
}

void _OpenPosition(const AurumDecision &d)
{
   double lot = NormalizeDouble(InpBaseLot * d.lot_mult, 2);
   double minlot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double lotstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(lot < minlot) lot = minlot;
   if(lotstep > 0) lot = MathFloor(lot / lotstep) * lotstep;

   // Floor the stop so spread/noise cannot pick it off; enforce min RR.
   double sl_atr = MathMax(d.sl_atr, InpMinSlAtr);
   double tp_atr = MathMax(d.tp_atr, InpMinRR * sl_atr);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   if(d.direction > 0)
   {
      double px = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = NormalizeDouble(px - sl_atr * g_atr, digits);
      double tp = NormalizeDouble(px + tp_atr * g_atr, digits);
      bool ok = trade.Buy(lot, _Symbol, px, sl, tp, "AURUM");
      PrintFormat("[AURUM] %s BUY lot=%.2f sl=%.1fATR tp=%.1fATR rc=%d",
                  ok ? "FIRED" : "FAILED", lot, sl_atr, tp_atr,
                  trade.ResultRetcode());
   }
   else
   {
      double px = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = NormalizeDouble(px + sl_atr * g_atr, digits);
      double tp = NormalizeDouble(px - tp_atr * g_atr, digits);
      bool ok = trade.Sell(lot, _Symbol, px, sl, tp, "AURUM");
      PrintFormat("[AURUM] %s SELL lot=%.2f sl=%.1fATR tp=%.1fATR rc=%d",
                  ok ? "FIRED" : "FAILED", lot, sl_atr, tp_atr,
                  trade.ResultRetcode());
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   if(InpRespectDeploy && !g_deploy_ok) return;
   if(!g_aurum_ready) return;

   // Trailing / break-even run on every tick so a fast move is captured.
   _ManageOpen();

   datetime cur = iTime(_Symbol, PERIOD_M5, 0);
   if(cur == g_last_m5) return;          // decisions only on a new M5 bar
   g_last_m5 = cur;

   g_atr = _AtrM5();
   if(g_atr <= 0) return;

   int pos_cnt = _PosCount();
   int pos_dir = _PosNetDir();

   // Timeout — a position that has gone nowhere for InpMaxHoldBars exits.
   if(pos_cnt > 0 && InpMaxHoldBars > 0 && _OldestBars() >= InpMaxHoldBars)
   {
      _CloseAll();
      if(InpVerboseLog) Print("[AURUM] timeout close");
      pos_cnt = 0; pos_dir = 0;
   }

   AurumDecision d = AURUM_Decide();
   if(InpVerboseLog)
      PrintFormat("[AURUM] dir=%d conf=%s pL=%.3f pS=%.3f regime=%d lot×=%.2f "
                  "pos=%d %s", d.direction, d.confident ? "Y" : "N",
                  d.p_long, d.p_short, d.regime, d.lot_mult, pos_cnt, d.reason);

   // NOT confident -> do nothing. Crucially, an OPEN position is NOT closed;
   // its SL / TP / trailing stop manage the exit.
   if(!d.confident || d.direction == 0)
      return;

   // Confident OPPOSITE call -> reverse (if enabled).
   if(pos_cnt > 0 && pos_dir != 0 && d.direction != pos_dir)
   {
      if(!InpExitOnReverse) return;       // hold; ignore the opposite signal
      _CloseAll();
      if(InpVerboseLog) Print("[AURUM] reverse — closed to flip");
      pos_cnt = 0; pos_dir = 0;
   }

   if(pos_cnt == 0)
   {
      // Fresh entry — gated by the regime filter AND the entry-quality
      // gate that rejects buy-the-top / sell-the-bottom reversals.
      if(!_RegimeAllowed(d.regime))
      {
         if(InpVerboseLog)
            PrintFormat("[AURUM] entry skipped — regime %d not in {%s}",
                        d.regime, InpTradeRegimes);
      }
      else if(_EntryRejected(d))
      {
         // reason already logged inside _EntryRejected
      }
      else
         _OpenPosition(d);
   }
   else if(d.direction == pos_dir && InpMaxStack > 1 && pos_cnt < InpMaxStack)
   {
      // Pyramid — add a unit ONLY when the stack is already in profit by
      // one more ATR step than the number of units already on.
      if(_AggProfitAtr() >= InpStackStepAtr * pos_cnt)
      {
         _OpenPosition(d);
         if(InpVerboseLog)
            PrintFormat("[AURUM] pyramid — unit %d added", pos_cnt + 1);
      }
   }
}
