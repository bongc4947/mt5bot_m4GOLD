//+------------------------------------------------------------------+
//| MT5bot_m4Gold_MetaTrend.mq5 — the validated GOLD strategy.        |
//|                                                                    |
//| The honest, evidence-backed edge (docs/RESEARCH_FINDINGS.md):      |
//|   * DIRECTION  — a slow EMA 50/200 cross. Deterministic.           |
//|   * FILTER     — an XGBoost meta-gate (ONNX) that decides when to  |
//|                  trust the trend. Net PF ~1.41, purged-CV,         |
//|                  5/6 folds strongly positive, cost-aware.          |
//|                                                                    |
//| It enters WITH the trend only when the meta-gate's P(act) clears   |
//| the spec threshold, holds for a multi-hour move, and exits when    |
//| the trend flips, a trailing stop is hit, or a timeout elapses.     |
//|                                                                    |
//| Bundle -> MT5 Common Files:                                        |
//|   M4GOLD_METATREND_GOLD.onnx / _GOLD_spec.json                     |
//| Decisions are taken once per new M5 bar, on CLOSED bars only.      |
//+------------------------------------------------------------------+
#property copyright "MT5bot_m4Gold — MetaTrend"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>
#include "includes/MetaGate.mqh"

input long   InpMagic         = 49200;
input double InpBaseLot       = 0.01;
input bool   InpRespectDeploy = true;    // honour spec deploy=false
input bool   InpVerboseLog    = true;
// --- risk / exits --------------------------------------------------
input double InpSlAtr         = 3.0;     // initial stop, ATR multiples (wide)
input bool   InpUseTrailing   = true;
input double InpTrailStartAtr = 2.0;     // begin trailing after +2 ATR
input double InpTrailAtr      = 3.0;     // loose trail — ride the trend
input int    InpMaxHoldBars   = 288;     // force-exit after N M5 bars (~24h)
input bool   InpExitOnFlip    = true;    // exit when the EMA trend flips

CTrade   trade;
datetime g_last_m5 = 0;
double   g_atr14   = 0.0;

//+------------------------------------------------------------------+
int OnInit()
{
   string s = _Symbol;
   if(StringFind(s, "GOLD") < 0 && StringFind(s, "XAU") < 0)
      PrintFormat("[MetaTrend] WARNING: attached to %s — GOLD-only.", s);
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetTypeFillingBySymbol(s);

   if(!MG_Init())
   {
      Print("[MetaTrend] meta-gate failed to load — EA idle.");
      return INIT_SUCCEEDED;
   }
   if(InpRespectDeploy && !g_mg_deploy)
      Print("[MetaTrend] EA idle — spec deploy=false. Set "
            "InpRespectDeploy=false to demo an ungated model.");
   else
      PrintFormat("[MetaTrend] LIVE — version=%s, EMA %d/%d trend + meta-gate.",
                  g_mg_version, MG_EMA_FAST, MG_EMA_SLOW);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   MG_Release();
   PrintFormat("[MetaTrend] deinit reason=%d", reason);
}

//+------------------------------------------------------------------+
//| Position helpers (scoped to this EA's magic).                     |
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
//| Loose ATR trailing stop — ratchet only.                          |
//+------------------------------------------------------------------+
void _ManageTrail()
{
   if(!InpUseTrailing || g_atr14 <= 0) return;
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      bool is_buy = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY);
      double op = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl = PositionGetDouble(POSITION_SL);
      double cur = is_buy ? bid : ask;
      double prof_atr = (is_buy ? (cur-op) : (op-cur)) / g_atr14;
      if(prof_atr < InpTrailStartAtr) continue;
      double tr = is_buy ? (cur - InpTrailAtr*g_atr14)
                         : (cur + InpTrailAtr*g_atr14);
      tr = NormalizeDouble(tr, digits);
      if((is_buy  && (sl == 0 || tr > sl)) ||
         (!is_buy && (sl == 0 || tr < sl)))
         trade.PositionModify(t, tr, PositionGetDouble(POSITION_TP));
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   if(!g_mg_ready) return;
   if(InpRespectDeploy && !g_mg_deploy) return;

   _ManageTrail();                                  // every tick

   datetime cur = iTime(_Symbol, PERIOD_M5, 0);
   if(cur == g_last_m5) return;                     // decisions per new bar
   g_last_m5 = cur;

   g_atr14 = _Atr14();
   if(g_atr14 <= 0) return;

   int prim = MG_Primary();                         // +1 / -1
   int pos  = _PosDir();

   // --- exits for an open position --------------------------------
   if(pos != 0)
   {
      if(InpExitOnFlip && prim != 0 && prim != pos)
      {
         _CloseAll();
         if(InpVerboseLog) Print("[MetaTrend] trend flipped — closed");
         pos = 0;
      }
      else if(InpMaxHoldBars > 0 && _PosBars() >= InpMaxHoldBars)
      {
         _CloseAll();
         if(InpVerboseLog) Print("[MetaTrend] timeout — closed");
         pos = 0;
      }
      else
         return;                                    // hold; trail manages it
   }

   // --- entry: trend + meta-gate agreement ------------------------
   if(prim == 0) return;
   double pact = MG_ActProb();
   if(pact < 0)
   {
      if(InpVerboseLog) Print("[MetaTrend] meta-gate run failed — no entry");
      return;
   }
   if(InpVerboseLog)
      PrintFormat("[MetaTrend] trend=%d  P(act)=%.3f  thr=%.2f",
                  prim, pact, g_mg_actthr);
   if(pact < g_mg_actthr) return;                   // meta-gate vetoes

   double lot = InpBaseLot;
   double minlot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(lot < minlot) lot = minlot;
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   if(prim > 0)
   {
      double px = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl = NormalizeDouble(px - InpSlAtr*g_atr14, digits);
      bool ok = trade.Buy(lot, _Symbol, px, sl, 0, "MetaTrend");
      PrintFormat("[MetaTrend] %s BUY lot=%.2f sl=%.1fATR P(act)=%.3f rc=%d",
                  ok ? "FIRED" : "FAILED", lot, InpSlAtr, pact,
                  trade.ResultRetcode());
   }
   else
   {
      double px = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = NormalizeDouble(px + InpSlAtr*g_atr14, digits);
      bool ok = trade.Sell(lot, _Symbol, px, sl, 0, "MetaTrend");
      PrintFormat("[MetaTrend] %s SELL lot=%.2f sl=%.1fATR P(act)=%.3f rc=%d",
                  ok ? "FIRED" : "FAILED", lot, InpSlAtr, pact,
                  trade.ResultRetcode());
   }
}
