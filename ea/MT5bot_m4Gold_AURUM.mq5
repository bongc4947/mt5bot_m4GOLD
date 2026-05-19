//+------------------------------------------------------------------+
//| MT5bot_m4Gold_AURUM.mq5 — AURUM v2 AI EA (GOLD-only).             |
//|                                                                    |
//| Trades GOLD purely from the AURUM ONNX stack:                      |
//|   - main net : multi-timeframe patch-transformer (M5/M15/H1)       |
//|   - meta gate: XGBoost P(act) filter                               |
//|   - conformal: trade only on a singleton confidence set            |
//|   - sizing   : quantile-Kelly + vol target lot multiplier          |
//|                                                                    |
//| Drop the AURUM bundle into MT5 Common Files:                       |
//|   M4GOLD_AURUM_GOLD.onnx / _META_GOLD.onnx / _GOLD_spec.json       |
//| then attach this EA to a GOLD chart. If the spec has deploy=false  |
//| the EA stays idle (training did not clear the deploy gate).        |
//|                                                                    |
//| Decisions are taken once per new M5 bar.                           |
//+------------------------------------------------------------------+
#property copyright "MT5bot_m4Gold — AURUM v2"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>
#include "includes/AurumAgent.mqh"

input long   InpMagic        = 49100;
input double InpBaseLot      = 0.01;     // multiplied by AURUM lot_mult
input bool   InpRespectDeploy = true;    // honour spec deploy=false
input bool   InpVerboseLog   = true;

CTrade   trade;
datetime g_last_m5 = 0;
bool     g_deploy_ok = true;

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
      return INIT_SUCCEEDED;     // stay attached but dormant
   }
   // Single source of truth — AURUM_Init already parsed the spec.
   g_deploy_ok = g_aurum_deploy;
   if(InpRespectDeploy && !g_deploy_ok)
   {
      if(!g_aurum_spec_ok)
         Print("[AURUM-EA] EA idle — the spec file is wrong/missing (see the "
               "*** WRONG SPEC FILE *** line above). Stage the correct "
               "M4GOLD_AURUM_GOLD_spec.json and reload.");
      else
         Print("[AURUM-EA] EA idle — spec deploy=false. This model did not "
               "clear the deploy gate; train a qualifying model or set "
               "InpRespectDeploy=false to demo it anyway.");
   }
   else
      PrintFormat("[AURUM-EA] LIVE — deploy=%s respect=%s, trading enabled.",
                  g_deploy_ok ? "true" : "false",
                  InpRespectDeploy ? "true" : "false");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   AURUM_Release();
   PrintFormat("[AURUM-EA] deinit reason=%d", reason);
}

//+------------------------------------------------------------------+
bool _HasPosition()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      return true;
   }
   return false;
}

void _ClosePosition()
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
void OnTick()
{
   if(InpRespectDeploy && !g_deploy_ok) return;

   datetime cur = iTime(_Symbol, PERIOD_M5, 0);
   if(cur == g_last_m5) return;
   g_last_m5 = cur;

   AurumDecision d = AURUM_Decide();
   if(InpVerboseLog)
      PrintFormat("[AURUM] dir=%d conf=%s pL=%.3f pS=%.3f regime=%d lot×=%.2f %s",
                  d.direction, d.confident ? "Y" : "N", d.p_long, d.p_short,
                  d.regime, d.lot_mult, d.reason);

   bool have = _HasPosition();

   // No confident signal: flatten any open position and wait.
   if(!d.confident || d.direction == 0)
   {
      if(have) { _ClosePosition(); if(InpVerboseLog) Print("[AURUM] flat -> closed"); }
      return;
   }

   // Confident: if an opposite position is open, flip it.
   if(have)
   {
      long ptype = -1;
      for(int i = 0; i < PositionsTotal(); i++)
      {
         ulong t = PositionGetTicket(i);
         if(t == 0 || !PositionSelectByTicket(t)) continue;
         if((long)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
         if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
         ptype = (long)PositionGetInteger(POSITION_TYPE);
      }
      bool aligned = (d.direction > 0 && ptype == POSITION_TYPE_BUY)
                  || (d.direction < 0 && ptype == POSITION_TYPE_SELL);
      if(aligned) return;          // already positioned correctly
      _ClosePosition();
   }

   double lot = NormalizeDouble(InpBaseLot * d.lot_mult, 2);
   double minlot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(lot < minlot) lot = minlot;

   double atr = 0;
   {
      MqlRates r[];
      if(CopyRates(_Symbol, PERIOD_M5, 0, 20, r) >= 16)
      {
         double s = 0; int n = 0;
         for(int k = 1; k < ArraySize(r); k++)
         {
            double tr = MathMax(r[k].high - r[k].low,
                        MathMax(MathAbs(r[k].high - r[k-1].close),
                                MathAbs(r[k].low  - r[k-1].close)));
            s += tr; n++;
         }
         atr = s / MathMax(1, n);
      }
   }
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double price, sl, tp;
   if(d.direction > 0)
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = NormalizeDouble(price - d.sl_atr * atr, digits);
      tp = NormalizeDouble(price + d.tp_atr * atr, digits);
      bool ok = trade.Buy(lot, _Symbol, price, sl, tp, "AURUM");
      PrintFormat("[AURUM] %s BUY lot=%.2f sl=%.2f tp=%.2f rc=%d",
                  ok ? "FIRED" : "FAILED", lot, sl, tp, trade.ResultRetcode());
   }
   else
   {
      price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = NormalizeDouble(price + d.sl_atr * atr, digits);
      tp = NormalizeDouble(price - d.tp_atr * atr, digits);
      bool ok = trade.Sell(lot, _Symbol, price, sl, tp, "AURUM");
      PrintFormat("[AURUM] %s SELL lot=%.2f sl=%.2f tp=%.2f rc=%d",
                  ok ? "FIRED" : "FAILED", lot, sl, tp, trade.ResultRetcode());
   }
}
