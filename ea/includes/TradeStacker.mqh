#ifndef TRADESTACKER_MQH
#define TRADESTACKER_MQH

#include "Defines.mqh"
#include "BrokerConfig.mqh"
#include "PositionSizer.mqh"

#include <Trade\Trade.mqh>

//=============================================================================
// Multi-entry position stacker. confidence tiers → 1–6 trades.
// All SL/TP and lot constraints from BrokerConfig.
//=============================================================================

CTrade g_trade;

class CTradeStacker
{
private:
   int m_magic;

   //--- How many entries for this confidence level
   int ConfidenceTier(double conf)
   {
      if(conf >= 0.90) return 6;
      if(conf >= 0.80) return 4;
      if(conf >= 0.70) return 3;
      if(conf >= 0.60) return 2;
      return 1;
   }

   //--- SL/TP prices
   bool CalcSLTP(const string canonical, int direction, double entry,
                 double &sl, double &tp)
   {
      double sl_dist = g_broker.CalcSLDist(canonical);
      double tp_dist = g_broker.CalcTPDist(canonical);
      int freeze     = g_broker.FreezeLevel(canonical);
      double pt      = g_broker.PointVal(canonical);

      // Ensure distance > freeze level
      double freeze_dist = (freeze + 2) * pt;
      sl_dist = MathMax(sl_dist, freeze_dist);
      tp_dist = MathMax(tp_dist, freeze_dist);

      if(direction > 0)
      {
         sl = entry - sl_dist;
         tp = entry + tp_dist;
      }
      else
      {
         sl = entry + sl_dist;
         tp = entry - tp_dist;
      }

      int digits = g_broker.Digits(canonical);
      sl = NormalizeDouble(sl, digits);
      tp = NormalizeDouble(tp, digits);
      return true;
   }

public:
   void Init(int magic)
   {
      m_magic = magic;
      g_trade.SetExpertMagicNumber(magic);
      g_trade.SetDeviationInPoints(10);
   }

   //--- Place stacked entries for a signal
   int PlaceEntries(const SSignal &sig, double meta_weight, double lot_scale = 1.0)
   {
      if(sig.skip) return 0;
      if(!g_broker.IsMarketOpen(sig.canonical, TimeCurrent())) return 0;
      if(!g_sizer.SpreadOK(sig.canonical)) return 0;
      if(!g_risk.CanOpenNew()) return 0;

      string bname   = g_broker.BrokerName(sig.canonical);
      int    tier    = ConfidenceTier(sig.confidence);
      double lot     = g_sizer.CalcLot(sig.canonical, sig.confidence,
                                        meta_weight, lot_scale);
      if(lot < g_broker.VolumeMin(sig.canonical)) return 0;

      double ask = SymbolInfoDouble(bname, SYMBOL_ASK);
      double bid = SymbolInfoDouble(bname, SYMBOL_BID);
      double entry = (sig.direction > 0) ? ask : bid;
      double sl = 0.0, tp = 0.0;
      CalcSLTP(sig.canonical, sig.direction, entry, sl, tp);

      int placed = 0;
      for(int t = 0; t < tier; t++)
      {
         // Each additional entry uses slightly reduced lot
         double tier_lot = g_broker.NormalizeLot(sig.canonical, lot * MathPow(0.8, t));
         if(tier_lot < g_broker.VolumeMin(sig.canonical)) break;

         bool ok;
         string comment_str = "HYDRA_" + sig.canonical + "_T" + IntegerToString(t+1);
         if(sig.direction > 0)
            ok = g_trade.Buy(tier_lot, bname, ask, sl, tp, comment_str);
         else
            ok = g_trade.Sell(tier_lot, bname, bid, sl, tp, comment_str);

         if(ok) { placed++; g_risk.CountOpen(); }
      }
      return placed;
   }
};

CTradeStacker g_stacker;

#endif // TRADESTACKER_MQH
