#ifndef TRAILINGMANAGER_MQH
#define TRAILINGMANAGER_MQH

#include "Defines.mqh"
#include "BrokerConfig.mqh"
#include <Trade\Trade.mqh>

//=============================================================================
// Adaptive trailing SL. Called every 100 cycles.
// Floor SL distance from BrokerConfig (never below broker minimum).
//=============================================================================

class CTrailingManager
{
private:
   int    m_magic;

   //--- Minimum trailing distance in price (broker-aware)
   double MinTrailDist(const string canonical)
   {
      double pt  = g_broker.PointVal(canonical);
      double pip = g_broker.PipSize(canonical);
      double spd = g_broker.SpreadPips(canonical);
      double min_d = (g_broker.MinStopPoints(canonical) + 5) * pt;
      return MathMax(min_d, spd * pip * 1.5);
   }

public:
   void Init(int magic) { m_magic = magic; }

   //--- Move SL to breakeven
   void MoveToBE(ulong ticket, double open_price, int digits)
   {
      if(!PositionSelectByTicket(ticket)) return;
      double cur_sl = PositionGetDouble(POSITION_SL);
      double cur_tp = PositionGetDouble(POSITION_TP);
      int type = (int)PositionGetInteger(POSITION_TYPE);
      double be_sl = NormalizeDouble(open_price, digits);
      // Only move SL in favorable direction
      if((type == POSITION_TYPE_BUY  && be_sl > cur_sl) ||
         (type == POSITION_TYPE_SELL && be_sl < cur_sl))
      {
         CTrade trd; trd.SetExpertMagicNumber(m_magic);
         trd.PositionModify(ticket, be_sl, cur_tp);
      }
   }

   //--- Trail SL by explicit pips (from modify model)
   void TrailByPips(ulong ticket, const string bname, const string canonical,
                    float trail_pips, double pip_size, int digits)
   {
      if(!PositionSelectByTicket(ticket)) return;
      double cur_sl = PositionGetDouble(POSITION_SL);
      double cur_tp = PositionGetDouble(POSITION_TP);
      double open_px = PositionGetDouble(POSITION_PRICE_OPEN);
      int type = (int)PositionGetInteger(POSITION_TYPE);

      double trail_dist = trail_pips * pip_size;
      double min_dist   = MinTrailDist(canonical);
      trail_dist = MathMax(trail_dist, min_dist);

      double new_sl = cur_sl;
      if(type == POSITION_TYPE_BUY)
      {
         double bid = SymbolInfoDouble(bname, SYMBOL_BID);
         double candidate = NormalizeDouble(bid - trail_dist, digits);
         if(candidate > cur_sl && bid > open_px) new_sl = candidate;
      }
      else
      {
         double ask = SymbolInfoDouble(bname, SYMBOL_ASK);
         double candidate = NormalizeDouble(ask + trail_dist, digits);
         if(candidate < cur_sl && ask < open_px) new_sl = candidate;
      }
      if(new_sl != cur_sl)
      {
         CTrade trd; trd.SetExpertMagicNumber(m_magic);
         trd.PositionModify(ticket, new_sl, cur_tp);
      }
   }

   //--- Walk through all open HYDRA positions and update trailing SL
   void Update()
   {
      int total = PositionsTotal();
      for(int i = total - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(!PositionSelectByTicket(ticket)) continue;
         if(PositionGetInteger(POSITION_MAGIC) != m_magic) continue;

         string bname   = PositionGetString(POSITION_SYMBOL);
         // Resolve canonical from broker name via linear search
         string canonical = bname;
         for(int s = 0; s < g_sym_count; s++)
            if(g_sym_specs[s].broker_name == bname) { canonical = g_sym_specs[s].canonical; break; }

         double trail_dist = MinTrailDist(canonical);
         int    digits     = g_broker.Digits(canonical);
         int    type       = (int)PositionGetInteger(POSITION_TYPE);
         double cur_sl     = PositionGetDouble(POSITION_SL);
         double cur_tp     = PositionGetDouble(POSITION_TP);
         double open_price = PositionGetDouble(POSITION_PRICE_OPEN);

         double new_sl = cur_sl;

         if(type == POSITION_TYPE_BUY)
         {
            double bid = SymbolInfoDouble(bname, SYMBOL_BID);
            double candidate = NormalizeDouble(bid - trail_dist, digits);
            // Only move SL up, never down; only if in profit
            if(candidate > cur_sl && bid > open_price)
               new_sl = candidate;
         }
         else
         {
            double ask = SymbolInfoDouble(bname, SYMBOL_ASK);
            double candidate = NormalizeDouble(ask + trail_dist, digits);
            if(candidate < cur_sl && ask < open_price)
               new_sl = candidate;
         }

         if(new_sl != cur_sl)
         {
            CTrade trd;
            trd.SetExpertMagicNumber(m_magic);
            trd.PositionModify(ticket, new_sl, cur_tp);
         }
      }
   }
};

CTrailingManager g_trailing;

#endif // TRAILINGMANAGER_MQH
