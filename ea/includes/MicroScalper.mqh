#ifndef MICROSCALPER_MQH
#define MICROSCALPER_MQH

#include "Defines.mqh"
#include "BrokerConfig.mqh"
#include "TradeStacker.mqh"

//=============================================================================
// Fast-close logic: close positions that hit 2×PipSize profit.
// Re-enters in same direction if signal still active.
//=============================================================================

class CMicroScalper
{
private:
   int m_magic;

   double ScalpTarget(const string canonical)
   {
      double pip = g_broker.PipSize(canonical);
      double spd = g_broker.SpreadPips(canonical);
      return MathMax(2.0 * pip, (spd + 1.0) * pip);
   }

public:
   void Init(int magic) { m_magic = magic; }

   //--- Scan open positions; close those that hit scalp target
   //    Returns number of positions closed
   int Update()
   {
      int closed = 0;
      int total = PositionsTotal();
      for(int i = total - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(!PositionSelectByTicket(ticket)) continue;
         if(PositionGetInteger(POSITION_MAGIC) != m_magic) continue;

         string bname = PositionGetString(POSITION_SYMBOL);
         string canonical = bname;
         for(int s = 0; s < g_sym_count; s++)
            if(g_sym_specs[s].broker_name == bname) { canonical = g_sym_specs[s].canonical; break; }

         double target   = ScalpTarget(canonical);
         int    type     = (int)PositionGetInteger(POSITION_TYPE);
         double profit   = PositionGetDouble(POSITION_PROFIT);
         double vol      = PositionGetDouble(POSITION_VOLUME);
         double open_p   = PositionGetDouble(POSITION_PRICE_OPEN);
         double cur_p    = (type == POSITION_TYPE_BUY)
                           ? SymbolInfoDouble(bname, SYMBOL_BID)
                           : SymbolInfoDouble(bname, SYMBOL_ASK);

         double dist_price = (type == POSITION_TYPE_BUY)
                             ? (cur_p - open_p)
                             : (open_p - cur_p);

         if(dist_price >= target && profit > 0.0)
         {
            CTrade trd;
            trd.SetExpertMagicNumber(m_magic);
            if(trd.PositionClose(ticket))
            {
               closed++;
               g_risk.CountScalp();
            }
         }
      }
      return closed;
   }
};

CMicroScalper g_scalper;

#endif // MICROSCALPER_MQH
