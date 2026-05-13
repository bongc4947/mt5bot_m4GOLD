#ifndef LIMITORDERMANAGER_MQH
#define LIMITORDERMANAGER_MQH

#include "Defines.mqh"
#include "BrokerConfig.mqh"

#include <Trade\Trade.mqh>

//=============================================================================
// CLimitOrderManager — places and tracks limit entries from timing < threshold.
// When timing score is in [TIMING_WAIT_MIN, TIMING_THRESHOLD], places a
// buy/sell limit 0.3×ATR offset from current close. Expires after LIMIT_EXPIRY_BARS.
//=============================================================================

#define MAX_PENDING_LIMITS 32

struct SPendingLimit
{
   ulong    ticket;
   string   canonical;
   int      direction;
   datetime placed_time;
   int      expiry_bars;
   float    sl_pips;
   float    tp_pips;
   float    lot;
   bool     active;
};

class CLimitOrderManager
{
private:
   int    m_magic;
   SPendingLimit m_limits[MAX_PENDING_LIMITS];
   int    m_count;

   int FindSlot()
   {
      for(int i = 0; i < MAX_PENDING_LIMITS; i++)
         if(!m_limits[i].active) return i;
      return -1;
   }

   double PipsToPrice(const string canonical, double pips)
   {
      return pips * g_broker.PipSize(canonical);
   }

public:
   void Init(int magic)
   {
      m_magic = magic;
      m_count = 0;
      for(int i = 0; i < MAX_PENDING_LIMITS; i++)
         m_limits[i].active = false;
   }

   //--- Place a limit order offset_pips from current close, expiry_bars TTL
   bool PlaceLimitEntry(const string canonical, int direction,
                        double entry_close, float offset_pips,
                        float sl_pips, float tp_pips,
                        float lot, int expiry_bars = LIMIT_EXPIRY_BARS)
   {
      int slot = FindSlot();
      if(slot < 0) return false;

      string bname   = g_broker.BrokerName(canonical);
      double pip     = g_broker.PipSize(canonical);
      int    digits  = g_broker.Digits(canonical);

      double offset  = offset_pips * pip;
      double limit_px;
      if(direction > 0)
         limit_px = NormalizeDouble(entry_close - offset, digits);  // buy limit below
      else
         limit_px = NormalizeDouble(entry_close + offset, digits);  // sell limit above

      double sl_dist = sl_pips * pip;
      double tp_dist = tp_pips * pip;
      double sl_px   = (direction > 0) ? limit_px - sl_dist : limit_px + sl_dist;
      double tp_px   = (direction > 0) ? limit_px + tp_dist : limit_px - tp_dist;

      sl_px = NormalizeDouble(sl_px, digits);
      tp_px = NormalizeDouble(tp_px, digits);

      CTrade trd;
      trd.SetExpertMagicNumber(m_magic);
      trd.SetDeviationInPoints(10);

      bool ok;
      datetime expiry_time = TimeCurrent() + expiry_bars * 5 * 60;  // M5 bars

      if(direction > 0)
         ok = trd.BuyLimit(lot, limit_px, bname, sl_px, tp_px, ORDER_TIME_SPECIFIED,
                           expiry_time, "HYDRA4_LMT_" + canonical);
      else
         ok = trd.SellLimit(lot, limit_px, bname, sl_px, tp_px, ORDER_TIME_SPECIFIED,
                            expiry_time, "HYDRA4_LMT_" + canonical);

      if(ok)
      {
         m_limits[slot].ticket      = trd.ResultOrder();
         m_limits[slot].canonical   = canonical;
         m_limits[slot].direction   = direction;
         m_limits[slot].placed_time = TimeCurrent();
         m_limits[slot].expiry_bars = expiry_bars;
         m_limits[slot].sl_pips     = sl_pips;
         m_limits[slot].tp_pips     = tp_pips;
         m_limits[slot].lot         = lot;
         m_limits[slot].active      = true;
         PrintFormat("LimitOrderManager: placed %s limit@%.5f  sl=%.1f  tp=%.1f",
                     canonical, limit_px, sl_pips, tp_pips);
      }
      return ok;
   }

   //--- Cancel expired or already-filled limits
   void Update()
   {
      datetime now = TimeCurrent();
      for(int i = 0; i < MAX_PENDING_LIMITS; i++)
      {
         if(!m_limits[i].active) continue;

         // Check if still pending
         if(!OrderSelect(m_limits[i].ticket)) { m_limits[i].active = false; continue; }
         if(OrderGetInteger(ORDER_STATE) != ORDER_STATE_PLACED)
            { m_limits[i].active = false; continue; }

         // Manual expiry check (belt-and-suspenders beyond broker expiry)
         int bars_elapsed = (int)((now - m_limits[i].placed_time) / (5 * 60));
         if(bars_elapsed >= m_limits[i].expiry_bars + 1)
         {
            CTrade trd;
            trd.SetExpertMagicNumber(m_magic);
            if(trd.OrderDelete(m_limits[i].ticket))
            {
               PrintFormat("LimitOrderManager: expired limit cancelled %s", m_limits[i].canonical);
               m_limits[i].active = false;
            }
         }
      }
   }

   //--- Return true if a pending limit already exists for this canonical
   bool HasPending(const string canonical) const
   {
      for(int i = 0; i < MAX_PENDING_LIMITS; i++)
         if(m_limits[i].active && m_limits[i].canonical == canonical)
            return true;
      return false;
   }
};

CLimitOrderManager g_limit_mgr;

#endif // LIMITORDERMANAGER_MQH
