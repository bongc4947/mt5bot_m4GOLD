#ifndef MARKETHOURS_MQH
#define MARKETHOURS_MQH

#include "Defines.mqh"

//=============================================================================
// CMarketHours — EA-side session checker (backup when exec model unavailable).
// Mirrors python/market_hours.py logic for session flags.
// Used only if session_gate from exec model is unavailable.
//=============================================================================

class CMarketHours
{
private:
   //--- Convert datetime to UTC hour (approximate — assumes server time ≈ UTC)
   int UtcHour(datetime t)
   {
      MqlDateTime dt;
      TimeToStruct(t, dt);
      return dt.hour;
   }

   int UtcMinute(datetime t)
   {
      MqlDateTime dt;
      TimeToStruct(t, dt);
      return dt.min;
   }

   int DayOfWeek(datetime t)
   {
      MqlDateTime dt;
      TimeToStruct(t, dt);
      return dt.day_of_week;   // 0=Sun, 6=Sat
   }

public:
   //--- Returns true if the current time is in a primary trading session
   //    (London 08:00-17:00 UTC or NY 13:00-22:00 UTC)
   bool IsActiveSession(datetime t = 0)
   {
      if(t == 0) t = TimeCurrent();
      int h   = UtcHour(t);
      int dow = DayOfWeek(t);

      // Weekend exclusions
      if(dow == 0) return false;                    // Sunday
      if(dow == 6) return false;                    // Saturday
      if(dow == 5 && h >= 20) return false;         // Friday after 20:00

      // London
      if(h >= 8 && h < 17) return true;
      // NY
      if(h >= 13 && h < 22) return true;

      return false;
   }

   //--- Returns true if near daily rollover (21:00-22:00 UTC)
   bool IsRolloverWindow(datetime t = 0)
   {
      if(t == 0) t = TimeCurrent();
      int h = UtcHour(t);
      return (h == 21);
   }

   //--- Returns true if London-NY overlap (highest liquidity, 13:00-17:00 UTC)
   bool IsOverlapSession(datetime t = 0)
   {
      if(t == 0) t = TimeCurrent();
      int h   = UtcHour(t);
      int dow = DayOfWeek(t);
      if(dow == 0 || dow == 6) return false;
      return (h >= 13 && h < 17);
   }

   //--- Minutes until next London open from time t
   int MinutesToLondonOpen(datetime t = 0)
   {
      if(t == 0) t = TimeCurrent();
      int h = UtcHour(t);
      int m = UtcMinute(t);
      if(h < 8) return (8 - h) * 60 - m;
      if(h >= 17) return (24 + 8 - h) * 60 - m;
      return 0;  // already in London
   }

   //--- Returns a session gate score [0,1] as fallback
   float SessionGateFallback(datetime t = 0)
   {
      if(t == 0) t = TimeCurrent();
      if(IsRolloverWindow(t)) return 0.0f;
      if(!IsActiveSession(t)) return 0.1f;
      if(IsOverlapSession(t)) return 1.0f;
      return 0.7f;
   }
};

CMarketHours g_market_hours;

#endif // MARKETHOURS_MQH
