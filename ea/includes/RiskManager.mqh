#ifndef RISKMANAGER_MQH
#define RISKMANAGER_MQH

#include "Defines.mqh"

//=============================================================================
// Global risk guard: daily DD limits, daily counters, shutdown logic.
//=============================================================================

class CRiskManager
{
private:
   double m_balance_sod;    // balance at start of day
   double m_daily_dd_pct;
   int    m_opened_today;
   int    m_scalps_today;
   bool   m_paused;
   bool   m_shutdown;
   string m_shutdown_reason;

public:
   CRiskManager()
      : m_balance_sod(0.0), m_daily_dd_pct(0.0),
        m_opened_today(0), m_scalps_today(0),
        m_paused(false), m_shutdown(false), m_shutdown_reason("") {}

   void OnDayStart()
   {
      m_balance_sod  = AccountInfoDouble(ACCOUNT_BALANCE);
      m_daily_dd_pct = 0.0;
      m_opened_today = 0;
      m_scalps_today = 0;
      m_paused       = false;
      // Shutdown is NOT auto-reset — requires manual EA re-attach
   }

   //--- Call every tick to refresh DD
   ENUM_HYDRA_STATE Update(ENUM_HYDRA_STATE current_state)
   {
      if(m_shutdown) return STATE_SHUTDOWN;

      double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      if(m_balance_sod <= 0.0) m_balance_sod = balance;

      m_daily_dd_pct = (m_balance_sod - equity) / m_balance_sod;

      if(m_daily_dd_pct >= DAILY_DD_SHUTDOWN)
      {
         m_shutdown = true;
         m_shutdown_reason = StringFormat("daily_dd=%.2f%% > %.0f%% limit",
                                          m_daily_dd_pct*100, DAILY_DD_SHUTDOWN*100);
         return STATE_SHUTDOWN;
      }
      if(m_daily_dd_pct >= DAILY_DD_PAUSE)
      {
         if(!m_paused)
            PrintFormat("HYDRA4 RISK: Daily DD %.2f%% reached pause threshold — new entries suspended.",
                        m_daily_dd_pct * 100);
         m_paused = true;
         if(current_state == STATE_LIVE) return STATE_LIVE_PAUSED;
      }
      else if(m_paused && m_daily_dd_pct < DAILY_DD_PAUSE * 0.5)
      {
         // Intraday recovery: DD dropped back below 50% of the pause threshold
         // (e.g. 2.5% if pause=5%) — resume new entries with a log message.
         m_paused = false;
         PrintFormat("HYDRA4 RISK: DD recovered to %.2f%% — resuming new entries.",
                     m_daily_dd_pct * 100);
         if(current_state == STATE_LIVE_PAUSED) return STATE_LIVE;
      }
      return current_state;
   }

   bool CanOpenNew()  const { return !m_paused && !m_shutdown; }
   bool IsShutdown()  const { return m_shutdown; }
   bool IsPaused()    const { return m_paused; }

   double DailyDDPct()    const { return m_daily_dd_pct; }
   int    OpenedToday()   const { return m_opened_today; }
   int    ScalpsToday()   const { return m_scalps_today; }
   string ShutdownReason() const { return m_shutdown_reason; }

   void CountOpen()  { m_opened_today++; }
   void CountScalp() { m_scalps_today++; }
};

CRiskManager g_risk;

#endif // RISKMANAGER_MQH
