#ifndef STATEMACHINE_MQH
#define STATEMACHINE_MQH

#include "Defines.mqh"

//=============================================================================
// CStateMachine — simplified 4-state machine (no TRAINING state in mk4).
//
// States:
//   LIVE          — all agents loaded, trading active
//   LIVE_PAUSED   — daily DD pause, no new entries
//   SHUTDOWN      — hard DD hit, all positions closed
//   MODEL_MISSING — one or more .onnx files not found, waits for Python
//
// Python handles training. EA continues on old model during hot-reload.
//=============================================================================

class CStateMachine
{
private:
   ENUM_HYDRA_STATE m_state;
   string           m_reason;
   datetime         m_state_changed;
   int              m_missing_count;

public:
   CStateMachine()
      : m_state(STATE_BOOT), m_reason(""), m_state_changed(0), m_missing_count(0) {}

   ENUM_HYDRA_STATE State() const { return m_state; }
   string Reason() const { return m_reason; }

   //--- Transition to new state with optional reason string
   void Transition(ENUM_HYDRA_STATE new_state, const string reason = "")
   {
      if(m_state == new_state) return;
      PrintFormat("StateMachine: %s → %s  (%s)", StateStr(m_state), StateStr(new_state), reason);
      m_state         = new_state;
      m_reason        = reason;
      m_state_changed = TimeCurrent();
   }

   void SetMissingCount(int n)
   {
      m_missing_count = n;
      if(n > 0 && m_state != STATE_SHUTDOWN)
         Transition(STATE_MODEL_MISSING, StringFormat("%d model(s) missing", n));
   }

   bool CanTrade() const
   {
      return m_state == STATE_LIVE;
   }

   bool CanOpenNew() const
   {
      return m_state == STATE_LIVE;
   }

   bool IsShutdown() const { return m_state == STATE_SHUTDOWN; }

   //--- Update from RiskManager result
   void ApplyRiskState(bool paused, bool shutdown)
   {
      if(shutdown && m_state != STATE_SHUTDOWN)
         Transition(STATE_SHUTDOWN, "risk_manager_shutdown");
      else if(paused && m_state == STATE_LIVE)
         Transition(STATE_LIVE_PAUSED, "daily_dd_pause");
      else if(!paused && m_state == STATE_LIVE_PAUSED)
         Transition(STATE_LIVE, "dd_recovered");
   }

   //--- Called after LoadAgents(). Goes LIVE if any agent loaded.
   //    MODEL_MISSING is only set when zero agents loaded (nothing to trade with).
   //    If some symbols are missing their ONNX files that's fine — those agents
   //    stay loaded=false and are skipped per-iteration; the rest can trade.
   void OnModelsReady(int n_active, int missing)
   {
      m_missing_count = missing;
      if(n_active > 0)
      {
         if(m_state == STATE_BOOT || m_state == STATE_MODEL_MISSING)
            Transition(STATE_LIVE, StringFormat("%d/%d agents ready", n_active, n_active + missing));
         if(missing > 0)
            PrintFormat("StateMachine: %d symbol(s) awaiting models — trading with %d loaded agent(s)",
                        missing, n_active);
      }
      else
      {
         Transition(STATE_MODEL_MISSING, "no agents loaded — waiting for Python");
      }
   }

   string StateStr(ENUM_HYDRA_STATE s) const
   {
      switch(s)
      {
         case STATE_BOOT:          return "BOOT";
         case STATE_LIVE:          return "LIVE";
         case STATE_LIVE_PAUSED:   return "LIVE_PAUSED";
         case STATE_SHUTDOWN:      return "SHUTDOWN";
         case STATE_MODEL_MISSING: return "MODEL_MISSING";
         default:                  return "UNKNOWN";
      }
   }

   string StateStr() const { return StateStr(m_state); }
};

CStateMachine g_state;

#endif // STATEMACHINE_MQH
