#ifndef METACONTROLLER_MQH
#define METACONTROLLER_MQH

#include "Defines.mqh"
#include "RiskManager.mqh"

//=============================================================================
// MetaController — Hierarchical Agentic Trading Authority
//
// Level 1 — CRO (Chief Risk Officer):
//   Monitors every agent for consecutive losses and margin violations.
//   Suspends offending agents for CRO_SUSPENSION_SEC.
//
// Level 2 — Senior Supervisors (per asset class):
//   Accumulates PnL per asset class. If drawdown exceeds ASSET_DD_MAX_PCT
//   of account balance, the entire asset class is blocked for SUPERVISOR_SEC.
//
// Level 3 — Worker Agents (GNN / PRISM / APEX / CE):
//   Individual traders. Sharpe-weighted via softmax. Throttled when weight
//   falls below 10%.  Promoted (higher lot scale) when Sharpe > median.
//=============================================================================

//--- CRO thresholds
#define CRO_CONSEC_LOSS_MAX   3      // suspend after N consecutive losses
#define CRO_SUSPENSION_SEC    1800   // 30-minute suspension
#define CRO_MARGIN_THRESHOLD  0.70   // kill any agent if margin-use exceeds 70%

//--- Senior Supervisor thresholds
#define SUPERVISOR_ASSET_DD   0.03   // block asset class after 3% drawdown
#define SUPERVISOR_SEC        900    // 15-minute block

#define SHARPE_WINDOW  252   // trade records per agent (~7 trading days)

// PnL ring per agent [agent][record] stored flat
double g_agent_pnl_ring[4 * SHARPE_WINDOW];
int    g_agent_pnl_head[4];
int    g_agent_pnl_count[4];

class CMetaController
{
private:
   //--- Worker level
   double   m_weights[4];
   double   m_sharpe[4];
   double   m_lot_mult[4];
   bool     m_throttled[4];

   //--- CRO level
   int      m_consec_losses[4];
   int      m_total_trades[4];
   int      m_wins[4];
   datetime m_suspended_until[4];

   //--- Senior Supervisor level (one slot per ENUM_ASSET_CLASS value, 0-5)
   double   m_asset_pnl_acc[6];      // rolling PnL accumulator per asset class
   datetime m_asset_blocked_until[6];

   void ComputeSharpe(int a)
   {
      int n = g_agent_pnl_count[a];
      if(n < 5) { m_sharpe[a] = 0.0; return; }
      double sum = 0.0, sum2 = 0.0;
      for(int i = 0; i < n; i++)
      {
         int pos = (g_agent_pnl_head[a] - 1 - i + SHARPE_WINDOW) % SHARPE_WINDOW;
         double v = g_agent_pnl_ring[a * SHARPE_WINDOW + pos];
         sum += v; sum2 += v*v;
      }
      double mean = sum / n;
      double var  = sum2/n - mean*mean;
      double std  = (var > 0.0) ? MathSqrt(var) : 1e-10;
      m_sharpe[a] = mean / std * MathSqrt(252.0);
   }

   void Softmax()
   {
      double ex[4]; double total = 0.0;
      for(int a = 0; a < 4; a++) { ex[a] = MathExp(MathMax(-10.0, MathMin(10.0, m_sharpe[a]))); total += ex[a]; }
      for(int a = 0; a < 4; a++) m_weights[a] = (total > 0.0) ? ex[a] / total : 0.25;
   }

public:
   CMetaController()
   {
      for(int a = 0; a < 4; a++)
      {
         m_weights[a]        = 0.25;
         m_sharpe[a]         = 0.0;
         m_lot_mult[a]       = 1.0;
         m_throttled[a]      = false;
         m_consec_losses[a]  = 0;
         m_total_trades[a]   = 0;
         m_wins[a]           = 0;
         m_suspended_until[a]= 0;
         g_agent_pnl_head[a]  = 0;
         g_agent_pnl_count[a] = 0;
      }
      for(int c = 0; c < 6; c++)
      {
         m_asset_pnl_acc[c]      = 0.0;
         m_asset_blocked_until[c]= 0;
      }
   }

   // -------------------------------------------------------------------------
   // Level 1 — CRO: record closed trade; apply consecutive-loss suspension
   // -------------------------------------------------------------------------
   void RecordPnL(int a, double pnl, int asset_class = 0)
   {
      if(a < 0 || a > 3) return;

      // Worker ring buffer (Sharpe computation)
      int h = g_agent_pnl_head[a];
      g_agent_pnl_ring[a * SHARPE_WINDOW + h] = pnl;
      g_agent_pnl_head[a]  = (h + 1) % SHARPE_WINDOW;
      if(g_agent_pnl_count[a] < SHARPE_WINDOW) g_agent_pnl_count[a]++;

      // CRO: consecutive loss counter
      m_total_trades[a]++;
      if(pnl < 0.0)
      {
         m_consec_losses[a]++;
         if(m_consec_losses[a] >= CRO_CONSEC_LOSS_MAX)
         {
            m_suspended_until[a] = TimeCurrent() + CRO_SUSPENSION_SEC;
            PrintFormat("CRO: Agent %d SUSPENDED %d min — %d consecutive losses  (total=%d win_rate=%.0f%%)",
                        a, CRO_SUSPENSION_SEC/60, CRO_CONSEC_LOSS_MAX,
                        m_total_trades[a],
                        m_total_trades[a] > 0 ? 100.0*m_wins[a]/m_total_trades[a] : 0.0);
            m_consec_losses[a] = 0;
         }
      }
      else
      {
         m_consec_losses[a] = 0;
         m_wins[a]++;
      }

      // Senior Supervisor: asset-class DD accumulator
      if(asset_class > 0 && asset_class < 6)
      {
         m_asset_pnl_acc[asset_class] += pnl;
         double balance = AccountInfoDouble(ACCOUNT_BALANCE);
         if(balance > 0.0 &&
            m_asset_pnl_acc[asset_class] < -balance * SUPERVISOR_ASSET_DD)
         {
            m_asset_blocked_until[asset_class] = TimeCurrent() + SUPERVISOR_SEC;
            PrintFormat("SUPERVISOR: Asset class %d BLOCKED %d min — class DD exceeded %.1f%%",
                        asset_class, SUPERVISOR_SEC/60, SUPERVISOR_ASSET_DD*100.0);
            m_asset_pnl_acc[asset_class] = 0.0;
         }
      }
   }

   // CRO instant kill for margin violations
   void RecordMarginViolation(int a)
   {
      if(a < 0 || a > 3) return;
      m_suspended_until[a] = TimeCurrent() + CRO_SUSPENSION_SEC;
      PrintFormat("CRO: Agent %d MARGIN KILL — suspended %d min", a, CRO_SUSPENSION_SEC/60);
   }

   // -------------------------------------------------------------------------
   // Level 2 — Senior Supervisor rebalance: Sharpe softmax + throttle/promote
   // -------------------------------------------------------------------------
   void Rebalance()
   {
      for(int a = 0; a < 4; a++) ComputeSharpe(a);
      Softmax();
      for(int a = 0; a < 4; a++)
      {
         m_throttled[a] = (m_weights[a] < 0.10);
         // Promote strong performers, penalise weak ones
         if(m_weights[a] > 0.50)       m_lot_mult[a] = 1.20;   // star trader: +20%
         else if(m_weights[a] > 0.30)  m_lot_mult[a] = 1.00;   // normal
         else                          m_lot_mult[a] = 0.75;   // under-performer: -25%

         if(!m_throttled[a] && m_sharpe[a] > 0.5)
            PrintFormat("META: Agent %d  Sharpe=%.2f  weight=%.2f  lot_mult=%.2f",
                        a, m_sharpe[a], m_weights[a], m_lot_mult[a]);
      }
   }

   // -------------------------------------------------------------------------
   // Query methods
   // -------------------------------------------------------------------------

   // Level 1 — CRO: is this agent currently suspended?
   bool IsSuspended(int a) const
   {
      if(a < 0 || a > 3) return false;
      return (TimeCurrent() < m_suspended_until[a]);
   }

   // Level 2 — Supervisor: is this asset class currently blocked?
   bool IsAssetBlocked(int asset_class) const
   {
      if(asset_class <= 0 || asset_class >= 6) return false;
      return (TimeCurrent() < m_asset_blocked_until[asset_class]);
   }

   // Level 3 — Worker: is this agent throttled (low Sharpe weight)?
   bool IsThrottled(int a) const { return (a >= 0 && a < 4) ? m_throttled[a] : false; }

   double GetWeight(int a)   const { return (a >= 0 && a < 4) ? m_weights[a]  : 0.25; }
   double GetSharpe(int a)   const { return (a >= 0 && a < 4) ? m_sharpe[a]   : 0.0;  }
   double GetLotMult(int a)  const { return (a >= 0 && a < 4) ? m_lot_mult[a] : 1.0;  }

   int    GetTotalTrades(int a) const { return (a >= 0 && a < 4) ? m_total_trades[a] : 0; }
   int    GetWins(int a)        const { return (a >= 0 && a < 4) ? m_wins[a]         : 0; }
   int    GetConsecLosses(int a)const { return (a >= 0 && a < 4) ? m_consec_losses[a]: 0; }

   void GetWeightsArray(double &out[]) const
   {
      ArrayResize(out, 4);
      for(int a = 0; a < 4; a++) out[a] = m_weights[a];
   }
};

CMetaController g_meta;

#endif // METACONTROLLER_MQH
