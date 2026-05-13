//+------------------------------------------------------------------+
//| MT5_Bot_mk4_HYDRA.mq5                                            |
//| HYDRA mk4 — Full ONNX Pipeline: Python Trains, MT5 Executes      |
//| Version: 4.0.0                                                    |
//+------------------------------------------------------------------+
#property copyright "HYDRA mk4"
#property version   "4.00"
#property strict

#include "includes/Defines.mqh"
#include "includes/BrokerScanner.mqh"   // must come before BrokerConfig
#include "includes/BrokerConfig.mqh"
#include "includes/SymbolFilter.mqh"
#include "includes/TickBuffer.mqh"
#include "includes/FeatureEncoder.mqh"
#include "includes/RiskManager.mqh"
#include "includes/PositionSizer.mqh"
#include "includes/MetaController.mqh"
#include "includes/TradeStacker.mqh"
#include "includes/TrailingManager.mqh"
#include "includes/MicroScalper.mqh"
#include "includes/MarketHours.mqh"
#include "includes/StateMachine.mqh"
#include "includes/OnnxAgent.mqh"
#include "includes/ExecAgent.mqh"
#include "includes/ModifyAgent.mqh"
#include "includes/MCDropout.mqh"
#include "includes/ModelWatcher.mqh"
#include "includes/LimitOrderManager.mqh"
#include "includes/Dashboard.mqh"
#include "includes/RunLogger.mqh"
// mk4.2: BrokerCompliance / EconomicCalendar / FundamentalLoader /
// QuantStateLoader / DynamicRiskLoader / SignalIntelLoader / SignalScraper
// includes removed — all sidecar JSON consumers gone. The EA now operates
// purely on broker-supplied bar data + the trained ONNX models.

//+------------------------------------------------------------------+
//| Input parameters                                                  |
//+------------------------------------------------------------------+
input int    InpMagic                = HYDRA_MAGIC;        // Magic number
input double InpMaxDDPause           = DAILY_DD_PAUSE;     // Daily DD pause (%)
input double InpMaxDDShutdown        = DAILY_DD_SHUTDOWN;  // Daily DD shutdown (%)
input bool   InpUseMicroScalper      = false;              // Enable MicroScalper
input bool   InpUseTradeStacker      = false;              // Enable multi-entry stacking (scalping mode)
input int    InpMaxStackEntries      = 4;                  // Max stacked entries per symbol (overrides tier if lower)
input int    InpDashboardRefreshBars = 5;                  // Dashboard refresh (bars)
input bool   InpAllowLong            = true;               // Allow BUY entries
input bool   InpAllowShort           = true;               // Allow SELL entries
input string InpAllowedSymbols       = "";                 // Manual symbol CSV override (non-empty wins over toggles below)
input string InpScanSymbols          = "";                 // Symbols to auto-scan if BrokerConfig missing

//--- Per-symbol on/off toggles  (used only when InpAllowedSymbols is empty)
//    Forex
input bool   InpAllow_EURUSD         = true;              // Trade EURUSD
input bool   InpAllow_GBPUSD         = true;              // Trade GBPUSD
input bool   InpAllow_USDJPY         = true;              // Trade USDJPY
//    Metals
input bool   InpAllow_GOLD           = true;              // Trade GOLD
input bool   InpAllow_SILVER         = true;              // Trade SILVER
input bool   InpAllow_PLATINUM       = true;              // Trade PLATINUM
input bool   InpAllow_COPPER         = true;              // Trade COPPER
//    Indices
input bool   InpAllow_US_500         = true;              // Trade US_500
input bool   InpAllow_UK_100         = true;              // Trade UK_100
// mk4.8: NAS100 toggle dropped — the active broker doesn't quote it,
// so no tick parquet is extracted and the rosters in python/config.py
// no longer include it.
//    Crypto / Energy
input bool   InpAllow_BTCUSD         = true;              // Trade BTCUSD
input bool   InpAllow_ETHUSD         = true;              // Trade ETHUSD
input bool   InpAllow_LTCUSD         = true;              // Trade LTCUSD
input bool   InpAllow_CrudeOIL       = true;              // Trade CrudeOIL
input bool   InpAllow_BRENT_OIL      = true;              // Trade BRENT_OIL
input bool   InpAllow_NATURAL_GAS    = true;              // Trade NATURAL_GAS

//+------------------------------------------------------------------+
//| Per-symbol ONNX agent set                                         |
//+------------------------------------------------------------------+
#define MAX_ACTIVE 16

struct SSymbolAgents
{
   string       canonical;
   string       agent_type;
   int          sym_idx;
   COnnxAgent   dir_agent;
   CExecAgent   exec_agent;
   CModifyAgent mod_agent;
   CMCDropout   mc;
   bool         loaded;
};

SSymbolAgents g_agents[MAX_ACTIVE];
int           g_n_agents = 0;
SDashSymbolRow g_dash_rows[MAX_ACTIVE];

//+------------------------------------------------------------------+
//| Signal log file                                                   |
//+------------------------------------------------------------------+
int g_signal_log_handle = INVALID_HANDLE;
int g_bar_counter = 0;
int g_diag_tick    = 0;        // counts OnTick calls for periodic diagnostics

// Per-agent last-known diagnostic state (updated every tick, flushed every DIAG_EVERY ticks)
struct SDiagRow
{
   string symbol;
   string agent;
   bool   mkt_open;
   bool   enc_valid;
   int    direction;
   float  conf;
   float  uncert;
   float  session_gate;
   string skip_reason;   // "mkt_closed","no_enc","uncertain","session","dir=0","conf_low","TRADE"
};
SDiagRow g_diag[MAX_ACTIVE];
#define DIAG_EVERY 50   // print summary every N ticks

// Direction bias tracker — ring buffer of last 100 non-flat signals (per agent)
#define BIAS_WINDOW 100
int g_bias_buf[MAX_ACTIVE * BIAS_WINDOW];   // +1=LONG, -1=SHORT
int g_bias_head[MAX_ACTIVE];
int g_bias_count[MAX_ACTIVE];

// Per-agent order failure cooldown — prevents spam-retrying rejected orders
#define ORDER_FAIL_COOLDOWN_MS 10000   // 10-second retry throttle after broker rejection
ulong g_last_fail_ms[MAX_ACTIVE];      // GetTickCount64() timestamp of last failure

// Boot warmup — prevents reflexive entry the instant EA is attached to chart
#define BOOT_WARMUP_SEC     60         // seconds to wait before any new entry after init

// Heartbeat — angry-boss check that wakes idle/stalled workers every 15 min
#define HEARTBEAT_SEC       900        // 15-minute worker health check
#define ADVERSE_EXIT_PCT    0.75       // close position if 75% of SL consumed & model reversed

datetime g_ready_time      = 0;        // earliest time new entries are allowed
datetime g_heartbeat_last  = 0;        // last heartbeat timestamp
datetime g_last_trade_time[MAX_ACTIVE]; // last trade-open timestamp per agent (heartbeat)

// Asset class lookup (mirrors SymbolToAgent mapping, uses ENUM_ASSET_CLASS from Defines.mqh)
ENUM_ASSET_CLASS _SymbolAssetClass(const string canonical)
{
   if(canonical=="GOLD"||canonical=="SILVER"||canonical=="PLATINUM"||canonical=="COPPER")
      return ASSET_METALS;
   if(canonical=="US_500"||canonical=="UK_100")
      return ASSET_INDICES;
   if(canonical=="BTCUSD"||canonical=="ETHUSD"||canonical=="LTCUSD")
      return ASSET_CRYPTO;
   if(canonical=="CrudeOIL"||canonical=="BRENT_OIL"||canonical=="NATURAL_GAS")
      return ASSET_ENERGY;
   if(canonical=="EURUSD"||canonical=="GBPUSD"||canonical=="USDJPY")
      return ASSET_FOREX;
   return ASSET_UNKNOWN;
}

// Per-asset-class SL ceiling as ATR multiple.
// Crypto ATR on M1 is $500-6000; capping at 1.5× prevents 20k-pip SL disasters.
// Indices and energy are also capped tighter than the labeler's 4× default.
double _SLMaxATRMult(const string canonical)
{
   ENUM_ASSET_CLASS ac = _SymbolAssetClass(canonical);
   switch(ac)
   {
      case ASSET_CRYPTO:  return 1.5;   // BTC/ETH/LTC: tight SL mandatory
      case ASSET_INDICES: return 2.0;   // US_500 / UK_100: moderately tight
      case ASSET_ENERGY:  return 2.5;   // CrudeOIL/BRENT: moderate
      case ASSET_METALS:  return 3.0;   // GOLD/SILVER: wider allowed
      default:            return 4.0;   // Forex: full labeler range
   }
}

//+------------------------------------------------------------------+
//| Position tracker — links open positions to their signal data     |
//| for MAE/MFE tracking and resolved-signal logging at close time.  |
//+------------------------------------------------------------------+
struct SPositionTracker
{
   ulong    ticket;
   string   canonical;
   string   agent;
   int      direction;          // +1=LONG / -1=SHORT
   double   entry_price;
   double   pip_size;
   double   worst_price;        // min(low) for LONG, max(high) for SHORT
   double   best_price;         // max(high) for LONG, min(low) for SHORT
   // Signal-time predictions (for resolved log)
   float    confidence;
   float    uncertainty;
   float    timing;
   float    sl_pips_pred;
   float    tp_pips_pred;
   float    vol_mult;
   float    session_gate;
   datetime signal_time;
   bool     active;
};

SPositionTracker g_pos_tracker[MAX_ACTIVE];
int              g_n_tracked = 0;

void _InitTrackers()
{
   for(int i = 0; i < MAX_ACTIVE; i++) g_pos_tracker[i].active = false;
   g_n_tracked = 0;
}

// Register a newly opened position in the tracker.
void _AddPositionTracker(ulong ticket, const string canonical, const string agent,
                         int direction, double entry_price, double pip_size,
                         float conf, float uncert, float timing,
                         float sl_pred, float tp_pred, float vol_mult, float sess_gate,
                         datetime signal_time)
{
   for(int i = 0; i < MAX_ACTIVE; i++)
   {
      if(!g_pos_tracker[i].active)
      {
         g_pos_tracker[i].ticket       = ticket;
         g_pos_tracker[i].canonical    = canonical;
         g_pos_tracker[i].agent        = agent;
         g_pos_tracker[i].direction    = direction;
         g_pos_tracker[i].entry_price  = entry_price;
         g_pos_tracker[i].pip_size     = pip_size > 0 ? pip_size : 0.0001;
         g_pos_tracker[i].worst_price  = entry_price;   // initialise to entry
         g_pos_tracker[i].best_price   = entry_price;
         g_pos_tracker[i].confidence   = conf;
         g_pos_tracker[i].uncertainty  = uncert;
         g_pos_tracker[i].timing       = timing;
         g_pos_tracker[i].sl_pips_pred = sl_pred;
         g_pos_tracker[i].tp_pips_pred = tp_pred;
         g_pos_tracker[i].vol_mult     = vol_mult;
         g_pos_tracker[i].session_gate = sess_gate;
         g_pos_tracker[i].signal_time  = signal_time;
         g_pos_tracker[i].active       = true;
         g_n_tracked++;
         return;
      }
   }
   PrintFormat("HYDRA4 WARN: position tracker full — cannot register ticket=%llu", ticket);
}

// Update worst/best price for all tracked positions (called each tick or bar).
void _UpdateTrackers()
{
   if(g_n_tracked == 0) return;
   for(int i = 0; i < MAX_ACTIVE; i++)
   {
      if(!g_pos_tracker[i].active) continue;
      ulong ticket = g_pos_tracker[i].ticket;
      if(!PositionSelectByTicket(ticket))
      {
         // Position may have just closed — OnTradeTransaction will remove it.
         continue;
      }
      string bname = PositionGetString(POSITION_SYMBOL);
      double bid   = SymbolInfoDouble(bname, SYMBOL_BID);
      double ask   = SymbolInfoDouble(bname, SYMBOL_ASK);
      double cur   = (g_pos_tracker[i].direction > 0) ? bid : ask;

      if(g_pos_tracker[i].direction > 0)
      {
         // LONG: worst = lowest price seen (adverse = we went below entry)
         //        best = highest price seen (favourable excursion)
         if(cur < g_pos_tracker[i].worst_price) g_pos_tracker[i].worst_price = cur;
         if(cur > g_pos_tracker[i].best_price)  g_pos_tracker[i].best_price  = cur;
      }
      else
      {
         // SHORT: worst = highest price seen, best = lowest price seen
         if(cur > g_pos_tracker[i].worst_price) g_pos_tracker[i].worst_price = cur;
         if(cur < g_pos_tracker[i].best_price)  g_pos_tracker[i].best_price  = cur;
      }
   }
}

// Find tracker by ticket.  Returns index or -1.
int _FindTracker(ulong ticket)
{
   for(int i = 0; i < MAX_ACTIVE; i++)
      if(g_pos_tracker[i].active && g_pos_tracker[i].ticket == ticket) return i;
   return -1;
}

// Remove tracker entry after the position closes.
void _RemoveTracker(int idx)
{
   if(idx < 0 || idx >= MAX_ACTIVE) return;
   g_pos_tracker[idx].active = false;
   if(g_n_tracked > 0) g_n_tracked--;
}

//+------------------------------------------------------------------+
//| Helpers                                                           |
//+------------------------------------------------------------------+

string SymbolToAgent(const string canonical)
{
   // Canonical names MUST match python/config.py and BrokerConfig canonical= values
   string forex[]   = {"EURUSD","GBPUSD","USDJPY"};
   string metals[]  = {"GOLD","SILVER","PLATINUM","COPPER"};
   string indices[] = {"US_500","UK_100"};
   for(int i = 0; i < 3; i++) if(forex[i] == canonical) return "PRISM";
   for(int i = 0; i < 4; i++) if(metals[i] == canonical) return "GNN";
   for(int i = 0; i < 2; i++) if(indices[i] == canonical) return "APEX";
   return "CE";
}

string _NormalizeAllowedList(const string csv)
{
   string items[];
   int n = StringSplit(csv, ',', items);
   string out = "";
   for(int i = 0; i < n; i++)
   {
      string tok = items[i];
      StringTrimLeft(tok);
      StringTrimRight(tok);
      if(tok == "") continue;
      if(out != "") out += ",";
      out += tok;
   }
   return out;
}

//--- Build a canonical-symbol CSV from the per-symbol input toggles.
//    Called only when InpAllowedSymbols is empty.
string _BuildSymbolCSVFromToggles()
{
   string out = "";
   #define _ADD(sym, flag) if(flag) { if(out != "") out += ","; out += sym; }
   _ADD("EURUSD",      InpAllow_EURUSD)
   _ADD("GBPUSD",      InpAllow_GBPUSD)
   _ADD("USDJPY",      InpAllow_USDJPY)
   _ADD("GOLD",        InpAllow_GOLD)
   _ADD("SILVER",      InpAllow_SILVER)
   _ADD("PLATINUM",    InpAllow_PLATINUM)
   _ADD("COPPER",      InpAllow_COPPER)
   _ADD("US_500",      InpAllow_US_500)
   _ADD("UK_100",      InpAllow_UK_100)
   _ADD("BTCUSD",      InpAllow_BTCUSD)
   _ADD("ETHUSD",      InpAllow_ETHUSD)
   _ADD("LTCUSD",      InpAllow_LTCUSD)
   _ADD("CrudeOIL",    InpAllow_CrudeOIL)
   _ADD("BRENT_OIL",   InpAllow_BRENT_OIL)
   _ADD("NATURAL_GAS", InpAllow_NATURAL_GAS)
   #undef _ADD
   return out;
}

int AgentToID(const string agent_type)
{
   if(agent_type == "PRISM")  return AGENT_PRISM;
   if(agent_type == "GNN")    return AGENT_GNN;
   if(agent_type == "APEX")   return AGENT_APEX;
   return AGENT_CE;
}

//+------------------------------------------------------------------+
//| Signal logger                                                     |
//+------------------------------------------------------------------+

void OpenSignalLog()
{
   g_signal_log_handle = FileOpen(SIGNAL_LOG,
      FILE_WRITE | FILE_CSV | FILE_COMMON | FILE_SHARE_READ, ',');
   if(g_signal_log_handle != INVALID_HANDLE)
      FileWrite(g_signal_log_handle,
         "timestamp","symbol","agent","direction","confidence","uncertainty",
         "regime","trade_opened","entry_price","sl_pips","tp_pips",
         "lot_size","spread_pips","equity_norm","dd_pct");
}

void LogSignal(const string canonical, const string agent,
               const SSignal &sig, bool trade_opened,
               double lot, double equity_norm)
{
   if(g_signal_log_handle == INVALID_HANDLE) return;
   string bname = g_broker.BrokerName(canonical);
   double ask   = SymbolInfoDouble(bname, SYMBOL_ASK);
   double bid   = SymbolInfoDouble(bname, SYMBOL_BID);
   double spd   = (ask - bid) / g_broker.PipSize(canonical);
   FileWrite(g_signal_log_handle,
      TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
      canonical, agent, sig.direction,
      DoubleToString(sig.confidence, 4),
      DoubleToString(sig.uncertainty, 4),
      (int)sig.regime, (int)trade_opened,
      DoubleToString((sig.direction > 0) ? ask : bid, g_broker.Digits(canonical)),
      DoubleToString(sig.sl_pips, 1),
      DoubleToString(sig.tp_pips, 1),
      DoubleToString(lot, 2),
      DoubleToString(spd, 1),
      DoubleToString(equity_norm, 4),
      DoubleToString(g_risk.DailyDDPct(), 4));
   FileFlush(g_signal_log_handle);
}

void LogTradeClose(ulong ticket, double pips, double pnl, int hold_bars, double max_dd)
{
   if(g_signal_log_handle == INVALID_HANDLE) return;
   FileWrite(g_signal_log_handle,
      TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
      "", "", "",  // no signal columns
      "", "", "", "",
      DoubleToString(0), DoubleToString(pips, 1),
      DoubleToString(pnl, 2), IntegerToString(hold_bars),
      DoubleToString(max_dd, 1));
   FileFlush(g_signal_log_handle);
}

//+------------------------------------------------------------------+
//| Load all ONNX agents for active symbols                           |
//+------------------------------------------------------------------+

int LoadAgents()
{
   int missing = 0;
   g_n_agents = 0;

   for(int i = 0; i < g_filter.Count() && i < MAX_ACTIVE; i++)
   {
      string can   = g_filter.Canonical(i);
      string agent = SymbolToAgent(can);

      g_agents[g_n_agents].canonical   = can;
      g_agents[g_n_agents].agent_type  = agent;
      g_agents[g_n_agents].sym_idx     = i;
      g_agents[g_n_agents].loaded      = false;

      g_agents[g_n_agents].dir_agent.Init(agent, can, FEATURE_DIM);
      g_agents[g_n_agents].exec_agent.Init(agent, can, EXEC_FEATURE_DIM);
      g_agents[g_n_agents].mod_agent.Init(agent, can);

      bool dir_ok  = g_agents[g_n_agents].dir_agent.Load();
      bool exec_ok = dir_ok && g_agents[g_n_agents].exec_agent.Load();
      bool mod_ok  = dir_ok && g_agents[g_n_agents].mod_agent.Load();

      if(!dir_ok)
      {
         // Direction model absent — cannot trade this symbol at all
         PrintFormat("HYDRA: Direction model missing for %s/%s — waiting for Python", agent, can);
         missing++;
         g_run_logger.LogModelLoaded(agent, can, 0, 0, 0, 0.0,
                                     g_agents[g_n_agents].dir_agent.MetaPath(), false);
      }
      else
      {
         double val_acc = g_agents[g_n_agents].dir_agent.ValAcc();

         // Block agents whose val_acc is below the trading threshold.
         // This catches stale / wrong-dimension models (e.g. old 80-dim files
         // loaded under a 136-dim EA) before they place any trades.
         if(val_acc < MIN_VAL_ACC_TRADE)
         {
            PrintFormat("HYDRA: BLOCKED %s/%s — val_acc=%.4f < %.2f (model too weak or wrong dim). Retrain.",
                        agent, can, val_acc, MIN_VAL_ACC_TRADE);
            missing++;
            g_run_logger.LogModelLoaded(agent, can,
               g_agents[g_n_agents].dir_agent.FeatureDim(), 0, 0,
               val_acc, g_agents[g_n_agents].dir_agent.MetaPath(), false);
         }
         else
         {
         // Direction model loaded — agent is live; exec/mod are optional
         g_agents[g_n_agents].mc.SetAgent(&g_agents[g_n_agents].dir_agent);
         g_agents[g_n_agents].loaded = true;

         if(!exec_ok)
            PrintFormat("HYDRA: Exec model missing for %s/%s — using default exec params", agent, can);
         if(!mod_ok)
            PrintFormat("HYDRA: Mod model missing for %s/%s — position modification disabled", agent, can);

         // Log loaded model details
         g_run_logger.LogModelLoaded(
            agent, can,
            g_agents[g_n_agents].dir_agent.FeatureDim(),
            exec_ok ? g_agents[g_n_agents].exec_agent.FeatureDim() : 0,
            mod_ok  ? g_agents[g_n_agents].mod_agent.FeatureDim()  : 0,
            g_agents[g_n_agents].dir_agent.ValAcc(),
            g_agents[g_n_agents].dir_agent.MetaPath(), true);

         // Register with ModelWatcher
         g_watcher.Register(&g_agents[g_n_agents].dir_agent,
                            &g_agents[g_n_agents].exec_agent,
                            &g_agents[g_n_agents].mod_agent);
         } // end val_acc gate
      } // end dir_ok

      g_n_agents++;
   }

   return missing;
}

//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+

int OnInit()
{
   PrintFormat("HYDRA mk4 v%s  magic=%d  init", HYDRA_VERSION, InpMagic);

   //--- Step 0: BrokerConfig gate + auto-scan
   //    If BrokerConfig does not exist, attempt to auto-generate it now.
   //    This eliminates the separate "run BrokerScan first" step for the user.
   if(!BrokerConfigExists())
   {
      Print("HYDRA: BrokerConfig not found — checking for Market_Watch.csv...");

      if(!MarketWatchCSVExists())
      {
         // Cannot scan without the CSV — halt with full instructions
         Print("");
         Print("┌──────────────────────────────────────────────────────────┐");
         Print("│  HYDRA mk4 — SETUP REQUIRED  (one-time step)            │");
         Print("├──────────────────────────────────────────────────────────┤");
         Print("│  BrokerConfig not found AND Market_Watch.csv missing.   │");
         Print("│                                                          │");
         Print("│  STEP 1 — Export Market Watch from MT5:                 │");
         Print("│    1. Open Market Watch  (View → Market Watch / Ctrl+M) │");
         Print("│    2. Add all symbols you want to trade (right-click    │");
         Print("│       → Show All, or add individually)                  │");
         Print("│    3. Right-click symbol list → Export                  │");
         Print("│    4. Save as 'Market_Watch.csv'                        │");
         Print("│    5. Copy to MT5 Common Files:                         │");
         Print("│       Help → Open Data Folder → Common → Files          │");
         Print("│");
         PrintFormat("│       Path: %s\\Files", TerminalInfoString(TERMINAL_COMMONDATA_PATH));
         Print("│");
         Print("│  STEP 2 — Re-attach this EA (or run BrokerScan script)  │");
         Print("│    The EA will auto-generate BrokerConfig on next start. │");
         Print("│                                                          │");
         Print("│  ALTERNATIVE: Run MT5_Bot_mk4_BrokerScan script instead │");
         Print("│    Scripts → MT5_Bot_mk4_BrokerScan → (double-click)    │");
         Print("└──────────────────────────────────────────────────────────┘");
         Print("");
         return INIT_FAILED;
      }

      // CSV found — run auto-scan now
      Print("HYDRA: Market_Watch.csv found — running auto BrokerScan...");
      string canon_list[];
      int n_req = 0;
      string scan_csv = _NormalizeAllowedList(InpScanSymbols);

      if(scan_csv != "")
      {
         n_req = StringSplit(scan_csv, ',', canon_list);
         for(int i = 0; i < n_req; i++)
         {
            StringTrimLeft(canon_list[i]);
            StringTrimRight(canon_list[i]);
         }
      }

      int n_ok = RunBrokerScan(canon_list, n_req, true);
      if(n_ok <= 0)
      {
         Print("HYDRA: Auto BrokerScan failed — see log above.");
         Print("       Fix the issue, then re-attach the EA.");
         return INIT_FAILED;
      }
      PrintFormat("HYDRA: Auto BrokerScan complete — %d symbol(s) configured.", n_ok);
   }

   //--- Load broker config
   if(!g_broker.Load()) return INIT_FAILED;

   // Build active symbol list from BrokerConfig, restricted by EA inputs.
   // Priority: InpAllowedSymbols (manual CSV) wins; if blank, use per-symbol
   // boolean toggles to build the CSV; if all toggles are off, falls through
   // to BrokerConfig with no restriction (all symbols enabled).
   string allowed_csv = _NormalizeAllowedList(InpAllowedSymbols);
   if(allowed_csv == "")
      allowed_csv = _BuildSymbolCSVFromToggles();
   g_filter.Build(allowed_csv);
   PrintFormat("HYDRA: %d symbols active%s", g_filter.Count(),
               (allowed_csv == "") ? "" : StringFormat(" (filter: %s)", allowed_csv));
   if(g_filter.Count() == 0)
   {
      if(allowed_csv == "")
         Print("HYDRA: No active symbols in BrokerConfig");
      else
         PrintFormat("HYDRA: No active symbols matched filter=%s", allowed_csv);
      return INIT_FAILED;
   }

   // Init subsystems
   g_tick_buf.Init();
   g_encoder.Init();
   g_risk.OnDayStart();
   g_stacker.Init(InpMagic);
   g_trailing.Init(InpMagic);
   g_scalper.Init(InpMagic);
   g_limit_mgr.Init(InpMagic);
   g_dashboard.Init();
   // mk4.2: BCF / FAE / QTW / DRM / SIE / SignalScraper init removed —
   // model now consumes only MT5-supplied bar features.

   // Open logger first so LogModelLoaded() calls inside LoadAgents() are captured.
   // Counts are unknown at this point — SetCounts() updates them after loading.
   g_run_logger.OpenSession();

   // Load ONNX agents
   int missing  = LoadAgents();
   int n_active = g_n_agents - missing;
   g_state.OnModelsReady(n_active, missing);

   // Update session header with real counts, then flush to disk
   g_run_logger.SetCounts(n_active, missing);
   g_run_logger.FlushSession();

   // Open signal logger
   OpenSignalLog();

   // Initialise position tracker (MAE/MFE + resolved-signal feedback)
   _InitTrackers();
   ArrayInitialize(g_last_fail_ms, 0);
   ArrayInitialize(g_last_trade_time, 0);

   // Boot warmup: block entries for BOOT_WARMUP_SEC after attach
   g_ready_time     = TimeCurrent() + BOOT_WARMUP_SEC;
   g_heartbeat_last = 0;
   PrintFormat("HYDRA: boot warmup active — entries blocked until %s (+%d sec)",
               TimeToString(g_ready_time, TIME_DATE|TIME_SECONDS), BOOT_WARMUP_SEC);

   // Start timer
   EventSetMillisecondTimer(TIMER_MS);

   PrintFormat("HYDRA mk4: init OK  state=%s  missing=%d", g_state.StateStr(), missing);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+

void OnDeinit(const int reason)
{
   EventKillTimer();
   g_dashboard.Cleanup();
   if(g_signal_log_handle != INVALID_HANDLE)
      FileClose(g_signal_log_handle);

   for(int i = 0; i < g_n_agents; i++)
   {
      g_agents[i].dir_agent.Release();
      g_agents[i].exec_agent.Release();
      g_agents[i].mod_agent.Release();
   }
   Print("HYDRA mk4: deinit");
}

//+------------------------------------------------------------------+
//| _HeartbeatCheck — 15-min angry-boss worker audit                  |
//+------------------------------------------------------------------+

void _HeartbeatCheck()
{
   datetime now = TimeCurrent();
   PrintFormat("=== HYDRA4 HEARTBEAT [%s] state=%s  pos=%d  equity=%.2f  dd=%.2f%% ===",
               TimeToString(now, TIME_DATE|TIME_SECONDS), g_state.StateStr(),
               PositionsTotal(), AccountInfoDouble(ACCOUNT_EQUITY),
               g_risk.DailyDDPct() * 100.0);

   int idle_agents = 0, suspended_agents = 0;

   for(int ai = 0; ai < g_n_agents; ai++)
   {
      if(!g_agents[ai].loaded) continue;
      string can   = g_agents[ai].canonical;
      string agent = g_agents[ai].agent_type;
      int    ag_id = AgentToID(agent);

      bool mkt_open  = g_broker.IsMarketOpen(can, now);
      bool suspended = g_meta.IsSuspended(ag_id);
      bool throttled = g_meta.IsThrottled(ag_id);
      double sharpe  = g_meta.GetSharpe(ag_id);
      double weight  = g_meta.GetWeight(ag_id);

      if(suspended) suspended_agents++;

      datetime last_trade = g_last_trade_time[ai];
      int mins_idle = (last_trade > 0) ? (int)((now - last_trade) / 60) : -1;

      // Worker is considered lazy if market is open, not suspended, and hasn't
      // traded or shown any signal in the last 2 heartbeat cycles
      bool lazy = mkt_open && !suspended &&
                  (last_trade == 0 || (now - last_trade) > (HEARTBEAT_SEC * 2));

      if(lazy)
      {
         idle_agents++;
         PrintFormat("  HB WARN [IDLE]: %s/%s — last_trade=%s mkt=%s susp=%s throttle=%s sharpe=%.2f wt=%.2f",
                     agent, can,
                     (mins_idle < 0) ? "never" : (IntegerToString(mins_idle)+"min"),
                     mkt_open ? "Y" : "N", suspended ? "Y" : "N", throttled ? "Y" : "N",
                     sharpe, weight);

         // Clear fail_cooldown so the worker gets an unblocked attempt next tick
         if(g_last_fail_ms[ai] > 0)
         {
            g_last_fail_ms[ai] = 0;
            PrintFormat("  HB: %s/%s fail_cooldown cleared (was blocking retries)", agent, can);
         }
      }
      else
      {
         PrintFormat("  HB OK: %s/%s  mkt=%s susp=%s throttle=%s sharpe=%.2f wt=%.2f  last_trade=%s",
                     agent, can,
                     mkt_open ? "Y" : "N", suspended ? "Y" : "N", throttled ? "Y" : "N",
                     sharpe, weight,
                     (last_trade > 0) ? TimeToString(last_trade, TIME_DATE|TIME_SECONDS) : "never");
      }
   }

   // Force Sharpe rebalance when any workers are stuck —
   // expired suspensions don't auto-update the throttle flag until next 1000-tick cycle
   if(idle_agents > 0 || suspended_agents > 0)
   {
      g_meta.Rebalance();
      PrintFormat("  HB: Rebalance forced — %d idle, %d suspended workers", idle_agents, suspended_agents);
   }

   PrintFormat("=== HEARTBEAT END ===");
}

//+------------------------------------------------------------------+
//| OnTimer — hot-reload check + 15-min heartbeat                     |
//+------------------------------------------------------------------+

void OnTimer()
{
   // 15-minute worker heartbeat
   datetime now_t = TimeCurrent();
   if(g_heartbeat_last == 0 || (now_t - g_heartbeat_last) >= HEARTBEAT_SEC)
   {
      _HeartbeatCheck();
      g_heartbeat_last = now_t;
   }

   // Hot-reload check
   g_watcher.CheckAll();

   // Limit order expiry
   g_limit_mgr.Update();

   // mk4.2: Macro snapshot reload + Phase 1-5 JSON reload removed —
   // model consumes only MT5-supplied bar features now.
}

//+------------------------------------------------------------------+
//| OnTick — main inference + execution loop                          |
//+------------------------------------------------------------------+

void OnTick()
{
   if(g_state.IsShutdown()) return;

   datetime now = TimeCurrent();
   MqlDateTime dt_now; TimeToStruct(now, dt_now);

   // New day reset
   static int last_day = -1;
   if(dt_now.day != last_day)
   {
      g_risk.OnDayStart();
      last_day = dt_now.day;
   }

   // Update state from risk manager
   bool paused   = g_risk.IsPaused();
   bool shutdown = g_risk.IsShutdown();
   g_state.ApplyRiskState(paused, shutdown);
   if(g_state.IsShutdown()) return;

   // Update risk manager with current equity
   g_risk.Update(g_state.State());

   // Update tick buffers for all active symbols
   for(int i = 0; i < g_filter.Count(); i++)
   {
      string can   = g_filter.Canonical(i);
      string bname = g_broker.BrokerName(can);
      MqlTick tick;
      if(!SymbolInfoTick(bname, tick)) continue;
      double spread_pts = (double)(tick.ask - tick.bid);
      g_tick_buf.Push(i, tick, spread_pts);

      // Encode direction features
      g_encoder.Encode(i, can);
   }

   // Trailing + scalper (always run even when paused)
   g_trailing.Update();
   if(InpUseMicroScalper) g_scalper.Update();

   // Update MAE/MFE trackers for all open positions (runs even when paused)
   _UpdateTrackers();

   if(g_state.CanOpenNew())
   {
   // Portfolio context for exec model
   double equity      = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance     = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity_norm = (balance > 0.0) ? equity / balance - 1.0 : 0.0;
   double equity_dd   = g_risk.DailyDDPct();
   int    open_pos    = PositionsTotal();
   double open_lots   = 0.0;
   for(int p = 0; p < open_pos; p++)
   {
      if(!PositionGetTicket(p)) continue;
      if(PositionGetInteger(POSITION_MAGIC) == InpMagic)
         open_lots += PositionGetDouble(POSITION_VOLUME);
   }
   float open_lots_norm = (float)MathMin(open_lots / 10.0, 1.0);
   float pnl_norm       = (float)MathMax(-1.0, MathMin(equity_norm, 1.0));

   // --- Main per-agent signal loop ---
   for(int ai = 0; ai < g_n_agents; ai++)
   {
      if(!g_agents[ai].loaded) continue;
      int sym_idx  = g_agents[ai].sym_idx;
      string can   = g_agents[ai].canonical;
      string agent = g_agents[ai].agent_type;
      int    ag_id = AgentToID(agent);

      // --- Diagnostic state capture (always, before any continue) ---
      g_diag[ai].symbol       = can;
      g_diag[ai].agent        = agent;
      g_diag[ai].enc_valid    = g_encoder.IsValid(sym_idx);
      g_diag[ai].mkt_open     = g_broker.IsMarketOpen(can, now);
      g_diag[ai].direction    = 0;
      g_diag[ai].conf         = 0.0f;
      g_diag[ai].uncert       = 0.0f;
      g_diag[ai].session_gate = 0.0f;
      g_diag[ai].skip_reason  = "pending";

      if(!g_encoder.IsValid(sym_idx))
         { g_diag[ai].skip_reason = "no_enc"; continue; }
      if(!g_broker.IsMarketOpen(can, now))
         { g_diag[ai].skip_reason = "mkt_closed"; continue; }

      // 1. Direction model — MC Dropout
      float x_dir[FEATURE_DIM];
      g_encoder.GetDirFeatures(sym_idx, x_dir);

      float mu = 0.5f, sigma_ep = 0.0f;
      g_agents[ai].mc.Infer(x_dir, mu, sigma_ep);

      // Skip if uncertain
      if(g_agents[ai].mc.IsUncertain(sigma_ep))
      {
         g_diag[ai].conf = mu; g_diag[ai].uncert = sigma_ep; g_diag[ai].skip_reason = "uncertain";
         g_run_logger.LogSkippedSignal(can, agent, 0, (double)mu, (double)sigma_ep,
            0.0, 0.0, 0.0, 0.0, equity_dd, "uncertain");
         continue;
      }

      // mk4.3: MTF alignment gate removed. The MTF block was at slots
      // 1132..1139 in the old 1180-dim layout; the parity-floored 200-dim
      // model has no such block. Reading those indices now is out of
      // bounds. The model itself implicitly captures alignment via its
      // M5 features.

      int    dir   = g_agents[ai].mc.Direction(mu);
      // conf = directional conviction: for SHORT signals mu is low (e.g. 0.30),
      // so flip it so conf always represents "how sure is the model of its direction".
      float  conf  = (dir < 0) ? (1.0f - mu) : mu;
      float  uncert = sigma_ep;
      g_diag[ai].direction = dir;
      g_diag[ai].conf      = conf;
      g_diag[ai].uncert    = uncert;

      // Regime (simple: use mu deviation from 0.5 as proxy)
      ENUM_REGIME regime = (mu > 0.6f) ? REGIME_BULL : (mu < 0.4f ? REGIME_BEAR : REGIME_SIDEWAYS);

      // 2. Execution model
      g_encoder.BuildExecFeatures(sym_idx, can, conf, uncert,
                                  (int)regime, open_lots_norm, pnl_norm,
                                  (float)equity_dd);
      float x_exec[EXEC_FEATURE_DIM];
      g_encoder.GetExecFeatures(sym_idx, x_exec);

      float exec_out[5];
      bool have_exec = g_agents[ai].exec_agent.Infer(x_exec, exec_out);

      float timing      = have_exec ? exec_out[EXEC_IDX_TIMING]  : 0.5f;
      float sl_pips     = have_exec ? exec_out[EXEC_IDX_SL]      : 20.0f;
      float tp_pips     = have_exec ? exec_out[EXEC_IDX_TP]      : 30.0f;
      float vol_mult    = have_exec ? exec_out[EXEC_IDX_VOL]     : 1.0f;
      float session_gate= have_exec ? exec_out[EXEC_IDX_SESSION] : g_market_hours.SessionGateFallback();

      // Exec model outputs SL and TP as ATR multiples (labels normalised in Python).
      // Step 1: multiply back to pips.  Step 2: clamp to safe ATR bounds.  Step 3: R:R gate.
      {
         double _pip        = g_broker.PipSize(can);
         double _atr_pips   = (_pip > 0.0) ? g_atr14[sym_idx] / _pip : 20.0;
         double _atr_safe   = MathMax(_atr_pips, 5.0);  // floor for tiny ATR values

         // Multiply ATR-multiple outputs → pips (undoes labeler_exec normalization)
         sl_pips = sl_pips * (float)_atr_safe;
         tp_pips = tp_pips * (float)_atr_safe;

         // Clamp to ATR-multiple bounds; tighter ceiling for crypto/indices to prevent
         // catastrophic SL on high-price assets where ATR × 4 = tens of thousands of pips.
         double _sl_max_mult = _SLMaxATRMult(can);
         sl_pips = (float)MathMax(0.5 * _atr_safe, MathMin(_sl_max_mult * _atr_safe, (double)sl_pips));
         tp_pips = (float)MathMax(1.0 * _atr_safe, MathMin(6.0 * _atr_safe, (double)tp_pips));

         // Enforce minimum R:R = 1.5 — skip trade rather than mangle TP.
         if(tp_pips < sl_pips * 1.5f)
         {
            g_diag[ai].skip_reason = "rr_fail";
            PrintFormat("HYDRA4 SKIP %s: R:R=%.2f (tp=%.1f sl=%.1f) below 1.5 min",
                        can, (double)tp_pips / (double)sl_pips, (double)tp_pips, (double)sl_pips);
            { double _spd = (_pip > 0) ? (SymbolInfoDouble(g_broker.BrokerName(can), SYMBOL_ASK)
                           - SymbolInfoDouble(g_broker.BrokerName(can), SYMBOL_BID)) / _pip : 0.0;
              g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
                 timing, sl_pips, tp_pips, _spd, equity_dd, "rr_fail"); }
            continue;
         }
      }

      // 3. Session gate
      g_diag[ai].session_gate = session_gate;
      if(session_gate < SESSION_THRESHOLD)
      {
         g_diag[ai].skip_reason = "session";
         { double _pip2 = g_broker.PipSize(can);
           double _spd2 = (_pip2 > 0) ? (SymbolInfoDouble(g_broker.BrokerName(can), SYMBOL_ASK)
                          - SymbolInfoDouble(g_broker.BrokerName(can), SYMBOL_BID)) / _pip2 : 0.0;
           g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
              timing, sl_pips, tp_pips, _spd2, equity_dd, "session"); }
         continue;
      }

      // 4. Build SSignal
      SSignal sig;
      sig.canonical    = can;
      sig.direction    = dir;
      sig.confidence   = conf;
      sig.uncertainty  = uncert;
      sig.timing       = timing;
      sig.sl_pips      = sl_pips;
      sig.tp_pips      = tp_pips;
      sig.vol_mult     = vol_mult;
      sig.session_gate = session_gate;
      sig.regime       = regime;
      sig.skip         = (dir == 0 || conf < CONF_THRESHOLD);

      // Update dashboard row
      g_dash_rows[ai].canonical    = can;
      g_dash_rows[ai].direction    = dir;
      g_dash_rows[ai].confidence   = conf;
      g_dash_rows[ai].uncertainty  = uncert;
      g_dash_rows[ai].session_gate = session_gate;
      g_dash_rows[ai].val_acc      = g_agents[ai].dir_agent.ValAcc();
      g_dash_rows[ai].last_reload  = TimeToString(g_agents[ai].dir_agent.LastLoaded(),
                                                  TIME_DATE | TIME_SECONDS);

      // Compute current spread once (used for skip logging below)
      double cur_spd_pips = 0.0;
      { double _pipS = g_broker.PipSize(can);
        if(_pipS > 0) cur_spd_pips = (SymbolInfoDouble(g_broker.BrokerName(can), SYMBOL_ASK)
                                     - SymbolInfoDouble(g_broker.BrokerName(can), SYMBOL_BID)) / _pipS; }

      if(sig.skip)
      {
         string skip_reason = (dir == 0) ? "dir=0" : "conf_low";
         g_diag[ai].skip_reason = skip_reason;
         g_run_logger.LogSignal(can, agent,
            dir, conf, uncert, (int)regime,
            cur_spd_pips, g_risk.DailyDDPct(),
            timing, sl_pips, tp_pips, vol_mult, session_gate,
            false, skip_reason);
         g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
            timing, sl_pips, tp_pips, cur_spd_pips, equity_dd, skip_reason);
         continue;
      }
      g_diag[ai].skip_reason = "SIGNAL";

      // Direction gate — block longs or shorts from the terminal inputs
      if(dir > 0 && !InpAllowLong)
      {
         g_diag[ai].skip_reason = "long_disabled";
         g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
            timing, sl_pips, tp_pips, cur_spd_pips, equity_dd, "long_disabled");
         continue;
      }
      if(dir < 0 && !InpAllowShort)
      {
         g_diag[ai].skip_reason = "short_disabled";
         g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
            timing, sl_pips, tp_pips, cur_spd_pips, equity_dd, "short_disabled");
         continue;
      }

      // 5. Hierarchical authority checks (CRO → Supervisor → Worker)

      // Level 1 — CRO: agent suspended for consecutive losses or margin violation
      if(g_meta.IsSuspended(ag_id))
      {
         g_diag[ai].skip_reason = "cro_suspended";
         g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
            timing, sl_pips, tp_pips, cur_spd_pips, equity_dd, "cro_suspended");
         continue;
      }

      // Level 2 — Senior Supervisor: asset class blocked due to drawdown
      ENUM_ASSET_CLASS sym_asset = _SymbolAssetClass(can);
      if(g_meta.IsAssetBlocked((int)sym_asset))
      {
         g_diag[ai].skip_reason = "supervisor_blocked";
         g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
            timing, sl_pips, tp_pips, cur_spd_pips, equity_dd, "supervisor_blocked");
         continue;
      }

      // Level 3 — Worker: Sharpe-weight throttle (low-performing agent)
      double meta_wt = g_meta.GetWeight(ag_id);
      if(g_meta.IsThrottled(ag_id))
      {
         g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
            timing, sl_pips, tp_pips, cur_spd_pips, equity_dd, "meta_throttled");
         continue;
      }

      // Bias gate: model with severe directional bias requires higher confidence
      if(g_bias_count[ai] >= BIAS_WINDOW / 2)
      {
         int n_same = 0;
         for(int b = 0; b < g_bias_count[ai]; b++)
            if(g_bias_buf[ai * BIAS_WINDOW + b] == dir) n_same++;
         double bias_pct = (double)n_same / g_bias_count[ai];
         if(bias_pct > 0.80 && conf < 0.75f)
         {
            g_diag[ai].skip_reason = "bias_filtered";
            g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
               timing, sl_pips, tp_pips, cur_spd_pips, equity_dd, "bias_filtered");
            continue;
         }
      }

      // Failure cooldown: skip if last order attempt was rejected recently
      if(g_last_fail_ms[ai] > 0 &&
         (GetTickCount64() - g_last_fail_ms[ai]) < ORDER_FAIL_COOLDOWN_MS)
      {
         g_diag[ai].skip_reason = "fail_cooldown";
         g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
            timing, sl_pips, tp_pips, cur_spd_pips, equity_dd, "fail_cooldown");
         continue;
      }

      // CRO margin check: if margin use > threshold, kill agent and skip
      double margin_used = AccountInfoDouble(ACCOUNT_MARGIN);
      double margin_free = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
      double margin_total= margin_used + margin_free;
      if(margin_total > 0.0 && margin_used / margin_total > CRO_MARGIN_THRESHOLD)
      {
         g_meta.RecordMarginViolation(ag_id);
         g_diag[ai].skip_reason = "cro_margin_kill";
         g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
            timing, sl_pips, tp_pips, cur_spd_pips, equity_dd, "cro_margin_kill");
         continue;
      }

      // Map vol_mult from exec model sigmoid output [0,1] → [VOL_MULT_MIN, VOL_MULT_MAX]
      // Exec model trains labels in [0.5, 2.0]; sigmoid output must be un-squeezed here.
      double vol_mult_scaled = VOL_MULT_MIN + (VOL_MULT_MAX - VOL_MULT_MIN) * (double)vol_mult;

      double pip  = g_broker.PipSize(can);
      double sl_dist = sl_pips * pip;
      double tp_dist = tp_pips * pip;
      double base_lot= g_sizer.CalcLotFromSL(can, sl_dist, MAX_RISK_PER_TRADE);
      // vol_mult_scaled: exec model risk adjuster [0.5, 2.0]
      // GetLotMult: meta performance multiplier [0.75, 1.20] — rewards star traders
      double lot     = g_broker.NormalizeLot(can, base_lot * vol_mult_scaled * g_meta.GetLotMult(ag_id));

      if(lot < g_broker.VolumeMin(can))
      {
         g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
            timing, sl_pips, tp_pips, cur_spd_pips, equity_dd, "lot_too_small");
         continue;
      }

      // Boot warmup: no new entries until BOOT_WARMUP_SEC has elapsed since init
      if(TimeCurrent() < g_ready_time)
      {
         g_diag[ai].skip_reason = "warmup";
         continue;
      }

      // One-position-per-symbol guard + direction-reversal early close
      {
         string bname_chk = g_broker.BrokerName(can);
         bool   has_pos   = false;
         ulong  open_tk   = 0;
         int    open_type = -1;
         double open_px_chk = 0.0;

         for(int px = 0; px < PositionsTotal(); px++)
         {
            ulong tk = PositionGetTicket(px);
            if(!PositionSelectByTicket(tk)) continue;
            if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
            if(PositionGetString(POSITION_SYMBOL) != bname_chk) continue;
            has_pos     = true;
            open_tk     = tk;
            open_type   = (int)PositionGetInteger(POSITION_TYPE);
            open_px_chk = PositionGetDouble(POSITION_PRICE_OPEN);
            break;
         }

         if(has_pos)
         {
            int  pos_dir2  = (open_type == POSITION_TYPE_BUY) ? 1 : -1;
            bool reversed  = (dir != 0 && dir != pos_dir2);

            // Reversal close — always runs regardless of stacking mode
            if(reversed && conf > (float)(CONF_THRESHOLD + 0.05))
            {
               double pip_chk   = g_broker.PipSize(can);
               double cur_chk   = (pos_dir2 > 0)
                                  ? SymbolInfoDouble(bname_chk, SYMBOL_BID)
                                  : SymbolInfoDouble(bname_chk, SYMBOL_ASK);
               double rev_pips  = (pos_dir2 > 0)
                                  ? (cur_chk - open_px_chk) / pip_chk
                                  : (open_px_chk - cur_chk) / pip_chk;
               CTrade trd_rev;
               trd_rev.SetExpertMagicNumber(InpMagic);
               if(trd_rev.PositionClose(open_tk))
                  PrintFormat("HYDRA4: REVERSAL_CLOSE %s ticket=%llu  old=%s new=%s conf=%.3f pnl_pips=%.1f",
                              can, open_tk,
                              (pos_dir2 > 0) ? "LONG" : "SHORT",
                              (dir > 0) ? "LONG" : "SHORT",
                              conf, rev_pips);
               g_diag[ai].skip_reason = "rev_closed";
               continue;  // let next tick re-evaluate with fresh position state
            }

            // Same-direction stacking gate
            if(!InpUseTradeStacker || conf < CONF_THRESHOLD)
            {
               g_diag[ai].skip_reason = "has_pos";
               continue;
            }

            // Count how many entries we already have on this symbol
            int existing_entries = 0;
            for(int px2 = 0; px2 < PositionsTotal(); px2++)
            {
               ulong tk2 = PositionGetTicket(px2);
               if(!PositionSelectByTicket(tk2)) continue;
               if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
               if(PositionGetString(POSITION_SYMBOL) != bname_chk) continue;
               existing_entries++;
            }

            // Tier limit from confidence: 0.90→6, 0.80→4, 0.70→3, 0.60→2, else 1
            int tier = (conf >= 0.90f) ? 6 : (conf >= 0.80f) ? 4 :
                       (conf >= 0.70f) ? 3 : (conf >= 0.60f) ? 2 : 1;
            int cap  = MathMin(tier, InpMaxStackEntries);
            if(existing_entries >= cap)
            {
               g_diag[ai].skip_reason = "stack_full";
               continue;
            }
            // Fall through — stacking entry is allowed below
         }
      }

      // 6. Place order based on timing
      bool trade_opened = false;
      if(timing >= TIMING_THRESHOLD)
      {
         // Market entry
         trade_opened = (PlaceEntryWithSLTP(can, dir, sl_dist, tp_dist, lot, meta_wt) > 0);
         g_risk.CountOpen();
      }
      else if(timing >= TIMING_WAIT_MIN && !g_limit_mgr.HasPending(can))
      {
         // Limit entry
         string bname = g_broker.BrokerName(can);
         double close_px = (dir > 0) ? SymbolInfoDouble(bname, SYMBOL_ASK)
                                     : SymbolInfoDouble(bname, SYMBOL_BID);
         float atr_pips = (float)(g_atr14[sym_idx] / pip);
         float offset   = atr_pips * (float)LIMIT_EXPIRY_BARS * 0.1f;
         trade_opened = g_limit_mgr.PlaceLimitEntry(can, dir, close_px,
                                                     offset, sl_pips, tp_pips,
                                                     (float)lot, LIMIT_EXPIRY_BARS);
      }

      g_diag[ai].skip_reason = trade_opened ? "TRADE" : "no_fill";
      LogSignal(can, agent, sig, trade_opened, lot, equity / balance);

      if(!trade_opened)
      {
         g_last_fail_ms[ai] = GetTickCount64();   // start failure cooldown
         g_run_logger.LogSkippedSignal(can, agent, dir, conf, uncert,
            timing, sl_pips, tp_pips, cur_spd_pips, equity_dd, "no_fill");
      }

      // Register newly opened trade in the position tracker for MAE/MFE + resolved log.
      // We look for the newest position matching our magic + canonical symbol.
      if(trade_opened) g_last_trade_time[ai] = TimeCurrent();   // heartbeat tracking
      if(trade_opened)
      {
         double entry_px = (dir > 0) ? SymbolInfoDouble(g_broker.BrokerName(can), SYMBOL_ASK)
                                     : SymbolInfoDouble(g_broker.BrokerName(can), SYMBOL_BID);
         // Find the ticket just opened (newest position on this symbol with our magic)
         ulong new_ticket = 0;
         datetime newest  = 0;
         for(int px = 0; px < PositionsTotal(); px++)
         {
            ulong tk = PositionGetTicket(px);
            if(!PositionSelectByTicket(tk)) continue;
            if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
            if(PositionGetString(POSITION_SYMBOL) != g_broker.BrokerName(can)) continue;
            datetime ot = (datetime)PositionGetInteger(POSITION_TIME);
            if(ot >= newest) { newest = ot; new_ticket = tk; }
         }
         if(new_ticket > 0)
            _AddPositionTracker(new_ticket, can, agent, dir, entry_px, pip,
                                conf, uncert, timing, sl_pips, tp_pips,
                                vol_mult, session_gate, TimeCurrent());
      }

      // Record direction in bias ring buffer (only non-flat signals)
      if(dir != 0)
      {
         int head = g_bias_head[ai];
         g_bias_buf[ai * BIAS_WINDOW + head] = dir;
         g_bias_head[ai]  = (head + 1) % BIAS_WINDOW;
         if(g_bias_count[ai] < BIAS_WINDOW) g_bias_count[ai]++;
      }

      // Structured run-logger (signals CSV)
      {
         string bname_rl = g_broker.BrokerName(can);
         double ask_rl   = SymbolInfoDouble(bname_rl, SYMBOL_ASK);
         double bid_rl   = SymbolInfoDouble(bname_rl, SYMBOL_BID);
         double spd_rl   = (ask_rl - bid_rl) / g_broker.PipSize(can);
         string skip_str = sig.skip ? "skip" : "";
         g_run_logger.LogSignal(can, agent,
            sig.direction, sig.confidence, sig.uncertainty, (int)sig.regime,
            spd_rl, g_risk.DailyDDPct(),
            timing, sl_pips, tp_pips, vol_mult, session_gate,
            trade_opened, skip_str);
      }
   }

   // --- Modification model pass for open positions ---
   for(int p = 0; p < PositionsTotal(); p++)
   {
      ulong ticket = PositionGetTicket(p);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;

      string bname  = PositionGetString(POSITION_SYMBOL);
      string can    = bname;
      for(int s = 0; s < g_sym_count; s++)
         if(g_sym_specs[s].broker_name == bname) { can = g_sym_specs[s].canonical; break; }

      // Find agent for this symbol
      int ai_idx = -1;
      for(int ai = 0; ai < g_n_agents; ai++)
         if(g_agents[ai].canonical == can) { ai_idx = ai; break; }
      if(ai_idx < 0 || !g_agents[ai_idx].loaded) continue;

      int sym_idx = g_agents[ai_idx].sym_idx;
      if(!g_encoder.IsValid(sym_idx)) continue;

      // Build mod features: dir_feat(136) at [0..135] + pos_context(8) at [136..143].
      // pos_context does NOT overwrite the direction block — avoids corrupting rolling stats.
      float x_dir[FEATURE_DIM];
      g_encoder.GetDirFeatures(sym_idx, x_dir);

      double pip      = g_broker.PipSize(can);
      double open_px  = PositionGetDouble(POSITION_PRICE_OPEN);
      double cur_px   = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                        ? SymbolInfoDouble(bname, SYMBOL_BID)
                        : SymbolInfoDouble(bname, SYMBOL_ASK);
      double float_pnl_pips = ((int)PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                              ? (cur_px - open_px) / pip
                              : (open_px - cur_px) / pip;
      double cur_sl   = PositionGetDouble(POSITION_SL);
      double cur_tp   = PositionGetDouble(POSITION_TP);
      double sl_pips  = MathAbs(open_px - cur_sl) / pip;
      double tp_pips  = MathAbs(cur_tp - open_px) / pip;

      // Build 144-dim mod input: copy dir features then append position context at end
      float x_mod[MOD_FEATURE_DIM];
      ArrayCopy(x_mod, x_dir, 0, 0, FEATURE_DIM);   // [0..135] = direction features (untouched)
      x_mod[136] = (float)MathMin(float_pnl_pips / 50.0, 2.0);   // floating PnL norm
      x_mod[137] = (float)MathMin(sl_pips / 50.0, 2.0);          // current SL norm
      x_mod[138] = (float)MathMin(tp_pips / 50.0, 2.0);          // current TP norm
      x_mod[139] = (float)g_agents[ai_idx].dir_agent.ValAcc();    // model quality
      // Fill MFE/MAE from live tracker if available — improves mod model input quality
      int tr_mod = _FindTracker(ticket);
      if(tr_mod >= 0 && g_pos_tracker[tr_mod].entry_price > 0 && pip > 0)
      {
         SPositionTracker trm = g_pos_tracker[tr_mod];
         double mfe_m = (trm.direction > 0)
            ? MathMax(0.0, (trm.best_price  - trm.entry_price) / pip)
            : MathMax(0.0, (trm.entry_price - trm.best_price)  / pip);
         double mae_m = (trm.direction > 0)
            ? MathMax(0.0, (trm.entry_price - trm.worst_price) / pip)
            : MathMax(0.0, (trm.worst_price - trm.entry_price) / pip);
         x_mod[140] = (float)MathMin(mfe_m / 50.0, 2.0);   // MFE norm
         x_mod[141] = (float)MathMin(mae_m / 50.0, 2.0);   // MAE norm
      }
      else
      {
         x_mod[140] = 0.0f;
         x_mod[141] = 0.0f;
      }
      x_mod[142] = 0.0f;
      x_mod[143] = 0.0f;

      SModSignal mod_sig;
      if(!g_agents[ai_idx].mod_agent.Infer(ticket, x_mod, mod_sig)) continue;

      // Apply modification signals
      int digits = g_broker.Digits(can);
      if(mod_sig.move_sl_to_be > MOD_BE_THRESHOLD && float_pnl_pips > 0.0)
      {
         g_trailing.MoveToBE(ticket, open_px, digits);
         // sl_pips_after = 0 because BE sets SL to entry price (0 pips of risk)
         g_run_logger.LogModEvent(can, ticket, "BE",
            (double)mod_sig.move_sl_to_be, float_pnl_pips,
            sl_pips, 0.0, (double)mod_sig.close_now);
      }
      else if(mod_sig.trail_sl_pips > 0.5f)
      {
         g_trailing.TrailByPips(ticket, bname, can, mod_sig.trail_sl_pips, pip, digits);
         g_run_logger.LogModEvent(can, ticket, "TRAIL",
            (double)mod_sig.trail_sl_pips, float_pnl_pips,
            sl_pips, (double)mod_sig.trail_sl_pips, (double)mod_sig.close_now);
      }

      if(mod_sig.close_now > MOD_CLOSE_THRESHOLD)
      {
         CTrade trd;
         trd.SetExpertMagicNumber(InpMagic);
         if(trd.PositionClose(ticket))
         {
            PrintFormat("HYDRA4: early close %s ticket=%llu close_now=%.2f",
                        can, ticket, mod_sig.close_now);
            g_run_logger.LogModEvent(can, ticket, "CLOSE_EARLY",
               (double)mod_sig.close_now, float_pnl_pips,
               sl_pips, sl_pips, (double)mod_sig.close_now);
            continue;
         }
      }

      // Adversity exit: position consumed >= ADVERSE_EXIT_PCT of its SL and the
      // direction model has flipped or gone uncertain — don't just wait for full stop.
      if(float_pnl_pips < 0.0 && sl_pips > 0.5)
      {
         float adverse_pct = (float)(-float_pnl_pips / sl_pips);
         if(adverse_pct >= (float)ADVERSE_EXIT_PCT)
         {
            int  pos_dir_adv  = ((int)PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
            int  last_dir_adv = (ai_idx >= 0) ? g_diag[ai_idx].direction : 0;
            bool model_flip   = (last_dir_adv != 0 && last_dir_adv != pos_dir_adv);
            bool model_unc    = (ai_idx >= 0 && g_diag[ai_idx].skip_reason == "uncertain");

            if(model_flip || model_unc)
            {
               CTrade trd_adv;
               trd_adv.SetExpertMagicNumber(InpMagic);
               if(trd_adv.PositionClose(ticket))
               {
                  PrintFormat("HYDRA4: ADVERSITY_EXIT %s ticket=%llu  adverse=%.0f%% of SL  flip=%s unc=%s  pnl_pips=%.1f",
                              can, ticket, adverse_pct * 100.0,
                              model_flip ? "Y" : "N", model_unc ? "Y" : "N",
                              float_pnl_pips);
                  g_run_logger.LogModEvent(can, ticket, "ADVERSITY_EXIT",
                     adverse_pct, float_pnl_pips, sl_pips, sl_pips, 1.0);
               }
               continue;
            }
         }
      }
   }

   // MetaController rebalance (every 1000 ticks)
   static int rebalance_counter = 0;
   if(++rebalance_counter >= 1000) { g_meta.Rebalance(); rebalance_counter = 0; }

   } // end if(g_state.CanOpenNew())

   // --- Periodic diagnostic heartbeat (every DIAG_EVERY ticks) ---
   if(++g_diag_tick >= DIAG_EVERY)
   {
      g_diag_tick = 0;
      PrintFormat("HYDRA4 DIAG [%s] state=%s  pos=%d",
                  TimeToString(now, TIME_DATE|TIME_SECONDS),
                  g_state.StateStr(), PositionsTotal());
      for(int ai = 0; ai < g_n_agents; ai++)
      {
         if(!g_agents[ai].loaded) continue;
         string sym = g_diag[ai].symbol;
         if(sym == "") sym = g_agents[ai].canonical;
         string reason = g_diag[ai].skip_reason;
         if(reason == "") reason = "pending";

         // Direction bias over last BIAS_WINDOW signals
         string bias_str = "n/a";
         int n_bias = g_bias_count[ai];
         if(n_bias > 0)
         {
            int n_long = 0, n_short = 0;
            for(int b = 0; b < n_bias; b++)
            {
               int d = g_bias_buf[ai * BIAS_WINDOW + b];
               if(d > 0) n_long++; else if(d < 0) n_short++;
            }
            double long_pct = 100.0 * n_long / n_bias;
            bias_str = StringFormat("L=%.0f%% S=%.0f%% (n=%d)%s",
                          long_pct, 100.0 - long_pct, n_bias,
                          (long_pct > 70.0 || long_pct < 30.0) ? " !" : "");
         }

         PrintFormat("  [%s/%s] mkt=%s enc=%s  dir=%d conf=%.3f unc=%.3f sess=%.2f  => %s  bias:%s",
            g_diag[ai].agent, sym,
            g_diag[ai].mkt_open  ? "Y" : "N",
            g_diag[ai].enc_valid ? "Y" : "N",
            g_diag[ai].direction,
            g_diag[ai].conf,
            g_diag[ai].uncert,
            g_diag[ai].session_gate,
            reason,
            bias_str);
      }
   }

   // Dashboard (every N bars)
   if(++g_bar_counter >= InpDashboardRefreshBars)
   {
      g_dashboard.Render(g_dash_rows, g_n_agents);
      g_bar_counter = 0;
   }
}

//+------------------------------------------------------------------+
//| OnTradeTransaction — record closed trades for MetaController      |
//+------------------------------------------------------------------+

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result_tr)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(trans.deal_type != DEAL_TYPE_BUY && trans.deal_type != DEAL_TYPE_SELL) return;

   ulong deal_ticket = trans.deal;
   if(!HistoryDealSelect(deal_ticket)) return;
   if(HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != InpMagic) return;
   if(HistoryDealGetInteger(deal_ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) return;

   double pnl       = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
   double exit_px   = HistoryDealGetDouble(deal_ticket, DEAL_PRICE);
   double volume    = HistoryDealGetDouble(deal_ticket, DEAL_VOLUME);
   ulong  pos_id    = (ulong)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID);
   string bname     = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
   string can       = bname;
   for(int s = 0; s < g_sym_count; s++)
      if(g_sym_specs[s].broker_name == bname) { can = g_sym_specs[s].canonical; break; }

   // Determine original direction (+1 LONG / -1 SHORT) from the exit deal type.
   // EXIT deal type is inverse of entry: DEAL_TYPE_SELL = closing a LONG position.
   int direction = (trans.deal_type == DEAL_TYPE_SELL) ? 1 : -1;

   // Look up the entry deal for this position to get entry price and SL/TP
   double entry_px = exit_px;   // fallback if history unavailable
   double sl_pips_closed = 0.0, tp_pips_closed = 0.0;
   datetime entry_time = 0;
   if(HistorySelectByPosition(pos_id))
   {
      int n_deals = HistoryDealsTotal();
      for(int d = 0; d < n_deals; d++)
      {
         ulong dh = HistoryDealGetTicket(d);
         if(HistoryDealGetInteger(dh, DEAL_ENTRY) == DEAL_ENTRY_IN &&
            HistoryDealGetInteger(dh, DEAL_MAGIC) == InpMagic)
         {
            entry_px   = HistoryDealGetDouble(dh, DEAL_PRICE);
            entry_time = (datetime)HistoryDealGetInteger(dh, DEAL_TIME);
            break;
         }
      }
   }

   // Compute pips (signed: positive = profitable direction)
   double pip_size = g_broker.PipSize(can);
   double pips = 0.0;
   if(pip_size > 0.0)
      pips = (direction > 0)
             ? (exit_px - entry_px) / pip_size
             : (entry_px - exit_px) / pip_size;

   // Hold time in M5 bars
   int hold_bars = (entry_time > 0 && TimeCurrent() > entry_time)
                   ? (int)((TimeCurrent() - entry_time) / 300)
                   : 0;

   // Find agent name for this symbol
   string agent_name = "UNKNOWN";
   int    ag_id      = -1;
   for(int ai = 0; ai < g_n_agents; ai++)
      if(g_agents[ai].canonical == can)
      {
         agent_name = g_agents[ai].agent_type;
         ag_id      = AgentToID(agent_name);
         break;
      }

   // Record PnL for MetaController (CRO + supervisor + Sharpe tracking)
   if(ag_id >= 0) g_meta.RecordPnL(ag_id, pnl, (int)_SymbolAssetClass(can));

   // -----------------------------------------------------------------------
   // Pull MAE/MFE from position tracker (if registered) and write both the
   // closed-trade log and the resolved-signal log.
   // -----------------------------------------------------------------------
   double mae_pips = 0.0, mfe_pips = 0.0;
   string close_reason = (pnl >= 0.0) ? "win" : "loss";

   int tr_idx = _FindTracker(pos_id);
   if(tr_idx >= 0)
   {
      SPositionTracker tr = g_pos_tracker[tr_idx];
      double ps = tr.pip_size > 0 ? tr.pip_size : pip_size;
      if(ps > 0 && tr.entry_price > 0)
      {
         if(tr.direction > 0)
         {
            mae_pips = (tr.entry_price - tr.worst_price) / ps;   // how far it went against us
            mfe_pips = (tr.best_price  - tr.entry_price) / ps;   // how far it moved in our favour
         }
         else
         {
            mae_pips = (tr.worst_price - tr.entry_price) / ps;
            mfe_pips = (tr.entry_price - tr.best_price)  / ps;
         }
         mae_pips = MathMax(0.0, mae_pips);
         mfe_pips = MathMax(0.0, mfe_pips);
      }

      // Write resolved signal (links predictions → outcome for retrain label injection)
      g_run_logger.LogResolvedSignal(
         can, agent_name, direction,
         (double)tr.confidence, (double)tr.uncertainty, (double)tr.timing,
         (double)tr.sl_pips_pred, (double)tr.tp_pips_pred,
         (double)tr.vol_mult, (double)tr.session_gate,
         tr.signal_time,
         entry_px, exit_px, pips, pnl, hold_bars,
         mae_pips, mfe_pips, close_reason);

      _RemoveTracker(tr_idx);
   }

   // Log to dedicated closed trades CSV (clean schema, no blank columns)
   g_run_logger.LogClosedTrade(
      can, agent_name, direction,
      entry_px, exit_px, pips, pnl, volume,
      hold_bars, sl_pips_closed, tp_pips_closed,
      mae_pips, mfe_pips, close_reason);

   PrintFormat("HYDRA4 CLOSED %s %s  pips=%.1f  pnl=%.2f  mae=%.1f  mfe=%.1f  hold=%d bars",
               (direction > 0) ? "LONG" : "SHORT", can, pips, pnl,
               mae_pips, mfe_pips, hold_bars);
}

//+------------------------------------------------------------------+
//| Extended TradeStacker helper — place entry with explicit SL/TP    |
//+------------------------------------------------------------------+

// Forward-declared helper (extend CTradeStacker without modifying mk3 include)
int PlaceEntryWithSLTP(const string canonical, int direction,
                       double sl_dist, double tp_dist,
                       double lot, double meta_weight)
{
   if(!g_broker.IsMarketOpen(canonical, TimeCurrent())) return 0;
   if(!g_sizer.SpreadOK(canonical)) return 0;
   if(!g_risk.CanOpenNew()) return 0;

   string bname  = g_broker.BrokerName(canonical);
   int    digits = g_broker.Digits(canonical);
   double pt     = g_broker.PointVal(canonical);
   int    freeze = g_broker.FreezeLevel(canonical);

   double ask    = SymbolInfoDouble(bname, SYMBOL_ASK);
   double bid    = SymbolInfoDouble(bname, SYMBOL_BID);
   double spread = ask - bid;

   // Enforce broker minimum SL/TP distance:
   //   stops_level enforces minimum points from market price
   //   spread must be cleared for the order side (buy SL is below bid)
   //   +4 points buffer to prevent edge rejections
   int    stops_lv = (int)SymbolInfoInteger(bname, SYMBOL_TRADE_STOPS_LEVEL);
   double min_dist = (stops_lv + freeze + 4) * pt + spread;
   sl_dist = MathMax(sl_dist, min_dist);
   tp_dist = MathMax(tp_dist, min_dist);

   double entry = (direction > 0) ? ask : bid;
   double sl = (direction > 0) ? NormalizeDouble(entry - sl_dist, digits)
                                : NormalizeDouble(entry + sl_dist, digits);
   double tp = (direction > 0) ? NormalizeDouble(entry + tp_dist, digits)
                                : NormalizeDouble(entry - tp_dist, digits);

   // lot is already risk-sized by CalcLotFromSL + vol_mult + meta lot_mult.
   // Do not multiply by meta_weight again — that was causing 4× margin saturation.
   lot = g_broker.NormalizeLot(canonical, lot);
   if(lot < g_broker.VolumeMin(canonical)) return 0;

   CTrade trd;
   trd.SetExpertMagicNumber(InpMagic);
   trd.SetDeviationInPoints(10);

   bool ok;
   string comment = "HYDRA4_" + canonical;
   if(direction > 0)
      ok = trd.Buy(lot, bname, ask, sl, tp, comment);
   else
      ok = trd.Sell(lot, bname, bid, sl, tp, comment);

   if(ok) g_risk.CountOpen();
   return ok ? 1 : 0;
}
