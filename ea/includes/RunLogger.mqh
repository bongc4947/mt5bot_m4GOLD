#ifndef RUNLOGGER_MQH
#define RUNLOGGER_MQH

#include "Defines.mqh"

//=============================================================================
// CRunLogger — Structured EA-side session and signal logging.
//
// Writes five files to Common Files:
//
//  HYDRA4_ea_session_<date>.json
//      One per EA start.  Records loaded models, dims, val_acc.
//
//  HYDRA4_ea_signals.csv
//      One row per bar per symbol.  Entry signal + outcome flag.
//
//  HYDRA4_closed_trades.csv
//      One row per closed trade.  Pips, PnL, MAE, MFE, hold_bars.
//
//  HYDRA4_skipped_signals.csv
//      One row every time a signal is generated but NOT traded.
//      Used to audit how much edge we are filtering away.
//
//  HYDRA4_resolved_signals.csv
//      One row per closed trade linking model predictions at signal time
//      to the actual outcome.  Primary source for retrain label injection
//      and execution-model calibration.
//
//  HYDRA4_mod_events.csv
//      One row each time the modification model fires a non-HOLD action
//      (BE move, trail adjust, early close).
//=============================================================================

class CRunLogger
{
private:
   int      m_signal_fh;       // HYDRA4_ea_signals.csv
   int      m_trades_fh;       // HYDRA4_closed_trades.csv
   int      m_skipped_fh;      // HYDRA4_skipped_signals.csv
   int      m_resolved_fh;     // HYDRA4_resolved_signals.csv
   int      m_mod_fh;          // HYDRA4_mod_events.csv
   int      m_compliance_fh;   // HYDRA4_compliance_log.csv  (Phase 1 BCF)
   int      m_drm_fh;          // HYDRA4_drm_events.csv      (Phase 4 DRM)
   string   m_session_path;
   string   m_models_buf;
   int      m_n_models;
   bool     m_open;
   int      m_n_active;
   int      m_n_missing;
   string   m_started_at;

   string EscStr(const string s) { return "\"" + s + "\""; }

   //--- Open a CSV in append mode, write header if new.
   //    Header is written via FileWriteString so commas in the header string
   //    are preserved as column separators, not CSV-escaped into a single field.
   int _OpenCSV(const string fname, const string header)
   {
      bool write_hdr = !FileIsExist(fname, FILE_COMMON);
      int fh = FileOpen(fname,
         FILE_WRITE | FILE_READ | FILE_CSV | FILE_COMMON | FILE_SHARE_READ, ',');
      if(fh == INVALID_HANDLE) return INVALID_HANDLE;
      FileSeek(fh, 0, SEEK_END);
      if(write_hdr)
         FileWriteString(fh, header + "\n");
      return fh;
   }

public:
   CRunLogger() : m_signal_fh(INVALID_HANDLE), m_trades_fh(INVALID_HANDLE),
                  m_skipped_fh(INVALID_HANDLE), m_resolved_fh(INVALID_HANDLE),
                  m_mod_fh(INVALID_HANDLE),
                  m_compliance_fh(INVALID_HANDLE), m_drm_fh(INVALID_HANDLE),
                  m_n_models(0), m_open(false),
                  m_n_active(0), m_n_missing(0) {}

   ~CRunLogger() { Close(); }

   //--------------------------------------------------------------------------
   // OpenSession — call from OnInit BEFORE LoadAgents
   //--------------------------------------------------------------------------
   void OpenSession(int n_active = 0, int n_missing = 0)
   {
      MqlDateTime mdt; TimeToStruct(TimeCurrent(), mdt);
      m_session_path = StringFormat("HYDRA4_ea_session_%04d%02d%02d_%02d%02d%02d.json",
         mdt.year, mdt.mon, mdt.day, mdt.hour, mdt.min, mdt.sec);

      m_started_at  = TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS);
      m_n_active    = n_active;
      m_n_missing   = n_missing;
      m_models_buf  = "";
      m_n_models    = 0;
      m_open        = true;

      // --- Entry signals CSV ---
      m_signal_fh = _OpenCSV("HYDRA4_ea_signals.csv",
         "timestamp,symbol,agent,direction,confidence,uncertainty,regime,"
         "spread_pips,equity_dd_pct,timing,sl_pips,tp_pips,vol_mult,"
         "session_gate,trade_opened,skip_reason");

      // --- Closed trades CSV ---
      m_trades_fh = _OpenCSV("HYDRA4_closed_trades.csv",
         "timestamp,symbol,agent,direction,entry_price,exit_price,"
         "pips,pnl_usd,lot_size,hold_bars,sl_pips,tp_pips,"
         "mae_pips,mfe_pips,close_reason");

      // --- Skipped signals CSV ---
      m_skipped_fh = _OpenCSV("HYDRA4_skipped_signals.csv",
         "timestamp,symbol,agent,direction,confidence,uncertainty,"
         "timing,sl_pips,tp_pips,spread_pips,equity_dd_pct,skip_reason");

      // --- Resolved signals CSV (signal → outcome link) ---
      m_resolved_fh = _OpenCSV("HYDRA4_resolved_signals.csv",
         "timestamp_signal,symbol,agent,direction,"
         "confidence,uncertainty,timing,sl_pips_pred,tp_pips_pred,vol_mult,session_gate,"
         "entry_price,exit_price,actual_pips,actual_pnl,hold_bars,"
         "mae_pips,mfe_pips,close_reason");

      // --- Modification events CSV ---
      m_mod_fh = _OpenCSV("HYDRA4_mod_events.csv",
         "timestamp,symbol,ticket,action,"
         "confidence,floating_pnl_pips,sl_pips_before,sl_pips_after,close_now_conf");

      // mk4.2: BCF compliance + DRM mode-change CSVs removed.
   }

   //--------------------------------------------------------------------------
   // LogModelLoaded — call once per successfully loaded model
   //--------------------------------------------------------------------------
   void LogModelLoaded(const string agent, const string symbol,
                       int feat_dim, int exec_dim, int mod_dim,
                       double val_acc, const string meta_path,
                       bool loaded)
   {
      if(!m_open) return;
      if(m_n_models > 0) m_models_buf += ",\n";
      m_models_buf += "    {\n";
      m_models_buf += "      \"agent\": "       + EscStr(agent)     + ",\n";
      m_models_buf += "      \"symbol\": "      + EscStr(symbol)    + ",\n";
      m_models_buf += StringFormat("      \"feat_dim\": %d,\n",      feat_dim);
      m_models_buf += StringFormat("      \"exec_dim\": %d,\n",      exec_dim);
      m_models_buf += StringFormat("      \"mod_dim\": %d,\n",       mod_dim);
      m_models_buf += StringFormat("      \"val_acc\": %.4f,\n",     val_acc);
      m_models_buf += "      \"meta\": "        + EscStr(meta_path) + ",\n";
      m_models_buf += StringFormat("      \"loaded\": %s\n",         loaded ? "true" : "false");
      m_models_buf += "    }";
      m_n_models++;
   }

   void SetCounts(int n_active, int n_missing)
   {
      m_n_active  = n_active;
      m_n_missing = n_missing;
   }

   //--------------------------------------------------------------------------
   // FlushSession — build full session JSON and write to disk
   //--------------------------------------------------------------------------
   void FlushSession()
   {
      if(!m_open) return;
      string buf = "{\n";
      buf += "  \"hydra_version\": " + EscStr(HYDRA_VERSION) + ",\n";
      buf += StringFormat("  \"magic\": %d,\n", HYDRA_MAGIC);
      buf += "  \"started_at\": " + EscStr(m_started_at) + ",\n";
      buf += StringFormat("  \"n_active\": %d,\n",          m_n_active);
      buf += StringFormat("  \"n_missing\": %d,\n",         m_n_missing);
      buf += StringFormat("  \"feature_dim\": %d,\n",       FEATURE_DIM);
      buf += StringFormat("  \"exec_feature_dim\": %d,\n",  EXEC_FEATURE_DIM);
      buf += StringFormat("  \"mod_feature_dim\": %d,\n",   MOD_FEATURE_DIM);
      buf += "  \"models\": [\n";
      buf += m_models_buf;
      buf += "\n  ]\n}\n";

      int fh = FileOpen(m_session_path, FILE_WRITE | FILE_TXT | FILE_COMMON);
      if(fh != INVALID_HANDLE)
      {
         FileWriteString(fh, buf);
         FileClose(fh);
         PrintFormat("RunLogger: session log → %s  active=%d  missing=%d",
                     m_session_path, m_n_active, m_n_missing);
      }
      else PrintFormat("RunLogger: WARN could not write session log err=%d", GetLastError());
   }

   //--------------------------------------------------------------------------
   // LogSignal — one row per bar per symbol (entry signals)
   //--------------------------------------------------------------------------
   void LogSignal(const string symbol, const string agent,
                  int direction, double confidence, double uncertainty,
                  int regime, double spread_pips, double equity_dd_pct,
                  double timing, double sl_pips, double tp_pips,
                  double vol_mult, double session_gate,
                  bool trade_opened, const string skip_reason = "")
   {
      if(m_signal_fh == INVALID_HANDLE) return;
      FileWrite(m_signal_fh,
         TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
         symbol, agent, direction,
         DoubleToString(confidence,   4),
         DoubleToString(uncertainty,  4),
         regime,
         DoubleToString(spread_pips,  2),
         DoubleToString(equity_dd_pct,4),
         DoubleToString(timing,       4),
         DoubleToString(sl_pips,      1),
         DoubleToString(tp_pips,      1),
         DoubleToString(vol_mult,     3),
         DoubleToString(session_gate, 4),
         (int)trade_opened,
         skip_reason);
      FileFlush(m_signal_fh);
   }

   //--------------------------------------------------------------------------
   // LogSkippedSignal — written whenever a signal with real model output is
   // suppressed (rr_fail / conf_low / timing_low / long_disabled / etc.)
   // Not written for pre-model skips (no_enc / mkt_closed).
   //--------------------------------------------------------------------------
   void LogSkippedSignal(const string symbol, const string agent,
                         int direction, double confidence, double uncertainty,
                         double timing, double sl_pips, double tp_pips,
                         double spread_pips, double equity_dd_pct,
                         const string skip_reason)
   {
      if(m_skipped_fh == INVALID_HANDLE) return;
      FileWrite(m_skipped_fh,
         TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
         symbol, agent, direction,
         DoubleToString(confidence,   4),
         DoubleToString(uncertainty,  4),
         DoubleToString(timing,       4),
         DoubleToString(sl_pips,      1),
         DoubleToString(tp_pips,      1),
         DoubleToString(spread_pips,  2),
         DoubleToString(equity_dd_pct,4),
         skip_reason);
      FileFlush(m_skipped_fh);
   }

   //--------------------------------------------------------------------------
   // LogClosedTrade — written on every OnTradeTransaction(DEAL_ENTRY_OUT)
   //--------------------------------------------------------------------------
   void LogClosedTrade(const string symbol, const string agent,
                       int    direction,
                       double entry_price, double exit_price,
                       double pips, double pnl_usd, double lot_size,
                       int    hold_bars, double sl_pips, double tp_pips,
                       double mae_pips = 0.0, double mfe_pips = 0.0,
                       const string close_reason = "")
   {
      if(m_trades_fh == INVALID_HANDLE) return;
      FileWrite(m_trades_fh,
         TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
         symbol, agent, direction,
         DoubleToString(entry_price, 5),
         DoubleToString(exit_price,  5),
         DoubleToString(pips,        1),
         DoubleToString(pnl_usd,     2),
         DoubleToString(lot_size,    2),
         hold_bars,
         DoubleToString(sl_pips,     1),
         DoubleToString(tp_pips,     1),
         DoubleToString(mae_pips,    1),
         DoubleToString(mfe_pips,    1),
         close_reason);
      FileFlush(m_trades_fh);
   }

   //--------------------------------------------------------------------------
   // LogResolvedSignal — links model predictions at signal-fire time to the
   // actual trade outcome.  Written at close time, not at signal time.
   // This is the primary data source for retrain label injection.
   //--------------------------------------------------------------------------
   void LogResolvedSignal(const string symbol, const string agent,
                          int    direction,
                          double confidence, double uncertainty,
                          double timing,
                          double sl_pips_pred, double tp_pips_pred, double vol_mult,
                          double session_gate,
                          datetime signal_time,
                          double entry_price, double exit_price,
                          double actual_pips, double actual_pnl,
                          int    hold_bars,
                          double mae_pips, double mfe_pips,
                          const string close_reason = "")
   {
      if(m_resolved_fh == INVALID_HANDLE) return;
      FileWrite(m_resolved_fh,
         TimeToString(signal_time, TIME_DATE | TIME_SECONDS),
         symbol, agent, direction,
         DoubleToString(confidence,    4),
         DoubleToString(uncertainty,   4),
         DoubleToString(timing,        4),
         DoubleToString(sl_pips_pred,  1),
         DoubleToString(tp_pips_pred,  1),
         DoubleToString(vol_mult,      3),
         DoubleToString(session_gate,  4),
         DoubleToString(entry_price,   5),
         DoubleToString(exit_price,    5),
         DoubleToString(actual_pips,   1),
         DoubleToString(actual_pnl,    2),
         hold_bars,
         DoubleToString(mae_pips,      1),
         DoubleToString(mfe_pips,      1),
         close_reason);
      FileFlush(m_resolved_fh);
   }

   //--------------------------------------------------------------------------
   // LogModEvent — written when the mod model fires a non-HOLD action.
   // action: "BE" | "TRAIL" | "CLOSE_EARLY"
   //--------------------------------------------------------------------------
   void LogModEvent(const string symbol, ulong ticket,
                    const string action,
                    double mod_confidence,
                    double floating_pnl_pips,
                    double sl_pips_before, double sl_pips_after,
                    double close_now_conf)
   {
      if(m_mod_fh == INVALID_HANDLE) return;
      FileWrite(m_mod_fh,
         TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
         symbol,
         (string)ticket,
         action,
         DoubleToString(mod_confidence,    4),
         DoubleToString(floating_pnl_pips, 1),
         DoubleToString(sl_pips_before,    1),
         DoubleToString(sl_pips_after,     1),
         DoubleToString(close_now_conf,    4));
      FileFlush(m_mod_fh);
   }

   //--------------------------------------------------------------------------
   // LogComplianceEvent — one row per BCF evaluation (shadow mode).
   // result: "PASS" | "BLOCK" (shadow-mode: always PASS, but still logged)
   //--------------------------------------------------------------------------
   void LogComplianceEvent(const string symbol, const string agent,
                           double bcf_score, const string result,
                           double confidence, double sl_pips, double spread_pips)
   {
      if(m_compliance_fh == INVALID_HANDLE) return;
      FileWrite(m_compliance_fh,
         TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
         symbol,
         agent,
         DoubleToString(bcf_score,    4),
         result,
         DoubleToString(confidence,   4),
         DoubleToString(sl_pips,      1),
         DoubleToString(spread_pips,  2));
      FileFlush(m_compliance_fh);
   }

   //--------------------------------------------------------------------------
   // LogDRMEvent — one row each time DRM trading mode changes.
   //--------------------------------------------------------------------------
   void LogDRMEvent(const string old_mode, const string new_mode,
                    double vol_regime, int risk_event,
                    double max_risk, int max_concurrent)
   {
      if(m_drm_fh == INVALID_HANDLE) return;
      FileWrite(m_drm_fh,
         TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
         old_mode,
         new_mode,
         DoubleToString(vol_regime,  4),
         (string)risk_event,
         DoubleToString(max_risk,    4),
         (string)max_concurrent);
      FileFlush(m_drm_fh);
   }

   //--------------------------------------------------------------------------
   // Close — flush and close all file handles
   //--------------------------------------------------------------------------
   void Close()
   {
      int handles[7];
      handles[0] = m_signal_fh;
      handles[1] = m_trades_fh;
      handles[2] = m_skipped_fh;
      handles[3] = m_resolved_fh;
      handles[4] = m_mod_fh;
      handles[5] = m_compliance_fh;
      handles[6] = m_drm_fh;
      for(int i = 0; i < 7; i++)
         if(handles[i] != INVALID_HANDLE) FileClose(handles[i]);
      m_signal_fh     = INVALID_HANDLE;
      m_trades_fh     = INVALID_HANDLE;
      m_skipped_fh    = INVALID_HANDLE;
      m_resolved_fh   = INVALID_HANDLE;
      m_mod_fh        = INVALID_HANDLE;
      m_compliance_fh = INVALID_HANDLE;
      m_drm_fh        = INVALID_HANDLE;
      m_open          = false;
   }
};

CRunLogger g_run_logger;

#endif // RUNLOGGER_MQH
