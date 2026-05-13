#ifndef MODELWATCHER_MQH
#define MODELWATCHER_MQH

#include "Defines.mqh"
#include "OnnxAgent.mqh"
#include "ExecAgent.mqh"
#include "ModifyAgent.mqh"

//=============================================================================
// CModelWatcher — polls meta.json modification time every MODEL_WATCH_SEC
// seconds; atomically hot-reloads dir + exec + modify models when a change
// is detected (i.e. Python has deployed new ONNX files).
//
// Usage:
//   g_watcher.Register(&dir_agent, &exec_agent, &mod_agent);  // once per symbol
//   g_watcher.CheckAll();                                       // called from OnTimer()
//=============================================================================

struct SWatchEntry
{
   COnnxAgent   *dir;
   CExecAgent   *exec;
   CModifyAgent *mod;
   datetime      last_mtime;   // last-seen meta.json modification datetime
   datetime      last_check;   // last filesystem poll time (throttle guard)
};

class CModelWatcher
{
private:
   SWatchEntry m_entries[MAX_SYMBOLS];
   int         m_count;

public:
   CModelWatcher() : m_count(0) {}

   //--- Register a symbol's three model pointers for hot-reload watching.
   //    Call once per symbol after successful initial Load().
   void Register(COnnxAgent *dir, CExecAgent *exec, CModifyAgent *mod)
   {
      if(m_count >= MAX_SYMBOLS)
      {
         Print("ModelWatcher: MAX_SYMBOLS reached — cannot register more agents");
         return;
      }
      m_entries[m_count].dir        = dir;
      m_entries[m_count].exec       = exec;
      m_entries[m_count].mod        = mod;
      m_entries[m_count].last_mtime = 0;
      m_entries[m_count].last_check = 0;
      m_count++;
   }

   //--- Poll all registered agents. Call from OnTimer().
   //    Throttled per-entry by MODEL_WATCH_SEC.
   //    On meta.json change: reloads dir → exec → modify in order.
   void CheckAll()
   {
      datetime now = TimeCurrent();

      for(int i = 0; i < m_count; i++)
      {
         // Throttle: skip if checked recently
         if(now - m_entries[i].last_check < MODEL_WATCH_SEC) continue;
         m_entries[i].last_check = now;

         COnnxAgent *dir = m_entries[i].dir;
         if(dir == NULL || !dir.IsLoaded()) continue;

         // Use meta.json modification time as change signal
         string meta = dir.MetaPath();
         if(!FileIsExist(meta, FILE_COMMON)) continue;

         datetime mtime = (datetime)FileGetInteger(meta, FILE_MODIFY_DATE, true);
         if(mtime == 0 || mtime == m_entries[i].last_mtime) continue;

         // Detected change — hot-reload all three models for this symbol
         m_entries[i].last_mtime = mtime;
         string sym = dir.Symbol();
         PrintFormat("ModelWatcher: [%s] meta.json changed (mtime=%s) — hot-reloading",
                     sym, TimeToString(mtime));

         bool dir_ok = dir.Load();
         if(!dir_ok)
         {
            PrintFormat("ModelWatcher: [%s] dir reload FAILED — keeping old model", sym);
            continue;
         }

         CExecAgent   *exec = m_entries[i].exec;
         CModifyAgent *mod  = m_entries[i].mod;

         bool exec_ok = (exec != NULL) ? exec.Load() : false;
         bool mod_ok  = (mod  != NULL) ? mod.Load()  : false;

         PrintFormat("ModelWatcher: [%s] hot-reload complete  val_acc=%.3f  exec=%s  mod=%s",
                     sym, dir.ValAcc(),
                     exec_ok ? "ok" : "fail",
                     mod_ok  ? "ok" : "fail");
      }
   }
};

//--- Global instance — referenced by MT5_Bot_mk4_HYDRA.mq5
CModelWatcher g_watcher;

#endif // MODELWATCHER_MQH
