#ifndef MODIFYAGENT_MQH
#define MODIFYAGENT_MQH

#include "Defines.mqh"

//=============================================================================
// CModifyAgent — wraps modification model OnnxRun.
// Input:  float[1, mod_feature_dim] (read from meta.json, falls back to MOD_FEATURE_DIM)
// Output: float[1, 3] → [move_sl_to_be, trail_sl_pips, close_now]
//=============================================================================

class CModifyAgent
{
private:
   long   m_model;
   string m_symbol;
   string m_agent_type;
   string m_onnx_path;
   string m_meta_path;
   bool   m_loaded;
   int    m_feature_dim;

   //--- Load meta JSON content via FileLoad
   string LoadMetaContent(const string path)
   {
      uchar buf[];
      int n = (int)FileLoad(path, buf, FILE_COMMON);
      if(n <= 0)
      {
         PrintFormat("ModifyAgent[%s/%s]: meta FileLoad failed: %s err=%d",
                     m_agent_type, m_symbol, path, GetLastError());
         return "";
      }
      return CharArrayToString(buf, 0, n, CP_UTF8);
   }

   //--- Parse a numeric int field from meta JSON
   int ParseIntField(const string content, const string key, int fallback)
   {
      int pos = StringFind(content, key);
      if(pos < 0) return fallback;
      int colon = StringFind(content, ":", pos);
      if(colon < 0) return fallback;
      int end = StringFind(content, ",", colon);
      if(end < 0) end = StringFind(content, "}", colon);
      if(end < 0) return fallback;
      int val = (int)StringToInteger(StringSubstr(content, colon + 1, end - colon - 1));
      return (val > 0) ? val : fallback;
   }

   //--- Parse mod_feature_dim from meta JSON.
   //    Priority: explicit "mod_feature_dim" > "feature_dim" > MOD_FEATURE_DIM.
   //    Mod model uses the same input dim as the direction model.
   int ParseModFeatureDim(const string path)
   {
      string content = LoadMetaContent(path);
      if(content == "") return MOD_FEATURE_DIM;

      // 1. Explicit field (written by new exporter)
      int explicit_dim = ParseIntField(content, "\"mod_feature_dim\"", -1);
      if(explicit_dim > 0) return explicit_dim;

      // 2. Same as direction model dim (mod model has same input as dir)
      int dir_dim = ParseIntField(content, "\"feature_dim\"", -1);
      if(dir_dim > 0) return dir_dim;

      // 3. Compiled constant fallback
      return MOD_FEATURE_DIM;
   }

   //--- Load one ONNX file, set input shape to feature_dim
   long LoadOne(const string path, int feature_dim)
   {
      uchar buf[];
      if(FileLoad(path, buf, FILE_COMMON) <= 0)
      {
         PrintFormat("ModifyAgent[%s/%s]: FileLoad failed: %s  err=%d",
                     m_agent_type, m_symbol, path, GetLastError());
         return INVALID_HANDLE;
      }
      long handle = OnnxCreateFromBuffer(buf, ONNX_DEFAULT);
      if(handle == INVALID_HANDLE)
      {
         PrintFormat("ModifyAgent[%s/%s]: OnnxCreateFromBuffer failed: %s  err=%d",
                     m_agent_type, m_symbol, path, GetLastError());
         return INVALID_HANDLE;
      }
      ulong shape[2] = {1, (ulong)feature_dim};
      if(!OnnxSetInputShape(handle, 0, shape))
      {
         PrintFormat("ModifyAgent[%s/%s]: OnnxSetInputShape failed dim=%d err=%d",
                     m_agent_type, m_symbol, feature_dim, GetLastError());
         OnnxRelease(handle);
         return INVALID_HANDLE;
      }
      // Explicitly set output shape [1,3] — same onnxsim cat-split issue as exec model.
      ulong out_shape[2] = {1, 3};
      if(!OnnxSetOutputShape(handle, 0, out_shape))
         PrintFormat("ModifyAgent[%s/%s]: OnnxSetOutputShape warn err=%d (may be ok)",
                     m_agent_type, m_symbol, GetLastError());
      return handle;
   }

public:
   CModifyAgent()
      : m_model(INVALID_HANDLE), m_loaded(false),
        m_feature_dim(MOD_FEATURE_DIM) {}

   ~CModifyAgent() { Release(); }

   void Init(const string agent_type, const string symbol)
   {
      m_agent_type  = agent_type;
      m_symbol      = symbol;
      m_feature_dim = MOD_FEATURE_DIM;

      string prefix = "HYDRA4_" + agent_type + "_" + symbol;
      m_onnx_path   = prefix + "_modify_det.onnx";
      m_meta_path   = prefix + "_meta.json";
   }

   bool Load()
   {
      if(m_model != INVALID_HANDLE) { OnnxRelease(m_model); m_model = INVALID_HANDLE; }

      // Read mod_feature_dim from meta.json to avoid err=5805 on dim mismatch.
      int feat_dim = ParseModFeatureDim(m_meta_path);
      m_feature_dim = feat_dim;

      m_model  = LoadOne(m_onnx_path, feat_dim);
      m_loaded = (m_model != INVALID_HANDLE);
      if(m_loaded)
         PrintFormat("ModifyAgent[%s/%s]: loaded  feat_dim=%d", m_agent_type, m_symbol, feat_dim);
      return m_loaded;
   }

   void Release()
   {
      if(m_model != INVALID_HANDLE) { OnnxRelease(m_model); m_model = INVALID_HANDLE; }
      m_loaded = false;
   }

   bool IsLoaded() const { return m_loaded; }
   string MetaPath() const { return m_meta_path; }
   string Symbol() const { return m_symbol; }
   int FeatureDim() const { return m_feature_dim; }

   //--- Run inference. Returns true and fills SModSignal.
   bool Infer(ulong ticket, const float &x[], SModSignal &sig_out)
   {
      sig_out.ticket       = ticket;
      sig_out.move_sl_to_be = 0.0f;
      sig_out.trail_sl_pips = 0.0f;
      sig_out.close_now     = 0.0f;

      if(m_model == INVALID_HANDLE) return false;

      float mod_feat[];
      ArrayResize(mod_feat, m_feature_dim);
      for(int i = 0; i < m_feature_dim; i++) mod_feat[i] = x[i];

      float mod_out[3];
      if(!OnnxRun(m_model, ONNX_NO_CONVERSION, mod_feat, mod_out))
      {
         Print("ModifyAgent Infer failed: ", GetLastError());
         return false;
      }

      sig_out.move_sl_to_be = mod_out[MOD_IDX_MOVE_BE];
      sig_out.trail_sl_pips = mod_out[MOD_IDX_TRAIL];
      sig_out.close_now     = mod_out[MOD_IDX_CLOSE];
      return true;
   }
};

#endif // MODIFYAGENT_MQH
