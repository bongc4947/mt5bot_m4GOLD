#ifndef EXECAGENT_MQH
#define EXECAGENT_MQH

#include "Defines.mqh"

//=============================================================================
// CExecAgent — wraps execution model OnnxRun.
// Input:  float[1, exec_feature_dim] (read from meta.json, falls back to EXEC_FEATURE_DIM)
// Output: float[1, 5] → [timing, sl_pips, tp_pips, vol_mult, session_gate]
//=============================================================================

class CExecAgent
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
         PrintFormat("ExecAgent[%s/%s]: meta FileLoad failed: %s err=%d",
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

   //--- Parse exec_feature_dim from meta JSON.
   //    Priority: explicit "exec_feature_dim" > "feature_dim"+120 > EXEC_FEATURE_DIM.
   //    The explicit field (written by exporter) is the authoritative source.
   int ParseExecFeatureDim(const string path)
   {
      string content = LoadMetaContent(path);
      if(content == "") return EXEC_FEATURE_DIM;

      // 1. Explicit field (written by new exporter — authoritative)
      int explicit_dim = ParseIntField(content, "\"exec_feature_dim\"", -1);
      if(explicit_dim > 0) return explicit_dim;

      // 2. Derive from direction model dim + 120-dim exec context block (mk4 default)
      int dir_dim = ParseIntField(content, "\"feature_dim\"", -1);
      if(dir_dim > 0) return dir_dim + 120;

      // 3. Compiled constant fallback
      return EXEC_FEATURE_DIM;
   }

   //--- Load one ONNX file, set input shape to feature_dim
   long LoadOne(const string path, int feature_dim)
   {
      uchar buf[];
      if(FileLoad(path, buf, FILE_COMMON) <= 0)
      {
         PrintFormat("ExecAgent[%s/%s]: FileLoad failed: %s  err=%d",
                     m_agent_type, m_symbol, path, GetLastError());
         return INVALID_HANDLE;
      }
      long handle = OnnxCreateFromBuffer(buf, ONNX_DEFAULT);
      if(handle == INVALID_HANDLE)
      {
         PrintFormat("ExecAgent[%s/%s]: OnnxCreateFromBuffer failed: %s  err=%d",
                     m_agent_type, m_symbol, path, GetLastError());
         return INVALID_HANDLE;
      }
      ulong shape[2] = {1, (ulong)feature_dim};
      if(!OnnxSetInputShape(handle, 0, shape))
      {
         PrintFormat("ExecAgent[%s/%s]: OnnxSetInputShape failed dim=%d err=%d",
                     m_agent_type, m_symbol, feature_dim, GetLastError());
         OnnxRelease(handle);
         return INVALID_HANDLE;
      }
      // Explicitly set output shape [1,5] so MT5 knows the single concatenated output.
      // Without this, onnxsim-simplified models split torch.cat([...],dim=1) back into
      // 5 separate [1,1] output nodes → OnnxRun fails with ERR_ONNX_INCORRECT_OUTPUT_COUNT.
      ulong out_shape[2] = {1, 5};
      if(!OnnxSetOutputShape(handle, 0, out_shape))
         PrintFormat("ExecAgent[%s/%s]: OnnxSetOutputShape warn err=%d (may be ok)",
                     m_agent_type, m_symbol, GetLastError());
      return handle;
   }

public:
   CExecAgent()
      : m_model(INVALID_HANDLE), m_loaded(false),
        m_feature_dim(EXEC_FEATURE_DIM) {}

   ~CExecAgent() { Release(); }

   void Init(const string agent_type, const string symbol,
             int feature_dim = EXEC_FEATURE_DIM)
   {
      m_agent_type  = agent_type;
      m_symbol      = symbol;
      m_feature_dim = feature_dim;

      string prefix  = "HYDRA4_" + agent_type + "_" + symbol;
      m_onnx_path    = prefix + "_exec_det.onnx";
      m_meta_path    = prefix + "_meta.json";
   }

   bool Load()
   {
      if(m_model != INVALID_HANDLE) { OnnxRelease(m_model); m_model = INVALID_HANDLE; }

      // Read exec_feature_dim from meta.json to avoid err=5805 on dim mismatch.
      int feat_dim = ParseExecFeatureDim(m_meta_path);
      m_feature_dim = feat_dim;

      m_model  = LoadOne(m_onnx_path, feat_dim);
      m_loaded = (m_model != INVALID_HANDLE);
      if(m_loaded)
         PrintFormat("ExecAgent[%s/%s]: loaded  feat_dim=%d", m_agent_type, m_symbol, feat_dim);
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

   //--- Run inference. Returns true and fills outputs on success.
   //    outputs[5]: [timing, sl_pips, tp_pips, vol_mult, session_gate]
   bool Infer(const float &x[], float &outputs[])
   {
      if(m_model == INVALID_HANDLE) return false;

      float exec_feat[];
      ArrayResize(exec_feat, m_feature_dim);
      for(int i = 0; i < m_feature_dim; i++) exec_feat[i] = x[i];

      float exec_out[5];
      if(!OnnxRun(m_model, ONNX_NO_CONVERSION, exec_feat, exec_out))
      {
         Print("ExecAgent Infer failed: ", GetLastError());
         return false;
      }

      ArrayResize(outputs, 5);
      for(int i = 0; i < 5; i++) outputs[i] = exec_out[i];

      // Map vol_mult from [0,1] → [0.5, 2.0]
      outputs[EXEC_IDX_VOL] = (float)(VOL_MULT_MIN +
                               outputs[EXEC_IDX_VOL] * (VOL_MULT_MAX - VOL_MULT_MIN));
      return true;
   }
};

#endif // EXECAGENT_MQH
