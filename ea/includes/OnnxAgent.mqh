#ifndef ONNXAGENT_MQH
#define ONNXAGENT_MQH

#include "Defines.mqh"

//=============================================================================
// COnnxAgent — wraps OnnxCreate/OnnxRun for direction model (det + mc pair).
// One instance per agent-symbol combination.
//=============================================================================

class COnnxAgent
{
private:
   long   m_model_det;       // deterministic ONNX handle
   long   m_model_mc;        // MC dropout ONNX handle
   string m_symbol;
   string m_agent_type;
   int    m_feature_dim;
   string m_onnx_det_path;
   string m_onnx_mc_path;
   string m_meta_path;
   datetime m_last_loaded;
   double   m_last_val_acc;
   string   m_version;
   bool     m_loaded;

   //--- Load one ONNX file from Common Files via buffer, set input shape
   long LoadOne(const string path, int feature_dim)
   {
      uchar buf[];
      if(FileLoad(path, buf, FILE_COMMON) <= 0)
      {
         PrintFormat("OnnxAgent[%s/%s]: FileLoad failed for %s err=%d",
                     m_agent_type, m_symbol, path, GetLastError());
         return INVALID_HANDLE;
      }

      long handle = OnnxCreateFromBuffer(buf, ONNX_DEFAULT);
      if(handle == INVALID_HANDLE)
      {
         PrintFormat("OnnxAgent[%s/%s]: OnnxCreateFromBuffer failed for %s err=%d",
                     m_agent_type, m_symbol, path, GetLastError());
         return INVALID_HANDLE;
      }

      // Use feature_dim read from meta.json — NOT the compiled constant.
      // This prevents err=5805 when the ONNX file was trained with a different
      // architecture than the current EA compile (e.g. old 80-dim vs new 96-dim).
      ulong shape[2] = {1, (ulong)feature_dim};
      if(!OnnxSetInputShape(handle, 0, shape))
      {
         PrintFormat("OnnxAgent[%s/%s]: OnnxSetInputShape failed dim=%d err=%d",
                     m_agent_type, m_symbol, feature_dim, GetLastError());
         OnnxRelease(handle);
         return INVALID_HANDLE;
      }
      // Direction model output is [1, 1]. MT5 cannot infer dimension[0] without
      // this call → "wrong dimension[0], try to use OnnxSetOutputShape" on every tick.
      ulong out_shape[2] = {1, 1};
      if(!OnnxSetOutputShape(handle, 0, out_shape))
         PrintFormat("OnnxAgent[%s/%s]: OnnxSetOutputShape warn err=%d",
                     m_agent_type, m_symbol, GetLastError());
      return handle;
   }

   //--- Load meta JSON content via FileLoad (same method used for ONNX files)
   string LoadMetaContent(const string path)
   {
      uchar buf[];
      int n = (int)FileLoad(path, buf, FILE_COMMON);
      if(n <= 0)
      {
         PrintFormat("OnnxAgent[%s/%s]: meta FileLoad failed: %s err=%d",
                     m_agent_type, m_symbol, path, GetLastError());
         return "";
      }
      return CharArrayToString(buf, 0, n, CP_UTF8);
   }

   //--- Parse feature_dim from meta JSON (falls back to compiled FEATURE_DIM)
   int ParseFeatureDim(const string path)
   {
      string content = LoadMetaContent(path);
      if(content == "") return FEATURE_DIM;

      int pos = StringFind(content, "\"feature_dim\"");
      if(pos < 0) return FEATURE_DIM;
      int colon = StringFind(content, ":", pos);
      if(colon < 0) return FEATURE_DIM;
      int end = StringFind(content, ",", colon);
      if(end < 0) end = StringFind(content, "}", colon);
      if(end < 0) return FEATURE_DIM;
      int dim = (int)StringToInteger(StringSubstr(content, colon + 1, end - colon - 1));
      return (dim > 0) ? dim : FEATURE_DIM;
   }

   //--- Parse val_acc from meta JSON
   double ParseValAcc(const string path)
   {
      string content = LoadMetaContent(path);
      if(content == "") return 0.0;

      int pos = StringFind(content, "\"val_acc\"");
      if(pos < 0) return 0.0;
      int colon = StringFind(content, ":", pos);
      if(colon < 0) return 0.0;
      int end = StringFind(content, ",", colon);
      if(end < 0) end = StringFind(content, "}", colon);
      if(end < 0) return 0.0;
      return StringToDouble(StringSubstr(content, colon + 1, end - colon - 1));
   }

   //--- Parse checksum from meta JSON
   string ParseChecksum(const string path)
   {
      string content = LoadMetaContent(path);
      if(content == "") return "";

      int pos = StringFind(content, "\"checksum\"");
      if(pos < 0) return "";
      int q1 = StringFind(content, "\"", StringFind(content, ":", pos));
      if(q1 < 0) return "";
      int q2 = StringFind(content, "\"", q1 + 1);
      if(q2 < 0) return "";
      return StringSubstr(content, q1 + 1, q2 - q1 - 1);
   }

public:
   COnnxAgent()
      : m_model_det(INVALID_HANDLE), m_model_mc(INVALID_HANDLE),
        m_feature_dim(FEATURE_DIM), m_last_loaded(0),
        m_last_val_acc(0.0), m_loaded(false) {}

   ~COnnxAgent() { Release(); }

   //--- Initialize paths (call before Load)
   void Init(const string agent_type, const string symbol, int feature_dim = FEATURE_DIM)
   {
      m_agent_type   = agent_type;
      m_symbol       = symbol;
      m_feature_dim  = feature_dim;

      string prefix = "HYDRA4_" + agent_type + "_" + symbol;
      m_onnx_det_path = prefix + "_dir_det.onnx";
      m_onnx_mc_path  = prefix + "_dir_mc.onnx";
      m_meta_path     = prefix + "_meta.json";
   }

   //--- Load / reload both ONNX files. Returns true on success.
   bool Load()
   {
      Release();

      // Read feature_dim from meta.json — overrides compiled constant so old ONNX
      // files (e.g. 80-dim) load without err=5805 after a Defines.mqh recompile.
      int feat_dim = ParseFeatureDim(m_meta_path);
      m_feature_dim = feat_dim;

      long det = LoadOne(m_onnx_det_path, feat_dim);
      if(det == INVALID_HANDLE) return false;

      long mc = LoadOne(m_onnx_mc_path, feat_dim);
      if(mc == INVALID_HANDLE)
      {
         OnnxRelease(det);
         return false;
      }

      m_model_det   = det;
      m_model_mc    = mc;
      m_last_loaded = TimeCurrent();
      m_last_val_acc = ParseValAcc(m_meta_path);
      m_loaded = true;

      PrintFormat("OnnxAgent[%s/%s]: loaded det+mc  feat_dim=%d  val_acc=%.3f",
                  m_agent_type, m_symbol, feat_dim, m_last_val_acc);
      return true;
   }

   void Release()
   {
      if(m_model_det != INVALID_HANDLE) { OnnxRelease(m_model_det); m_model_det = INVALID_HANDLE; }
      if(m_model_mc  != INVALID_HANDLE) { OnnxRelease(m_model_mc);  m_model_mc  = INVALID_HANDLE; }
      m_loaded = false;
   }

   bool IsLoaded() const { return m_loaded; }
   string Symbol() const { return m_symbol; }
   string AgentType() const { return m_agent_type; }
   double ValAcc() const { return m_last_val_acc; }
   datetime LastLoaded() const { return m_last_loaded; }
   string MetaPath() const { return m_meta_path; }
   int FeatureDim() const { return m_feature_dim; }

   //--- Single deterministic inference. Returns raw logit.
   float Infer(const float &x[])
   {
      if(m_model_det == INVALID_HANDLE) return 0.0f;

      float feat[];
      ArrayResize(feat, m_feature_dim);
      ArrayCopy(feat, x, 0, 0, m_feature_dim);

      float out_arr[1];
      if(!OnnxRun(m_model_det, ONNX_NO_CONVERSION, feat, out_arr))
      {
         Print("OnnxAgent Infer failed: ", GetLastError());
         return 0.0f;
      }
      return out_arr[0];
   }

   //--- MC uncertainty: T passes with dropout model → mu, sigma_ep
   void InferMC(const float &x[], int T, float &mu, float &sigma_ep)
   {
      if(m_model_mc == INVALID_HANDLE) { mu = 0.0f; sigma_ep = 1.0f; return; }

      float mc_feat[];
      ArrayResize(mc_feat, m_feature_dim);
      for(int i = 0; i < m_feature_dim; i++) mc_feat[i] = x[i];

      double sum_logit = 0.0, sum_sq = 0.0;
      float mc_out[1];
      for(int t = 0; t < T; t++)
      {
         if(OnnxRun(m_model_mc, ONNX_NO_CONVERSION, mc_feat, mc_out))
         {
            double sig = 1.0 / (1.0 + MathExp(-(double)mc_out[0])); // sigmoid
            sum_logit += sig;
            sum_sq    += sig * sig;
         }
      }
      mu       = (float)(sum_logit / T);
      double var = sum_sq / T - (sum_logit / T) * (sum_logit / T);
      sigma_ep = (float)MathSqrt(MathMax(0.0, var));
   }
};

#endif // ONNXAGENT_MQH
