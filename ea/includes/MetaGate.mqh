//+------------------------------------------------------------------+
//| MetaGate.mqh - MT5-side runner for the meta-trend strategy.       |
//|                                                                    |
//| The validated GOLD edge (docs/RESEARCH_FINDINGS.md): a slow EMA    |
//| 50/200 cross supplies the trade DIRECTION; an XGBoost meta-gate    |
//| (ONNX) decides WHEN to trust it. Net PF ~1.41 under leak-free      |
//| purged CV, 5/6 folds strongly positive.                            |
//|                                                                    |
//| The 18-feature vector built here MUST match python/aurum/          |
//| metatrend.py::build_features bit-for-bit - train/serve parity.     |
//|                                                                    |
//| Bundle (MT5 Common Files):                                         |
//|   M4GOLD_METATREND_GOLD.onnx        meta-gate  float[1,18]->[1,2]  |
//|   M4GOLD_METATREND_GOLD_spec.json   threshold + deploy flag        |
//+------------------------------------------------------------------+
#ifndef METAGATE_MQH
#define METAGATE_MQH

// ---- constants - mirror python/aurum/metatrend.py -------------------
#define MG_EMA_FAST     50
#define MG_EMA_SLOW     200
#define MG_N_FEATURES   18
#define MG_HISTORY      1500     // closed M5 bars pulled (EMA200 converges)

// ---- agent state ----------------------------------------------------
long   g_mg_handle   = INVALID_HANDLE;
bool   g_mg_ready    = false;
bool   g_mg_deploy   = false;
double g_mg_actthr   = 0.55;
string g_mg_version  = "";

//+------------------------------------------------------------------+
//| Minimal JSON helpers (spec is single-byte ASCII - FILE_ANSI).     |
//+------------------------------------------------------------------+
double _MGJsonNum(const string js, const string key, double dflt)
{
   int p = StringFind(js, "\"" + key + "\"");
   if(p < 0) return dflt;
   int c = StringFind(js, ":", p);
   if(c < 0) return dflt;
   int s = c + 1;
   while(s < StringLen(js))
   {
      string ch = StringSubstr(js, s, 1);
      if(ch != " " && ch != "\t" && ch != "\n" && ch != "\r") break;
      s++;
   }
   int e = s;
   while(e < StringLen(js))
   {
      string ch = StringSubstr(js, e, 1);
      if(ch == "," || ch == "}" || ch == "\n") break;
      e++;
   }
   return StringToDouble(StringSubstr(js, s, e - s));
}

bool _MGJsonBool(const string js, const string key)
{
   int p = StringFind(js, "\"" + key + "\"");
   if(p < 0) return false;
   int c = StringFind(js, ":", p);
   if(c < 0) return false;
   return (StringFind(StringSubstr(js, c, 12), "true") >= 0);
}

string _MGJsonStr(const string js, const string key)
{
   int p = StringFind(js, "\"" + key + "\"");
   if(p < 0) return "";
   int c = StringFind(js, ":", p);
   if(c < 0) return "";
   int q1 = StringFind(js, "\"", c);
   if(q1 < 0) return "";
   int q2 = StringFind(js, "\"", q1 + 1);
   if(q2 < 0) return "";
   return StringSubstr(js, q1 + 1, q2 - q1 - 1);
}

//+------------------------------------------------------------------+
//| Load the meta-gate ONNX + spec.                                   |
//+------------------------------------------------------------------+
bool MG_Init()
{
   g_mg_handle = OnnxCreate("M4GOLD_METATREND_GOLD.onnx", ONNX_COMMON_FOLDER);
   if(g_mg_handle == INVALID_HANDLE)
   {
      Print("[MetaGate] M4GOLD_METATREND_GOLD.onnx not found in Common Files.");
      return false;
   }
   // XGBoost ONNX: 1 input float[1,18], 2 outputs (label int64[1] +
   // probabilities float[1,2]). Both output shapes must be declared.
   ulong in_shape[]  = {1, MG_N_FEATURES};
   ulong lbl_shape[] = {1};
   ulong prb_shape[] = {1, 2};
   OnnxSetInputShape (g_mg_handle, 0, in_shape);
   OnnxSetOutputShape(g_mg_handle, 0, lbl_shape);
   OnnxSetOutputShape(g_mg_handle, 1, prb_shape);

   int h = FileOpen("M4GOLD_METATREND_GOLD_spec.json",
                    FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h != INVALID_HANDLE)
   {
      string js = "";
      while(!FileIsEnding(h)) js += FileReadString(h);
      FileClose(h);
      g_mg_actthr  = _MGJsonNum(js, "act_threshold", 0.55);
      g_mg_deploy  = _MGJsonBool(js, "deploy");
      g_mg_version = _MGJsonStr(js, "version");
      if(StringFind(js, "\"strategy\"") < 0)
         Print("[MetaGate] *** WRONG SPEC FILE *** - no strategy key.");
   }
   else
      Print("[MetaGate] *** SPEC MISSING *** - M4GOLD_METATREND_GOLD_spec.json "
            "not in Common Files.");

   g_mg_ready = true;
   PrintFormat("[MetaGate] ready  version=%s  deploy=%s  act_thr=%.2f",
               g_mg_version == "" ? "?" : g_mg_version,
               g_mg_deploy ? "true" : "false", g_mg_actthr);
   return true;
}

void MG_Release()
{
   if(g_mg_handle != INVALID_HANDLE) OnnxRelease(g_mg_handle);
   g_mg_ready = false;
}

//+------------------------------------------------------------------+
//| EMA over a closed-bar close array, forward recursion (matches     |
//| pandas ewm(span, adjust=False): ema[0]=x[0]).                     |
//+------------------------------------------------------------------+
double _MG_EmaLast(const double &close[], int n, int span)
{
   double k = 2.0 / (span + 1.0);
   double e = close[0];
   for(int i = 1; i < n; i++) e = close[i] * k + e * (1.0 - k);
   return e;
}

// sample std (ddof=1) of the last `w` one-bar log returns ending at `last`
double _MG_RetStd(const double &cls[], int last, int w)
{
   double eps = 1e-12, m = 0;
   for(int j = last-w+1; j <= last; j++)
      m += MathLog(MathMax(cls[j],eps) / MathMax(cls[j-1],eps));
   m /= w;
   double s = 0;
   for(int j = last-w+1; j <= last; j++)
   {
      double rr = MathLog(MathMax(cls[j],eps)/MathMax(cls[j-1],eps)) - m;
      s += rr*rr;
   }
   return MathSqrt(s / (w-1.0));
}

//+------------------------------------------------------------------+
//| Build the 18-feature vector. CLOSED bars only (shift 1). Returns  |
//| false if history is insufficient. Mirrors metatrend.build_features|
//+------------------------------------------------------------------+
bool MG_BuildFeatures(float &f[])
{
   ArrayResize(f, MG_N_FEATURES);
   MqlRates r[];
   int got = CopyRates(_Symbol, PERIOD_M5, 1, MG_HISTORY, r);  // shift 1 = closed
   if(got < 320) return false;                                  // need 288 + slack
   int n = ArraySize(r);
   int last = n - 1;                                            // current closed bar
   double eps = 1e-12;

   double cls[], hi[], lo[];
   ArrayResize(cls, n); ArrayResize(hi, n); ArrayResize(lo, n);
   for(int i = 0; i < n; i++)
   { cls[i] = r[i].close; hi[i] = r[i].high; lo[i] = r[i].low; }
   double c = cls[last];

   // --- returns at lags ---
   double ret12 = MathLog(MathMax(c,eps) / MathMax(cls[last-12],eps));
   double ret48 = MathLog(MathMax(c,eps) / MathMax(cls[last-48],eps));
   double ret96 = MathLog(MathMax(c,eps) / MathMax(cls[last-96],eps));

   // --- ATR(14) and ATR(48) - rolling mean of true range ---
   double atr14 = 0, atr48 = 0;
   for(int j = last-13; j <= last; j++)
   {
      double tr = MathMax(r[j].high-r[j].low,
                  MathMax(MathAbs(r[j].high-r[j-1].close),
                          MathAbs(r[j].low-r[j-1].close)));
      atr14 += tr;
   }
   atr14 /= 14.0;
   for(int j = last-47; j <= last; j++)
   {
      double tr = MathMax(r[j].high-r[j].low,
                  MathMax(MathAbs(r[j].high-r[j-1].close),
                          MathAbs(r[j].low-r[j-1].close)));
      atr48 += tr;
   }
   atr48 /= 48.0;

   // --- realised vol: sample std (ddof=1) of 1-bar log returns ---
   double rv24 = _MG_RetStd(cls, last, 24);
   double rv96 = _MG_RetStd(cls, last, 96);

   // --- position in range ---
   double hh96=hi[last], ll96=lo[last], hh288=hi[last], ll288=lo[last];
   for(int j = last-95;  j <= last; j++)
   { if(hi[j]>hh96)  hh96=hi[j];   if(lo[j]<ll96)  ll96=lo[j]; }
   for(int j = last-287; j <= last; j++)
   { if(hi[j]>hh288) hh288=hi[j];  if(lo[j]<ll288) ll288=lo[j]; }
   double pos96  = (c-ll96)  / MathMax(hh96-ll96,  eps);
   double pos288 = (c-ll288) / MathMax(hh288-ll288,eps);

   // --- EMAs ---
   double ef = _MG_EmaLast(cls, n, MG_EMA_FAST);
   double es = _MG_EmaLast(cls, n, MG_EMA_SLOW);
   double ema_fast_dist = (c-ef) / MathMax(atr14,eps);
   double ema_slow_dist = (c-es) / MathMax(atr14,eps);
   double ema_gap       = (ef-es) / MathMax(atr14,eps);

   // --- bars since 96-bar high / low (0=now, ~1=oldest), /96 ---
   int arg_hi = last, arg_lo = last;
   double mhi=hi[last], mlo=lo[last];
   for(int j = last-95; j <= last; j++)
   {
      if(hi[j] >= mhi) { mhi = hi[j]; arg_hi = j; }
      if(lo[j] <= mlo) { mlo = lo[j]; arg_lo = j; }
   }
   double bars_since_hi96 = (double)(last-arg_hi) / 96.0;
   double bars_since_lo96 = (double)(last-arg_lo) / 96.0;

   // --- trend age: bars since the EMA fast/slow cross, /200 capped ---
   //     recompute the EMA pair forward, track the last flip.
   double kf = 2.0/(MG_EMA_FAST+1.0), ks = 2.0/(MG_EMA_SLOW+1.0);
   double e_f = cls[0], e_s = cls[0];
   int prev_dir = 0, age = 0, last_flip_age = 0;
   for(int i = 0; i < n; i++)
   {
      if(i > 0)
      { e_f = cls[i]*kf + e_f*(1.0-kf); e_s = cls[i]*ks + e_s*(1.0-ks); }
      int d = (e_f > e_s) ? 1 : -1;
      if(i == 0) { prev_dir = d; age = 0; }
      else { age = (d != prev_dir) ? 0 : age + 1; prev_dir = d; }
      last_flip_age = age;
   }
   double trend_age = MathMin(last_flip_age / 200.0, 1.0);

   // --- up/down close streak, /10 clamped to [-1,1] ---
   int s = 0;
   for(int i = 1; i < n; i++)
   {
      if(cls[i] > cls[i-1])      s = (s >= 0) ? s+1 : 1;
      else if(cls[i] < cls[i-1]) s = (s <= 0) ? s-1 : -1;
      else                        s = 0;
   }
   double up_streak = MathMax(-1.0, MathMin(1.0, s / 10.0));

   // --- hour-of-day sin/cos (bar open time) ---
   MqlDateTime dt;
   TimeToStruct(r[last].time, dt);
   double hod = dt.hour + dt.min / 60.0;
   double hod_sin = MathSin(2.0*M_PI*hod/24.0);
   double hod_cos = MathCos(2.0*M_PI*hod/24.0);

   // --- pack in the exact META_FEATURES order ---
   f[0]=(float)ret12;          f[1]=(float)ret48;          f[2]=(float)ret96;
   f[3]=(float)(atr14/MathMax(c,eps)); f[4]=(float)(atr48/MathMax(c,eps));
   f[5]=(float)rv24;           f[6]=(float)rv96;
   f[7]=(float)pos96;          f[8]=(float)pos288;
   f[9]=(float)ema_fast_dist;  f[10]=(float)ema_slow_dist; f[11]=(float)ema_gap;
   f[12]=(float)bars_since_hi96; f[13]=(float)bars_since_lo96;
   f[14]=(float)trend_age;     f[15]=(float)up_streak;
   f[16]=(float)hod_sin;       f[17]=(float)hod_cos;
   return true;
}

//+------------------------------------------------------------------+
//| Primary trend direction: +1 long / -1 short (EMA fast vs slow).   |
//+------------------------------------------------------------------+
int MG_Primary()
{
   double cls[];
   MqlRates r[];
   int got = CopyRates(_Symbol, PERIOD_M5, 1, MG_HISTORY, r);
   if(got < 250) return 0;
   int n = ArraySize(r);
   ArrayResize(cls, n);
   for(int i = 0; i < n; i++) cls[i] = r[i].close;
   double ef = _MG_EmaLast(cls, n, MG_EMA_FAST);
   double es = _MG_EmaLast(cls, n, MG_EMA_SLOW);
   return (ef > es) ? 1 : -1;
}

//+------------------------------------------------------------------+
//| Meta-gate P(act). Returns -1.0 on failure.                        |
//+------------------------------------------------------------------+
double MG_ActProb()
{
   if(!g_mg_ready || g_mg_handle == INVALID_HANDLE) return -1.0;
   float f[];
   if(!MG_BuildFeatures(f)) return -1.0;
   long  lbl[];  ArrayResize(lbl, 1);
   float prob[]; ArrayResize(prob, 2);
   if(!OnnxRun(g_mg_handle, ONNX_DEFAULT, f, lbl, prob)) return -1.0;
   return (double)prob[1];        // P(class 1 = act)
}

#endif // METAGATE_MQH
