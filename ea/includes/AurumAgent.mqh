//+------------------------------------------------------------------+
//| AurumAgent.mqh — MT5-side runner for the AURUM v2 AI stack.        |
//|                                                                    |
//| Loads the deployable AURUM bundle from MT5 Common Files:           |
//|   M4GOLD_AURUM_GOLD.onnx        main net  float[1,2048]->float[1,13]|
//|   M4GOLD_AURUM_META_GOLD.onnx   meta gate float[1,13] ->float[1,2] |
//|   M4GOLD_AURUM_GOLD_spec.json   conformal q_hat, thresholds, norm   |
//|                                                                    |
//| Pipeline per decision (see docs/DESIGN_AURUM.md):                  |
//|   1. build the 2048-dim multi-timeframe feature vector             |
//|   2. main net  -> [dir(3), quant(3), exec(3), regime(4)]           |
//|   3. conformal singleton test on dir probabilities                 |
//|   4. meta gate -> P(act); require >= act_threshold                 |
//|   5. quantile-Kelly + vol-target lot multiplier                    |
//|                                                                    |
//| Channel maths MUST mirror python/aurum/datamodule.py exactly.      |
//+------------------------------------------------------------------+
#ifndef AURUM_AGENT_MQH
#define AURUM_AGENT_MQH

// ---- contract constants (mirror python/aurum/aurum_config.py) -------
#define AURUM_N_CH        8
#define AURUM_L_M5        128
#define AURUM_L_M15       64
#define AURUM_L_H1        64
#define AURUM_INPUT_DIM   2048   // 128*8 + 64*8 + 64*8
#define AURUM_OUTPUT_DIM  13     // dir3 + quant3 + exec3 + regime4
#define AURUM_META_DIM    13

// ---- decision result ------------------------------------------------
struct AurumDecision
{
   int    direction;     // +1 long / -1 short / 0 flat-or-abstain
   bool   confident;     // conformal singleton AND meta gate passed
   double p_long;
   double p_short;
   double lot_mult;      // multiply onto base lot
   double sl_atr;
   double tp_atr;
   int    regime;        // argmax regime class
   double q50;           // model's median forward-return prediction
   string reason;
};

// ---- agent state ----------------------------------------------------
long   g_aurum_main   = INVALID_HANDLE;
long   g_aurum_meta   = INVALID_HANDLE;
bool   g_aurum_ready  = false;
double g_aurum_qhat   = 0.5;
double g_aurum_actthr = 0.55;
double g_aurum_kellyf = 0.25;
double g_aurum_voltgt = 0.10;
double g_aurum_maxlot = 2.0;
double g_aurum_minlot = 0.25;
bool   g_aurum_deploy = false;   // spec's deploy flag
string g_aurum_version = "";     // spec version string
bool   g_aurum_spec_ok = false;  // spec file was found AND parsed sanely

//+------------------------------------------------------------------+
//| Minimal JSON scalar extractor (spec files are flat enough).       |
//+------------------------------------------------------------------+
double _AurumJsonNum(const string js, const string key, double dflt)
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

// Boolean extractor — looks for `"key": true/false`.
bool _AurumJsonBool(const string js, const string key)
{
   int p = StringFind(js, "\"" + key + "\"");
   if(p < 0) return false;
   int c = StringFind(js, ":", p);
   if(c < 0) return false;
   return (StringFind(StringSubstr(js, c, 12), "true") >= 0);
}

// String extractor — looks for `"key": "value"`.
string _AurumJsonStr(const string js, const string key)
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
//| Load the AURUM bundle. Returns false if the main net is missing.  |
//+------------------------------------------------------------------+
bool AURUM_Init()
{
   g_aurum_main = OnnxCreate("M4GOLD_AURUM_GOLD.onnx", ONNX_COMMON_FOLDER);
   if(g_aurum_main == INVALID_HANDLE)
   {
      Print("[AURUM] main net M4GOLD_AURUM_GOLD.onnx not found — AURUM OFF");
      return false;
   }
   ulong in_shape[]  = {1, AURUM_INPUT_DIM};
   ulong out_shape[] = {1, AURUM_OUTPUT_DIM};
   OnnxSetInputShape (g_aurum_main, 0, in_shape);
   OnnxSetOutputShape(g_aurum_main, 0, out_shape);

   g_aurum_meta = OnnxCreate("M4GOLD_AURUM_META_GOLD.onnx", ONNX_COMMON_FOLDER);
   if(g_aurum_meta != INVALID_HANDLE)
   {
      // The XGBoost meta gate ONNX has ONE input and TWO outputs:
      //   output 0 = "label"          int64 [1]
      //   output 1 = "probabilities"  float [1,2]
      // Both shapes must be declared, and OnnxRun must be passed BOTH
      // output arrays, or it fails with "incorrect parameters count".
      ulong m_in[]   = {1, AURUM_META_DIM};
      ulong m_lbl[]  = {1};
      ulong m_prob[] = {1, 2};
      OnnxSetInputShape (g_aurum_meta, 0, m_in);
      OnnxSetOutputShape(g_aurum_meta, 0, m_lbl);
      OnnxSetOutputShape(g_aurum_meta, 1, m_prob);
   }
   else Print("[AURUM] meta gate missing — proceeding without meta filter");

   // spec JSON — FILE_ANSI is REQUIRED: the spec is single-byte ASCII
   // (Python json.dumps). Without FILE_ANSI, MT5 defaults FILE_TXT to
   // UTF-16 and reads the file as garbage, so every key lookup fails.
   int h = FileOpen("M4GOLD_AURUM_GOLD_spec.json",
                    FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h != INVALID_HANDLE)
   {
      string js = "";
      while(!FileIsEnding(h)) js += FileReadString(h);
      FileClose(h);
      g_aurum_qhat    = _AurumJsonNum(js, "q_hat",         0.5);
      g_aurum_actthr  = _AurumJsonNum(js, "act_threshold", 0.55);
      g_aurum_kellyf  = _AurumJsonNum(js, "kelly_fraction",0.25);
      g_aurum_voltgt  = _AurumJsonNum(js, "vol_target",    0.10);
      g_aurum_maxlot  = _AurumJsonNum(js, "max_lot_mult",  2.0);
      g_aurum_minlot  = _AurumJsonNum(js, "min_lot_mult",  0.25);
      g_aurum_deploy  = _AurumJsonBool(js, "deploy");
      g_aurum_version = _AurumJsonStr(js, "version");
      // A genuine AURUM spec always carries a "q_hat" key. If it came back
      // as the bare default, the wrong file was staged.
      g_aurum_spec_ok = (StringFind(js, "\"q_hat\"") >= 0
                         && StringFind(js, "\"strategy\"") >= 0);
      if(!g_aurum_spec_ok)
         Print("[AURUM] *** WRONG SPEC FILE *** — M4GOLD_AURUM_GOLD_spec.json "
               "in Common Files has no q_hat/strategy key. You staged the "
               "wrong file (e.g. AURUM_report.json or an old spec). Re-copy "
               "M4GOLD_AURUM_GOLD_spec.json from the deploy=True training run.");
   }
   else
      Print("[AURUM] *** SPEC MISSING *** — M4GOLD_AURUM_GOLD_spec.json not "
            "in MT5 Common Files. Copy the bundle there, then reload the EA.");

   g_aurum_ready = true;
   PrintFormat("[AURUM] spec: version=%s  deploy=%s  q_hat=%.4f  act_thr=%.2f  "
               "kelly_f=%.2f", g_aurum_version == "" ? "?" : g_aurum_version,
               g_aurum_deploy ? "true" : "false",
               g_aurum_qhat, g_aurum_actthr, g_aurum_kellyf);
   return true;
}

void AURUM_Release()
{
   if(g_aurum_main != INVALID_HANDLE) OnnxRelease(g_aurum_main);
   if(g_aurum_meta != INVALID_HANDLE) OnnxRelease(g_aurum_meta);
   g_aurum_ready = false;
}

//+------------------------------------------------------------------+
//| Compute the 8 microstructure channels for `lookback` bars of `tf` |
//| and write them, row-major [bar,ch] oldest->newest, into `dst`     |
//| starting at `off`. Mirrors datamodule._compute_channels.          |
//+------------------------------------------------------------------+
void _AurumFillChannels(ENUM_TIMEFRAMES tf, int lookback,
                        float &dst[], int off)
{
   // Pull extra history for the rolling stats (z-score 200, vol-ma 50, atr 14).
   // Start at shift 1 — bar 0 is the still-FORMING candle. The model was
   // trained on CLOSED bars only; feeding it a partial bar de-syncs the
   // model from real time-price movement and mistimes entries.
   int warm = 220;
   int need = lookback + warm;
   MqlRates r[];
   int got = CopyRates(_Symbol, tf, 1, need, r);
   if(got < lookback + 30)
   {
      // not enough history — zero-fill (model still runs, low confidence)
      for(int i = 0; i < lookback * AURUM_N_CH; i++) dst[off + i] = 0.0;
      return;
   }
   int n = ArraySize(r);            // r[0] oldest .. r[n-1] newest
   int start = n - lookback;        // first bar we actually emit
   for(int b = 0; b < lookback; b++)
   {
      int j = start + b;            // index into r
      double o = r[j].open, hi = r[j].high, lo = r[j].low, c = r[j].close;
      double v = (double)r[j].tick_volume;
      double pc = (j > 0) ? r[j-1].close : c;
      double eps = 1e-12;

      double ret  = (pc > eps) ? MathLog(MathMax(c,eps)/MathMax(pc,eps)) : 0.0;
      double hl   = (c > eps) ? (hi - lo) / c : 0.0;
      double body = (c > eps) ? (c - o) / c : 0.0;
      double up   = (c > eps) ? (hi - MathMax(o,c)) / c : 0.0;
      double dn   = (c > eps) ? (MathMin(o,c) - lo) / c : 0.0;

      // signed volume z-score over the trailing 200 bars
      double sgn = (ret > 0) ? 1.0 : ((ret < 0) ? -1.0 : 0.0);
      double sv  = sgn * v;
      double m = 0, s = 0; int cnt = 0;
      for(int k = MathMax(0, j-199); k <= j; k++)
      {
         double pck = (k>0)? r[k-1].close : r[k].close;
         double rk  = (pck>eps)? MathLog(MathMax(r[k].close,eps)/MathMax(pck,eps)):0.0;
         double sgk = (rk>0)?1.0:((rk<0)?-1.0:0.0);
         m += sgk * (double)r[k].tick_volume; cnt++;
      }
      m /= MathMax(1, cnt);
      cnt = 0;
      for(int k = MathMax(0, j-199); k <= j; k++)
      {
         double pck = (k>0)? r[k-1].close : r[k].close;
         double rk  = (pck>eps)? MathLog(MathMax(r[k].close,eps)/MathMax(pck,eps)):0.0;
         double sgk = (rk>0)?1.0:((rk<0)?-1.0:0.0);
         double d = sgk*(double)r[k].tick_volume - m; s += d*d; cnt++;
      }
      s = MathSqrt(s / MathMax(1, cnt));
      double sv_z = (s > eps) ? (sv - m) / s : 0.0;

      // volume ratio over trailing 50 bars
      double vma = 0; int vc = 0;
      for(int k = MathMax(0, j-49); k <= j; k++)
      { vma += (double)r[k].tick_volume; vc++; }
      vma /= MathMax(1, vc);
      double vr = (vma > eps) ? v / vma : 1.0;

      // ATR(14) / close
      double atr = 0; int ac = 0;
      for(int k = MathMax(1, j-13); k <= j; k++)
      {
         double tr = MathMax(r[k].high - r[k].low,
                     MathMax(MathAbs(r[k].high - r[k-1].close),
                             MathAbs(r[k].low  - r[k-1].close)));
         atr += tr; ac++;
      }
      atr /= MathMax(1, ac);
      double atrn = (c > eps) ? atr / c : 0.0;

      int base = off + b * AURUM_N_CH;
      dst[base+0]=(float)ret;  dst[base+1]=(float)hl;   dst[base+2]=(float)body;
      dst[base+3]=(float)up;   dst[base+4]=(float)dn;   dst[base+5]=(float)sv_z;
      dst[base+6]=(float)vr;   dst[base+7]=(float)atrn;
   }
}

//+------------------------------------------------------------------+
//| Build the full 2048-dim AURUM input vector.                       |
//+------------------------------------------------------------------+
void _AurumBuildInput(float &x[])
{
   ArrayResize(x, AURUM_INPUT_DIM);
   _AurumFillChannels(PERIOD_M5,  AURUM_L_M5,  x, 0);
   _AurumFillChannels(PERIOD_M15, AURUM_L_M15, x, AURUM_L_M5 * AURUM_N_CH);
   _AurumFillChannels(PERIOD_H1,  AURUM_L_H1,  x,
                      (AURUM_L_M5 + AURUM_L_M15) * AURUM_N_CH);
}

//+------------------------------------------------------------------+
//| Quantile-Kelly + vol-target lot multiplier (mirrors sizing.py).   |
//+------------------------------------------------------------------+
double _AurumLotMult(double q10, double q50, double q90, double rvol)
{
   double win  = MathMax(1e-6, q90);
   double loss = MathMax(1e-6, -q10);
   double R    = win / loss;
   double span = MathMax(1e-9, q90 - q10);
   double p    = (q50 - q10) / span;
   if(p < 0.05) p = 0.05; if(p > 0.95) p = 0.95;
   double f = p - (1.0 - p) / R;
   if(f < 0) f = 0;
   double kelly = f * g_aurum_kellyf;
   double vscale = g_aurum_voltgt / MathMax(1e-6, rvol);
   double m = kelly * vscale;
   if(m < g_aurum_minlot) m = g_aurum_minlot;
   if(m > g_aurum_maxlot) m = g_aurum_maxlot;
   return m;
}

//+------------------------------------------------------------------+
//| Run the full AURUM decision pipeline.                             |
//+------------------------------------------------------------------+
AurumDecision AURUM_Decide()
{
   AurumDecision d;
   d.direction = 0; d.confident = false; d.lot_mult = g_aurum_minlot;
   d.p_long = 0; d.p_short = 0; d.sl_atr = 1.0; d.tp_atr = 2.0;
   d.regime = 0; d.reason = "";

   if(!g_aurum_ready) { d.reason = "agent not ready"; return d; }

   float x[];
   _AurumBuildInput(x);

   float out[];
   ArrayResize(out, AURUM_OUTPUT_DIM);
   if(!OnnxRun(g_aurum_main, ONNX_DEFAULT, x, out))
   {
      d.reason = "main OnnxRun failed"; return d;
   }
   // output layout: [dir3, quant3, exec3, regime4] — dir & regime softmaxed.
   double p_short = out[0], p_flat = out[1], p_long = out[2];
   double q10 = out[3], q50 = out[4], q90 = out[5];
   d.sl_atr = MathMax(0.3, MathMin(5.0, (double)out[6]));
   d.tp_atr = MathMax(0.5, MathMin(8.0, (double)out[7]));
   d.p_long = p_long; d.p_short = p_short;
   d.q50 = q50;            // median forward-return — used as a sanity gate

   int reg = 9, rbest = 9;
   double rmax = out[9];
   for(int k = 10; k < 13; k++) if(out[k] > rmax) { rmax = out[k]; rbest = k; }
   d.regime = rbest - 9;

   // raw direction = argmax
   int raw = 1;                                  // flat
   if(p_long >= p_short && p_long >= p_flat) raw = 2;
   else if(p_short >= p_long && p_short >= p_flat) raw = 0;

   // L5 conformal singleton test: class in set iff (1 - p) <= q_hat.
   int set_size = 0;
   if(1.0 - p_short <= g_aurum_qhat) set_size++;
   if(1.0 - p_flat  <= g_aurum_qhat) set_size++;
   if(1.0 - p_long  <= g_aurum_qhat) set_size++;
   bool singleton = (set_size == 1);

   // L4 meta gate
   bool meta_ok = true;
   if(g_aurum_meta != INVALID_HANDLE)
   {
      float mf[]; ArrayResize(mf, AURUM_META_DIM);
      // dir_conf = top probability minus 2nd — must match Python
      // meta_label.build_meta_features (sorted top1-top2 of the 3 dir probs).
      double mx = MathMax(p_short, MathMax(p_flat, p_long));
      double mn = MathMin(p_short, MathMin(p_flat, p_long));
      double mid = (p_short + p_flat + p_long) - mx - mn;
      double conf = mx - mid;
      // realized vol = std of the 128 M5 ret channel values just built
      double rm = 0; for(int i=0;i<AURUM_L_M5;i++) rm += x[i*AURUM_N_CH];
      rm /= AURUM_L_M5;
      double rv = 0; for(int i=0;i<AURUM_L_M5;i++)
      { double dd=x[i*AURUM_N_CH]-rm; rv+=dd*dd; }
      rv = MathSqrt(rv / AURUM_L_M5);
      double atrn = x[(AURUM_L_M5-1)*AURUM_N_CH + 7];
      mf[0]=(float)p_short; mf[1]=(float)p_flat; mf[2]=(float)p_long;
      mf[3]=(float)conf; mf[4]=(float)q10; mf[5]=(float)q50; mf[6]=(float)q90;
      mf[7]=out[9]; mf[8]=out[10]; mf[9]=out[11]; mf[10]=out[12];
      mf[11]=(float)rv; mf[12]=(float)atrn;
      // meta ONNX has 2 outputs: label int64[1] + probabilities float[1,2].
      long  mlbl[];  ArrayResize(mlbl,  1);
      float mprob[]; ArrayResize(mprob, 2);
      if(OnnxRun(g_aurum_meta, ONNX_DEFAULT, mf, mlbl, mprob))
         meta_ok = ((double)mprob[1] >= g_aurum_actthr);   // P(act)
   }

   d.confident = (singleton && meta_ok && raw != 1);
   if(d.confident)
   {
      d.direction = (raw == 2) ? +1 : -1;
      // realized vol again for sizing (same definition)
      double rm = 0; for(int i=0;i<AURUM_L_M5;i++) rm += x[i*AURUM_N_CH];
      rm /= AURUM_L_M5;
      double rv = 0; for(int i=0;i<AURUM_L_M5;i++)
      { double dd=x[i*AURUM_N_CH]-rm; rv+=dd*dd; }
      rv = MathSqrt(rv / AURUM_L_M5);
      d.lot_mult = _AurumLotMult(q10, q50, q90, rv);
      d.reason = "confident";
   }
   else
   {
      d.reason = singleton ? (meta_ok ? "flat" : "meta veto")
                           : "conformal set not singleton";
   }
   return d;
}

#endif // AURUM_AGENT_MQH
