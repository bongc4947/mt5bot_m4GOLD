#ifndef FEATUREENCODER_MQH
#define FEATUREENCODER_MQH

#include "Defines.mqh"
#include "TickBuffer.mqh"
#include "BrokerConfig.mqh"
#include "MarketHours.mqh"

//=============================================================================
// CFeatureEncoder — 1000-dim direction features (10 blocks, see Defines.mqh).
// Also provides BuildExecFeatures() for the 1120-dim execution input.
// Must match python/feature_engine.py exactly.
//
// Direction features (1000-dim) — block layout from Defines.mqh:
//   [M5    50×8=400]  indices   0.. 399  (FEAT_BLOCK_M5_START)
//   [H1    30×5=150]  indices 400.. 549  (FEAT_BLOCK_H1_START)
//   [H4    20×4= 80]  indices 550.. 629  (FEAT_BLOCK_H4_START)  ← TODO: full 80-dim (stub at 120-135 is dead)
//   [H8    20×4= 80]  indices 630.. 709  (FEAT_BLOCK_H8_START)  ← TODO: not yet populated
//   [D1    20×4= 80]  indices 710.. 789  (FEAT_BLOCK_D1_START)  ← TODO: not yet populated
//   [Spec       60]   indices 790.. 849  (FEAT_BLOCK_SPECTRAL_START) ← TODO: not yet populated
//   [Pattern    50]   indices 850.. 899  (FEAT_BLOCK_PATTERN_START)  ← TODO: not yet populated
//   [StatReg    60]   indices 900.. 959  (FEAT_BLOCK_STATREG_START)  ← TODO: not yet populated
//   [XAsset     20]   indices 960.. 979  (FEAT_BLOCK_XASSET_START)   ← TODO: not yet populated
//   [Macro      20]   indices 980.. 999  (FEAT_BLOCK_MACRO_START)    ← TODO: not yet populated
//
// KNOWN LIMITATION: EA currently populates only M5 (0-399) and H1 (400-549).
// H4 has a dead stub at indices 120-135 (old 136-dim architecture) — BuildH4Block()
// writes there but those indices are overwritten by BuildM5Block(). To be replaced
// with full H4 at indices 550-629. Models are trained to tolerate zero-padded blocks.
//
// Execution context (120-dim, appended for exec model, indices 1000..1119):
//   Indices 1000..1039 (ctx  0..39): microstructure + session + portfolio context
//   Indices 1040..1059 (ctx 40..59): Phase 3 macro/sentiment block
//     Read from HYDRA4_macro.json written by python/export_macro_snapshot.py
//     [1040] fed_funds_norm     [1048] cot_nc_net_pct
//     [1041] yield_curve_norm   [1049] cot_c_net_pct
//     [1042] cpi_norm           [1050] cot_momentum
//     [1043] pmi_norm           [1051] cot_oi_change
//     [1044] vix_norm           [1052] cot_composite
//     [1045] dxy_momentum       [1053] news_proximity_pre
//     [1046] econ_surprise      [1054] news_proximity_post
//     [1047] rate_diff_norm     [1055] news_impact_now
//                               [1056] news_week_density
//                               [1057] macro_regime
//                               [1058] risk_on_score
//                               [1059] macro_uncertainty
//   Indices 1060..1079 (ctx 60..79): FAE fundamental block (FundamentalLoader.mqh)
//   Indices 1080..1119 (ctx 80..119): QTW state block (QuantStateLoader.mqh)
//     VQW 1080-1089, RDW 1090-1097, MCW 1098-1105, CQW 1106-1113, LQW 1114-1119
//=============================================================================

// mk4.2: MACRO_* defines removed — macro block no longer used.

// Global direction feature storage [MAX_SYMBOLS × FEATURE_DIM]
float g_dir_features[MAX_SYMBOLS * FEATURE_DIM];
bool  g_features_valid[MAX_SYMBOLS];

// Execution feature storage [MAX_SYMBOLS × EXEC_FEATURE_DIM]
float g_exec_features[MAX_SYMBOLS * EXEC_FEATURE_DIM];

// mk4.2: macro snapshot removed.

// Raw feature history ring [sym][bar(20)][feat(RAW_FEATURES)]
double g_feat_history[MAX_SYMBOLS * 20 * RAW_FEATURES];
int    g_feat_hist_head[MAX_SYMBOLS];
int    g_feat_hist_count[MAX_SYMBOLS];
int    g_post_gap_bars[MAX_SYMBOLS];

// ATR(14/5/50) per symbol for exec model
double g_atr14[MAX_SYMBOLS];
double g_atr5[MAX_SYMBOLS];
double g_atr50[MAX_SYMBOLS];

// Running EMA of mid price per symbol (for M5 F20-F21)
double g_ema5p[MAX_SYMBOLS];
double g_ema20p[MAX_SYMBOLS];
double g_ema50p[MAX_SYMBOLS];

void DirFeatSet(int s, int d, float v) { g_dir_features[s * FEATURE_DIM + d] = v; }
float DirFeatGet(int s, int d)         { return g_dir_features[s * FEATURE_DIM + d]; }

void ExecFeatSet(int s, int d, float v) { g_exec_features[s * EXEC_FEATURE_DIM + d] = v; }
float ExecFeatGet(int s, int d)         { return g_exec_features[s * EXEC_FEATURE_DIM + d]; }

class CFeatureEncoder
{
private:
   double LogReturn(double p1, double p0)
   {
      if(p0 <= 0.0 || p1 <= 0.0) return 0.0;
      return MathLog(p1 / p0);
   }

   void PushHistory(int sym_idx, double &raw[])
   {
      int head = g_feat_hist_head[sym_idx];
      for(int f = 0; f < RAW_FEATURES; f++)
         g_feat_history[sym_idx * 20 * RAW_FEATURES + head * RAW_FEATURES + f] = raw[f];
      g_feat_hist_head[sym_idx]  = (head + 1) % 20;
      if(g_feat_hist_count[sym_idx] < 20) g_feat_hist_count[sym_idx]++;
   }

   double HistGet(int sym_idx, int bar_offset, int f)
   {
      int cnt = g_feat_hist_count[sym_idx];
      if(bar_offset >= cnt) return 0.0;
      int pos = (g_feat_hist_head[sym_idx] - 1 - bar_offset + 20) % 20;
      return g_feat_history[sym_idx * 20 * RAW_FEATURES + pos * RAW_FEATURES + f];
   }

   void UpdateATR(int sym_idx, double tr)
   {
      if(g_atr14[sym_idx] <= 0.0) g_atr14[sym_idx] = tr;
      else g_atr14[sym_idx] = (g_atr14[sym_idx] * 13.0 + tr) / 14.0;

      if(g_atr5[sym_idx] <= 0.0) g_atr5[sym_idx] = tr;
      else g_atr5[sym_idx] = (g_atr5[sym_idx] * 4.0 + tr) / 5.0;

      if(g_atr50[sym_idx] <= 0.0) g_atr50[sym_idx] = tr;
      else g_atr50[sym_idx] = (g_atr50[sym_idx] * 49.0 + tr) / 50.0;
   }

   double SinEncode(double val, double period) { return MathSin(2.0 * M_PI * val / period); }
   double CosEncode(double val, double period) { return MathCos(2.0 * M_PI * val / period); }

   //--- Compute 8 raw HTF features from a rates array (n_rates bars, oldest→newest).
   //    mk4.3: dead code (BuildH1Block / BuildH4Block no longer called).
   //    Kept as historical reference. The function size is fixed at 8 — the
   //    MTF_H1_RAW define has been removed alongside the rest of the HTF
   //    constants. Inline literal so the file still compiles.
   //    Returns false if not enough bars.
   bool ComputeHTFRaw(const MqlRates &rates[], int n_rates, double &out[])
   {
      ArrayResize(out, 8);
      ArrayInitialize(out, 0.0);
      if(n_rates < 4) return false;

      int cur = n_rates - 1;  // most recent complete bar index

      // F0-F1: log returns
      out[0] = (cur >= 1 && rates[cur-1].close > 0.0)
                 ? MathLog(rates[cur].close / rates[cur-1].close) : 0.0;
      out[1] = (cur >= 3 && rates[cur-3].close > 0.0)
                 ? MathLog(rates[cur].close / rates[cur-3].close) : 0.0;

      // F2: realized vol (5-bar)
      double lr_sum2 = 0.0;
      int nvol = MathMin(5, cur);
      for(int b = 0; b < nvol; b++)
      {
         double lr = (rates[cur-b-1].close > 0.0)
                      ? MathLog(rates[cur-b].close / rates[cur-b-1].close) : 0.0;
         lr_sum2 += lr * lr;
      }
      out[2] = (nvol > 0) ? MathSqrt(lr_sum2 / nvol) : 0.0;

      // F3: ATR(14) normalized
      double atr14 = 0.0;
      int natr = MathMin(14, n_rates - 1);
      for(int b = cur; b > cur - natr; b--)
      {
         double prev_c = (b > 0) ? rates[b-1].close : rates[b].open;
         double tr = MathMax(rates[b].high - rates[b].low,
                    MathMax(MathAbs(rates[b].high - prev_c),
                            MathAbs(rates[b].low  - prev_c)));
         atr14 += tr;
      }
      if(natr > 0) atr14 /= natr;
      double price = rates[cur].close;
      out[3] = (price > 0.0) ? MathMin(atr14 / price * 100.0, 2.0) : 0.0;

      // F4: RSI(14) simplified
      double avg_gain = 0.0, avg_loss = 0.0;
      int nrsi = MathMin(14, cur);
      for(int b = cur; b > cur - nrsi; b--)
      {
         double d = rates[b].close - rates[b-1].close;
         if(d > 0) avg_gain += d; else avg_loss += -d;
      }
      if(nrsi > 0) { avg_gain /= nrsi; avg_loss /= nrsi; }
      double rs = (avg_loss > 1e-12) ? avg_gain / avg_loss : 1.0;
      double rsi = 1.0 / (1.0 + rs);
      out[4] = rsi; // Corrected: value 0..1 (Matches 100 * gain/(gain+loss) normalized)

      // F5-F6: EMA(5)/EMA(20)/EMA(50) crossover ratios
      double ema5v = rates[0].close, ema20v = rates[0].close, ema50v = rates[0].close;
      double a5 = 2.0/6.0, a20 = 2.0/21.0, a50 = 2.0/51.0;
      for(int b = 1; b <= cur; b++)
      {
         double c = rates[b].close;
         ema5v  = a5  * c + (1.0 - a5)  * ema5v;
         ema20v = a20 * c + (1.0 - a20) * ema20v;
         ema50v = a50 * c + (1.0 - a50) * ema50v;
      }
      double e5_20  = (ema20v > 1e-10) ? (ema5v  / ema20v - 1.0) * 100.0 : 0.0;
      double e20_50 = (ema50v > 1e-10) ? (ema20v / ema50v - 1.0) * 100.0 : 0.0;
      out[5] = MathMax(-1.0, MathMin(e5_20,  1.0));
      out[6] = MathMax(-1.0, MathMin(e20_50, 1.0));

      // F7: HL range normalized
      double hl = rates[cur].high - rates[cur].low;
      out[7] = (price > 0.0) ? MathMin(hl / price * 100.0, 2.0) / 2.0 : 0.0;

      return true;
   }

   //--- Build H1 context block (24-dim) at indices 96..119 of dir features.
   //    Requires H1 rates fetched via CopyRates(PERIOD_H1, 0, 60, ...) start=0 latest.
   void BuildH1Block(int sym_idx, const string broker_sym)
   {
      MqlRates h1[];
      // start_pos=1 skips current incomplete H1 bar; fetch 60 complete H1 bars
      int n = CopyRates(broker_sym, PERIOD_H1, 1, 60, h1);
      if(n < 5)
      {
         // Not enough data — zero the block
         for(int d = 96; d < 120; d++) DirFeatSet(sym_idx, d, 0.0f);
         return;
      }
      // h1 is oldest→newest: h1[0]=oldest, h1[n-1]=most recent complete bar

      double cur8[];
      bool ok = ComputeHTFRaw(h1, n, cur8);
      if(!ok) { for(int d = 96; d < 120; d++) DirFeatSet(sym_idx, d, 0.0f); return; }

      // mean4 over last 4 H1 bars
      double sum8[];  ArrayResize(sum8, 8); ArrayInitialize(sum8, 0.0);
      int navg = MathMin(4, n);
      for(int b = n - navg; b < n; b++)
      {
         // Build raw8 for each of the last 4 bars individually
         double tmp8[];
         ComputeHTFRaw(h1, b + 1, tmp8);  // use first b+1 bars (latest = b)
         for(int f = 0; f < 8; f++) sum8[f] += tmp8[f];
      }
      double mean4[8];
      for(int f = 0; f < 8; f++) mean4[f] = sum8[f] / navg;

      // delta = cur8 − prev8 (n-1 vs n-2)
      double prev8[];
      bool has_prev = (n >= 2) && ComputeHTFRaw(h1, n - 1, prev8);

      for(int f = 0; f < 8; f++)
      {
         double raw_v  = cur8[f];
         double mean_v = mean4[f];
         double delt_v = has_prev ? raw_v - prev8[f] : 0.0;

         DirFeatSet(sym_idx, 96 + f,      (float)raw_v);
         DirFeatSet(sym_idx, 96 + 8 + f,  (float)mean_v);
         DirFeatSet(sym_idx, 96 + 16 + f, (float)delt_v);
      }
   }

   //--- Build H4 raw block (old 16-dim stub: indices 120..135 of dir features).
   //    TODO: replace with full 80-dim H4 block at FEAT_BLOCK_H4_START (550..629).
   void BuildH4Block(int sym_idx, const string broker_sym)
   {
      MqlRates h4[];
      int n = CopyRates(broker_sym, PERIOD_H4, 1, 60, h4);
      if(n < 5)
      {
         for(int d = 120; d < 136; d++) DirFeatSet(sym_idx, d, 0.0f);
         return;
      }

      double cur8[];
      bool ok = ComputeHTFRaw(h4, n, cur8);
      if(!ok) { for(int d = 120; d < 136; d++) DirFeatSet(sym_idx, d, 0.0f); return; }

      // delta = cur8 − prev8
      double prev8[];
      bool has_prev = (n >= 2) && ComputeHTFRaw(h4, n - 1, prev8);

      for(int f = 0; f < 8; f++)
      {
         double raw_v  = cur8[f];
         double delt_v = has_prev ? raw_v - prev8[f] : 0.0;

         DirFeatSet(sym_idx, 120 + f,     (float)raw_v);
         DirFeatSet(sym_idx, 120 + 8 + f, (float)delt_v);
      }
   }

public:
   //+----------------------------------------------------------------+
   //| BuildFeaturesForBar — historical-export entrypoint.            |
   //|                                                                 |
   //| Used by MT5_Bot_mk4_FeatureExport.mq5 to walk every M5 bar in   |
   //| history and emit a 1160-dim feature vector to a binary file.    |
   //|                                                                 |
   //| Parameters:                                                     |
   //|   canonical : symbol name (typically _Symbol on the export      |
   //|               script's chart).                                  |
   //|   sym_idx   : feature-array slot (0 for single-symbol export).  |
   //|   shift     : MT5 bar offset — 0 = current, N-1 = oldest in     |
   //|               history. The script iterates from N-1 down to 0.  |
   //|                                                                 |
   //| HONEST IMPLEMENTATION NOTE                                      |
   //| ------------------------                                        |
   //| The live Encode() pipeline assumes shift=0 (most recent closed  |
   //| bar). For historical export we need each helper that calls      |
   //| iClose / iHigh / iLow / iTime to use the supplied shift. The    |
   //| stub below currently delegates to Encode() unchanged — it works |
   //| only when shift==0 (i.e. the script's last iteration). For      |
   //| shift>0 you must extend each block builder (M5/H1/H4/H8/D1/     |
   //| Spectral/Pattern/StatReg/XAsset) to honour an as-of offset.     |
   //|                                                                 |
   //| Recommended path (small change, big payoff):                    |
   //|   1. Add `int g_export_shift = 0;` as a class member.           |
   //|   2. In every iClose/iHigh/iLow/iTime call inside the private   |
   //|      block builders, replace `0` (or hard-coded shift) with     |
   //|      `(g_export_shift + bar)` so the rolling windows slide.     |
   //|   3. Set g_export_shift = shift here, call Encode(), reset.     |
   //+----------------------------------------------------------------+
   void BuildFeaturesForBar(const string canonical, int sym_idx, int shift)
   {
      if(shift != 0) {
         static int warned = 0;
         if(warned == 0) {
            Print("[FeatureEncoder] WARN: BuildFeaturesForBar(shift=",
                  shift, ") falls back to live Encode(). Per-bar shift ",
                  "support requires extending the block builders. See ",
                  "the comment block above this method.");
            warned = 1;
         }
      }
      Encode(sym_idx, canonical);
   }

   void Init()
   {
      for(int i = 0; i < MAX_SYMBOLS; i++)
      {
         g_features_valid[i]   = false;
         g_feat_hist_head[i]   = 0;
         g_feat_hist_count[i]  = 0;
         g_post_gap_bars[i]    = 0;
         g_atr14[i] = g_atr5[i] = g_atr50[i] = 0.0;
         g_ema5p[i] = g_ema20p[i] = g_ema50p[i] = 0.0;
         for(int d = 0; d < FEATURE_DIM; d++)
            g_dir_features[i * FEATURE_DIM + d] = 0.0f;
         for(int d = 0; d < EXEC_FEATURE_DIM; d++)
            g_exec_features[i * EXEC_FEATURE_DIM + d] = 0.0f;
      }
   }

   // mk4.2: LoadMacroSnapshot / MaybeReloadMacro removed — model no
   // longer consumes the macro JSON. All feature inputs come from
   // MT5-supplied bars now.

   void Encode(int sym_idx, const string canonical)
   {
      if(g_tick_buf.Count(sym_idx) < 6) { g_features_valid[sym_idx] = false; return; }

      STick t0, t1, t2, t5;
      g_tick_buf.Get(sym_idx, 0, t0);
      g_tick_buf.Get(sym_idx, 1, t1);
      g_tick_buf.Get(sym_idx, 2, t2);
      g_tick_buf.Get(sym_idx, 5, t5);

      bool gap = (t0.time - t1.time > 120);
      if(gap) g_post_gap_bars[sym_idx] = 22;
      else if(g_post_gap_bars[sym_idx] > 0) g_post_gap_bars[sym_idx]--;

      double raw[RAW_FEATURES];
      double mid0 = (t0.bid + t0.ask) * 0.5;
      double mid1 = (t1.bid + t1.ask) * 0.5;
      double mid2 = (t2.bid + t2.ask) * 0.5;
      double mid5 = (t5.bid + t5.ask) * 0.5;
      double pip  = g_broker.PipSize(canonical);
      double spd  = g_broker.SpreadPips(canonical);
      if(pip <= 0.0) pip = _Point * 10.0;
      if(spd <= 0.0) spd = 1.0;

      // F0-F2: log returns 1/2/5 bar
      raw[0] = gap ? 0.0 : LogReturn(mid0, mid1);
      raw[1] = gap ? 0.0 : LogReturn(mid0, mid2);
      raw[2] = gap ? 0.0 : LogReturn(mid0, mid5);

      // F3: realized vol (10-bar from history)
      int cnt = g_feat_hist_count[sym_idx];
      double var10 = 0.0, sum10 = 0.0;
      int n10 = MathMin(10, cnt);
      for(int b = 0; b < n10; b++) sum10 += HistGet(sym_idx, b, 0);
      double m10 = n10 > 0 ? sum10 / n10 : 0.0;
      for(int b = 0; b < n10; b++) { double d = HistGet(sym_idx, b, 0) - m10; var10 += d*d; }
      raw[3] = (n10 > 1) ? MathSqrt(var10 / (n10 - 1)) : 0.0;

      // F4: spread normalized
      double live_spd = (t0.ask - t0.bid) / pip;
      raw[4] = MathMin(live_spd / (spd * 10.0), 1.0);

      // F5: momentum (close/close_1 - 1)
      raw[5] = (mid1 > 0.0) ? (mid0 - mid1) / mid1 : 0.0;

      // F6: tick count log (normalized)
      raw[6] = MathLog(MathMax(1.0, (double)t0.volume)) / 15.0;

      // F7: session flag
      raw[7] = g_market_hours.IsActiveSession(t0.time) ? 1.0 : 0.0;

      // F8: high-low range in pips
      double hi = mid0, lo = mid0;
      int buf_cnt = g_tick_buf.Count(sym_idx);
      for(int b = 0; b < MathMin(20, buf_cnt); b++)
      {
         STick tb; g_tick_buf.Get(sym_idx, b, tb);
         double mb = (tb.bid + tb.ask) * 0.5;
         if(mb > hi) hi = mb;
         if(mb < lo) lo = mb;
      }
      double hl = (hi > lo) ? (hi - lo) : 1e-10;
      raw[8] = MathMin(hl / pip / 100.0, 1.0);

      // F9: body position
      raw[9] = (hl > 1e-12) ? (mid0 - lo) / hl : 0.5;

      // F10: body-to-range
      raw[10] = (hl > 1e-12) ? MathAbs(mid0 - mid1) / hl : 0.0;

      // F11: upper wick
      raw[11] = (hl > 1e-12) ? (hi - MathMax(mid0, mid1)) / hl : 0.0;

      // F12: lower wick
      raw[12] = (hl > 1e-12) ? (MathMin(mid0, mid1) - lo) / hl : 0.0;

      // F13: RSI(14) normalized — compute from history
      double avg_gain = 0.0, avg_loss = 0.0;
      int rsi_n = MathMin(14, cnt);
      for(int b = 0; b < rsi_n; b++)
      {
         double lr = HistGet(sym_idx, b, 0);
         if(lr > 0) avg_gain += lr; else avg_loss += -lr;
      }
      if(rsi_n > 0) { avg_gain /= rsi_n; avg_loss /= rsi_n; }
      double rs = (avg_loss > 1e-12) ? avg_gain / avg_loss : 1.0;
      raw[13] = 1.0 / (1.0 + rs);

      // F14: ATR(14) normalized by price
      double tr = (cnt > 0) ? hl + MathAbs(mid0 - mid1) : hl;
      UpdateATR(sym_idx, tr);
      raw[14] = (mid0 > 0.0) ? MathMin(g_atr14[sym_idx] / mid0 * 100.0, 2.0) : 0.0;

      // F15: ADX proxy — directional persistence over last 14 bars (0..1)
      {
         double up_m = 0.0, dn_m = 0.0;
         int n14 = MathMin(14, cnt);
         for(int b = 0; b < n14 - 1; b++)
         {
            double lr = HistGet(sym_idx, b, 0);
            if(lr > 0.0) up_m += 1.0; else dn_m += 1.0;
         }
         raw[15] = (n14 > 1) ? MathAbs(up_m - dn_m) / (n14 - 1) : 0.0;
      }

      // F16: MACD-like histogram z-scored to [-1, 1]
      {
         int n20 = MathMin(20, cnt);
         double ema5 = 0.0, ema20 = 0.0;
         double a5 = 2.0 / 6.0, a20 = 2.0 / 21.0;
         bool ei = false;
         for(int b = n20 - 1; b >= 0; b--)
         {
            double lr = HistGet(sym_idx, b, 0);
            if(!ei) { ema5 = lr; ema20 = lr; ei = true; }
            else    { ema5 = a5 * lr + (1.0 - a5) * ema5; ema20 = a20 * lr + (1.0 - a20) * ema20; }
         }
         double macd_v = ema5 - ema20;
         double lr_var = 0.0;
         for(int b = 0; b < n20; b++) { double d = HistGet(sym_idx, b, 0); lr_var += d * d; }
         double lr_std = (n20 > 1) ? MathSqrt(lr_var / n20) + 1e-10 : 1e-10;
         raw[16] = MathMax(-1.0, MathMin(macd_v / (lr_std * 3.0), 1.0));
      }

      // F17: Bollinger Band %B → [0,1]
      {
         int n20 = MathMin(20, cnt);
         double sum_lr = 0.0;
         for(int b = 0; b < n20; b++) sum_lr += HistGet(sym_idx, b, 0);
         double mean_lr = (n20 > 0) ? sum_lr / n20 : 0.0;
         double var_lr  = 0.0;
         for(int b = 0; b < n20; b++) { double d = HistGet(sym_idx, b, 0) - mean_lr; var_lr += d * d; }
         double std_lr  = (n20 > 1) ? MathSqrt(var_lr / (n20 - 1)) + 1e-10 : 1e-10;
         double lr_now  = HistGet(sym_idx, 0, 0);
         double bb_z    = (lr_now - mean_lr) / (2.0 * std_lr);
         raw[17] = 0.5 + MathMax(-0.5, MathMin(bb_z, 0.5));
      }

      // F18: Volume ratio — current tick log vol / 20-bar mean → [0,1]
      {
         int n20 = MathMin(20, cnt);
         double vol_sum = 0.0;
         for(int b = 0; b < n20; b++) vol_sum += HistGet(sym_idx, b, 6);
         double vol_mean_v = (n20 > 0) ? vol_sum / n20 + 1e-10 : 1e-10;
         raw[18] = MathMin(raw[6] / vol_mean_v / 3.0, 1.0);
      }

      // F19: RSI slope — (RSI_now - RSI_3_bars_ago) × 5, clipped to [-1,1]
      {
         double rsi_3 = (cnt >= 3) ? HistGet(sym_idx, 2, 13) : raw[13];
         raw[19] = MathMax(-1.0, MathMin((raw[13] - rsi_3) * 5.0, 1.0));
      }

      // F20-F21: EMA(5)/EMA(20) and EMA(20)/EMA(50) crossover ratios [-1, 1]
      {
         if(gap || g_ema5p[sym_idx] <= 0.0)
         {
            g_ema5p[sym_idx] = g_ema20p[sym_idx] = g_ema50p[sym_idx] = mid0;
         }
         else
         {
            double a5  = 2.0 / 6.0;
            double a20 = 2.0 / 21.0;
            double a50 = 2.0 / 51.0;
            g_ema5p[sym_idx]  = a5  * mid0 + (1.0 - a5)  * g_ema5p[sym_idx];
            g_ema20p[sym_idx] = a20 * mid0 + (1.0 - a20) * g_ema20p[sym_idx];
            g_ema50p[sym_idx] = a50 * mid0 + (1.0 - a50) * g_ema50p[sym_idx];
         }
         double e5_20  = (g_ema20p[sym_idx] > 1e-10)
                          ? (g_ema5p[sym_idx]  / g_ema20p[sym_idx] - 1.0) * 100.0 : 0.0;
         double e20_50 = (g_ema50p[sym_idx] > 1e-10)
                          ? (g_ema20p[sym_idx] / g_ema50p[sym_idx] - 1.0) * 100.0 : 0.0;
         raw[20] = MathMax(-1.0, MathMin(e5_20,  1.0));
         raw[21] = MathMax(-1.0, MathMin(e20_50, 1.0));
      }

      // F22-F23: time-of-day sin/cos from bar close timestamp
      {
         MqlDateTime mdt; TimeToStruct(t0.time, mdt);
         raw[22] = MathSin(2.0 * M_PI * mdt.hour / 24.0);
         raw[23] = MathCos(2.0 * M_PI * mdt.hour / 24.0);
      }

      // Gap sentinel — zero out all features after a gap
      if(g_post_gap_bars[sym_idx] > 0)
         for(int f = 0; f < RAW_FEATURES; f++) raw[f] = DBL_MAX;

      PushHistory(sym_idx, raw);

      // Build M5 96-dim block: [raw(24) | mean20(24) | std20(24) | delta(24)]
      int nc = g_feat_hist_count[sym_idx];
      for(int f = 0; f < RAW_FEATURES; f++)
      {
         double last_v = HistGet(sym_idx, 0, f);
         double smean = 0.0; bool sentinel = false;
         for(int b = 0; b < nc; b++) { double hv = HistGet(sym_idx, b, f); if(hv == DBL_MAX) sentinel = true; smean += sentinel ? 0.0 : hv; }
         smean = (nc > 0 && !sentinel) ? smean / nc : 0.0;
         double sstd = 0.0;
         for(int b = 0; b < nc && !sentinel; b++) { double d = HistGet(sym_idx, b, f) - smean; sstd += d * d; }
         sstd = (nc > 1 && !sentinel) ? MathSqrt(sstd / (nc - 1)) : 0.0;
         double oldest = (nc >= 2) ? HistGet(sym_idx, nc - 1, f) : last_v;
         double delta  = (oldest == DBL_MAX || last_v == DBL_MAX) ? 0.0 : last_v - oldest;

         DirFeatSet(sym_idx, f,                     (float)((last_v == DBL_MAX) ? 0.0 : last_v));
         DirFeatSet(sym_idx, RAW_FEATURES + f,      (float)smean);
         DirFeatSet(sym_idx, RAW_FEATURES * 2 + f,  (float)sstd);
         DirFeatSet(sym_idx, RAW_FEATURES * 3 + f,  (float)delta);
      }

      // mk4.3: BuildH1Block / BuildH4Block disabled. They wrote to slots
      // 96..135 — which OVERWROTE the M5 block we just computed. The
      // dim layout is now M5-only (slots 0..199); there are no HTF slots
      // to write. Methods are kept in this file as historical reference
      // for future re-implementation but are not called.

      g_features_valid[sym_idx] = (g_post_gap_bars[sym_idx] == 0 && nc >= 5);
   }

   //--- Build execution features (1120-dim) for sym_idx.
   //    Concatenates dir features (1000) + exec context block (120).
   //    FAE (1060-1079) and QTW (1080-1119) slots are zero-padded here;
   //    FundamentalLoader and QuantStateLoader overwrite them after this call.
   //    Call after Encode().
   void BuildExecFeatures(int sym_idx, const string canonical,
                          float dir_confidence, float dir_uncertainty,
                          int regime,
                          float open_lots_norm, float pnl_norm, float equity_dd)
   {
      for(int d = 0; d < FEATURE_DIM; d++)
         ExecFeatSet(sym_idx, d, DirFeatGet(sym_idx, d));

      // Exec context block (indices 1000..1059)
      datetime now  = TimeCurrent();
      MqlDateTime mdt; TimeToStruct(now, mdt);
      int dow = mdt.day_of_week;
      int hr  = mdt.hour;
      int mi  = mdt.min;

      double mid = 0.0;
      if(g_tick_buf.Count(sym_idx) > 0)
      {
         STick t0; g_tick_buf.Get(sym_idx, 0, t0);
         mid = (t0.bid + t0.ask) * 0.5;
      }

      double pip = g_broker.PipSize(canonical);
      if(pip <= 0.0) pip = 0.0001;

      double atr14_i = g_atr14[sym_idx];
      double atr5_i  = g_atr5[sym_idx];
      double atr50_i = MathMax(g_atr50[sym_idx], 1e-10);
      double norm_c  = (mid > 0.0) ? 1.0 / mid : 1.0;

      double spd_pips = g_broker.SpreadPips(canonical);

      int j = FEATURE_DIM;  // start of exec context block

      ExecFeatSet(sym_idx, j, (float)MathMin(atr14_i * norm_c * 100.0, 2.0)); j++;
      ExecFeatSet(sym_idx, j, (float)MathMin(atr14_i / pip / 100.0, 1.0)); j++;
      ExecFeatSet(sym_idx, j, (float)MathMin(atr5_i / atr50_i, 5.0) / 5.0f); j++;
      ExecFeatSet(sym_idx, j, (float)MathMin(spd_pips / 20.0, 1.0)); j++;
      ExecFeatSet(sym_idx, j, 0.0f); j++;
      ExecFeatSet(sym_idx, j, 0.0f); j++;

      for(int d = 1; d <= 5; d++) { ExecFeatSet(sym_idx, j, (float)(dow == d ? 1.0 : 0.0)); j++; }

      ExecFeatSet(sym_idx, j, (float)SinEncode(hr, 24)); j++;
      ExecFeatSet(sym_idx, j, (float)CosEncode(hr, 24)); j++;
      ExecFeatSet(sym_idx, j, (float)SinEncode(mi, 60)); j++;
      ExecFeatSet(sym_idx, j, (float)CosEncode(mi, 60)); j++;

      ExecFeatSet(sym_idx, j, (float)(hr >= 8 && hr < 17 ? 1 : 0)); j++;
      ExecFeatSet(sym_idx, j, (float)(hr >= 13 && hr < 22 ? 1 : 0)); j++;
      ExecFeatSet(sym_idx, j, (float)(hr < 9 ? 1 : 0)); j++;
      ExecFeatSet(sym_idx, j, (float)(hr >= 21 || hr < 6 ? 1 : 0)); j++;
      ExecFeatSet(sym_idx, j, (float)(hr >= 13 && hr < 17 ? 1 : 0)); j++;

      int min_now = hr * 60 + mi;
      ExecFeatSet(sym_idx, j, (float)MathMax(0, 8*60 - min_now) / 1440.0f); j++;
      ExecFeatSet(sym_idx, j, (float)MathMax(0, 17*60 - min_now) / 1440.0f); j++;
      ExecFeatSet(sym_idx, j, (float)MathMax(0, 21*60 - min_now) / 1440.0f); j++;
      int days_to_fri = (5 - mdt.day_of_week + 7) % 7;
      ExecFeatSet(sym_idx, j, (float)(days_to_fri * 1440 + MathMax(0, 21*60 - min_now)) / 10080.0f); j++;

      ExecFeatSet(sym_idx, j, 0.0f); j++;
      ExecFeatSet(sym_idx, j, 0.0f); j++;
      ExecFeatSet(sym_idx, j, 0.0f); j++;
      ExecFeatSet(sym_idx, j, 0.0f); j++;

      for(int x = 0; x < 4; x++) { ExecFeatSet(sym_idx, j, 0.0f); j++; }

      ExecFeatSet(sym_idx, j, (float)MathMin(open_lots_norm, 1.0f)); j++;
      ExecFeatSet(sym_idx, j, (float)MathMax(-1.0f, MathMin(pnl_norm, 1.0f))); j++;
      ExecFeatSet(sym_idx, j, (float)MathMin(equity_dd, 1.0f)); j++;

      ExecFeatSet(sym_idx, j, dir_confidence); j++;
      ExecFeatSet(sym_idx, j, dir_uncertainty); j++;

      for(int r = 0; r < 3; r++) { ExecFeatSet(sym_idx, j, (float)(regime == r ? 1 : 0)); j++; }

      // mk4.2: macro/FAE/QTW blocks removed. Zero-pad remaining exec-context
      // slots up to EXEC_FEATURE_DIM (1200). Slots 4..39 are EA-injected
      // position context — written by the trade-management code, not here.
      while(j < EXEC_FEATURE_DIM) { ExecFeatSet(sym_idx, j, 0.0f); j++; }
   }

   void GetDirFeatures(int sym_idx, float &out[])
   {
      ArrayResize(out, FEATURE_DIM);
      for(int d = 0; d < FEATURE_DIM; d++)
         out[d] = g_dir_features[sym_idx * FEATURE_DIM + d];
   }

   void GetExecFeatures(int sym_idx, float &out[])
   {
      ArrayResize(out, EXEC_FEATURE_DIM);
      for(int d = 0; d < EXEC_FEATURE_DIM; d++)
         out[d] = g_exec_features[sym_idx * EXEC_FEATURE_DIM + d];
   }

   bool IsValid(int sym_idx) { return g_features_valid[sym_idx]; }
};

CFeatureEncoder g_encoder;

#endif // FEATUREENCODER_MQH
