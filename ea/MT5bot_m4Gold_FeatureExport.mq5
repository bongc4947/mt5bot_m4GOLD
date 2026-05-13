//+------------------------------------------------------------------+
//|        MT5_Bot_mk4_FeatureExport.mq5                              |
//|        HYDRA mk4.2.1 — MT5-canonical feature exporter             |
//|                                                                    |
//| Compute the 1160-dim direction-feature vector for every M5 bar    |
//| of the chart-attached symbol and write the result to two files    |
//| in the MT5 Common Files folder, ready for Python training:        |
//|                                                                    |
//|   HYDRA4_FEAT_<SYMBOL>_M5.bin           float32, (N, FEATURE_DIM)  |
//|   HYDRA4_FEAT_<SYMBOL>_M5.meta.json     N, dim, time range, etc.   |
//|   HYDRA4_FEAT_<SYMBOL>_M5_times.bin     int64 UTC seconds          |
//|                                                                    |
//| The file format is consumed by python/load_mt5_features.py and    |
//| python/_train_agent.py (--mt5-features). Feature math comes from  |
//| FeatureEncoder.mqh — the same code that runs at inference. Train  |
//| and live therefore share a single feature implementation.         |
//|                                                                    |
//| USAGE (one symbol at a time):                                      |
//|   1. Open the chart for the symbol you want (M5 timeframe).        |
//|   2. Drag this script onto the chart (Navigator -> Scripts).       |
//|   3. Adjust the InpMaxBars input if you only want a subset.        |
//|   4. Wait for the "Done" log line — output lands in Common Files.  |
//|   5. Repeat for each symbol you want to train on.                  |
//|                                                                    |
//| HONEST NOTE: this file calls FeatureEncoder.mqh's public           |
//| BuildFeaturesForBar() helper. If your local FeatureEncoder.mqh     |
//| does not expose that helper yet, see the scaffold in               |
//| ea/includes/FeatureEncoder.mqh (top of file) and add a thin batch  |
//| wrapper around the existing internal Build() pipeline. The Python  |
//| side, file format, and trainer wiring are complete and tested.     |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict
#property copyright "HYDRA mk4.2.1"
#property version   "4.21"

#include "includes/Defines.mqh"
#include "includes/MarketHours.mqh"
#include "includes/TickBuffer.mqh"
#include "includes/FeatureEncoder.mqh"

input int    InpMaxBars     = 0;       // 0 = all available history; else cap
input bool   InpWriteTimes  = true;    // also write per-row UTC epoch sidecar
input string InpOutPrefix   = "HYDRA4_FEAT";  // file basename prefix

//+------------------------------------------------------------------+
void OnStart()
{
   string sym = _Symbol;
   datetime started_at = TimeGMT();

   //--- 1. Resolve how many bars we have on this M5 chart.
   int total_bars = (int)Bars(sym, PERIOD_M5);
   if(total_bars <= 0) {
      PrintFormat("[FeatureExport] %s: no bars available", sym);
      return;
   }
   int n = (InpMaxBars > 0 && InpMaxBars < total_bars) ? InpMaxBars : total_bars;
   PrintFormat("[FeatureExport] %s: exporting %d M5 bars (FEATURE_DIM=%d)",
               sym, n, FEATURE_DIM);

   //--- 2. Open output files.
   string bin_name   = StringFormat("%s_%s_M5.bin",       InpOutPrefix, sym);
   string meta_name  = StringFormat("%s_%s_M5.meta.json", InpOutPrefix, sym);
   string times_name = StringFormat("%s_%s_M5_times.bin", InpOutPrefix, sym);

   int bin_fh = FileOpen(bin_name, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(bin_fh == INVALID_HANDLE) {
      PrintFormat("[FeatureExport] FileOpen failed for %s (err %d)",
                  bin_name, GetLastError());
      return;
   }

   int times_fh = INVALID_HANDLE;
   if(InpWriteTimes) {
      times_fh = FileOpen(times_name, FILE_WRITE | FILE_BIN | FILE_COMMON);
      if(times_fh == INVALID_HANDLE)
         PrintFormat("[FeatureExport] WARN: times sidecar disabled (err %d)",
                     GetLastError());
   }

   //--- 3. Iterate from oldest to newest. Bar index 0 in MT5 is *current*,
   //       so we walk shift = (n-1) -> 0 to write rows in time order.
   ulong t0 = GetTickCount64();
   datetime first_time = 0, last_time = 0;
   int sym_idx = 0;          // single-symbol export -> always 0
   float fbuf[];
   ArrayResize(fbuf, FEATURE_DIM);

   for(int i = n - 1; i >= 0; i--)
   {
      datetime bt = iTime(sym, PERIOD_M5, i);
      if(i == n - 1) first_time = bt;
      if(i == 0)     last_time  = bt;

      // Compute features for this bar via the live encoder. The wrapper
      // takes (sym, sym_idx, shift) and fills g_dir_features[sym_idx, *].
      // See FeatureEncoder.mqh::BuildFeaturesForBar — a thin batch shim.
      g_encoder.BuildFeaturesForBar(sym, sym_idx, i);
      g_encoder.GetDirFeatures(sym_idx, fbuf);

      // FileWriteArray writes count*sizeof(float) bytes — exactly what
      // numpy.fromfile(dtype=np.float32) consumes on the Python side.
      FileWriteArray(bin_fh, fbuf, 0, FEATURE_DIM);

      if(times_fh != INVALID_HANDLE) {
         long ts = (long)bt;
         FileWriteLong(times_fh, ts);
      }

      if((n - i) % 50000 == 0)
         PrintFormat("[FeatureExport] %s: %d / %d bars",
                     sym, n - i, n);
   }
   FileClose(bin_fh);
   if(times_fh != INVALID_HANDLE) FileClose(times_fh);

   double elapsed_s = (GetTickCount64() - t0) / 1000.0;
   PrintFormat("[FeatureExport] %s: %d bars in %.1f s  (%.0f bars/s)",
               sym, n, elapsed_s, n / MathMax(elapsed_s, 1e-9));

   //--- 4. Write JSON sidecar (hand-rolled — MQL5 has no json writer).
   string j  = "{\n";
   j += "  \"schema_version\": \"1.0\",\n";
   j += StringFormat("  \"symbol\": \"%s\",\n", sym);
   j += "  \"timeframe\": \"M5\",\n";
   j += StringFormat("  \"n_rows\": %d,\n", n);
   j += StringFormat("  \"dim\": %d,\n", FEATURE_DIM);
   j += StringFormat("  \"time_start\": \"%s\",\n", TimeToString(first_time, TIME_DATE | TIME_SECONDS));
   j += StringFormat("  \"time_end\":   \"%s\",\n", TimeToString(last_time,  TIME_DATE | TIME_SECONDS));
   j += "  \"exported_by\": \"HYDRA4_FeatureExport.mq5\",\n";
   j += StringFormat("  \"exported_at\": \"%s\",\n",
                      TimeToString(started_at, TIME_DATE | TIME_SECONDS));
   j += StringFormat("  \"ea_version\": \"%s\",\n", HYDRA_VERSION);
   j += StringFormat("  \"feature_dim_dir\": %d,\n", FEATURE_DIM);
   j += "  \"block_starts\": {\n";
   j += StringFormat("    \"M5\": %d,\n",       FEAT_BLOCK_M5_START);
   j += StringFormat("    \"H1\": %d,\n",       FEAT_BLOCK_H1_START);
   j += StringFormat("    \"H4\": %d,\n",       FEAT_BLOCK_H4_START);
   j += StringFormat("    \"H8\": %d,\n",       FEAT_BLOCK_H8_START);
   j += StringFormat("    \"D1\": %d,\n",       FEAT_BLOCK_D1_START);
   j += StringFormat("    \"Spectral\": %d,\n", FEAT_BLOCK_SPECTRAL_START);
   j += StringFormat("    \"Pattern\": %d,\n",  FEAT_BLOCK_PATTERN_START);
   j += StringFormat("    \"StatReg\": %d,\n",  FEAT_BLOCK_STATREG_START);
   j += StringFormat("    \"XAsset\": %d\n",    FEAT_BLOCK_XASSET_START);
   j += "  }\n";
   j += "}\n";

   int meta_fh = FileOpen(meta_name, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(meta_fh == INVALID_HANDLE) {
      PrintFormat("[FeatureExport] FileOpen failed for meta %s (err %d)",
                  meta_name, GetLastError());
      return;
   }
   FileWriteString(meta_fh, j);
   FileClose(meta_fh);

   PrintFormat("[FeatureExport] Done. Wrote %s + %s%s",
               bin_name, meta_name, InpWriteTimes ? (" + " + times_name) : "");
   PrintFormat("[FeatureExport] Common Files folder: %s",
               TerminalInfoString(TERMINAL_COMMONDATA_PATH));
}
//+------------------------------------------------------------------+
