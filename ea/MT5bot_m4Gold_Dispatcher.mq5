//+------------------------------------------------------------------+
//| MT5bot_m4Gold_Dispatcher.mq5 — GOLD-only multi-strategy EA.       |
//|                                                                    |
//| Runs any combination of three strategies, each gated by its own    |
//| spec JSON's `deploy: true` flag. Drop spec files into MT5's        |
//| Common Files folder and the corresponding strategy auto-enables.   |
//|                                                                    |
//|   H4 TREND    spec: M4GOLD_H4TREND_GOLD_spec.json                  |
//|     - state-based long/short on H1 bars (MA cross or momentum)    |
//|     - position held until trend flips                             |
//|                                                                    |
//|   H5 SCALP    spec: M4GOLD_H5SCALP_GOLD_spec.json                  |
//|     - intraday pullback entries on M5 inside H4 trend direction    |
//|     - triple-barrier exits (TP, SL, timeout)                       |
//|                                                                    |
//|   H6 MR       spec: M4GOLD_H6MR_GOLD_spec.json                     |
//|     - GOLD-only intraday mean-reversion (z-score on H1 log price)  |
//|     - long when oversold, short when overbought, exit on reversion |
//|                                                                    |
//| AI gate (optional): if `M4GOLD_GOLD_GOLD_dir_det.onnx` is present  |
//| in Common Files, the trained ONNX direction head must AGREE with   |
//| each strategy's entry signal before the EA fires. Disable with     |
//| InpUseAiGate=false.                                                |
//|                                                                    |
//| MAGIC LAYOUT (per-strategy isolation):                             |
//|   InpMagicBase + 4   = H4 trend                                   |
//|   InpMagicBase + 5   = H5 scalp                                   |
//|   InpMagicBase + 6   = H6 mean-reversion                          |
//+------------------------------------------------------------------+
#property copyright "MT5bot_m4Gold — GOLD-only dispatcher"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>
#include "includes/TrendRule.mqh"
#include "includes/FeatureEncoder.mqh"

// ====================================================================
// INPUTS
// ====================================================================
input string  InpEnabledStrategies = "H4,H5,H6";
input long    InpMagicBase         = 49000;
input double  InpLotsH4            = 0.01;
input double  InpLotsH5            = 0.01;
input double  InpLotsH6            = 0.01;
input bool    InpUseAiGate         = true;        // require ONNX agreement on entries
input string  InpAiOnnxFile        = "M4GOLD_GOLD_GOLD_dir_det.onnx";
input double  InpAiConfMin         = 0.55;        // P(class) threshold
input double  InpH5PullbackK       = 1.0;
input double  InpH5SlAtr           = 0.7;
input double  InpH5TpAtr           = 1.5;
input int     InpH5TimeoutBars     = 12;
input double  InpH5MaWindow        = 20;
input int     InpH5AtrPeriod       = 14;
input double  InpH6ZIn             = 2.0;
input double  InpH6ZOut            = 0.5;
input double  InpH6ZStop           = 3.5;
input int     InpH6ZWindow         = 200;
input int     InpH6TimeoutBars     = 48;
input bool    InpVerboseLog        = true;

// ====================================================================
// STATE
// ====================================================================
bool   g_h4_enabled = false;
bool   g_h5_enabled = false;
bool   g_h6_enabled = false;
bool   g_ai_loaded  = false;
long   g_ai_handle  = INVALID_HANDLE;
TR_Spec  g_h4_spec;
TR_State g_h4_state_prev;
datetime g_last_m5 = 0;
datetime g_last_h1 = 0;
int    g_h6_open_dir   = 0;
ulong  g_h6_ticket     = 0;
double g_h6_open_z     = 0;
datetime g_h6_open_time = 0;

CTrade trade_h4, trade_h5, trade_h6;

// ====================================================================
// HELPERS
// ====================================================================
bool _IsOn(const string list, const string key)
{
   return StringFind(list, key) >= 0;
}

bool _JsonGetStr(const string content, const string key, string &out)
{
   int p = StringFind(content, "\"" + key + "\"");
   if(p < 0) return false;
   int colon = StringFind(content, ":", p);
   if(colon < 0) return false;
   int start = colon + 1;
   while(start < StringLen(content) &&
         (StringSubstr(content, start, 1) == " " ||
          StringSubstr(content, start, 1) == "\t" ||
          StringSubstr(content, start, 1) == "\n")) start++;
   if(StringSubstr(content, start, 1) == "\"")
   {
      int q2 = StringFind(content, "\"", start + 1);
      if(q2 < 0) return false;
      out = StringSubstr(content, start + 1, q2 - start - 1);
      return true;
   }
   int end1 = StringFind(content, ",", start);
   int end2 = StringFind(content, "}", start);
   int end = (end1 < 0) ? end2 : ((end2 < 0) ? end1 : MathMin(end1, end2));
   if(end < 0) return false;
   out = StringSubstr(content, start, end - start);
   StringTrimLeft(out); StringTrimRight(out);
   return true;
}

bool _JsonGetBool(const string content, const string key)
{
   string v;
   if(!_JsonGetStr(content, key, v)) return false;
   return StringFind(v, "true") >= 0;
}

double _JsonGetDouble(const string content, const string key, double dflt)
{
   string v;
   if(!_JsonGetStr(content, key, v)) return dflt;
   return StringToDouble(v);
}

bool _ReadSpec(const string fname, string &out)
{
   // FILE_ANSI required — Python-written spec JSON is single-byte ASCII;
   // without it MT5 reads FILE_TXT as UTF-16 and the parse fails.
   int h = FileOpen(fname, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE) { out = ""; return false; }
   out = "";
   while(!FileIsEnding(h)) out += FileReadString(h);
   FileClose(h);
   return true;
}

// AI gate: returns +1 / 0 / -1 (model's directional vote). If the model
// is not loaded or the confidence floor is missed, returns 0 (no vote,
// i.e. veto). The 200-dim feature window is built by FeatureEncoder.mqh.
int _AiVote()
{
   if(!InpUseAiGate || !g_ai_loaded) return 0;
   float feats[];
   if(!FE_Encode(_Symbol, PERIOD_M5, feats)) return 0;
   if(ArraySize(feats) != FE_DIM) return 0;
   float out[];
   ArrayResize(out, 3);  // [P_short, P_flat, P_long]
   ulong in_shape[]  = {1, FE_DIM};
   ulong out_shape[] = {1, 3};
   if(!OnnxSetInputShape (g_ai_handle, 0, in_shape))  return 0;
   if(!OnnxSetOutputShape(g_ai_handle, 0, out_shape)) return 0;
   if(!OnnxRun(g_ai_handle, ONNX_DEFAULT, feats, out)) return 0;
   double p_short = out[0], p_flat = out[1], p_long = out[2];
   if(p_long >= InpAiConfMin && p_long > p_short)  return +1;
   if(p_short >= InpAiConfMin && p_short > p_long) return -1;
   return 0;
}

// ====================================================================
// OnInit
// ====================================================================
int OnInit()
{
   string s = _Symbol;
   if(StringFind(s, "GOLD") < 0 && StringFind(s, "XAU") < 0)
   {
      PrintFormat("[m4Gold] WARNING: attached to %s — this EA is GOLD-only "
                  "by design. Verify your broker's gold symbol naming.", s);
   }
   trade_h4.SetExpertMagicNumber(InpMagicBase + 4);
   trade_h5.SetExpertMagicNumber(InpMagicBase + 5);
   trade_h6.SetExpertMagicNumber(InpMagicBase + 6);
   trade_h4.SetTypeFillingBySymbol(s);
   trade_h5.SetTypeFillingBySymbol(s);
   trade_h6.SetTypeFillingBySymbol(s);

   // AI gate
   if(InpUseAiGate)
   {
      g_ai_handle = OnnxCreate(InpAiOnnxFile, ONNX_COMMON_FOLDER);
      if(g_ai_handle != INVALID_HANDLE)
      {
         g_ai_loaded = true;
         PrintFormat("[m4Gold] AI gate ON  model=%s  conf_min=%.2f",
                     InpAiOnnxFile, InpAiConfMin);
      }
      else
      {
         PrintFormat("[m4Gold] AI gate requested but ONNX %s NOT FOUND in "
                     "Common Files — running strategies with no AI gate. "
                     "Train it with: python python/train.py gold",
                     InpAiOnnxFile);
      }
   }

   // H4
   if(_IsOn(InpEnabledStrategies, "H4"))
   {
      string spec = "M4GOLD_H4TREND_" + s + "_spec.json";
      if(TR_LoadSpec(spec, g_h4_spec))
      {
         string raw;
         if(_ReadSpec(spec, raw) && _JsonGetBool(raw, "deploy"))
         {
            g_h4_enabled = true;
            g_h4_state_prev.position = 0;
            PrintFormat("[m4Gold] H4 ON  kind=%s  tf=%s  fast=%d  slow=%d",
                        g_h4_spec.kind == TR_MA_CROSS ? "ma_cross" : "momentum",
                        EnumToString(g_h4_spec.timeframe),
                        g_h4_spec.fast, g_h4_spec.slow);
         }
         else PrintFormat("[m4Gold] H4 spec %s found but deploy=false — H4 OFF", spec);
      }
      else PrintFormat("[m4Gold] H4 spec %s missing — H4 OFF", spec);
   }

   // H5
   if(_IsOn(InpEnabledStrategies, "H5"))
   {
      string spec = "M4GOLD_H5SCALP_" + s + "_spec.json";
      string raw;
      if(_ReadSpec(spec, raw) && _JsonGetBool(raw, "deploy"))
      {
         g_h5_enabled = true;
         PrintFormat("[m4Gold] H5 ON  pullback_k=%.2f  SL=%.2f×ATR  TP=%.2f×ATR  to=%d",
                     InpH5PullbackK, InpH5SlAtr, InpH5TpAtr, InpH5TimeoutBars);
      }
      else PrintFormat("[m4Gold] H5 spec %s missing or deploy=false — H5 OFF", spec);
   }

   // H6 (GOLD-only mean reversion)
   if(_IsOn(InpEnabledStrategies, "H6"))
   {
      string spec = "M4GOLD_H6MR_" + s + "_spec.json";
      string raw;
      if(_ReadSpec(spec, raw) && _JsonGetBool(raw, "deploy"))
      {
         g_h6_enabled = true;
         PrintFormat("[m4Gold] H6 ON  z_in=%.2f  z_out=%.2f  z_stop=%.2f  win=%d",
                     InpH6ZIn, InpH6ZOut, InpH6ZStop, InpH6ZWindow);
      }
      else PrintFormat("[m4Gold] H6 spec %s missing or deploy=false — H6 OFF", spec);
   }

   if(!g_h4_enabled && !g_h5_enabled && !g_h6_enabled)
      Print("[m4Gold] no strategies enabled — EA idle. Run "
            "`python python/train_strategies.py` then drop the spec JSONs "
            "into MT5\\Files\\Common\\.");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_ai_loaded && g_ai_handle != INVALID_HANDLE)
      OnnxRelease(g_ai_handle);
   PrintFormat("[m4Gold] deinit  reason=%d", reason);
}

// ====================================================================
// MAIN TICK LOOP
// ====================================================================
void OnTick()
{
   datetime cur_m5 = iTime(_Symbol, PERIOD_M5, 0);
   datetime cur_h1 = iTime(_Symbol, PERIOD_H1, 0);

   bool new_h1 = (cur_h1 != g_last_h1);
   bool new_m5 = (cur_m5 != g_last_m5);
   if(new_h1) g_last_h1 = cur_h1;
   if(new_m5) g_last_m5 = cur_m5;

   if(g_h4_enabled && new_h1) _ServeH4();
   if(g_h5_enabled && new_m5) _ServeH5();
   if(g_h6_enabled && new_h1) _ServeH6();
}

// ====================================================================
// H4 — TREND
// ====================================================================
void _ServeH4()
{
   TR_State st = TR_ComputeState(_Symbol, g_h4_spec);
   if(st.position == g_h4_state_prev.position) return;
   if(InpVerboseLog)
      PrintFormat("[H4] state change %d -> %d  reason=%s",
                  g_h4_state_prev.position, st.position, st.reason);
   int rule_dir = st.position;
   int ai_dir   = _AiVote();
   if(InpUseAiGate && g_ai_loaded && rule_dir != 0 && ai_dir != rule_dir)
   {
      if(InpVerboseLog)
         PrintFormat("[H4] AI veto — rule=%d ai=%d, holding", rule_dir, ai_dir);
      // close existing position if AI disagrees, but do not open new one
      _CloseAllH4();
      g_h4_state_prev.position = 0;
      return;
   }
   g_h4_state_prev = st;
   _CloseAllH4();
   if(rule_dir == 0) return;
   double price = (rule_dir > 0)
      ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
      : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   bool ok = (rule_dir > 0)
      ? trade_h4.Buy (InpLotsH4, _Symbol, price, 0, 0, "M4GOLD-H4")
      : trade_h4.Sell(InpLotsH4, _Symbol, price, 0, 0, "M4GOLD-H4");
   PrintFormat("[H4] %s %s @ %.5f rc=%d", ok ? "FIRED" : "FAILED",
               rule_dir > 0 ? "BUY" : "SELL", price, trade_h4.ResultRetcode());
}

void _CloseAllH4()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (InpMagicBase + 4)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      trade_h4.PositionClose(t);
   }
}

// ====================================================================
// H5 — SCALP (M5 pullback inside H4 trend)
// ====================================================================
void _ServeH5()
{
   if(_HasPosWithMagic(InpMagicBase + 5)) { _CloseH5Timeouts(); return; }
   int trend = g_h4_state_prev.position;
   if(trend == 0) return;

   int need = (int)InpH5MaWindow + InpH5AtrPeriod + 5;
   MqlRates r[];
   int got = CopyRates(_Symbol, PERIOD_M5, 0, need, r);
   if(got < need) return;
   int last = ArraySize(r) - 2;
   double ma = 0;
   for(int k = 0; k < (int)InpH5MaWindow; k++) ma += r[last - k].close;
   ma /= InpH5MaWindow;
   double tr_sum = 0;
   for(int k = 0; k < InpH5AtrPeriod; k++)
   {
      int j = last - k;
      double pc = r[j - 1].close;
      double tr = MathMax(r[j].high - r[j].low,
                          MathMax(MathAbs(r[j].high - pc),
                                  MathAbs(r[j].low  - pc)));
      tr_sum += tr;
   }
   double atr = tr_sum / InpH5AtrPeriod;
   if(atr <= 0) return;
   double price = r[last].close;
   double threshold_long  = ma - InpH5PullbackK * atr;
   double threshold_short = ma + InpH5PullbackK * atr;
   int direction = 0;
   if(trend > 0 && r[last].low <= threshold_long && price > threshold_long)
      direction = +1;
   else if(trend < 0 && r[last].high >= threshold_short && price < threshold_short)
      direction = -1;
   if(direction == 0) return;
   if(InpUseAiGate && g_ai_loaded)
   {
      int ai = _AiVote();
      if(ai != direction)
      {
         if(InpVerboseLog)
            PrintFormat("[H5] AI veto — rule=%d ai=%d", direction, ai);
         return;
      }
   }

   double sl, tp;
   if(direction > 0) { sl = price - InpH5SlAtr * atr; tp = price + InpH5TpAtr * atr; }
   else              { sl = price + InpH5SlAtr * atr; tp = price - InpH5TpAtr * atr; }
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   sl = NormalizeDouble(sl, digits); tp = NormalizeDouble(tp, digits);
   double entry = (direction > 0)
      ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
      : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   bool ok = (direction > 0)
      ? trade_h5.Buy (InpLotsH5, _Symbol, entry, sl, tp, "M4GOLD-H5")
      : trade_h5.Sell(InpLotsH5, _Symbol, entry, sl, tp, "M4GOLD-H5");
   PrintFormat("[H5] %s %s entry=%.5f sl=%.5f tp=%.5f atr=%.5f rc=%d",
               ok ? "FIRED" : "FAILED", direction > 0 ? "BUY" : "SELL",
               entry, sl, tp, atr, trade_h5.ResultRetcode());
}

void _CloseH5Timeouts()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (InpMagicBase + 5)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
      int bars_open = iBarShift(_Symbol, PERIOD_M5, opened);
      if(bars_open >= InpH5TimeoutBars)
      {
         trade_h5.PositionClose(t);
         if(InpVerboseLog)
            PrintFormat("[H5] TIMEOUT-CLOSE ticket=%I64u bars=%d", t, bars_open);
      }
   }
}

// ====================================================================
// H6 — GOLD-only mean reversion (z-score on H1 log close)
// ====================================================================
void _ServeH6()
{
   int need = InpH6ZWindow + 5;
   MqlRates r[];
   if(CopyRates(_Symbol, PERIOD_H1, 0, need, r) < need) return;
   int last = ArraySize(r) - 2;

   double sum = 0;
   for(int k = 0; k < InpH6ZWindow; k++) sum += MathLog(r[last - k].close);
   double mean = sum / InpH6ZWindow;
   double sd2 = 0;
   for(int k = 0; k < InpH6ZWindow; k++)
   {
      double d = MathLog(r[last - k].close) - mean;
      sd2 += d * d;
   }
   double sd = MathSqrt(sd2 / (InpH6ZWindow - 1));
   if(sd < 1e-12) return;
   double z = (MathLog(r[last].close) - mean) / sd;
   if(InpVerboseLog)
      PrintFormat("[H6] z=%+.3f  open_dir=%d", z, g_h6_open_dir);

   if(g_h6_open_dir != 0)
   {
      bool exit = false; string reason = "";
      datetime now = TimeCurrent();
      if(MathAbs(z) <= InpH6ZOut)        { exit = true; reason = "mean_revert"; }
      else if(MathAbs(z) >= InpH6ZStop)  { exit = true; reason = "stop"; }
      else if(now - g_h6_open_time >= (long)InpH6TimeoutBars * 3600)
                                          { exit = true; reason = "timeout"; }
      if(exit)
      {
         _CloseH6();
         PrintFormat("[H6] EXIT (%s) z=%.3f", reason, z);
         g_h6_open_dir = 0;
         g_h6_ticket = 0;
      }
      return;
   }

   int dir = 0;
   if(z >=  InpH6ZIn) dir = -1;     // overbought: short GOLD
   else if(z <= -InpH6ZIn) dir = +1; // oversold: long GOLD
   if(dir == 0) return;
   if(InpUseAiGate && g_ai_loaded)
   {
      int ai = _AiVote();
      // For MR entries, AI should NOT strongly disagree. Equal vote or
      // neutral (0) is acceptable; only veto when AI is on the opposite
      // side with conviction.
      if(ai == -dir)
      {
         if(InpVerboseLog)
            PrintFormat("[H6] AI veto — rule=%d ai=%d", dir, ai);
         return;
      }
   }

   double price = (dir > 0)
      ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
      : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   bool ok = (dir > 0)
      ? trade_h6.Buy (InpLotsH6, _Symbol, price, 0, 0, "M4GOLD-H6")
      : trade_h6.Sell(InpLotsH6, _Symbol, price, 0, 0, "M4GOLD-H6");
   if(ok)
   {
      g_h6_open_dir = dir;
      g_h6_ticket   = trade_h6.ResultOrder();
      g_h6_open_z   = z;
      g_h6_open_time = TimeCurrent();
      PrintFormat("[H6] ENTRY dir=%d z=%+.3f @ %.5f", dir, z, price);
   }
   else PrintFormat("[H6] FAILED rc=%d", trade_h6.ResultRetcode());
}

void _CloseH6()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != (InpMagicBase + 6)) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      trade_h6.PositionClose(t);
   }
}

// ====================================================================
// SHARED
// ====================================================================
bool _HasPosWithMagic(const long magic)
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0 || !PositionSelectByTicket(t)) continue;
      if((long)PositionGetInteger(POSITION_MAGIC) != magic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      return true;
   }
   return false;
}
