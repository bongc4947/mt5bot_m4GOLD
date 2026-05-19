//+------------------------------------------------------------------+
//| TrendRule.mqh — H4 long-horizon trend-following.                  |
//|                                                                    |
//| Two deterministic rule kinds; pick via spec JSON written by       |
//| python/train_h4_trend.py (HYDRA4_H4TREND_<sym>_spec.json):         |
//|                                                                    |
//|   ma_cross :  long when MA(fast)  > MA(slow) on closed H1/H4 bars,|
//|               short (if allowed) when MA(fast) < MA(slow).         |
//|   momentum :  long when close > close[lookback], short otherwise. |
//|                                                                    |
//| Returns TR_State in {-1, 0, +1} — the EA holds POSITION until     |
//| state changes (not event-based). Reverses on state flip.          |
//|                                                                    |
//| Strictly backward — uses only fully closed higher-timeframe bars. |
//+------------------------------------------------------------------+
#ifndef TREND_RULE_MQH
#define TREND_RULE_MQH

enum TR_Kind { TR_MA_CROSS = 0, TR_MOMENTUM = 1 };

struct TR_Spec
{
    TR_Kind kind;
    ENUM_TIMEFRAMES timeframe;
    int    fast;          // ma_cross only
    int    slow;          // ma_cross only
    int    lookback;      // momentum only
    bool   allow_short;
};

struct TR_State
{
    int    position;      // -1, 0, +1
    double ma_fast;
    double ma_slow;
    string reason;
};

ENUM_TIMEFRAMES TR_TimeframeFromString(const string s)
{
    if(s == "1h" || s == "H1") return PERIOD_H1;
    if(s == "4h" || s == "H4") return PERIOD_H4;
    if(s == "1d" || s == "D1") return PERIOD_D1;
    return PERIOD_H1;
}

bool TR_LoadSpec(const string spec_path, TR_Spec &out)
{
    // Spec is a JSON file in MQL5\Files\HYDRA4_H4TREND_<sym>_spec.json.
    // We parse a few key fields by simple string search rather than pulling
    // in a full JSON library — the keys we care about are fixed.
    // FILE_ANSI required — spec JSON is single-byte ASCII; without it
    // MT5 reads FILE_TXT as UTF-16 and every key lookup fails.
    int h = FileOpen(spec_path, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
    if(h == INVALID_HANDLE)
    {
        PrintFormat("[TrendRule] cannot open spec %s (err=%d)",
                    spec_path, GetLastError());
        return false;
    }
    string content = "";
    while(!FileIsEnding(h)) content += FileReadString(h);
    FileClose(h);

    // rule_kind
    int kpos = StringFind(content, "\"rule_kind\"");
    if(kpos < 0) { PrintFormat("[TrendRule] spec missing rule_kind"); return false; }
    string tail = StringSubstr(content, kpos);
    if(StringFind(tail, "\"ma_cross\"") >= 0)      out.kind = TR_MA_CROSS;
    else if(StringFind(tail, "\"momentum\"") >= 0) out.kind = TR_MOMENTUM;
    else { PrintFormat("[TrendRule] unknown rule_kind in spec"); return false; }

    // timeframe
    string tf = "1h";
    int tpos = StringFind(content, "\"timeframe\"");
    if(tpos >= 0)
    {
        int q1 = StringFind(content, "\"", tpos + 13);
        int q2 = StringFind(content, "\"", q1 + 1);
        if(q1 >= 0 && q2 > q1) tf = StringSubstr(content, q1 + 1, q2 - q1 - 1);
    }
    out.timeframe = TR_TimeframeFromString(tf);

    // params: fast, slow, lookback (cheap grep)
    out.fast = 50; out.slow = 200; out.lookback = 240;
    int fp = StringFind(content, "\"fast\":");
    int sp = StringFind(content, "\"slow\":");
    int lp = StringFind(content, "\"lookback\":");
    if(fp >= 0) out.fast     = (int)StringToInteger(StringSubstr(content, fp + 7, 8));
    if(sp >= 0) out.slow     = (int)StringToInteger(StringSubstr(content, sp + 7, 8));
    if(lp >= 0) out.lookback = (int)StringToInteger(StringSubstr(content, lp + 11, 8));

    int ap = StringFind(content, "\"allow_short\":");
    if(ap >= 0)
        out.allow_short = (StringFind(content, "true", ap) >= 0
                            && StringFind(content, "false", ap) < 0);
    else
        out.allow_short = true;
    return true;
}

TR_State TR_ComputeState(const string symbol, const TR_Spec &spec)
{
    TR_State st;
    st.position = 0; st.ma_fast = 0; st.ma_slow = 0; st.reason = "";

    if(spec.kind == TR_MA_CROSS)
    {
        int need = spec.slow + 3;
        double closes[];
        int got = CopyClose(symbol, spec.timeframe, 0, need, closes);
        if(got < need) { st.reason = "insufficient bars"; return st; }
        int last = got - 2;
        double s_fast = 0;
        for(int i = 0; i < spec.fast; i++) s_fast += closes[last - i];
        s_fast /= spec.fast;
        double s_slow = 0;
        for(int i = 0; i < spec.slow; i++) s_slow += closes[last - i];
        s_slow /= spec.slow;
        st.ma_fast = s_fast; st.ma_slow = s_slow;
        if(s_fast > s_slow)
        {
            st.position = +1;
            st.reason = StringFormat("MA cross UP  fast=%.5f > slow=%.5f", s_fast, s_slow);
        }
        else if(s_fast < s_slow && spec.allow_short)
        {
            st.position = -1;
            st.reason = StringFormat("MA cross DOWN  fast=%.5f < slow=%.5f", s_fast, s_slow);
        }
        else
        {
            st.reason = "MA flat / shorts disabled";
        }
    }
    else // TR_MOMENTUM
    {
        int need = spec.lookback + 3;
        double closes[];
        int got = CopyClose(symbol, spec.timeframe, 0, need, closes);
        if(got < need) { st.reason = "insufficient bars"; return st; }
        int last = got - 2;
        int back = last - spec.lookback;
        if(back < 0) { st.reason = "lookback warmup"; return st; }
        if(closes[last] > closes[back])
        {
            st.position = +1;
            st.reason = StringFormat("MOM UP  close=%.5f > close[lb]=%.5f",
                                     closes[last], closes[back]);
        }
        else if(closes[last] < closes[back] && spec.allow_short)
        {
            st.position = -1;
            st.reason = StringFormat("MOM DOWN  close=%.5f < close[lb]=%.5f",
                                     closes[last], closes[back]);
        }
        else
        {
            st.reason = "MOM flat / shorts disabled";
        }
    }
    return st;
}

#endif // TREND_RULE_MQH
