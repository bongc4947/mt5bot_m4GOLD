//+------------------------------------------------------------------+
//| SessionBreakoutRule.mqh — H2 session-open Donchian breakout.      |
//|                                                                    |
//| Each closed M5 bar inside the London (07:00-08:00 UTC) or NY      |
//| (13:30-14:30 UTC) open window, check whether close > prior N-bar  |
//| high  (LONG)  or close < prior N-bar low  (SHORT). If yes, emit   |
//| an SB_Signal with SL/TP based on ATR(14).                          |
//|                                                                    |
//| Strictly backward — donchian envelope uses bars [i-N .. i-1]. The |
//| current bar i is the one being TESTED.                            |
//|                                                                    |
//| Default thresholds (must match train_h2_session.py):              |
//|   donchian_window = 20      (M5 bars, ≈ 100 min lookback)         |
//|   sl_atr_mult     = 0.5                                            |
//|   tp_atr_mult     = 1.5                                            |
//|   timeout_bars    = 12      (≈ 1 hour, force-close if no SL/TP)   |
//|                                                                    |
//| If HYDRA4_H2SES_<sym>.onnx is present and InpUseMetaFilter=true   |
//| the dispatcher will gate signals via the trained meta classifier. |
//+------------------------------------------------------------------+
#ifndef SESSION_BREAKOUT_RULE_MQH
#define SESSION_BREAKOUT_RULE_MQH

#define SB_LONDON_OPEN_HOUR   7
#define SB_LONDON_OPEN_MIN    0
#define SB_LONDON_CLOSE_HOUR  8
#define SB_LONDON_CLOSE_MIN   0
#define SB_NY_OPEN_HOUR       13
#define SB_NY_OPEN_MIN        30
#define SB_NY_CLOSE_HOUR      14
#define SB_NY_CLOSE_MIN       30
#define SB_FEATURE_DIM        11

struct SB_Signal
{
    int    direction;    // +1 long, -1 short, 0 no-trade
    int    session_id;   // 0=London 1=NY
    double sl_price;
    double tp_price;
    double atr;
    double donchian_h;
    double donchian_l;
    string reason;
};

bool _SB_in_window(const datetime t, const int oh, const int om,
                    const int ch, const int cm)
{
    MqlDateTime dt; TimeToStruct(t, dt);
    int t_min = dt.hour * 60 + dt.min;
    int s     = oh * 60 + om;
    int e     = ch * 60 + cm;
    return (t_min >= s && t_min < e);
}

SB_Signal SB_ComputeSignal(const string symbol,
                            const int    donchian_window = 20,
                            const double sl_atr_mult     = 0.5,
                            const double tp_atr_mult     = 1.5,
                            const int    atr_period      = 14)
{
    SB_Signal sig;
    sig.direction = 0; sig.session_id = -1;
    sig.sl_price = 0; sig.tp_price = 0; sig.atr = 0;
    sig.donchian_h = 0; sig.donchian_l = 0; sig.reason = "";

    int need = donchian_window + atr_period + 5;
    MqlRates rates[];
    int got = CopyRates(symbol, PERIOD_M5, 0, need, rates);
    if(got < need) { sig.reason = "insufficient bars"; return sig; }
    int last = ArraySize(rates) - 2;   // last closed bar
    if(last < donchian_window + atr_period) { sig.reason = "warmup"; return sig; }

    datetime t = rates[last].time;
    bool in_london = _SB_in_window(t, SB_LONDON_OPEN_HOUR, SB_LONDON_OPEN_MIN,
                                       SB_LONDON_CLOSE_HOUR, SB_LONDON_CLOSE_MIN);
    bool in_ny     = _SB_in_window(t, SB_NY_OPEN_HOUR,     SB_NY_OPEN_MIN,
                                       SB_NY_CLOSE_HOUR,    SB_NY_CLOSE_MIN);
    if(!in_london && !in_ny) { sig.reason = "outside session window"; return sig; }
    sig.session_id = in_london ? 0 : 1;

    // Donchian envelope over PREVIOUS donchian_window bars (exclude current).
    double hh = -DBL_MAX, ll = DBL_MAX;
    for(int k = 1; k <= donchian_window; k++)
    {
        int i = last - k;
        if(i < 0) { sig.reason = "donchian warmup"; return sig; }
        hh = MathMax(hh, rates[i].high);
        ll = MathMin(ll, rates[i].low);
    }
    sig.donchian_h = hh; sig.donchian_l = ll;

    // ATR(14) over the last N closed bars (backward).
    double tr_sum = 0;
    for(int k = 0; k < atr_period; k++)
    {
        int i = last - k;
        if(i <= 0) { sig.reason = "atr warmup"; return sig; }
        double prev_close = rates[i - 1].close;
        double tr = MathMax(rates[i].high - rates[i].low,
                            MathMax(MathAbs(rates[i].high - prev_close),
                                    MathAbs(rates[i].low  - prev_close)));
        tr_sum += tr;
    }
    sig.atr = tr_sum / atr_period;
    if(sig.atr <= 0) { sig.reason = "atr<=0"; return sig; }

    double close = rates[last].close;
    if(close > hh)
    {
        sig.direction = +1;
        sig.sl_price = close - sl_atr_mult * sig.atr;
        sig.tp_price = close + tp_atr_mult * sig.atr;
        sig.reason = StringFormat("BREAKOUT_LONG  close=%.5f > donchian_h=%.5f",
                                  close, hh);
    }
    else if(close < ll)
    {
        sig.direction = -1;
        sig.sl_price = close + sl_atr_mult * sig.atr;
        sig.tp_price = close - tp_atr_mult * sig.atr;
        sig.reason = StringFormat("BREAKOUT_SHORT close=%.5f < donchian_l=%.5f",
                                  close, ll);
    }
    else
    {
        sig.reason = StringFormat("inside donchian envelope [%.5f, %.5f]", ll, hh);
    }
    return sig;
}

#endif // SESSION_BREAKOUT_RULE_MQH
