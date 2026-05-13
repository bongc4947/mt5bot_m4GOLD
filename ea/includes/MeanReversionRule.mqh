//+------------------------------------------------------------------+
//| MeanReversionRule.mqh — pure-rule signal helper.                  |
//|                                                                    |
//| Implements the z-score fade + vol-regime-gate rule validated in   |
//| BACKTEST_RESULTS.md (PF 1.71–4.22 over ~10 years out-of-sample on  |
//| 12 symbols at retail 2-pip cost):                                  |
//|                                                                    |
//|   z = z_score(log_return, window=20)                              |
//|   vol_ratio = std(last 20 returns) / std(last 200 returns)        |
//|   if vol_ratio < 0.70:               skip   (LOW vol regime)      |
//|   if z >= +2.5:  SHORT (fade up-spike)  SL = price + 1×ATR(14)   |
//|                                          TP = price - 2×ATR(14)   |
//|   if z <= -2.5:  LONG  (fade down-spike) SL = price - 1×ATR(14)   |
//|                                          TP = price + 2×ATR(14)   |
//|   timeout = 20 M5 bars (force-close at market)                    |
//|                                                                    |
//| No ML, no ONNX, no broker depth. Pure price math. Validated path.  |
//+------------------------------------------------------------------+
#ifndef MEAN_REVERSION_RULE_MQH
#define MEAN_REVERSION_RULE_MQH

struct MR_Signal
{
    int    direction;   // +1 long, -1 short, 0 no-trade
    double sl_price;
    double tp_price;
    double z_score;
    double atr;
    double vol_ratio;
    string reason;
};

// Mean and (sample) std of `a[start .. start+n-1]`.
void _MR_arr_stats(const double &a[], int start, int n, double &mean, double &stdv)
{
    double s = 0.0;
    for(int i = 0; i < n; i++) s += a[start + i];
    mean = s / n;
    double sq = 0.0;
    for(int i = 0; i < n; i++)
    {
        double d = a[start + i] - mean;
        sq += d * d;
    }
    stdv = MathSqrt(sq / MathMax(1, n - 1));
}

//+------------------------------------------------------------------+
//| MR_ComputeSignal                                                  |
//|                                                                    |
//| Inspects the most recent CLOSED M5 bar (rates[size-2]) of the     |
//| given symbol and decides if the rule fires. Returns MR_Signal     |
//| with direction=0 when there is no entry.                          |
//|                                                                    |
//| Must be called once per new bar. Output sl_price / tp_price are   |
//| absolute prices ready for OrderSend.                              |
//+------------------------------------------------------------------+
MR_Signal MR_ComputeSignal(const string symbol,
                            const double z_thresh           = 2.5,
                            const int    return_window      = 20,
                            const int    vol_baseline_window = 200,
                            const double vol_med_threshold  = 0.70,
                            const int    atr_period         = 14,
                            const double sl_atr_mult        = 1.0,
                            const double tp_atr_mult        = 2.0)
{
    MR_Signal sig;
    sig.direction = 0;
    sig.sl_price  = 0.0;
    sig.tp_price  = 0.0;
    sig.z_score   = 0.0;
    sig.atr       = 0.0;
    sig.vol_ratio = 0.0;
    sig.reason    = "";

    int need = vol_baseline_window + return_window + atr_period + 5;
    MqlRates rates[];
    int got = CopyRates(symbol, PERIOD_M5, 0, need, rates);
    if(got < need)
    {
        sig.reason = StringFormat("insufficient bars (got %d need %d)", got, need);
        return sig;
    }

    int n = ArraySize(rates);
    // Use the LAST CLOSED bar (n-2). The current bar (n-1) is incomplete.
    int last = n - 2;
    if(last < return_window) { sig.reason = "last-index too small"; return sig; }

    // 1. Log-returns close[i] -> close[i+1].
    double log_returns[];
    ArrayResize(log_returns, last);
    for(int i = 1; i <= last; i++)
    {
        if(rates[i - 1].close <= 0.0)
        {
            sig.reason = "non-positive close";
            return sig;
        }
        log_returns[i - 1] = MathLog(rates[i].close / rates[i - 1].close);
    }
    int last_ret = last - 1;
    if(last_ret < return_window) { sig.reason = "too few returns"; return sig; }

    // 2. z-score of the most recent return vs last `return_window` returns.
    double mean_r, std_r;
    _MR_arr_stats(log_returns, last_ret - return_window + 1, return_window, mean_r, std_r);
    if(std_r < 1e-12) { sig.reason = "zero short-window std"; return sig; }
    sig.z_score = (log_returns[last_ret] - mean_r) / std_r;

    // 3. Vol-regime gate — std(last 20) vs std(last 200), backward only.
    double vs_mean, vs_std;
    _MR_arr_stats(log_returns, last_ret - return_window + 1, return_window, vs_mean, vs_std);
    int vl_start = MathMax(0, last_ret - vol_baseline_window + 1);
    int vl_n     = last_ret - vl_start + 1;
    double vl_mean, vl_std;
    _MR_arr_stats(log_returns, vl_start, vl_n, vl_mean, vl_std);
    if(vl_std < 1e-12) { sig.reason = "zero baseline std"; return sig; }
    sig.vol_ratio = vs_std / vl_std;
    if(sig.vol_ratio < vol_med_threshold)
    {
        sig.reason = StringFormat("vol-regime LOW (ratio=%.2f < %.2f)",
                                  sig.vol_ratio, vol_med_threshold);
        return sig;
    }

    // 4. ATR for SL / TP placement.
    double atr_sum = 0.0;
    for(int i = last - atr_period + 1; i <= last; i++)
    {
        double prev_close = rates[i - 1].close;
        double tr = MathMax(rates[i].high - rates[i].low,
                            MathMax(MathAbs(rates[i].high - prev_close),
                                    MathAbs(rates[i].low  - prev_close)));
        atr_sum += tr;
    }
    sig.atr = atr_sum / atr_period;
    if(sig.atr <= 0.0) { sig.reason = "non-positive ATR"; return sig; }

    // 5. Entry decision.
    double price = rates[last].close;   // last closed-bar close as reference
    if(sig.z_score >= z_thresh)
    {
        sig.direction = -1;
        sig.sl_price  = price + sl_atr_mult * sig.atr;
        sig.tp_price  = price - tp_atr_mult * sig.atr;
        sig.reason    = StringFormat("FADE up-spike z=%.2f vol_ratio=%.2f",
                                     sig.z_score, sig.vol_ratio);
    }
    else if(sig.z_score <= -z_thresh)
    {
        sig.direction = +1;
        sig.sl_price  = price - sl_atr_mult * sig.atr;
        sig.tp_price  = price + tp_atr_mult * sig.atr;
        sig.reason    = StringFormat("FADE down-spike z=%.2f vol_ratio=%.2f",
                                     sig.z_score, sig.vol_ratio);
    }
    else
    {
        sig.reason = StringFormat("|z|=%.2f below threshold %.2f",
                                  MathAbs(sig.z_score), z_thresh);
    }

    return sig;
}

#endif // MEAN_REVERSION_RULE_MQH
