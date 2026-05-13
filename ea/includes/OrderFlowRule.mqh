//+------------------------------------------------------------------+
//| OrderFlowRule.mqh — live H1 order-flow imbalance helper.          |
//|                                                                    |
//| Maintains a rolling buffer of the last K ticks and exposes:        |
//|   OF_BuildFeatures(symbol, out_feats[16])                          |
//| which mirrors python/train_h1_orderflow.build_h1_features in       |
//| index order so the trained ONNX (HYDRA4_H1OF_<sym>.onnx) consumes   |
//| the same vector live as it saw in training.                       |
//|                                                                    |
//| Feature index map (must match H1_FEATURE_COLUMNS in Python):       |
//|    0  ofi_now            (signed-vol / total-vol of current bar)   |
//|    1  ofi_ema5           (EMA(5)   of bar OFI)                     |
//|    2  ofi_ema20          (EMA(20)  of bar OFI)                     |
//|    3  cvd_norm           (cum signed-vol / rolling abs-delta sum)  |
//|    4  taker_now          (fraction of ticks with non-zero direction)|
//|    5  taker_ema20        (EMA(20)  of taker_ratio)                 |
//|    6  ret_1              (log return over last 1 tick-bar  * 1e4)  |
//|    7  ret_5              (log return over last 5 tick-bars * 1e4)  |
//|    8  ret_20             (log return over last 20 bars    * 1e4)   |
//|    9  vol_5              (std of 5  last 1-bar log returns *1e4)   |
//|   10  vol_20             (std of 20 last 1-bar log returns *1e4)   |
//|   11  microdrift_5       (mid drift over last 5 bars  *1e4)        |
//|   12  microdrift_20      (mid drift over last 20 bars *1e4)        |
//|   13  spread_now / mid * 1e4                                       |
//|   14  spread_now / rolling_60_median_spread, clipped [0,5]/5       |
//|   15  tick intensity z-score (rolling 200 bars of total_volume)    |
//|                                                                    |
//| Caller responsibility: call OF_OnTick() once per incoming tick to  |
//| feed the buffer; call OF_OnBarClose() every N ticks (e.g. 100) to  |
//| roll the buffer into a fresh bar.                                  |
//+------------------------------------------------------------------+
#ifndef ORDER_FLOW_RULE_MQH
#define ORDER_FLOW_RULE_MQH

#define OF_FEATURE_DIM   16
#define OF_DEFAULT_TICKS_PER_BAR  100
#define OF_BUFFER_BARS   220   // enough for 200-bar tick intensity + warmup

struct OF_Bar
{
    datetime t_close;
    double   mid;
    double   spread;
    double   ofi;        // signed-vol / total-vol, [-1, 1]
    double   taker;      // fraction of non-zero signed ticks
    double   total_volume;
    double   signed_volume;
};

// Live ring buffer per symbol. The EA owns one of these via the dispatcher.
struct OF_State
{
    int     ticks_per_bar;
    int     tick_count_in_bar;
    double  sv_in_bar;     // running signed volume in current building bar
    double  av_in_bar;     // running |volume| in current building bar
    int     nz_in_bar;     // running count of non-zero-direction ticks
    double  spread_sum;
    double  mid_open;
    double  mid_high;
    double  mid_low;
    double  mid_last;
    OF_Bar  bars[OF_BUFFER_BARS];
    int     n_bars;
    int     head;          // ring head; bars[head-1] is the most recent
};

void OF_Init(OF_State &st, const int ticks_per_bar = OF_DEFAULT_TICKS_PER_BAR)
{
    st.ticks_per_bar = ticks_per_bar;
    st.tick_count_in_bar = 0;
    st.sv_in_bar = 0; st.av_in_bar = 0; st.nz_in_bar = 0;
    st.spread_sum = 0;
    st.mid_open = st.mid_high = st.mid_low = st.mid_last = 0;
    st.n_bars = 0;
    st.head = 0;
    for(int i = 0; i < OF_BUFFER_BARS; i++)
    {
        st.bars[i].t_close = 0;
        st.bars[i].mid = 0; st.bars[i].spread = 0;
        st.bars[i].ofi = 0; st.bars[i].taker = 0;
        st.bars[i].total_volume = 0; st.bars[i].signed_volume = 0;
    }
}

// Feed a single tick. Returns true if the call CLOSED a bar (so caller can
// optionally rebuild features and run the ONNX model).
bool OF_OnTick(OF_State &st, const double bid, const double ask,
                const double last_price, const double volume,
                const datetime t)
{
    double mid    = (bid + ask) * 0.5;
    double spread = ask - bid;
    if(st.tick_count_in_bar == 0)
    {
        st.mid_open = mid; st.mid_high = mid; st.mid_low = mid;
    }
    st.mid_high = MathMax(st.mid_high, mid);
    st.mid_low  = MathMin(st.mid_low,  mid);
    st.mid_last = mid;
    st.spread_sum += spread;
    // Lee-Ready trade direction
    int s = 0;
    if(last_price > mid + 1e-12) s = 1;
    else if(last_price < mid - 1e-12) s = -1;
    double v = (volume > 0) ? volume : 1.0;
    st.sv_in_bar += s * v;
    st.av_in_bar += v;
    if(s != 0) st.nz_in_bar++;
    st.tick_count_in_bar++;
    if(st.tick_count_in_bar >= st.ticks_per_bar)
    {
        OF_Bar bar;
        bar.t_close = t;
        bar.mid = st.mid_last;
        bar.spread = st.spread_sum / st.tick_count_in_bar;
        bar.total_volume = st.av_in_bar;
        bar.signed_volume = st.sv_in_bar;
        bar.ofi = (st.av_in_bar > 0) ? (st.sv_in_bar / st.av_in_bar) : 0.0;
        bar.ofi = MathMax(-1.0, MathMin(1.0, bar.ofi));
        bar.taker = (double)st.nz_in_bar / st.tick_count_in_bar;
        st.bars[st.head] = bar;
        st.head = (st.head + 1) % OF_BUFFER_BARS;
        if(st.n_bars < OF_BUFFER_BARS) st.n_bars++;
        // Reset
        st.tick_count_in_bar = 0;
        st.sv_in_bar = 0; st.av_in_bar = 0; st.nz_in_bar = 0;
        st.spread_sum = 0;
        return true;
    }
    return false;
}

// Bar accessor; offset=0 -> most recent closed bar, offset=1 -> prior, ...
bool OF_Bar_At(const OF_State &st, const int offset, OF_Bar &out)
{
    if(offset < 0 || offset >= st.n_bars) return false;
    int idx = (st.head - 1 - offset + OF_BUFFER_BARS) % OF_BUFFER_BARS;
    out = st.bars[idx];
    return true;
}

double _OF_ema(const OF_State &st, const int field, const int span)
{
    // field: 0=ofi, 1=taker. Backward EMA over the available bars.
    if(st.n_bars == 0) return 0.0;
    double alpha = 2.0 / (span + 1.0);
    double ema = 0.0; bool first = true;
    int count = MathMin(st.n_bars, OF_BUFFER_BARS);
    // Iterate oldest -> newest so EMA accumulates correctly.
    for(int k = count - 1; k >= 0; k--)
    {
        OF_Bar b;
        if(!OF_Bar_At(st, k, b)) continue;
        double v = (field == 0) ? b.ofi : b.taker;
        if(first) { ema = v; first = false; }
        else      { ema = alpha * v + (1.0 - alpha) * ema; }
    }
    return ema;
}

double _OF_std(const double &arr[], int n)
{
    if(n < 2) return 0.0;
    double m = 0; for(int i = 0; i < n; i++) m += arr[i]; m /= n;
    double s = 0; for(int i = 0; i < n; i++) { double d = arr[i] - m; s += d*d; }
    return MathSqrt(s / (n - 1));
}

bool OF_BuildFeatures(const OF_State &st, double &out[])
{
    if(st.n_bars < 60) return false;   // need warmup
    ArrayResize(out, OF_FEATURE_DIM);

    OF_Bar b0; OF_Bar_At(st, 0, b0);

    out[0] = b0.ofi;
    out[1] = _OF_ema(st, 0, 5);
    out[2] = _OF_ema(st, 0, 20);

    // CVD norm — running cumulative signed vol over recent buffer,
    // divided by rolling 1000-bar |delta| sum (cap at buffer length).
    int win = MathMin(st.n_bars, 1000);
    double cum = 0, abssum = 0;
    for(int k = 0; k < win; k++)
    {
        OF_Bar b; if(!OF_Bar_At(st, k, b)) continue;
        cum += b.signed_volume;
        abssum += MathAbs(b.signed_volume);
    }
    double cvd = (abssum > 0) ? cum / abssum : 0.0;
    out[3] = MathMax(-1.0, MathMin(1.0, cvd));

    out[4] = b0.taker;
    out[5] = _OF_ema(st, 1, 20);

    // Returns and vols over last 1, 5, 20 bars in 1e4 units.
    OF_Bar b1, b5, b20;
    bool has1  = OF_Bar_At(st, 1,  b1);
    bool has5  = OF_Bar_At(st, 5,  b5);
    bool has20 = OF_Bar_At(st, 20, b20);
    double ret1  = (has1  && b1.mid  > 0) ? MathLog(b0.mid / b1.mid)  : 0.0;
    double ret5  = (has5  && b5.mid  > 0) ? MathLog(b0.mid / b5.mid)  : 0.0;
    double ret20 = (has20 && b20.mid > 0) ? MathLog(b0.mid / b20.mid) : 0.0;
    out[6] = ret1  * 1e4;
    out[7] = ret5  * 1e4;
    out[8] = ret20 * 1e4;

    // Vols: std of 1-bar log returns over last 5 / 20 closed bars.
    double r5[5], r20[20];
    int n5 = 0, n20 = 0;
    for(int k = 0; k < 20; k++)
    {
        OF_Bar a, c;
        if(!OF_Bar_At(st, k, a) || !OF_Bar_At(st, k + 1, c)) break;
        if(c.mid <= 0) continue;
        double lr = MathLog(a.mid / c.mid);
        if(k < 5)  { r5[n5++] = lr; }
        r20[n20++] = lr;
    }
    out[9]  = _OF_std(r5,  n5)  * 1e4;
    out[10] = _OF_std(r20, n20) * 1e4;

    // Microprice drift
    out[11] = (has5  && b5.mid  > 0) ? (b0.mid - b5.mid)  / b5.mid  * 1e4 : 0.0;
    out[12] = (has20 && b20.mid > 0) ? (b0.mid - b20.mid) / b20.mid * 1e4 : 0.0;

    // Spread now (pip-scaled) + spread regime ratio over 60-bar median.
    out[13] = (b0.mid > 0) ? b0.spread / b0.mid * 1e4 : 0.0;
    int nm = MathMin(60, st.n_bars);
    double spreads[60];
    for(int k = 0; k < nm; k++)
    {
        OF_Bar b; if(!OF_Bar_At(st, k, b)) spreads[k] = 0; else spreads[k] = b.spread;
    }
    // Simple median by partial sort
    for(int i = 0; i < nm; i++)
        for(int j = i + 1; j < nm; j++)
            if(spreads[j] < spreads[i]) { double t = spreads[i]; spreads[i] = spreads[j]; spreads[j] = t; }
    double med = (nm > 0) ? spreads[nm / 2] : 1.0;
    double ratio = (med > 1e-12) ? b0.spread / med : 0.0;
    out[14] = MathMax(0.0, MathMin(5.0, ratio)) / 5.0;

    // Tick intensity z-score: total_volume of current bar vs rolling 200-bar
    // mean/std.
    int nv = MathMin(200, st.n_bars);
    double vsum = 0, vsq = 0;
    for(int k = 0; k < nv; k++)
    {
        OF_Bar b; if(!OF_Bar_At(st, k, b)) continue;
        vsum += b.total_volume;
        vsq  += b.total_volume * b.total_volume;
    }
    double vmean = (nv > 0) ? vsum / nv : 0.0;
    double vvar  = (nv > 1) ? vsq / nv - vmean * vmean : 0.0;
    double vstd  = (vvar > 1e-12) ? MathSqrt(vvar) : 1.0;
    double z = (b0.total_volume - vmean) / vstd;
    out[15] = MathMax(-1.0, MathMin(1.0, z / 5.0));
    return true;
}

#endif // ORDER_FLOW_RULE_MQH
