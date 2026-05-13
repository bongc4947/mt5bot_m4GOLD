#ifndef POSITIONSIZER_MQH
#define POSITIONSIZER_MQH

#include "Defines.mqh"
#include "BrokerConfig.mqh"
#include "RiskManager.mqh"

//=============================================================================
// Calculates lot size from confidence × drawdown × meta weight.
// All constraints from BrokerConfig.
//=============================================================================

class CPositionSizer
{
public:
   //--- Calculate lot
   //    confidence: 0..1 from agent
   //    meta_weight: 0..1 capital share for this agent
   //    lot_scale: extra multiplier (from TD3/GA chromosome)
   double CalcLot(const string canonical, double confidence,
                  double meta_weight, double lot_scale = 1.0)
   {
      double dd   = g_risk.DailyDDPct();
      double dd_f = MathMax(0.0, 1.0 - dd / DAILY_DD_PAUSE); // fade out as DD approaches 5%
      double raw  = g_broker.VolumeMin(canonical)
                    * confidence
                    * meta_weight * 4.0      // scale so weight 0.25 → 1× vmin
                    * dd_f
                    * lot_scale;

      return g_broker.NormalizeLot(canonical, raw);
   }

   //--- Fixed-fractional lot from explicit SL distance (mk4 primary method)
   //    sl_price_dist: SL distance in price units (from exec model × pip_size)
   //    risk_pct: fraction of balance to risk (e.g. 0.01 = 1%)
   double CalcLotFromSL(const string canonical, double sl_price_dist, double risk_pct = 0.01)
   {
      if(sl_price_dist <= 0.0) return g_broker.VolumeMin(canonical);
      double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
      double risk_money = balance * risk_pct;
      // Fade out near daily DD limit
      double dd_f = MathMax(0.0, 1.0 - g_risk.DailyDDPct() / DAILY_DD_PAUSE);
      risk_money *= dd_f;
      double tick_val  = g_broker.TickValue(canonical);
      double tick_size = g_broker.PipSize(canonical) / 10.0;
      if(tick_val <= 0.0 || tick_size <= 0.0) return g_broker.VolumeMin(canonical);
      double ticks_in_sl = sl_price_dist / tick_size;
      double raw = (ticks_in_sl > 0.0) ? risk_money / (ticks_in_sl * tick_val) : 0.0;
      return g_broker.NormalizeLot(canonical, raw);
   }

   //--- DRM-aware lot calculation (Phase 4)
   //    Caller passes g_drm_loader.MaxRisk(canonical) as risk_pct.
   //    Defaults to MAX_RISK_PER_TRADE when DRM is not yet loaded.
   double CalcLotDRM(const string canonical, double sl_price_dist,
                     double risk_pct = MAX_RISK_PER_TRADE)
   {
      return CalcLotFromSL(canonical, sl_price_dist, risk_pct);
   }

   //--- Spread guard: true = spread is acceptable
   bool SpreadOK(const string canonical)
   {
      string bname   = g_broker.BrokerName(canonical);
      double pt      = g_broker.PointVal(canonical);
      double pip     = g_broker.PipSize(canonical);
      int    live_pts= (int)SymbolInfoInteger(bname, SYMBOL_SPREAD);
      double live_pip= (pip > 0.0) ? live_pts * pt / pip : live_pts * 0.1;
      double cfg_spd = g_broker.SpreadPips(canonical);
      return (live_pip <= cfg_spd * 2.5);
   }
};

CPositionSizer g_sizer;

#endif // POSITIONSIZER_MQH
