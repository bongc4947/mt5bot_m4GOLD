#ifndef BROKERCONFIG_MQH
#define BROKERCONFIG_MQH

#include "Defines.mqh"

//=============================================================================
// CBrokerConfig — loads BrokerConfig_[Broker].ini and provides per-symbol
// accessors. Zero file I/O after Load(). All data in fixed arrays.
//=============================================================================

SSymbolSpec g_sym_specs[MAX_SYMBOLS];
int         g_sym_count  = 0;
string      g_broker_name_loaded = "";
string      g_account_currency   = "";
bool        g_broker_loaded      = false;

class CBrokerConfig
{
private:
   //--- resolve canonical → index (linear; 16 syms is fast enough)
   int Find(const string canonical) const
   {
      for(int i = 0; i < g_sym_count; i++)
         if(g_sym_specs[i].canonical == canonical)
            return i;
      return -1;
   }

   //--- parse "HH:MM-HH:MM" or "CLOSED" into SSession
   void ParseSession(const string val, SSession &s)
   {
      if(val == "CLOSED") { s.closed = true; s.from_min = 0; s.to_min = 0; return; }
      s.closed = false;
      string parts[];
      StringSplit(val, '-', parts);
      if(ArraySize(parts) < 2) { s.closed = true; return; }
      // from
      string fp[]; StringSplit(parts[0], ':', fp);
      s.from_min = (ArraySize(fp) >= 2) ? (int)StringToInteger(fp[0])*60 + (int)StringToInteger(fp[1]) : 0;
      // to
      string tp2[]; StringSplit(parts[1], ':', tp2);
      s.to_min   = (ArraySize(tp2) >= 2) ? (int)StringToInteger(tp2[0])*60 + (int)StringToInteger(tp2[1]) : 1439;
   }

   ENUM_ASSET_CLASS ParseAssetClass(const string v)
   {
      if(v == "FOREX")   return ASSET_FOREX;
      if(v == "METALS")  return ASSET_METALS;
      if(v == "INDICES") return ASSET_INDICES;
      if(v == "CRYPTO")  return ASSET_CRYPTO;
      if(v == "ENERGY")  return ASSET_ENERGY;
      return ASSET_UNKNOWN;
   }

   ENUM_HYDRA_TRADE_MODE ParseTradeMode(const string v)
   {
      if(v == "LONGONLY")  return HTRADE_LONGONLY;
      if(v == "SHORTONLY") return HTRADE_SHORTONLY;
      if(v == "CLOSEONLY") return HTRADE_CLOSEONLY;
      if(v == "DISABLED")  return HTRADE_DISABLED;
      return HTRADE_FULL;
   }

   ENUM_SWAP_KIND ParseSwapType(const string v)
   {
      if(v == "MONEY")    return SWAP_MONEY;
      if(v == "INTEREST") return SWAP_INTEREST;
      if(v == "CURRENCY") return SWAP_CURRENCY;
      return SWAP_POINTS;
   }

   void ApplyKeyValue(int idx, const string key, const string val)
   {
      if(key == "broker_name")   g_sym_specs[idx].broker_name    = val;
      else if(key == "canonical")     g_sym_specs[idx].canonical      = val;
      else if(key == "asset_class")   g_sym_specs[idx].asset_class    = ParseAssetClass(val);
      else if(key == "digits")        g_sym_specs[idx].digits         = (int)StringToInteger(val);
      else if(key == "point")         g_sym_specs[idx].point_val      = StringToDouble(val);
      else if(key == "pip_size")      g_sym_specs[idx].pip_size       = StringToDouble(val);
      else if(key == "tick_size")     g_sym_specs[idx].tick_size      = StringToDouble(val);
      else if(key == "tick_value")    g_sym_specs[idx].tick_value     = StringToDouble(val);
      else if(key == "contract_size") g_sym_specs[idx].contract_size  = StringToDouble(val);
      else if(key == "spread_pips")   g_sym_specs[idx].spread_pips    = StringToDouble(val);
      else if(key == "spread_type")   g_sym_specs[idx].spread_type    = (val=="FIXED") ? SPREAD_FIXED : SPREAD_VARIABLE;
      else if(key == "stops_level")   g_sym_specs[idx].stops_level    = (int)StringToInteger(val);
      else if(key == "freeze_level")  g_sym_specs[idx].freeze_level   = (int)StringToInteger(val);
      else if(key == "volume_min")    g_sym_specs[idx].volume_min     = StringToDouble(val);
      else if(key == "volume_max")    g_sym_specs[idx].volume_max     = StringToDouble(val);
      else if(key == "volume_step")   g_sym_specs[idx].volume_step    = StringToDouble(val);
      else if(key == "trade_mode")    g_sym_specs[idx].trade_mode     = ParseTradeMode(val);
      else if(key == "swap_long")     g_sym_specs[idx].swap_long      = StringToDouble(val);
      else if(key == "swap_short")    g_sym_specs[idx].swap_short     = StringToDouble(val);
      else if(key == "swap_type")     g_sym_specs[idx].swap_type      = ParseSwapType(val);
      else if(key == "session_sun")   ParseSession(val, g_sym_specs[idx].sessions[0]);
      else if(key == "session_mon")   ParseSession(val, g_sym_specs[idx].sessions[1]);
      else if(key == "session_tue")   ParseSession(val, g_sym_specs[idx].sessions[2]);
      else if(key == "session_wed")   ParseSession(val, g_sym_specs[idx].sessions[3]);
      else if(key == "session_thu")   ParseSession(val, g_sym_specs[idx].sessions[4]);
      else if(key == "session_fri")   ParseSession(val, g_sym_specs[idx].sessions[5]);
      else if(key == "session_sat")   ParseSession(val, g_sym_specs[idx].sessions[6]);
   }

public:
   //--- Load INI file. Returns false if file not found (→ CONFIG_MISSING).
   bool Load()
   {
      g_broker_loaded = false;
      g_sym_count     = 0;

      string raw_broker = AccountInfoString(ACCOUNT_COMPANY);
      // sanitize: same rules as BrokerScan
      StringReplace(raw_broker, " ", "-");
      StringReplace(raw_broker, "/", "-");
      StringReplace(raw_broker, "\\", "-");
      StringReplace(raw_broker, ".", "");   // "Ava Trade Ltd." → "Ava-Trade-Ltd"
      g_broker_name_loaded  = raw_broker;
      g_account_currency    = AccountInfoString(ACCOUNT_CURRENCY);

      string fname = HYDRA_FOLDER + CONFIG_PREFIX + raw_broker + CONFIG_EXT;
      int fh = FileOpen(fname, FILE_READ | FILE_COMMON | FILE_TXT | FILE_SHARE_READ | FILE_ANSI);
      if(fh == INVALID_HANDLE)
      {
         PrintFormat("HYDRA HALT: Config not found: %s  Run BrokerScan script.", fname);
         return false;
      }

      int   cur_idx    = -1;
      bool  in_broker  = false;

      while(!FileIsEnding(fh))
      {
         string line = FileReadString(fh);
         StringTrimLeft(line);
         StringTrimRight(line);

         if(StringLen(line) == 0) continue;
         if(StringGetCharacter(line, 0) == ';') continue;  // comment

         // section header
         if(StringGetCharacter(line, 0) == '[')
         {
            string sec = StringSubstr(line, 1, StringLen(line) - 2);
            if(sec == "BROKER") { in_broker = true; cur_idx = -1; continue; }
            in_broker = false;
            if(g_sym_count < MAX_SYMBOLS)
            {
               cur_idx = g_sym_count;
               g_sym_count++;
               // zero-init
               g_sym_specs[cur_idx].canonical    = sec;
               g_sym_specs[cur_idx].broker_name  = sec;
               g_sym_specs[cur_idx].mw_present   = false;
               g_sym_specs[cur_idx].asset_class  = ASSET_UNKNOWN;
               g_sym_specs[cur_idx].trade_mode   = HTRADE_FULL;
               g_sym_specs[cur_idx].spread_type  = SPREAD_VARIABLE;
               g_sym_specs[cur_idx].swap_type    = SWAP_POINTS;
               g_sym_specs[cur_idx].digits       = 5;
               g_sym_specs[cur_idx].point_val    = 0.00001;
               g_sym_specs[cur_idx].pip_size     = 0.0001;
               g_sym_specs[cur_idx].volume_min   = 0.01;
               g_sym_specs[cur_idx].volume_step  = 0.01;
               g_sym_specs[cur_idx].volume_max   = 100.0;
               for(int d = 0; d < 7; d++) { g_sym_specs[cur_idx].sessions[d].closed = true; }
            }
            continue;
         }

         // key=value
         int eq = StringFind(line, "=");
         if(eq < 1) continue;
         string key = StringSubstr(line, 0, eq);
         string val = StringSubstr(line, eq + 1);
         StringTrimRight(key); StringTrimLeft(val);

         if(in_broker)
         {
            if(key == "account_currency") g_account_currency = val;
            continue;
         }
         if(cur_idx >= 0) ApplyKeyValue(cur_idx, key, val);
      }
      FileClose(fh);

      // Validate symbols against Market Watch
      for(int i = 0; i < g_sym_count; i++)
      {
         bool is_custom = false;
         bool present = SymbolExist(g_sym_specs[i].broker_name, is_custom);
         g_sym_specs[i].mw_present = present;
         if(!present)
            PrintFormat("HYDRA WARNING: [%s] broker_name '%s' not in Market Watch (config may be stale)",
                        g_sym_specs[i].canonical, g_sym_specs[i].broker_name);
      }

      g_broker_loaded = true;
      PrintFormat("HYDRA: BrokerConfig loaded. Broker=%s  Symbols=%d", raw_broker, g_sym_count);
      return true;
   }

   bool IsLoaded() const { return g_broker_loaded; }
   int  SymCount()  const { return g_sym_count; }
   string GetBrokerName() const { return g_broker_name_loaded; }

   //--- Per-symbol accessors (use canonical name)
   string BrokerName(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].broker_name : canonical;
   }

   double PipSize(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].pip_size : _Point * 10.0;
   }

   double PointVal(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].point_val : _Point;
   }

   double TickValue(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].tick_value : 1.0;
   }

   double ContractSize(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].contract_size : 100000.0;
   }

   int MinStopPoints(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].stops_level : 10;
   }

   int FreezeLevel(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].freeze_level : 0;
   }

   double VolumeMin(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].volume_min : 0.01;
   }

   double VolumeMax(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].volume_max : 100.0;
   }

   double VolumeStep(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].volume_step : 0.01;
   }

   double SpreadPips(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].spread_pips : 1.5;
   }

   ENUM_SPREAD_KIND SpreadType(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].spread_type : SPREAD_VARIABLE;
   }

   ENUM_HYDRA_TRADE_MODE TradeMode(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].trade_mode : HTRADE_FULL;
   }

   ENUM_ASSET_CLASS AssetClass(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].asset_class : ASSET_UNKNOWN;
   }

   bool IsMWPresent(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].mw_present : false;
   }

   int Digits(const string canonical) const
   {
      int i = Find(canonical);
      return (i >= 0) ? g_sym_specs[i].digits : 5;
   }

   //--- Market hours check — uses broker's live trade mode, NOT static INI sessions.
   // Static session times captured by BrokerScan can be stale (e.g. broker opens
   // earlier than recorded). SYMBOL_TRADE_MODE reflects the broker's real-time state.
   bool IsMarketOpen(const string canonical, const datetime t) const
   {
      int i = Find(canonical);
      string bname = (i >= 0) ? g_sym_specs[i].broker_name : canonical;

      // Live check: broker sets SYMBOL_TRADE_MODE_DISABLED when market is closed.
      long trade_mode = SymbolInfoInteger(bname, SYMBOL_TRADE_MODE);
      if(trade_mode == SYMBOL_TRADE_MODE_DISABLED ||
         trade_mode == SYMBOL_TRADE_MODE_CLOSEONLY)
         return false;

      // Sanity: bid must be non-zero (symbol streaming)
      if(SymbolInfoDouble(bname, SYMBOL_BID) <= 0.0)
         return false;

      return true;
   }

   //--- SL/TP distance helpers (in price)
   double CalcSLDist(const string canonical) const
   {
      double pt    = PointVal(canonical);
      double pip   = PipSize(canonical);
      double spd   = SpreadPips(canonical);
      double mind  = (MinStopPoints(canonical) + 5) * pt;
      return MathMax(mind, 2.0 * spd * pip);
   }

   double CalcTPDist(const string canonical) const
   {
      double pt    = PointVal(canonical);
      double pip   = PipSize(canonical);
      double spd   = SpreadPips(canonical);
      double mind  = (MinStopPoints(canonical) + 5) * pt;
      return MathMax(mind * 1.5, 4.0 * spd * pip);
   }

   //--- Normalize lot to broker constraints
   double NormalizeLot(const string canonical, double raw_lot) const
   {
      double step = VolumeStep(canonical);
      double vmin = VolumeMin(canonical);
      double vmax = VolumeMax(canonical);
      if(step <= 0.0) step = 0.01;
      double lot = MathFloor(raw_lot / step) * step;
      return MathMax(vmin, MathMin(vmax, lot));
   }

   //--- Iterate all loaded symbols
   string GetCanonical(int idx) const
   {
      if(idx < 0 || idx >= g_sym_count) return "";
      return g_sym_specs[idx].canonical;
   }
};

// Global singleton
CBrokerConfig g_broker;

#endif // BROKERCONFIG_MQH
