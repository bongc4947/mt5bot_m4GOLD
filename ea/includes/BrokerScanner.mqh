#ifndef BROKERSCANNER_MQH
#define BROKERSCANNER_MQH

//=============================================================================
// BrokerScanner.mqh — Shared broker config generator for HYDRA mk4.
//
// MANDATORY FIRST STEP:
//   Place Market_Watch.csv in MT5 Common Files before running.
//   Export from MT5: right-click Market Watch panel → "Export" → save as
//   "Market_Watch.csv" → copy to MT5 Common Files directory.
//
// CSV format (semicolon-delimited, as exported by MT5):
//   Symbol;Bid;Ask;Spread;Tick Size;Tick Value;Pip Size;Pip Value;Contract Size
//
// The CSV is the primary source for:
//   broker symbol names, pip_size, tick_size, tick_value, contract_size, spread
//
// Live SymbolInfo calls fill the remainder:
//   digits, stops_level, freeze_level, volume limits, session hours, swap
//
// Shared by: MT5_Bot_mk4_BrokerScan.mq5 (script) and
//            MT5_Bot_mk4_HYDRA.mq5 (EA auto-scan on missing config)
//=============================================================================

//--- Expected CSV location in MT5 Common Files
#define MWC_FILENAME     "Market_Watch.csv"
#define MWC_FILENAME_ALT "Market_Watch"      // prefix for auto-detect

#define SCAN_CONFIG_PREFIX  "BrokerConfig_"
#define SCAN_CONFIG_EXT     ".ini"
#define SCAN_MAX_SYMS       512

//--- Parsed row from Market Watch CSV
struct SMWRow
{
   string symbol;
   double bid;
   double ask;
   double spread_pts;   // in points (raw column)
   double tick_size;
   double tick_value;
   double pip_size;
   double pip_value;
   double contract_size;
   bool   valid;
};

//--- In-memory CSV table
SMWRow  g_mw_rows[SCAN_MAX_SYMS];
int     g_mw_count = 0;

//--- Alias table (canonical → pipe-separated broker variants)
// mk4.8: N_ALIASES dropped 16 -> 15 (NAS100 removed — broker doesn't quote
// it, no tick parquet extracted). Index numbers shifted: UK_100 .. NATURAL_GAS
// all decrement by 1 below.
#define N_ALIASES 15
string g_scan_canonical[N_ALIASES];
string g_scan_aliases[N_ALIASES];

string g_scan_forex_list  = "EURUSD|GBPUSD|USDJPY|USDCHF|AUDUSD|NZDUSD|USDCAD|EURGBP|EURJPY|GBPJPY";
string g_scan_metals_list = "GOLD|SILVER|PLATINUM|COPPER";
string g_scan_index_list  = "US_500|UK_100|GER30|GER40|FRA40|JPN225|AUS200|HK50";
string g_scan_crypto_list = "BTCUSD|ETHUSD|LTCUSD|XRPUSD|BNBUSD|ADAUSD|DOGEUSD";
string g_scan_energy_list = "CRUDEOIL|BRENT_OIL|NATURAL_GAS";

string g_day_keys[7] = {"session_sun","session_mon","session_tue","session_wed","session_thu","session_fri","session_sat"};

//=============================================================================
// CSV parser
//=============================================================================

//--- Find and open Market_Watch.csv (or newest "Market Watch*.csv") in Common Files.
//    Returns file handle or INVALID_HANDLE.
int OpenMarketWatchCSV(string &found_name)
{
   // 1. Try exact name
   if(FileIsExist(MWC_FILENAME, FILE_COMMON))
   {
      int fh = FileOpen(MWC_FILENAME,
                        FILE_READ | FILE_CSV | FILE_COMMON | FILE_ANSI,
                        ';');
      if(fh != INVALID_HANDLE) { found_name = MWC_FILENAME; return fh; }
   }

   // 2. Search for "Market Watch*.csv" pattern using FileFindFirst/Next
   string search_name = "";
   long   search_h    = FileFindFirst("Market Watch*.csv", search_name, FILE_COMMON);
   if(search_h != INVALID_HANDLE)
   {
      // Pick first match (newest if there are multiple — MT5 lists alphabetically)
      string best = search_name;
      string next_name = "";
      while(FileFindNext(search_h, next_name))
         if(next_name > best) best = next_name;
      FileFindClose(search_h);

      int fh = FileOpen(best, FILE_READ | FILE_CSV | FILE_COMMON | FILE_ANSI, ';');
      if(fh != INVALID_HANDLE) { found_name = best; return fh; }
   }

   found_name = "";
   return INVALID_HANDLE;
}

//--- Load and parse Market_Watch.csv into g_mw_rows[].
//    Returns number of rows parsed, or -1 if file not found.
int LoadMarketWatchCSV()
{
   g_mw_count = 0;

   string found_name = "";
   int fh = OpenMarketWatchCSV(found_name);

   if(fh == INVALID_HANDLE)
   {
      Print("[BrokerScanner] ERROR: Market_Watch.csv not found in Common Files.");
      Print("[BrokerScanner] *** MANDATORY STEP: ***");
      Print("[BrokerScanner]   1. In MT5: right-click Market Watch panel");
      Print("[BrokerScanner]   2. Select 'Symbols' or click the gear icon");
      Print("[BrokerScanner]   3. Right-click in the symbol list → Export");
      Print("[BrokerScanner]   4. Save the file");
      Print("[BrokerScanner]   5. Rename it to 'Market_Watch.csv'");
      Print("[BrokerScanner]   6. Copy it to: MT5 Common Files directory");
      Print("[BrokerScanner]      (Help → Open Data Folder → MQL5/../Common/Files)");
      Print("[BrokerScanner]   7. Re-run BrokerScan or restart the EA.");
      return -1;
   }

   PrintFormat("[BrokerScanner] Loaded Market Watch file: %s", found_name);

   // Skip header row
   bool first = true;
   while(!FileIsEnding(fh) && g_mw_count < SCAN_MAX_SYMS)
   {
      // Read one CSV row — FileReadString reads one cell when file opened with separator
      string sym = FileReadString(fh); if(FileIsEnding(fh) && sym == "") break;
      string bid_s  = FileReadString(fh);
      string ask_s  = FileReadString(fh);
      string spd_s  = FileReadString(fh);
      string tksz_s = FileReadString(fh);
      string tkvl_s = FileReadString(fh);
      string ppsz_s = FileReadString(fh);
      string ppvl_s = FileReadString(fh);
      string ctrsz_s= FileReadString(fh);

      StringTrimLeft(sym); StringTrimRight(sym);

      // Skip header
      if(first) { first = false; if(sym == "Symbol") continue; }
      if(StringLen(sym) == 0) continue;

      SMWRow row;
      row.symbol        = sym;
      row.bid           = StringToDouble(bid_s);
      row.ask           = StringToDouble(ask_s);
      row.spread_pts    = StringToDouble(spd_s);
      row.tick_size     = StringToDouble(tksz_s);
      row.tick_value    = StringToDouble(tkvl_s);
      row.pip_size      = StringToDouble(ppsz_s);
      row.pip_value     = StringToDouble(ppvl_s);
      row.contract_size = StringToDouble(ctrsz_s);
      row.valid         = (row.pip_size > 0.0 && row.contract_size > 0.0);

      g_mw_rows[g_mw_count] = row;
      g_mw_count++;
   }

   FileClose(fh);
   PrintFormat("[BrokerScanner] Parsed %d symbols from Market Watch CSV", g_mw_count);
   return g_mw_count;
}

//--- Find a CSV row by broker symbol name. Returns index or -1.
int MWFindByBrokerName(const string broker_sym)
{
   for(int i = 0; i < g_mw_count; i++)
      if(g_mw_rows[i].symbol == broker_sym) return i;
   return -1;
}

//--- Find a CSV row that matches canonical (via alias list + fuzzy).
//    Also returns the matched broker symbol name.
int MWFindCanonical(const string canonical, string &out_broker_sym)
{
   // Build alias row for this canonical
   string alias_row = "";
   for(int a = 0; a < N_ALIASES; a++)
      if(g_scan_canonical[a] == canonical) { alias_row = g_scan_aliases[a]; break; }

   // Try all alias variants against CSV
   string variants[];
   int n_v = 0;
   if(alias_row != "")
      n_v = StringSplit(alias_row, '|', variants);
   if(n_v == 0) { ArrayResize(variants, 1); variants[0] = canonical; n_v = 1; }

   for(int v = 0; v < n_v; v++)
   {
      int idx = MWFindByBrokerName(variants[v]);
      if(idx >= 0) { out_broker_sym = g_mw_rows[idx].symbol; return idx; }
   }

   // Fuzzy: strip trailing non-alpha from CSV symbols, match to canonical
   string cu = canonical; StringToUpper(cu);
   for(int i = 0; i < g_mw_count; i++)
   {
      string bsym = g_mw_rows[i].symbol;
      int len = StringLen(bsym);
      while(len > 0)
      {
         ushort ch = StringGetCharacter(bsym, len - 1);
         if((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) break;
         len--;
      }
      string stripped = StringSubstr(bsym, 0, len);
      string su = stripped; StringToUpper(su);
      if(su == cu) { out_broker_sym = bsym; return i; }
   }

   out_broker_sym = "";
   return -1;
}

//=============================================================================
// Alias table
//=============================================================================

void BuildScanAliasTable()
{
   g_scan_canonical[0]  = "EURUSD";  g_scan_aliases[0]  = "EURUSD|EURUSDm|EURUSD.|EURUSD_i|EURUSD+|EURUSD_";
   g_scan_canonical[1]  = "GBPUSD";  g_scan_aliases[1]  = "GBPUSD|GBPUSDm|GBPUSD.|GBPUSD_i";
   g_scan_canonical[2]  = "USDJPY";  g_scan_aliases[2]  = "USDJPY|USDJPYm|USDJPY.|USDJPY_i";
   g_scan_canonical[3]  = "GOLD";      g_scan_aliases[3]  = "GOLD|XAUUSD|XAUUSDm|XAUUSD.|GOLDm|GOLD.";
   g_scan_canonical[4]  = "SILVER";   g_scan_aliases[4]  = "SILVER|XAGUSD|XAGUSDm|XAGUSD.|SILVERm";
   g_scan_canonical[5]  = "PLATINUM"; g_scan_aliases[5]  = "PLATINUM|XPTUSD|XPTUSDm|XPTUSD.";
   g_scan_canonical[6]  = "COPPER";   g_scan_aliases[6]  = "COPPER|XCUUSD|XCUUSDm|XCUUSD.";
   g_scan_canonical[7]  = "US_500";   g_scan_aliases[7]  = "US_500|US500|SPX500|SP500|USA500|US.500|US500m|SPXUSD";
   g_scan_canonical[8]  = "UK_100";   g_scan_aliases[8]  = "UK_100|UK100|FTSE100|UK100m|FTSE|GBXUSD";
   g_scan_canonical[9]  = "BTCUSD";   g_scan_aliases[9]  = "BTCUSD|BITCOIN|BTC|BTCUSDm|BTCUSDT|BTC/USD";
   g_scan_canonical[10] = "ETHUSD";   g_scan_aliases[10] = "ETHUSD|ETHEREUM|ETH|ETHUSDm|ETH/USD";
   g_scan_canonical[11] = "LTCUSD";   g_scan_aliases[11] = "LTCUSD|LITECOIN|LTC|LTCUSDm|LTC/USD";
   g_scan_canonical[12] = "CrudeOIL";    g_scan_aliases[12] = "CrudeOIL|USOIL|WTI|OIL|CRUDE|USOILSPOT|OILm|WTIUSD";
   g_scan_canonical[13] = "BRENT_OIL";   g_scan_aliases[13] = "BRENT_OIL|UKOIL|BRENT|BRENTOIL|UKOILm|BRTUSD";
   g_scan_canonical[14] = "NATURAL_GAS"; g_scan_aliases[14] = "NATURAL_GAS|NGAS|NATURALGAS|NGASm|NATGAS|GASUSD";
}

//=============================================================================
// Asset class detection
//=============================================================================

string DetectAssetClass(const string canonical)
{
   string cu = canonical; StringToUpper(cu);
   if(StringFind(g_scan_forex_list,  cu) >= 0) return "FOREX";
   if(StringFind(g_scan_metals_list, cu) >= 0) return "METALS";
   if(StringFind(g_scan_index_list,  cu) >= 0) return "INDICES";
   if(StringFind(g_scan_crypto_list, cu) >= 0) return "CRYPTO";
   if(StringFind(g_scan_energy_list, cu) >= 0) return "ENERGY";
   return "FOREX";
}

//=============================================================================
// Session formatter
//=============================================================================

string FormatSession(const string broker_sym, const int day)
{
   datetime from = 0, to = 0;
   if(!SymbolInfoSessionTrade(broker_sym, (ENUM_DAY_OF_WEEK)day, 0, from, to))
      return "CLOSED";
   if(from == 0 && to == 0) return "CLOSED";
   int fh2 = (int)(from / 3600), fm = (int)((from % 3600) / 60);
   int th  = (int)(to   / 3600), tm = (int)((to   % 3600) / 60);
   if(th >= 24) { th = 23; tm = 59; }
   return StringFormat("%02d:%02d-%02d:%02d", fh2, fm, th, tm);
}

//=============================================================================
// WriteSymbol — write one [CANONICAL] section to the config file.
// CSV-derived values take priority over live SymbolInfo for pip/tick/spread.
//=============================================================================

void WriteSymbolScan(int file_h, const string canonical,
                     const string broker_sym, const int mw_idx)
{
   SymbolSelect(broker_sym, true);
   Sleep(150);

   //--- Live SymbolInfo values
   int    digits        = (int)SymbolInfoInteger(broker_sym, SYMBOL_DIGITS);
   double pt            = SymbolInfoDouble(broker_sym, SYMBOL_POINT);
   int    spread_pts_live = (int)SymbolInfoInteger(broker_sym, SYMBOL_SPREAD);
   bool   spread_float  = (bool)SymbolInfoInteger(broker_sym, SYMBOL_SPREAD_FLOAT);
   int    stops_level   = (int)SymbolInfoInteger(broker_sym, SYMBOL_TRADE_STOPS_LEVEL);
   int    freeze_level  = (int)SymbolInfoInteger(broker_sym, SYMBOL_TRADE_FREEZE_LEVEL);
   double vol_min       = SymbolInfoDouble(broker_sym, SYMBOL_VOLUME_MIN);
   double vol_max       = SymbolInfoDouble(broker_sym, SYMBOL_VOLUME_MAX);
   double vol_step      = SymbolInfoDouble(broker_sym, SYMBOL_VOLUME_STEP);
   double swap_long     = SymbolInfoDouble(broker_sym, SYMBOL_SWAP_LONG);
   double swap_short    = SymbolInfoDouble(broker_sym, SYMBOL_SWAP_SHORT);
   int    swap_mode_raw = (int)SymbolInfoInteger(broker_sym, SYMBOL_SWAP_MODE);
   int    trade_mode_raw= (int)SymbolInfoInteger(broker_sym, SYMBOL_TRADE_MODE);

   //--- CSV-derived values (take priority — broker-verified pip/tick data)
   double pip_size, tick_size, tick_value, contract_size, spread_pips;

   if(mw_idx >= 0 && g_mw_rows[mw_idx].valid)
   {
      // CSV is the authoritative source for these values
      pip_size      = g_mw_rows[mw_idx].pip_size;
      tick_size     = g_mw_rows[mw_idx].tick_size;
      tick_value    = g_mw_rows[mw_idx].tick_value;
      contract_size = g_mw_rows[mw_idx].contract_size;
      // Spread: CSV gives it in points; convert to pips using CSV pip_size
      double spd_pts = g_mw_rows[mw_idx].spread_pts;
      spread_pips = (pip_size > 0.0 && pt > 0.0)
                    ? NormalizeDouble(spd_pts * pt / pip_size, 2)
                    : NormalizeDouble(spread_pts_live * pt / (pip_size > 0 ? pip_size : pt), 2);
      PrintFormat("[BrokerScanner]   CSV data: pip=%.6f  tick=%.6f  tv=%.5f  ctr=%.1f  spd=%.2f pips",
                  pip_size, tick_size, tick_value, contract_size, spread_pips);
   }
   else
   {
      // Fallback: compute from live SymbolInfo
      pip_size      = (digits == 5 || digits == 3) ? pt * 10.0 : pt;
      tick_size     = SymbolInfoDouble(broker_sym, SYMBOL_TRADE_TICK_SIZE);
      tick_value    = SymbolInfoDouble(broker_sym, SYMBOL_TRADE_TICK_VALUE);
      contract_size = SymbolInfoDouble(broker_sym, SYMBOL_TRADE_CONTRACT_SIZE);
      spread_pips   = NormalizeDouble(spread_pts_live * pt / (pip_size > 0.0 ? pip_size : pt), 2);
      PrintFormat("[BrokerScanner]   Fallback (no CSV row): pip=%.6f  spd=%.2f pips",
                  pip_size, spread_pips);
   }

   //--- Swap type string
   string swap_type;
   switch(swap_mode_raw)
   {
      case SYMBOL_SWAP_MODE_POINTS:            swap_type = "POINTS";   break;
      case SYMBOL_SWAP_MODE_CURRENCY_SYMBOL:
      case SYMBOL_SWAP_MODE_CURRENCY_MARGIN:
      case SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT:  swap_type = "MONEY";    break;
      case SYMBOL_SWAP_MODE_INTEREST_CURRENT:
      case SYMBOL_SWAP_MODE_INTEREST_OPEN:     swap_type = "INTEREST"; break;
      default:                                 swap_type = "POINTS";   break;
   }

   //--- Trade mode string
   string trade_mode_str;
   switch(trade_mode_raw)
   {
      case SYMBOL_TRADE_MODE_LONGONLY:  trade_mode_str = "LONGONLY";  break;
      case SYMBOL_TRADE_MODE_SHORTONLY: trade_mode_str = "SHORTONLY"; break;
      case SYMBOL_TRADE_MODE_CLOSEONLY: trade_mode_str = "CLOSEONLY"; break;
      case SYMBOL_TRADE_MODE_DISABLED:  trade_mode_str = "DISABLED";  break;
      default:                          trade_mode_str = "FULL";      break;
   }

   //--- Write section
   FileWriteString(file_h, "\n[" + canonical + "]\n");
   FileWriteString(file_h, "broker_name="    + broker_sym                            + "\n");
   FileWriteString(file_h, "canonical="      + canonical                             + "\n");
   FileWriteString(file_h, "asset_class="    + DetectAssetClass(canonical)           + "\n");
   FileWriteString(file_h, "digits="         + IntegerToString(digits)               + "\n");
   FileWriteString(file_h, "point="          + DoubleToString(pt, 10)                + "\n");
   FileWriteString(file_h, "pip_size="       + DoubleToString(pip_size, 10)          + "\n");
   FileWriteString(file_h, "tick_size="      + DoubleToString(tick_size, 10)         + "\n");
   FileWriteString(file_h, "tick_value="     + DoubleToString(tick_value, 5)         + "\n");
   FileWriteString(file_h, "contract_size="  + DoubleToString(contract_size, 2)      + "\n");
   FileWriteString(file_h, "spread_points="  + IntegerToString(spread_pts_live)      + "\n");
   FileWriteString(file_h, "spread_pips="    + DoubleToString(spread_pips, 2)        + "\n");
   FileWriteString(file_h, "spread_type="    + (spread_float ? "VARIABLE" : "FIXED") + "\n");
   FileWriteString(file_h, "stops_level="    + IntegerToString(stops_level)          + "\n");
   FileWriteString(file_h, "freeze_level="   + IntegerToString(freeze_level)         + "\n");
   FileWriteString(file_h, "volume_min="     + DoubleToString(vol_min, 4)            + "\n");
   FileWriteString(file_h, "volume_max="     + DoubleToString(vol_max, 2)            + "\n");
   FileWriteString(file_h, "volume_step="    + DoubleToString(vol_step, 4)           + "\n");
   FileWriteString(file_h, "trade_mode="     + trade_mode_str                        + "\n");
   FileWriteString(file_h, "swap_long="      + DoubleToString(swap_long, 4)          + "\n");
   FileWriteString(file_h, "swap_short="     + DoubleToString(swap_short, 4)         + "\n");
   FileWriteString(file_h, "swap_type="      + swap_type                             + "\n");
   // source tag for debugging
   FileWriteString(file_h, "pip_source="     + (mw_idx >= 0 ? "market_watch_csv" : "live_api") + "\n");

   for(int d = 0; d < 7; d++)
      FileWriteString(file_h, g_day_keys[d] + "=" + FormatSession(broker_sym, d) + "\n");
}

//=============================================================================
// RunBrokerScan — full scan pipeline.
//
// canon_list:  array of canonical symbol names to configure
// n_canon:     size of canon_list
// require_csv: if true, halt with error when Market_Watch.csv is missing
//              if false, fall through to live-API-only scan (not recommended)
//
// Returns: number of symbols successfully written, or -1 on fatal error.
//=============================================================================

int RunBrokerScan(const string &canon_list[], int n_canon,
                  bool require_csv = true)
{
   BuildScanAliasTable();

   //--- Step 1: Load Market Watch CSV (mandatory)
   int mw_loaded = LoadMarketWatchCSV();
   if(mw_loaded < 0)
   {
      if(require_csv) return -1;
      PrintFormat("[BrokerScanner] WARNING: Continuing without CSV — pip values from live API only.");
   }

   //--- Step 2: Determine output filename
   string broker_raw  = AccountInfoString(ACCOUNT_COMPANY);
   string broker_safe = broker_raw;
   StringReplace(broker_safe, " ",  "-");
   StringReplace(broker_safe, "/",  "-");
   StringReplace(broker_safe, "\\", "-");
   StringReplace(broker_safe, ".",  "");

   string account_cur = AccountInfoString(ACCOUNT_CURRENCY);
   string fname = SCAN_CONFIG_PREFIX + broker_safe + SCAN_CONFIG_EXT;

   PrintFormat("[BrokerScanner] Broker : %s", broker_raw);
   PrintFormat("[BrokerScanner] Account: %s  Currency: %s", AccountInfoString(ACCOUNT_NAME), account_cur);
   PrintFormat("[BrokerScanner] Output : Common Files/%s", fname);

   //--- Step 3: Open output file
   int fh = FileOpen(fname, FILE_WRITE | FILE_COMMON | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE)
   {
      PrintFormat("[BrokerScanner] ERROR: Cannot create %s  (error %d)", fname, GetLastError());
      return -1;
   }

   string now_str = TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS);
   FileWriteString(fh, "; HYDRA mk4 Broker Configuration\n");
   FileWriteString(fh, "; Generated : " + now_str + "\n");
   FileWriteString(fh, "; Broker    : " + broker_raw + "\n");
   FileWriteString(fh, "; Currency  : " + account_cur + "\n");
   FileWriteString(fh, "; CSV source: " + (mw_loaded >= 0 ? "Market_Watch.csv" : "live_api_only") + "\n");
   FileWriteString(fh, "; CSV rows  : " + IntegerToString(g_mw_count) + "\n");
   FileWriteString(fh, "; Symbols   : " + IntegerToString(n_canon) + " requested\n\n");

   FileWriteString(fh, "[BROKER]\n");
   FileWriteString(fh, "name="             + broker_safe   + "\n");
   FileWriteString(fh, "account_currency=" + account_cur   + "\n");
   FileWriteString(fh, "scan_date="        + TimeToString(TimeCurrent(), TIME_DATE)    + "\n");
   FileWriteString(fh, "scan_time="        + TimeToString(TimeCurrent(), TIME_SECONDS) + "\n");
   FileWriteString(fh, "csv_rows="         + IntegerToString(g_mw_count)               + "\n");

   //--- Step 4: Process each requested canonical symbol
   int n_ok = 0, n_fail = 0, n_csv_hit = 0;
   for(int i = 0; i < n_canon; i++)
   {
      string canonical = canon_list[i];
      if(StringLen(canonical) == 0) continue;

      //--- Find broker name via CSV first, then live alias resolution
      string broker_sym = "";
      int mw_idx = MWFindCanonical(canonical, broker_sym);

      if(mw_idx >= 0)
      {
         // CSV match found — verify symbol exists in live terminal
         bool is_custom = false;
         if(!SymbolExist(broker_sym, is_custom))
         {
            // Symbol in CSV but not in terminal — try live alias resolution as fallback
            PrintFormat("[BrokerScanner]   %s: CSV says '%s' but not in terminal — trying live lookup",
                        canonical, broker_sym);
            broker_sym = "";
            mw_idx = -1;
         }
         else
         {
            n_csv_hit++;
            PrintFormat("[BrokerScanner] %s -> '%s' (CSV match)", canonical, broker_sym);
         }
      }

      // If CSV didn't resolve, try live alias lookup
      if(broker_sym == "")
      {
         // Try alias table
         string alias_row = "";
         for(int a = 0; a < N_ALIASES; a++)
            if(g_scan_canonical[a] == canonical) { alias_row = g_scan_aliases[a]; break; }

         string variants[];
         int n_v = (alias_row != "") ? StringSplit(alias_row, '|', variants) : 0;
         if(n_v == 0) { ArrayResize(variants, 1); variants[0] = canonical; n_v = 1; }

         bool is_custom = false;
         for(int v = 0; v < n_v; v++)
         {
            is_custom = false;
            if(SymbolExist(variants[v], is_custom)) { broker_sym = variants[v]; break; }
         }

         // Fuzzy fallback
         if(broker_sym == "")
         {
            string cu = canonical; StringToUpper(cu);
            int total = SymbolsTotal(false);
            for(int s = 0; s < total && broker_sym == ""; s++)
            {
               string bsym = SymbolName(s, false);
               int len = StringLen(bsym);
               while(len > 0)
               {
                  ushort ch = StringGetCharacter(bsym, len - 1);
                  if((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) break;
                  len--;
               }
               string stripped = StringSubstr(bsym, 0, len);
               string su = stripped; StringToUpper(su);
               if(su == cu) broker_sym = bsym;
            }
         }

         if(broker_sym == "")
         {
            PrintFormat("[BrokerScanner] WARNING: %s -> NOT FOUND in CSV or terminal (skipped)", canonical);
            n_fail++;
            continue;
         }
         PrintFormat("[BrokerScanner] %s -> '%s' (live lookup — no CSV match)", canonical, broker_sym);
      }

      WriteSymbolScan(fh, canonical, broker_sym, mw_idx);
      n_ok++;
   }

   FileClose(fh);

   PrintFormat("[BrokerScanner] ========================================");
   PrintFormat("[BrokerScanner] OK: %d   CSV-sourced: %d   Skipped: %d",
               n_ok, n_csv_hit, n_fail);
   PrintFormat("[BrokerScanner] Config written: Common Files/%s", fname);
   if(n_fail > 0)
      PrintFormat("[BrokerScanner] %d symbol(s) not found — add them to Market Watch in MT5 and rescan.", n_fail);
   PrintFormat("[BrokerScanner] DONE.");

   return n_ok;
}

//--- Convenience: check if a BrokerConfig file exists for the current broker
bool BrokerConfigExists()
{
   string broker_raw  = AccountInfoString(ACCOUNT_COMPANY);
   string broker_safe = broker_raw;
   StringReplace(broker_safe, " ",  "-");
   StringReplace(broker_safe, "/",  "-");
   StringReplace(broker_safe, "\\", "-");
   StringReplace(broker_safe, ".",  "");
   string fname = SCAN_CONFIG_PREFIX + broker_safe + SCAN_CONFIG_EXT;
   return FileIsExist(fname, FILE_COMMON);
}

//--- Check if Market_Watch.csv (or variant) exists in Common Files
bool MarketWatchCSVExists()
{
   if(FileIsExist(MWC_FILENAME, FILE_COMMON)) return true;
   string dummy = "";
   long h = FileFindFirst("Market Watch*.csv", dummy, FILE_COMMON);
   if(h != INVALID_HANDLE) { FileFindClose(h); return true; }
   return false;
}

#endif // BROKERSCANNER_MQH
