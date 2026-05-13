#ifndef SYMBOLFILTER_MQH
#define SYMBOLFILTER_MQH

#include "Defines.mqh"
#include "BrokerConfig.mqh"

//=============================================================================
// Builds the active symbol list from BrokerConfig.
// Maps canonical names to agent IDs and assigns sym_idx in global arrays.
//=============================================================================

// All canonical symbols in the order they appear in g_sym_specs
// sym_idx used everywhere = index in this ordered list
string  g_active_canonical[MAX_SYMBOLS];
int     g_active_count = 0;

// Reverse map: canonical → active index
int     g_canon_to_active[MAX_SYMBOLS];   // index matches g_sym_specs order

class CSymbolFilter
{
private:
   bool _AllowAll(const string allowed_csv) const
   {
      string s = allowed_csv;
      StringTrimLeft(s);
      StringTrimRight(s);
      return (s == "" || s == "*" || s == "ALL" || s == "all");
   }

   bool _IsAllowed(const string canonical, const string allowed_csv) const
   {
      if(_AllowAll(allowed_csv)) return true;

      string items[];
      int n = StringSplit(allowed_csv, ',', items);
      for(int i = 0; i < n; i++)
      {
         string tok = items[i];
         StringTrimLeft(tok);
         StringTrimRight(tok);
         if(tok == canonical)
            return true;
      }
      return false;
   }

public:
   //--- Build active list from BrokerConfig. Returns count.
   int Build(const string allowed_csv = "")
   {
      g_active_count = 0;
      for(int i = 0; i < MAX_SYMBOLS; i++)
         g_canon_to_active[i] = -1;

      for(int i = 0; i < g_broker.SymCount() && i < MAX_SYMBOLS; i++)
      {
         string can = g_broker.GetCanonical(i);
         if(!g_broker.IsMWPresent(can)) continue;
         if(g_broker.TradeMode(can) == HTRADE_DISABLED) continue;
         if(!_IsAllowed(can, allowed_csv)) continue;

         g_active_canonical[g_active_count] = can;
         g_canon_to_active[i]               = g_active_count;
         g_active_count++;
      }
      return g_active_count;
   }

   //--- Get active index for a canonical name (-1 if not active)
   int IndexOf(const string canonical) const
   {
      for(int i = 0; i < g_active_count; i++)
         if(g_active_canonical[i] == canonical) return i;
      return -1;
   }

   //--- Build sym_map arrays for each agent (maps agent node → active sym idx)
   void BuildAgentMaps(int &forex_map[], int &metals_map[],
                       int &indices_map[], int &ce_map[])
   {
      string forex_c[]   = {"EURUSD","GBPUSD","USDJPY"};
      string metals_c[]  = {"GOLD","SILVER","PLATINUM","COPPER"};
      // mk4.8: NAS100 dropped (no broker quote / tick parquet)
      string indices_c[] = {"US_500","UK_100"};
      string ce_c[]      = {"BTCUSD","ETHUSD","LTCUSD","CrudeOIL","BRENT_OIL","NATURAL_GAS"};

      ArrayResize(forex_map,   3);
      ArrayResize(metals_map,  4);
      ArrayResize(indices_map, 2);
      ArrayResize(ce_map,      6);

      for(int i = 0; i < 3; i++)  forex_map[i]   = MathMax(0, IndexOf(forex_c[i]));
      for(int i = 0; i < 4; i++)  metals_map[i]   = MathMax(0, IndexOf(metals_c[i]));
      for(int i = 0; i < 2; i++)  indices_map[i]  = MathMax(0, IndexOf(indices_c[i]));
      for(int i = 0; i < 6; i++)  ce_map[i]       = MathMax(0, IndexOf(ce_c[i]));
   }

   int  Count()               const { return g_active_count; }
   string Canonical(int i)    const { return (i >= 0 && i < g_active_count) ? g_active_canonical[i] : ""; }
};

CSymbolFilter g_filter;

#endif // SYMBOLFILTER_MQH
