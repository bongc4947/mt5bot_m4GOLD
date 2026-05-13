#ifndef TICKBUFFER_MQH
#define TICKBUFFER_MQH

#include "Defines.mqh"

//=============================================================================
// Circular tick ring — one per symbol. Stored as global fixed 2D-style arrays
// because MQL5 forbids 2D arrays as class members.
//=============================================================================

struct STick
{
   datetime time;
   double   bid;
   double   ask;
   double   last;
   double   volume;
   double   spread_pts;
};

// Global tick storage: [sym_idx][tick_idx]
// Use flat 1D with manual indexing: idx = sym * TICK_BUFFER_SIZE + pos
STick    g_tick_ring[MAX_SYMBOLS * TICK_BUFFER_SIZE];
int      g_tick_head[MAX_SYMBOLS];   // next write position (circular)
int      g_tick_count[MAX_SYMBOLS];  // filled count (up to TICK_BUFFER_SIZE)

class CTickBuffer
{
public:
   void Init()
   {
      for(int i = 0; i < MAX_SYMBOLS; i++) { g_tick_head[i] = 0; g_tick_count[i] = 0; }
   }

   //--- Clear one symbol's ring (used by historical feature exporter to
   //    inject synthetic ticks bar-by-bar without leaking state across calls).
   void ResetSymbol(int sym_idx)
   {
      if(sym_idx < 0 || sym_idx >= MAX_SYMBOLS) return;
      g_tick_head[sym_idx]  = 0;
      g_tick_count[sym_idx] = 0;
   }

   //--- Push latest tick for symbol index sym_idx
   void Push(int sym_idx, const MqlTick &t, double spread_pts)
   {
      if(sym_idx < 0 || sym_idx >= MAX_SYMBOLS) return;
      int pos = g_tick_head[sym_idx];
      int flat = sym_idx * TICK_BUFFER_SIZE + pos;
      g_tick_ring[flat].time       = t.time;
      g_tick_ring[flat].bid        = t.bid;
      g_tick_ring[flat].ask        = t.ask;
      g_tick_ring[flat].last       = t.last;
      g_tick_ring[flat].volume     = (double)t.volume;
      g_tick_ring[flat].spread_pts = spread_pts;
      g_tick_head[sym_idx] = (pos + 1) % TICK_BUFFER_SIZE;
      if(g_tick_count[sym_idx] < TICK_BUFFER_SIZE) g_tick_count[sym_idx]++;
   }

   //--- Get tick at offset from head (0 = newest, 1 = one before, etc.)
   bool Get(int sym_idx, int offset, STick &out) const
   {
      if(sym_idx < 0 || sym_idx >= MAX_SYMBOLS) return false;
      int cnt = g_tick_count[sym_idx];
      if(offset >= cnt) return false;
      int pos  = (g_tick_head[sym_idx] - 1 - offset + TICK_BUFFER_SIZE) % TICK_BUFFER_SIZE;
      out = g_tick_ring[sym_idx * TICK_BUFFER_SIZE + pos];
      return true;
   }

   int Count(int sym_idx) const
   {
      if(sym_idx < 0 || sym_idx >= MAX_SYMBOLS) return 0;
      return g_tick_count[sym_idx];
   }

   //--- Mid price of newest tick
   double Mid(int sym_idx) const
   {
      STick t;
      if(!Get(sym_idx, 0, t)) return 0.0;
      return (t.bid + t.ask) * 0.5;
   }
};

CTickBuffer g_tick_buf;

#endif // TICKBUFFER_MQH
