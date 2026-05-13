#ifndef DASHBOARD_MQH
#define DASHBOARD_MQH

#include "Defines.mqh"
#include "StateMachine.mqh"
#include "MetaController.mqh"
#include "RiskManager.mqh"

//=============================================================================
// CDashboard — renders text overlay on the chart showing:
//   system state, model versions, val_acc, win rate, MetaController weights,
//   last hot-reload timestamp, daily DD, per-symbol last signal.
//=============================================================================

#define DASH_PREFIX "HYDRA4_DASH_"
#define DASH_X      10
#define DASH_Y_START 20
#define DASH_LINE_H  16
#define DASH_FONT    "Courier New"
#define DASH_FONT_SZ 8

struct SDashSymbolRow
{
   string canonical;
   int    direction;
   float  confidence;
   float  uncertainty;
   float  session_gate;
   double val_acc;
   string last_reload;
};

class CDashboard
{
private:
   int       m_n_rows;
   string    m_obj_names[64];
   int       m_obj_count;

   //--- Create or update a text label on the chart
   void SetLabel(const string name, int x, int y, const string text,
                 color clr = clrWhite, int font_sz = DASH_FONT_SZ)
   {
      string full_name = DASH_PREFIX + name;
      if(ObjectFind(0, full_name) < 0)
      {
         ObjectCreate(0, full_name, OBJ_LABEL, 0, 0, 0);
         ObjectSetInteger(0, full_name, OBJPROP_CORNER,    CORNER_LEFT_UPPER);
         ObjectSetInteger(0, full_name, OBJPROP_XDISTANCE, x);
         ObjectSetInteger(0, full_name, OBJPROP_SELECTABLE, false);
         ObjectSetString(0,  full_name, OBJPROP_FONT,      DASH_FONT);
         ObjectSetInteger(0, full_name, OBJPROP_FONTSIZE,  font_sz);
         // Track for cleanup
         if(m_obj_count < 64)
            m_obj_names[m_obj_count++] = full_name;
      }
      ObjectSetInteger(0, full_name, OBJPROP_YDISTANCE, y);
      ObjectSetString(0,  full_name, OBJPROP_TEXT,      text);
      ObjectSetInteger(0, full_name, OBJPROP_COLOR,     clr);
   }

   string DirStr(int d)
   {
      if(d > 0)  return "LONG ";
      if(d < 0)  return "SHORT";
      return "FLAT ";
   }

   color StateColor(ENUM_HYDRA_STATE s)
   {
      switch(s)
      {
         case STATE_LIVE:          return clrLimeGreen;
         case STATE_LIVE_PAUSED:   return clrOrange;
         case STATE_SHUTDOWN:      return clrRed;
         case STATE_MODEL_MISSING: return clrYellow;
         default:                  return clrGray;
      }
   }

public:
   CDashboard() : m_n_rows(0), m_obj_count(0) {}

   void Init() { m_obj_count = 0; }

   void Cleanup()
   {
      for(int i = 0; i < m_obj_count; i++)
         ObjectDelete(0, m_obj_names[i]);
      m_obj_count = 0;
   }

   //--- Full render. Call from OnTick() or OnTimer() at low frequency.
   void Render(const SDashSymbolRow &rows[], int n_rows)
   {
      int y = DASH_Y_START;

      // Header
      ENUM_HYDRA_STATE st = g_state.State();
      SetLabel("hdr_title", DASH_X, y,
               "HYDRA mk4  v" + HYDRA_VERSION + "  " + TimeToString(TimeCurrent()),
               clrCyan, DASH_FONT_SZ + 1);
      y += DASH_LINE_H + 4;

      SetLabel("hdr_state", DASH_X, y,
               "State: " + g_state.StateStr() +
               (g_state.Reason() != "" ? "  [" + g_state.Reason() + "]" : ""),
               StateColor(st));
      y += DASH_LINE_H;

      // Risk
      SetLabel("hdr_risk", DASH_X, y,
               StringFormat("DD: %.2f%%  (pause@%.0f%%  halt@%.0f%%)",
                            g_risk.DailyDDPct() * 100,
                            DAILY_DD_PAUSE * 100, DAILY_DD_SHUTDOWN * 100),
               g_risk.DailyDDPct() > DAILY_DD_PAUSE ? clrOrange : clrWhite);
      y += DASH_LINE_H;

      // MetaController weights
      string wstr = "Weights: ";
      string agent_names[4] = {"PRISM","GNN","APEX","CE"};
      for(int a = 0; a < 4; a++)
         wstr += agent_names[a] + "=" + DoubleToString(g_meta.GetWeight(a) * 100, 0) + "%  ";
      SetLabel("hdr_weights", DASH_X, y, wstr, clrSkyBlue);
      y += DASH_LINE_H + 4;

      // Column headers
      SetLabel("col_hdr", DASH_X, y,
               "Symbol       Dir    Conf  Uncrt  Gate  Val%  Reload",
               clrDarkGray);
      y += DASH_LINE_H;

      // Per-symbol rows
      for(int i = 0; i < n_rows; i++)
      {
         string lbl_name = "row_" + IntegerToString(i);
         color  row_clr  = clrWhite;
         if(rows[i].direction > 0)       row_clr = clrLimeGreen;
         else if(rows[i].direction < 0)  row_clr = clrTomato;
         if(rows[i].session_gate < SESSION_THRESHOLD) row_clr = clrDarkGray;

         string row_txt = StringFormat(
            "%-12s %s  %.3f  %.3f  %.2f  %.1f%%  %s",
            rows[i].canonical,
            DirStr(rows[i].direction),
            rows[i].confidence,
            rows[i].uncertainty,
            rows[i].session_gate,
            rows[i].val_acc * 100.0,
            rows[i].last_reload
         );
         SetLabel(lbl_name, DASH_X, y, row_txt, row_clr);
         y += DASH_LINE_H;
      }

      ChartRedraw();
   }
};

CDashboard g_dashboard;

#endif // DASHBOARD_MQH
