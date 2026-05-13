#ifndef DEFINES_MQH
#define DEFINES_MQH

//=============================================================================
// HYDRA mk4.3 — MT5-canonical feature parity floor.
//
// The model now consumes only what FeatureEncoder.mqh::Encode() actually
// populates: the M5 block (50 raw features × 4 transforms = 200 dims).
// HTF / spectral / pattern / statreg / xasset blocks were never wired
// MQL5-side and have been culled from the Python feature engine too.
// Closing this gap eliminates the silent zero-padding that caused
// trained models to receive 83% null inputs at live inference.
//
// Constants below MUST stay in sync with python/config.py.
//=============================================================================

//--- Version
#define HYDRA_VERSION       "4.3.0"
#define HYDRA_MAGIC         20260506

//--- Feature dimensions (parity-floored)
//    Slot layout (200 dims):
//      0..49     raw value of features 0..49
//      50..99    rolling mean over last 20 bars
//      100..149  rolling std  over last 20 bars
//      150..199  delta = current - 20-bars-ago
#define RAW_FEATURES        50
#define M5_DIM              200      // 50 raw × 4 transforms

//--- Feature block start indices
#define FEAT_BLOCK_M5_START   0

#define FEATURE_DIM         200      // direction model input
#define EXEC_CTX_DIM         40      // 4 microstructure + 36 EA-injected pos slots
#define EXEC_FEATURE_DIM    240      // exec model input  (200 + 40)
#define MOD_FEATURE_DIM     208      // modify model input (200 + 8 pos_context)
                                     // pos_context at indices 200..207
#define FEATURE_WARMUP_BARS 100

//--- Network dims (must match python/config.py)
//    mk4.3.1: rescaled for 200-dim input.
#define PRISM_H0            256
#define PRISM_H1            128
#define PRISM_H2            64
#define PRISM_H3            32
#define APEX_H0             384
#define APEX_H1             192
#define APEX_H2             96
#define APEX_H3             48
#define GNN_H0              128
#define GNN_HIDDEN          32
#define GNN_NODES           4
#define CE_H1               96
#define CE_H2               48
#define EXEC_H1             192
#define EXEC_H2             96
#define MOD_H1              96
#define MOD_H2              48

//--- Labels
#define LABEL_FORWARD_BARS  20
#define MAX_SYMBOLS         16
#define TICK_BUFFER_SIZE    500

//--- MC Dropout
#define MC_T                20
#define MC_DROPOUT_RATE     0.30
#define MC_UNCERTAINTY_CAP  0.15

//--- Inference thresholds (must match python/config.py)
#define CONF_THRESHOLD      0.60
#define SESSION_THRESHOLD   0.00
#define TIMING_THRESHOLD    0.60
#define TIMING_WAIT_MIN     0.30
#define LIMIT_EXPIRY_BARS   3

//--- Execution model output indices
#define EXEC_IDX_TIMING     0
#define EXEC_IDX_SL         1
#define EXEC_IDX_TP         2
#define EXEC_IDX_VOL        3
#define EXEC_IDX_SESSION    4

//--- Modification model output indices
#define MOD_IDX_MOVE_BE     0
#define MOD_IDX_TRAIL       1
#define MOD_IDX_CLOSE       2
#define MOD_BE_THRESHOLD    0.6
#define MOD_CLOSE_THRESHOLD 0.7

//--- Vol mult mapping: sigmoid 0..1 -> [0.5, 2.0]
#define VOL_MULT_MIN        0.5
#define VOL_MULT_MAX        2.0

//--- Model quality gate
#define MIN_VAL_ACC_TRADE   0.60

// mk4.3: MTF alignment indices removed (block doesn't exist in 200-dim layout).

//--- Risk
#define DAILY_DD_PAUSE      0.05
#define DAILY_DD_SHUTDOWN   0.10
#define MAX_RISK_PER_TRADE  0.01

//--- Hot-reload watcher
#define MODEL_WATCH_SEC     30
#define TIMER_MS            100

//--- File paths (Common Files root)
#define HYDRA_FOLDER        ""
#define CONFIG_PREFIX       "BrokerConfig_"
#define CONFIG_EXT          ".ini"
#define META_EXT            "_meta.json"
#define ONNX_EXT            ".onnx"
#define SIGNAL_LOG          "HYDRA4_signals.csv"
#define MONITOR_JSON        "HYDRA4_monitor.json"

//--- Enums

enum ENUM_HYDRA_STATE
{
   STATE_BOOT          = 0,
   STATE_LIVE          = 1,
   STATE_LIVE_PAUSED   = 2,
   STATE_SHUTDOWN      = 3,
   STATE_MODEL_MISSING = 4
};

enum ENUM_ASSET_CLASS
{
   ASSET_UNKNOWN = 0,
   ASSET_FOREX,
   ASSET_METALS,
   ASSET_INDICES,
   ASSET_CRYPTO,
   ASSET_ENERGY
};

enum ENUM_HYDRA_TRADE_MODE
{
   HTRADE_FULL = 0,
   HTRADE_LONGONLY,
   HTRADE_SHORTONLY,
   HTRADE_CLOSEONLY,
   HTRADE_DISABLED
};

enum ENUM_SPREAD_KIND
{
   SPREAD_FIXED = 0,
   SPREAD_VARIABLE
};

enum ENUM_SWAP_KIND
{
   SWAP_POINTS = 0,
   SWAP_MONEY,
   SWAP_INTEREST,
   SWAP_CURRENCY
};

enum ENUM_REGIME
{
   REGIME_BULL     = 0,
   REGIME_SIDEWAYS = 1,
   REGIME_BEAR     = 2
};

enum ENUM_AGENT_ID
{
   AGENT_PRISM  = 0,
   AGENT_GNN    = 1,
   AGENT_APEX   = 2,
   AGENT_CE     = 3
};

//--- Structs

struct SSession
{
   int  from_min;
   int  to_min;
   bool closed;
};

struct SSymbolSpec
{
   string   canonical;
   string   broker_name;
   ENUM_ASSET_CLASS asset_class;
   int      digits;
   double   point_val;
   double   pip_size;
   double   tick_size;
   double   tick_value;
   double   contract_size;
   double   spread_pips;
   ENUM_SPREAD_KIND spread_type;
   int      stops_level;
   int      freeze_level;
   double   volume_min;
   double   volume_max;
   double   volume_step;
   ENUM_HYDRA_TRADE_MODE trade_mode;
   double   swap_long;
   double   swap_short;
   ENUM_SWAP_KIND swap_type;
   SSession sessions[7];
   bool     mw_present;
};

struct SSignal
{
   string   canonical;
   int      direction;        // +1=LONG, -1=SHORT, 0=FLAT
   float    confidence;
   float    uncertainty;
   float    timing;
   float    sl_pips;
   float    tp_pips;
   float    vol_mult;
   float    session_gate;
   ENUM_REGIME regime;
   bool     skip;
};

struct SModSignal
{
   ulong  ticket;
   float  move_sl_to_be;
   float  trail_sl_pips;
   float  close_now;
};

struct STradeResult
{
   ulong  ticket;
   bool   success;
   int    error;
};

#endif // DEFINES_MQH
