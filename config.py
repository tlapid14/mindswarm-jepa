"""Shared constants. Everything that more than one module needs lives here."""

from pathlib import Path

# Filesystem layout
PROJECT_ROOT:   Path = Path(__file__).resolve().parent
DATA_DIR:       Path = PROJECT_ROOT / "data"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"
RESULTS_DIR:    Path = PROJECT_ROOT / "results"

# World geometry
WORLD_WIDTH:  int = 1200
WORLD_HEIGHT: int = 900
SIM_DT: float = 1.0

# Agent counts
N_BLUE:  int = 16
N_RELAY: int = 6
N_RED:   int = 3

# Padded maxima for state-vector layout. Equal to the live counts in v2,
# but kept separate so curriculum / ablation runs don't need to reshape.
MAX_BLUE:  int = N_BLUE
MAX_RELAY: int = N_RELAY

# Kinematics (px/step)
BLUE_MAX_SPEED: float = 4.0
BLUE_ACCEL:     float = 0.8
RED_MAX_SPEED:  float = 1.5

# Comms + jamming.
# Per-Red intensity: I(d) = JAM_STRENGTH / (1 + (d / JAM_REFERENCE)**2),
# summed over Reds. An edge breaks when the sum at its midpoint exceeds
# JAM_BREAK_THRESHOLD.
COMM_RANGE:          float = 450.0
JAM_STRENGTH:        float = 1.0
JAM_REFERENCE:       float = 80.0
JAM_BREAK_THRESHOLD: float = 0.3
# COMM_RANGE started at 220; with a 3x2 relay grid that left a
# coverage hole in the middle of the world that auto-failed episodes
# regardless of jamming. 450 makes the relay backbone actually cover.

# Connectivity rule: a Blue counts as connected iff a path of unbroken
# edges reaches some relay. Episode fails if the connected ratio stays
# below the threshold for FAILURE_SUSTAIN_STEPS in a row.
CONNECTIVITY_FAILURE_THRESHOLD: float = 0.5
FAILURE_SUSTAIN_STEPS: int = 3

# How stale the observed state is. Constant per episode for now.
COMM_LATENCY_STEPS: int = 3

# Episodes
MAX_EPISODE_STEPS: int = 400
N_OBJECTIVES:      int = 3

# Supervised problem
PAST_WINDOW:     int = 10   # encoder input length
FUTURE_OFFSET:   int = 20   # prediction horizon
FAILURE_HORIZON: int = 35   # label = 1 iff failure inside this many future steps
DATASET_STRIDE:  int = 3    # sliding window stride per episode

# State vector layout:
#   blues  (MAX_BLUE  x 5):  x, y, vx, vy, connected
#   relays (MAX_RELAY x 2):  x, y
#   scalars (5):             conn_ratio, avg_path_len, jam_coverage,
#                            mean_red_dist, fraction_blues_jammed
PER_BLUE_FEATURES:  int = 5
PER_RELAY_FEATURES: int = 2
SCALAR_FEATURES:    int = 5
STATE_DIM: int = (
    MAX_BLUE  * PER_BLUE_FEATURES
    + MAX_RELAY * PER_RELAY_FEATURES
    + SCALAR_FEATURES
)  # 80 + 12 + 5 = 97

# Models
LATENT_DIM:               int = 64
GRU_HIDDEN_DIM:           int = 128
PREDICTOR_HIDDEN_DIM:     int = 128
RISK_HEAD_HIDDEN_DIM:     int = 64
BASELINE_LSTM_HIDDEN_DIM: int = 128

# Training
SEED:            int   = 0
BCE_LOSS_WEIGHT: float = 1.0   # lambda in: loss = mse + lambda * bce

# Viz
FPS: int = 30
COLOR_BG                = ( 15,  15,  25)
COLOR_BLUE              = ( 80, 160, 255)
COLOR_BLUE_DISCONNECTED = (160, 160, 160)
COLOR_RELAY             = (255, 220,  80)
COLOR_RED               = (220,  80,  80)
COLOR_JAM_FIELD         = (220,  80,  80)
COLOR_EDGE_OK           = ( 80, 200, 120)
COLOR_EDGE_BROKEN       = ( 80,  80,  80)
COLOR_TEXT              = (220, 220, 220)
COLOR_SPARK_JEPA        = (100, 230, 200)
COLOR_SPARK_BASELINE    = (255, 180,  90)
