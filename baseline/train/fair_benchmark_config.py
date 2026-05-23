from __future__ import annotations

# Main-model-aligned benchmark protocol shared by all baseline trainers.

ENV_VARS = ["thetao", "chl", "uo", "vo", "so", "zos", "o2"]

SEQ_LEN = 12
PRED_LEN = 1
ROLLOUT_2024_HORIZON = 12
ROLLOUT_START_INDEX = 144

BATCH_SIZE = 2
ACCUMULATION_STEPS = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

TRAIN_LOSS = "mse"
HOTSPOT_THRESHOLD = 0.2755

LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 8
MIN_LR = 1e-6
EARLY_STOPPING_PATIENCE = 20
MAX_GRAD_NORM = 5.0

# Model-specific auxiliary-loss weights.
PREDRNN_V2_DECOUPLE_BETA = 0.1
EXPRECAST_FACL_WEIGHT = 1e-5
