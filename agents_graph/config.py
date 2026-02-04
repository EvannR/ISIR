"""
Configuration constants for the graph-based multi-agent environment.
Mirrors the 2D environment settings where applicable.
"""

# ----------------------------------------
# Simulation Settings
# ----------------------------------------
NUM_EPISODES = 300
MAX_STEPS = 200
NUM_AGENTS = 5
NUM_SIMULATIONS_PER_CONDITION = 3

# ----------------------------------------
# Graph Settings
# ----------------------------------------
DEFAULT_GRAPH_TYPE = 'dynamic'  # 'random', 'dynamic', 'small_world', 'fully_connected', 'ring'
EDGE_PROBABILITY = 0.4          # For random graph
SMALL_WORLD_K = 2               # Initial neighbors per side (Watts-Strogatz)
SMALL_WORLD_P = 0.3             # Rewiring probability (Watts-Strogatz)
MAX_NEIGHBORS = 8               # Maximum neighbors for observation padding

# Dynamic graph (emerging connections; edges can appear and disappear)
DYNAMIC_P_BASE = 0.05           # Base probability to connect each step
DYNAMIC_P_MAX = 0.95            # Max connection probability (reinforcement cap)
DYNAMIC_DELTA = 0.02            # Increase in p when edge is present this step
DYNAMIC_DECAY = 0.9             # Decay toward p_base when edge absent (p = decay*p + (1-decay)*p_base)

# ----------------------------------------
# Resource Pool Settings
# ----------------------------------------
INITIAL_POOL = 50
MAX_POOL_CAPACITY = 100
REGENERATION_RATE = 1.0
REGENERATION_MODE = 'constant'  # 'constant', 'proportional', 'none'

# ----------------------------------------
# Reward Settings
# ----------------------------------------
EXPLOIT_REWARD = 1.0
EMPATHIC_PENALTY = -1.0
STEP_COST = -0.01
WAIT_REWARD = 0.0
FAILED_CONSUMPTION_PENALTY = -0.1

# ----------------------------------------
# Agent Settings (matching 2D)
# ----------------------------------------
MEMORY_SIZE = 10                # Meal history buffer
BETA = 0.5                      # Satisfaction: weight of last meal vs history
THRESHOLD = 0.5                 # Emotion threshold
SMOOTHING = 'linear'            # Emotion smoothing: 'linear' or 'sigmoid'
SIGMOID_GAIN = 10.0

# ----------------------------------------
# DQN Hyperparameters (matching 2D)
# ----------------------------------------
LEARNING_RATE = 0.002
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.997           # Slightly slower decay (simpler action space)
EPSILON_MIN = 0.05
BATCH_SIZE = 32
HIDDEN_SIZE = 64                # Smaller than 2D (simpler observations)
UPDATE_TARGET_EVERY = 20
REPLAY_BUFFER_CAPACITY = 50000

# ----------------------------------------
# Action Constants
# ----------------------------------------
ACTION_WAIT = 0
ACTION_CONSUME = 1
NUM_ACTIONS = 2

# ----------------------------------------
# Observation Space
# ----------------------------------------
# [own_satisfaction, own_emotion, pool_level, neighbor_emotions_and_weights...]
# Dimension: 3 + max_neighbors * 2
OBSERVATION_DIM = 3 + MAX_NEIGHBORS * 2  # = 19 for max_neighbors=8
