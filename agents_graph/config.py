"""
Configuration constants for the graph-based multi-agent environment.
"""

# ----------------------------------------
# Simulation Settings
# ----------------------------------------
NUM_EPISODES = 300
MAX_STEPS = 200
NUM_AGENTS = 6
NUM_SIMULATIONS_PER_CONDITION = 3

# ----------------------------------------
# Graph Settings
# ----------------------------------------
DEFAULT_GRAPH_TYPE = 'directed_random'
EDGE_PROBABILITY = 0.4

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
ALPHA = 0.5         # Empathy weight (0=selfish, 1=fully empathic)

# ----------------------------------------
# Agent Settings
# ----------------------------------------
MEMORY_SIZE = 10
BETA = 0.5          # Satisfaction: weight of last meal vs history
THRESHOLD = 0.5
SMOOTHING = 'linear'
SIGMOID_GAIN = 10.0

# ----------------------------------------
# DQN Hyperparameters
# ----------------------------------------
LEARNING_RATE = 0.002
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.997
EPSILON_MIN = 0.05
BATCH_SIZE = 32
HIDDEN_SIZE = 64
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
# [own_satisfaction, own_emotion, pool_level, mean_neighbor_emotion]
OBSERVATION_DIM = 4
