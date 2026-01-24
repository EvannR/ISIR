# Graph-Based Multi-Agent Empathic Environment

## Session Summary (2026-01-24)

Created a new `agents_graph/` folder implementing a graph-based variant of the 2D empathic resource sharing simulation.

## Key Design Decisions

### Architecture Changes from 2D Grid
- **No movement**: Agents are fixed nodes in a social graph
- **Shared resource pool**: Central pool instead of spatial resources
- **Graph-based visibility**: Edges determine who can see whose emotions
- **Binary actions**: `[0=wait, 1=consume]` instead of 4 directional movements

### What Was Reused from 2D
- `Agent` class (meal history, satisfaction/emotion calculation)
- `EmpathicRewardCalculator` (reward formula)
- `ReplayBuffer` and `Experience` (DQN training)
- Training loop structure and CSV logging format

## File Structure

```
agents_graph/
├── __init__.py          # Package exports
├── config.py            # Configuration constants
├── social_graph.py      # SocialGraph class (topology management)
├── resource_pool.py     # ResourcePool class (centralized resources)
├── agent_graph.py       # GraphDQNAgent (imports from 2D)
├── env_graph.py         # EmpathicGraphEnv (main environment)
├── main_graph.py        # Training script
└── CLAUDE.md            # This file
```

## Key Classes

### SocialGraph (`social_graph.py`)
- Manages graph topology (adjacency matrix)
- Supports: `random`, `small_world`, `fully_connected`, `ring`, `custom`
- Methods: `get_neighbors()`, `get_neighbor_weights()`
- **Affiliation hooks ready**: `update_weight()` for future weighted connections

### ResourcePool (`resource_pool.py`)
- Centralized pool with configurable regeneration
- Modes: `constant`, `proportional`, `none`
- Handles simultaneous consumption with fair random allocation

### EmpathicGraphEnv (`env_graph.py`)
- Gym-compatible environment
- Observation: `[own_satisfaction, own_emotion, pool_level, neighbor_emotions_and_weights...]`
- Observation dim: `3 + max_neighbors * 2` (default 19)
- Action space: `Discrete(2)`

### GraphDQNAgent (`agent_graph.py`)
- Wraps `Agent` from 2D module
- Flat observation processing
- Binary action selection

## Running Experiments

```bash
cd agents_graph
python main_graph.py
```

Results saved to `agents_graph/graph_simulation_results/`

## Future Work (Not Implemented)

### Affiliation Theory
Infrastructure is ready in `SocialGraph.update_weight()`:
- Cooperative interactions strengthen bonds
- Competitive interactions weaken bonds
- Weights influence emotional consideration

### Dynamic Topology
Currently static edges, but could evolve based on:
- Interaction frequency
- Reciprocal behavior
- Social learning

## Configuration (`config.py`)

Key parameters:
- `NUM_AGENTS = 5`
- `MAX_NEIGHBORS = 8` (for observation padding)
- `INITIAL_POOL = 50`
- `REGENERATION_MODE = 'constant'`
- `DEFAULT_GRAPH_TYPE = 'random'`
- `EDGE_PROBABILITY = 0.4`

## Imports from 2D Module

The `agent_graph.py` file imports from `../2D/agent_2d.py`:
```python
from agent_2d import Agent, EmpathicRewardCalculator, ReplayBuffer, Experience
```

This means the 2D folder must remain in place for the graph module to work.
