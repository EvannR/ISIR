# 2D Grid Environment for Empathic Resource Sharing

This directory contains a standalone implementation of a multi-agent grid environment where agents learn to navigate and share resources empathetically.

## Overview

The simulation implements a game-theoretic scenario where agents must:
1. **Learn navigation**: Move on a grid to find and reach resources
2. **Learn empathic behavior**: Avoid taking resources when they are more satisfied than nearby agents

Each agent observes a square area around itself (local vision) and receives:
- **Positive reward** when successfully exploiting a resource
- **Negative reward (penalty)** when exploiting a resource while being more satisfied than nearby agents

## Files

- **env_2d.py**: Grid environment implementation with local observations and empathic reward system
- **agent_2d.py**: Agent class with meal history tracking, satisfaction calculation, and DQN implementation
- **main_2d.py**: Main simulation loop with training and CSV data recording

## Key Features

### Environment
- Grid-based world (configurable size)
- Resources placed randomly on the grid
- Agents with local vision (square around themselves)
- Actions: Move (up, down, left, right) - resources are automatically consumed when agent moves onto them

### Observation Space
Each agent observes a `(vision_size, vision_size, 3)` tensor:
- **Channel 0**: Terrain information
  - `-1`: Out of bounds
  - `0`: Empty cell
  - `1`: Resource
  - `2`: Another agent
- **Channel 1**: Satisfaction levels of other agents in view (0-1)
- **Channel 2**: Agent's own satisfaction (broadcasted for self-awareness)

### Reward System
- **Step cost**: Small negative reward per step (encourages efficiency)
- **Exploit reward**: Positive reward when successfully exploiting a resource
- **Empathic penalty**: Minimum (maximum punishment) consumption bonus when nearby agents are unsatisfied
- **Empathy alpha (optional)**: Controls reward shaping by mixing **own satisfaction** and **nearby others’ emotion** when consuming
- **Loop penalty (optional)**: Additional negative reward when an agent repeats an action pattern (length ≥ 2) more than 3 times (≥4 repetitions)

### Agent Learning
- **DQN**: Deep Q-Network for learning optimal policies
- **Experience replay**: Stores and samples past experiences
- **Target network**: Stabilizes training
- **Epsilon-greedy exploration**: Balances exploration vs exploitation

## Usage

### Basic Usage

```python
from env_2d import EmpathicGridEnv
from agent_2d import DQNAgent

# Create environment
env = EmpathicGridEnv(
    grid_size=10,
    num_agents=6,
    num_resources=15,
    vision_radius=2
)

# Create agents
agents = []
for i in range(6):
    agent = DQNAgent(
        agent_id=i,
        state_size=env.observation_space.shape,
        action_size=env.action_space.n
    )
    agents.append(agent)

env.agents = agents

# Run episode
obs, _ = env.reset()
for step in range(1000):
    actions = [agent.select_action(obs[i]) for i, agent in enumerate(agents)]
    next_obs, rewards, done, truncated, info = env.step(actions)
    
    # Update agents
    for i, agent in enumerate(agents):
        agent.step(next_obs[i], rewards[i], done)
    
    obs = next_obs
    if done:
        break
```

### Run Full Simulation

```bash
python main_2d.py
```

This will:
- Run 30 independent simulations
- Each simulation has 500 episodes
- Save results to `2D_simulation_results/` directory
- Generate CSV files with episode summaries

## Configuration

Edit constants in `main_2d.py`:

- **Grid settings**: `GRID_SIZE`, `NUM_RESOURCES`, `VISION_RADIUS`
- **Agent settings**: `NB_AGENTS`, `BETA`, `THRESHOLD`, `SMOOTHING`
- **Reward settings**: `EXPLOIT_REWARD`, `EMPATHIC_PENALTY`, `STEP_COST`
- **Empathy settings**: `EMPATHY_ALPHA`
- **Loop settings**: `LOOP_PENALTY`, `LOOP_MAX_PATTERN_LENGTH`, `LOOP_MIN_REPETITIONS`
- **Training settings**: `EPISODE_NUMBER`, `MAX_STEPS`, `SIMULATION_NUMBER`
- **DQN hyperparameters**: `PARAMS_DQN` dictionary

## Output Data

The simulation generates CSV files with the following columns:

- `simulation_number`: Simulation run index
- `seed`: Random seed used
- `episode`: Episode number
- `total_steps`: Number of steps in episode
- `resources_remaining`: Resources left at end of episode
- `total_reward_{i}`: Total reward for agent i
- `total_exploits_{i}`: Total exploit attempts for agent i
- `total_empathic_penalties_{i}`: Times agent i received empathic penalty
- `total_loop_events_{i}`: Number of steps where a loop pattern was detected for agent i
- `final_satisfaction_{i}`: Final satisfaction level of agent i
- `final_emotion_{i}`: Final emotion level of agent i

## Dependencies

- numpy
- torch (PyTorch)
- gymnasium
- tqdm
- csv (standard library)
- collections (standard library)
- random (standard library)
- os (standard library)

## Differences from Original Game-Theoretic Implementation

1. **Spatial representation**: Grid-based instead of abstract resource pool
2. **Navigation learning**: Agents must learn to move on the grid
3. **Local observations**: Agents only see a square around themselves
4. **Action space**: Only movement actions (4 directions) - resources consumed automatically when stepped on
5. **Empathic behavior**: Negative reward when consuming resource while more satisfied than nearby agents

## Notes

- Agents observe satisfaction levels of nearby agents in their vision
- Satisfaction is calculated from meal history using beta-weighted average
- Emotion is derived from satisfaction using linear or sigmoid mapping
- The empathic penalty encourages agents to leave resources for less satisfied neighbors

