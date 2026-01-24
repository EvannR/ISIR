import numpy as np
import random
import csv
import os
import time
import torch
from tqdm import tqdm
from env_2d import EmpathicGridEnv
from agent_2d import DQNAgent


# ----------------------------------------
# Utility functions (aligned with notebook)
# ----------------------------------------

def compute_gini_coefficient(values):
    """
    Compute the Gini coefficient for a list of values.
    Gini = 0 means perfect equality (everyone has the same)
    Gini = 1 means perfect inequality (one person has everything)
    """
    values = np.array(values, dtype=float)
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0

    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)

    # Compute Gini using the formula: G = (2 * sum(i * x_i) - (n + 1) * sum(x_i)) / (n * sum(x_i))
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values) - (n + 1) * np.sum(sorted_values)) / (n * np.sum(sorted_values))


def compute_average_inter_agent_distance(positions):
    """
    Compute the average Manhattan distance between all pairs of agents.
    """
    if len(positions) < 2:
        return 0.0

    total_distance = 0
    num_pairs = 0

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            total_distance += abs(x1 - x2) + abs(y1 - y2)
            num_pairs += 1

    return total_distance / num_pairs if num_pairs > 0 else 0.0


# ----------------------------------------
# Constants for the simulation
# ----------------------------------------

SIMULATION_NUMBER = 3      # Number of simulation runs (also used as seed per run)
EPISODE_NUMBER = 300        # Number of episodes per simulation
NB_AGENTS = 6              # Number of agents (matching empathy_comparison notebook)
MAX_STEPS = 200           # Maximum steps per episode
GRID_SIZE = 5             # Grid size (grid_size x grid_size)
NUM_RESOURCES = 10         # Number of resources on the grid
VISION_RADIUS = 2          # Vision radius (agent sees (2*radius+1) x (2*radius+1) square)

# Agent & emotion settings
BETA = 0.5                 # Weight of last meal vs history for satisfaction
THRESHOLD = 0.3            # Consumption rate threshold for neutral emotion (matching notebook default)
SMOOTHING = 'linear'       # Emotion smoothing: 'linear' or 'sigmoid'
SIGMOID_GAIN = 10.0        # Steepness for sigmoid mapping

# Environment reward settings
EXPLOIT_REWARD = 1.0       # Reward for successfully exploiting a resource
EMPATHIC_PENALTY = -1.0    # Penalty when exploiting while more satisfied than nearby agents (matching notebook)
STEP_COST = -0.01          # Cost per step

# Empathy shaping (game-theoretic style): mix (self satisfaction) and (others emotion)
# alpha = weight on others (0 = selfish, 1 = fully social).
EMPATHY_ALPHA = 0.5

# Loop penalty (reward shaping; OFF when 0.0)
# Loop definition:
# - pattern length >= 2
# - repeated more than 3 times (>=4 occurrences)
LOOP_PENALTY = -0.05
LOOP_MAX_PATTERN_LENGTH = 6
LOOP_MIN_REPETITIONS = 4

# RL agent hyperparameters (matching empathy_comparison notebook)
PARAMS_DQN = {
    "learning_rate": 0.002,      # Notebook uses 0.002 (faster learning)
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.995,      # Notebook uses 0.995 (faster decay)
    "epsilon_min": 0.05,
    "batch_size": 32,
    "hidden_size": 128,
    "update_target_every": 20   # Notebook uses 20 (more frequent target updates)
}


# ----------------------------------------
# Initialization functions
# ----------------------------------------

def initialize_agents_and_env(seed=None, empathy_alpha=None):
    """
    Create environment and agents for a new simulation.
    
    Args:
        seed: Random seed for reproducibility
        empathy_alpha: Empathy parameter (0.0 for non-empathic, 0.5 for empathic)
                       If None, uses global EMPATHY_ALPHA
    
    Returns:
        env: Environment instance
        agents: List of agent instances
    """
    # Use provided empathy_alpha or default to global value
    if empathy_alpha is None:
        empathy_alpha = EMPATHY_ALPHA
    
    # Create environment
    env = EmpathicGridEnv(
        grid_size=GRID_SIZE,
        num_agents=NB_AGENTS,
        num_resources=NUM_RESOURCES,
        vision_radius=VISION_RADIUS,
        beta=BETA,
        threshold=THRESHOLD,
        smoothing=SMOOTHING,
        sigmoid_gain=SIGMOID_GAIN,
        memory_size=10,
        exploit_reward=EXPLOIT_REWARD,
        empathic_penalty=EMPATHIC_PENALTY,
        empathy_alpha=empathy_alpha,
        step_cost=STEP_COST,
        loop_penalty=LOOP_PENALTY,
        loop_max_pattern_length=LOOP_MAX_PATTERN_LENGTH,
        loop_min_repetitions=LOOP_MIN_REPETITIONS,
        max_steps=MAX_STEPS,
        regeneration_mode='dynamic',  # Dynamic regeneration: fewer resources = lower respawn probability
        render_mode=None
    )
    
    # Create agents
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agents = []
    
    for i in range(NB_AGENTS):
        agent = DQNAgent(
            agent_id=i,
            state_size=state_size,
            action_size=action_size,
            memory_size=10,
            beta=BETA,
            threshold=THRESHOLD,
            smoothing=SMOOTHING,
            sigmoid_gain=SIGMOID_GAIN,
            **PARAMS_DQN
        )
        agents.append(agent)
    
    # Link agents to environment (environment expects Agent objects, not DQNAgent objects)
    env.agents = [agent.agent for agent in agents]
    
    return env, agents


# ----------------------------------------
# Step processing
# ----------------------------------------

def run_step(env, agents, simulation_index, episode, step, obs):
    """
    Execute one step: select actions, apply to env, collect info, update agents.
    
    Returns:
        record: Step record dictionary
        next_obs: Next observations
        done: Whether episode is done
    """
    # Select actions for all agents (with action masking to prevent invalid moves)
    actions = []
    for i, agent in enumerate(agents):
        valid_actions = env.get_valid_actions(i)
        action = agent.select_action(obs[i], training=True, valid_actions=valid_actions)
        actions.append(action)
    
    # Execute step in environment
    next_obs, rewards, done, truncated, info = env.step(actions)
    
    # Create step record
    record = {
        'simulation_number': simulation_index,
        'seed': simulation_index,
        'episode': episode,
        'step': step,
        'resources_remaining': info['resources_remaining'],
        'actions': actions.copy(),
        'rewards': rewards.copy(),
        'satisfactions': info['satisfactions'].copy(),
        'loop_flags': info.get('loop_flags', [False] * NB_AGENTS),
        'consumed': info.get('consumed', [False] * NB_AGENTS),
        'empathic_penalty_events': info.get('empathic_penalty_events', [False] * NB_AGENTS)
    }
    
    # Learning updates for all agents (with action masking)
    for i, agent in enumerate(agents):
        valid_actions_next = env.get_valid_actions(i)
        agent.step(next_state=next_obs[i], reward=rewards[i], done=done, valid_actions=valid_actions_next)
    
    return record, next_obs, done


# ----------------------------------------
# Simulation logic
# ----------------------------------------

def run_simulation_with_progressive_saving(simulation_index, summary_file, seed,
                                         episode_number=EPISODE_NUMBER, 
                                         step_count=MAX_STEPS, 
                                         verbose=True,
                                         empathy_alpha=None,
                                         condition_name=""):
    """
    Run a complete simulation with multiple episodes and save results progressively.
    
    Args:
        simulation_index: Index of current simulation
        summary_file: Path to summary CSV file
        seed: Random seed for this simulation
        episode_number: Number of episodes to run
        step_count: Maximum steps per episode
        verbose: Whether to show progress bars
        empathy_alpha: Empathy parameter (0.0 for non-empathic, 0.5 for empathic)
                      If None, uses global EMPATHY_ALPHA
        condition_name: Name of condition (for progress prints)
    """
    start_time = time.time()
    np.random.seed(seed)
    random.seed(seed)
    env, agents = initialize_agents_and_env(seed=seed, empathy_alpha=empathy_alpha)
    
    episode_iter = tqdm(range(episode_number), 
                       desc=f"Simulation {simulation_index + 1}/{SIMULATION_NUMBER}") if verbose else range(episode_number)
    
    # Progress update intervals
    progress_interval = max(1, episode_number // 10)  # Update every 10% of episodes
    
    for episode in episode_iter:
        # FIX: Use episode-specific seed to ensure different random sequences per episode
        episode_seed = seed + episode * 1000
        np.random.seed(episode_seed)
        random.seed(episode_seed)

        obs, _ = env.reset(seed=episode_seed)

        # Reset agents
        for i, agent in enumerate(agents):
            agent.reset(observation=obs[i])

        # Episode statistics
        total_rewards = np.zeros(NB_AGENTS)
        total_exploits = np.zeros(NB_AGENTS)
        total_empathic_penalties = np.zeros(NB_AGENTS)
        total_loop_events = np.zeros(NB_AGENTS)
        episode_step_records = []

        # NEW: Additional metrics tracking (aligned with notebook)
        episode_distances = []
        resource_counts_during_episode = []
        depletion_step = step_count  # Default: never depleted
        first_depletion_recorded = False
        consumption_per_agent = [0] * NB_AGENTS

        step_iter = tqdm(range(step_count), desc=f"Episode {episode}", leave=False) if verbose else range(step_count)

        for step in step_iter:
            # Track inter-agent distance before step
            episode_distances.append(compute_average_inter_agent_distance(env.agent_positions))

            # Track resources BEFORE step
            resource_counts_during_episode.append(len(env.resource_positions))

            record, obs, done = run_step(env, agents, simulation_index, episode, step, obs)

            # FIX: Check depletion AFTER step (using info from record)
            # This catches when resources hit 0, even if they respawn later
            if record['resources_remaining'] == 0 and not first_depletion_recorded:
                depletion_step = step
                first_depletion_recorded = True

            # Track resources AFTER step
            resource_counts_during_episode.append(record['resources_remaining'])

            episode_step_records.append(record)
            total_rewards += np.array(record['rewards'])
            total_loop_events += np.array(record.get('loop_flags', [False] * NB_AGENTS), dtype=float)

            # Count resource consumptions and empathic-penalty events (shaped reward)
            for i in range(NB_AGENTS):
                if record.get('consumed', [False] * NB_AGENTS)[i]:
                    total_exploits[i] += 1
                    consumption_per_agent[i] += 1
                if record.get('empathic_penalty_events', [False] * NB_AGENTS)[i]:
                    total_empathic_penalties[i] += 1

            if done:
                break
        
        # Create episode summary
        final_satisfactions = [agent.get_satisfaction() for agent in agents]
        final_emotions = [agent.get_emotion() for agent in agents]

        # NEW: Compute Gini coefficients and average distance (aligned with notebook)
        gini_consumption = compute_gini_coefficient(consumption_per_agent)
        gini_satisfaction = compute_gini_coefficient(final_satisfactions)
        avg_inter_agent_distance = np.mean(episode_distances) if episode_distances else 0.0
        avg_resources_during_episode = np.mean(resource_counts_during_episode) if resource_counts_during_episode else 0.0
        empathic_penalties_count = int(np.sum(total_empathic_penalties))

        summary = {
            'simulation_number': simulation_index,
            'seed': seed,
            'episode': episode,
            'total_steps': step + 1,
            'resources_remaining': len(env.resource_positions),
            'total_rewards': total_rewards.tolist(),
            'total_exploits': total_exploits.tolist(),
            'total_empathic_penalties': total_empathic_penalties.tolist(),
            'total_loop_events': total_loop_events.tolist(),
            'final_satisfactions': final_satisfactions,
            'final_emotions': final_emotions,
            # NEW metrics (aligned with notebook)
            'depletion_step': depletion_step,
            'avg_resources_during_episode': avg_resources_during_episode,
            'gini_consumption': gini_consumption,
            'gini_satisfaction': gini_satisfaction,
            'avg_inter_agent_distance': avg_inter_agent_distance,
            'consumption_per_agent': consumption_per_agent,
            'empathic_penalties_count': empathic_penalties_count
        }
        
        # Write summary to CSV
        write_summary_csv([summary], simulation_index, seed, filename=summary_file)
        
        # Progress updates at intervals
        if (episode + 1) % progress_interval == 0 or (episode + 1) == episode_number:
            elapsed_time = time.time() - start_time
            episodes_completed = episode + 1
            episodes_remaining = episode_number - episodes_completed
            
            if episodes_completed > 0:
                avg_time_per_episode = elapsed_time / episodes_completed
                estimated_remaining = avg_time_per_episode * episodes_remaining
                
                elapsed_str = f"{elapsed_time/60:.1f} min" if elapsed_time > 60 else f"{elapsed_time:.1f} sec"
                remaining_str = f"{estimated_remaining/60:.1f} min" if estimated_remaining > 60 else f"{estimated_remaining:.1f} sec"
                
                if condition_name:
                    print(f"\n  [{condition_name}] Episode {episodes_completed}/{episode_number} | "
                          f"Elapsed: {elapsed_str} | Est. remaining: {remaining_str}")
    
        # Final timing summary
    total_time = time.time() - start_time
    total_time_str = f"{total_time/60:.1f} minutes" if total_time > 60 else f"{total_time:.1f} seconds"
    avg_episode_time = total_time / episode_number
    avg_time_str = f"{avg_episode_time:.2f} sec/episode"
    
    if condition_name:
        print(f"\n  [{condition_name}] Completed {episode_number} episodes in {total_time_str} "
              f"(avg: {avg_time_str})")
    
    # GPU memory info (if using GPU)
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
        current_memory = torch.cuda.memory_allocated() / 1e6  # MB
        if condition_name:
            print(f"  [{condition_name}] GPU Memory: {current_memory:.1f} MB used (peak: {peak_memory:.1f} MB)")
        torch.cuda.empty_cache()  # Clear cache after each condition


# ----------------------------------------
# CSV writing functions
# ----------------------------------------

def write_summary_csv(summaries, simulation_index, seed, filename=None):
    """
    Append per-episode summary data to CSV. Write header only if file does not exist.
    Includes new metrics aligned with notebook: depletion_step, avg_resources, gini, distances.
    """
    if filename is None:
        filename = filename_definer(simulation_index, suffix="episode_summary")

    file_exists = os.path.exists(filename)

    header = (
        ["simulation_number", "seed", "episode", "total_steps", "resources_remaining",
         # NEW metrics (aligned with notebook)
         "depletion_step", "avg_resources_during_episode",
         "gini_consumption", "gini_satisfaction", "avg_inter_agent_distance",
         "empathic_penalties_count"] +
        [f"total_reward_{i}" for i in range(NB_AGENTS)] +
        [f"total_exploits_{i}" for i in range(NB_AGENTS)] +
        [f"total_empathic_penalties_{i}" for i in range(NB_AGENTS)] +
        [f"total_loop_events_{i}" for i in range(NB_AGENTS)] +
        [f"final_satisfaction_{i}" for i in range(NB_AGENTS)] +
        [f"final_emotion_{i}" for i in range(NB_AGENTS)] +
        [f"consumption_per_agent_{i}" for i in range(NB_AGENTS)]
    )

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        for rec in summaries:
            row = (
                [
                    rec['simulation_number'],
                    seed,
                    rec['episode'],
                    rec['total_steps'],
                    rec['resources_remaining'],
                    # NEW metrics (aligned with notebook)
                    rec.get('depletion_step', rec['total_steps']),
                    rec.get('avg_resources_during_episode', 0.0),
                    rec.get('gini_consumption', 0.0),
                    rec.get('gini_satisfaction', 0.0),
                    rec.get('avg_inter_agent_distance', 0.0),
                    rec.get('empathic_penalties_count', 0),
                ]
                + rec['total_rewards']
                + rec['total_exploits']
                + rec['total_empathic_penalties']
                + rec['total_loop_events']
                + rec['final_satisfactions']
                + rec['final_emotions']
                + rec.get('consumption_per_agent', [0] * NB_AGENTS)
            )
            writer.writerow(row)


# ----------------------------------------
# Filename builder
# ----------------------------------------

def filename_definer(simulation_index: int, suffix: str, empathy_alpha=None) -> str:
    """
    Build a filename with simulation parameters.
    Format: results_<sim>_<episodes>_<grid>_<agents>_<resources>_<vision>_
            <exploit_reward>_<empathic_penalty>_<alpha>_<params>_<random>_<suffix>.csv
    
    Args:
        simulation_index: Index of simulation
        suffix: Suffix for filename
        empathy_alpha: Empathy parameter to include in filename (if None, uses global EMPATHY_ALPHA)
    """
    if empathy_alpha is None:
        empathy_alpha = EMPATHY_ALPHA
    
    param_order = ["learning_rate", "gamma", "epsilon", "epsilon_decay", "epsilon_min",
                   "batch_size", "hidden_size", "update_target_every"]
    param_values = "_".join(str(PARAMS_DQN[key]) for key in param_order)
    random_suffix = ''.join(str(random.randint(0, 9)) for _ in range(6))
    
    filename = (
        f"results_{simulation_index:03d}_"
        f"{EPISODE_NUMBER}_"
        f"grid{GRID_SIZE}_"
        f"agents{NB_AGENTS}_"
        f"resources{NUM_RESOURCES}_"
        f"vision{VISION_RADIUS}_"
        f"exploit{EXPLOIT_REWARD}_"
        f"penalty{EMPATHIC_PENALTY}_"
        f"alpha{empathy_alpha}_"
        f"loop{LOOP_PENALTY}_"
        f"{param_values}_"
        f"{random_suffix}_"
        f"{suffix}.csv"
    )
    
    # Ensure no file overwriting
    base_filename = filename.replace('.csv', '')
    version = 1
    while os.path.exists(filename):
        filename = f"{base_filename}_{version}.csv"
        version += 1
    
    return filename


# ----------------------------------------
# Main entry
# ----------------------------------------

if __name__ == '__main__':
    # GPU Configuration and Verification
    print(f"\n{'='*70}")
    print("HARDWARE CONFIGURATION")
    print(f"{'='*70}")
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"✓ CUDA Available: Using GPU")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Performance optimizations
        torch.backends.cudnn.benchmark = True  # Auto-tune for best performance
        torch.backends.cudnn.enabled = True
        print(f"  cuDNN Enabled: {torch.backends.cudnn.enabled}")
        print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        
        # Clear cache and reset peak memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        DEVICE = torch.device("cpu")
        print(f"✗ CUDA Not Available: Using CPU")
        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())
        print(f"  CPU Threads: {torch.get_num_threads()}")
    
    print(f"{'='*70}\n")
    
    overall_start_time = time.time()
    # Create results folder inside the 2D folder (where this script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(script_dir, "2D_simulation_results")
    os.makedirs(folder_name, exist_ok=True)
    
    SHOW_SIMULATION_PROGRESS = True
    
    # Run two simulations: one with empathy (alpha=0.5) and one without (alpha=0.0)
    empathy_conditions = [
        (0.5, "empathic"),
        (0.0, "non_empathic")
    ]
    
    print(f"\n{'='*70}")
    print(f"Starting simulation run: {len(empathy_conditions)} conditions × {EPISODE_NUMBER} episodes each")
    print(f"{'='*70}\n")
    
    for condition_idx, (empathy_alpha, condition_name) in enumerate(empathy_conditions):
        condition_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"CONDITION {condition_idx + 1}/{len(empathy_conditions)}: {condition_name.upper()}")
        print(f"  Empathy Alpha: {empathy_alpha}")
        print(f"  Episodes: {EPISODE_NUMBER}")
        print(f"  Max Steps per Episode: {MAX_STEPS}")
        print(f"  Regeneration Mode: dynamic (fewer resources = lower respawn probability)")
        print(f"{'='*70}\n")
        
        # Use different seeds for each condition
        seed = 42 if empathy_alpha == 0.5 else 123
        np.random.seed(seed)
        random.seed(seed)
        
        summary_csv_name = filename_definer(simulation_index=0,
                                            suffix="episode_summary",
                                            empathy_alpha=empathy_alpha)
        summary_csv_path = os.path.join(folder_name, summary_csv_name)
        
        print(f"  Output file: {summary_csv_path}\n")
        
        # Run simulation
        run_simulation_with_progressive_saving(
            episode_number=EPISODE_NUMBER,
            step_count=MAX_STEPS,
            simulation_index=0,
            summary_file=summary_csv_path,
            seed=seed,
            verbose=True,
            empathy_alpha=empathy_alpha,
            condition_name=condition_name.upper()
        )
        
        condition_time = time.time() - condition_start_time
        condition_time_str = f"{condition_time/60:.1f} minutes" if condition_time > 60 else f"{condition_time:.1f} seconds"
        
        print(f"\n{'='*70}")
        print(f"✓ {condition_name.upper()} condition completed in {condition_time_str}")
        print(f"  Results saved to: {summary_csv_path}")
        print(f"{'='*70}")
        
        # Estimate remaining time
        if condition_idx < len(empathy_conditions) - 1:
            remaining_conditions = len(empathy_conditions) - condition_idx - 1
            estimated_remaining = condition_time * remaining_conditions
            estimated_str = f"{estimated_remaining/60:.1f} minutes" if estimated_remaining > 60 else f"{estimated_remaining:.1f} seconds"
            print(f"  Estimated time for remaining {remaining_conditions} condition(s): ~{estimated_str}\n")
    
    total_time = time.time() - overall_start_time
    total_time_str = f"{total_time/60:.1f} minutes" if total_time > 60 else f"{total_time:.1f} seconds"
    
    print(f"\n{'='*70}")
    print(f"🎉 ALL SIMULATIONS COMPLETED!")
    print(f"   Total time: {total_time_str}")
    print(f"   Average per condition: {total_time/len(empathy_conditions)/60:.1f} minutes")
    
    # Final GPU memory summary
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
        print(f"   GPU Peak Memory: {peak_memory:.1f} MB")
    
    print(f"{'='*70}\n")

