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
# Utility functions (from notebook)
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

    sorted_values = np.sort(values)
    n = len(sorted_values)
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
# Constants (matching notebook)
# ----------------------------------------

NUM_EPISODES = 300
MAX_STEPS = 200
NB_AGENTS = 5              # Notebook uses 5 agents
GRID_SIZE = 5
NUM_RESOURCES = 10
VISION_RADIUS = 2
NUM_SIMULATIONS_PER_CONDITION = 3  # Number of independent simulations per condition

# Environment reward settings (matching notebook)
EXPLOIT_REWARD = 1.0
EMPATHIC_PENALTY = -1.0    # Notebook uses -2.0
STEP_COST = -0.01
LOOP_PENALTY = -0.05

# DQN hyperparameters (matching notebook)
LEARNING_RATE = 0.002
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
BATCH_SIZE = 32
HIDDEN_SIZE = 128
UPDATE_TARGET_EVERY = 20


# ----------------------------------------
# CSV writing functions (from main_2d.py)
# ----------------------------------------

def write_summary_csv(summary, filename):
    """
    Append a single episode summary to CSV. Write header only if file does not exist.
    """
    file_exists = os.path.exists(filename)

    header = (
        ["simulation_number", "seed", "episode", "total_steps", "resources_remaining",
         "depletion_step", "avg_resources_during_episode",
         "gini_consumption", "gini_satisfaction", "avg_inter_agent_distance",
         "empathic_penalties_count", "total_consumption"] +
        [f"total_reward_{i}" for i in range(NB_AGENTS)] +
        [f"total_exploits_{i}" for i in range(NB_AGENTS)] +
        [f"final_satisfaction_{i}" for i in range(NB_AGENTS)] +
        [f"consumption_per_agent_{i}" for i in range(NB_AGENTS)]
    )

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        row = (
            [
                summary['simulation_number'],
                summary['seed'],
                summary['episode'],
                summary['total_steps'],
                summary['resources_remaining'],
                summary['depletion_step'],
                summary['avg_resources_during_episode'],
                summary['gini_consumption'],
                summary['gini_satisfaction'],
                summary['avg_inter_agent_distance'],
                summary['empathic_penalties_count'],
                summary['total_consumption'],
            ]
            + summary['episode_rewards']
            + summary['consumption_per_agent']
            + summary['final_satisfactions']
            + summary['consumption_per_agent']
        )
        writer.writerow(row)


def filename_definer(empathy_alpha: float, simulation_index: int = 0) -> str:
    """
    Build a filename with simulation parameters matching notebook configuration.
    
    Args:
        empathy_alpha: Empathy parameter value
        simulation_index: Index of the simulation (0-based)
    """
    random_suffix = ''.join(str(random.randint(0, 9)) for _ in range(6))

    filename = (
        f"results_{simulation_index:03d}_{NUM_EPISODES}_"
        f"grid{GRID_SIZE}_"
        f"agents{NB_AGENTS}_"
        f"resources{NUM_RESOURCES}_"
        f"vision{VISION_RADIUS}_"
        f"exploit{EXPLOIT_REWARD}_"
        f"penalty{EMPATHIC_PENALTY}_"
        f"alpha{empathy_alpha}_"
        f"loop{LOOP_PENALTY}_"
        f"{LEARNING_RATE}_{GAMMA}_{EPSILON}_{EPSILON_DECAY}_{EPSILON_MIN}_"
        f"{BATCH_SIZE}_{HIDDEN_SIZE}_{UPDATE_TARGET_EVERY}_"
        f"{random_suffix}_"
        f"episode_summary.csv"
    )

    return filename


# ----------------------------------------
# Main experiment function (notebook logic)
# ----------------------------------------

def run_experiment(empathy_alpha, seed, summary_file, simulation_index=0, show_progress=True):
    """
    Run experiment with notebook logic but save to CSV progressively.

    Args:
        empathy_alpha: 0.0 for non-empathic, 0.5 for empathic
        seed: Random seed for this simulation
        summary_file: Path to output CSV file
        simulation_index: Index of this simulation (for tracking in CSV)
        show_progress: Show tqdm progress bar
    """
    # Set all random seeds (matching notebook)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    start_time = time.time()

    # Create environment (notebook style - minimal parameters)
    env = EmpathicGridEnv(
        grid_size=GRID_SIZE,
        num_agents=NB_AGENTS,
        num_resources=NUM_RESOURCES,
        vision_radius=VISION_RADIUS,
        exploit_reward=EXPLOIT_REWARD,
        empathic_penalty=EMPATHIC_PENALTY,
        empathy_alpha=empathy_alpha,
        step_cost=STEP_COST,
        loop_penalty=LOOP_PENALTY,
        max_steps=MAX_STEPS,
        regeneration_mode='dynamic'
    )

    # Create DQN agents (notebook style - minimal parameters)
    dqn_agents = []
    for agent_id in range(NB_AGENTS):
        agent = DQNAgent(
            agent_id=agent_id,
            state_size=env.observation_space.shape,
            action_size=env.action_space.n,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
            epsilon_min=EPSILON_MIN,
            batch_size=BATCH_SIZE,
            hidden_size=HIDDEN_SIZE,
            update_target_every=UPDATE_TARGET_EVERY
        )
        dqn_agents.append(agent)

    # Verify GPU usage
    if len(dqn_agents) > 0:
        device = next(dqn_agents[0].policy_network.parameters()).device
        print(f"  Agents running on: {device}")

    # Link agents to environment
    env.agents = [agent.agent for agent in dqn_agents]

    # Progress bar
    episode_iter = tqdm(range(NUM_EPISODES), desc="Training", disable=not show_progress)
    progress_interval = max(1, NUM_EPISODES // 10)

    for episode in episode_iter:
        # Episode-specific seed (matching notebook)
        episode_seed = seed + episode * 1000
        obs, _ = env.reset(seed=episode_seed)

        for agent_idx, dqn_agent in enumerate(dqn_agents):
            dqn_agent.reset(observation=obs[agent_idx])

        # Episode tracking
        episode_consumption = [0] * NB_AGENTS
        episode_rewards = [0.0] * NB_AGENTS
        episode_distances = []
        resource_counts_during_episode = []
        depletion_step = MAX_STEPS
        first_depletion_recorded = False
        empathic_penalty_count = 0

        for step in range(MAX_STEPS):
            # Track inter-agent distance
            episode_distances.append(compute_average_inter_agent_distance(env.agent_positions))

            # Track resources BEFORE step
            resource_counts_during_episode.append(len(env.resource_positions))

            # Collect actions
            actions = []
            for agent_idx, dqn_agent in enumerate(dqn_agents):
                valid_actions = env.get_valid_actions(agent_idx)
                action = dqn_agent.select_action(obs[agent_idx], training=True, valid_actions=valid_actions)
                actions.append(action)

            # Step environment
            next_obs, rewards, done, truncated, info = env.step(actions)

            # Check depletion AFTER step (notebook fix)
            if info['resources_remaining'] == 0 and not first_depletion_recorded:
                depletion_step = step
                first_depletion_recorded = True

            # Track resources AFTER step
            resource_counts_during_episode.append(info['resources_remaining'])

            # Track consumption and penalties
            for agent_idx in range(NB_AGENTS):
                if info['consumed'][agent_idx]:
                    episode_consumption[agent_idx] += 1
                if info['empathic_penalty_events'][agent_idx]:
                    empathic_penalty_count += 1
                episode_rewards[agent_idx] += rewards[agent_idx]

            # Update agents
            for agent_idx, dqn_agent in enumerate(dqn_agents):
                valid_actions_next = env.get_valid_actions(agent_idx)
                dqn_agent.step(next_obs[agent_idx], rewards[agent_idx], done, valid_actions=valid_actions_next)

            obs = next_obs
            if done:
                break

        # Compute episode metrics
        final_satisfactions = [agent.get_satisfaction() for agent in env.agents]

        summary = {
            'simulation_number': simulation_index,
            'seed': seed,
            'episode': episode,
            'total_steps': step + 1,
            'resources_remaining': len(env.resource_positions),
            'depletion_step': depletion_step,
            'avg_resources_during_episode': np.mean(resource_counts_during_episode),
            'gini_consumption': compute_gini_coefficient(episode_consumption),
            'gini_satisfaction': compute_gini_coefficient(final_satisfactions),
            'avg_inter_agent_distance': np.mean(episode_distances),
            'empathic_penalties_count': empathic_penalty_count,
            'total_consumption': sum(episode_consumption),
            'episode_rewards': episode_rewards,
            'consumption_per_agent': episode_consumption,
            'final_satisfactions': final_satisfactions,
        }

        # Save to CSV progressively
        write_summary_csv(summary, summary_file)

        # Update progress bar
        if show_progress:
            episode_iter.set_postfix({
                'Resources': len(env.resource_positions),
                'Gini': f"{summary['gini_consumption']:.2f}",
                'ε': f"{dqn_agents[0].epsilon:.2f}"
            })

        # Progress updates at intervals
        if (episode + 1) % progress_interval == 0:
            elapsed_time = time.time() - start_time
            episodes_completed = episode + 1
            episodes_remaining = NUM_EPISODES - episodes_completed
            avg_time_per_episode = elapsed_time / episodes_completed
            estimated_remaining = avg_time_per_episode * episodes_remaining

            elapsed_str = f"{elapsed_time/60:.1f} min" if elapsed_time > 60 else f"{elapsed_time:.1f} sec"
            remaining_str = f"{estimated_remaining/60:.1f} min" if estimated_remaining > 60 else f"{estimated_remaining:.1f} sec"

            print(f"\n  Episode {episodes_completed}/{NUM_EPISODES} | "
                  f"Elapsed: {elapsed_str} | Est. remaining: {remaining_str}")

    # Final timing
    total_time = time.time() - start_time
    total_time_str = f"{total_time/60:.1f} minutes" if total_time > 60 else f"{total_time:.1f} seconds"
    print(f"\n  Completed {NUM_EPISODES} episodes in {total_time_str}")

    # Clear GPU cache
    if torch.cuda.is_available():
        print(f"  Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")
        torch.cuda.empty_cache()


# ----------------------------------------
# Main entry
# ----------------------------------------

if __name__ == '__main__':
    # GPU Configuration
    print(f"\n{'='*70}")
    print("HARDWARE CONFIGURATION")
    print(f"{'='*70}")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"✓ CUDA Available: Using GPU")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print(f"  cuDNN Enabled: {torch.backends.cudnn.enabled}")
        print(f"  cuDNN Benchmark: {torch.backends.cudnn.benchmark}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        DEVICE = torch.device("cpu")
        print(f"✗ CUDA Not Available: Using CPU")
        torch.set_num_threads(torch.get_num_threads())
        print(f"  CPU Threads: {torch.get_num_threads()}")

    print(f"{'='*70}\n")

    # Create results folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(script_dir, "2D_simulation_results")
    os.makedirs(folder_name, exist_ok=True)

    overall_start_time = time.time()

    # Conditions (matching notebook: Non-empathic=42, Empathic=123)
    empathy_conditions = [
        (0.0, "non_empathic", 42),
        (0.5, "empathic", 123)
    ]

    print(f"\n{'='*70}")
    print(f"SIMULATION CONFIGURATION (matching notebook)")
    print(f"{'='*70}")
    print(f"  Agents: {NB_AGENTS}")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Resources: {NUM_RESOURCES}")
    print(f"  Empathic Penalty: {EMPATHIC_PENALTY}")
    print(f"  Episodes per simulation: {NUM_EPISODES}")
    print(f"  Max Steps per episode: {MAX_STEPS}")
    print(f"  Simulations per condition: {NUM_SIMULATIONS_PER_CONDITION}")
    print(f"  Total simulations: {len(empathy_conditions) * NUM_SIMULATIONS_PER_CONDITION}")
    print(f"{'='*70}\n")

    for condition_idx, (empathy_alpha, condition_name, base_seed) in enumerate(empathy_conditions):
        condition_start_time = time.time()

        print(f"\n{'='*70}")
        print(f"CONDITION {condition_idx + 1}/{len(empathy_conditions)}: {condition_name.upper()}")
        print(f"  Empathy Alpha: {empathy_alpha}")
        print(f"  Base Seed: {base_seed}")
        print(f"  Simulations: {NUM_SIMULATIONS_PER_CONDITION}")
        print(f"{'='*70}\n")

        # Run multiple simulations for this condition
        for sim_idx in range(NUM_SIMULATIONS_PER_CONDITION):
            sim_start_time = time.time()
            
            # Generate unique seed for this simulation
            simulation_seed = base_seed + sim_idx * 1000
            
            print(f"\n  {'-'*68}")
            print(f"  SIMULATION {sim_idx + 1}/{NUM_SIMULATIONS_PER_CONDITION} ({condition_name.upper()})")
            print(f"    Simulation Index: {sim_idx}")
            print(f"    Seed: {simulation_seed}")
            print(f"  {'-'*68}\n")

            # Generate filename and path
            summary_csv_name = filename_definer(empathy_alpha=empathy_alpha, simulation_index=sim_idx)
            summary_csv_path = os.path.join(folder_name, summary_csv_name)

            print(f"    Output file: {summary_csv_path}\n")

            # Run experiment
            run_experiment(
                empathy_alpha=empathy_alpha,
                seed=simulation_seed,
                summary_file=summary_csv_path,
                simulation_index=sim_idx,
                show_progress=True
            )

            sim_time = time.time() - sim_start_time
            sim_time_str = f"{sim_time/60:.1f} minutes" if sim_time > 60 else f"{sim_time:.1f} seconds"

            print(f"\n    ✓ Simulation {sim_idx + 1}/{NUM_SIMULATIONS_PER_CONDITION} completed in {sim_time_str}")
            print(f"      Results saved to: {summary_csv_path}")

            # Estimate remaining time for this condition
            if sim_idx < NUM_SIMULATIONS_PER_CONDITION - 1:
                remaining_sims = NUM_SIMULATIONS_PER_CONDITION - sim_idx - 1
                estimated_remaining = sim_time * remaining_sims
                estimated_str = f"{estimated_remaining/60:.1f} minutes" if estimated_remaining > 60 else f"{estimated_remaining:.1f} seconds"
                print(f"      Estimated time for remaining {remaining_sims} simulation(s): ~{estimated_str}")

        condition_time = time.time() - condition_start_time
        condition_time_str = f"{condition_time/60:.1f} minutes" if condition_time > 60 else f"{condition_time:.1f} seconds"

        print(f"\n{'='*70}")
        print(f"✓ {condition_name.upper()} condition completed in {condition_time_str}")
        print(f"  {NUM_SIMULATIONS_PER_CONDITION} simulation(s) completed")
        print(f"{'='*70}")

        # Estimate remaining time
        if condition_idx < len(empathy_conditions) - 1:
            remaining_conditions = len(empathy_conditions) - condition_idx - 1
            estimated_remaining = condition_time * remaining_conditions
            estimated_str = f"{estimated_remaining/60:.1f} minutes" if estimated_remaining > 60 else f"{estimated_remaining:.1f} seconds"
            print(f"  Estimated time for remaining {remaining_conditions} condition(s): ~{estimated_str}\n")

    total_time = time.time() - overall_start_time
    total_time_str = f"{total_time/60:.1f} minutes" if total_time > 60 else f"{total_time:.1f} seconds"

    total_simulations = len(empathy_conditions) * NUM_SIMULATIONS_PER_CONDITION
    
    print(f"\n{'='*70}")
    print(f"ALL SIMULATIONS COMPLETED!")
    print(f"   Total simulations: {total_simulations}")
    print(f"   Total time: {total_time_str}")
    print(f"   Average per condition: {total_time/len(empathy_conditions)/60:.1f} minutes")
    print(f"   Average per simulation: {total_time/total_simulations/60:.1f} minutes")

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e6
        print(f"   GPU Peak Memory: {peak_memory:.1f} MB")

    print(f"{'='*70}\n")
