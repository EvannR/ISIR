"""
Main Training Script for Graph-Based Empathic Environment

Runs experiments comparing empathic vs non-empathic agents
in a graph-based resource sharing setting.
"""

import numpy as np
import random
import csv
import os
import time
import torch
from tqdm import tqdm

from env_graph import EmpathicGraphEnv
from agent_graph import GraphDQNAgent
from config import (
    NUM_EPISODES, MAX_STEPS, NUM_AGENTS, NUM_SIMULATIONS_PER_CONDITION,
    DEFAULT_GRAPH_TYPE, EDGE_PROBABILITY, SMALL_WORLD_K, SMALL_WORLD_P, MAX_NEIGHBORS,
    DYNAMIC_P_BASE, DYNAMIC_P_MAX, DYNAMIC_DELTA, DYNAMIC_DECAY,
    INITIAL_POOL, MAX_POOL_CAPACITY, REGENERATION_RATE, REGENERATION_MODE,
    EXPLOIT_REWARD, EMPATHIC_PENALTY, STEP_COST, WAIT_REWARD, FAILED_CONSUMPTION_PENALTY,
    MEMORY_SIZE, BETA, THRESHOLD, SMOOTHING, SIGMOID_GAIN,
    LEARNING_RATE, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN,
    BATCH_SIZE, HIDDEN_SIZE, UPDATE_TARGET_EVERY, REPLAY_BUFFER_CAPACITY,
    OBSERVATION_DIM, NUM_ACTIONS,
)


# ----------------------------------------
# Utility Functions
# ----------------------------------------

def compute_gini_coefficient(values):
    """
    Compute the Gini coefficient for a list of values.
    Gini = 0 means perfect equality
    Gini = 1 means perfect inequality
    """
    values = np.array(values, dtype=float)
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values) - (n + 1) * np.sum(sorted_values)) / (n * np.sum(sorted_values))


def compute_average_degree(graph):
    """Compute average node degree in the graph."""
    if graph is None:
        return 0.0
    return graph.get_average_degree()


# ----------------------------------------
# CSV Writing Functions
# ----------------------------------------

def write_summary_csv(summary, filename):
    """
    Append a single episode summary to CSV. Write header only if file does not exist.
    """
    file_exists = os.path.exists(filename)

    header = (
        ["simulation_number", "seed", "episode", "total_steps", "pool_remaining",
         "depletion_step", "avg_pool_during_episode",
         "gini_consumption", "gini_satisfaction",
         "empathic_penalties_count", "total_consumption",
         "graph_type", "graph_density", "graph_avg_degree"] +
        [f"total_reward_{i}" for i in range(NUM_AGENTS)] +
        [f"total_consumes_{i}" for i in range(NUM_AGENTS)] +
        [f"final_satisfaction_{i}" for i in range(NUM_AGENTS)] +
        [f"consumption_per_agent_{i}" for i in range(NUM_AGENTS)]
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
                summary['pool_remaining'],
                summary['depletion_step'],
                summary['avg_pool_during_episode'],
                summary['gini_consumption'],
                summary['gini_satisfaction'],
                summary['empathic_penalties_count'],
                summary['total_consumption'],
                summary['graph_type'],
                summary['graph_density'],
                summary['graph_avg_degree'],
            ]
            + summary['episode_rewards']
            + summary['consumption_per_agent']
            + summary['final_satisfactions']
            + summary['consumption_per_agent']
        )
        writer.writerow(row)


def filename_definer(empathy_alpha: float, graph_type: str, simulation_index: int = 0) -> str:
    """
    Build a filename with simulation parameters.
    """
    random_suffix = ''.join(str(random.randint(0, 9)) for _ in range(6))

    filename = (
        f"graph_results_{simulation_index:03d}_{NUM_EPISODES}_"
        f"agents{NUM_AGENTS}_"
        f"pool{INITIAL_POOL}_"
        f"regen{REGENERATION_MODE}_"
        f"graph{graph_type}_"
        f"alpha{empathy_alpha}_"
        f"{LEARNING_RATE}_{GAMMA}_{EPSILON}_{EPSILON_DECAY}_{EPSILON_MIN}_"
        f"{BATCH_SIZE}_{HIDDEN_SIZE}_{UPDATE_TARGET_EVERY}_"
        f"{random_suffix}_"
        f"episode_summary.csv"
    )

    return filename


# ----------------------------------------
# Main Experiment Function
# ----------------------------------------

def run_experiment(
    empathy_alpha: float,
    seed: int,
    summary_file: str,
    simulation_index: int = 0,
    graph_type: str = 'random',
    show_progress: bool = True
):
    """
    Run experiment with graph-based environment.

    Args:
        empathy_alpha: 0.0 for non-empathic, 0.5 for empathic
        seed: Random seed for this simulation
        summary_file: Path to output CSV file
        simulation_index: Index of this simulation
        graph_type: Type of social graph
        show_progress: Show tqdm progress bar
    """
    # Set all random seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    start_time = time.time()

    # Create environment
    env = EmpathicGraphEnv(
        num_agents=NUM_AGENTS,
        max_neighbors=MAX_NEIGHBORS,
        initial_pool=INITIAL_POOL,
        max_pool_capacity=MAX_POOL_CAPACITY,
        regeneration_rate=REGENERATION_RATE,
        regeneration_mode=REGENERATION_MODE,
        graph_type=graph_type,
        edge_probability=EDGE_PROBABILITY,
        small_world_k=SMALL_WORLD_K,
        small_world_p=SMALL_WORLD_P,
        dynamic_p_base=DYNAMIC_P_BASE,
        dynamic_p_max=DYNAMIC_P_MAX,
        dynamic_delta=DYNAMIC_DELTA,
        dynamic_decay=DYNAMIC_DECAY,
        memory_size=MEMORY_SIZE,
        beta=BETA,
        threshold=THRESHOLD,
        smoothing=SMOOTHING,
        sigmoid_gain=SIGMOID_GAIN,
        exploit_reward=EXPLOIT_REWARD,
        empathic_penalty=EMPATHIC_PENALTY,
        empathy_alpha=empathy_alpha,
        step_cost=STEP_COST,
        wait_reward=WAIT_REWARD,
        failed_consumption_penalty=FAILED_CONSUMPTION_PENALTY,
        max_steps=MAX_STEPS,
    )

    # Create DQN agents
    dqn_agents = []
    for agent_id in range(NUM_AGENTS):
        agent = GraphDQNAgent(
            agent_id=agent_id,
            observation_dim=OBSERVATION_DIM,
            action_size=NUM_ACTIONS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
            epsilon_min=EPSILON_MIN,
            batch_size=BATCH_SIZE,
            hidden_size=HIDDEN_SIZE,
            update_target_every=UPDATE_TARGET_EVERY,
            replay_capacity=REPLAY_BUFFER_CAPACITY,
            memory_size=MEMORY_SIZE,
            beta=BETA,
            threshold=THRESHOLD,
            smoothing=SMOOTHING,
            sigmoid_gain=SIGMOID_GAIN,
        )
        dqn_agents.append(agent)

    # Verify GPU usage
    if len(dqn_agents) > 0:
        device = next(dqn_agents[0].policy_network.parameters()).device
        print(f"  Agents running on: {device}")

    # Link agents to environment
    env.set_agents(dqn_agents)

    # Progress bar
    episode_iter = tqdm(range(NUM_EPISODES), desc="Training", disable=not show_progress)
    progress_interval = max(1, NUM_EPISODES // 10)

    for episode in episode_iter:
        # Episode-specific seed
        episode_seed = seed + episode * 1000
        obs, _ = env.reset(seed=episode_seed)

        for agent_idx, dqn_agent in enumerate(dqn_agents):
            dqn_agent.reset(observation=obs[agent_idx])

        # Episode tracking
        episode_consumption = [0] * NUM_AGENTS
        episode_rewards = [0.0] * NUM_AGENTS
        pool_levels_during_episode = []
        depletion_step = MAX_STEPS
        first_depletion_recorded = False
        empathic_penalty_count = 0

        for step in range(MAX_STEPS):
            # Track pool level BEFORE step
            pool_levels_during_episode.append(env.pool.get_level())

            # Collect actions
            actions = []
            for agent_idx, dqn_agent in enumerate(dqn_agents):
                action = dqn_agent.select_action(obs[agent_idx], training=True)
                actions.append(action)

            # Step environment
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # Check depletion AFTER step
            if info['pool_level'] == 0 and not first_depletion_recorded:
                depletion_step = step
                first_depletion_recorded = True

            # Track pool level AFTER step
            pool_levels_during_episode.append(info['pool_level'])

            # Track consumption and penalties
            for agent_idx in range(NUM_AGENTS):
                if info['consumed'][agent_idx]:
                    episode_consumption[agent_idx] += 1
                if info['empathic_penalty_events'][agent_idx]:
                    empathic_penalty_count += 1
                episode_rewards[agent_idx] += rewards[agent_idx]

            # Update agents
            for agent_idx, dqn_agent in enumerate(dqn_agents):
                dqn_agent.step(next_obs[agent_idx], rewards[agent_idx], done)

            obs = next_obs
            if done:
                break

        # Compute episode metrics
        final_satisfactions = [dqn_agent.get_satisfaction() for dqn_agent in dqn_agents]

        summary = {
            'simulation_number': simulation_index,
            'seed': seed,
            'episode': episode,
            'total_steps': step + 1,
            'pool_remaining': env.pool.get_level(),
            'depletion_step': depletion_step,
            'avg_pool_during_episode': np.mean(pool_levels_during_episode),
            'gini_consumption': compute_gini_coefficient(episode_consumption),
            'gini_satisfaction': compute_gini_coefficient(final_satisfactions),
            'empathic_penalties_count': empathic_penalty_count,
            'total_consumption': sum(episode_consumption),
            'graph_type': graph_type,
            'graph_density': env.graph.get_density(),
            'graph_avg_degree': env.graph.get_average_degree(),
            'episode_rewards': episode_rewards,
            'consumption_per_agent': episode_consumption,
            'final_satisfactions': final_satisfactions,
        }

        # Save to CSV progressively
        write_summary_csv(summary, summary_file)

        # Update progress bar
        if show_progress:
            episode_iter.set_postfix({
                'Pool': env.pool.get_level(),
                'Gini': f"{summary['gini_consumption']:.2f}",
                'Eps': f"{dqn_agents[0].epsilon:.2f}"
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
# Main Entry
# ----------------------------------------

if __name__ == '__main__':
    # GPU Configuration
    print(f"\n{'='*70}")
    print("HARDWARE CONFIGURATION")
    print(f"{'='*70}")

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"CUDA Available: Using GPU")
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
        print(f"CUDA Not Available: Using CPU")
        torch.set_num_threads(torch.get_num_threads())
        print(f"  CPU Threads: {torch.get_num_threads()}")

    print(f"{'='*70}\n")

    # Create results folder (use dedicated subfolder for dynamic graph for comparison notebook)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_name = os.path.join(
        script_dir,
        "graph_simulation_results_dynamic" if DEFAULT_GRAPH_TYPE == "dynamic" else "graph_simulation_results",
    )
    os.makedirs(folder_name, exist_ok=True)

    overall_start_time = time.time()

    # Conditions
    empathy_conditions = [
        (0.0, "non_empathic", 42),
        (0.5, "empathic", 123)
    ]

    graph_type = DEFAULT_GRAPH_TYPE

    print(f"\n{'='*70}")
    print(f"GRAPH-BASED SIMULATION CONFIGURATION")
    print(f"{'='*70}")
    print(f"  Agents: {NUM_AGENTS}")
    print(f"  Graph Type: {graph_type}")
    print(f"  Edge Probability: {EDGE_PROBABILITY}")
    print(f"  Max Neighbors (obs): {MAX_NEIGHBORS}")
    print(f"  Initial Pool: {INITIAL_POOL}")
    print(f"  Regeneration Mode: {REGENERATION_MODE}")
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
            summary_csv_name = filename_definer(
                empathy_alpha=empathy_alpha,
                graph_type=graph_type,
                simulation_index=sim_idx
            )
            summary_csv_path = os.path.join(folder_name, summary_csv_name)

            print(f"    Output file: {summary_csv_path}\n")

            # Run experiment
            run_experiment(
                empathy_alpha=empathy_alpha,
                seed=simulation_seed,
                summary_file=summary_csv_path,
                simulation_index=sim_idx,
                graph_type=graph_type,
                show_progress=True
            )

            sim_time = time.time() - sim_start_time
            sim_time_str = f"{sim_time/60:.1f} minutes" if sim_time > 60 else f"{sim_time:.1f} seconds"

            print(f"\n    Simulation {sim_idx + 1}/{NUM_SIMULATIONS_PER_CONDITION} completed in {sim_time_str}")
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
        print(f"{condition_name.upper()} condition completed in {condition_time_str}")
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
    print(f"ALL GRAPH SIMULATIONS COMPLETED!")
    print(f"   Total simulations: {total_simulations}")
    print(f"   Total time: {total_time_str}")
    print(f"   Average per condition: {total_time/len(empathy_conditions)/60:.1f} minutes")
    print(f"   Average per simulation: {total_time/total_simulations/60:.1f} minutes")

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e6
        print(f"   GPU Peak Memory: {peak_memory:.1f} MB")

    print(f"{'='*70}\n")
