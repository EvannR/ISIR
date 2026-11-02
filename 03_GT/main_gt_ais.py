"""
main entry point for the AIS-2026 revision experiments (02_GT_AIS).

derived from empathy_tragedy/game_theoretic_matrix_setup/main_game_theoretic_new_version.py,
restructured around named EXPERIMENTS that map one-to-one onto the reviewers'
requests (see EXPERIMENT_TRACKER.md):

  exp1_resource_obs        : 2x2 (see_emotions x alpha) with see_resource=True   [R1 main fix]
  exp2_replication_no_res  : 2x2 with see_resource=False (paper conditions,
                             corrected training)                                  [links new to old results]
  exp3_fixed_baselines     : random / always_exploit non-learning references     [R1+R2 baselines]
  exp4_inequity_aversion   : Hughes et al. 2018 subjective reward                [R2 literature baseline]
  exp5_gamma_robustness    : conditions A and D at gamma in {0.9, 0.999}         [R2 robustness]
  exp6_graded_alpha        : paper Study 2 (alpha in {0,.25,.5,.75,.99}) with
                             resource observation and corrected training         [dose-response re-run]
  exp7_social_discount     : 03_GT, empathic stream discounted with its own
                             gamma_others (two-head DQN) + matched-alpha controls [horizon of empathy]

other changes vs. the manuscript runs:
  - replications controlled from the command line (--runs, default 1); overall
    target 20 seeds/condition (R2), split across machines via --sim-start
  - epsilon decays per EPISODE (runner calls agent.end_episode())
  - replay buffers persist across episodes (env no longer clears them)
  - per-episode meal counts and Gini coefficient written in the summary CSVs
  - human-readable filenames + one JSON config sidecar per run

usage:
    python main_gt_ais.py                            # 1 replication/condition, experiments in RUN_EXPERIMENTS
    python main_gt_ais.py --runs 7 --sim-start 0     # machine 1 (seeds/simulations 0..6)
    python main_gt_ais.py --runs 7 --sim-start 7     # machine 2 (simulations 7..13)
    python main_gt_ais.py --runs 6 --sim-start 14    # machine 3 (simulations 14..19)
    python main_gt_ais.py --exp exp1_resource_obs    # restrict to specific experiments
    python main_gt_ais.py --eval-every 0             # disable the greedy evaluation episodes
    python main_gt_ais.py --smoke                    # 2-episode sanity check of every condition

every simulation index maps to a unique deterministic seed, so different
machines running different --sim-start ranges never collide; existing summary
CSVs are skipped, so the same command can be relaunched after an interruption.

every --eval-every training episodes (default 10), one extra GREEDY episode
(epsilon = 0, no learning) is run and logged with phase='eval' in the summary
CSV: these rows measure the learned policy without exploration noise and are
the ones to use for the final analyses. 03_GT also evaluates the untrained
policy once before training (episode = -1) and after each of the first
--eval-dense-first episodes (default 20), so the early learning phase is visible.

for DQN conditions, the initial and final policy-network weights of the 6
agents are saved next to the summary CSV as <run>_weights.pt (see save_weights).

output goes to  02_GT_AIS/results/<experiment>/  (smoke: 02_GT_AIS/results_smoke/).
"""

from env_gt_ais import GameTheoreticEnv
from agent_policies_gt_ais import QAgent, DQNAgent, FixedPolicyAgent
import argparse
import numpy as np
import random
import csv
import json
import os
from tqdm import tqdm
import torch


# ----------------------------------------
# constants for the simulation
# ----------------------------------------

NUM_RUNS_PER_CONDITION = 3   # default replications per launch; override with --runs.
                             # 03_GT first batch: 3 seeds/condition (pilot); the
                             # 20-seed target can later be split across machines
                             # via --sim-start
EVAL_EVERY_DEFAULT = 10      # greedy eval episode every N training episodes (0 = off)
EVAL_DENSE_FIRST = 20        # also eval after EACH of the first N training episodes,
                             # plus once before any training (logged as episode -1):
                             # the greedy policy converges within ~10 episodes, so
                             # the sparse schedule alone misses the learning phase
EPISODE_NUMBER = 500         # number of episodes per simulation
NB_AGENTS = 6
MAX_STEPS = 1000             # number of steps per episode
INITIAL_RESOURCES = 500      # resources at the beginning of each episode
ENVIRONMENT_TYPE = "stochastic"  # 'deterministic' or 'stochastic'

# emotion settings (unchanged from the manuscript runs)
EMOTION_TYPE = "vector"    # 'average' (one slot: mean of the others) or 'vector'
                           # (03_GT default since 02/10/2026: one slot per other agent,
                           # fixed order = agent index without self, independent noise
                           # draw per slot at precision rho). the empathic REWARD term
                           # is still the mean of the others' emotions.
BETA = 0.5                 # valuation of last meal
SMOOTHING = 'linear'       # "sigmoid" OR "linear"
SIGMOID_GAIN = 5.0
THRESHOLD = 0.5            # ratio of reward in meal history for the emotion to be neutral
EMOTION_DECIMALS = 2       # emotion's number of decimals

INCLUDE_TIME = False       # optional time slot in the observation (R1 mentions it as an alternative proxy)

# 03_GT: 'homeostatic' (r in [0,1] for every alpha, meal value decreasing with
# satiety), 'homeostatic_restraint' (same personal term, empathic term = bonus
# for abstaining while better off than the others; default since 05/10/2026,
# the first exp1 batch used 'homeostatic') or 'legacy_ais2026' (manuscript
# formula) — see agent_policies_gt_ais
REWARD_SCHEME = "homeostatic_restraint"

# RL agent hyperparameters
# NOTE epsilon_decay is now applied PER EPISODE: 0.99^460 ~ 0.01, so exploration
# anneals over ~460 of the 500 episodes instead of dying within episode 1.
PARAMS_DQN = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.99,
    "epsilon_min": 0.01,
    "batch_size": 64,
    "hidden_size": 64,
    "update_target_every": 10,
    "buffer_capacity": 100000
}
PARAMS_QLEARNING = {
    "learning_rate": 0.1,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_decay": 0.99,
    "epsilon_min": 0.01
}

# Hughes et al. 2018 inequity-aversion parameters (advantageous-only by default)
PARAMS_IA = {
    "alpha_ia": 0.0,
    "beta_ia": 0.5,
    "lambda_ia": 0.95
}


# ----------------------------------------
# experiment definitions
# each condition is a dict; missing keys fall back to defaults in make_condition()
# ----------------------------------------

def make_condition(label, agent="DQN", see_resource=True, see_emotions=True,
                   see_own_meals=True, alpha=0.5, gamma=None, gamma_others=None,
                   reward_mode="social", fixed_policy=None, emotion_precision=None):
    if gamma_others is not None and (agent != "DQN" or reward_mode != "social"):
        raise ValueError("gamma_others needs agent='DQN' and reward_mode='social'")
    return {
        "label": label,
        "agent": agent,                # 'DQN', 'QLearning', 'random', 'always_exploit', 'always_abstain'
        "see_resource": see_resource,
        "see_emotions": see_emotions,
        "emotion_precision": emotion_precision,  # rho in [0,1]; None -> 1.0 if see_emotions else 0.0
        "see_own_meals": see_own_meals,  # own recent consumption rate in obs (paper: absent)
        "alpha": alpha,
        "gamma": gamma,                # None -> default from PARAMS_*
        "gamma_others": gamma_others,  # None -> one Q head, one gamma (AIS model);
                                       # else two-head DQN, empathic stream discounted with it
        "reward_mode": reward_mode,
        "fixed_policy": fixed_policy,  # set automatically for fixed agents
    }


EXPERIMENTS = {
    # R1 main fix: same 2x2 as the manuscript, but agents also observe r_t / r_0
    "exp1_resource_obs": [
        make_condition("A_res_blind_non_empathic",    see_resource=True, see_emotions=False, alpha=0.0),
        make_condition("B_res_blind_reward_empathic", see_resource=True, see_emotions=False, alpha=0.5),
        make_condition("C_res_sees_ignores",          see_resource=True, see_emotions=True,  alpha=0.0),
        make_condition("D_res_full_empathy",          see_resource=True, see_emotions=True,  alpha=0.5),
    ],
    # paper conditions re-run under the corrected training regime (buffer kept,
    # epsilon per episode, 20 seeds) — bridges old and new results.
    # see_own_meals=False too: the paper's agents observed NOTHING but o_i.
    "exp2_replication_no_res": [
        make_condition("A_nores_blind_non_empathic",    see_resource=False, see_emotions=False, see_own_meals=False, alpha=0.0),
        make_condition("B_nores_blind_reward_empathic", see_resource=False, see_emotions=False, see_own_meals=False, alpha=0.5),
        make_condition("C_nores_sees_ignores",          see_resource=False, see_emotions=True,  see_own_meals=False, alpha=0.0),
        make_condition("D_nores_full_empathy",          see_resource=False, see_emotions=True,  see_own_meals=False, alpha=0.5),
    ],
    # non-learning reference points
    "exp3_fixed_baselines": [
        make_condition("random_policy",  agent="random",         alpha=0.0),
        make_condition("always_exploit", agent="always_exploit", alpha=0.0),
    ],
    # Hughes et al. 2018 baseline: same observations as D_res, IA subjective reward
    "exp4_inequity_aversion": [
        make_condition("inequity_aversion", see_resource=True, see_emotions=True,
                       alpha=0.0, reward_mode="inequity_aversion"),
    ],
    # R2: does the fairness reversal persist under different discounting?
    "exp5_gamma_robustness": [
        make_condition("A_res_gamma090",  see_resource=True, see_emotions=False, alpha=0.0, gamma=0.9),
        make_condition("D_res_gamma090",  see_resource=True, see_emotions=True,  alpha=0.5, gamma=0.9),
        make_condition("A_res_gamma0999", see_resource=True, see_emotions=False, alpha=0.0, gamma=0.999),
        make_condition("D_res_gamma0999", see_resource=True, see_emotions=True,  alpha=0.5, gamma=0.999),
        # myopic regime: effective horizon 1/(1-gamma) = 5 steps, i.e. BELOW the
        # 10-step meal_history window the emotion signal is computed over.
        # the empathic reward is an immediate term, so it needs no foresight:
        # if the empathy effect survives here, it cannot be attributed to agents
        # being artificially far-sighted. appended LAST on purpose — condition_idx
        # feeds the seed formula, so never insert above existing conditions.
        make_condition("A_res_gamma080",  see_resource=True, see_emotions=False, alpha=0.0, gamma=0.8),
        make_condition("D_res_gamma080",  see_resource=True, see_emotions=True,  alpha=0.5, gamma=0.8),
    ],
    # re-run of the paper's Study 2 (graded empathic weight, observation always on)
    # under the corrected regime, with resource observation
    "exp6_graded_alpha": [
        make_condition("alpha000_res", see_resource=True, see_emotions=True, alpha=0.0),
        make_condition("alpha025_res", see_resource=True, see_emotions=True, alpha=0.25),
        make_condition("alpha050_res", see_resource=True, see_emotions=True, alpha=0.5),
        make_condition("alpha075_res", see_resource=True, see_emotions=True, alpha=0.75),
        make_condition("alpha099_res", see_resource=True, see_emotions=True, alpha=0.99),
    ],
}

# 03_GT: social discount factor (analysis/analysis_01_social_discount_factor.ipynb).
# every condition uses the two-head DQN (gamma_others set) so the architecture is
# identical across the grid, including the gamma_others = gamma reference.
GAMMA_SELF_EXP7 = PARAMS_DQN["gamma"]
ALPHA_EXP7 = 0.5
GAMMA_OTHERS_EXP7 = [0.99, 0.95, 0.9, 0.8, 0.0, 0.995, 0.999]


def matched_alpha(alpha, gamma_self, gamma_others):
    """
    alpha' giving a single-gamma agent the same TOTAL empathic share as
    (alpha, gamma_others), on a constant reward stream:
        alpha' = alpha/(1-g_o) / ((1-alpha)/(1-g_s) + alpha/(1-g_o)).
    lowering alpha cuts empathy at every horizon, lowering gamma_others only in
    the future: the matched control separates "less empathy" from "shorter empathy".
    """
    w_self = (1.0 - alpha) / (1.0 - gamma_self)
    w_others = alpha / (1.0 - gamma_others)
    return round(w_others / (w_self + w_others), 4)


def _go_tag(g):
    return f"{int(round(g * 1000)):04d}"   # 0.95 -> '0950', 0.999 -> '0999'


EXPERIMENTS["exp7_social_discount"] = (
    # manipulation: alpha fixed, horizon of the empathic stream varied
    [make_condition(f"D_res_go{_go_tag(g)}", see_resource=True, see_emotions=True,
                    alpha=ALPHA_EXP7, gamma_others=g)
     for g in GAMMA_OTHERS_EXP7]
    # controls: single horizon (gamma_others = gamma), alpha matched on the total share
    + [make_condition(f"D_res_alphamatch_go{_go_tag(g)}", see_resource=True, see_emotions=True,
                      alpha=matched_alpha(ALPHA_EXP7, GAMMA_SELF_EXP7, g),
                      gamma_others=GAMMA_SELF_EXP7)
       for g in GAMMA_OTHERS_EXP7 if g != GAMMA_SELF_EXP7]
)

# default experiments when --exp is not given.
# 03_GT first batch (02/10/2026): the 2x2 of exp1 only (A-D, single gamma: no
# gamma_others, one Q head), with the individual-emotion observation
# (EMOTION_TYPE = "vector"). pass other experiments explicitly with --exp.
RUN_EXPERIMENTS = [
    "exp1_resource_obs",
]

FIXED_POLICIES = {"random": "random",
                  "always_exploit": "always_exploit",
                  "always_abstain": "always_abstain"}


# ----------------------------------------
# initialization of agents and environment for a new simulation
# ----------------------------------------


def init_env(condition,
             n_agents=NB_AGENTS,
             env_type=ENVIRONMENT_TYPE,
             initial_resources=INITIAL_RESOURCES,
             emotion_type=EMOTION_TYPE,
             beta=BETA,
             smoothing=SMOOTHING,
             sigmoid_gain=SIGMOID_GAIN,
             threshold=THRESHOLD,
             round_emotions=EMOTION_DECIMALS,
             max_steps=MAX_STEPS):
    """
    create environment and agents for one condition dict (see make_condition).
    all flags are embedded into each agent's config so they become per-agent
    attributes, enabling factorial and (later) mixed-population designs.
    """
    agent_kind = condition["agent"]

    if agent_kind in FIXED_POLICIES:
        AgentClass = FixedPolicyAgent
        base_params = {"policy": FIXED_POLICIES[agent_kind]}
    elif agent_kind == "DQN":
        AgentClass = DQNAgent
        base_params = dict(PARAMS_DQN)
    elif agent_kind == "QLearning":
        AgentClass = QAgent
        base_params = dict(PARAMS_QLEARNING)
    else:
        raise ValueError(f"Invalid agent type: {agent_kind}")

    if condition["gamma"] is not None and agent_kind in ("DQN", "QLearning"):
        base_params["gamma"] = condition["gamma"]
    if condition["gamma_others"] is not None:
        base_params["gamma_others"] = condition["gamma_others"]

    per_agent_config = {**base_params,
                        'see_emotions': condition["see_emotions"],
                        'see_resource': condition["see_resource"],
                        'see_own_meals': condition["see_own_meals"],
                        'alpha': condition["alpha"],
                        'emotion_precision': condition["emotion_precision"]}
    agent_configs = [per_agent_config.copy() for _ in range(n_agents)]

    env = GameTheoreticEnv(
        nb_agents=n_agents,
        env_type=env_type,
        initial_resources=initial_resources,
        emotion_type=emotion_type,
        see_emotions=condition["see_emotions"],
        see_resource=condition["see_resource"],
        see_own_meals=condition["see_own_meals"],
        agent_class=AgentClass,
        agent_configs=agent_configs,
        alpha=condition["alpha"],
        beta=beta,
        smoothing=smoothing,
        sigmoid_gain=sigmoid_gain,
        threshold=threshold,
        round_emotions=round_emotions,
        include_time=INCLUDE_TIME,
        max_steps=max_steps,
        reward_mode=condition["reward_mode"],
        ia_params=PARAMS_IA if condition["reward_mode"] == "inequity_aversion" else None,
        reward_scheme=REWARD_SCHEME
    )

    return env, env.agents


# ----------------------------------------
# metrics
# ----------------------------------------


def gini(x):
    """Gini coefficient of a non-negative array (0 = perfect equality)."""
    x = np.sort(np.array(x, dtype=float))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / total).sum()) / n)


# ----------------------------------------
# step processing
# ----------------------------------------


def run_step(env, agents, simulation_index, episode, step, obs, save_steps=False):
    """
    execute one step: select actions, apply to env, collect info, update agents.
    returns the step record (None unless save_steps), reward arrays, per-agent
    meals, next obs, done.
    """
    # act() = the action PLAYED, remembered by the agent for its next step():
    # the transition stored in the replay buffer carries exactly this action
    actions = [agent.act(obs[i]) for i, agent in enumerate(agents)]

    next_obs, rewards, done, info = env.step(actions)

    personal_arr = np.array(info['personal_reward'])
    empathic_arr = np.array(info['empathic_reward'])
    combined_arr = np.array(info['combined_reward'])
    meals_arr = np.array(info['immediate_rewards'])

    record = None
    if save_steps:
        record = {
            'simulation_number': simulation_index,
            'episode': episode,
            'step': step,
            'resource': env.resource,
            'observations': [o.tolist() for o in obs],
            'actions': actions,
            'personal': personal_arr.tolist(),
            'empathic': empathic_arr.tolist(),
            'combined_reward': combined_arr.tolist(),
            'meals': meals_arr.tolist()
        }

    # learning updates
    for i, agent in enumerate(agents):
        agent.step(next_state=next_obs[i], reward=rewards[i], done=done,
                   reward_parts=(personal_arr[i], empathic_arr[i]))

    return record, personal_arr, empathic_arr, combined_arr, meals_arr, next_obs, done


# ----------------------------------------
# simulation logic for one simulation with multiple episodes
# ----------------------------------------


def run_eval_episode(env, agents, simulation_index, seed, episode, step_count,
                     isolate_rng=False):
    """
    run ONE greedy episode (epsilon=0, no learning, no buffer writes) to
    measure the current policy without exploration noise. agents' epsilons are
    restored afterwards; training state is untouched (the next training episode
    starts with its own env.reset()).

    isolate_rng=True restores the python / numpy / torch RNG states afterwards,
    so the episode leaves the training trajectory bit-identical to a run
    without it. used for the evals added in 03_GT (pre-training and dense
    early evals); the sparse every-eval_every evals keep consuming the RNG as
    in the first 03_GT batch, so training matches that batch seed for seed.
    """
    if isolate_rng:
        rng_states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    saved_epsilons = [getattr(a, 'epsilon', 0.0) for a in agents]
    for a in agents:
        a.epsilon = 0.0

    obs = env.reset()
    total_personal = np.zeros(NB_AGENTS)
    total_empathic = np.zeros(NB_AGENTS)
    total_combined = np.zeros(NB_AGENTS)
    episode_meals = np.zeros(NB_AGENTS)

    for step in range(step_count):
        actions = [a.select_action(obs[i]) for i, a in enumerate(agents)]
        obs, rewards, done, info = env.step(actions)
        total_personal += np.array(info['personal_reward'])
        total_empathic += np.array(info['empathic_reward'])
        total_combined += np.array(info['combined_reward'])
        episode_meals += np.array(info['immediate_rewards'])
        if done:
            break

    for a, eps in zip(agents, saved_epsilons):
        a.epsilon = eps
    if isolate_rng:
        random.setstate(rng_states[0])
        np.random.set_state(rng_states[1])
        torch.set_rng_state(rng_states[2])

    return {
        'phase': 'eval',
        'simulation_number': simulation_index,
        'seed': seed,
        'episode': episode,
        'total_steps': step + 1,
        'resource_remaining': env.resource,
        'epsilon': 0.0,
        'meals': episode_meals.tolist(),
        'gini_meals': gini(episode_meals),
        'personal_totals': total_personal.tolist(),
        'empathic_totals': total_empathic.tolist(),
        'combined_totals': total_combined.tolist()
    }


def observation_labels(nb_agents=NB_AGENTS):
    """names of the observation slots, in the order of env.get_observations()
    (= the input units of the DQN). social slots of agent i are the other
    agents in index order, i itself skipped."""
    labels = ["resource", "own_meals"]
    if EMOTION_TYPE == "average":
        labels += ["emotion_mean_others"]
    else:
        labels += [f"emotion_other_{k}" for k in range(nb_agents - 1)]
    if INCLUDE_TIME:
        labels += ["time"]
    return labels


def snapshot_networks(agents):
    """detached copy of every DQN's policy-network weights (None for agents
    without a network). the clone matters: state_dict() returns live tensors
    that the optimizer keeps updating in place."""
    return [{k: v.detach().cpu().clone() for k, v in a.policy_network.state_dict().items()}
            if hasattr(a, 'policy_network') else None
            for a in agents]


def save_weights(path, condition, seed, initial, final, agents, episodes_trained):
    """one .pt per run: initial and final policy-network weights of each agent
    + what is needed to rebuild and read the networks (DQNNetwork(state_size,
    action_size, hidden_size, n_heads), input slot names, per-agent alpha)."""
    if all(w is None for w in final):
        return  # no DQN in this condition (fixed policies, tabular Q)
    dqn = next(a for a in agents if hasattr(a, 'policy_network'))
    torch.save({
        "condition": condition,
        "seed": seed,
        "episodes_trained": episodes_trained,
        "state_size": dqn.state_size,
        "action_size": dqn.action_size,
        "hidden_size": dqn.policy_network.fc1.out_features,
        "n_heads": dqn.policy_network.n_heads,
        "input_labels": observation_labels(len(agents)),
        "action_labels": ["abstain", "exploit"],
        "agent_ids": [getattr(a, 'agent_id', i) for i, a in enumerate(agents)],
        "initial": initial,
        "final": final,
        "final_epsilon": [getattr(a, 'epsilon', None) for a in agents],
    }, path)


def run_simulation(condition, simulation_index, seed, step_file, summary_file,
                   episode_number=EPISODE_NUMBER, step_count=MAX_STEPS,
                   verbose=True, save_steps=False, eval_every=EVAL_EVERY_DEFAULT,
                   eval_dense_first=EVAL_DENSE_FIRST, weights_file=None):
    set_global_seed(seed)
    env, agents = init_env(condition)
    initial_weights = snapshot_networks(agents) if weights_file else None

    # greedy eval of the untrained policy (episode -1): baseline of the curve
    if eval_every:
        eval_summary = run_eval_episode(env, agents, simulation_index,
                                        seed, -1, step_count, isolate_rng=True)
        write_summary_csv([eval_summary], condition, filename=summary_file)

    episode_iter = tqdm(range(episode_number),
                        desc=f"{condition['label']} seed {seed}") if verbose else range(episode_number)

    for episode in episode_iter:
        obs = env.reset()
        total_personal = np.zeros(NB_AGENTS)
        total_empathic = np.zeros(NB_AGENTS)
        total_combined = np.zeros(NB_AGENTS)
        episode_meals = np.zeros(NB_AGENTS)
        episode_step_records = []

        for step in range(step_count):
            record, pers_r, emp_r, comb_r, meals, obs, done = \
                run_step(env, agents, simulation_index, episode, step, obs,
                        save_steps=save_steps)

            if save_steps:
                episode_step_records.append(record)

            total_personal += pers_r
            total_empathic += emp_r
            total_combined += comb_r
            episode_meals += meals

            if done:
                break

        # per-episode epsilon decay (fix of the per-step decay of the old setup)
        for agent in agents:
            agent.end_episode()

        if save_steps:
            write_step_csv(episode_step_records, simulation_index, seed, step_file)

        summary = {
            'phase': 'train',
            'simulation_number': simulation_index,
            'seed': seed,
            'episode': episode,
            'total_steps': step + 1,
            'resource_remaining': env.resource,
            'epsilon': getattr(agents[0], 'epsilon', 0.0),
            'meals': episode_meals.tolist(),
            'gini_meals': gini(episode_meals),
            'personal_totals': total_personal.tolist(),
            'empathic_totals': total_empathic.tolist(),
            'combined_totals': total_combined.tolist()
        }
        write_summary_csv([summary], condition, filename=summary_file)

        # greedy evaluation: clean measurement of the learned policy, after every
        # episode early on (dense), then every eval_every episodes
        sparse_eval = bool(eval_every) and (episode + 1) % eval_every == 0
        dense_eval = bool(eval_every) and episode < eval_dense_first
        if sparse_eval or dense_eval:
            eval_summary = run_eval_episode(env, agents, simulation_index,
                                            seed, episode, step_count,
                                            isolate_rng=not sparse_eval)
            write_summary_csv([eval_summary], condition, filename=summary_file)

    if weights_file:
        save_weights(weights_file, condition, seed, initial_weights,
                     snapshot_networks(agents), agents, episode_number)


# ----------------------------------------
# functions used to write CSV / config sidecars
# ----------------------------------------


def summary_header(condition):
    return (
        ["condition", "agent", "see_resource", "see_emotions", "emotion_precision",
         "see_own_meals", "alpha", "gamma", "gamma_others", "reward_scheme",
         "reward_mode", "phase", "simulation_number", "seed", "episode", "total_steps",
         "resource_remaining", "initial_resources", "max_steps", "epsilon",
         "gini_meals"] +
        [f"meals_{i}" for i in range(NB_AGENTS)] +
        [f"total_personal_reward_{i}" for i in range(NB_AGENTS)] +
        [f"total_empathic_reward_{i}" for i in range(NB_AGENTS)] +
        [f"total_combined_reward_{i}" for i in range(NB_AGENTS)]
    )


def effective_emotion_precision(condition):
    """rho actually used by the agents (mirrors Agent.get_emotion_precision)."""
    if condition["emotion_precision"] is not None:
        return condition["emotion_precision"]
    return 1.0 if condition["see_emotions"] else 0.0


def write_summary_csv(summaries, condition, filename):
    """append per-episode summary data to CSV. write header only if file does not exist."""
    file_exists = os.path.exists(filename)
    gamma = condition["gamma"] if condition["gamma"] is not None else \
        (PARAMS_DQN["gamma"] if condition["agent"] == "DQN"
         else PARAMS_QLEARNING["gamma"] if condition["agent"] == "QLearning"
         else "")

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(summary_header(condition))
        for rec in summaries:
            row = (
                [condition["label"],
                 condition["agent"],
                 condition["see_resource"],
                 condition["see_emotions"],
                 effective_emotion_precision(condition),
                 condition["see_own_meals"],
                 condition["alpha"],
                 gamma,
                 condition["gamma_others"] if condition["gamma_others"] is not None else "",
                 REWARD_SCHEME,
                 condition["reward_mode"],
                 rec.get('phase', 'train'),
                 rec['simulation_number'],
                 rec['seed'],
                 rec['episode'],
                 rec['total_steps'],
                 rec['resource_remaining'],
                 INITIAL_RESOURCES,
                 MAX_STEPS,
                 rec['epsilon'],
                 rec['gini_meals']]
                + rec['meals']
                + rec['personal_totals']
                + rec['empathic_totals']
                + rec['combined_totals']
            )
            writer.writerow(row)


def write_step_csv(step_records, simulation_index, seed, filename):
    """write step-level records (only used when save_steps=True)."""
    file_exists = os.path.exists(filename)
    header = (
        ["simulation_number", "seed", "episode", "step", "resource_remaining"] +
        [f"observation_{i}" for i in range(NB_AGENTS)] +
        [f"action_{i}" for i in range(NB_AGENTS)] +
        [f"meal_{i}" for i in range(NB_AGENTS)] +
        [f"personal_reward_{i}" for i in range(NB_AGENTS)] +
        [f"empathic_reward_{i}" for i in range(NB_AGENTS)] +
        [f"total_reward_{i}" for i in range(NB_AGENTS)]
    )
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for record in step_records:
            row = (
                [record['simulation_number'], seed, record['episode'],
                 record['step'], record['resource']]
                + [json.dumps(o) for o in record['observations']]
                + record['actions']
                + record['meals']
                + record['personal']
                + record['empathic']
                + record['combined_reward']
            )
            writer.writerow(row)


def write_run_config(condition, experiment_name, seed, folder, basename,
                     episode_number=EPISODE_NUMBER, eval_every=EVAL_EVERY_DEFAULT,
                     eval_dense_first=EVAL_DENSE_FIRST):
    """write a JSON sidecar with the full parameterisation of one run."""
    # record the EFFECTIVE hyperparameters: a per-condition gamma (exp5) overrides
    # the module-level default in init_env, so dumping PARAMS_* raw would report
    # the wrong discount factor in the sidecar. mirrors the override in init_env.
    params_dqn = dict(PARAMS_DQN)
    params_qlearning = dict(PARAMS_QLEARNING)
    if condition["gamma"] is not None:
        if condition["agent"] == "DQN":
            params_dqn["gamma"] = condition["gamma"]
        elif condition["agent"] == "QLearning":
            params_qlearning["gamma"] = condition["gamma"]
    if condition["gamma_others"] is not None:
        params_dqn["gamma_others"] = condition["gamma_others"]

    config = {
        "experiment": experiment_name,
        "condition": condition,
        "seed": seed,
        "nb_agents": NB_AGENTS,
        "episode_number": episode_number,
        "eval_every": eval_every,
        "eval_dense_first": eval_dense_first if eval_every else 0,
        "eval_before_training": bool(eval_every),
        "max_steps": MAX_STEPS,
        "initial_resources": INITIAL_RESOURCES,
        "environment_type": ENVIRONMENT_TYPE,
        "emotion_type": EMOTION_TYPE,
        "beta": BETA,
        "smoothing": SMOOTHING,
        "sigmoid_gain": SIGMOID_GAIN,
        "threshold": THRESHOLD,
        "emotion_decimals": EMOTION_DECIMALS,
        "include_time": INCLUDE_TIME,
        "reward_scheme": REWARD_SCHEME,
        "emotion_precision_effective": effective_emotion_precision(condition),
        "params_dqn": params_dqn,
        "params_qlearning": params_qlearning,
        "params_ia": PARAMS_IA,
        "epsilon_decay_schedule": "per_episode",
        "replay_buffer_persistent": True,
        "stored_action_is_played_action": True,
    }
    with open(os.path.join(folder, basename + "_config.json"), 'w') as f:
        json.dump(config, f, indent=2)


# ----------------------------------------
# seed management
# ----------------------------------------


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----------------------------------------
# main entry point
# ----------------------------------------

BASE_SEED = 102



def parse_args():
    parser = argparse.ArgumentParser(
        description="AIS-2026 revision experiments (see module docstring)")
    parser.add_argument("--runs", type=int, default=NUM_RUNS_PER_CONDITION,
                        help=f"replications per condition for THIS launch (default {NUM_RUNS_PER_CONDITION})")
    parser.add_argument("--sim-start", type=int, default=0,
                        help="first simulation index; use a different value per machine "
                             "(e.g. 0 / 7 / 14) so seeds never collide")
    parser.add_argument("--exp", nargs="+", choices=list(EXPERIMENTS.keys()),
                        default=RUN_EXPERIMENTS, metavar="EXP",
                        help=f"experiments to run (default: {' '.join(RUN_EXPERIMENTS)}; "
                             f"choices: {', '.join(EXPERIMENTS.keys())})")
    parser.add_argument("--conditions", nargs="+", default=None, metavar="LABEL",
                        help="restrict to specific condition labels within the "
                             "selected --exp (e.g. --conditions D_res_gamma090 "
                             "A_res_gamma0999); default: all conditions")
    parser.add_argument("--episodes", type=int, default=EPISODE_NUMBER,
                        help="training episodes per simulation (default 500)")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY_DEFAULT,
                        help="greedy eval episode every N training episodes (0 = off)")
    parser.add_argument("--eval-dense-first", type=int, default=EVAL_DENSE_FIRST,
                        help="also eval after each of the first N training episodes "
                             f"(default {EVAL_DENSE_FIRST}; 0 = sparse schedule only)")
    parser.add_argument("--smoke", action="store_true",
                        help="sanity check: 2 episodes, 1 run, ALL experiments")
    return parser.parse_args()


def main():
    args = parse_args()
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    episode_number = args.episodes
    runs_per_condition = args.runs
    sim_start = args.sim_start
    eval_every = args.eval_every
    experiments = args.exp
    results_root = os.path.join(_script_dir, "csv_results")

    if args.smoke:
        episode_number = 2
        runs_per_condition = 1
        sim_start = 0
        eval_every = 1
        results_root = os.path.join(_script_dir, "results_smoke")
        experiments = list(EXPERIMENTS.keys())
        print("SMOKE MODE: 2 episodes, 1 run per condition, all experiments, eval every episode")

    print(f"Experiments: {', '.join(experiments)}")
    print(f"Simulations {sim_start}..{sim_start + runs_per_condition - 1} "
          f"({runs_per_condition} per condition), eval_every={eval_every}")

    if args.conditions:
        unknown = set(args.conditions) - {c["label"] for exp in experiments
                                          for c in EXPERIMENTS[exp]}
        if unknown:
            raise SystemExit(f"Unknown --conditions label(s): {', '.join(sorted(unknown))}")

    for experiment_name in experiments:
        conditions = EXPERIMENTS[experiment_name]
        out_folder = os.path.join(results_root, experiment_name)
        os.makedirs(out_folder, exist_ok=True)

        for condition_idx, condition in enumerate(conditions):
            # condition_idx feeds the seed formula below, so filtering here
            # (instead of slicing `conditions`) keeps seeds identical to a full run.
            if args.conditions and condition["label"] not in args.conditions:
                continue
            print(f"\n{'='*70}")
            print(f"{experiment_name} — condition: {condition['label']}")
            print(f"  agent={condition['agent']}, see_resource={condition['see_resource']}, "
                  f"see_emotions={condition['see_emotions']}, "
                  f"emotion_precision={effective_emotion_precision(condition)}, see_own_meals={condition['see_own_meals']}, "
                  f"alpha={condition['alpha']}, gamma={condition['gamma'] or 'default'}, "
                  f"gamma_others={condition['gamma_others'] if condition['gamma_others'] is not None else '-'}, "
                  f"reward_mode={condition['reward_mode']}")
            print(f"{'='*70}")

            for simulation_number in range(sim_start, sim_start + runs_per_condition):
                seed = BASE_SEED + condition_idx * 1000 + simulation_number

                basename = f"{condition['label']}_seed{seed:04d}"
                summary_path = os.path.join(out_folder, basename + "_episode_summary.csv")
                step_path = os.path.join(out_folder, basename + "_step_data.csv")
                weights_path = os.path.join(out_folder, basename + "_weights.pt")

                if os.path.exists(summary_path):
                    print(f"  SKIP (already exists): {basename}")
                    continue

                write_run_config(condition, experiment_name, seed, out_folder, basename,
                                 episode_number=episode_number, eval_every=eval_every,
                                 eval_dense_first=args.eval_dense_first)
                run_simulation(
                    condition=condition,
                    simulation_index=simulation_number,
                    seed=seed,
                    step_file=step_path,
                    summary_file=summary_path,
                    episode_number=episode_number,
                    step_count=MAX_STEPS,
                    verbose=True,
                    save_steps=False,
                    eval_every=eval_every,
                    eval_dense_first=args.eval_dense_first,
                    weights_file=weights_path,
                )
                print(f"  Saved: {basename}_episode_summary.csv")

    print(f"\nDone. Results in {results_root}/")


if __name__ == '__main__':
    main()
