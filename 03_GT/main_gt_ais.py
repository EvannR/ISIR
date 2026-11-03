"""
this file defines:
  - PARAMS_DQN and the constants of a simulation.
  - EXPERIMENTS               : the conditions of each experiment, see below.
  - init_env / run_simulation : run one replication and write its CSV and final weights.
  - config_id / run_seed      : identity of a run, see below.
  - main                      : command line, runs every condition of the selected experiments.

experiments, numbered as in the plan of the journal paper (method.tex, Table 1):
  exp1_resource_obs        : Axis 1, 2x2 (see_emotions x alpha) with see_resource=True
  exp2_replication_no_res  : Axis 1 moderator, same 2x2 with see_resource=False
  exp3_graded_alpha        : Axis 2, alpha in {0,.25,.5,.75,.99}, emotions seen
  exp4_graded_precision    : Axis 3, rho in {0,.25,.5,.75,1} x alpha in {0,.5}
  exp5_gamma_robustness    : horizon, 2x2 at gamma in {0.8, 0.9, 0.999}, one head
  exp6_social_discount     : horizon, empathic stream discounted with its own
                             gamma_others (two-head DQN) + matched-alpha controls
exp7 (reference policies) and exp8 (frozen weights) involve no training: they are analysis notebooks.

usage:
    python main_gt_ais.py --runs 20                  # exp1 (RUN_EXPERIMENTS), default: 3 replications per condition
    python main_gt_ais.py --runs 7 --sim-start 7     # replications 7..13, for another machine or terminal
    python main_gt_ais.py --exp exp3_graded_alpha    # restrict to specific experiments
    python main_gt_ais.py --eval-every 0             # disable the greedy evaluation episodes
    python main_gt_ais.py --smoke                    # 2-episode sanity check of every condition

run exp1 first: exp3 reuses its runs C-D and exp4 its runs A-D, and the weights keep the label of
the experiment that computed them (analysis_02 and exp8 read it).

a run = a configuration (config_id, without the label) + a replication index, which give its seed.
runs live in csv_results/runs/<config_id>/, each experiment folder gets a copy of its summary CSVs.
relaunching is safe: finished runs are skipped (the launch stops if the code changed), interrupted
ones restarted. a .lock file stops a second process on the same run; if the process was killed, delete it.

greedy evaluations (epsilon = 0, no learning) are logged with phase='eval', the rows to analyse:
every --eval-every episodes, before training (episode = -1) and after the first --eval-dense-first episodes.
"""

from env_gt_ais import GameTheoreticEnv
from agent_policies_gt_ais import DQNAgent, FixedPolicyAgent
import argparse
import ast
import hashlib
import io
import tokenize
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

NUM_RUNS_PER_CONDITION = 3   # default replications per launch; the campaign uses 20
EVAL_EVERY_DEFAULT = 10      # greedy eval every N training episodes (0 = off)
EVAL_DENSE_FIRST = 20        # also after each of the first N: the greedy policy converges within ~10 episodes
EPISODE_NUMBER = 500
NB_AGENTS = 6
MAX_STEPS = 1000             # per episode
INITIAL_RESOURCES = 500      # at the start of each episode
ENVIRONMENT_TYPE = "stochastic"  # 'deterministic' or 'stochastic'

EMOTION_TYPE = "vector"    # observation: 'vector' (one slot per other agent, one noise draw each) or
                           # 'average' (their mean). the empathic reward is always the mean
BETA = 0.5                 # weight of the last meal (legacy scheme only)
SMOOTHING = 'linear'
SIGMOID_GAIN = 5.0
THRESHOLD = 0.5            # consumption rate at which the emotion is neutral
EMOTION_DECIMALS = 2

INCLUDE_TIME = False       # optional time slot in the observation


REWARD_SCHEME = "homeostatic_signed" # journal model: meal value decreasing with satiety, empathic term and observed emotions on [-1, 1], r in [-alpha, 1]. see agent_policies_gt_ais for the other schemes

# epsilon decays per episode: 0.99^460 ~ 0.01, exploration lasts ~460 of the 500 episodes
PARAMS_DQN = {"learning_rate": 0.001,"gamma": 0.99,"epsilon": 1.0,"epsilon_decay": 0.99,"epsilon_min": 0.01,"batch_size": 64,"hidden_size": 64,"update_target_every": 10,"buffer_capacity": 100000,"learn_every": 4 }  # one update every 4 actions (Mnih et al. 2015)

# ----------------------------------------
# experiment definitions
# ----------------------------------------

def make_condition(label, agent="DQN", see_resource=True, see_emotions=True,
                   see_own_meals=True, alpha=0.5, gamma=None, gamma_others=None,
                   emotion_precision=None):  # , fixed_policy=None
    if gamma_others is not None and agent != "DQN":
        raise ValueError("gamma_others needs agent='DQN'")
    return {
        "label": label,
        "agent": agent,                # 'DQN', 'random', 'always_exploit', 'always_abstain'
        "see_resource": see_resource,
        "see_emotions": see_emotions,
        "emotion_precision": emotion_precision,  # rho in [0,1]; None -> 1.0 if see_emotions else 0.0
        "see_own_meals": see_own_meals,  # own recent consumption rate in obs
        "alpha": alpha,
        "gamma": gamma,                # None -> default from PARAMS_DQN
        "gamma_others": gamma_others,  # None -> one Q head; else two heads, empathic stream discounted with it
        # "fixed_policy": fixed_policy,  # set automatically for fixed agents
    }


EXPERIMENTS = {
    # Axis 1: agents observe r_t / r_0
    "exp1_resource_obs": [
        make_condition("A_res_blind_non_empathic",    see_resource=True, see_emotions=False, alpha=0.0),
        make_condition("B_res_blind_reward_empathic", see_resource=True, see_emotions=False, alpha=0.5),
        make_condition("C_res_sees_ignores",          see_resource=True, see_emotions=True,  alpha=0.0),
        make_condition("D_res_full_empathy",          see_resource=True, see_emotions=True,  alpha=0.5),
    ],
    # moderator: same as exp1 without r_t / r_0. own meals stay visible, to isolate its effect
    "exp2_replication_no_res": [
        make_condition("A_nores_blind_non_empathic",    see_resource=False, see_emotions=False, alpha=0.0),
        make_condition("B_nores_blind_reward_empathic", see_resource=False, see_emotions=False, alpha=0.5),
        make_condition("C_nores_sees_ignores",          see_resource=False, see_emotions=True,  alpha=0.0),
        make_condition("D_nores_full_empathy",          see_resource=False, see_emotions=True,  alpha=0.5),
    ],
    # Axis 2: graded empathic weight. alpha 0 and 0.5 are the runs C and D of exp1
    "exp3_graded_alpha": [
        make_condition(f"alpha{int(alpha * 100):03d}_res", see_resource=True, see_emotions=True, alpha=alpha)
        for alpha in (0.0, 0.25, 0.5, 0.75, 0.99)
    ],
    # Axis 3: graded perception. rho 0 and 1 are the runs A-D of exp1
    "exp4_graded_precision": [
        make_condition(f"rho{int(rho * 100):03d}_alpha{int(alpha * 100):03d}", see_resource=True,
                       see_emotions=True, emotion_precision=rho, alpha=alpha)
        for alpha in (0.0, 0.5) for rho in (0.0, 0.25, 0.5, 0.75, 1.0)
    ],
    # horizon: full 2x2 at each gamma, for the gamma x rho x alpha ANOVA. gamma = 0.8 is the myopic
    # regime: horizon of 5 steps, below the 10-step meal history. the empathic reward is immediate,
    # so an effect surviving here does not come from far-sighted agents
    "exp5_gamma_robustness": [
        make_condition(f"{cell}_res_gamma{tag}", see_resource=True, see_emotions=see_emotions, alpha=alpha, gamma=gamma)
        for tag, gamma in (("090", 0.9), ("0999", 0.999), ("080", 0.8))
        for cell, see_emotions, alpha in (("A", False, 0.0), ("B", False, 0.5), ("C", True, 0.0), ("D", True, 0.5))
    ],
}

GAMMA_SELF_EXP6 = PARAMS_DQN["gamma"] # social discount factor (see analysis_01): two heads everywhere, for one architecture across the grid, including the reference gamma_others = gamma
ALPHA_EXP6 = 0.5
GAMMA_OTHERS_EXP6 = [0.99, 0.95, 0.9, 0.8, 0.0, 0.995, 0.999]


def matched_alpha(alpha, gamma_self, gamma_others):
    """
    alpha' giving a single-gamma agent the same total empathic share as (alpha, gamma_others),
    on a constant reward stream: alpha' = alpha/(1-g_o) / ((1-alpha)/(1-g_s) + alpha/(1-g_o)).
    this control separates "less empathy" from "shorter empathy".
    """
    w_self = (1.0 - alpha) / (1.0 - gamma_self)
    w_others = alpha / (1.0 - gamma_others)
    return round(w_others / (w_self + w_others), 4)


def _go_tag(g):
    return f"{int(round(g * 1000)):04d}"   # 0.95 -> '0950', 0.999 -> '0999'


EXPERIMENTS["exp6_social_discount"] = (
    # manipulation: alpha fixed, horizon of the empathic stream varied
    [make_condition(f"D_res_go{_go_tag(g)}", see_resource=True, see_emotions=True,
                    alpha=ALPHA_EXP6, gamma_others=g)
     for g in GAMMA_OTHERS_EXP6]
    # controls: single horizon (gamma_others = gamma), alpha matched on the total share
    + [make_condition(f"D_res_alphamatch_go{_go_tag(g)}", see_resource=True, see_emotions=True,
                      alpha=matched_alpha(ALPHA_EXP6, GAMMA_SELF_EXP6, g),
                      gamma_others=GAMMA_SELF_EXP6)
       for g in GAMMA_OTHERS_EXP6 if g != GAMMA_SELF_EXP6]
)

# when --exp is not given
RUN_EXPERIMENTS = [
    "exp1_resource_obs",
]

FIXED_POLICIES = {"random": "random",
                  "always_exploit": "always_exploit",
                  "always_abstain": "always_abstain"}


# -----------------------------------------------------------------------------------------------
# initialization of agents and environment for a new simulation
# -----------------------------------------------------------------------------------------------


def init_env(condition, emotion_type=EMOTION_TYPE):
    """
    create environment and agents for one condition (see make_condition).
    """
    agent_kind = condition["agent"]

    if agent_kind in FIXED_POLICIES:
        AgentClass = FixedPolicyAgent
        base_params = {"policy": FIXED_POLICIES[agent_kind]}
    elif agent_kind == "DQN":
        AgentClass = DQNAgent
        base_params = dict(PARAMS_DQN)
    else:
        raise ValueError(f"Invalid agent type: {agent_kind}")

    if condition["gamma"] is not None and agent_kind == "DQN":
        base_params["gamma"] = condition["gamma"]
    if condition["gamma_others"] is not None:
        base_params["gamma_others"] = condition["gamma_others"]

    per_agent_config = {**base_params,
                        'see_emotions': condition["see_emotions"],
                        'see_resource': condition["see_resource"],
                        'see_own_meals': condition["see_own_meals"],
                        'alpha': condition["alpha"],
                        'emotion_precision': condition["emotion_precision"]}
    agent_configs = [per_agent_config.copy() for _ in range(NB_AGENTS)]

    env = GameTheoreticEnv(
        nb_agents=NB_AGENTS,
        env_type=ENVIRONMENT_TYPE,
        initial_resources=INITIAL_RESOURCES,
        emotion_type=emotion_type,
        # see_emotions=condition["see_emotions"],  # the agent_configs carry them
        # see_resource=condition["see_resource"],
        # see_own_meals=condition["see_own_meals"],
        agent_class=AgentClass,
        agent_configs=agent_configs,
        # alpha=condition["alpha"],
        beta=BETA,
        smoothing=SMOOTHING,
        sigmoid_gain=SIGMOID_GAIN,
        threshold=THRESHOLD,
        round_emotions=EMOTION_DECIMALS,
        include_time=INCLUDE_TIME,
        max_steps=MAX_STEPS,
        reward_scheme=REWARD_SCHEME
    )

    return env, env.agents


# -------------------------------------------------------------------------------------------
# metric
# --------------------------------------------------------------------------------------------


def gini(x):
    """Gini coefficient of a non-negative array (0 = perfect equality)."""
    x = np.sort(np.array(x, dtype=float))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / total).sum()) / n)


# -------------------------------------------------------------------------------------------
# step processing
# -------------------------------------------------------------------------------------------


def run_step(env, agents, obs):  # , simulation_index, episode, step, save_steps=False
    """
    execute one step: select actions, apply to env, collect info, update agents.
    returns the reward arrays, per-agent meals, next obs, done.
    """
    # act() remembers the action played: it is the one stored by step()
    actions = [agent.act(obs[i]) for i, agent in enumerate(agents)]

    next_obs, rewards, done, info = env.step(actions)

    personal_arr = np.array(info['personal_reward'])
    empathic_arr = np.array(info['empathic_reward'])
    combined_arr = np.array(info['combined_reward'])
    meals_arr = np.array(info['immediate_rewards'])

    # record = None
    # if save_steps:
    #     record = {
    #         'simulation_number': simulation_index,
    #         'episode': episode,
    #         'step': step,
    #         'resource': env.resource,
    #         'observations': [o.tolist() for o in obs],
    #         'actions': actions,
    #         'personal': personal_arr.tolist(),
    #         'empathic': empathic_arr.tolist(),
    #         'combined_reward': combined_arr.tolist(),
    #         'meals': meals_arr.tolist()
    #     }

    # learning updates
    for i, agent in enumerate(agents):
        agent.step(next_state=next_obs[i], reward=rewards[i], done=done,
                   reward_parts=(personal_arr[i], empathic_arr[i]))

    return personal_arr, empathic_arr, combined_arr, meals_arr, next_obs, done


# -------------------------------------------------------------------------------------------------------------
# one simulation
# -------------------------------------------------------------------------------------------------------------


def episode_record(phase, simulation_index, seed, episode, step, env, epsilon, totals):
    """summary row of a finished episode. totals: per-agent sums of (personal, empathic, combined, meals)."""
    personal, empathic, combined, meals = totals.tolist()
    return {
        'phase': phase,
        'simulation_number': simulation_index,
        'seed': seed,
        'episode': episode,
        'total_steps': step + 1,
        'resource_remaining': env.resource,
        'epsilon': epsilon,
        'meals': meals,
        'gini_meals': gini(meals),
        'personal_totals': personal,
        'empathic_totals': empathic,
        'combined_totals': combined
    }


def run_eval_episode(env, agents, simulation_index, seed, episode, step_count,
                     isolate_rng=False):
    """
    run one greedy episode (epsilon = 0, no learning, no buffer writes), epsilons restored afterwards.
    isolate_rng=True also restores the RNG states, so the episode does not change the training:
    used for the evals before training and the dense ones. the sparse evals consume the RNG.
    """
    if isolate_rng:
        rng_states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    saved_epsilons = [getattr(a, 'epsilon', 0.0) for a in agents]
    for a in agents:
        a.epsilon = 0.0

    obs = env.reset()
    totals = np.zeros((4, NB_AGENTS))

    for step in range(step_count):
        actions = [a.select_action(obs[i]) for i, a in enumerate(agents)]
        obs, rewards, done, info = env.step(actions)
        totals += (info['personal_reward'], info['empathic_reward'],
                   info['combined_reward'], info['immediate_rewards'])
        if done:
            break

    for a, eps in zip(agents, saved_epsilons):
        a.epsilon = eps
    if isolate_rng:
        random.setstate(rng_states[0])
        np.random.set_state(rng_states[1])
        torch.set_rng_state(rng_states[2])

    return episode_record('eval', simulation_index, seed, episode, step, env, 0.0, totals)


def observation_labels(nb_agents=NB_AGENTS):
    """names of the observation slots (the inputs of the DQN). social slots: the others, in index order."""
    labels = ["resource", "own_meals"]
    if EMOTION_TYPE == "average":
        labels += ["emotion_mean_others"]
    else:
        labels += [f"emotion_other_{k}" for k in range(nb_agents - 1)]
    if INCLUDE_TIME:
        labels += ["time"]
    return labels


def snapshot_networks(agents):
    """copy of each agent's policy-network weights (None without a network).
    clone: state_dict() returns live tensors, which the optimizer keeps updating."""
    return [{k: v.detach().cpu().clone() for k, v in a.policy_network.state_dict().items()}
            if hasattr(a, 'policy_network') else None
            for a in agents]


def save_weights(path, condition, seed, initial, final, agents, episodes_trained):
    """one .pt per run: initial and final weights of each agent + what is needed to rebuild the networks."""
    if all(w is None for w in final):
        return  # no DQN in this condition (fixed policies)
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


def run_simulation(condition, simulation_index, seed, summary_file,  # , step_file
                   episode_number=EPISODE_NUMBER, step_count=MAX_STEPS,
                   verbose=True, eval_every=EVAL_EVERY_DEFAULT,  # , save_steps=False
                   eval_dense_first=EVAL_DENSE_FIRST, weights_file=None):
    set_global_seed(seed)
    env, agents = init_env(condition)
    initial_weights = snapshot_networks(agents) if weights_file else None

    # untrained policy (episode -1)
    if eval_every:
        eval_summary = run_eval_episode(env, agents, simulation_index,
                                        seed, -1, step_count, isolate_rng=True)
        write_summary_csv([eval_summary], condition, filename=summary_file)

    episode_iter = tqdm(range(episode_number),
                        desc=f"{condition['label']} seed {seed}") if verbose else range(episode_number)

    for episode in episode_iter:
        obs = env.reset()
        totals = np.zeros((4, NB_AGENTS))
        # episode_step_records = []

        for step in range(step_count):
            pers_r, emp_r, comb_r, meals, obs, done = run_step(env, agents, obs)

            # if save_steps:
            #     episode_step_records.append(record)

            totals += (pers_r, emp_r, comb_r, meals)

            if done:
                break

        # epsilon decay
        for agent in agents:
            agent.end_episode()

        # if save_steps:
        #     write_step_csv(episode_step_records, simulation_index, seed, step_file)

        summary = episode_record('train', simulation_index, seed, episode, step, env,
                                 getattr(agents[0], 'epsilon', 0.0), totals)
        write_summary_csv([summary], condition, filename=summary_file)

        # greedy evaluation: dense first, then every eval_every episodes
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


# ----------------------------------------------------------------------------------------------------
# CSV and JSON config
# ----------------------------------------------------------------------------------------------------


def summary_header():  # condition
    return (
        ["condition", "agent", "see_resource", "see_emotions", "emotion_precision",
         "see_own_meals", "alpha", "gamma", "gamma_others", "reward_scheme",
         "phase", "simulation_number", "seed", "episode", "total_steps",
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
    """append summary rows to the CSV, with the header if the file is new."""
    file_exists = os.path.exists(filename)
    gamma = condition["gamma"] if condition["gamma"] is not None else \
        (PARAMS_DQN["gamma"] if condition["agent"] == "DQN"
         else "")

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(summary_header())
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


# def write_step_csv(step_records, simulation_index, seed, filename):
#     """write step-level records (only used when save_steps=True)."""
#     file_exists = os.path.exists(filename)
#     header = (
#         ["simulation_number", "seed", "episode", "step", "resource_remaining"] +
#         [f"observation_{i}" for i in range(NB_AGENTS)] +
#         [f"action_{i}" for i in range(NB_AGENTS)] +
#         [f"meal_{i}" for i in range(NB_AGENTS)] +
#         [f"personal_reward_{i}" for i in range(NB_AGENTS)] +
#         [f"empathic_reward_{i}" for i in range(NB_AGENTS)] +
#         [f"total_reward_{i}" for i in range(NB_AGENTS)]
#     )
#     with open(filename, 'a', newline='') as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(header)
#         for record in step_records:
#             row = (
#                 [record['simulation_number'], seed, record['episode'],
#                  record['step'], record['resource']]
#                 + [json.dumps(o) for o in record['observations']]
#                 + record['actions']
#                 + record['meals']
#                 + record['personal']
#                 + record['empathic']
#                 + record['combined_reward']
#             )
#             writer.writerow(row)


# functions of this file that run the simulation; the rest of a run is written in its JSON config
RUNNER_FUNCTIONS = {"init_env", "gini", "run_step", "episode_record", "run_eval_episode",
                    "run_simulation", "set_global_seed"}


def _code_text(path, only_functions=None):
    """source without comments, docstrings and blank lines. works on the text itself,
    so the result does not depend on the Python version, unlike ast.dump."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    lines = src.splitlines()
    tree = ast.parse(src)
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            lines[row - 1] = lines[row - 1][:col]
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                drop.update(range(first.lineno - 1, first.end_lineno))
    keep = range(len(lines))
    if only_functions is not None:
        keep = {i for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in only_functions
                for i in range(n.lineno - 1, n.end_lineno)}
    return "\n".join(lines[i].rstrip() for i in sorted(keep)
                     if i not in drop and lines[i].strip())


def code_version():
    """short sha256 of the simulation code, without comments and docstrings.
    a new function of this file that influences the simulation goes in RUNNER_FUNCTIONS."""
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    h.update(_code_text(os.path.join(here, "env_gt_ais.py")).encode())
    h.update(_code_text(os.path.join(here, "agent_policies_gt_ais.py")).encode())
    h.update(_code_text(os.path.join(here, "main_gt_ais.py"), RUNNER_FUNCTIONS).encode())
    return h.hexdigest()[:12]


def run_parameters(condition, episode_number, eval_every, eval_dense_first):
    """every parameter of one run except its identity (experiment, label, seed)."""
    # the values really used: same override as in init_env
    params_dqn = dict(PARAMS_DQN)
    if condition["gamma"] is not None and condition["agent"] == "DQN":
        params_dqn["gamma"] = condition["gamma"]
    if condition["gamma_others"] is not None:
        params_dqn["gamma_others"] = condition["gamma_others"]

    return {
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
        "epsilon_decay_schedule": "per_episode",
        "replay_buffer_persistent": True,
        "stored_action_is_played_action": True,
    }


def config_id(condition, episode_number, eval_every, eval_dense_first):
    """short hash of what defines a run, with the effective gamma and rho and without the label:
    a condition shared by several experiments (exp1 D = exp3 alpha 0.5) has one identity."""
    identity = run_parameters(condition, episode_number, eval_every, eval_dense_first)
    identity.update({k: condition[k] for k in ("agent", "see_resource", "see_own_meals",
                                               "alpha")})
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]


def run_seed(cid, replication):
    """deterministic seed of one replication of one configuration (fits 32 bits)."""
    return int(hashlib.sha256(f"{cid}:{replication}".encode()).hexdigest()[:8], 16)


def write_run_config(condition, experiment_name, seed, folder, basename,
                     episode_number=EPISODE_NUMBER, eval_every=EVAL_EVERY_DEFAULT,
                     eval_dense_first=EVAL_DENSE_FIRST):
    """write the JSON config of one run."""
    config = {
        "experiment": experiment_name,
        "condition": condition,
        "seed": seed,
        "config_id": config_id(condition, episode_number, eval_every, eval_dense_first),
        "code_version": code_version(),
        **run_parameters(condition, episode_number, eval_every, eval_dense_first),
    }
    with open(os.path.join(folder, basename + "_config.json"), 'w') as f:
        json.dump(config, f, indent=2)


def run_is_complete(summary_path, episode_number):
    """last training episode reached, not the row count: a finished run can miss a row."""
    if not os.path.exists(summary_path):
        return False
    with open(summary_path, newline='') as f:
        episodes = [int(r["episode"]) for r in csv.DictReader(f) if r["phase"] == "train"]
    return bool(episodes) and max(episodes) == episode_number - 1


def copy_summary(summary_path, target_path, condition):
    """copy a run's summary into an experiment folder, with this experiment's label and see_emotions:
    the run may have been computed for another experiment (same effective rho)."""
    with open(summary_path, newline='') as f:
        rows = list(csv.reader(f))
    label_column = rows[0].index("condition")
    see_emotions_column = rows[0].index("see_emotions")
    for row in rows[1:]:
        row[label_column] = condition["label"]
        row[see_emotions_column] = condition["see_emotions"]
    with open(target_path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)


# --------------------------------------------------------------------------------------------------------
# seed management
# --------------------------------------------------------------------------------------------------------


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="experiments of the journal paper (see module docstring)")
    parser.add_argument("--runs", type=int, default=NUM_RUNS_PER_CONDITION,
                        help=f"replications per condition for THIS launch (default {NUM_RUNS_PER_CONDITION})")
    parser.add_argument("--sim-start", type=int, default=0,
                        help="first replication index; use a different value per machine "
                             "(e.g. 0 / 7 / 14) so runs never collide")
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
    torch.set_num_threads(1)  # these small networks run slower on several threads
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(script_dir, "csv_results")

    if args.smoke:
        args.episodes = 2
        args.runs = 1
        args.sim_start = 0
        args.eval_every = 1
        args.exp = list(EXPERIMENTS.keys())
        results_root = os.path.join(script_dir, "results_smoke")
        print("SMOKE MODE: 2 episodes, 1 run per condition, all experiments, eval every episode")

    print(f"Experiments: {', '.join(args.exp)}")
    print(f"Simulations {args.sim_start}..{args.sim_start + args.runs - 1} "
          f"({args.runs} per condition), eval_every={args.eval_every}")

    if args.conditions:
        unknown = set(args.conditions) - {c["label"] for exp in args.exp
                                          for c in EXPERIMENTS[exp]}
        if unknown:
            raise SystemExit(f"Unknown --conditions label(s): {', '.join(sorted(unknown))}")

    current_code = code_version()

    def identify(condition):
        return config_id(condition, args.episodes, args.eval_every, args.eval_dense_first)

    for experiment_name in args.exp:
        conditions = EXPERIMENTS[experiment_name]
        out_folder = os.path.join(results_root, experiment_name)
        os.makedirs(out_folder, exist_ok=True)
        # the notebooks read every CSV of this folder: never mix configurations
        expected = {f"_{identify(c)}_rep" for c in conditions}
        foreign = [n for n in os.listdir(out_folder) if n.endswith("_episode_summary.csv")
                   and not any(tag in n for tag in expected)]
        if foreign:
            raise SystemExit(f"{out_folder} holds {len(foreign)} summary CSV(s) from another "
                             f"configuration (e.g. {foreign[0]}): rename or move this folder.")

        for condition in conditions:
            if args.conditions and condition["label"] not in args.conditions:
                continue
            print(f"\n{'='*70}")
            print(f"{experiment_name} — condition: {condition['label']}")
            print(f"  agent={condition['agent']}, see_resource={condition['see_resource']}, "
                  f"see_emotions={condition['see_emotions']}, "
                  f"emotion_precision={effective_emotion_precision(condition)}, see_own_meals={condition['see_own_meals']}, "
                  f"alpha={condition['alpha']}, gamma={condition['gamma'] or 'default'}, "
                  f"gamma_others={condition['gamma_others'] if condition['gamma_others'] is not None else '-'}")
            print(f"{'='*70}")

            cid = identify(condition)
            run_folder = os.path.join(results_root, "runs", cid)
            os.makedirs(run_folder, exist_ok=True)

            for simulation_number in range(args.sim_start, args.sim_start + args.runs):
                seed = run_seed(cid, simulation_number)
                tag = f"{cid}_rep{simulation_number:03d}"
                basename = f"{tag}_seed{seed}"
                summary_path = os.path.join(run_folder, basename + "_episode_summary.csv")
                # step_path = os.path.join(run_folder, basename + "_step_data.csv")
                weights_path = os.path.join(run_folder, basename + "_weights.pt")
                copy_path = os.path.join(out_folder,
                                         f"{condition['label']}_{tag}_episode_summary.csv")

                # the weights are written last: without them, the final eval may be missing
                if run_is_complete(summary_path, args.episodes) and os.path.exists(weights_path):
                    with open(os.path.join(run_folder, basename + "_config.json")) as f:
                        run_code = json.load(f)["code_version"]
                    if run_code != current_code:
                        raise SystemExit(f"{basename} was computed with the code {run_code}, the current "
                                         f"code is {current_code}: move {run_folder} or restore the code.")
                    print(f"  SKIP (already complete): {basename}")
                    copy_summary(summary_path, copy_path, condition)
                    continue

                lock_path = os.path.join(run_folder, basename + ".lock")
                try:
                    open(lock_path, 'x').close()
                except FileExistsError:
                    raise SystemExit(f"another process is computing {basename}. if none is, it was "
                                     f"killed: delete {lock_path} and relaunch.")
                try:
                    if os.path.exists(summary_path):
                        print(f"  RESTART (interrupted): {basename}")
                    for stale in (summary_path, weights_path):
                        if os.path.exists(stale):
                            os.remove(stale)

                    write_run_config(condition, experiment_name, seed, run_folder, basename,
                                     episode_number=args.episodes, eval_every=args.eval_every,
                                     eval_dense_first=args.eval_dense_first)
                    run_simulation(
                        condition=condition,
                        simulation_index=simulation_number,
                        seed=seed,
                        # step_file=step_path,
                        summary_file=summary_path,
                        episode_number=args.episodes,
                        step_count=MAX_STEPS,
                        verbose=True,
                        # save_steps=False,
                        eval_every=args.eval_every,
                        eval_dense_first=args.eval_dense_first,
                        weights_file=weights_path,
                    )
                    copy_summary(summary_path, copy_path, condition)
                    print(f"  Saved: {basename}_episode_summary.csv")
                finally:
                    os.remove(lock_path)

    print(f"\nDone. Results in {results_root}/")


if __name__ == '__main__':
    main()
