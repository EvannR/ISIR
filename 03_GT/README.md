# 03_GT: simulations of the journal paper

Multi-agent reinforcement learning simulations of a common-pool resource dilemma. Six agents share a
resource, and the question is how empathy, split into perceiving the others' emotions and being
motivated by them, changes how long the resource lasts and how evenly it is shared.

## The model

**Environment.** At each step, each agent abstains or tries to eat. An attempt succeeds with
probability r_t / r_0, and each meal removes one unit of the resource. Agents act at the same time,
from the same stock. An episode ends when the resource is depleted, or after 1000 steps. A run lasts
500 episodes, with r_0 = 500.

**Emotion.** h_i is the share of meals in the agent's last 10 steps, starting each episode at the
neutral value 0.5. The emotion is e_i = 2 h_i − 1, on [−1, 1].

**Reward** (`homeostatic_signed`):

    r_i = (1 − α) p_i + α ẽ_i

- p_i = meal_i × (1 − h_i before the meal): a meal is worth less the more sated the agent is, and
  abstaining is worth 0.
- ẽ_i is the mean true emotion of the five others, so their distress is a penalty.

**Observation.** Seven values, whatever the condition, so the network never changes:

| Slot | Value | When hidden |
|------|-------|-------------|
| Resource | r_t / r_0 | 0 (`see_resource=False`) |
| Own meals | h_i | 0 (`see_own_meals=False`) |
| Emotions of the five others | one slot each, rounded to 2 decimals | uniform noise (ρ = 0) |

The emotions are seen through a precision ρ in [0, 1]: each slot is drawn from a Beta distribution
whose mode is the true value, with concentration κ = 2ρ / (1 − ρ). ρ = 1 gives the exact value and
ρ = 0 uniform noise. `see_emotions=True` and `False` are shortcuts for ρ = 1 and ρ = 0. The reward
always uses the true emotions, not the perceived ones.

**Agents.** Each agent has its own DQN with a replay buffer that persists across episodes and a
target network (two hidden layers of 64 units, Adam, learning rate 0.001, γ = 0.99, one update every
4 actions). ε decays once per episode. In exp6, the network has two heads, combined as
Q = (1 − α) Q^p + α Q^e: the empathic stream is discounted with its own γ_o, and both heads
bootstrap on the same action.

## The experiments

Numbered as in the plan of the paper (`method.tex`), at 20 replications per condition. A run shared
by several experiments is computed once.

| Experiment | Design | Purpose | Conditions | New runs |
|------------|------------|---------|-----|------|
| `exp1_resource_obs` | 2×2: ρ ∈ {0, 1} × α ∈ {0, 0.5}, resource seen | Axis 1, reference | 4 | 80 |
| `exp2_replication_no_res` | Same 2×2, resource hidden | Axis 1, moderator | 4 | 80 |
| `exp3_graded_alpha` | α ∈ {0, .25, .5, .75, .99}, ρ = 1 | Axis 2, motivation | 5 | 60 |
| `exp4_graded_precision` | ρ ∈ {0, .25, .5, .75, 1} × α ∈ {0, .5} | Axis 3, perception | 10 | 120 |
| `exp5_gamma_robustness` | 2×2 at γ ∈ {0.8, 0.9, 0.999} | Horizon, shared γ | 12 | 240 |
| `exp6_social_discount` | 7 levels of γ_o + 6 matched-α controls | Horizon, independent γ and γ_o | 13 | 260 |
| exp7, reference policies | Random, always exploit, always abstain | Notebook only, no training | 3 | none |
| exp8, frozen weights | Trained policies under logistic regeneration | Notebook only, no training | n/a | none |

The campaign is 840 runs.

## Files

```
03_GT/
  main_gt_ais.py              # experiments, command line, run identity
  env_gt_ais.py               # environment
  agent_policies_gt_ais.py    # agents and reward
  analysis/                   # one notebook per experiment, plus the checks below
  csv_results/                # results, see below
  figures/                    # figures of the notebooks
```

In `analysis/`:
- `analysis_00_reward_observation_checks.ipynb`: checks of the reward and the observation.
- `analysis_01_social_discount_factor.ipynb`: reasoning behind γ_o.
- `analysis_02_emotion_probes.ipynb`: how the trained networks read the individual emotions.
- `analysis_exp1_*` to `analysis_exp8_*`: one analysis per experiment.

## Running

Install the dependencies from the root of the repository, then run from this folder:

```bash
pip install -r requirements.txt
cd 03_GT
python main_gt_ais.py --smoke      # 2 episodes of every condition, written to results_smoke/
```

**Run exp1 first, and alone.** exp3 reuses its runs C and D, and exp4 its runs A to D. The saved
weights keep the label of the experiment that computed them, and `analysis_02` and exp8 read it.

```bash
python main_gt_ais.py --runs 20    # exp1
python main_gt_ais.py --runs 20 --exp exp2_replication_no_res exp3_graded_alpha
python main_gt_ais.py --runs 20 --exp exp4_graded_precision exp5_gamma_robustness exp6_social_discount
```

The default is 3 replications per condition, so pass `--runs 20` for the campaign.

**Several terminals or machines.** Split the replications with `--sim-start`. Each run has its own
seed, so this does not change the values. On the test PC, the total speed stopped improving beyond
about 8 runs side by side.

```bash
python main_gt_ais.py --runs 5 --sim-start 0
python main_gt_ais.py --runs 5 --sim-start 5
python main_gt_ais.py --runs 5 --sim-start 10
python main_gt_ais.py --runs 5 --sim-start 15
```

**Relaunching is safe:**
- A finished run is skipped, and an interrupted one starts again.
- If a finished run was computed with other code (`code_version`), the launch stops. Comments and
  docstrings do not count as code. Move the run's folder or restore the code.
- A `.lock` file stops a second process on the same run. It only stays if the process was killed:
  delete it, as the error message says.
- A folder in `csv_results/<experiment>/` holding runs of another configuration is refused: rename it.

## Results

```
csv_results/
  runs/<config_id>/           # one folder per configuration, one set of files per replication
    <config_id>_rep<NNN>_seed<seed>_episode_summary.csv
    <config_id>_rep<NNN>_seed<seed>_config.json     # parameters, code_version
    <config_id>_rep<NNN>_seed<seed>_weights.pt      # initial and final weights of the 6 agents
  <experiment>/               # copies of the summary CSVs, read by the notebooks
```

A run is identified by its parameters (`config_id`, without the label) and its replication number,
which give its seed.

In the CSV, **use the rows with `phase='eval'` for the analyses**: greedy episodes (ε = 0, no
learning), run every 10 episodes, before training (`episode = -1`) and after each of the first 20
episodes. The rows with `phase='train'` include the exploration noise.
