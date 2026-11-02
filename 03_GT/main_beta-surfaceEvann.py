"""
main_beta.py -- quick ablations on why the effect of empathy is smaller in
03_GT than in the AIS 2026 paper, on conditions C and D of exp1 (emotions seen,
resource observed, alpha 0 vs 0.5).

Batch 1 (14/10/2026, 100 episodes, csv_results/beta/): one 03_GT setting
changed at a time.
  legacy_average      : AIS 2026 reward + mean of the others' emotions
  homeostatic_average : homeostatic reward + mean emotion
  homeostatic_vector  : homeostatic reward + individual emotions (current 03_GT)
  -> D - C: +110 / +58 / -3 steps. The reward change and the individual
     emotions both shrink the effect; the bug fixes do not.

Batch 2 (default, 200 episodes, 3 runs per condition, csv_results/beta2/):
from the current 03_GT model (homeostatic reward, individual emotions), one
change at a time.
  homeostatic_vector  : reference, same model over 200 episodes
  signed_empathy      : empathic term on [-1, 1] as in the AIS paper, so that
                        the others' distress is a penalty
  learn_every5        : one learning update every 5 steps instead of every step
  (lr_div5, learning rate divided by 5, is defined but not run for now)

Batch 3 (default, 500 episodes, 3 runs per condition, csv_results/beta3/): the
full 2x2 of exp1 (A, B, C, D) with the journal configuration decided on
27/10/2026 (global_v1): signed empathic term, individual emotions, one learning
update every 4 actions.

Each run: greedy evaluation before training, after each of the first 20
episodes and every 10 episodes after that. Everything else is main_gt_ais.

usage:
    python main_beta.py                              # batch 3: global_v1 x 4 conditions x 3 runs
    python main_beta.py --batch 2                    # batch 2: 3 variants x 2 conditions x 3 runs
    python main_beta.py --variants signed_empathy    # one variant only
    python main_beta.py --report                     # tables + figure from existing CSVs
    python main_beta.py --batch 1 --report           # report of batch 1
"""

import argparse
import functools
import os

import main_gt_ais as M

# order matters: the variant index feeds the seed formula, so append new
# variants at the end. "dqn" overrides entries of main_gt_ais.PARAMS_DQN.
VARIANTS = {
    "legacy_average":      {"reward_scheme": "legacy_ais2026",     "emotion_type": "average"},
    "homeostatic_average": {"reward_scheme": "homeostatic",        "emotion_type": "average"},
    "homeostatic_vector":  {"reward_scheme": "homeostatic",        "emotion_type": "vector"},
    "signed_empathy":      {"reward_scheme": "homeostatic_signed", "emotion_type": "vector"},
    "learn_every5":        {"reward_scheme": "homeostatic",        "emotion_type": "vector",
                            "dqn": {"learn_every": 5}},
    "lr_div5":             {"reward_scheme": "homeostatic",        "emotion_type": "vector",
                            "dqn": {"learning_rate": M.PARAMS_DQN["learning_rate"] / 5}},
    # journal configuration decided on 27/10/2026: signed empathic term,
    # individual emotions, one learning update every 4 actions
    "global_v1":           {"reward_scheme": "homeostatic_signed", "emotion_type": "vector",
                            "dqn": {"learn_every": 4}},
}
BATCHES = {
    1: {"variants": ["legacy_average", "homeostatic_average", "homeostatic_vector"],
        "episodes": 100, "runs": 5, "folder": "beta"},
    # lr_div5 defined above but left out for now (14/10/2026)
    2: {"variants": ["homeostatic_vector", "signed_empathy", "learn_every5"],
        "episodes": 200, "runs": 3, "folder": "beta2"},
    # the full 2x2 of exp1 with the journal configuration, exp1's 500 episodes
    3: {"variants": ["global_v1"], "conditions": "ABCD",
        "episodes": 500, "runs": 3, "folder": "beta3"},
}
EXP1 = M.EXPERIMENTS["exp1_resource_obs"]   # conditions A, B, C, D
EVAL_EVERY = 10
EVAL_DENSE_FIRST = 20
# far from the main runner's seeds (102 + 1000 * condition + simulation)
BASE_SEED = 900_000

_HERE = os.path.dirname(os.path.abspath(__file__))
_INIT_ENV = M.init_env
_PARAMS_DQN = dict(M.PARAMS_DQN)


def apply_variant(variant):
    """set the 03_GT module to one variant. init_env reads REWARD_SCHEME and
    PARAMS_DQN at call time but binds its emotion_type default at import, so
    emotion_type is passed explicitly; the globals are also what the CSVs and
    config sidecars record."""
    M.REWARD_SCHEME = variant["reward_scheme"]
    M.EMOTION_TYPE = variant["emotion_type"]
    M.PARAMS_DQN = {**_PARAMS_DQN, **variant.get("dqn", {})}
    M.init_env = functools.partial(_INIT_ENV, emotion_type=variant["emotion_type"])


def run(variants, runs, episodes, out_root, letters="CD"):
    conditions = [c for c in EXP1 if c["label"][0] in letters]
    for v_idx, name in enumerate(VARIANTS):
        if name not in variants:
            continue
        apply_variant(VARIANTS[name])
        folder = os.path.join(out_root, name)
        os.makedirs(folder, exist_ok=True)
        for c_idx, condition in enumerate(conditions):
            for r in range(runs):
                seed = BASE_SEED + 1000 * v_idx + 100 * c_idx + r
                basename = f"{condition['label']}_seed{seed}"
                summary = os.path.join(folder, basename + "_episode_summary.csv")
                if os.path.exists(summary):
                    print(f"SKIP (already exists): {name}/{basename}")
                    continue
                M.write_run_config(condition, f"beta_{name}", seed, folder, basename,
                                   episode_number=episodes, eval_every=EVAL_EVERY,
                                   eval_dense_first=EVAL_DENSE_FIRST)
                M.run_simulation(condition, r, seed,
                                 step_file=os.path.join(folder, basename + "_step_data.csv"),
                                 summary_file=summary, episode_number=episodes,
                                 step_count=M.MAX_STEPS, verbose=True, save_steps=False,
                                 eval_every=EVAL_EVERY, eval_dense_first=EVAL_DENSE_FIRST,
                                 weights_file=os.path.join(folder, basename + "_weights.pt"))


def report(out_root):
    """greedy episode length per variant and condition: mean per window of
    training episodes, end-of-run value, and the effect of alpha D - C."""
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt

    files = glob.glob(os.path.join(out_root, "*", "*_episode_summary.csv"))
    if not files:
        raise SystemExit(f"no CSV under {out_root}")
    df = pd.concat(pd.read_csv(f).assign(variant=os.path.basename(os.path.dirname(f)))
                   for f in files)
    ev = df[df.phase == "eval"]
    # one value per episode and run: a sparse eval may share its episode
    # number with a dense one (episodes 9 and 19)
    ev = (ev.assign(cond=ev.condition.str[0])
            .groupby(["variant", "cond", "seed", "episode"], as_index=False).total_steps.mean())

    # windows of training episodes, averaged per run first (the run is the unit)
    windows = [(-1, -1), (0, 4), (5, 19), (20, 49), (50, 99), (100, 149), (150, 199),
               (200, 299), (300, 399), (400, 499)]
    table = {}
    for lo, hi in windows:
        w = ev[(ev.episode >= lo) & (ev.episode <= hi)]
        if w.empty:
            continue
        per_run = w.groupby(["variant", "cond", "seed"]).total_steps.mean()
        table["before" if hi < 0 else f"{lo}-{hi}"] = per_run.groupby(["variant", "cond"]).mean()
    table = pd.DataFrame(table)
    print("\nGreedy episode length per window of training episodes (mean over runs):")
    print(table.round(0).to_string())
    conds = set(table.index.get_level_values("cond"))
    for hi, lo, label in (("D", "C", "emotions seen"), ("B", "A", "emotions hidden")):
        if {hi, lo} <= conds:
            effect = table.xs(hi, level="cond") - table.xs(lo, level="cond")
            print(f"\nEffect of alpha per window, {hi} - {lo} ({label}; "
                  f"AIS 2026 paper: about +135 steps):")
            print(effect.round(0).to_string())

    # end of run: last 5 sparse evaluations of each run
    sparse = ev[(ev.episode >= 0) & ((ev.episode + 1) % EVAL_EVERY == 0)].sort_values("episode")
    end = (sparse.groupby(["variant", "cond", "seed"]).tail(5)
                 .groupby(["variant", "cond", "seed"]).total_steps.mean()
                 .groupby(["variant", "cond"]).agg(["mean", "std", "count"]))
    print("\nEnd of run (last 5 sparse evaluations):")
    print(end.round(1).to_string())

    curve = (ev.groupby(["variant", "cond", "episode"]).total_steps.mean()
               .unstack("episode"))
    variants = list(dict.fromkeys(curve.index.get_level_values(0)))
    fig, axes = plt.subplots(1, len(variants), figsize=(4.4 * len(variants), 3.6),
                             sharey=True, squeeze=False)
    for ax, variant in zip(axes[0], variants):
        for cond, style in (("A", ":"), ("B", "-."), ("C", "-"), ("D", "--")):
            if (variant, cond) in curve.index:
                row = curve.loc[(variant, cond)].dropna()
                ax.plot(row.index, row.values, style, marker=".",
                        label=f"{cond} (alpha = {0.0 if cond in 'AC' else 0.5})")
        ax.axvline(EVAL_DENSE_FIRST, color="grey", lw=0.8, ls=":")
        ax.set_title(variant)
        ax.set_xlabel("training episode")
    axes[0][0].set_ylabel("greedy episode length (steps)")
    axes[0][0].legend()
    fig.tight_layout()
    path = os.path.join(out_root, "beta_report.png")
    fig.savefig(path, dpi=150)
    print(f"\nFigure: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", type=int, choices=list(BATCHES), default=3)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None,
                        help="default: the variants of the batch")
    parser.add_argument("--runs", type=int, default=None,
                        help="default: the runs per condition of the batch")
    parser.add_argument("--episodes", type=int, default=None,
                        help="default: the episodes of the batch")
    parser.add_argument("--out", default=None, help="default: csv_results/<batch folder>")
    parser.add_argument("--report", action="store_true",
                        help="only print the tables and save the figure")
    args = parser.parse_args()
    batch = BATCHES[args.batch]
    out = args.out or os.path.join(_HERE, "csv_results", batch["folder"])
    if not args.report:
        run(args.variants or batch["variants"], args.runs or batch["runs"],
            args.episodes or batch["episodes"], out, batch.get("conditions", "CD"))
    report(out)


if __name__ == "__main__":
    main()
