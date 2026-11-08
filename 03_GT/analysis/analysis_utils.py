"""
code shared by the 03_GT analysis notebooks: loading the runs, statistics and figures.

the unit of analysis is the run: end_of_training gives one value per (condition, seed), the mean
over its last LAST_N_EVAL greedy evaluation episodes, so that the correlated episodes of one run
are never counted as independent observations.
"""
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats
from scipy.ndimage import uniform_filter1d
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from visualisation_module_test import POSTER_BG, POSTER_PALETTE, apply_style, create_poster_config

LAST_N_EVAL = 10      # eval episodes averaged to define "end of training"
ALPHA_LEVEL = 0.05    # significance threshold
MIN_ETA_SQ = 0.06     # smallest partial eta squared for a factor to count as a moderator
TARGET_SEEDS = 20     # replications per condition of the full campaign
SMOOTH_WINDOW = 5     # eval episodes averaged in the learning curves (0 = raw)

METRIC_LABEL = {
    "depletion_step": "Depletion step (steps before the pool empties)",
    "gini_meals": "Gini coefficient of meals",
    "total_meals": "Total meals consumed per episode",
    "personal_reward": "Personal reward per agent (episode sum)",
}

# ---------------------------------------------------------------------------------------------
# chart style: recessive grid and axes, neutral text, series colour reserved for the marks
# ---------------------------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "baseline": "#c3c2b7"}
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]            # categorical slots, never cycled
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # light end clears 2:1


def apply_house_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "figure.dpi": 110, "font.size": 10,
        "axes.edgecolor": INK["baseline"], "axes.linewidth": 1.0,
        "axes.labelcolor": INK["secondary"], "axes.labelsize": 10,
        "axes.titlecolor": INK["primary"], "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": INK["grid"], "grid.linewidth": 0.8,
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "xtick.labelcolor": INK["secondary"], "ytick.labelcolor": INK["secondary"],
        "text.color": INK["primary"],
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0,
    })


# ---------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------


def load_runs(folder):
    """every *_episode_summary.csv of one folder, without the interrupted runs, with the derived metrics."""
    folder = Path(folder)
    files = sorted(folder.glob("*_episode_summary.csv"))
    if not files:
        raise FileNotFoundError(f"no *_episode_summary.csv under {folder.resolve()}")
    df = pd.concat([pd.read_csv(p).assign(source_file=p.name) for p in files], ignore_index=True)
    print(f"{df['source_file'].nunique()} run files, {len(df)} episode rows")
    assert df["reward_scheme"].nunique() == 1, "mixed reward schemes in one folder"

    # last train episode reached, not the row count: a finished run of beta3 misses one train row
    last = df[df["phase"] == "train"].groupby("source_file")["episode"].max()
    interrupted = sorted(last.index[last < last.max()])
    if interrupted:
        df = df[~df["source_file"].isin(interrupted)].reset_index(drop=True)
        print(f"{len(interrupted)} interrupted run(s) dropped (< {last.max() + 1} train episodes):")
        for name in interrupted:
            print(f"    {name}  ({last[name] + 1} train episodes)")

    dupes = int(df.duplicated(subset=["condition", "seed", "phase", "episode"]).sum())
    assert dupes == 0, f"{dupes} duplicated episode rows: check for a re-run written twice"

    # the episode ends when the pool hits 0, so total_steps is the depletion step, except
    # when the episode is cut off at max_steps with resource left (right-censoring)
    df["depletion_step"] = df["total_steps"]
    df["censored"] = (df["total_steps"] >= df["max_steps"]) & (df["resource_remaining"] > 0)
    meal_cols = [c for c in df.columns if re.fullmatch(r"meals_\d+", c)]
    df["total_meals"] = df[meal_cols].sum(axis=1)
    personal_cols = [c for c in df.columns if re.fullmatch(r"total_personal_reward_\d+", c)]
    df["personal_reward"] = df[personal_cols].mean(axis=1)
    return df


def seed_inventory(df):
    """seeds, runs and episodes per condition, with a warning below the campaign target."""
    inventory = (df.groupby("condition", observed=True)
                   .agg(seeds=("seed", "nunique"),
                        runs=("source_file", "nunique"),
                        train_episodes=("phase", lambda s: int((s == "train").sum())),
                        eval_episodes=("phase", lambda s: int((s == "eval").sum())))
                   .sort_index())
    inventory["seeds_missing_vs_target"] = TARGET_SEEDS - inventory["seeds"]
    display(inventory)
    if (inventory["seeds"] < TARGET_SEEDS).any():
        print(f"WARNING: only {int(inventory['seeds'].min())}-{int(inventory['seeds'].max())} seeds "
              f"per condition (target {TARGET_SEEDS}). Every test below is underpowered: treat the "
              f"p-values as provisional until the full campaign has landed.")
    else:
        print(f"All conditions at the {TARGET_SEEDS}-seed target.")


def report_censoring(ev, keys=("condition",)):
    """share of evaluation episodes cut off at max_steps with resource left, per group."""
    table = (ev.groupby(list(keys), observed=True)
               .agg(eval_episodes=("censored", "size"),
                    censored_episodes=("censored", "sum"),
                    censored_rate=("censored", "mean"),
                    max_steps=("max_steps", "max"))
               .sort_index())
    display(table.style.format({"censored_rate": "{:.1%}"}))
    overall = ev["censored"].mean()
    if overall > 0.10:
        print(f"WARNING: {overall:.1%} of eval episodes never depleted the pool and were cut off at "
              f"max_steps={int(ev['max_steps'].max())}. For those, the depletion step is a LOWER "
              f"BOUND: conditions differing only above the cap cannot be separated by these tests.")
    else:
        print(f"Censoring is limited ({overall:.1%} of eval episodes): the depletion step is "
              f"observed almost everywhere.")


def end_of_training(eval_df, group_keys, last_n=LAST_N_EVAL):
    """one row per (group, seed): mean over its last `last_n` evaluation episodes."""
    keys = list(group_keys) + ["seed"]
    tail = eval_df.sort_values("episode").groupby(keys, observed=True).tail(last_n)
    return (tail.groupby(keys, observed=True)
                .agg(depletion_step=("depletion_step", "mean"),
                     gini_meals=("gini_meals", "mean"),
                     total_meals=("total_meals", "mean"),
                     personal_reward=("personal_reward", "mean"),
                     censored_frac=("censored", "mean"),
                     n_eval=("episode", "size"),
                     from_episode=("episode", "min"),
                     to_episode=("episode", "max"))
                .reset_index())


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def mean_ci(values, conf=0.95):
    """mean and half-width of its t-based CI (0 when n < 2)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        return (float(values.mean()) if n else np.nan), 0.0
    half = stats.t.ppf(1 - (1 - conf) / 2, n - 1) * values.std(ddof=1) / np.sqrt(n)
    return float(values.mean()), float(half)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def hedges_g(a, b):
    """Hedges' g (bias-corrected Cohen's d) and its 95% CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / pooled
    g = d * (1 - 3 / (4 * (na + nb) - 9))
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb - 2)))
    return float(g), float(g - 1.96 * se), float(g + 1.96 * se)


def spearman_ci(rho, n, conf=0.95):
    """Fisher-z CI for Spearman's rho, with the Bonett-Wright variance correction."""
    if n < 6 or not np.isfinite(rho) or abs(rho) >= 1:
        return (np.nan, np.nan)
    delta = stats.norm.ppf(1 - (1 - conf) / 2) * np.sqrt((1 + rho ** 2 / 2) / (n - 3))
    return float(np.tanh(np.arctanh(rho) - delta)), float(np.tanh(np.arctanh(rho) + delta))


def jonckheere_terpstra(groups):
    """
    ordered-alternatives trend test (normal approximation, ties at 0.5). `groups` must be given
    in the hypothesised increasing order. unlike Kruskal-Wallis it uses the ordering of the
    levels, so it is the right rank test for a dose-response claim.
    """
    jt = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = np.asarray(groups[i])[:, None]
            b = np.asarray(groups[j])[None, :]
            jt += float((a < b).sum() + 0.5 * (a == b).sum())
    ns = np.array([len(g) for g in groups], dtype=float)
    total = ns.sum()
    mean = (total ** 2 - (ns ** 2).sum()) / 4
    var = (total ** 2 * (2 * total + 3) - (ns ** 2 * (2 * ns + 3)).sum()) / 72
    z = (jt - mean) / np.sqrt(var)
    return jt, float(z), float(2 * stats.norm.sf(abs(z)))


def anova_table(formula, data):
    """type-II ANOVA with partial eta squared: (fitted model, table)."""
    model = ols(formula, data=data).fit()
    table = anova_lm(model, typ=2)
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + table.loc["Residual", "sum_sq"])
    table.loc["Residual", "partial_eta_sq"] = np.nan
    table["sig"] = [stars(p) if pd.notna(p) else "" for p in table["PR(>F)"]]
    return model, table


def moderating_terms(table, factor):
    """interactions with `factor` meeting the rule of the method: p < .05 and partial eta^2 >= .06."""
    return [t for t in table.index if factor in t and ":" in t
            and table.loc[t, "PR(>F)"] < ALPHA_LEVEL and table.loc[t, "partial_eta_sq"] >= MIN_ETA_SQ]


def describe_groups(seed_df, group_col, metrics):
    """mean, 95% CI half-width and SD per group, for each metric."""
    rows = []
    for key, sub in seed_df.groupby(group_col, observed=True):
        row = {group_col: key, "n_seeds": len(sub)}
        for metric in metrics:
            m, half = mean_ci(sub[metric])
            row[f"{metric}_mean"] = m
            row[f"{metric}_ci95"] = half
            row[f"{metric}_sd"] = sub[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def assumption_checks(residuals, groups):
    """Shapiro-Wilk on the model residuals, Levene on the raw groups."""
    sw_stat, sw_p = stats.shapiro(residuals)
    lv_stat, lv_p = stats.levene(*groups)
    return pd.DataFrame([
        {"test": "Shapiro-Wilk (residual normality)", "statistic": sw_stat, "p": sw_p,
         "verdict": "ok" if sw_p >= ALPHA_LEVEL else "violated -> read the rank test"},
        {"test": "Levene (homoscedasticity)", "statistic": lv_stat, "p": lv_p,
         "verdict": "ok" if lv_p >= ALPHA_LEVEL else "violated -> read the rank test"},
    ])


def pairwise_ranktests(seed_df, metric, group_col, order):
    """all pairwise Mann-Whitney U tests with Holm-corrected p-values."""
    rows = []
    for i, left in enumerate(order):
        for right in order[i + 1:]:
            a = seed_df.loc[seed_df[group_col] == left, metric].to_numpy()
            b = seed_df.loc[seed_df[group_col] == right, metric].to_numpy()
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            g, lo, hi = hedges_g(a, b)
            rows.append({"pair": f"{left} vs {right}", "U": u, "p_raw": p,
                         "mean_diff": a.mean() - b.mean(),
                         "hedges_g": g, "g_lo": lo, "g_hi": hi})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_raw"])
    out["sig"] = out["p_holm"].map(stars)
    return out


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------


def learning_curve(ax, metric, series_specs, smooth=SMOOTH_WINDOW, show_ci=True, clip=None):
    """
    eval-phase metric vs episode, mean (+/- 95% CI) across seeds. `series_specs` holds one
    {"data", "color", "label", "linestyle"} dict per line. `show_ci=False` past ~4 series:
    overlapping bands become unreadable. `clip` bounds the band, e.g. (0, 1) for a Gini.
    """
    for spec in series_specs:
        agg = spec["data"].groupby("episode", observed=True)[metric].agg(["mean", "std", "count"])
        n = agg["count"].to_numpy()
        half = pd.Series(np.where(n > 1,
                                  stats.t.ppf(0.975, np.maximum(n - 1, 1))
                                  * np.nan_to_num(agg["std"].to_numpy()) / np.sqrt(np.maximum(n, 1)),
                                  0.0),
                         index=agg.index)
        centre = agg["mean"]
        if smooth and smooth > 1:
            centre = centre.rolling(smooth, center=True, min_periods=1).mean()
            half = half.rolling(smooth, center=True, min_periods=1).mean()
        if show_ci:
            low, high = centre - half, centre + half
            if clip is not None:
                low, high = low.clip(lower=clip[0]), high.clip(upper=clip[1])
            ax.fill_between(agg.index, low, high, color=spec["color"], alpha=0.13, linewidth=0)
        ax.plot(agg.index, centre, color=spec["color"], lw=2,
                linestyle=spec.get("linestyle", "-"), label=spec["label"])
    ax.set_xlabel("Training episode")
    ax.set_ylabel(METRIC_LABEL[metric])


def box_strip(ax, seed_df, metric, group_col, order, colors, ticklabels=None):
    """boxplot with every seed overlaid: at these sample sizes a box alone hides the n."""
    groups = [seed_df.loc[seed_df[group_col] == g, metric].to_numpy() for g in order]
    bp = ax.boxplot(groups, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK["primary"], lw=1.6),
                    whiskerprops=dict(color=INK["baseline"], lw=1.2),
                    capprops=dict(color=INK["baseline"], lw=1.2))
    for patch, colour in zip(bp["boxes"], colors):
        patch.set_facecolor(colour)
        patch.set_alpha(0.22)
        patch.set_edgecolor(colour)
        patch.set_linewidth(1.4)
    rng = np.random.default_rng(0)
    for i, (values, colour) in enumerate(zip(groups, colors), start=1):
        jitter = i + rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(jitter, values, s=34, color=colour, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(ticklabels or [str(o) for o in order])
    ax.set_ylabel(METRIC_LABEL[metric])


# ---------------------------------------------------------------------------------------------
# paper figures, in the house style of the LaTeX manuscript
# ---------------------------------------------------------------------------------------------

POSTER_YLABEL = {"depletion_step": "Steps to Depletion",
                 "gini_meals": "Gini Coefficient of Meals",
                 "personal_reward": "Personal Reward per Agent"}
POSTER_FILE_TAG = {"depletion_step": "steps", "gini_meals": "gini", "personal_reward": "reward"}
POSTER_SMOOTH = 5     # only ~50 eval points per run: a wider window would flatten the curve
POSTER_BOX_WIDTH = 0.30
POSTER_Z = stats.norm.ppf(0.975)

# "" is the full-width version. "_half" is sized for the side-by-side (a)/(b) layout of Results.tex,
# printed at 0.37 and 0.60 of the 12.2 cm LNCS line: halving the canvas takes the print reduction
# from ~23% to ~49%, so labels land near 8 pt instead of 4 pt. its boxplot is near-square because a
# 2.7 aspect at half width is too flat for the bracket stack, and the condition names and the CI
# note move to the LaTeX caption.
POSTER_LAYOUTS = {
    "": dict(fig_w=8.0, fig_h=5.0, fig_w_box=13.0,
             label_fs=16, tick_fs=14, legend_fs=14, legend_ncol=1, handlelen=2.5,
             box_label_fs=20, box_tick_fs=20, star_fs=14,
             line_w=None, pad=0.8,
             pt_size=34, pt_edge=1.2, pt_jitter=0.09,
             ci_in_ylabel=True, short_labels=False, tight=True),
    "_half": dict(fig_w=3.6, fig_h=3.4, fig_w_box=3.9,
                  label_fs=15, tick_fs=12, legend_fs=11, legend_ncol=1, handlelen=1.6,
                  box_label_fs=15, box_tick_fs=12, star_fs=9,
                  line_w=1.8, pad=0.15,
                  pt_size=11, pt_edge=0.6, pt_jitter=0.07,
                  ci_in_ylabel=False, short_labels=True, tight=False),
}


def _flat(label):
    return str(label).replace("\n", " | ")     # two-line tick labels are unreadable in print


def add_sig_brackets(ax, data, x_col, y_col, groups, star_fs=14, verbose=True):
    """pairwise Mann-Whitney U with Bonferroni correction, brackets on the significant pairs only."""
    pairs = list(combinations(range(len(groups)), 2))
    sig_pairs = []
    if verbose:
        print(f"\n--- Significance tests for {y_col} ---")
    for i, j in pairs:
        g1 = data[data[x_col] == groups[i]][y_col].dropna()
        g2 = data[data[x_col] == groups[j]][y_col].dropna()
        if len(g1) < 2 or len(g2) < 2:
            continue
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        p_adj = min(p * len(pairs), 1.0)
        star = stars(p_adj)
        if verbose:
            print(f"  {_flat(groups[i])} vs {_flat(groups[j])}: "
                  f"U={stat:.0f}, p={p:.4e}, p_adj={p_adj:.4e}  {star}")
        if star != "ns":
            sig_pairs.append((i, j, star))

    if not sig_pairs:
        if verbose:
            print("  No significant pairs.")
        return

    y_vals = data[y_col].dropna()
    y_range = y_vals.quantile(0.99) - y_vals.quantile(0.01)
    bar_tip = y_range * 0.03
    bar_sep = y_range * 0.13
    y_start = y_vals.max() + y_range * 0.05
    sig_pairs.sort(key=lambda t: t[1] - t[0])
    for k, (i, j, star) in enumerate(sig_pairs):
        y = y_start + k * bar_sep
        ax.plot([i, i, j, j], [y, y + bar_tip, y + bar_tip, y],
                color="black", linewidth=1.2, clip_on=False)
        ax.text((i + j) / 2, y + bar_tip * 1.2, star,
                ha="center", va="bottom", fontsize=star_fs, color="black", clip_on=False)
    ax.set_ylim(top=y_start + len(sig_pairs) * bar_sep + bar_sep * 1.5)


def _style_axes(ax, L):
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    sns.despine(ax=ax)
    plt.tight_layout(pad=L["pad"])


def _save_and_show(fig, fig_dir, name, tight):
    # tight=False keeps the canvas at exactly figsize: the LaTeX panel widths are computed from it
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{name}.{ext}", format=ext, bbox_inches="tight" if tight else None,
                    dpi=300, facecolor=fig.get_facecolor())
    display(fig)
    plt.close(fig)


def _poster_trajectory(traj_df, group_col, groups, metric, L, config, palette):
    line_w = L["line_w"] or config.line_width
    fig, ax = plt.subplots(figsize=(L["fig_w"], L["fig_h"]), dpi=config.dpi)
    fig.patch.set_facecolor(POSTER_BG)
    ax.set_facecolor(POSTER_BG)
    for i, group in enumerate(groups):
        color = palette[i % len(palette)]
        grouped = traj_df[traj_df[group_col] == group].groupby("episode", observed=True)[metric]
        xs = grouped.mean().index.values
        ys = uniform_filter1d(grouped.mean().to_numpy(dtype=float), size=POSTER_SMOOTH)
        sem = uniform_filter1d(grouped.sem().to_numpy(dtype=float), size=POSTER_SMOOTH)
        ax.fill_between(xs, ys - POSTER_Z * sem, ys + POSTER_Z * sem, alpha=0.15, color=color, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=line_w, label=_flat(group))
    ax.set_xlabel("Episode", fontsize=L["label_fs"], labelpad=4, color="black", fontweight="bold")
    ylabel = POSTER_YLABEL[metric] + ("\n(mean ± 95% CI, SEM)" if L["ci_in_ylabel"] else "")
    ax.set_ylabel(ylabel, fontsize=L["label_fs"], labelpad=8, color="black", fontweight="bold")
    ax.legend(title=None, fontsize=L["legend_fs"], ncol=L["legend_ncol"], columnspacing=1.0,
              frameon=True, facecolor="white", edgecolor="none", labelcolor="black",
              handlelength=L["handlelen"], handleheight=1.5, borderpad=0.5)
    ax.tick_params(axis="both", labelsize=L["tick_fs"], colors="black", pad=3)
    _style_axes(ax, L)
    return fig


def _poster_boxplot(box_df, group_col, groups, metric, L, config, palette, verbose):
    line_w = L["line_w"] or config.line_width
    fig, ax = plt.subplots(figsize=(L["fig_w_box"], L["fig_h"]), dpi=config.dpi)
    fig.patch.set_facecolor(POSTER_BG)
    ax.set_facecolor(POSTER_BG)
    sns.boxplot(data=box_df, x=group_col, y=metric, order=groups, hue=group_col, hue_order=groups,
                width=POSTER_BOX_WIDTH, linewidth=line_w * 0.8, showfliers=False, legend=False, ax=ax)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="gray", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # every seed over its box: fliers are off because these points already show the extremes.
    # the jitter is seeded, so re-running the cell writes the same figure
    rng = np.random.default_rng(0)
    for i, group in enumerate(groups):
        values = box_df.loc[box_df[group_col] == group, metric].to_numpy()
        jitter = i + rng.uniform(-L["pt_jitter"], L["pt_jitter"], size=len(values))
        ax.scatter(jitter, values, s=L["pt_size"], color=palette[i % len(palette)], zorder=3,
                   edgecolor=POSTER_BG, linewidth=L["pt_edge"])

    add_sig_brackets(ax, box_df, group_col, metric, groups, star_fs=L["star_fs"], verbose=verbose)
    ax.set_xlabel("")
    ax.set_ylabel(POSTER_YLABEL[metric], fontsize=L["box_label_fs"], labelpad=8, color="black", fontweight="bold")
    ax.set_xticks(range(len(groups)))
    labels = [g.split("\n")[0] for g in groups] if L["short_labels"] else groups
    ax.set_xticklabels(labels, fontsize=L["box_tick_fs"], rotation=0, ha="center", color="black", fontweight="bold")
    ax.tick_params(axis="x", pad=8, colors="black")
    ax.tick_params(axis="y", labelsize=L["tick_fs"], colors="black", pad=3)
    _style_axes(ax, L)
    return fig


def poster_figures(box_df, traj_df, group_col, groups, metrics, fig_dir, prefix):
    """
    trajectory and boxplot of each metric, in both layouts, shown inline and written as PDF + PNG.
    box_df has one row per (group, seed), traj_df one per (group, seed, episode). the style is
    applied inside a context, so the charts of the notebook keep theirs.
    """
    fig_dir = Path(fig_dir).resolve()
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("Seeds per condition:")
    display(box_df.groupby(group_col, observed=True)["seed"].nunique().reindex(groups).rename("n_seeds").to_frame())

    with plt.rc_context():
        config = create_poster_config()
        config.label_fontsize = config.label_size = 16
        config.tick_fontsize = config.tick_size = 14
        config.legend_fontsize = config.legend_size = 14
        apply_style(config)
        plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                             "axes.edgecolor": "black", "axes.labelcolor": "black",
                             "xtick.color": "black", "ytick.color": "black", "text.color": "black"})
        palette = sns.color_palette(POSTER_PALETTE)
        for metric in metrics:
            for suffix, L in POSTER_LAYOUTS.items():
                print(f"\n===== {POSTER_FILE_TAG[metric]}, layout '{suffix or 'full'}' =====")
                name = f"{prefix}_{POSTER_FILE_TAG[metric]}"
                fig = _poster_trajectory(traj_df, group_col, groups, metric, L, config, palette)
                _save_and_show(fig, fig_dir, f"{name}_trajectory{suffix}", L["tight"])
                fig = _poster_boxplot(box_df, group_col, groups, metric, L, config, palette, verbose=suffix == "")
                _save_and_show(fig, fig_dir, f"{name}_boxplot{suffix}", L["tight"])
