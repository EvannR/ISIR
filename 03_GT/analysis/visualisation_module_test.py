"""
Metrics Visualization Module for Multi-Agent Simulation Analysis

Provides publication-quality plots tuned for the Iste & Rabeau (2026) poster
template (white background, teal/navy headers, colorblind palette, ticks style).

Plot types
----------
density_plot   – KDE / filled density curves
boxplot        – box-and-whisker
violinplot     – violin (seaborn 0.13+ compatible)
evolution_plot – line chart with optional smoothing & CI band

Author: Iste Jérôme, Rabeau Evann
"""

import textwrap
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import uniform_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# Poster colour identity  (extracted from the Nanterre 2026 template)
# ─────────────────────────────────────────────────────────────────────────────
POSTER_PALETTE   = "colorblind"   # matches existing figures exactly
POSTER_BG        = "white"
POSTER_TITLE_CLR = "#1a1a2e"      # near-black navy used for section headers
POSTER_SPINE_CLR = "#444444"      # thin dark-grey axes
POSTER_GRID_CLR  = "#e8e8e8"      # very light grid (off by default)


# ─────────────────────────────────────────────────────────────────────────────
# StyleConfig
# ─────────────────────────────────────────────────────────────────────────────

class StyleConfig:
    """
    Central configuration for all visualization styles.

    Supports two naming conventions (both accepted):
    - Long form : title_fontsize, label_fontsize, tick_fontsize, legend_fontsize
    - Short form: title_size,     label_size,     tick_size,     legend_size

    Final rendered size = fontsize * font_scale.

    Use ``create_poster_config()`` to get a preset perfectly matched to the
    Iste & Rabeau Nanterre 2026 poster template.
    """

    def __init__(
        self,
        # ── font sizes (long form, canonical) ────────────────────────────────
        title_fontsize:  int   = 13,
        label_fontsize:  int   = 10,
        tick_fontsize:   int   = 9,
        legend_fontsize: int   = 9,
        # ── font sizes (short-form aliases) ───────────────────────────────────
        title_size:  Optional[int] = None,
        label_size:  Optional[int] = None,
        tick_size:   Optional[int] = None,
        legend_size: Optional[int] = None,
        # ── seaborn / mpl globals ─────────────────────────────────────────────
        palette:     str   = POSTER_PALETTE,
        style:       str   = "ticks",
        context:     str   = "poster",
        figsize:     Tuple = (5.5, 3.5),
        dpi:         int   = 300,
        line_width:  float = 1.8,
        marker_size: float = 4,
        font_scale:  float = 1.0,
        rc_params:   Optional[Dict] = None,
        # ── poster-specific tweaks ────────────────────────────────────────────
        title_color:      str  = POSTER_TITLE_CLR,
        spine_color:      str  = POSTER_SPINE_CLR,
        despine:          bool = True,
        show_grid:        bool = False,
        legend_frameon:   bool = False,
        title_fontweight: str  = "bold",
    ):
        # resolve short-form aliases (short wins if provided)
        self.title_fontsize  = title_size   if title_size   is not None else title_fontsize
        self.label_fontsize  = label_size   if label_size   is not None else label_fontsize
        self.tick_fontsize   = tick_size    if tick_size    is not None else tick_fontsize
        self.legend_fontsize = legend_size  if legend_size  is not None else legend_fontsize

        # expose both naming conventions as attributes
        self.title_size  = self.title_fontsize
        self.label_size  = self.label_fontsize
        self.tick_size   = self.tick_fontsize
        self.legend_size = self.legend_fontsize

        self.palette     = palette
        self.style       = style
        self.context     = context
        self.figsize     = figsize
        self.dpi         = dpi
        self.line_width  = line_width
        self.marker_size = marker_size
        self.font_scale  = font_scale
        self.rc_params   = rc_params or {}

        self.title_color      = title_color
        self.spine_color      = spine_color
        self.despine          = despine
        self.show_grid        = show_grid
        self.legend_frameon   = legend_frameon
        self.title_fontweight = title_fontweight

        self._apply()

    def _apply(self):
        """Push all settings to seaborn + matplotlib globals."""
        sns.set_style(self.style)
        sns.set_context(self.context, font_scale=self.font_scale)
        sns.set_palette(self.palette)

        s = self.font_scale
        plt.rcParams.update({
            # figure
            "figure.dpi":           self.dpi,
            "figure.facecolor":     POSTER_BG,
            "axes.facecolor":       POSTER_BG,
            # title
            "axes.titlesize":       self.title_fontsize  * s,
            "axes.titleweight":     self.title_fontweight,
            "axes.titlecolor":      self.title_color,
            # labels
            "axes.labelsize":       self.label_fontsize  * s,
            "axes.labelcolor":      self.spine_color,
            # ticks
            "xtick.labelsize":      self.tick_fontsize   * s,
            "ytick.labelsize":      self.tick_fontsize   * s,
            "xtick.color":          self.spine_color,
            "ytick.color":          self.spine_color,
            "xtick.major.width":    0.8,
            "ytick.major.width":    0.8,
            "xtick.major.size":     3.5,
            "ytick.major.size":     3.5,
            # spines
            "axes.edgecolor":       self.spine_color,
            "axes.linewidth":       0.8,
            # legend
            "legend.fontsize":      self.legend_fontsize * s,
            "legend.frameon":       self.legend_frameon,
            "legend.borderpad":     0.4,
            "legend.labelspacing":  0.3,
            # lines
            "lines.linewidth":      self.line_width,
            "lines.markersize":     self.marker_size,
        })
        plt.rcParams.update(self.rc_params)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_style(config: StyleConfig) -> None:
    """Re-apply a StyleConfig (useful after other cells reset mpl globals)."""
    config._apply()


def wrap_title(title: str, width: int = 55) -> str:
    """Wrap long titles for poster readability."""
    return "\n".join(textwrap.wrap(title, width))


def _finish_ax(ax, config: StyleConfig) -> None:
    """Poster-consistent finishing touches on an Axes."""
    if config.despine:
        sns.despine(ax=ax)
    if config.show_grid:
        ax.yaxis.grid(True, color=POSTER_GRID_CLR, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=3)


# ─────────────────────────────────────────────────────────────────────────────
# MetricsVisualizer
# ─────────────────────────────────────────────────────────────────────────────

class MetricsVisualizer:
    """
    Publication-quality plots matched to the Nanterre 2026 poster template.

    All methods return a ``matplotlib.figure.Figure`` and accept an optional
    ``figsize`` override so individual plots can deviate from the config default.
    """

    def __init__(self, config: Optional[StyleConfig] = None):
        self.config = config or StyleConfig()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _make_fig(self, figsize) -> Tuple:
        fig, ax = plt.subplots(
            figsize=figsize or self.config.figsize,
            dpi=self.config.dpi,
        )
        fig.patch.set_facecolor(POSTER_BG)
        ax.set_facecolor(POSTER_BG)
        return fig, ax

    def _set_labels(self, ax, title, xlabel, ylabel) -> None:
        if title:
            ax.set_title(
                wrap_title(title),
                fontweight=self.config.title_fontweight,
                color=self.config.title_color,
                pad=6,
            )
        if xlabel is not None:
            ax.set_xlabel(xlabel, labelpad=4)
        if ylabel is not None:
            ax.set_ylabel(ylabel, labelpad=4)

    def _maybe_rotate_xlabels(self, ax) -> None:
        """Rotate x-tick labels when they are long (avoids overlap)."""
        labels = [t.get_text() for t in ax.get_xticklabels()]
        if labels and max((len(l) for l in labels), default=0) > 12:
            ax.set_xticklabels(
                labels, rotation=25, ha="right",
                fontsize=self.config.tick_fontsize,
            )

    # ── density plot ──────────────────────────────────────────────────────────

    def density_plot(
        self,
        data:      pd.DataFrame,
        column:    str,
        hue:       Optional[str]            = None,
        title:     Optional[str]            = None,
        xlabel:    Optional[str]            = None,
        ylabel:    str                      = "Density",
        hue_order: Optional[List]           = None,
        figsize:   Optional[Tuple[int,int]] = None,
        fill:      bool                     = True,
        alpha:     float                    = 0.45,
    ):
        """KDE / filled density curves — best for comparing distributions."""
        fig, ax = self._make_fig(figsize)

        groups  = hue_order or (sorted(data[hue].unique()) if hue else [None])
        palette = sns.color_palette(self.config.palette, len(groups))

        for i, group in enumerate(groups):
            subset = (
                data[data[hue] == group][column].dropna()
                if hue else data[column].dropna()
            )
            sns.kdeplot(
                subset, fill=fill, alpha=alpha,
                label=str(group) if group is not None else column,
                color=palette[i], linewidth=self.config.line_width, ax=ax,
            )

        if hue:
            ax.legend(title=hue)

        self._set_labels(ax, title or f"Distribution of {column}", xlabel or column, ylabel)
        _finish_ax(ax, self.config)
        plt.tight_layout(pad=0.8)
        return fig

    # ── boxplot ───────────────────────────────────────────────────────────────

    def boxplot(
        self,
        data:      pd.DataFrame,
        column:    str,
        hue:       Optional[str]            = None,
        title:     Optional[str]            = None,
        xlabel:    Optional[str]            = None,
        ylabel:    Optional[str]            = None,
        hue_order: Optional[List]           = None,
        figsize:   Optional[Tuple[int,int]] = None,
        width:     float                    = 0.55,
        fliersize: float                    = 2.0,
    ):
        """Box-and-whisker — compact style matched to poster."""
        fig, ax = self._make_fig(figsize)

        sns.boxplot(
            data=data, x=hue, y=column,
            order=hue_order, hue=hue, hue_order=hue_order,
            width=width, linewidth=self.config.line_width * 0.8,
            fliersize=fliersize, legend=False, ax=ax,
        )

        self._maybe_rotate_xlabels(ax)
        self._set_labels(ax, title or column, xlabel or hue or "", ylabel or column)
        _finish_ax(ax, self.config)
        plt.tight_layout(pad=0.8)
        return fig

    # ── violin plot ───────────────────────────────────────────────────────────

    def violinplot(
        self,
        data:      pd.DataFrame,
        column:    str,
        hue:       Optional[str]            = None,
        title:     Optional[str]            = None,
        xlabel:    Optional[str]            = None,
        ylabel:    Optional[str]            = None,
        hue_order: Optional[List]           = None,
        figsize:   Optional[Tuple[int,int]] = None,
        inner:     str                      = "box",
        # legacy — silently ignored (removed in seaborn 0.13)
        split:     bool                     = False,
    ):
        """
        Violin plot — KDE shape + summary stats.
        ``split`` is kept for backwards compatibility but ignored.
        """
        fig, ax = self._make_fig(figsize)

        sns.violinplot(
            data=data, x=hue, y=column,
            order=hue_order, hue=hue, hue_order=hue_order,
            inner=inner, linewidth=self.config.line_width * 0.8,
            legend=False, ax=ax,
        )

        self._maybe_rotate_xlabels(ax)
        self._set_labels(ax, title or column, xlabel or hue or "", ylabel or column)
        _finish_ax(ax, self.config)
        plt.tight_layout(pad=0.8)
        return fig

    # ── evolution plot ────────────────────────────────────────────────────────

    def evolution_plot(
        self,
        data:      pd.DataFrame,
        x:         str,
        y:         str,
        hue:       Optional[str]            = None,
        hue_order: Optional[List]           = None,
        title:     Optional[str]            = None,
        xlabel:    Optional[str]            = None,
        ylabel:    Optional[str]            = None,
        figsize:   Optional[Tuple[int,int]] = None,
        smooth:              bool  = False,
        smoothing_window:    int   = 10,
        smooth_method:       str   = "uniform",   # "uniform" | "ewm"
        confidence_interval: Optional[float] = 95,
        markers:             bool  = False,
    ):
        """
        Line chart with optional smoothing and shaded CI band.

        Aggregates ``y`` by ``x`` per ``hue`` group, drawing:
        - smoothed (or raw) mean line
        - shaded ±z·SEM band when ``confidence_interval`` is not None
        """
        fig, ax = self._make_fig(figsize)

        groups  = hue_order or (sorted(data[hue].unique()) if hue else [None])
        palette = sns.color_palette(self.config.palette, len(groups))

        z = 0.0
        if confidence_interval is not None:
            from scipy.stats import norm
            z = norm.ppf(0.5 + confidence_interval / 200.0)

        for i, group in enumerate(groups):
            color  = palette[i]
            subset = data[data[hue] == group] if (hue and group is not None) else data

            grouped  = subset.groupby(x)[y]
            agg      = grouped.mean()
            xs, ys   = agg.index.values, agg.values
            ys_plot  = self._smooth(ys, smoothing_window, smooth_method) if smooth else ys

            if confidence_interval is not None:
                sem        = grouped.sem().values
                sem_smooth = self._smooth(sem, smoothing_window, smooth_method) if smooth else sem
                ax.fill_between(
                    xs,
                    ys_plot - z * sem_smooth,
                    ys_plot + z * sem_smooth,
                    alpha=0.15, color=color, linewidth=0,
                )

            ax.plot(
                xs, ys_plot,
                color=color,
                label=str(group) if group is not None else y,
                marker="o" if markers else None,
                linewidth=self.config.line_width,
                markersize=self.config.marker_size if markers else 0,
            )

        if hue:
            ax.legend(title=hue)

        self._set_labels(ax, title or f"Evolution of {y}", xlabel or x, ylabel or y)
        _finish_ax(ax, self.config)
        plt.tight_layout(pad=0.8)
        return fig

    # ── smoothing helper ──────────────────────────────────────────────────────

    @staticmethod
    def _smooth(values: np.ndarray, window: int, method: str) -> np.ndarray:
        if method == "ewm":
            return pd.Series(values).ewm(span=window, min_periods=1).mean().values
        return uniform_filter1d(values.astype(float), size=window)


# ─────────────────────────────────────────────────────────────────────────────
# Preset configs
# ─────────────────────────────────────────────────────────────────────────────

def create_poster_config() -> StyleConfig:
    """
    StyleConfig perfectly matched to the Iste & Rabeau Nanterre 2026 poster.

    Design choices
    --------------
    - White background, no top/right spines  → matches poster template
    - colorblind palette                     → matches existing figures
    - Compact figsize (5.5 × 3.5 in)        → many plots per poster column
    - Font sizes 13/10/9 pt                  → readable at poster scale
    - Frameless legend, thin spines          → clean, airy aesthetic
    - Near-black navy title colour           → matches section header colour
    """
    return StyleConfig(
        title_fontsize  = 13,
        label_fontsize  = 10,
        tick_fontsize   = 9,
        legend_fontsize = 9,
        font_scale      = 1.0,
        palette         = POSTER_PALETTE,
        style           = "ticks",
        context         = "poster",
        figsize         = (5.5, 3.5),
        dpi             = 300,
        line_width      = 1.8,
        marker_size     = 4,
        despine         = True,
        show_grid       = False,
        legend_frameon  = False,
        title_color     = POSTER_TITLE_CLR,
        spine_color     = POSTER_SPINE_CLR,
    )


def create_poster_wide_config() -> StyleConfig:
    """
    Same as ``create_poster_config`` but wider (10 × 3.5 in) — for trajectory
    plots that span a full poster column.
    """
    return StyleConfig(
        title_fontsize  = 13,
        label_fontsize  = 10,
        tick_fontsize   = 9,
        legend_fontsize = 9,
        font_scale      = 1.0,
        palette         = POSTER_PALETTE,
        style           = "ticks",
        context         = "poster",
        figsize         = (10, 3.5),
        dpi             = 300,
        line_width      = 1.8,
        marker_size     = 4,
        despine         = True,
        show_grid       = False,
        legend_frameon  = False,
        title_color     = POSTER_TITLE_CLR,
        spine_color     = POSTER_SPINE_CLR,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Usage example
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
from visualisation_module_test import (
    create_poster_config, create_poster_wide_config,
    MetricsVisualizer, apply_style, wrap_title,
)

# ── compact plots (boxplot, violin, density) ───────────────────────────────
config = create_poster_config()
viz    = MetricsVisualizer(config=config)

fig = viz.boxplot(
    data=df, column="gini_coef", hue="group_label",
    title="Gini Coefficient by Condition",
    ylabel="Gini Coefficient", xlabel="Condition",
    hue_order=group_order,
)
fig.savefig("boxplot.pdf", bbox_inches="tight")

# ── wide trajectory plot ──────────────────────────────────────────────────
config_wide = create_poster_wide_config()
viz_wide    = MetricsVisualizer(config=config_wide)

fig_evo = viz_wide.evolution_plot(
    data=df, x="episode", y="gini_coef", hue="group_label",
    smooth=True, smoothing_window=15, confidence_interval=95,
    title="Evolution of Gini Coefficient",
    hue_order=group_order,
)
fig_evo.savefig("evolution.pdf", bbox_inches="tight")
""")