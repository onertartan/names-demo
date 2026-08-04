import numpy as np
import pandas as pd
import streamlit as st
from kneed import KneeLocator
from matplotlib import pyplot as plt

from clustering.evaluation.cvi_registry import CVI_REGISTRY

# Panels per figure row-group; more metrics wrap into further figures.
MAX_PANEL_COLUMNS = 4

TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 11


class OptimalKPlotter:
    @staticmethod
    def _panel_specs(engine_class, metrics_all):
        """One spec per figure panel: every CVI_REGISTRY entry in registry
        order, then the engine's declared extras, then ARI. No metric name
        is written out by hand here."""
        specs = [
            {"kind": "cvi", "key": cvi.key, "title": cvi.label, "maximize": cvi.maximize}
            for cvi in CVI_REGISTRY.values()
        ]
        for key in engine_class.extra_metric_keys():
            if key in metrics_all:
                title = "Inertia (Elbow)" if key == "Inertia" else key
                specs.append({"kind": "extra", "key": key, "title": title})
        specs.append({"kind": "ari", "key": None, "title": "ARI Stability"})
        return specs

    @staticmethod
    def _selected_k_index(values, maximize):
        """Index of the selected k by the entry's polarity; None if the whole
        curve is NaN."""
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).any():
            return None
        return int(np.nanargmax(values)) if maximize else int(np.nanargmin(values))

    @staticmethod
    def _plot_curve_panel(ax, k_values, values, title, maximize=None, elbow=False):
        """One metric curve; NaN points simply drop out, an all-NaN metric
        gets a labelled panel, and CVI panels mark the selected k by the
        registry polarity."""
        k_list = list(k_values)
        values = np.asarray(values, dtype=float)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        if not np.isfinite(values).any():
            ax.text(0.5, 0.5, "unavailable\n(all NaN)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")
            return
        ax.plot(k_list, values, "o-")
        if elbow:
            try:
                locator = KneeLocator(k_list, values, curve="convex", direction="decreasing")
                if locator.elbow:
                    ax.axvline(locator.elbow, color="r", linestyle="--")
            except Exception:
                pass
        if maximize is not None:
            idx = OptimalKPlotter._selected_k_index(values, maximize)
            if idx is not None:
                ax.axvline(k_list[idx], color="g", linestyle="--", alpha=0.8)

    @staticmethod
    def build_optimal_k_figures(engine_class, num_seeds_to_plot, k_values, random_states,
                                metrics_all, metrics_mean, ari_mean, ari_std):
        """Build the sweep figures: panels chunked at MAX_PANEL_COLUMNS per
        figure, each figure keeping the per-seed rows plus the mean row."""
        specs = OptimalKPlotter._panel_specs(engine_class, metrics_all)
        num_seeds_to_plot = min(num_seeds_to_plot, len(random_states))
        k_list = list(k_values)
        chunks = [specs[i:i + MAX_PANEL_COLUMNS]
                  for i in range(0, len(specs), MAX_PANEL_COLUMNS)]

        figures = []
        for chunk in chunks:
            n_cols = len(chunk)
            fig, axs = plt.subplots(
                num_seeds_to_plot + 1,
                n_cols,
                figsize=(4.8 * n_cols, 4.2 * (num_seeds_to_plot + 1)),
                dpi=200,
                sharex="col",
                squeeze=False,
            )
            # ---- per-seed rows ----
            for seed in range(num_seeds_to_plot):
                for j, spec in enumerate(chunk):
                    ax = axs[seed, j]
                    if spec["kind"] == "ari":  # ARI has no per-seed curve
                        ax.axis("off")
                        continue
                    OptimalKPlotter._plot_curve_panel(
                        ax, k_list, metrics_all[spec["key"]][seed],
                        f"Seed {seed}: {spec['title']}",
                        maximize=spec.get("maximize"),
                        elbow=(spec["key"] == "Inertia"),
                    )
                    ax.grid(True)
                    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

            # ---- mean row ----
            row = num_seeds_to_plot
            for j, spec in enumerate(chunk):
                ax = axs[row, j]
                if spec["kind"] == "ari":
                    ax.plot(k_list, ari_mean, "o-", label="Mean ARI")
                    ax.fill_between(
                        k_list,
                        np.array(ari_mean) - np.array(ari_std),
                        np.array(ari_mean) + np.array(ari_std),
                        alpha=0.2,
                        label="±1 std",
                    )
                    ax.set_ylim(0, 1.05)
                    ax.legend()
                    ax.set_title("ARI Stability", fontsize=TITLE_FONTSIZE)
                else:
                    OptimalKPlotter._plot_curve_panel(
                        ax, k_list, metrics_mean[spec["key"]],
                        f"Mean {spec['title']}",
                        maximize=spec.get("maximize"),
                        elbow=(spec["key"] == "Inertia"),
                    )
                ax.set_xlabel("Number of clusters (k)", fontsize=AXIS_LABEL_FONTSIZE)
                ax.grid(True)

            fig.tight_layout(pad=2.5)
            figures.append(fig)
        return figures

    @staticmethod
    def plot_optimal_k_analysis(
            engine_class,
            num_seeds_to_plot,
            k_values,
            random_states,
            metrics_all,
            metrics_mean,
            ari_mean,
            ari_std,
            kwargs,
    ):
        st.header(f"Running {engine_class.__name__} Optimal k Analysis ({len(random_states)} seeds)")
        st.write("Using params:", kwargs)
        for fig in OptimalKPlotter.build_optimal_k_figures(
                engine_class, num_seeds_to_plot, k_values, random_states,
                metrics_all, metrics_mean, ari_mean, ari_std):
            st.pyplot(fig)

    @staticmethod
    def print_optimal_k_analysis(df_summary,using_same_data=False):
        # using_same_data: multiple runs with different random states on same data.
        # if same data is used ARI can be calculated, otherwise not (e.g. synthetic data generator is used)
        col1, col2 = st.columns(2)
        st.write("Results")
        st.dataframe(OptimalKPlotter.style_metrics_dataframe(df_summary,using_same_data))

    @staticmethod
    def style_metrics_dataframe(df: pd.DataFrame, using_same_data: bool):
        display = pd.DataFrame(index=df.index)

        def mean_pm_std(mean_col, std_col, prec=3):
            return (
                    df[mean_col].map(lambda x: f"{x:.{prec}f}") +
                    " ± " +
                    df[std_col].map(lambda x: f"{x:.{prec}f}")
            )

        # (display label, mean column, std column, objective, precision):
        # registry entries in registry order, then the engine extras that made
        # it into the summary. Objectives come from each entry's maximize
        # flag, never from a name check.
        column_specs = []
        if using_same_data:
            column_specs.append(("ARI", "ARI_mean", "ARI_std", "max", 3))
        for cvi in CVI_REGISTRY.values():
            column_specs.append((cvi.label, cvi.mean_column, cvi.std_column,
                                 "max" if cvi.maximize else "min", 3))
        column_specs.append(("BIC", "BIC_mean", "BIC_std", "min", 0))
        column_specs.append(("AIC", "AIC_mean", "AIC_std", "min", 0))

        highlight_specs = []
        for label, mean_col, std_col, objective, prec in column_specs:
            if mean_col not in df.columns:
                continue
            display[label] = mean_pm_std(mean_col, std_col, prec)
            highlight_specs.append((label, mean_col, objective))

        def highlight_best(mean_col, objective):
            values = df[mean_col]
            if not values.notna().any():
                return ["" for _ in df.index]
            best = values.idxmax() if objective == "max" else values.idxmin()
            return [
                "background-color: #d4f7d4" if idx == best else ""
                for idx in df.index
            ]

        styler = display.style
        for label, mean_col, objective in highlight_specs:
            styler = styler.apply(
                lambda _, mc=mean_col, obj=objective: highlight_best(mc, obj),
                axis=0,
                subset=[label],
            )
        return styler
