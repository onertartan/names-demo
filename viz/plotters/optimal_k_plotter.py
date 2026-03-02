import pandas as pd
import numpy as np
from kneed import KneeLocator
from matplotlib import pyplot as plt
import streamlit as st

from clustering.base_clustering import BaseClustering
from clustering.models.kmeans import KMeansEngine
from clustering.models.gmm import GMMEngine

METRIC_OBJECTIVES = {
    "Silhouette_mean": "max",
    "ARI_mean": "max",
    "Consensus": "max",
    "DaviesBouldin_mean": "min",

    # Only present for GMM
    "BIC_mean": "min",
    "AIC_mean": "min",
}

class OptimalKPlotter:
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

        TITLE_FONTSIZE = 14
        AXIS_LABEL_FONTSIZE = 12
        TICK_LABEL_FONTSIZE = 11

        # ---- column specification (ORDER MATTERS) ----
        columns = [("Silhouette Score (cosine)", "Silhouette Score (cosine)", "max"),
            ("Silhouette Score (euclidean)", "Silhouette Score (euclidean)", "max")]

        if engine_class is KMeansEngine:
            columns.append(("Inertia", "Inertia (Elbow)", "min"))

        if engine_class is GMMEngine:
            columns.append(("AIC", "AIC", "min"))
            columns.append(("BIC", "BIC", "min"))

        # last column reserved for ARI (mean ± std)
        columns.append(("ARI", "ARI Stability", "max"))

        num_cols = len(columns)
        num_seeds_to_plot = min(num_seeds_to_plot, len(random_states))

        fig, axs = plt.subplots(
            num_seeds_to_plot + 1,
            num_cols,
            figsize=(4.8 * num_cols, 4.2 * (num_seeds_to_plot + 1)),
            dpi=200,
            sharex="col",
        )

        # ---- per-seed rows ----
        for i, seed in enumerate(random_states[:num_seeds_to_plot]):
            for j, (key, title, _) in enumerate(columns):
                ax = axs[i, j]
                if key.startswith("Silhouette"):
                    ax.plot(k_values, metrics_all[key][i], "o-")
                    ax.set_title(f"Seed {seed}: {title}", fontsize=TITLE_FONTSIZE)

                elif key == "Inertia":
                    ax.plot(k_values, metrics_all["Inertia"][i], "o-")
                    elbow = KneeLocator(
                        k_values,
                        metrics_all["Inertia"][i],
                        curve="convex",
                        direction="decreasing",
                    )
                    if elbow.elbow:
                        ax.axvline(elbow.elbow, color="r", linestyle="--")
                    ax.set_title(f"Seed {seed}: {title}", fontsize=TITLE_FONTSIZE)

                elif key in ("AIC", "BIC"):
                    ax.plot(k_values, metrics_all[key][i], "o-")
                    ax.set_title(f"Seed {seed}: {title}", fontsize=TITLE_FONTSIZE)

                else:  # ARI column → hidden for seed rows
                    ax.axis("off")

                ax.grid(True)
                ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

        # ---- mean row ----
        r = num_seeds_to_plot
        st.header(metrics_mean.keys())

        for j, (key, title, _) in enumerate(columns):
            ax = axs[r, j]

            if key.startswith("Silhouette"):
                ax.plot(k_values, metrics_mean[key], "o-")
                ax.set_title(f"Mean {title}", fontsize=TITLE_FONTSIZE)

            elif key == "Inertia":
                ax.plot(k_values, metrics_mean["Inertia"], "o-")
                elbow = KneeLocator(
                    k_values,
                    metrics_mean[key],
                    curve="convex",
                    direction="decreasing",
                )
                if elbow.elbow:
                    ax.axvline(elbow.elbow, color="r", linestyle="--")
                ax.set_title("Mean Inertia", fontsize=TITLE_FONTSIZE)

            elif key in ("AIC", "BIC"):
                ax.plot(k_values, metrics_mean[key], "o-")
                ax.set_title(f"Mean {title}", fontsize=TITLE_FONTSIZE)

            elif key == "ARI":
                ax.plot(k_values, ari_mean, "o-", label="Mean ARI")
                ax.fill_between(
                    k_values,
                    np.array(ari_mean) - np.array(ari_std),
                    np.array(ari_mean) + np.array(ari_std),
                    alpha=0.2,
                    label="±1 std",
                )
                ax.set_ylim(0, 1.05)
                ax.legend()
                ax.set_title("ARI Stability", fontsize=TITLE_FONTSIZE)

            ax.set_xlabel("Number of clusters (k)", fontsize=AXIS_LABEL_FONTSIZE)
            ax.grid(True)

        fig.tight_layout(pad=2.5)
        st.pyplot(fig)

    @staticmethod
    def print_optimal_k_analysis(df_summary):
        col1, col2 = st.columns(2)
        col1.write("Formatted results")
        col1.dataframe(OptimalKPlotter.style_metrics_dataframe(df_summary))
        col2.write("Raw results")
        col2.dataframe(df_summary)

    @staticmethod
    def style_metrics_dataframe(df: pd.DataFrame):
        display = pd.DataFrame(index=df.index)
        def mean_pm_std(mean_col_, std_col, prec=3):
            return (
                    df[mean_col_].map(lambda x: f"{x:.{prec}f}") +
                    " ± " +
                    df[std_col].map(lambda x: f"{x:.{prec}f}")
            )

        # ---- Always-present metrics ----
        display["Silhouette Score (cosine)"] = mean_pm_std("Silhouette_mean (cosine)", "Silhouette_std (cosine)")
        display["Silhouette Score (euclidean)"] = mean_pm_std("Silhouette_mean (euclidean)", "Silhouette_std (euclidean)")
        display["Davies–Bouldin"] = mean_pm_std("DaviesBouldin_mean", "DaviesBouldin_std" )
        display["ARI"] = mean_pm_std("ARI_mean", "ARI_std")

        # ---- GMM-only metrics (guarded) ----
        if "BIC_mean" in df.columns:
            display["BIC"] = mean_pm_std("BIC_mean", "BIC_std", prec=0)

        if "AIC_mean" in df.columns:
            display["AIC"] = mean_pm_std("AIC_mean", "AIC_std", prec=0)

        # ---- Highlighting logic ----
        def highlight_best(_mean_col):
            values = df[_mean_col]
            if METRIC_OBJECTIVES[_mean_col] == "max":
                best = values.idxmax()
            else:
                best = values.idxmin()

            return [
                "background-color: #d4f7d4" if idx == best else ""
                for idx in df.index
            ]

        styler = display.style

        for mean_col in METRIC_OBJECTIVES:

            if mean_col not in df.columns:
                continue
            label = (
                mean_col
                .replace("_mean", "")
                .replace("DaviesBouldin", "Davies–Bouldin")
            )
            if label == "Silhouette":
                label += " Score"

            if label in display.columns:
                styler = styler.apply(
                    lambda _, mc=mean_col: highlight_best(mc),  # ← FIX
                    axis=0,
                    subset=[label]
                )
            else:
                st.header(label+" not found in dataframe columns!"+str(df.columns))

        return styler