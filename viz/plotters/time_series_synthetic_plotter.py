import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


class TimeSeriesSyntheticPlotter:
    """Plots for the synthetic time-series sub-tab of the Experiment page.

    SyntheticDataPlotter's 2-D feature scatter would put year 1880 against
    year 1881 here, so the series are drawn on the year axis instead.
    """

    def plot_series_by_cluster(self, df_pivot: pd.DataFrame):
        """Overlay every generated series, coloured by its predicted cluster."""
        clusters = df_pivot["clusters"].to_numpy()
        df_features = df_pivot.drop(columns=["clusters"])
        years = df_features.columns.to_numpy()
        cluster_ids = np.unique(clusters)
        cmap = plt.get_cmap("tab10")

        fig, ax = plt.subplots(figsize=(9, 5))
        for row, cluster in zip(df_features.to_numpy(), clusters):
            ax.plot(years, row, color=cmap(int(cluster) % 10), alpha=0.3, linewidth=0.8)
        handles = [
            plt.Line2D([], [], color=cmap(int(cluster) % 10), label=f"Cluster {cluster}")
            for cluster in cluster_ids
        ]
        ax.legend(handles=handles, title="Clusters", loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        ax.set_xlabel("Year")
        ax.set_ylabel("z-score")
        ax.set_title("Generated series by predicted cluster")
        ax.set_xticks([year for year in years if int(year) % 10 == 0])
        ax.grid(True, alpha=0.25)

        col_plot, _ = st.columns([7, 3])
        col_plot.pyplot(fig)

    def build_prototype_preview_figure(self, years, protos, labels,
                                       samples=None, sample_labels=None):
        """Overlay class prototypes on the year axis; noisy sample draws are
        alpha-blended behind them when given. Returns the figure so tests can
        compare renderings pixel by pixel."""
        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(9, 5))
        if samples is not None:
            for row, cls in zip(samples, sample_labels):
                ax.plot(years, row, color=cmap(int(cls) % 10), alpha=0.15, linewidth=0.7)
        for idx, (proto, label) in enumerate(zip(protos, labels)):
            ax.plot(years, proto, color=cmap(idx % 10), linewidth=2.2, label=label)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                  title="Classes")
        ax.set_xlabel("Year")
        ax.set_ylabel("z-score")
        ax.set_title("Class prototypes (bold) with sample draws")
        ax.set_xticks([year for year in years if int(year) % 10 == 0])
        ax.grid(True, alpha=0.25)
        return fig

    def plot_prototype_preview(self, years, protos, labels,
                               samples=None, sample_labels=None):
        st.pyplot(self.build_prototype_preview_figure(
            years, protos, labels, samples, sample_labels))

    def build_separation_heatmap_figure(self, matrix, labels):
        """Signed-rho separation matrix as an annotated heatmap."""
        size = len(labels)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(size), labels=labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(size), labels=labels, fontsize=8)
        for i in range(size):
            for j in range(size):
                ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(matrix[i, j]) > 0.6 else "black")
        fig.colorbar(image, ax=ax, label="signed rho")
        ax.set_title("Separation matrix (signed rho)")
        fig.tight_layout()
        return fig

    def plot_separation_heatmap(self, matrix, labels):
        st.pyplot(self.build_separation_heatmap_figure(matrix, labels))
