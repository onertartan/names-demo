import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


class TimeSeriesSyntheticPlotter:
    """Plots for the synthetic time-series sub-tab of the Experiment page.

    SyntheticDataPlotter's 2-D feature scatter would put year 1901 against
    year 1902 here, so the series are drawn on the year axis instead.
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
