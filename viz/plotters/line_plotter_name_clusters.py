from __future__ import annotations

import abc
from typing import Literal

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import streamlit as st

# Column name BasePage.tab_clustering appends to the clustered pivot
CLUSTER_COLUMN = "clusters"


def clusters_from_pivot(clustered_pivot: pd.DataFrame, cluster_column: str = CLUSTER_COLUMN) -> pd.DataFrame:
    """Extract the name→cluster mapping from a clustered pivot (names as index)."""
    return pd.DataFrame(
        {"name": clustered_pivot.index.tolist(), "cluster": clustered_pivot[cluster_column].values}
    )


def split_clustered_pivot(
    clustered_pivot: pd.DataFrame, cluster_column: str = CLUSTER_COLUMN
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a clustered pivot into the (pivot_df, clusters_df) pair the plotters use.

    Expected input: names as index, years as columns, cluster labels in
    ``cluster_column`` — the frame returned by ``BasePage.tab_clustering`` for
    tab_name_trend_clustering. The returned pivot_df has years as index and
    names as columns (the orientation the plot engines draw from).
    """
    clusters_df = clusters_from_pivot(clustered_pivot, cluster_column)
    pivot_df = clustered_pivot.drop(columns=[cluster_column]).T
    return pivot_df, clusters_df


class ClusterLinePlotter(abc.ABC):
    ENGINE: str

    def __init__(
        self,
        pivot_df: pd.DataFrame,
        clusters_df: pd.DataFrame | None = None,
        title: str = "Name Clusters Over Time",
    ):
        if clusters_df is None:
            if CLUSTER_COLUMN not in pivot_df.columns:
                raise ValueError(
                    f"clusters_df was not given and pivot_df has no '{CLUSTER_COLUMN}' column. "
                    "Pass either (pivot_df, clusters_df) or a clustered pivot "
                    f"(names as index, years as columns, labels in '{CLUSTER_COLUMN}')."
                )
            pivot_df, clusters_df = split_clustered_pivot(pivot_df)
        self.pivot_df = pivot_df
        self.clusters_df = clusters_df
        self.title = title

    def _cluster_ids(self) -> list:
        if self.pivot_df.empty or self.clusters_df.empty:
            return []
        return self.clusters_df["cluster"].dropna().sort_values().unique().tolist()

    def _cluster_names(self, cluster_id) -> list[str]:
        return self.clusters_df.loc[self.clusters_df["cluster"] == cluster_id, "name"].tolist()

    def _cluster_frame(self, cluster_id) -> pd.DataFrame:
        cluster_names = self._cluster_names(cluster_id)
        frame = self.pivot_df[cluster_names].copy()
        frame["year"] = frame.index
        return frame

    def _year_ticks(self) -> list[int]:
        years = pd.Index(self.pivot_df.index).dropna().tolist()
        if not years:
            return []

        years = sorted({int(year) for year in years})
        if len(years) <= 12:
            return years

        start, end = years[0], years[-1]
        span = end - start
        if span > 80:
            step = 10
        elif span > 35:
            step = 5
        elif span > 15:
            step = 2
        else:
            step = 1

        first_tick = ((start + step - 1) // step) * step
        ticks = list(range(first_tick, end + 1, step))
        if not ticks or ticks[0] != start:
            ticks = [start] + ticks
        return sorted(set(ticks))

    def _legend_columns(self, item_count: int) -> int:
        return max(1, (item_count + 9) // 40)

    def _name_palette(self, item_count: int) -> list[str]:
        if item_count <= 10:
            return sns.color_palette("tab10", item_count).as_hex()
        if item_count <= 20:
            return sns.color_palette("tab20", item_count).as_hex()
        return sns.color_palette("husl", item_count).as_hex()

    @staticmethod
    def _render_target(col_plot):
        """Streamlit container to draw into; falls back to the page itself."""
        return col_plot if col_plot is not None else st

    @abc.abstractmethod
    def plot(self, col_plot=None) -> None:
        raise NotImplementedError


class MatplotlibClusterPlotter(ClusterLinePlotter):
    ENGINE = "Matplotlib"

    def _draw_cluster(self, ax, cluster_id, cluster_names: list[str], colors: list[str]) -> None:
        for name, color in zip(cluster_names, colors):
            ax.plot(
                self.pivot_df.index,
                self.pivot_df[name],
                color=color,
                alpha=0.7,
                linewidth=1.2,
                label=name,
                zorder=1,
            )

    def _decorate_axis(self, ax, cluster_id, cluster_names: list[str], year_ticks: list[int]) -> None:
        ax.set_title(f"Cluster {cluster_id} ({len(cluster_names)} names)")
        ax.set_ylabel("Ratio")
        if year_ticks:
            ax.set_xticks(year_ticks)
            ax.set_xticklabels([str(year) for year in year_ticks], rotation=45, ha="right")
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(True, alpha=0.2)
        legend_cols = self._legend_columns(len(cluster_names) + 1)
        ax.legend(
            loc="upper left",
            ncol=legend_cols,
            fontsize=7,
            frameon=True,
            handlelength=1.6,
            columnspacing=0.8,
            borderaxespad=0.3,
        )

    def plot(self, col_plot=None) -> None:
        cluster_ids = self._cluster_ids()
        if not cluster_ids:
            st.info("No cluster line plot can be drawn for the current data.")
            return

        n_clusters = len(cluster_ids)
        year_ticks = self._year_ticks()
        fig, axes = plt.subplots(
            n_clusters,
            1,
            figsize=(12, max(3, 2.8 * n_clusters)),
            sharex=False,
        )
        if n_clusters == 1:
            axes = [axes]

        fig.suptitle(self.title, fontsize=14, y=0.995)

        for ax, cluster_id in zip(axes, cluster_ids):
            cluster_names = self._cluster_names(cluster_id)
            colors = self._name_palette(len(cluster_names))

            self._draw_cluster(ax, cluster_id, cluster_names, colors)
            ax.plot(
                self.pivot_df.index,
                self.pivot_df[cluster_names].mean(axis=1),
                color="black",
                linewidth=3,
                label="Cluster mean",
                zorder=3,
            )
            self._decorate_axis(ax, cluster_id, cluster_names, year_ticks)

        for ax in axes:
            ax.set_xlabel("Year")
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        self._render_target(col_plot).pyplot(fig, use_container_width=True)
        plt.close(fig)


class SeabornClusterPlotter(MatplotlibClusterPlotter):
    ENGINE = "Seaborn"

    def _draw_cluster(self, ax, cluster_id, cluster_names: list[str], colors: list[str]) -> None:
        cluster_frame = self._cluster_frame(cluster_id)
        melted = cluster_frame.melt(id_vars="year", var_name="name", value_name="ratio")
        melted = melted[melted["name"].isin(cluster_names)]

        sns.lineplot(
            data=melted,
            x="year",
            y="ratio",
            hue="name",
            hue_order=cluster_names,
            palette=colors,
            units="name",
            estimator=None,
            ax=ax,
            alpha=0.7,
            linewidth=1.2,
            legend="full",
        )


class PlotlyClusterPlotter(ClusterLinePlotter):
    ENGINE = "Plotly"

    def plot(self, col_plot=None) -> None:
        cluster_ids = self._cluster_ids()
        if not cluster_ids:
            st.info("No cluster line plot can be drawn for the current data.")
            return

        fig = make_subplots(
            rows=len(cluster_ids),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.05,
            subplot_titles=[f"Cluster {cluster_id}" for cluster_id in cluster_ids],
        )
        year_ticks = self._year_ticks()

        for row_idx, cluster_id in enumerate(cluster_ids, start=1):
            cluster_names = self._cluster_names(cluster_id)
            cluster_frame = self._cluster_frame(cluster_id)
            colors = self._name_palette(len(cluster_names))

            for name, color in zip(cluster_names, colors):
                fig.add_trace(
                    go.Scatter(
                        x=cluster_frame["year"],
                        y=cluster_frame[name],
                        mode="lines",
                        line=dict(color=color, width=1.2),
                        opacity=0.7,
                        name=name,
                        legendgroup=f"cluster-{cluster_id}",
                        legendgrouptitle_text=f"Cluster {cluster_id}" if name == cluster_names[0] else None,
                        showlegend=True,
                    ),
                    row=row_idx,
                    col=1,
                )

            fig.add_trace(
                go.Scatter(
                    x=cluster_frame["year"],
                    y=cluster_frame.drop(columns="year").mean(axis=1),
                    mode="lines",
                    line=dict(color="black", width=3),
                    name="Cluster mean",
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )

            fig.update_yaxes(title_text="Ratio", row=row_idx, col=1)
            if year_ticks:
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=year_ticks,
                    ticktext=[str(year) for year in year_ticks],
                    tickangle=45,
                    showticklabels=True,
                    row=row_idx,
                    col=1,
                )

        fig.update_layout(
            title=self.title,
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                traceorder="grouped",
                yanchor="top",
                y=-0.18,
                xanchor="left",
                x=0,
                entrywidth=90,
                entrywidthmode="pixels",
            ),
            margin=dict(t=80, b=180, l=50, r=50),
        )
        self._render_target(col_plot).plotly_chart(fig, use_container_width=True)


class AltairClusterPlotter(ClusterLinePlotter):
    ENGINE = "Altair"

    def plot(self, col_plot=None) -> None:
        cluster_ids = self._cluster_ids()
        if not cluster_ids:
            st.info("No cluster line plot can be drawn for the current data.")
            return

        cluster_charts = []
        year_ticks = self._year_ticks()
        for cluster_id in cluster_ids:
            cluster_names = self._cluster_names(cluster_id)
            cluster_frame = self._cluster_frame(cluster_id)
            colors = self._name_palette(len(cluster_names))
            legend_cols = self._legend_columns(len(cluster_names))
            melted = cluster_frame.melt(id_vars="year", var_name="name", value_name="ratio")
            melted = melted[melted["name"].isin(cluster_names)]
            mean_df = pd.DataFrame(
                {
                    "year": cluster_frame["year"].values,
                    "ratio": cluster_frame.drop(columns="year").mean(axis=1).values,
                }
            )

            names_chart = (
                alt.Chart(melted)
                .mark_line(opacity=0.25, strokeWidth=1.2)
                .encode(
                    x=alt.X("year:O", title="Year", axis=alt.Axis(values=year_ticks, labelAngle=45)),
                    y=alt.Y("ratio:Q", title="Ratio"),
                    color=alt.Color(
                        "name:N",
                        scale=alt.Scale(domain=cluster_names, range=colors),
                        legend=alt.Legend(title="Name", columns=legend_cols, labelLimit=200),
                    ),
                )
            )
            mean_chart = (
                alt.Chart(mean_df)
                .mark_line(color="black", strokeWidth=3)
                .encode(
                    x=alt.X("year:O", title="Year", axis=alt.Axis(values=year_ticks, labelAngle=45)),
                    y=alt.Y("ratio:Q", title="Ratio"),
                )
            )
            title =f"Cluster {cluster_id}: " if len(cluster_ids) > 1 else ""
            cluster_charts.append(
                alt.layer(names_chart, mean_chart)
                .properties(title=f"{title} {len(cluster_names)} names")
            )

        chart = (
            alt.vconcat(*cluster_charts)
            .resolve_scale(x="independent", y="independent", color="independent")
            .properties(title=self.title)
        )

        self._render_target(col_plot).altair_chart(chart, use_container_width=True)


def _collect_cluster_plotters() -> dict[str, type[ClusterLinePlotter]]:
    def walk(cls: type[ClusterLinePlotter]) -> list[type[ClusterLinePlotter]]:
        collected: list[type[ClusterLinePlotter]] = []
        for subcls in cls.__subclasses__():
            collected.append(subcls)
            collected.extend(walk(subcls))
        return collected

    return {cls.ENGINE: cls for cls in walk(ClusterLinePlotter) if getattr(cls, "ENGINE", None)}


ENGINES: dict[str, type[ClusterLinePlotter]] = _collect_cluster_plotters()


def get_line_plotter_for_temporal_name_clusters(
    engine: Literal["Matplotlib", "Seaborn", "Plotly", "Altair"],
    pivot_df: pd.DataFrame,
    clusters_df: pd.DataFrame | None = None,
    title: str = "Temporal Trajectories of Name Clusters",
) -> ClusterLinePlotter:
    """
    Build a cluster line plotter.

    Either pass ``pivot_df`` (years × names) together with ``clusters_df``
    (columns ``name``/``cluster``), or pass only a clustered pivot
    (names as index, years as columns, labels in a ``clusters`` column) —
    the frame ``BasePage.tab_clustering`` returns.
    """
    return ENGINES[engine](pivot_df=pivot_df, clusters_df=clusters_df, title=title)


def line_plotter_name_clusters(
    pivot_df: pd.DataFrame,
    clusters_df: pd.DataFrame | None = None,
    col_plot=None,
    title: str = "Name Clusters Over Time",
) -> None:
    """
    Backward-compatible wrapper for the previous function-style API.
    """
    get_line_plotter_for_temporal_name_clusters(
        "Matplotlib",
        pivot_df=pivot_df,
        clusters_df=clusters_df,
        title=title,
    ).plot(col_plot=col_plot)
