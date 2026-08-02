from __future__ import annotations
import abc
from typing import Literal
import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from viz.gui_helpers.base_page_names.plot_helpers import (
    get_title_statement
)
import plotly.express as px

from viz.gui_helpers.base_page_names.validate_df import validate_df


class LinePlotter(abc.ABC):
    ENGINE: str

    def __init__(self, gender: str, page_name: str):
        self.gender = gender
        self.page_name = page_name
        self.title = self._build_title()

    def _build_title(self) -> str:
        return f"Temporal Trend of {get_title_statement(self.gender, self.page_name)}"

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_df(df)
        return df.sort_values(["name", "year"])

    @abc.abstractmethod
    def plot(
        self,
        df: pd.DataFrame,
        col_plot: st.delta_generator.DeltaGenerator,
        show_column: str
    ) -> None:
        pass


class MatplotlibLinePlotter(LinePlotter):
    ENGINE = "Matplotlib"

    def plot(self, df, col_plot, show_column="count"):
        df = self._prepare_df(df)

        fig, ax = plt.subplots(figsize=(12, 7))

        for name, sub in df.groupby("name"):
            ax.plot(sub["year"], sub[show_column], label=name)

        ax.set_title(self.title)
        ax.set_xlabel("Year")
        ax.set_ylabel(show_column.title())

        # optional: legend control (important for many names)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        col_plot.pyplot(fig)
        plt.close(fig)


class SeabornLinePlotter(LinePlotter):
    ENGINE = "Seaborn"

    def plot(self, df, col_plot, show_column):
        df = self._prepare_df(df)

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(data=df, x="year", y=show_column, hue="name", ax=ax)

        ax.set_title(self.title)
        ax.set_xlabel("Year")
        ax.set_ylabel(show_column.title())
        # legend outside (critical for readability)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        col_plot.pyplot(fig)

        plt.close(fig)

class PlotlyLinePlotter(LinePlotter):
    ENGINE = "Plotly"

    def plot(self, df, col_plot, show_column):
        df = self._prepare_df(df)

        fig = px.line(
            df,
            x="year",
            y=show_column,
            color="name",
            markers=True,
            title=self.title
        )

        fig.update_layout(
            legend=dict(orientation="v"),
            hovermode="x unified"
        )
        col_plot.plotly_chart(fig, use_container_width=True)


class AltairLinePlotter(LinePlotter):
    ENGINE = "Altair"

    def plot(self, df, col_plot, show_column="count"):
        df = self._prepare_df(df)

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y(f"{show_column}:Q", title=show_column.title()),
                color=alt.Color("name:N", legend=alt.Legend(title="Name"))
            )
            .properties(title=self.title)
        )
        col_plot.altair_chart(chart)
        st.dataframe(df)




# ------------- factory -------------
ENGINES: dict[str, type[LinePlotter]] = {
    cls.ENGINE: cls
    for cls in (
        MatplotlibLinePlotter,
        SeabornLinePlotter,
        PlotlyLinePlotter,
        AltairLinePlotter,
    )
}


def get_line_plotter(
    engine: Literal["Matplotlib", "Seaborn", "Plotly", "Altair"],
    gender: str,
    page_name: str,
) -> LinePlotter:
    return ENGINES[engine](gender, page_name)