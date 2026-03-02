# bar_plotter.py  (vertical bars, 0-100 values, labels, col_plot support)
from __future__ import annotations
import abc
from typing import Literal
import pandas as pd
import streamlit as st

# ------------- base -------------
class BarPlotter(abc.ABC):
    ENGINE: str

    @staticmethod
    @abc.abstractmethod
    def plot(
        df: pd.DataFrame,
        col_plot: st.delta_generator.DeltaGenerator,
        title: str = "Percentage by province",
    ) -> None:
        """Draw the chart inside the supplied Streamlit column."""


# ------------- matplotlib -------------
class MatplotlibPlotter(BarPlotter):
    ENGINE = "matplotlib"

    @staticmethod
    def plot(
        df: pd.DataFrame,
        col_plot: st.delta_generator.DeltaGenerator,
        title: str = "Percentage by province",
    ) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.6), 4))
        bars = ax.bar(df.index, df.percentage, color="#29b5e8")
        ax.set_ylabel("Percentage")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")
        ax.set_xlabel("")

        # add text labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,               # slightly above the bar
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=14,
            )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        col_plot.pyplot(fig)


# ------------- seaborn -------------
class SeabornPlotter(BarPlotter):
    ENGINE = "seaborn"

    @staticmethod
    def plot(
        df: pd.DataFrame,
        col_plot: st.delta_generator.DeltaGenerator,
        title: str = "Percentage by province",
    ) -> None:
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.6), 4))
        sns.barplot(x=df.index, y="percentage", data=df, ax=ax, color="#29b5e8")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")
        ax.set_xlabel("")

        # add text labels
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + .4,
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=14,
            )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        col_plot.pyplot(fig)


# ------------- plotly -------------
class PlotlyPlotter(BarPlotter):
    ENGINE = "plotly"

    @staticmethod
    def plot(
        df: pd.DataFrame,
        col_plot: st.delta_generator.DeltaGenerator,
        title: str = "Percentage by province",
    ) -> None:
        print("PLOTLYY",df.head())
        import plotly.express as px
        fig = px.bar(
            df,
            x="province",
            y="percentage",
            text="percentage",          # use the same column for labels
            title=title,
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(yaxis_tickformat=".1f", yaxis_ticksuffix="%")
        col_plot.plotly_chart(fig, use_container_width=True)


# ------------- pandas -------------
class PandasPlotter(BarPlotter):
    ENGINE = "pandas"

    @staticmethod
    def plot(
        df: pd.DataFrame,
        col_plot: st.delta_generator.DeltaGenerator,
        title: str = "Percentage by province",
    ) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.6), 4))
        df["percentage"].plot(kind="bar", ax=ax, color="#29b5e8")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0f}%")
        ax.set_xlabel("")

        # add text labels
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + .3,
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=14,
            )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        col_plot.pyplot(fig)


# ------------- altair -------------

class AltairPlotter(BarPlotter):
    ENGINE = "altair"

    @staticmethod
    def plot(
        df: pd.DataFrame,
        col_plot: st.delta_generator.DeltaGenerator,
        title: str = "Percentage by province",
        height: int = 600,  # Default height of 500 pixels

    ) -> None:
        import altair as alt

        bars = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("province:N", sort="-y", axis=alt.Axis(labelAngle=-90),   labelFontSize=12,  titleFontSize=14),
                y=alt.Y("percentage:Q", axis=alt.Axis(format=".1f", labelExpr="datum.value + '%'")),
                tooltip=["province", alt.Tooltip("percentage:Q", format=".1f", title="percentage")],
            )
        )
        text = bars.mark_text(
            align="center", baseline="bottom", dy=-2
        ).encode(text=alt.Text("percentage:Q", format=".1f"))
        chart = (bars + text).properties(title=title, height=height)
        col_plot.altair_chart(chart, use_container_width=True)
# ------------- factory -------------
ENGINES: dict[str, type[BarPlotter]] = {
    cls.ENGINE: cls
    for cls in (
        MatplotlibPlotter,
        SeabornPlotter,
        PlotlyPlotter,
        PandasPlotter,
        AltairPlotter,
    )
}


def get_plotter(engine: Literal["matplotlib", "seaborn", "plotly", "pandas", "altair"]) -> type[BarPlotter]:
    return ENGINES[engine]