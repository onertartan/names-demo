from __future__ import annotations
import abc
import random
from typing import Literal
from collections import defaultdict
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import altair as alt
from viz.gui_helpers.base_page_names.plot_helpers import get_title_statement
import plotly.graph_objects as go

from viz.gui_helpers.base_page_names.validate_df import validate_df
from viz.plotters.helpers import year_ticks


def resolve_label_overlaps(label_df: pd.DataFrame, min_gap: float =-1) -> pd.DataFrame:
    """
    Returns a copy with 'rank_label' column — spread apart so no two labels
    are closer than min_gap rank-units. Original 'rank' is untouched.
    """
    df_s = label_df.sort_values("rank").copy().reset_index(drop=True)
    pos = df_s["rank"].tolist()
    if min_gap==-1: # do not make any changes
        return df_s
    elif min_gap==-2:
        max_rank = max(pos)
        min_gap = max_rank//20
    for _ in range(1000):          # iterate until stable
        changed = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < min_gap:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1] = mid - min_gap / 2
                pos[i]     = mid + min_gap / 2
                changed = True
        if not changed:
            break

    df_s["rank_label"] = pos
    return df_s

# ------------- base -------------
class BumpPlotter(abc.ABC):
    ENGINE: str

    def __init__(self, gender: str, page_name: str):
        self.gender = gender
        self.page_name = page_name
        self.title = self._build_title()

    def _build_title(self) -> str:
        return f"Rank Evolution of {get_title_statement(self.gender, self.page_name)} Over Years"

    @staticmethod
    def _build_year_ticks(years: list[int], max_ticks: int = 12) -> list[int]:
        if not years:
            return []

        start, end = years[0], years[-1]
        span = end - start
        if span > 80:
            step = 5
        elif span > 35:
            step = 3
        elif span > 15:
            step = 2
        else:
            step = 1

        first_tick = ((start + step - 1) // step) * step
        ticks = list(range(first_tick, end + 1, step))
        if not ticks or ticks[0] != start:
            ticks = [start] + ticks
        return sorted(set(ticks))

    @staticmethod
    def _first_seen_points(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.sort_values(["name", "year"])
            .groupby("name", as_index=False)
            .first()[["name", "year", "rank"]]
        )

    @staticmethod
    def _get_max_rank(df: pd.DataFrame) -> int | None:
        rank_series = pd.to_numeric(df["rank"], errors="coerce")
        rank_series = rank_series.dropna()
        if rank_series.empty:
            return None,None
        return  int(rank_series.min()), int(rank_series.max())

    @abc.abstractmethod
    def plot(
        self,
        df: pd.DataFrame,
        col_plot: st.delta_generator.DeltaGenerator,
            *args
    ) -> None:
        pass


# ------------- matplotlib implementation -------------
class MatplotlibBumpPlotter(BumpPlotter):
    ENGINE = "Matplotlib"
    # show_column is not used (added for compatibility with bar plotters which accept 3 parameters)

    def plot(self, df: pd.DataFrame, col_plot: st.delta_generator.DeltaGenerator,show_column:str):
        validate_df(df)
        df_pivot = pd.pivot_table(
            df,
            values="rank",
            index="year",
            columns="name",
            aggfunc=lambda x: x,
            dropna=False
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.get_cmap("tab20", 20)
        colors = [cmap(i) for i in range(20)]

        for i, name in enumerate(df_pivot.columns):
            color = colors[i % len(colors)]
            series = df_pivot[name]
            not_nan = series.notna()
            segments = []
            current_segment = []

            for year, valid in zip(series.index, not_nan):
                if valid:
                    current_segment.append((year, series.loc[year]))
                else:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = []
            if current_segment:
                segments.append(current_segment)

            for j, segment in enumerate(segments):
                years, ranks = zip(*segment)
                ax.plot(
                    years, ranks,
                    marker="o", markersize=8, linewidth=2,
                    alpha=0.7, color=color,
                    label=name if j == 0 else None
                )

        min_rank,max_rank = self._get_max_rank(df)
        if max_rank is None:
            col_plot.warning("No rank data available for the selected filters.")
            return
        ax.invert_yaxis()
        ax.set_yticks(range(1, min(51, max_rank + 1)))
        ax.set_ylim(min(51, max_rank + 1), 0)
        years = [int(y) for y in df_pivot.index]
        tick_years = self._build_year_ticks(years)
        ax.set_xticks(tick_years)
        ax.set_xticklabels([str(y) for y in tick_years], rotation=45, ha="right")
        ax.margins(x=0.15, y=0.1)
        fig.tight_layout(pad=2.0)
        ax.set_title(self.title, fontsize=20, pad=20)
        ax.set_xlabel("Year")
        ax.set_ylabel("Rank")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        y_offset_first = defaultdict(int)
        y_offset_last = defaultdict(int)

        for name in df_pivot.columns:
            last_rank = df_pivot[name].iloc[-1]
            last_year = df_pivot.index[-1]
            offset_l = y_offset_last[last_rank] * 0.2
            ax.text(last_year + 0.1, last_rank + offset_l, name,
                    ha="left", va="center", fontsize=9, alpha=0.7)
            y_offset_last[last_rank] += 1

        for _, row in self._first_seen_points(df).iterrows():
            ax.text(
                row["year"] - 0.1,
                row["rank"] - 0.15,
                row["name"],
                ha="right",
                va="center",
                fontsize=9,
                alpha=0.7
            )

        col_plot.pyplot(fig)
        fig.savefig("temp/eps/X.pdf", format="pdf",dpi=300, bbox_inches="tight")
        plt.close(fig)

# ------------- plotly implementation -------------
class PlotlyBumpPlotter(BumpPlotter):
    ENGINE = "Plotly"

    def _selected_top_n(self) -> int | None:
        try:
            value = st.session_state.get(f"rank_{self.page_name}")
        except Exception:
            return None

        try:
            top_n = int(value)
        except (TypeError, ValueError):
            return None
        return top_n if top_n > 0 else None

    @staticmethod
    def _with_segment_ids(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["name", "year"]).copy()
        ordered_years = sorted(df["year"].dropna().unique().tolist())
        year_positions = {year: index for index, year in enumerate(ordered_years)}

        df["_year_position"] = df["year"].map(year_positions)
        starts_new_segment = (
            df.groupby("name")["_year_position"]
            .diff()
            .fillna(1)
            .ne(1)
        )
        df["_segment"] = starts_new_segment.groupby(df["name"]).cumsum()
        return df.drop(columns="_year_position")

    def plot(
        self,
        df: pd.DataFrame,
        col_plot: st.delta_generator.DeltaGenerator,
        show_column: str,
    ) -> None:
        validate_df(df)
        df_reset = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df.copy()

        top_n = self._selected_top_n()
        if top_n is not None:
            df_reset = df_reset[df_reset["rank"] <= top_n].copy()

        if df_reset.empty:
            col_plot.warning("No names are in the selected top-n range for this year interval.")
            return

        max_rank = top_n if top_n is not None else int(df_reset["rank"].max())
        names = sorted(df_reset["name"].dropna().unique().tolist())
        colors = px.colors.qualitative.Dark24
        color_by_name = {name: colors[index % len(colors)] for index, name in enumerate(names)}
        segmented_df = self._with_segment_ids(df_reset)

        fig = go.Figure()
        line_traces = []
        marker_traces = []

        for name in names:
            name_df = segmented_df[segmented_df["name"] == name]
            color = color_by_name[name]

            for _, segment_df in name_df.groupby("_segment", sort=False):
                if len(segment_df) < 2:
                    continue

                line_traces.append(
                    go.Scatter(
                        x=segment_df["year"],
                        y=segment_df["rank"],
                        name=name,
                        mode="lines",
                        line=dict(color=color, width=2),
                        legendgroup=name,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

            marker_traces.append(
                go.Scatter(
                    x=name_df["year"],
                    y=name_df["rank"],
                    name=name,
                    mode="markers",
                    marker=dict(color=color, size=8),
                    legendgroup=name,
                    showlegend=True,
                    customdata=name_df[["name"]],
                    hovertemplate=(
                        "Name: %{customdata[0]}<br>"
                        "Year: %{x}<br>"
                        "Rank: %{y}<extra></extra>"
                    ),
                )
            )

        for trace in line_traces:
            fig.add_trace(trace)
        for trace in marker_traces:
            fig.add_trace(trace)

        years = sorted(df_reset["year"].dropna().unique().tolist())
        tick_years = self._build_year_ticks([int(year) for year in years])
        fig.update_layout(
            width=1000,
            height=500,
            xaxis_title="Year",
            yaxis_title="Rank",
            margin=dict(l=120, r=120, t=80, b=50),
            legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="right", x=1.15),
            title=dict(text=self.title, font=dict(size=20), automargin=True, yref="paper"),
            yaxis_title_font=dict(size=14),
            xaxis_title_font=dict(size=14),
        )
        fig.update_yaxes(
            tickvals=list(range(1, max_rank + 1)),
            range=[max_rank + 0.5, 0.5],
            dtick=1,
            tickfont=dict(size=12),
        )
        fig.update_xaxes(
            showline=True,
            tickfont=dict(size=12),
            tickmode="array",
            tickvals=tick_years,
            ticktext=[str(year) for year in tick_years],
        )

        last_year = max(years)
        for _, row in df_reset[df_reset["year"] == last_year].iterrows():
            fig.add_annotation(
                x=row["year"],
                y=row["rank"],
                text=row["name"],
                showarrow=False,
                xshift=8,
                font=dict(size=11, color=color_by_name.get(row["name"])),
                xanchor="left",
            )

        for _, row in self._first_seen_points(df_reset).iterrows():
            fig.add_annotation(
                x=row["year"],
                y=row["rank"],
                text=row["name"],
                showarrow=False,
                xshift=-0,
                yshift=-10 if row["rank"] != 2 else 10,
                font=dict(size=11, color=color_by_name.get(row["name"])),
                xanchor="center",
            )

        col_plot.plotly_chart(fig, use_container_width=True)
        col_plot.dataframe(df)


# ------------- seaborn implementation -------------
class SeabornBumpPlotter(BumpPlotter):
    ENGINE = "Seaborn"

    def plot(self, df: pd.DataFrame, col_plot: st.delta_generator.DeltaGenerator, show_column:str) -> None:
        # show_column is not used (added for compatibility with bar plotters which accept 3 parameters)
        validate_df(df)
        df_pivot = pd.pivot_table(
            df, values="rank", index="year", columns="name",
            aggfunc=lambda x: x, dropna=False
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("tab20", len(df_pivot.columns))

        for i, name in enumerate(df_pivot.columns):
            series = df_pivot[name].dropna()
            sns.lineplot(
                x=series.index, y=series.values,
                ax=ax, color=palette[i], marker="o",
                markersize=8, linewidth=2, label=name
            )

        min_rank, max_rank = self._get_max_rank(df)
        if max_rank is None:
            col_plot.warning("No rank data available for the selected filters.")
            return
        ax.invert_yaxis()
        ax.set_yticks(range(1, min(51, max_rank + 1)))
        ax.set_ylim(min(51, max_rank + 1), 0)
        years = [int(y) for y in df_pivot.index]
        tick_years = self._build_year_ticks(years)
        ax.set_xticks(tick_years)
        ax.set_xticklabels([str(y) for y in tick_years], rotation=45, ha="right")
        ax.margins(x=0.15, y=0.1)
        ax.set_title(self.title, fontsize=20, pad=20)
        ax.set_xlabel("Year")
        ax.set_ylabel("Rank")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # --- endpoint labels ---
        y_offset_first = defaultdict(int)
        y_offset_last = defaultdict(int)
        for name in df_pivot.columns:
            series = df_pivot[name].dropna()
            if series.empty:
                continue
            first_year, last_year = series.index[0], series.index[-1]
            first_rank, last_rank = series.iloc[0], series.iloc[-1]
            if first_year!=df.index[0]:
                ax.text(first_year - 0.1, first_rank + y_offset_first[first_rank] * 0.2,
                        name, ha="right", va="center", fontsize=9, alpha=0.7)
                ax.text(last_year + 0.1, last_rank + y_offset_last[last_rank] * 0.2,
                        name, ha="left", va="center", fontsize=9, alpha=0.7)

            y_offset_first[first_rank] += 1
            y_offset_last[last_rank] += 1

        for _, row in self._first_seen_points(df).iterrows():
            ax.text(
                row["year"] + 0.1,
                row["rank"] - 0.15,
                row["name"],
                ha="left",
                va="center",
                fontsize=9,
                alpha=0.7
            )

        fig.tight_layout(pad=2.0)
        col_plot.pyplot(fig)
        plt.close(fig)
        col_plot.dataframe(df)

# ------------- altair implementation -------------

class AltairBumpPlotter(BumpPlotter):
    ENGINE = "Altair"

    def plot(self, df: pd.DataFrame, col_plot: st.delta_generator.DeltaGenerator,show_column:str) -> None:
        # show_column is not used (added for compatibility with bar plotters which accept 3 parameters)
        validate_df(df)
        min_rank, max_rank = self._get_max_rank(df)
        st.dataframe(df)
        labelFontSize=10
        labelFontSize_y_tick=10
        titleFontSize=16
        if max_rank is None:
            col_plot.warning("No rank data available for the selected filters.")
            return
        names = sorted(df["name"].dropna().unique().tolist())
        if len(names)<=10:
            color_scale = alt.Scale(scheme="tableau10", domain=names)
        else:
            color_scale = alt.Scale(scheme="tableau20", domain=names)

        years = [int(y) for y in df["year"].unique()]
        first_year,last_year= years[0], years[-1]
        tick_years = self._build_year_ticks(years)

        first_year_ranks = set(df[(df["year"] == first_year)  ]["rank"].unique() ) | {max_rank}
        last_year_ranks = set(df[(df["year"] == last_year)]["rank"].unique())
        last_year_ranks = sorted(int(v) for v in last_year_ranks)
        first_year_ranks = sorted(int(v) for v in first_year_ranks)
        st.header("YEAR RANKS (bump plotter line 460)")
        st.write(str(first_year_ranks))
        base = alt.Chart(df).encode(
            x=alt.X("year:O", title="Year",
                    axis=alt.Axis(labelFontSize=labelFontSize, titleFontSize=titleFontSize, values=tick_years)),
            y=alt.Y("rank:Q", scale=alt.Scale(domain=[max_rank + 0.1, 0.5]), axis=None),  # ← rank_label


            color=alt.Color("name:N", scale=color_scale, legend=alt.Legend(title="Name")),
        )

        lines = base.mark_line(strokeWidth=2)
        points = base.mark_point(filled=True, size=60)

        # endpoint labels — first year
        first_year = df["year"].min()
        last_year = df["year"].max()
        first_seen_df = (
            df.sort_values(["name", "year"])
            .groupby("name", as_index=False)
            .first()[["name", "year", "rank"]]
        )
        min_gap_first_year,min_gap_last_year=20,100

        first_year_labels = alt.Chart(resolve_label_overlaps(df[df["year"] == first_year],min_gap_first_year)).mark_text(
            align="center", dx=-50, dy=0, fontSize=labelFontSize#df=-55
        ).encode(
            x=alt.X("year:O"),
            y=alt.Y("rank_label:Q", scale=alt.Scale(domain=[max_rank + 0.1, 0.5]),axis=None),  # ← rank_label
            text=alt.Text("name:N"),
            color=alt.Color("name:N", scale=color_scale, legend=None)
        )

       # karen_seen_df = first_seen_df[ (first_seen_df["year"] != first_year) & (first_seen_df["name"] == "Karen")]
        first_seen_df = first_seen_df[ (first_seen_df["year"] != first_year)]

        first_seen_year_ranks = sorted(first_seen_df["rank"].unique())

        # Define the axis once as a variable to keep it DRY
        rank_axis = alt.Axis(
            values=first_seen_year_ranks,  # <-- Now a sorted list
            grid=True,
            labels=False,
            ticks=False,
            title=None,
            gridColor='lightgray',
            gridOpacity=0.5
        )
        threshold= 3 if max_rank<10 else max_rank
        odd_rows = alt.Chart(first_seen_df[(first_seen_df["rank"] % 2 == 1) & (first_seen_df["rank"] <= threshold)]).mark_text(
            align="right", dx=0, dy=-8, fontSize=labelFontSize
        ).encode(
            x=alt.X("year:O"),
            y=alt.Y("rank:Q", scale=alt.Scale(domain=[max_rank + 0.1, 0.5]),axis=rank_axis),
            text=alt.Text("name:N"),
            color=alt.Color("name:N", scale=color_scale, legend=None),
        )
        even_rows = alt.Chart(first_seen_df[(first_seen_df["rank"]%2==0) & (first_seen_df["rank"]<=threshold)]).mark_text(
            align="right", dx=0, dy=-8, fontSize=labelFontSize
        ).encode(
            x=alt.X("year:O"),
            y=alt.Y("rank:Q", scale=alt.Scale(domain=[max_rank + 0.1, 0.5]), axis=rank_axis
                    ),
            text=alt.Text("name:N"),
            color=alt.Color("name:N", scale=color_scale, legend=None),
        )
        """
        karen_row = alt.Chart(karen_seen_df).mark_text(
            align="right", dx=22, dy=-13, fontSize=labelFontSize
        ).encode(
            x=alt.X("year:O"),
            y=alt.Y("rank:Q", scale=alt.Scale(domain=[max_rank + 0.1, 0.5]), axis=rank_axis
                    ),
            text=alt.Text("name:N"),
            color=alt.Color("name:N", scale=color_scale, legend=None),
        )
        """
        first_seen_labels = odd_rows + even_rows #+karen_row


        last_labels = alt.Chart(resolve_label_overlaps(df[df["year"] == last_year],min_gap_last_year) ).mark_text(
            align="left", dx=38, fontSize=labelFontSize
        ).encode(
            x=alt.X("year:O"),
            y=alt.Y("rank_label:Q", scale=alt.Scale(domain=[max_rank + 0.1, 0.5]),axis=None),  # ← rank_label
            text=alt.Text("name:N"),
            color=alt.Color("name:N", scale=color_scale, legend=None),
        )
        left_axis = alt.Chart(df).mark_line(opacity=0).encode(
            y=alt.Y("rank:Q",
                    scale=alt.Scale(domain=[max_rank + 0.1, 0.5], nice=False),  # Ana skalayla aynı
                    axis=alt.Axis(
                        orient='left',  # Ekseni sola koy
                        values=first_year_ranks+first_seen_year_ranks,#[:4]+first_year_ranks[5:],#+first_seen_year_ranks[:5]+first_seen_year_ranks[6:],#[0:1],#+first_year_ranks[2:4]+first_year_ranks[5:],#+first_seen_year_ranks,#[0:1]+first_seen_year_ranks[-2:],#+last_year_ranks[:1]+last_year_ranks[2:3]+last_year_ranks[4:],#[0:3]+first_year_ranks[4:],  # Aynı tick değerleri
                        format="d",
                        labelFontSize=labelFontSize_y_tick,  # İsterseniz küçültün
                        title="Rank",  # Başlığı boş bırakın (isteğe bağlı)
                        labelOverlap=False,
                        labelAlign='right',  # sol eksende etiketler sağa hizalanır (grafik dışına)
                        labelPadding=4,
                        grid=True,  # <-- turn gridlines on
                        titlePadding=40
                    )
                    )
        )

        """
        df = df.copy()
        df["rank_left2"] = df["rank"]  # duplicate field
        left_axis2 = alt.Chart(df).mark_line(opacity=0).encode(
            y=alt.Y("rank_left2:Q",
                    scale=alt.Scale(domain=[max_rank + 0.1, 0.5], nice=False),
                    axis=alt.Axis(
                        orient='left',
                        values=first_year_ranks,#[-2:],
                        format="d",
                        labelFontSize=labelFontSize_y_tick,
                        title=None,
                        labelOverlap=False,
                        labelAlign='right',
                        labelPadding=10,
                        # grid=False (or omitted)
                    )
                    )
        )
        """
        df = df.copy()
        df["rank_right"] = df["rank"]

        right_axis = alt.Chart(df).mark_line(opacity=0).encode(
            y=alt.Y("rank_right:Q",
                    scale=alt.Scale(domain=[max_rank + 0.1, 0.5], nice=False),  # Ana skalayla aynı
                    axis=alt.Axis(
                        orient='right',  # Ekseni sağa koy
                        values=last_year_ranks[0:1]+last_year_ranks[3:],#[:7]+last_year_ranks[8::2],#+last_year_ranks[4:],  # Aynı tick değerleri
                        format="d",
                        labelFontSize=labelFontSize_y_tick,  # İsterseniz küçültün
                        title=None,  # Başlığı boş bırakın (isteğe bağlı)
                        labelOverlap=False,
                        labelAlign='left',  # Sağ eksende etiketler sola hizalanır (grafik dışına)
                        labelPadding=4,
                        grid=True,  # <-- turn gridlines on

                    )
                    )
        )
        chart = (lines + points + first_year_labels + first_seen_labels + last_labels + left_axis+right_axis).properties(
            width=1050, height=500,
            title=alt.TitleParams(text=self.title, fontSize=20, anchor="middle")
        ).resolve_scale(
        y='independent'   # <-- now genuinely separate scales, since fields differ
    ).resolve_axis(
            y='independent'  # <-- keeps left and right y-axes separate instead of merging
        ).configure_axisX(labelFontSize=10, titleFontSize=16).configure_axisY(
            labelAlign='left',titleFontSize=16,
            labelPadding=15,
            titlePadding=0,
        ).configure_legend(
            padding=40,
        )


        col_plot.altair_chart(chart, use_container_width=True)
        st.write("Unique names(607):"+str(sorted(df["name"].unique().tolist())))
        st.write("Number of unique names(608):"+ str(len(df["name"].unique().tolist())) )
        chart.save("temp/eps/top_male_20.pdf")  # saves as PDF

# ------------- factory -------------
ENGINES: dict[str, type[BumpPlotter]] = {
    cls.ENGINE: cls
    for cls in (
        MatplotlibBumpPlotter,
        SeabornBumpPlotter,
        PlotlyBumpPlotter,
        AltairBumpPlotter,
    )
}

def get_bump_plotter(
        engine: Literal["Matplotlib", "Seaborn", "Plotly", "Altair"],
        gender: str,
        page_name: str,
) -> BumpPlotter:
    return ENGINES[engine](gender, page_name)
