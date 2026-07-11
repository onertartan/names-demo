from typing import List
import pandas as pd
import polars as pl
import geopandas as gpd
import streamlit as st
from sklearn.metrics import adjusted_rand_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from clustering.models.time_series_k_means import TimeSeriesKMeansEngine
from clustering.models.trend_correlation_hierarchical import trend_correlation_hierarchical
from viz.plotters.bar_plotter_names import get_bar_plotter
from viz.plotters.bump_plotter import get_bump_plotter
from viz.plotters.line_plotter_names import get_line_plotter


def get_rank_plotter(tab_selected, plot_style, plotter_engine, gender, page_name):
    if "bump" in tab_selected:
        return get_bump_plotter(plotter_engine, gender, page_name)
    if "Bar" in plot_style:
        return get_bar_plotter(plotter_engine, gender, page_name)
    return get_line_plotter(plotter_engine, gender, page_name)

def rank_again(df: pl.DataFrame, geo_column: str) -> pl.DataFrame:
    accumulated = (
        df.group_by([geo_column, "name", "gender"])
        .agg(pl.col("count").sum().alias("count"))
        .with_columns(
            pl.col("count")
            .rank(method="min", descending=True)
            .over([geo_column, "gender"])
            .alias("rank")
        )
        .sort([geo_column, "rank", "count", "name"], descending=[False, False, True, False])
    )

    if "total_count" in df.columns:
        total_counts = (
            df.select(["year", geo_column, "total_count"])
            .unique()
            .group_by(geo_column)
            .agg(pl.col("total_count").sum().alias("total_count"))
        )
        accumulated = accumulated.join(total_counts, on=geo_column, how="left")

    return accumulated


def preprocess_for_nth_most_common_tab(df: pl.DataFrame, gdf_borders:gpd.GeoDataFrame,year: int, target_rank: int, include_top_n: bool,geo_column:str) -> gpd.GeoDataFrame:
    # Helper for Tab 2.1
    # (2.1.1:Select names and filter if they are in top-n  & 2.1.2:Nth most common)
    # Step 1: filter by year
    # If 'year' is a single value, wrap it in a list; otherwise use it as is
    year_list = year if isinstance(year, list) else [year]
    # Filter rows where the 'year' column is in year_list
    df_year = df if ("year" not in df.columns or year is None) else df.filter(pl.col("year").is_in(year_list))


    # Step 2: filter by rank
    if include_top_n:# Limits the rank limit to 5, since range-30 results in many names
        df_year_rank = df_year.filter(pl.col("rank") <= target_rank)
    else:
        df_year_rank = df_year.filter(pl.col("rank") == target_rank)
    df_year_rank=df_year_rank.to_pandas()
    # Step 3: aggregate names cleanly
    df_year_rank = df_year_rank.sort_values(["rank","gender"]).groupby(geo_column).apply(lambda g: "\n".join("-".join(rank_group["name"].tolist())
            for _, rank_group in g.groupby("rank", sort=False))).reset_index(name="name")
    # Step 4: attach geometry
    gdf_result = gdf_borders.merge(df_year_rank, on=geo_column)
    return gdf_result

def process_for_subtabs_bump_and_line_plots(df, gdf_borders, names_from_multi_select, year, rank, geo_column):
    # Helper for Tab 2.1.1 (under 2.1 plot map tab)
    # If 'year' is a single value, wrap it in a list; otherwise use it as is
    year_list = year if isinstance(year, list) else [year]
    # Filter rows where the 'year' column is in year_list
    df_year = df if ("year" not in df.columns or year is None) else df.filter(pl.col("year").is_in(year_list))
    if names_from_multi_select:
        # filter by name and rank
        df_result = df_year.filter(pl.col("name").is_in(names_from_multi_select) & (pl.col("rank") <= rank))
        # group by province and join names with newline (replaces groupby + apply lambda)
        df_result_grouped = df_result.group_by(geo_column).agg(pl.col("name").sort()).with_columns( pl.col("name").list.join("\n "))  # joins the list into a single string
        # merge with geodataframe borders (Polars → Pandas for GeoPandas compatibility)
        df_result_not_null = gdf_borders.merge(df_result_grouped.to_pandas(), left_on=geo_column, right_on=geo_column)
        # left join to keep all provinces including those with no match (#df_result_with_nulls)
        df_result = gdf_borders.merge(df_result_not_null[[geo_column, "name"]], left_on=geo_column, right_on=geo_column,how="left")
        return df_result, df_result_not_null
    return None,None

def preprocess_for_rank_bar_tabs(df: pl.DataFrame, use_rank_filtering:bool, include_all_years_option:bool, selected_names:List, top_n:int, show_column:str, secondary_top_k_filter:str, second_filter_option:int) -> pl.DataFrame:
    # Tabs 2.2,2.3,2.4
    # Drop geography by aggregating counts over year + name.
    df = df.group_by(["year", "name"]).agg(pl.col("count").sum()).sort(["year", "count"], descending=[False, True])
    if show_column=="ratio":
        df_year_totals = df.group_by("year").agg(pl.col("count").sum().alias("year_total"))
        df = df.join(df_year_totals, on="year").with_columns((pl.col("count") / pl.col("year_total")).alias("ratio")).sort(["year", "count"], descending=[False, True])

    if df.is_empty():
        return df.to_pandas()

    # vectorized rank per year
    df = df.with_columns(pl.col("count").rank(method="min", descending=True).over("year").alias("rank"))
    if use_rank_filtering:
        n_years = df.select(pl.col("year").n_unique()).item()
        if include_all_years_option :
            ever_top_n = df.filter(pl.col("rank") <= top_n)["name"].unique()
            df = df.filter(pl.col("name").is_in(ever_top_n))

            if secondary_top_k_filter :
                if  secondary_top_k_filter == "Max-rank":
                    threshold= df.select(pl.col("rank").max()).item()

                else:
                    threshold = int(secondary_top_k_filter.split("top-")[1])
                # Her ismin veri setinde toplam kaç yıl göründüğünü hesapla
                name_stats = (
                    df.group_by("name")
                    .agg([
                        pl.col("year").n_unique().alias("total_appeared_years"),
                        pl.col("year").filter(pl.col("rank") <= threshold).n_unique().alias("years_in_threshold")
                    ])
                )
                ################################
                st.dataframe(name_stats.to_pandas())
                st.header(str(second_filter_option)+", n_years: "+str(n_years))
                if second_filter_option==0:
                    # True: Sadece veri setindeki TÜM yıllarda (n_years) kesintisiz görünen
                    # ve her yıl belirlenen threshold içinde kalan isimler.
                    always_top_names = name_stats.filter(
                        pl.col("years_in_threshold") == n_years
                    ).select("name")
                    df = df.filter(  (pl.col("name").is_in(always_top_names["name"]))   )

                elif second_filter_option==1:
                    # sadece Her yıl görünmeyen (total_appeared_years < n_years), yani NaN içeren
                    # ancak fiilen göründüğü tüm yıllarda threshold içinde kalan isimler.
                    always_top_names = name_stats.filter(
                        (pl.col("total_appeared_years") < n_years) &  (pl.col("years_in_threshold") == pl.col("total_appeared_years"))
                    ).select("name")
                    df = df.filter(  (pl.col("name").is_in(always_top_names["name"])) )

            # elif second_filter_option==2:
                #    always_top_names = name_stats.filter(
                  #      (pl.col("total_appeared_years") < n_years)
                   # ).select("name")


        else:
            df = df.filter(pl.col("rank") <= top_n)

    if selected_names:
        df = df.filter(pl.col("name").is_in(selected_names))

    return df.to_pandas()


def preprocess_for_trend(df:pd.DataFrame,window):
    pivot_df = df.pivot(index='year', columns='name', values='ratio').fillna(0)
    # pivot_df: years=rows, names=columns

    # Original names order — for dendrogram labels and fcluster alignment
    original_names = pivot_df.columns.tolist()
    years = pivot_df.index.tolist()
    st.header("SHAPE OF pivot_df:" + str(pivot_df.shape))
    if window > 1:
        half_window = window // 2  # 11 için bu değer 5 olacaktır.
        # Hareketli ortalama uygulama ve boş değerleri temizleme
        pivot_df_processed = pivot_df.rolling(window=window, center=True).mean().dropna()
        # start and end are trimmed due to the averaging
        years = years[half_window:-half_window]
    else:
        pivot_df_processed = pivot_df
    return pivot_df,pivot_df_processed,years, original_names

def window_ari_analysis(df, n_cluster):
    k_values = [n_cluster]  # fixed k across all windows
    base_window = 11        # primary analysis window
    window_range = range(1,20,2)
    hc_labels_per_window = {}
    tsk_labels_per_window = {}

    for window in window_range:
        pivot_df, pivot_df_processed, years, original_names = preprocess_for_trend(df, window)

        # HC labels
        _, _, _, df_hc_labels = trend_correlation_hierarchical(
            pivot_df_processed, original_names, n_cluster
        )
        hc_labels_per_window[window] = df_hc_labels  # index=name, column='cluster'

        # TSK labels — scale from pivot_df_processed directly
        ts_features = TimeSeriesScalerMeanVariance().fit_transform(
            pivot_df_processed.T
        ).squeeze()  # (n_names, n_years)

        ts_features_df = pd.DataFrame(ts_features, index=original_names, columns=years)
        _, _, _, _, _, consensus_labels_all = TimeSeriesKMeansEngine.optimal_k_analysis(
            ts_features_df,
            random_states=range(0, 100),
            k_values=k_values,
            model_kwargs={"n_clusters":-1}
        )
        tsk_labels_per_window[window] = pd.DataFrame(
            index=ts_features_df.index,
            data={"cluster": consensus_labels_all[n_cluster]}
        )

    # ARI comparisons — base window vs all windows
    st.subheader("Sensitivity Analysis Results")

    base_hc = hc_labels_per_window[base_window]
    base_tsk = tsk_labels_per_window[base_window]

    rows = []
    for window in window_range:
        hc_w = hc_labels_per_window[window]
        tsk_w = tsk_labels_per_window[window]

        # Align everything on a common name set, identical order, before scoring
        common_hc_names = base_hc.index.intersection(hc_w.index)
        common_tsk_names = base_tsk.index.intersection(tsk_w.index)
        common_cross_names = hc_w.index.intersection(tsk_w.index)

        if len(common_hc_names) < len(base_hc) or len(common_hc_names) < len(hc_w):
            st.warning(
                f"Window={window}: HC name mismatch — "
                f"{len(common_hc_names)} common out of "
                f"{len(base_hc)} (base) / {len(hc_w)} (window)."
            )

        ari_hc = adjusted_rand_score(
            base_hc.loc[common_hc_names, "cluster"].astype(int),
            hc_w.loc[common_hc_names, "cluster"].astype(int),
        )
        ari_tsk = adjusted_rand_score(
            base_tsk.loc[common_tsk_names, "cluster"].astype(int),
            tsk_w.loc[common_tsk_names, "cluster"].astype(int),
        )
        ari_cross = adjusted_rand_score(
            hc_w.loc[common_cross_names, "cluster"].astype(int),
            tsk_w.loc[common_cross_names, "cluster"].astype(int),
        )

        rows.append({
            "Window": window,
            f"ARI(HC_w{base_window}, HC_w)": round(ari_hc, 3),
            f"ARI(TSK_w{base_window}, TSK_w)": round(ari_tsk, 3),
            "ARI(HC_w, TSK_w)": round(ari_cross, 3),
        })

    sensitivity_df = pd.DataFrame(rows).set_index("Window")
    st.dataframe(sensitivity_df, use_container_width=True)
    return sensitivity_df