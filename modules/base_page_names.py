from clustering.models.time_series_k_means import TimeSeriesKMeansEngine
from clustering.models.trend_correlation_hierarchical import trend_correlation_hierarchical, \
    trend_timeserieskmeans_hierarchical
from clustering.scaling import scale
from modules.base_page import BasePage
import streamlit as st
import locale
from utils import SessionAdapter
from utils.base_page_names.names_helpers import process_for_subtabs_bump_and_line_plots, \
    preprocess_for_nth_most_common_tab, \
    preprocess_for_rank_bar_tabs, rank_again, preprocess_for_trend, window_ari_analysis, get_rank_plotter
from utils.session_adapter import PageKeys
from viz import OptimalKPlotter
from viz.config import CLUSTER_COLOR_MAPPING, VA_POSITIONS, HA_POSITIONS
from viz.gui_helpers.base_page_names.plot_helpers import create_title_for_plot
from viz.gui_helpers.base_page.helpers import province_selector, sidebar_controls_basic_setup
from viz.gui_helpers.base_page_names.render_tab_selection import render_tab_selection
from viz.gui_helpers.base_page_names.tab_helpers_rank_and_trend import render_rank_and_trend_sub_tabs, get_window_size, get_n_cluster
from viz.gui_helpers.clustering_helpers import render_name_clustering_specific_ui, gui_scaler
from viz.gui_helpers.base_page_names.render_tabs_helpers import render_plot_map_sub_tab,  render_gender_name_surname_filters
from viz.plotters.dendogram_plotter import plot_dendrogram
from viz.plotters.geo_names_plotter import get_map_plotter
from viz.plotters.heatmap_plotter import plot_heatmap
from viz.plotters.line_plotter_name_clusters import get_line_plotter_for_temporal_name_clusters, clusters_from_pivot
from viz.plotters.network_plotter import plot_umap_tsne, plot_mds_provinces
import polars as pl
import geopandas as gpd
import pandas as pd

locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')

class PageNames(BasePage):
    geo_level = "province"
    country = "turkiye"


    def preprocessing_initial_filtering(self, name_surname_selection, selected_years, gender_list, cols, geo_column="province"):
        """
        Load the selected names/surnames dataset and apply the initial UI filters.

        The method selects the appropriate dataframe from ``self.data``, renders the
        geography selector, then filters rows by the chosen years, selected
        geographic units, and gender selection when a ``gender`` column is present.

        Parameters
        ----------
        name_surname_selection(str) : Dataset key chosen in the UI, such as ``"name"`` or ``"surname"``.
        selected_years (list)       : Years selected by the user.
        gender_list (list)          : Selected gender values, e.g. ``["male", "female"]``.
        cols (list)                 : Streamlit columns used to place the geography selector.
        geo_column (str)            : Geography column to filter on, for example ``"province"`` or ``"state"``.

        Returns
        df (pl.DataFrame)          : Filtered dataframe matching the current UI selections.
        """
        # 1. data is a dictionary with the key "names" (it can have another key "surnames")
        df = self.data[name_surname_selection.lower()]
        # 2. Get unique province(state) names
        if self.page_name != "Experiment":
            if geo_column:
                with cols[2]:
                    selected_provinces = province_selector(
                        df.select(pl.col(geo_column)).unique().to_series().to_list(),
                        key_prefix=f"{self.page_name}{geo_column}")
                # 3. Filter according to the selected provinces
                df = df.filter( pl.col(geo_column).is_in(selected_provinces)  )
        # 4. Filter according to the selected year(s)
        df = df.filter( pl.col("year").is_in(selected_years) )
        # 5. Filter according to the gender
        # If surname is not selected there is a gender column (the line below is sufficent)  if name_surname_selection != "surname":
        if "gender" in df.columns:
                df = df.filter( pl.col("gender").is_in(gender_list) )
        return df

    def preprocess_clustering(self, df, geo_scale,selected_tab,scaler):
        page_name,geo_column = self.page_name, self.geo_level
        unique_years = df.index.get_level_values(0).unique() # year(s)
        scaler_method = st.session_state["scaler"]
        df_year = df.loc[unique_years] # gerek var mı
        if self.session.get(self.keys.gender_list) == ["male", "female"]:  # if both sexes are selected
            df_year_male, df_year_female = df[df["gender"] == "male"].loc[unique_years], df[df["gender"] == "female"].loc[unique_years]
            overlapping_names = set(df_year_male["name"]) & set(df_year_female["name"])
            nonoverlapping_names =set(df_year_male["name"]) - set(df_year_female["name"])
            nonoverlapping_names2 =set(df_year_female["name"]) - set(df_year_male["name"])
            #st.header("Size overlapping:"+str(len(overlapping_names))+", Size nonoverlapping:"+str(len(nonoverlapping_names))+", Size nonoverlapping2:"+str(len(nonoverlapping_names2)))
            df_year_male['name'] = df_year_male.apply(
                lambda x: f"{x['name']}_male" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year_female['name'] = df_year_female.apply(
                lambda x: f"{x['name']}_female" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year = pd.concat([df_year_male, df_year_female])


       # else: # Use top-30 names for clustering otherwise, pivot table becomes a very high-dimensional sparse matrix
       #     df=df.filter((pl.col("rank") <= 30))
        # Get unique cumulative total counts over years(for each province)

        if selected_tab == "tab_name_trend_clustering":#tab_time_windowed_aligned_clustering":
            # List to hold each year's pivoted DataFrame
            pivots = []
            if geo_scale:#time aligned geospatial clustering bölümüne taşı(tab_name_trend_clustering'a ait değil)
                for year in unique_years:
                    st.dataframe(df_year.loc[year])
                    df_pivot = pd.pivot_table(df_year.loc[year], values='count', index=geo_scale, columns=['name'],
                                               aggfunc="first", dropna=False, fill_value=0) #!!! check aggfunc ="first" works
                    df_total_count = df.loc[year, "total_count"].groupby(level=0).first()
                    df_pivot = scale(scaler_method, df_pivot, df_total_count)
                    pivots.append(df_pivot)
                result = pd.concat(pivots, axis=1,keys=unique_years)
            else: # countrywide temporal
                 df_pivot = pd.pivot_table(df_year, values='count', index=df_year.index, columns=['name'],
                                          aggfunc="first", dropna=False, fill_value=0)
                 df_pivot=df_pivot.T
        else:
            df_total_counts = df[["total_count"]].groupby(level=["year", geo_column]).first()
            total_counts = df_total_counts.groupby(geo_column).sum()
            if isinstance(df_year.index, pd.MultiIndex):  # If applying temporal clustering (multiple years given as year)
                df_year = df_year.droplevel(0)  # Drop the first level year(position 0), if so index becomes province/state

            df_year = df_year.groupby([df_year.index, 'name']).agg({'count': 'sum'})
            # Merge
            df_year = df_year.merge(total_counts, left_index=True, right_index=True, how="outer")
            df_year = df_year.reset_index().set_index(geo_column)
            df_pivot = pd.pivot_table(df_year, values='count', index=df_year.index, columns=['name'], aggfunc=lambda x: x, dropna=False, fill_value=0)
            total_counts = df_year.loc[:, "total_count"]
            total_counts_unique=total_counts.groupby(level=0).first()
            df_pivot = scale(scaler_method, df_pivot, total_counts_unique)
            if selected_tab == "tab_name_clustering":  # transpose_for_name_clustering:
                df_pivot = df_pivot.T
        return df_pivot

    def render(self):
        page_name, geo_level= self.page_name, self.geo_level
        gdf_borders = self.gdf[geo_level] if geo_level else None
        self.session.set(self.keys.geo_scale,self.geo_level)#st.session_state["geo_scale"] = "state" if "usa" in self.page_name else "province"

        header = "Names & Surnames Analysis" if self.page_name == "names_surnames" else "Baby Names Analysis"
        start_year, end_year = self.data["name"].select(pl.col("year")).min().item(),  self.data["name"].select(pl.col("year")).max().item()
        sidebar_controls_basic_setup(start_year, end_year)

        cols = st.columns([1, 1, 3, 2])
        name_surname_selection, selected_years, gender_list = render_gender_name_surname_filters(page_name,cols)
        df = self.preprocessing_initial_filtering(name_surname_selection, selected_years, gender_list, cols, geo_level)
        tab_selected = render_tab_selection(page_name,geo_level)
        if  tab_selected=="tab_geo_clustering":  # tabs- 1.1, 1.2, 1.3
            self.tab_geo_and_name_clustering(df,  tab_selected, geo_level) # clustering tab
        elif tab_selected=="tab_name_trend_clustering":
            self.tab_name_trend_clustering(df, page_name, tab_selected, geo_level=None)
        elif tab_selected == "tab_name_trend_analysis":
            self.tab_name_trend_plotting(df, page_name, tab_selected, geo_level)  # clustering tab
        elif tab_selected == "tab_map":  # tab 2.3 Map Plot
            self.tab_plot_names_on_map(df, gdf_borders, page_name, geo_level)
        else:  # elif tab_selected in ["rank_bump", "rank_line_bar"]:  # Main Tab-2: Sub-tabs: 2-3
            self.tabs_rank_and_line_plot(df, page_name, tab_selected, geo_level)

    def tab_geo_and_name_clustering(self, df,  tab_selected, geo_level):
        # Geographical clustering & Name Clustering
        if "rank" in df.columns:
            max_rank = df.select(pl.col("rank")).max().item()
            use_data_option, top_n_names = render_name_clustering_specific_ui(df,30,tab_selected)# 30 is compatible with Türkiye, max_rank can be 5000 in USA
            if "top-n" in use_data_option:
                df = df.filter(pl.col("rank") <= top_n_names)

        # Convert Polars dataframe to Pandas dataframe
        df = df.to_pandas().set_index(['year', geo_level]).sort_index()
        gender=self.session.get(self.keys.gender)#gender=st.session_state['gender_radio_widget_' + page_name].lower()
        save_folder =  f"results/{self.country}/{tab_selected}/{gender}"

        scaler= gui_scaler(self.page_name)
        df=self.preprocess_clustering( df, geo_level, tab_selected, scaler)
        df_pivot = self.tab_clustering(df, geo_level, save_folder,None,scaler) # df, geo_scale, save_folder="", data_generator=None,

        if tab_selected == "tab_geo_clustering":
            if df_pivot is not None:
                plot_umap_tsne(df_pivot.copy(), CLUSTER_COLOR_MAPPING)
                plot_mds_provinces(df_pivot)

    def tab_name_trend_plotting(self, df, page_name, tab_selected, geo_level=None):
        """ Name Trend Plotting Tab: Uses the same ui with subtab2.1 rank bump"""
        clusters = []
        names = sorted(df["name"].unique(), key=locale.strxfrm)
        n_years=df["year"].n_unique()

        params_dict, use_province_or_cluster, show_column, \
            selected_n_cluster, show_provinces_separately, plotter_engine, plot_style,col2 =  render_rank_and_trend_sub_tabs(
            page_name, clusters, names, geo_level, tab_selected)
        if n_years == 1:
            st.warning("Note: For temporal analysis you should select a range of years and rank filtering / multiple names.")
            return
        if not params_dict["use_rank_filtering"] and params_dict["selected_names"] == []:
            st.warning("You have not selected any names/surnames or activate RANK FILTERING")
            return

        with col2:
            plot_trend=st.button("Plot trend", key="plot_trend_" + page_name,use_container_width=True)
            window = get_window_size(n_years)
        col_df,_=st.columns([9,1])

        if  plot_trend:
            df = preprocess_for_rank_bar_tabs(df, **params_dict)
            pivot_df,pivot_df_processed, pivot_df_normalized,years, original_names = preprocess_for_trend(df, window)
            if plot_trend:
                with col_df:
                     get_line_plotter_for_temporal_name_clusters(
                        plotter_engine,
                        pivot_df=pivot_df,
                        clusters_df=pd.DataFrame({"cluster": 1,"name":pivot_df.columns}),
                        title="Cluster Name Trajectories",
                    ).plot()

    def tab_name_trend_clustering(self, df, page_name, tab_selected, geo_level=None):
        """ Name Trend Plotting Tab: Uses the same ui with subtab2.1 rank bump"""
        clusters = []
        names = sorted(df["name"].unique(), key=locale.strxfrm)
        n_years=df["year"].n_unique()

        params_dict, use_province_or_cluster, show_column, \
            selected_n_cluster, show_provinces_separately, plotter_engine, plot_style,col2 =  render_rank_and_trend_sub_tabs(
            page_name, clusters, names, geo_level, tab_selected)

        #with col2:
        #    run_hierarchical_clustering = st.button("Apply hierarchical clustering and visualize", key="run_hierarchical_clustering_" + page_name,use_container_width=True)
        #    run_timeserieskmeans_clustering = st.button("Apply TimeSeriesKMeans clustering and visualize", key="run_timeserieskmeans_clustering_" + page_name,use_container_width=True)
        window = get_window_size(n_years)
        #    n_cluster = get_n_cluster()

        if n_years == 1:
            st.warning("Note: For temporal analysis you should select a range of years and rank filtering / multiple names.")
            return
        if not params_dict["use_rank_filtering"] and params_dict["selected_names"] == []:
            st.warning("You have not selected any names/surnames or activate RANK FILTERING")
            return

        df = preprocess_for_rank_bar_tabs(df, **params_dict)
        pivot_df,pivot_df_processed,pivot_df_normalized, years, original_names = preprocess_for_trend(df, window)

        # Cluster name trajectories: samples = names (rows), features = years (columns);
        # tab_clustering appends the labels as a "clusters" column on the frame it returns.
        df_clustered = self.tab_clustering(pivot_df_normalized.T, geo_level, "", None, "Z-norm")
        if df_clustered is None:  # no clustering algorithm selected yet
            return
        col_df,_=st.columns([9,1])
        get_line_plotter_for_temporal_name_clusters(
            plotter_engine,
            pivot_df=pivot_df,  # raw ratio trajectories (years × names)
            clusters_df=clusters_from_pivot(df_clustered),
            title="Temporal Trajectories of Name Clusters",
        ).plot(col_plot=col_df)


    def tab_plot_names_on_map(self, df:pl.DataFrame, gdf_borders:gpd.GeoDataFrame, page_name:str, geo_level:str):
        # Tab 2.3 Map Plot (2.1.1.Select Baby Names if they are in top-n , 2.1.2. Nth Most Common)
        names = sorted(df["name"].unique(), key=locale.strxfrm) # names variable contains surnames if surnames checkbox is selected
        rank, display_option,include_top_n,accumulate,plotter_engine = render_plot_map_sub_tab(names,page_name)
        def plot_map(year, title):
            effective_df = rank_again(df, geo_level) if accumulate else df
            # Display option 1: Show the nth most common baby names
            if display_option == "top-n to filter":  # Display option 2: Select single year, name(s) and top-n number to filter
                names_from_multi_select = self.session.get("names")
                df_result, df_result_not_null = process_for_subtabs_bump_and_line_plots(effective_df, gdf_borders,
                                                                                        names_from_multi_select, year, rank,
                                                                                        geo_level)
                if df_result_not_null is not None:
                        df_results.append(df_result_not_null)
            elif display_option == "nth most common":
                df_result = preprocess_for_nth_most_common_tab(effective_df, gdf_borders, year, rank, include_top_n, geo_level)
                df_results.append(df_result)

            if df_result is not None:
                map_plotter = get_map_plotter(plotter_engine, HA_POSITIONS, VA_POSITIONS, CLUSTER_COLOR_MAPPING)
                map_plotter.plot(df_result, title, col_plot, geo_level)
        # --- Plot map ---
        col_plot, _ = st.columns([99, 1])
        # Display results on map if a button is clicked
        if display_option:
            st.session_state["visualization_option"] = "matplotlib"
            df_results = []  # for animation?
            year_1, year_2 = st.session_state["year_1"], st.session_state["year_2"]
            if accumulate:
                title, _ = create_title_for_plot(rank, f"{year_1}-{year_2}", display_option, page_name)
                plot_map([year_1,year_2], title)
            else:
                for i, year in enumerate(sorted({year_1, year_2})):
                    title, _ = create_title_for_plot(rank, year, display_option, page_name)
                    plot_map(year,title)

    def tabs_rank_and_line_plot(self, df, page_name, tab_selected, geo_level=None):
        """Rank Bump Plot , Rank Bar & Line Plot"""
        clusters = []
        if "n_clusters_" + page_name in st.session_state:
            #clusters = range(1, self.session.get("n_clusters") + 1)
            clusters = range(1, st.session_state["n_clusters_" + page_name] + 1)
        gender = self.session.get("gender_radio_widget")
        names = sorted(df["name"].unique(), key=locale.strxfrm)
        params_dict,use_province_or_cluster,show_column,\
            selected_n_cluster,show_provinces_separately,plotter_engine,plot_style, col2 = render_rank_and_trend_sub_tabs(page_name, clusters, names, geo_level, tab_selected)
        plotter_object = get_rank_plotter(tab_selected, plot_style, plotter_engine, gender, page_name)

        if df.is_empty():
            st.error("You have not selected any data (provinces)")
            return
        if not params_dict["use_rank_filtering"] and params_dict["selected_names"] == []:
            st.error("You have not selected any names/surnames or activate RANK FILTERING")
            return
        col_plot, _ = st.columns([99, 1])
        col_plot.title("Counts by Name and Year")
        if use_province_or_cluster == "":  # empty string is for nationwide data (no geo_level available)
            df_plot = preprocess_for_rank_bar_tabs(df, **params_dict)
            plotter_object.plot(df_plot, col_plot, show_column)
        elif use_province_or_cluster == f"use {geo_level}s":

            if show_provinces_separately:
                for province in df[geo_level].unique(maintain_order=True).to_list():
                    df_province = df.filter(pl.col(geo_level) == province)
                    if not df_province.is_empty():
                        col_plot.subheader(province)
                        df_plot = preprocess_for_rank_bar_tabs(df_province,**params_dict)
                        plotter_object.plot(df_plot, col_plot,show_column)
            else:
                plotter_object.plot(preprocess_for_rank_bar_tabs(df,**params_dict), col_plot,show_column)
        elif use_province_or_cluster == "Use clusters" and selected_n_cluster:
            df_pivot = self.preprocess_clustering(df)
            df_pivot, _ = self.kmeans(df_pivot) # _ --> closest indices ( not used here)

            df_clusters = df_pivot["clusters"]
            df['clusters'] = df.index.get_level_values("province").map(df_clusters)
            if st.session_state["aggregate_totals_" + self.page_name]:
                df = df[df["clusters"].isin(selected_n_cluster)]
                plotter_object.plot(preprocess_for_rank_bar_tabs(df,**params_dict), col_plot,show_column)
            else:
                for cluster in selected_n_cluster:
                    df_cluster = df[df["clusters"] == cluster]
                    col_plot.subheader(f"Cluster {cluster}")
                    plotter_object.plot(preprocess_for_rank_bar_tabs(df_cluster,**params_dict), col_plot)
       # else:  # if not any selected, select all provinces
        #    plotter_object.plot(preprocess_for_rank_bar_tabs(df,params), col_plot)
