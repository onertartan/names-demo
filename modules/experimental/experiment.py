from modules.base_page_names import PageNames
from modules.experimental.shape_library import ShapeInstance
from modules.experimental.synthetic_data_generator import BlobsSyntheticDataGenerator, \
    TimeSeriesSyntheticDataGenerator
import streamlit as st
import polars as pl
from viz.gui_helpers.base_page_names.render_tabs_helpers import render_gender_name_surname_filters, \
    render_synthetic_data
from viz.gui_helpers.base_page.helpers import sidebar_controls_basic_setup
import extra_streamlit_components as stx


class Experiment(PageNames):
    page_name = "Experiment"
    geo_level= "province"
    country = "turkiye"
    @staticmethod
    @st.cache_data
    def get_data():
        df = pl.read_parquet("data/preprocessed/population/names_baby_turkiye.parquet")
        df_data = {"name":df }
        return df_data

    def render_tabs(self):
        tabs_main = [stx.TabBarItemData(id="tab_synthetic_clustering", title="Synthetic Data", description=""),
                     stx.TabBarItemData(id="tab_geo_clustering", title="Names Data", description="")]
        tab_main_selected = stx.tab_bar(data=tabs_main, default="tab_geo_clustering")

        sub_tab_selected = None
        if tab_main_selected == "tab_synthetic_clustering":
            tabs = [stx.TabBarItemData(id="blobs", title="Blobs Data", description=""),
                    stx.TabBarItemData(id="time_series", title="Time Series Data", description="")]
            sub_tab_selected = stx.tab_bar(data=tabs, default="blobs")
        # Main-tab and sub-tab ids live in separate session keys:
        # BasePage.tab_clustering branches on the main id, so the sub id must
        # never overwrite it.
        self.session.set(self.keys.selected_tab, tab_main_selected)
        self.session.set(self.keys.selected_sub_tab, sub_tab_selected)
        return tab_main_selected, sub_tab_selected

    def preprocess_clustering(self, df, tab_main_selected):
        # data_generator parameter is for compatibility, it is passed as *args to tab_clustering
        if tab_main_selected == "tab_geo_clustering":
            return super().preprocess_clustering(df,"", tab_main_selected)
        else:
            return df
    @staticmethod
    def time_series_synthetic_kwargs():
        # Step-1 vertical slice: fixed three-class list. The class-builder UI
        # (render_time_series_synthetic_data) replaces this in the next stage.
        return {
            "instances": [ShapeInstance("peak", 1925, 15),
                          ShapeInstance("trough", 1955, 15),
                          ShapeInstance("level_shift", 1970)],
            "n_per_cluster": 20,
            "sigma": 0.3,
            "znorm": True,
            "amplitude_jitter": False,
            "seed": 0,
        }

    def render(self):
        page_name, geo_level= self.page_name, self.geo_level
        gdf_borders = self.gdf[geo_level]

        st.session_state["geo_scale"] = "province"
        start_year, end_year = self.data["name"].select(pl.col("year")).min().item(),  self.data["name"].select(pl.col("year")).max().item()
        sidebar_controls_basic_setup(start_year, end_year)
        cols = st.columns([1, 1, 3, 2])
        tab_main_selected, sub_tab_selected = self.render_tabs()
        if tab_main_selected == "tab_geo_clustering":
            name_surname_selection, selected_years, gender_list = render_gender_name_surname_filters(page_name,cols)
            df = self.preprocessing_initial_filtering(name_surname_selection, selected_years, gender_list, cols, geo_level)
            df = df.to_pandas().set_index(['year', geo_level]).sort_index()
            data_generator = None
        elif sub_tab_selected == "time_series":
            data_generator = TimeSeriesSyntheticDataGenerator(self.time_series_synthetic_kwargs())
            df, ground_truth_labels = data_generator.generate()
        else: # blobs synthetic data
            synthetic_kwargs = render_synthetic_data()
            data_generator = BlobsSyntheticDataGenerator(synthetic_kwargs)
            df,ground_truth_labels= data_generator.generate()
        df_pivot = self.tab_clustering(df, geo_level, "results/experiment", data_generator,"No scaling")

Experiment().run()
