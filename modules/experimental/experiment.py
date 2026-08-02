from modules.base_page_names import PageNames
from modules.experimental.synthetic_data_generator import BlobsSyntheticDataGenerator
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
        page_name,geo_column = self.page_name, self.geo_level

        tabs_main = [stx.TabBarItemData(id="tab_synthetic_clustering", title="Synthetic Data", description=""),
                     stx.TabBarItemData(id="tab_geo_clustering", title="Names Data", description="")]
        tab_selected = stx.tab_bar(data=tabs_main, default="tab_geo_clustering")

        if tab_selected == "tab_synthetic_clustering":
            tabs = [stx.TabBarItemData(id="blobs", title="Blobs Data", description=""),
                    stx.TabBarItemData(id="time_series", title="Time Series Data", description="")]
            tab_selected = stx.tab_bar(data=tabs, default="blobs")
        st.session_state["selected_tab_" + page_name] = tab_selected

        # st.session_state["selected_tab_" + page_name] = stx.tab_bar(data=tabs, default="tab_map")
        return tab_selected

    def preprocess_clustering(self, df, tab_main_selected):
        # data_generator parameter is for compatibility, it is passed as *args to tab_clustering
        if tab_main_selected == "tab_geo_clustering":
            return super().preprocess_clustering(df,"", tab_main_selected)
        else:
            return df
    def render(self):
        page_name, geo_level= self.page_name, self.geo_level
        gdf_borders = self.gdf[geo_level]

        st.session_state["geo_scale"] = "province"
        start_year, end_year = self.data["name"].select(pl.col("year")).min().item(),  self.data["name"].select(pl.col("year")).max().item()
        sidebar_controls_basic_setup(start_year, end_year)
        cols = st.columns([1, 1, 3, 2])
        tab_main_selected = self.render_tabs()
        if tab_main_selected == "tab_geo_clustering":
            name_surname_selection, selected_years, gender_list = render_gender_name_surname_filters(page_name,cols)
            df = self.preprocessing_initial_filtering(name_surname_selection, selected_years, gender_list, cols, geo_level)
            df = df.to_pandas().set_index(['year', geo_level]).sort_index()
            data_generator = None
        else: # make blobs synthetic data
            synthetic_kwargs = render_synthetic_data()
            data_generator = BlobsSyntheticDataGenerator(synthetic_kwargs)
            df,ground_truth_labels= data_generator.generate()
        df_pivot = self.tab_clustering(df, geo_level, "results/experiment", data_generator,"No scaling")

Experiment().run()
