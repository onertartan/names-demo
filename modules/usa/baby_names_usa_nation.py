from modules.base_page_names import PageNames
import streamlit as st
import polars as pl
from viz.gui_helpers.base_page.helpers import sidebar_controls_basic_setup
from viz.gui_helpers.base_page_names.render_tab_selection import render_tab_selection
from viz.gui_helpers.base_page_names.render_tabs_helpers import  render_gender_name_surname_filters

class PageBabyNamesNation(PageNames):
    page_name = "baby_names_usa_nation"
    geo_level= None
    country = "usa"
    @staticmethod
    @st.cache_data
    def get_data():
        df = pl.read_parquet("data/preprocessed/usa/names_usa_nation.parquet")
        df_data = {"name":df }
        return df_data

PageBabyNamesNation().run()