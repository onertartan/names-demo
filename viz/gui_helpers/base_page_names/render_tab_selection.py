import extra_streamlit_components as stx
import streamlit as st

def render_tab_selection(page_name,geo_level=None):
    # if "selected_tab" not in st.session_state:
    #    st.session_state["selected_tab_"+self.page_name] = "map"
    tabs_main = [stx.TabBarItemData(id="tab_main_algorithmic", title="Clustering & Trend Analysis Tabs", description=""),
                 stx.TabBarItemData(id="tab_main_plot", title="Plot Tabs", description="")]

    tab_main_selected = stx.tab_bar(data=tabs_main, default="tab_main_algorithmic")

    if tab_main_selected == "tab_main_algorithmic":
        tabs=[stx.TabBarItemData(id="tab_name_trend_analysis", title="Name Trend Analysis", description="")]
        if geo_level:
            tabs.extend([stx.TabBarItemData(id="tab_geo_clustering", title="Geographical Clustering", description=""),
                    stx.TabBarItemData(id="tab_name_clustering", title="Name Clustering", description=""),
              stx.TabBarItemData(id="tab_time_windowed_aligned_clustering", title="Time-Window-Aligned Spatial Clustering", description="")])
        st.session_state["selected_tab_" +page_name] = stx.tab_bar(data=tabs, default="tab_name_trend_analysis")

    else:
        tabs=[stx.TabBarItemData(id="rank_bump", title="Rank Bump Plot", description=""),
        stx.TabBarItemData(id="rank_bar_line", title="Bar & Line Plot", description="")]
        if geo_level:
            tabs.append(stx.TabBarItemData(id="tab_map", title="Map Plot", description=""))
        st.session_state["selected_tab_" + page_name] = stx.tab_bar(data=tabs, default="rank_bump")
    return st.session_state["selected_tab_" +page_name]