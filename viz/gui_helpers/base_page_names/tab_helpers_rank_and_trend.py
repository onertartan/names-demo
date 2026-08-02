import extra_streamlit_components as stx
import streamlit as st
# TREND helpers
def get_n_cluster():
    return st.slider( label="Select number of clusters to observe)", min_value=2, max_value=15, value=2, step=1,key="n_cluster_trend" )
def get_window_size(n_max):
    col1,_=st.columns([1,2])
    max_window_size= max(2,min(n_max//5,21))
    window_size = col1.slider(
    label="Temporal Smoothing Window (Years)",
    min_value=1,  # Prevents window=0 (NaN error)
    max_value=max_window_size,  # Upper bound suitable for capturing multi-decade naming shifts
    value=1,  # Default: 1 year (No smoothing / Raw data mode)
    step=2,  # Restricts to odd numbers for symmetric centering
    help="Select 1 to view raw annual frequencies. Select 3 or higher to smooth out short-term noise and highlight underlying demographic trajectories.",
        key="window_size",
)
    # 2. Add a dynamic label to give the user explicit context
    if window_size == 1:
        st.caption("📈 Raw annual data (no smoothing applied).")
    else:
        st.caption(f"📊  {window_size}-year centered moving average.")
    return window_size

def render_rank_and_trend_sub_tabs_helper_rank_filtering_panel(col_1, page_name, names):
    expression_in_sentence = "names or surnames" if page_name == "names_surnames" else "baby names"
    selected_names = col_1.multiselect(f"Filter {expression_in_sentence} (optional)", names)
    use_rank_filtering = col_1.toggle("Use rank filtering", value=True)
    with col_1.container(border=True):
        col_1_1, _ = st.columns([3, 1])
        top_n = col_1_1.selectbox(f"Filter (appeared at least once in) top-n",
                                  range(1, 11) if "usa" in page_name else range(1, 31), index=4,
                                  key="rank_" + page_name,
                                  disabled=not use_rank_filtering)
        include_all_years = col_1_1.radio("Select an option",
                                          ["Show Only Years When Names Are in Top-n",
                                           "Include All Years for Names Ever in Top-n"],
                                          key="include_all_years", disabled=not use_rank_filtering)
        include_all_years = include_all_years=="Include All Years for Names Ever in Top-n"
        thresholds = [10, 20, 50, 100, 200, 300, 400, 500, 600, 1000, 2000, 3000, 4000, 5000, 6000,7000,8000,9000,10000, 12000, 20000]
        options = ["Max-rank"]
        for t in thresholds:
            options.append(f"Always in top-{t}")

        key = "always_top_filter_" + page_name
        disabled = not use_rank_filtering or not include_all_years
        secondary_top_k_filter = col_1_1.selectbox(
            'Secondary filter (only applies if "include all years option" is selected', options, key=key,
            disabled=disabled)
        options = ["Show names that always appeared in top-k every year",
                   "Show names with missing years (NaN) that appeared in top-k when ranked",
                   "Show all"]

        selected_label = st.radio("Select how to apply the second filter:", options)
        second_filter_option = options.index(selected_label)

    return selected_names, use_rank_filtering, top_n, include_all_years, secondary_top_k_filter, second_filter_option


def render_rank_and_trend_sub_tabs(page_name, clusters, names, geo_level, tab_selected):
    """ Helper function for rendering
      - 'Rank Bump Plot'
     - 'Rank Bar & Line Plot' (sub-tabs 2.2 & 2.3 of 'Plots Tab')
     - 'Name Trend Plotting' (1.1  of 'Clustering & Trend Analysis Tab')
     - Name Trend Clustering (1.2 of 'Clustering & Trend Analysis Tab') """
    col_1, col_23,col_4,col_5 =st.columns([3,3,1,1])
    col_2, col_3 = st.columns([1,2])
    if geo_level and not "trend" in tab_selected:
        show_provinces_separately = col_23.toggle(
            f"Show provinces separately(does not aggregate counts for selected provinces)", value=False)
    else:
        show_provinces_separately = False

    selected_names, use_rank_filtering, top_n, include_all_years, secondary_top_k_filter,second_filter_option= render_rank_and_trend_sub_tabs_helper_rank_filtering_panel(col_1, page_name, names)
    use_province_or_cluster, selected_n_cluster = None, None
    if tab_selected=="rank_bar_line" : # ratio is only used for line plot
        use_count_or_ratio = col_2.radio("Select an option:", ["Use count", "Use ratio"])
        show_column="ratio" if "ratio" in use_count_or_ratio else "count"
    else:
        show_column="ratio"

    if tab_selected=="rank_bar_line" and geo_level != None:
        use_province_or_cluster = col_3.radio("Select an option", options=[f"Use {geo_level}s", "Use clusters"],
                                          key="province_or_cluster").lower()
        selected_n_cluster = col_3.multiselect(f"Select clusters (default all)", clusters)
    else:
        use_province_or_cluster = ""

    plotter_engine = col_4.radio("Select plotter engine",
                      options=["Matplotlib", "Seaborn", "Plotly",  "Altair"],
                      index=3,
                      # key=f"bump_engine_{page_name}",
                      )
    # col_5 is used for bar&line plots (sub-tab 2.3 & 2.4), it is not used for rank bump plot
    plot_style = ""
    if "line" in tab_selected:
        plot_style = col_5.radio("Select plot style:", ["Line plot","Bar plot"] )

    params = {"use_rank_filtering": use_rank_filtering,
              "include_all_years_option": include_all_years,
              "selected_names": selected_names,
              "top_n": top_n,
              "show_column": show_column,
              "secondary_top_k_filter": secondary_top_k_filter,
              "second_filter_option": second_filter_option}
    return params,use_province_or_cluster,show_column,selected_n_cluster,show_provinces_separately,plotter_engine,plot_style,col_23