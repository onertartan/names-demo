import streamlit as st
from typing import List, Optional

from matplotlib import pyplot as plt


def province_selector(all_provinces, key_prefix: str = "province",  default_excluded: Optional[List[str]] = None) -> List[str]:
    """
    Ultra-compact province selector using only exclusion.

    Args:
        key_prefix: Unique prefix for session state keys and widget IDs
        all_provinces: Complete list of all provinces
        default_excluded: Provinces to exclude by default on first load

    Returns:
        List of currently selected provinces (all minus excluded)
    """
    # Session state initialization
    if f"{key_prefix}_excluded" not in st.session_state:
        if default_excluded is not None:
            st.session_state[f"{key_prefix}_excluded"] = default_excluded.copy()
        else:
            st.session_state[f"{key_prefix}_excluded"] = []

    excluded_key = f"{key_prefix}_excluded"

    # Tek bir searchable multiselect ile hariç tutma
    st.markdown("**Province Selection**")
    st.caption("Exclude provinces from analysis (all others are included)")
    col_header = st.columns([3, 1])

    with col_header[0]:
        # Searchable multiselect with better UX
        excluded = st.multiselect(
            "Exclude provinces:",
            options=all_provinces,
            default=st.session_state[excluded_key],
            key=f"{key_prefix}_exclude_compact",
            label_visibility="collapsed",
            placeholder="Search and select provinces to exclude..."
        )
    with col_header[1]:
        if st.button("Clear", key=f"{key_prefix}_clear_btn", type="secondary"):
            st.session_state[excluded_key] = []
            st.rerun()

    if excluded != st.session_state[excluded_key]:
        st.session_state[excluded_key] = excluded.copy()
        st.rerun()

    # Calculate final selected (all minus excluded)
    final_selected = [p for p in all_provinces if p not in st.session_state[excluded_key]]

    # Mini summary
    st.caption(
        f"**{len(final_selected)}** provinces selected, "
        f"**{len(st.session_state[excluded_key])}** excluded"
    )

    return final_selected


def sidebar_controls_basic_setup(*args):
    """
       Renders the sidebar controls
       Parameters: starting year, ending year
    """
    # Inject custom CSS to set the width of the sidebar
    #   st.markdown("""<style>section[data-testid="stSidebar"] {width: 300px; !important;} </style> """,  unsafe_allow_html=True)
    if "visualization_option" not in st.session_state:
        st.session_state["visualization_option"] = "matplotlib"
    start_year = args[0]
    end_year = args[1]
    with (st.sidebar):
        st.header('Visualization options')
         # if ifadesine gerek olmadığı düşünülerek (hata olursa bu if kalktığı için olabilir) metot classmethod'dan static'e dönüştü. Böylelikle higher-education kullanabildi.
        # options= list(range(start_year, end_year + 1)) if cls.page_name != "sex_age_edu_elections" else [2018,2023]
        options = list(range(start_year, end_year + 1))
        # Create a slider to select a single year
        st.select_slider("Select a year", options, 2023, on_change=update_selected_slider_and_years, args=[1],  key="slider_year_1")
        # Create sliders to select start and end years
        st.select_slider("Or select start and end years",options, [options[0],options[-1]],on_change=update_selected_slider_and_years, args=[2], key="slider_year_2")

        if "selected_slider" not in st.session_state:
            st.session_state["selected_slider"] = 1
        update_selected_slider_and_years(st.session_state["selected_slider"])

        if st.session_state["selected_slider"] == 1:
            st.write("You have selected a single year from the first slider.")
            st.write("Selected year:", st.session_state["year_1"])
        else:
            st.write("You have selected start and end years from the second slider.")
            st.write("Selected start year:", st.session_state["year_1"], "\nSelected end year:", st.session_state["year_2"])

        # Main content
        if "animation_images_generated" not in st.session_state:
            st.session_state["animation_images_generated"] = False


def update_selected_slider_and_years(slider_index):
    st.session_state["selected_slider"] = slider_index
    if slider_index == 1:
        st.session_state["year_1"] = st.session_state["year_2"] = int(st.session_state.slider_year_1)
    else:
        st.session_state["year_1"], st.session_state["year_2"] = int(st.session_state.slider_year_2[0]), int(st.session_state.slider_year_2[1])

def figure_setup(display_change=False):
    if st.session_state["visualization_option"] != "matplotlib":
        return None, None
    if st.session_state["year_1"] == st.session_state["year_2"] or st.session_state["selected_slider"] == 1 or \
            st.session_state["animate"]:
        n_rows = 1
    elif display_change:
        n_rows = 3
    else:
        n_rows = 2
    fig, axs = plt.subplots(n_rows, 1, squeeze=False, figsize=(10, 4 * n_rows),
                            gridspec_kw={'wspace': 0, 'hspace': 0.1})  # axs has size (3,1)
    # fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=0.5, hspace=0.5)
    return fig, axs