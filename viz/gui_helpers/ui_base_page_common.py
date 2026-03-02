import streamlit as st
import extra_streamlit_components as stx


def apply_custom_css():
    """Sayfa için gerekli CSS stillerini tek bir yerden yönetir."""
    st.markdown("""
        <style> 
            .main > div {padding-left:1rem; padding-right:1rem; padding-top:4rem;}
            [role=radiogroup] { gap: 0rem; }
            [data-testid="stHorizontalBlock"] { align-items: top; }
        </style>
    """, unsafe_allow_html=True)

def gui_basic_setup(col_weights):
    cols_title = st.columns(2)
    cols_title[0].markdown("<h3 style='color: red;'>Select primary parameters.</h3>", unsafe_allow_html=True)
    cols_title[0].markdown("<br><br><br>", unsafe_allow_html=True)

    # Checkbox to switch between population and percentage display
    cols_title[1].markdown("<h3 style='color: blue;'>Select secondary parameters.</h3>", unsafe_allow_html=True)
    cols_title[1].checkbox("Check to get ratio: primary parameters/secondary parameters.", key="display_percentage")
    cols_title[1].write("Uncheck to show counts of primary parameters.")

    cols_all = st.columns(col_weights)  # There are 2*n columns(for example: 3 for nominator,3 for denominator)
    with cols_all[len(cols_all) // 2]:
        st.html(
            '''
                <div class="divider-vertical-line"></div>
                <style>
                    .divider-vertical-line {
                        border-left: 1px solid rgba(49, 51, 63, 0.2);
                        height: 180px;
                        margin: auto;
                    }
                </style>
            '''
        )
    cols_nom_denom = {"nominator": cols_all[0:len(col_weights) // 2],
                      "denominator": cols_all[len(col_weights) // 2 + 1:]}
    return cols_nom_denom

def sidebar_controls_plot_options_setup():
    with (st.sidebar):
        if "selected_tab" not in st.session_state:
            st.session_state["selected_tab"] = "map"

        tabs = [stx.TabBarItemData(id="map", title="Map/Race plot", description="")]
        if st.session_state["page_name"] in ["sex_age", "marital_status"]:
            tabs.append(stx.TabBarItemData(id="pyramid", title="Pop. Pyramid", description=""))

        #   tab_map, tab_pyramid= st.tabs(["Map", "Population pyramid"])
        st.session_state["selected_tab"] = stx.tab_bar(data=tabs, default="map")

        if st.session_state["selected_tab"] == "map":
            st.session_state["visualization_option"] = st.radio("Choose visualization option",
                                                                ["Matplotlib", "Folium",
                                                                 "Raceplotly"]).lower()
        else: # There are two option for population pyramid
            st.session_state["visualization_option"] = st.radio("Choose visualization option",
                                                                ["Matplotlib", "Plotly"]).lower()

        if st.session_state["visualization_option"] != "raceplotly":

                # Dropdown menu for colormap selection
                st.selectbox("Select a colormap\n (colormaps in the red boxes are available only for matplotlib) ",
                             st.session_state["colormap"][st.session_state["visualization_option"]],
                             key="selected_cmap")
                # Display a static image
                st.image("images/colormaps.jpg")