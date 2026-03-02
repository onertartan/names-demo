from sklearn import preprocessing
import os
import time
import pandas as pd
import streamlit.components.v1 as components
from PIL import Image
from folium import GeoJsonPopup, GeoJsonTooltip
from matplotlib.colors import to_hex
from raceplotly.plots import barplot
from matplotlib import colors, pyplot as plt, patches
from sklearn.cluster import KMeans
from streamlit_folium import folium_static
import folium
from modules.base_page import BasePage
import plotly.graph_objects as go
import plotly.express as px

from viz.gui_helpers.ui_base_page import sidebar_controls_basic_setup, figure_setup
from viz.gui_helpers.ui_base_page_common import *

class PageCommon(BasePage):

    def _render_map_tab(self, df_data, cols_nom_denom, gdf_borders):
        with st.form("submit_form"):
            (col_show_results, col_animation) = st.columns(2)
            show_results = col_show_results.form_submit_button("Show results")
            if self.animation_available:
                play_animation = col_animation.form_submit_button("Play animation")
                col_animation.write("Animation Controls")
                col_animation.slider("Animation Speed (seconds)", min_value=0.5, max_value=5., value=1., step=1.,
                                     key="animation_speed")
                col_animation.checkbox("Auto-play", value=True, key="auto_play")
                if play_animation:
                    st.session_state["animate"] = True
                else:
                    st.session_state["animate"] = False

            # Run on first load OR when form is submitted
            selected_features = self.get_selected_features(cols_nom_denom)
            if show_results or (self.animation_available and play_animation):
                #  st.session_state["animation_images_generated"] = False
                # delete_temp_files()
                col_plot, col_df = st.columns((4, 1), gap="small")
                print("df_data : LOL", df_data["denominator"]["district"])
                self.plot_main(col_plot, col_df, df_data, gdf_borders, selected_features,  [st.session_state["geo_scale"]])

    def _render_clustering_tab(self, df_data, cols_nom_denom, geo_scale):
        selected_features = self.get_selected_features(cols_nom_denom)
        years_selected = sorted({st.session_state["year_1"], st.session_state["year_2"]})
        df_result = self.get_df_result(df_data, selected_features, geo_scale, years_selected, give_total=True)
        self.tab_clustering(df_result, "",df_data, years_selected, selected_features, geo_scale)

    def render(self):
        apply_custom_css()
        st.session_state["geo_scale"] = self.top_row_cols[0].radio("Choose geographic scale",self.geo_scales).split()[0]
        self.fun_extras() # for optional columns at the top row
        cols_nom_denom = gui_basic_setup(self.col_weights)
        #cols_nom_denom = self.ui_basic_setup()
        # get cached data
        df_data = self.get_data()
        geo_scale = "province" if st.session_state["geo_scale"]!="district" else "district"
        gdf_borders = self.gdf[geo_scale]
        # determine year interval
        start_year = df_data["nominator"][geo_scale].index.get_level_values(0).min()
        end_year = df_data["nominator"][geo_scale].index.get_level_values(0).max()
        sidebar_controls_basic_setup(start_year, end_year)# File: viz/gui_helpers/clustering_helpers.py
        sidebar_controls_plot_options_setup()

        tabs = [stx.TabBarItemData(id="tab_map", title="Map Plot", description="") ,
                stx.TabBarItemData(id="tab_geo_clustering", title="Geographical Clustering", description="")]
        tab_selected = stx.tab_bar(data=tabs, default="tab_map")
        st.session_state["selected_tab_" + self.page_name]=tab_selected

        st.session_state["clustering_" + st.session_state["page_name"]] = (tab_selected == "tab_geo_clustering")
        if tab_selected == "tab_map":
            self._render_map_tab(df_data, cols_nom_denom, gdf_borders)
        else:
            self._render_clustering_tab(df_data, cols_nom_denom, geo_scale)


    # Overriden method
    def preprocess_clustering(self, df_result, df_data, years, selected_features, geo_scale):
        numeric_cols = list(df_result.select_dtypes(include=['number']).columns)
        df_pivot= df_result.loc[:, numeric_cols]
        if st.session_state["display_percentage"]:
            df_denom_result = self.get_df_year_and_features(df_data, "denominator", years, selected_features, geo_scale, give_total=True)
            df_pivot = df_pivot.div(df_denom_result["result"], axis=0)  # .div(df_denom_result.droplevel(0,axis=0)["result"],axis=0)
        df_pivot = df_pivot.groupby(level=1).sum()
        #df_pivot = scale(df_pivot)
        return df_pivot

    def get_df_result(self, df_data, selected_features, geo_scale, years, give_total=True):
        clustering = st.session_state["clustering_" + st.session_state["page_name"]]
        df_result = df_nom_result = self.get_df_year_and_features(df_data, "nominator", years, selected_features, geo_scale, not clustering)

        if st.session_state["display_percentage"] and not clustering:
            # df_data_nom and df_data_denom is same for maritial_status, sex-age pages, but different for birth
            df_denom_result = self.get_df_year_and_features(df_data, "denominator", years, selected_features, geo_scale, give_total=True)
            print("vcxz", df_result)
            print("çömn:", df_denom_result)  # df_denom_result.droplevel(0,axis=0))
            # Calculate the percentage
            df_result["result"] = df_nom_result["result"] / df_denom_result["result"]

           # n_clusters = st.session_state["n_clusters_" + st.session_state["page_name"]]
            #k_means = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++', n_init=50).fit(self.scale(df_result.loc[:, numeric_cols]))
            #df_result["clusters"] = k_means.labels_

            #color_map = plt.get_cmap('tab20')  # tab20 has 20 distinct colors
            #df_result["color"] = df_result["clusters"].apply(lambda x: color_map(x))
            #df_result["clusters"] +=  1 # +1 for displaying clusters to use from 1 not 0
            # keep descriptive(region, province, code) and cluster,color columns(drop other e.g. 5-9,10-14)
            #non_numeric_cols = df_result.select_dtypes(exclude=['number']).columns.tolist() + ["clusters"]
            #df_result = df_result.loc[:, non_numeric_cols]
          #  df_result = df_result.loc[:, numeric_cols]

        print("SONUÇ:", df_result.head())
        # if not k_means:
        #    df_result = df_result[["result"] + df_result.columns[:-1].tolist()]  # Reorder dataframe (result to first column) ARTIK GEREKMİYOR GİBİ, result zaten ilk sütun olmuş
        return df_result

    def handle_duplicate_columns(self,df):
        # Step 1: Identify duplicated column names
        duplicated_cols = df.columns[df.columns.duplicated(keep=False)].unique()

        # Step 2: Create new column names with '_m' and '_f' suffixes
        col_counts = {}
        new_columns = []
        for col in df.columns:
            if col in duplicated_cols:
                if col not in col_counts:
                    col_counts[col] = 1
                    new_columns.append(f"{col}_m")
                else:
                    col_counts[col] += 1
                    new_columns.append(f"{col}_f")
            else:
                new_columns.append(col)

        # Step 3: Assign new column names to the DataFrame
        df.columns = new_columns
        return df

    def get_df_year_and_features(self, df_data, nom_denom_selection, year, selected_features_dict, geo_scale, give_total=True):
        df_codes = pd.read_csv("data/preprocessed/region_codes.csv", index_col=0)
        print("YYY", year, "df_codesçç", df_codes.head(), "geo_scale", geo_scale)
        if "district" in geo_scale:
            print("1. çekpoint ", nom_denom_selection)
            df = df_data[nom_denom_selection]["district"]
            print("1. çekdf", df)
        else:
            print("2. çekpoint")
            df = df_data[nom_denom_selection]["province"]
        print("nom_denom_selection:", nom_denom_selection, "selected_features_dict:", selected_features_dict)
        selected_features = selected_features_dict[nom_denom_selection]

        print("nom_denom_selection:", nom_denom_selection, "START:\n", df.head(), "\nssselected_features:",
              selected_features)
        if len(selected_features) == 1:
            selected_features = selected_features[0]
        df = pd.DataFrame(df.loc[year, selected_features])  # if it is Pandas Series it is converted to dataframe
        for i in range(df.columns.nlevels - 1):
            df = df.droplevel(0, axis=1)
        if give_total:
            # if df.columns.nlevels > 1:  # Check if the DataFrame has multiple column levels
            #   df = df.droplevel(1, axis=1)  # .sum(axis=1))  # .reset_index()
            df = df.sum(axis=1).to_frame()
            print("nom_denom_selection:", nom_denom_selection, "zxzxzx", df.head())

            df.rename(columns={df.columns[-1]: "result"}, inplace=True)  # aggregation result(sum of selected cols)
            df = df.sort_values(by=["year", "result"], ascending=False)

        # else:
        #    df = df.droplevel(0,axis=0) # k-means clustering drop year from index to merge with df_codes (make df compatible)
        #   df = df.droplevel(0, axis=1)  # .sum(axis=1))  # .reset_index()

        print("geo_scale:", geo_scale, "nom_denom_selection", nom_denom_selection, "GGG:", df.head(), df.shape)

        print("\ndf_codes HHH:", df_codes.head())
        #  print("ÜĞP:", df.loc[(slice(None), "İstanbul", slice(None))])
        df = self.handle_duplicate_columns(df)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        if geo_scale != ["district"]:
            print("before join shape df",df.shape,"len df_codes",len(df_codes))
            df = df.join(df_codes, on="province")  # left_index=True  DEĞİŞTİRİLDİ left_on="province" yapıldı
            print("after  join shape df",df.shape,"len df_codes",len(df_codes))
            print(df.columns)
            print(len(df.columns) != len(set(df.columns)))  # True if duplicates exist
        if geo_scale == ["sub-region"]:
            agg_funs = {col: "sum" for col in numeric_cols}
            agg_funs.update(
                {"province": lambda x: ",".join(x), "ibbs1 code": 'first', "region": "first", "ibbs2 code": 'first'})
            df = df.reset_index().groupby(["year", "sub-region"]).agg(agg_funs)
            print("FFF:\n", df.head())
        elif geo_scale == ["region"]:
            print("SSS:", df.head())
            print("NNUMM:", numeric_cols)
            agg_funs = {col: "sum" for col in numeric_cols}
            agg_funs.update(
                {"province": lambda x: ",".join(x.astype(str)), "ibbs1 code": 'first', "sub-region": lambda x: ",".join(x.unique().astype(str))})
            df = df.reset_index().groupby(["year", "region"]).agg(agg_funs)
        return df

    def get_df_change(self, df_result):
        # display_change: Show the change between end and start years in the third figure
        if st.session_state["year_1"] != st.session_state["year_2"]:
            if st.session_state["clustering_" + st.session_state["page_name"]]:
                print("TTTT:", df_result.select_dtypes(include=['number']).loc[st.session_state["year_1"]].index)
                df1, df2 = df_result.loc[st.session_state["year_1"]], df_result.loc[st.session_state["year_2"]]
                df1, df2 = df1.align(df2, join="inner", axis=1)  # Align columns
                df1, df2 = df1.align(df2, join="inner", axis=0)  # Align rows
                df_change = df2.copy()
                print("PŞM:", df1.head())
                print("OIK:", df2.head())
                df_change = df2.select_dtypes(include=['number']) - df1.select_dtypes(include=['number'])
                print("ÇÖKJ:", df1.shape, df2.shape)
                if st.session_state["display_percentage"]:
                    numeric_cols = df_change.select_dtypes(include=['number']).columns
                    df_change.loc[:, numeric_cols] = df_change.loc[:, numeric_cols] / df1.loc[:, numeric_cols] * 100
            else:
                df_change = pd.DataFrame({"result": df_result.loc[st.session_state["year_2"], "result"] - df_result.loc[
                    st.session_state["year_1"], "result"]})
                if st.session_state["display_percentage"]:
                    df_change["result"] = df_change["result"] / df_result.loc[
                        st.session_state["year_1"], "result"] * 100
        print("CCHHAA:", df_change.head())
        return df_change

    def plot_main(self,col_plot, col_df, df_data, gdf_borders, selected_features, geo_scale):
        plotter = getattr(self, "plot_"+ st.session_state["selected_tab"] + "_"+st.session_state["visualization_option"])

        if geo_scale == ["district"]:
            geo_scale = geo_scale + ["province"]  # geo_scale = ["province", "district"]
        # Population Pyramid - Plotly
        if st.session_state["selected_tab"] == "pyramid":
            plotter(df_data, selected_features)
        elif st.session_state["selected_tab"] == "map":
            # Racebar plot
            if st.session_state["visualization_option"] == "raceplotly":
                df_result = self.get_df_result(df_data, selected_features, geo_scale, list(
                    range(st.session_state["slider_year_2"][0], st.session_state["slider_year_2"][1] + 1)))
                self.plot_map_raceplotly(df_result, geo_scale[0])
            else:
                # Map plots
                if not st.session_state["animate"]:  # if animation is not clicked plotter will work in the standard way
                    # st.session_state["animation_images_generated"] = False
                    # delete_temp_files()
                    years_selected = sorted({st.session_state["year_1"], st.session_state["year_2"]})
                    df_result = self.get_df_result(df_data, selected_features, geo_scale, years_selected)
                else:  # In the next step plotter will generate images according to df_result for A RANGE OF YEARS
                    self.delete_temp_files()# 22 NİSAN 2025 ÖNCE KLASÖRÜ TEMİZLE
                    years_selected = list( range(st.session_state["slider_year_2"][0], st.session_state["slider_year_2"][1] + 1))
                    df_result = self.get_df_result(df_data, selected_features, geo_scale, years_selected)

                fig, axs = figure_setup((st.session_state["year_1"] != st.session_state["year_2"]))
                self.plot_map_generic(col_plot, col_df, gdf_borders, df_result, geo_scale, plotter, years_selected, fig, axs)

                if st.session_state["animate"]:
                    self.animate(col_plot)


    def plot_map_generic(self,col_plot, col_df, gdf_borders, df_result, geo_scale, plotter_func, years_selected, fig=None, axs=None):
        # Add CSS for layout control
        st.markdown(""" <style>[role=checkbox]{ gap: 1rem; }</style>""", unsafe_allow_html=True)
        display_change = (st.session_state["year_1"] != st.session_state["year_2"])
        years = sorted(list(df_result.index.get_level_values(0).unique()))
        col_df.write("""<style>.dataframe-margin { margin-bottom: 80px;} </style>""", unsafe_allow_html=True)

        for i, year in enumerate(years):
            print("abcde:\n", df_result.loc[year].head(), "\nres:\n", gdf_borders.head())
            # gdf_result = get_geo_df_result(gdf_borders, df_result.loc[year], geo_scale)
            gdf_result = gdf_borders.dissolve(by=geo_scale)[["id", "geometry"]].merge(df_result, left_index=True,
                                                                                      right_index=True)  # after dissolving index becomes geo_scale,so common index is geo_scale(example:province)

            title = f'Results for {years_selected[i]}'
            if axs is not None:
                if st.session_state["animate"]:
                    ax = axs[0, 0]
                else:
                    ax = axs[i, 0]
            else:
                ax = None
            with col_plot:
                col_df.markdown('<div class="dataframe-margin">', unsafe_allow_html=True)
                plotter_func(gdf_result, title, geo_scale, ax)  # plot map or save figure for each year if the state of animate is True
                col_df.markdown('</div>', unsafe_allow_html=True)

            if not st.session_state["animate"] and not st.session_state["clustering_" + st.session_state["page_name"]]:
                # display dataframe on the right side if not in animation mode
                col_df.dataframe(df_result.loc[year].sort_values(by="result", ascending=False))


        if st.session_state["animate"]:
            # generated and saved image for each year calling plotterFunction in the for loop above
            st.session_state["animation_images_generated"] = True
        elif display_change:
            df_change = self.get_df_change(df_result)
            # after dissolving index becomes geo_scale,so common index is geo_scale(example:province)
            gdf_result = gdf_borders.dissolve(by=geo_scale)[["id", "geometry"]].merge(df_result, left_index=True,
                                                                                      right_index=True)
            with col_plot:
                start_word = "Change"
                if st.session_state["display_percentage"]:
                    start_word = "Relative change in ratio"
                title = f"{start_word} between years {st.session_state.year_1} and {st.session_state.year_2}"
                plotter_func(gdf_result, title, geo_scale, axs[2, 0] if axs is not None else None)
            # if not k-means clustering show result col
            if not st.session_state["clustering_" + st.session_state["page_name"]]:
                col_df.markdown('<div class="dataframe-margin">', unsafe_allow_html=True)
                col_df.dataframe(df_change.sort_values(by="result", ascending=False))
                col_df.markdown('</div>', unsafe_allow_html=True)
            elif st.session_state["elbow"]:  # if k-means and elbow selected
                print("--x-")
                random_states = range(st.session_state["number_of_seeds"])  # 50 random seeds
                fig, _ = self.optimal_k_analysis(df_result.iloc[:, :-5], random_states, k_values= list(range(2, 15)))
                col_df.pyplot(fig)
        if fig and not st.session_state["animate"]:
            col_plot.pyplot(fig)


    def plot_map_matplotlib(self, gdf_result, title, geo_scale, ax):
        ax.cla()
        norm = None
        if not st.session_state["clustering_" + st.session_state["page_name"]] and st.session_state[
            "display_percentage"] and gdf_result["result"].min() < 0 and gdf_result["result"].max() > 0:
            norm = colors.TwoSlopeNorm(vmin=gdf_result["result"].min(), vcenter=0, vmax=gdf_result["result"].max())
        if st.session_state["clustering_" + st.session_state["page_name"]]:

            gdf_result.plot(ax=ax, color=gdf_result["color"], edgecolor="black",
                            linewidth=.2, norm=norm,  # legend=True, legend_kwds={"shrink": .6},
                            antialiased=True)  # column=selected_feature,
        else:
            gdf_result.plot(ax=ax, column=gdf_result["result"], cmap=st.session_state["selected_cmap"],
                            edgecolor="black",
                            linewidth=.2, norm=norm,  # legend=True, legend_kwds={"shrink": .6},
                            antialiased=True)  # column=selected_feature,
        print("ömnb3", ax.get_legend())

        ax.legend(fontsize=4, bbox_to_anchor=(.01, 0.01), loc='lower right', fancybox=True, shadow=True)
        ax.set_title(title)
        region_text_vertical_shift_dict = {"Akdeniz": -1.05, "Batı Marmara": -.6, "Doğu Marmara": .2,
                                           "Ortadoğu Anadolu": .25, "Doğu Karadeniz": -0.1}
        region_text_horizontal_shift_dict = {"Akdeniz": -.5}
        font_size_dict = {"region": 6.5, "sub-region": 5}
        if "district" not in geo_scale:  # if geo_scale is region or sub-region
            gdf_result.reset_index().apply(
                lambda x: ax.annotate(text=x[geo_scale[0]], size=font_size_dict.get(geo_scale[0], 4.5),
                                      xy=(
                                      x.geometry.centroid.x + region_text_horizontal_shift_dict.get(x[geo_scale[0]], 0),
                                      x.geometry.centroid.y + region_text_vertical_shift_dict.get(x[geo_scale[0]], 0)),
                                      # x.geometry.centroid.coords[0]   ,
                                      ha='center', va="bottom"), axis=1)
        ax.axis("off")
        ax.margins(x=0, y=0)
        # Get the figure object
        fig = ax.get_figure()
        if st.session_state["animate"]:
            # Adjust the layout to prevent text cutoff
            fig.tight_layout()
            # Save the figure as JPEG
            fig.savefig("temp/" + title + ".jpg", dpi=150, bbox_inches='tight', format='jpg')  # JPEG quality (0-100)
            # Clean up
            plt.close(fig)

    def plot_map_folium(self, gdf_result, *args):
        gdf_result = gdf_result.reset_index()

        # Prepare common parameters
        explore_kwargs = {
            "tooltip": "province",
            "popup": True,
            "tiles": "CartoDB positron",
            "columns": [col for col in gdf_result.columns if col != "year"],
            "style_kwds": dict(color="black", line_width=.01),
        }
        # Determine if we're doing clustering
        is_clustering = st.session_state["clustering_" + st.session_state["page_name"]]
        # If custom colors are available in the dataframe
        if "color" in gdf_result.columns:
            gdf_result["color"] = gdf_result["color"].apply(lambda x: to_hex(x))
            # Use custom colors for either clustering or regular display
            if is_clustering:
                explore_kwargs.update({
                    "column": "clusters",
                    "color": "color"  # Use the pre-calculated colors
                })
            else:
                explore_kwargs.update({
                    "column": "result",
                    "color": "color"  # Use the pre-calculated colors
                })
        else:
            # Fall back to colormap when no custom colors are available
            explore_kwargs.update({
                "column": "clusters" if is_clustering else "result",
                "cmap": st.session_state["selected_cmap"]
            })

        # Create and display the map
        m = gdf_result.explore(**explore_kwargs)
        folium_static(m, width=1100, height=450)

    def plot_map_raceplotly(self, df_result, geo_scale):
        # Generate colors using Matplotlib
        pyplot_colors = plt.get_cmap('tab10').colors + plt.get_cmap('Accent').colors + plt.get_cmap('Pastel2').colors
        colors_255 = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in pyplot_colors]

        # Get unique geoscales (e.g. provinces)
        df_first_year = df_result.loc[st.session_state["slider_year_2"][0]]
        unique_geoscales = df_first_year.reset_index().sort_values(by="result", ascending=False).loc[
                           :min(20, len(df_first_year) - 1), geo_scale]

        # Create a mapping of provinces to colors
        color_mapping = {province: "rgb" + str(colors_255[i % len(colors_255)]) for i, province in
                         enumerate(unique_geoscales)}
        # Assign colors to the 'color' column based on the province
        df_result = df_result.reset_index()
        my_raceplot = barplot(df_result,
                              item_column=geo_scale,
                              value_column='result',
                              time_column='year',
                              item_color=color_mapping)

        fig = my_raceplot.plot(
            title=f'Top 10 {geo_scale}s from {st.session_state["slider_year_2"][0]} to {st.session_state["slider_year_2"][1]} ',
            item_label=f'Top 10 {geo_scale}s',
            value_label='Result',
            frame_duration=800)
        fig.update_layout(margin=dict(l=20, r=60, t=20, b=20))
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    def delete_temp_files(self, folder_path="temp"):
        try:
            # List all files in the folder
            files = os.listdir(folder_path)
            # Delete each file
            for file in files:
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file}")
        except FileNotFoundError:
            print(f"Folder not found: {folder_path}")
        except PermissionError:
            print("Permission denied")

    def load_images(self, image_dir):
        """
          Load all JPEG images from the specified directory
          """
        images = []
        for filename in sorted(os.listdir(image_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                images.append(Image.open(image_path))
        return images

    def animate(self, col_plot):
        # Create a placeholder for the image
        image_placeholder = col_plot.empty()
        # Directory containing your JPEG images
        image_dir = "temp"  # Replace with your image directory path
        try:
            # Load all images
            images = self.load_images(image_dir)
            if not images:
                st.error("No JPEG images found in the specified directory!")
                return
            # Display animation
            while st.session_state["auto_play"] and st.session_state["animate"]:
                for image in images:
                    image_placeholder.image(image, use_column_width=True)
                    time.sleep(st.session_state["animation_speed"])
                # Break the loop if auto-play is unchecked
                if not st.session_state["auto_play"]:
                    break
            # If auto-play is off, add a manual control
            if not st.session_state["auto_play"]:
                selected_frame = st.slider("Select Frame", min_value=0, max_value=len(images) - 1, value=0)
                image_placeholder.image(images[selected_frame], use_column_width=True)
        except Exception as e:
            st.error(f"Error loading images: {str(e)}")

    def get_pyramid_dfs(self,df_data, selected_features):
        # Adding Male data to the figure
        df = df_data["nominator"]["province"].loc[st.session_state.slider_year_2[0]:st.session_state.slider_year_2[1],
             selected_features["nominator"]].reset_index().drop(columns=["province"]).groupby("year").sum()

        #  df = df.loc[:,pd.IndexSlice["male",age_groups_ordered]]
        print("şş:", df.columns.nlevels)
        for i in range(df.columns.nlevels):
            df = df.stack(future_stack=True)
        df = df.reset_index()
        print("FGH:\n", df)
        df.rename(columns={df.columns[-1]: "Population"}, inplace=True)
        return df

    def plot_pyramid_plotly(self, df_data, selected_features, *args):
        print("213456",df_data)
        df = self.get_pyramid_dfs(df_data, selected_features)

        # total_population = df_male.sum() + df_female.sum()
        # Create the male and female bar traces
        # trace_male = go.Bar(x=df["male"], y=df_male.index,  name='Male', orientation='h', marker=dict(color="#1f77b4"))
        # trace_female = go.Bar(x=-df_female, y=df_female.index, name='Female', orientation='h', marker=dict(color="#d62728"))
        # max_population = max(df_female.max(), df_male.max()) * 1.2  # Find the max count either male or female
        layout_dict = {"title": "Population Pyramid", "title_font_size": 22, "barmode": 'overlay',
                       "yaxis": dict(title="Age"), "color": "#1f77b4",
                       "bargroupgap": 0,
                       "bargap": .3}
        if isinstance(selected_features["nominator"][-1], list):
            n_bars = len(selected_features["nominator"][-1]) * 2
        else:
            n_bars = 2 * len(st.session_state["age_group_keys"][self.page_name])
        barmode = "group"
        if barmode != "group":
            df.loc[df["sex"] == "female", "Population"] *= -1
        fig = px.bar(df, x="Population", y="age_group", orientation='h', color="sex", animation_frame='year',
                     height=600,
                     color_discrete_map={
                         'male': 'light blue',
                         'female': 'pink'
                     }
                     )
        layout_dict = {"title": "Population Pyramid", "title_font_size": 22, "barmode": 'group',
                       "yaxis": dict(title="Age"),
                       "bargroupgap": 0,
                       "bargap": .03}
        fig.update_layout(layout_dict)
        # for idx in range(len(fig.data)):
        #  print("ÖÖ:",dir(fig.data[idx].xaxis))
        print("ÖÖ", (fig.data[0]._props["xaxis"]))
        #   r = range(fig.layout["xaxis"]["range"][0], fig.layout["xaxis"]["range"][1], stepSize=2)
        #    fig.update_layout(xaxis={"tickvals": list(r),   "ticktext": [t if t >= 0 else '' for t in r]})
        # Create the layout
        """
        layout = go.Layout(title="Population Pyramid",title_font_size = 22, barmode = 'overlay',
                           yaxis=dict(title="Age"),
                            bargroupgap=0,   xaxis=go.layout.XAxis(
                         #  range=[-max_population, max_population],
                         #  ticktext = ['6M', '4M', '2M', '0',  '2M', '4M', '6M'],
                                  title = 'Population in Millions',
                                  title_font_size = 14 ),
                           bargap=.3)
        """
        # Create the figure
        #   fig = go.Figure(data=[trace_male, trace_female], layout=layout,)

        #  fig.update_xaxes(range=[-max_population*1.2,max_population*1.2])

        st.plotly_chart(fig)

    def plot_pyramid_matplotlib(self, df_data, selected_features, *args):
        print("piram", df_data)
        cols = st.columns([1, 8, 1])
        cols[1].write("Note: First slider is used for year selection.")
        df = df_data["nominator"]["province"].loc[st.session_state.slider_year_1, selected_features["nominator"]]

        df_male = df["male"].sum().reset_index()
        df_female = df["female"].sum().reset_index()
        df_male.rename(columns={df_male.columns[-1]: "Population"}, inplace=True)
        df_female.rename(columns={df_female.columns[-1]: "Population"}, inplace=True)

        print("jjj\n", df_male)
        # Calculate total population for percentage
        total_population = df_male["Population"].sum() + df_female["Population"].sum()
        fig, ax = plt.subplots(figsize=(9, 6))

        shift = total_population / 100  # Adjust this based on your data scale
        # Plot bars with a custom size and color
        bars_male = ax.barh(df_male["age_group"], df_male["Population"], color='#4583b5', align='center', height=0.7,
                            left=shift, label='Male',
                            zorder=3)

        bars_female = ax.barh(df_female["age_group"], -df_female["Population"], color='#ef7a84', align='center',
                              height=0.7, left=-shift,
                              label='Female',
                              zorder=3)
        ax.set_yticklabels([])

        # Set titles and labels
        fig.suptitle(f'Population Distribution by Age and Sex in the year {st.session_state["slider_year_1"]}',
                     fontsize=14, x=0.06, y=0.98, ha="left")  # Customize title
        ax.set_xlabel('Population', labelpad=10)

        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Y-axis
        ax.spines['left'].set_position(('data', shift))  # Center the y-axis
        ax.yaxis.set_ticks_position('none')  # Remove y-axis ticks
        for label in df_male["age_group"]:
            ax.text(0, label, f' {label} ', va='center', ha='center', color='black',
                    backgroundcolor='#fafafa')  # Center y-axis labels

        # X-axis
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, pos: f'{int(abs(x))}'))  # Set custom tick labels to show absolute values
        max_population = max(df_female["Population"].max(),
                             df_male["Population"].max()) * 1.2  # Find the max count either male or female
        ax.set_xlim(left=-max_population, right=max_population)  # Adjust x-axis limits for centering

        # Add data labels
        fontsize = 9  # Font size for the labels
        max_bar_width = max(max([bar.get_width() for bar in bars_male]),
                            max([abs(bar.get_width()) for bar in bars_female]))
        label_shift = max_bar_width / 4
        for bar in bars_female:
            width = bar.get_width()
            label_x_pos = bar.get_x() + width - label_shift  # Adjust position outside the bar
            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                    f'{abs(width) / total_population:.1%}', va='center', ha='left', color='#ef7a84', fontsize=fontsize,
                    fontweight='bold')

        for bar in bars_male:
            width = bar.get_width()
            label_x_pos = bar.get_x() + width + label_shift  # Adjust position outside the bar
            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2,
                    f'{width / total_population:.1%}', va='center', ha='right', color='#4583b5', fontsize=fontsize,
                    fontweight='bold')

        # Adding a custom rectangle as a border around the figure
        border_radius = 0.015
        rect = patches.FancyBboxPatch((0.03, -0.03), 1, 1.08, transform=fig.transFigure, facecolor="#fafafa",
                                      edgecolor='black', linewidth=1, clip_on=False, zorder=-3, linestyle='-',
                                      boxstyle=f"round,pad=0.03,rounding_size={border_radius}")
        fig.patches.extend([rect])
        ax.set_facecolor('#fafafa')  # Set the background color of the axes

        # Adding data for males and females in corners
        ax.text(1.11, 0.95,
                f'Male: {df_male["Population"].sum()} ({df_male["Population"].sum() / total_population:.1%})',
                transform=ax.transAxes,
                fontsize=9,
                ha='right',
                va='top',
                color="white",
                weight='bold',
                bbox=dict(facecolor='#4583b5', edgecolor='#4583b5', boxstyle=f"round,pad=1.2,rounding_size={0.4}"))

        ax.text(0.21, 0.95,
                f'Female: {df_female["Population"].sum()} ({df_female["Population"].sum() / total_population:.1%})',
                transform=ax.transAxes,
                fontsize=9,
                ha='right',
                va='top',
                color="white",
                weight='bold',
                bbox=dict(facecolor='#ef7a84', edgecolor='#ef7a84', boxstyle=f"round,pad=1.2,rounding_size={0.4}"))

        cols[1].pyplot(fig)

    # METHODS BELOW ARE NOT USED ANYMORE, SINCE GDF.EXPLORE PROVIDES ENOUGH HIGH LEVEL CONTROL
    # THEY CAN BE USED IF MORE CONTROL IS NEEDED

    def plotter_folium_low_level(self, gdf_result, title, geo_scale, *args):
        st.markdown(f"<h3 style='text-align: center; color: grey;'>{title}</h1>", unsafe_allow_html=True)
        gdf_result = gdf_result.reset_index()
        print("756756", gdf_result.columns)
        # Create the map
        m = folium.Map(location=[36.5, 35], zoom_start=6.4, tiles="cartodb positron")
        # Create the choropleth layer
        if st.session_state["clustering_cb_" + st.session_state["page_name"]]:
            # For clustered data
            folium.GeoJson(
                data=gdf_result,
                style_function=lambda feature: {
                    'fillColor': to_hex(feature['properties']["color"]),
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7,
                    'line_opacity': 0.2
                },
                popup=GeoJsonPopup(
                    fields=[col for col in gdf_result.columns if col not in ['geometry', 'year']],
                    aliases=[col for col in gdf_result.columns if col not in ['geometry', 'year']],
                    localize=True
                )
                # ,    tooltip=GeoJsonTooltip(
                #  fields=[geo_scale[0], "result"],
                #  aliases=[geo_scale[0], "Result"],
                #  localize=True  )
            ).add_to(m)
        else:
            # For regular choropleth
            folium.Choropleth(
                geo_data=gdf_result,
                name="choropleth",
                data=gdf_result,
                columns=geo_scale + ["result"] if "district" not in geo_scale else ["id", "result"],
                key_on=f'feature.properties.{geo_scale[0]}' if "district" not in geo_scale else "feature.properties.id",
                fill_color=st.session_state["selected_cmap"],
                nan_fill_color="black",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="Result"
            ).add_to(m)

            # Add tooltips
            style_function = lambda x: {'fillColor': '#ffffff',
                                        'color': '#000000',
                                        'fillOpacity': 0.1,
                                        'weight': 0.1}
            highlight_function = lambda x: {'fillColor': '#000000',
                                            'color': '#000000',
                                            'fillOpacity': 0.50,
                                            'weight': 0.1}

            NIL = folium.features.GeoJson(
                gdf_result,
                style_function=style_function,
                control=False,
                highlight_function=highlight_function,
                tooltip=folium.features.GeoJsonTooltip(
                    fields=[geo_scale[0], 'result'],
                    aliases=[geo_scale[0], 'Result'],
                    style=(
                        "background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
                )
            )
            m.add_child(NIL)
            m.keep_in_front(NIL)

        # Convert to HTML with responsive container
        self.resize_folium_map(m)

    def resize_folium_map(self,m, height=450):
        # Convert map to HTML string
        map_html = m._repr_html_()

        # Custom HTML with responsive iframe
        html = f"""
        <style>
            .responsive-map {{
                position: relative;
                padding-bottom: 45%; /* 4:3 Aspect Ratio */
                height: 0;
                overflow: hidden;
            }}
            .responsive-map iframe {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }}
        </style>
        <div class="responsive-map">
            {map_html}
        </div>
        """

        # Render the HTML
        components.html(html, height=450)

