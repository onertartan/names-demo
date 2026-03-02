from collections import defaultdict
from adjustText import adjust_text

from clustering.scaling import scale
from modules.base_page import BasePage
import pandas as pd
import geopandas as gpd
import streamlit as st
from matplotlib import colormaps, pyplot as plt, cm
import locale
import plotly.express as px
from matplotlib.patches import Patch
import altair as alt
from viz.config import COLORS, CLUSTER_COLOR_MAPPING, VA_POSITIONS, HA_POSITIONS
from viz.color_mapping import create_cluster_color_mapping
from viz.gui_helpers.ui_base_page import province_selector, sidebar_controls_basic_setup, figure_setup
from viz.gui_helpers.ui_base_page_names import render_tab_selection, render_gender_name_surname_filters, \
    render_top_n_selector, sidebar_controls_plot_options_setup

locale.setlocale(locale.LC_ALL, 'tr_TR.utf8')

class PageNames(BasePage):

    def initialize_multiindex_gdf_clusters(self, df: pd.DataFrame, year: object) -> None:
        new_index = pd.MultiIndex.from_product([[year], df.index], names=["year", "city"])
        self.gdf_clusters = df.copy()
        self.gdf_clusters.index = new_index

    def preprocessing_initial_filtering(self,name_surname_selection,selected_years,gender_list_state_key):
        # ---   Apply Filter ---
        # Only filter by gender if we are looking at names
        df = self.data[name_surname_selection.lower()]
        top_n_names = render_top_n_selector()
        selected_provinces = province_selector(df.index.get_level_values("province").unique().tolist())
        idx = pd.IndexSlice
        df = df.loc[idx[selected_years, selected_provinces], :]
        df = df[df["rank"] <= top_n_names]  # use top-n for clustering (n=30 for all)

        if name_surname_selection != "surname":
            # Ensure the column 'sex' exists before filtering
            if 'sex' in df.columns:
                df = df[df['sex'].isin(st.session_state[gender_list_state_key])]
        return df

    def preprocess_clustering(self, df):
        """"
        returns:df_pivot
        """
        year = df.index.get_level_values(0).unique() # year(s)
        page_name = self.page_name
        # top_n = st.session_state["n_" + page_name]  # CANCELED : NOW ALL 30 FEATURES ARE USE IN K-MEANS
        df_year = df.loc[year]
        if page_name == "names_surnames" and "name_surname_rb" in st.session_state and st.session_state["name_surname_rb"] == "surname":
            df_year = df.loc[year]
        elif st.session_state["sex_" + page_name] == ["male", "female"]:  # if both sexes are selected
            df_year_male, df_year_female = df[df["sex"] == "male"].loc[year], df[df["sex"] == "female"].loc[year]
            #     df_year_male = df_year_male[df_year_male["rank"] <= top_n]
            #    df_year_female = df_year_female[df_year_female["rank"] <= top_n]
            st.header("df_year_male shape:"+str(df_year_male.head()))
            st.header("Unique male names"+str(df_year_male["name"].nunique()))
            st.header("Unique female names"+str(df_year_female["name"].nunique()))

            st.header("df_year_female shape:"+str(df_year_female.shape))

            overlapping_names = set(df_year_male["name"]) & set(df_year_female["name"])
            df_year_male['name'] = df_year_male.apply(
                lambda x: f"{x['name']}_female" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year_female['name'] = df_year_female.apply(
                lambda x: f"{x['name']}_male" if x['name'] in overlapping_names else x['name'], axis=1)
            df_year = pd.concat([df_year_male, df_year_female])
     #   else:  # single gender selected for names  # gender selection handled in preprocessing_initial now
     #       sex = st.session_state["sex_" + page_name]
     #       df_year = df[df["sex"].isin(sex)].loc[year]
        if isinstance(df_year.index, pd.MultiIndex):  # If applying temporal clustering (multiple years given as year)
             print("bebe",df_year.head())
             df_year = df_year.droplevel(0)  # Drop the first level year(position 0)

          # Get unique cumulative total counts over years(for each province)
        total_counts = df[["total_count"]].groupby(level=["year", "province"]).first()
        total_counts = total_counts.groupby("province").sum()

        df_year = df_year.groupby([df_year.index, 'name']).agg({'count': 'sum'})
        # Merge
        df_year = df_year.merge(total_counts, left_index=True, right_index=True, how="outer")
        df_year = df_year.reset_index().set_index("province")
        df_pivot = pd.pivot_table(df_year, values='count', index=df_year.index, columns=['name'], aggfunc=lambda x: x,
                                  dropna=False, fill_value=0)
        total_counts = df_year.loc[:, "total_count"]
        scaler_method = st.session_state["scaler"]
        df_pivot = scale(scaler_method, df_pivot, total_counts)
        if st.session_state["selected_tab_" + self.page_name] == "tab_name_clustering":  # transpose_for_name_clustering:
            df_pivot = df_pivot.T
        return df_pivot

    @staticmethod
    def get_ordinal(n):
        if 11 <= (n % 100) <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    @staticmethod
    def preprocess_for_map(df, year, target_rank, n_to_top_inclusive):
        df_year = df.loc[year]
        if n_to_top_inclusive:  # select the names above the target rank(inclusive)
            df_year_rank = df_year[df_year["rank"] <= target_rank]
        else:  # select the names with target rank
            df_year_rank = df_year[df_year["rank"] == target_rank]
        # if surname is selected or only single gender selected, we do not need combinatiobs
        if "sex" not in df_year_rank.columns or len(df_year_rank["sex"].unique()) == 1:  # if st.session_state["name_surname_rb"]=="surname":
            return df_year_rank
        # generate combinations if both genders are selected and surname is not selected
        results = {}
        for province in df_year_rank.index.unique():
                province_data = df_year_rank.loc[province]
                male_names = province_data[province_data['sex'] == 'male']['name'].tolist()
                female_names = province_data[province_data['sex'] == 'female']['name'].tolist()
                combinations = []
                for male in male_names:
                    for female in female_names:
                        combinations.append(f"{male}-{female}")
                results[province] = '\n'.join(combinations)
        # Create the final dataframe
        final_df = pd.DataFrame(results.items(), columns=['province', 'name'])
        final_df.set_index('province', inplace=True)
        return final_df

    def preprocess_for_rank_bar_tabs(self, df):
        df.index = df.index.droplevel(1)  # drop provinces from multiindex
        df = df.groupby([df.index, "name"]).aggregate({"count": "sum"}).reset_index(level="name")
        df = df.sort_values(by=["year", "count"], ascending=False)
        if "rank" in st.session_state["selected_tab_"+self.page_name]:  # rank tabs work for cumulative results
            # Create rank column
            for year in df.index:
                df.loc[year, 'rank'] = df.loc[year, 'count'].rank(axis=0, method='min', ascending=False)

        if "rank" in st.session_state["selected_tab_" + self.page_name]:
            # preprocess rank
            if st.session_state["include_all_years"] == "Include All Years for Names Ever in Top-n":
                selected_names = df[df['rank'] <= st.session_state["rank_" + self.page_name]]["name"]
                df = df[df["name"].isin(selected_names)]
            else:
                df = df[df['rank'] <= st.session_state["rank_" + self.page_name]]
        else:  # st.session_state["selected_tab" + cls.page_name] == "custom_bar":
            selected_names = st.session_state["names_" + self.page_name]
            df = df[df["name"].isin(selected_names)]
        return df

    def render(self):
        # Apply CSS to all radio groups except the first
        st.session_state["geo_scale"] = "province"
        header = "Names & Surnames Analysis" if self.page_name == "names_surnames" else "Baby Names Analysis"
        st.header(header)
        start_year, end_year = self.data["name"].index.get_level_values(0).min(), self.data["name"].index.get_level_values(0).max()
        sidebar_controls_basic_setup(start_year, end_year)
        sidebar_controls_plot_options_setup(self.page_name)
        name_surname_selection, selected_years, gender_list_state_key = render_gender_name_surname_filters(self.page_name)
        df = self.preprocessing_initial_filtering(name_surname_selection, selected_years, gender_list_state_key)
        tab_selected = render_tab_selection(self.page_name)
        col_1, col_2, col_3, col_4, _ = st.columns([1, 1, 1, 1, 1])

        # if "display_option_"+self.page_name not in st.session_state:
        #     st.session_state["display_option_"+self.page_name] = 0

        col_plot, col_df = st.columns([5, 1])
        if tab_selected == "tab_geo_clustering" or tab_selected == "tab_name_clustering":  # Main Tab-1
            self.tab_clustering(df=df,save_sub_folder=st.session_state["gender_radio_widget_" + self.page_name].lower())
        elif tab_selected == "tab_map":  # Main Tab-2: Tab-1
            self.tab_2_map(df)
        elif tab_selected in ["rank_bump", "rank_bar", "custom_bar"]:  # Main Tab-2: Tab 3-4-5
            self.tab_3_4_5(df, col_plot, col_df, col_2, col_3, col_4)

    def tab_2_map(self, df):
        # Expression depending on page
        expr = "names or surnames" if self.page_name == "names_surnames" else "baby names"
        options = list(range(1, 31))  # Options are [1-30]
        btn_col1, btn_col2 = st.columns([1, 1])
        plot_value = 0
        display_option = None
        button_clicked = btn_col1.button("Select & Filter", use_container_width=True)
        top_n = btn_col1.selectbox('Choose a number top-n to filter', options, index=1, key="top_n_filter")
        btn_col1.multiselect(f"Select {expr}", sorted(df["name"].unique(), key=locale.strxfrm), key="names_" + self.page_name)
        if button_clicked:
            plot_value = top_n
            display_option = "top-n to filter"
        # Use container with custom class
        button_clicked = btn_col2.button("Nth Common", use_container_width=True)
        n_most_common = btn_col2.selectbox('Choose a number n for the "nth most common"', options)
        if button_clicked:
            plot_value = n_most_common
            display_option = "nth most common"
        # --- Plot map ---
        col_plot, col_df = st.columns([5, 1])
        # Display results on map if a button is clicked
        if display_option:
            self.plot_map(col_plot, col_df, df, plot_value, display_option)

    def tab_3_4_5(self, df, col_plot, col_df,col_2,col_3,col_4):
        tab_selected = st.session_state["selected_tab_" + self.page_name]
        if "rank" in tab_selected:  # add rank selectbox if tab_selected == "tab_rank_bump" or tab_selected == "tab_rank_bar": Tabs 2-3
            col_2.selectbox(f"Select rank", range(1, 21), index=4, key="rank_" + self.page_name)
            col_2.radio("Select an option",
                        ["Show Only Years When Names Are in Top-n", "Include All Years for Names Ever in Top-n"],
                        key="include_all_years")
        elif tab_selected == "custom_bar":  # add name selector for custom name selection instead of top-rank names
            expression_in_sentence = "names or surnames" if self.page_name == "names_surnames" else "baby names"
            # names_surnames has extra name-surname radio group overlapping with name selector,if so move the selector to right col
            empty_col = col_4 if self.page_name == "names_surnames" else col_2
            empty_col.multiselect(f"Select {expression_in_sentence}", sorted(df["name"].unique(), key=locale.strxfrm),
                                  key="names_" + self.page_name)
        if tab_selected in ["rank_bump", "rank_bar", "custom_bar"]:  # if tab is one of tab-2,3 or 4
            col_3.radio("Select an option", options=["Use provinces", "Use clusters"],  key="province_or_cluster").lower()
            provinces = sorted(df.index.get_level_values(1).unique(), key=locale.strxfrm)
            if "n_clusters_" + self.page_name in st.session_state:
                clusters = range(1, st.session_state["n_clusters_" + self.page_name] + 1)
            else:
                clusters= []
            col_4.multiselect(f"Select provinces (default all)", provinces, key="provinces_" + self.page_name)
            col_4.multiselect(f"Select clusters (default all)", clusters, key="clusters_" + self.page_name)
            col_4.checkbox(f"Show aggregated totals (sum counts for selected provinces)", provinces,
                           key="aggregate_totals_" + self.page_name)

        # works for two tabs: Rank Bar Plot and Custom Name Bar Plot (tabs 4 & 5)
        selected_provinces = st.session_state["provinces_" + self.page_name]
        selected_clusters = st.session_state["clusters_" + self.page_name]

        if "bar" in st.session_state["selected_tab_"+self.page_name]:
            plot_method = getattr(self, "plot_rank_bar")
        else:
            plot_method = getattr(self, "plot_rank_bump") #getattr(cls, "plot_" + st.session_state["selected_tab"+cls.page_name])
        col_plot.title("Counts by Name and Year")
        idx = pd.IndexSlice
        if st.session_state["province_or_cluster"] == "Use provinces" and selected_provinces:
            if st.session_state["aggregate_totals_" + self.page_name]:
                idx = pd.IndexSlice
                df = df.loc[idx[:, selected_provinces], :]
                plot_method(self.preprocess_for_rank_bar_tabs(df), col_plot)
            else:
                for province in selected_provinces:
                    df_province = df.loc[idx[:, province], :]
                    col_plot.subheader(province)
                    plot_method(self.preprocess_for_rank_bar_tabs(df_province), col_plot)
            col_df.dataframe(df)
        elif st.session_state["province_or_cluster"] == "Use clusters" and selected_clusters:
            df_pivot = self.preprocess_clustering(df)
            df_pivot, _ = self.kmeans(df_pivot) # _ --> closest indices ( not used here)

            df_clusters = df_pivot["clusters"]
            df['clusters'] = df.index.get_level_values("province").map(df_clusters)
            if st.session_state["aggregate_totals_" + self.page_name]:
                df = df[df["clusters"].isin(selected_clusters)]
                plot_method(self.preprocess_for_rank_bar_tabs(df), col_plot)
            else:
                for cluster in selected_clusters:
                    df_cluster = df[df["clusters"] == cluster]
                    col_plot.subheader(f"Cluster {cluster}")
                    plot_method(self.preprocess_for_rank_bar_tabs(df_cluster), col_plot)
            col_df.dataframe(df)
        else:  # if not any selected, select all provinces
            plot_method(self.preprocess_for_rank_bar_tabs(df), col_plot)


    def create_title_for_plot(self, rank):
        page_name = st.session_state["page_name"]
        names_or_surnames = "names"
        selected_gender = "male and female" if len(st.session_state["sex_" + self.page_name]) != 1 else st.session_state["sex_" + self.page_name][0]
        # Adjust phrasing based on rank
        if rank == 1:
            title_prefix = "The most common "
        else:
            title_prefix = f"The {PageNames.get_ordinal(rank)} most common "

        if page_name == "names_surnames":
            if st.session_state["name_surname_rb"] == "Surname":
                title = title_prefix + "surnames"
            else:
                title = title_prefix + selected_gender+" names"
        else:
            title = title_prefix + selected_gender + " baby names"
        return title, names_or_surnames

    # Plot methods of three tabs

    def plot_map(self, col_plot, col_df, df, n, display_option):
        st.session_state["visualization_option"] = "matplotlib"
        gdf_borders = self.gdf["province"]
        title, names_or_surnames = self.create_title_for_plot(n)
        df_results = []
        year_1, year_2 = st.session_state["year_1"], st.session_state["year_2"]
        fig, axs = figure_setup()
        for i, year in enumerate(sorted({year_1, year_2})):
            # Display option 1: Show the nth most common baby names
            if display_option == "nth most common":
                df_year_rank = PageNames.preprocess_for_map(df, year, target_rank=n, n_to_top_inclusive=False)
                df_result = gdf_borders.merge(df_year_rank, left_on="province", right_index=True)
                df_result = df_result.sort_values(by=['province', 'name'], ascending=[True, True]) # to prevent different orders like "Asel, Defne"  and "Defne, Asel"
                df_result = df_result.groupby(["geometry", "province"])["name"].apply(
                    lambda x: "%s" % '\n'.join(x)).to_frame().reset_index()
                df_results.append(df_result)
                self.plot_names(df_results[i], axs[i, 0])
                axs[i, 0].set_title(title + f' in {year}')
            elif display_option == "top-n to filter":  # Display option 2: Select single year, name(s) and top-n number to filter
                names_from_multi_select = st.session_state["names_" + self.page_name]
                df_year = df.loc[year].reset_index()
                if names_from_multi_select:
                    df_result = df_year[(df_year["name"].isin(names_from_multi_select)) & (df_year["rank"] <= n)]
                    # drop "s" if single name or surname selected
                    names_or_surnames_statement = names_or_surnames[:-1] + " is" if len(names_from_multi_select) == 1 else names_or_surnames + " are"
                    if df_result.empty:
                        st.write(f"Selected {names_or_surnames_statement} are not in the top {n} for the year {year}")

                    df_result_not_null = gdf_borders.merge(df_result, left_on="province", right_on="province")
                    df_result_not_null = df_result_not_null.groupby(["geometry", "province"])["name"].apply(lambda x: "%s" % '\n '.join(x)).to_frame().reset_index()
                    df_results.append(df_result_not_null)
                    df_result_with_nulls = gdf_borders.merge(df_result_not_null[["province", "name"]],
                                                             left_on="province", right_on="province", how="left")
                    print("456987",df_result_with_nulls)
                    self.plot_names(df_result_with_nulls, axs[i, 0])#, sorted(df_result_not_null['name'].unique()))  -->GEREKLİ Mİ, fonksiyondan parametre kalkmıştı, buradan yollamaya gerek var mı?
                    axs[i, 0].set_title(f"Provinces where selected {names_or_surnames_statement} in the top {n} for {year}")
            # else:  # K-means
            #     self.k_means_clustering(df,  year)   # ilk ve son yıl için kullanılıyordu artık
            #     #self.plot_clusters(axs[i, 0], year)  # artık for üstündeki ilk if çalıştığı için kullanılmıyor
            #     df_results.append(self.gdf_clusters)
        if axs[0, 0].has_data():
            col_plot.pyplot(fig)
        else:
            st.write("No results found.")

    def get_title_statement(self):
        # Male Baby Names, Female Names
        gender = st.session_state["sex_" + self.page_name]
        if self.page_name == "names_surnames" and st.session_state["name_surname_rb"] == "Surname":
            title_statement = "Surnames"
        else:
            title_statement = "Names"
        if self.page_name == "baby_names":
            title_statement = "Baby " + title_statement
        if len(gender) == 1 and title_statement!="Surnames":
            gender = gender[0]
            title_statement = " " + gender.capitalize() + " " + title_statement
        return title_statement

    # Tab-2.1:  nth most common
    def plot_names(self, df_result, ax):
        # Tab-1.1: Plots nth most common names on map
        # Create a color map
        df_result["clusters"] = df_result["name"].factorize()[0]
        color_map = create_cluster_color_mapping(df_result.set_index("name"), CLUSTER_COLOR_MAPPING)
        # Assign colors to each row in the GeoDataFrame
        print("LŞL",color_map)
        df_result['color'] = df_result['clusters'].map(color_map).fillna("gray") #-->GEREK YOK mu
        # After groupby df_result becomes Pandas dataframe, we have to convert it to GeoPandas dataframe
        df_result = gpd.GeoDataFrame(df_result, geometry='geometry')
        # Plotting
        df_result.plot(ax=ax, color=df_result['color'], legend=True,  edgecolor="black", linewidth=.2)
        bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6)
        df_result.apply(lambda x: ax.annotate(
            text=x["province"].upper() + "\n" + x['name'].title() if isinstance(x['name'], str) else x["province"],
            size=4, xy=x.geometry.centroid.coords[0], ha=HA_POSITIONS.get(x["province"], "center"), va=VA_POSITIONS.get(x["province"], "center"), bbox=bbox), axis=1)
        ax.axis("off")
        ax.margins(x=0)
        # # Add a table (positioned like a legend)
        df_count = (df_result['name'].str.split('\n')  # Split names by newline
                    .explode()        # Create separate row for each name
                    .value_counts()   # Count occurrences
                    .rename_axis('name')
                    .reset_index(name='count')
                   )
        # Format the DataFrame as a string for the legend
        legend_text = "\n".join(f"{row['name']}: {row['count']}" for _, row in df_count.iterrows())
        # Create list of text entries instead of single multi-line string
        legend_entries = [f"{row['name']}: {row['count']}" for _, row in df_count.iterrows()]
        # Add custom legend entry with text only (no line): Create invisible handles for each entry
        handles = [Patch(color='none', label=entry, linewidth=0) for entry in legend_entries]
        ax.plot([], [], label=legend_text)  # Invisible plot
        # Show legend
        ncols = 1+len(legend_entries)//9
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 0.165 if ncols>2 else 0.21), fontsize=4,ncols=ncols,  # Two columns
                  # Reduce left margin parameters:
                  handlelength=0,  # Remove space for legend handle (invisible line)
                  handletextpad=0,  # Remove padding between handle and text
                  alignment='left'  # Force left-aligned text
                  )

    # Add a legend
       # for name, color in set(zip(df_result['name'], df_result['color'])):
       #     ax.plot([], [], color=color, label=name, linestyle='None', marker='o')
       #     ax.legend(title='Names', fontsize=4, bbox_to_anchor=(0.01, 0.01), loc='lower right', fancybox=True, shadow=True)


    # Tab-2.2 plot method
    def plot_rank_bump(self, df, col_plot):  # pyplot
        # 2nd tab: Bump chart using pyplot
        df_pivot = pd.pivot_table(df, values='rank', index=df.index, columns=['name'],
                                  aggfunc=lambda x: x, dropna=False)
        fig, ax = plt.subplots(figsize=(10, 6))  # Explicit figure creation
        cmap = plt.cm.get_cmap("tab20", 20)  # good for categorical lines
        COLORS = [cmap(i) for i in range(20)]
        # Plot lines and dots for each name
        for i, name in enumerate(df_pivot.columns):
            color =  COLORS[i]
            series = df_pivot[name]

            # Identify all non-NaN segments, including single points
            not_nan = series.notna()
            segments = []
            current_segment = []

            for year, valid in zip(series.index, not_nan):
                if valid:
                    current_segment.append((year, series.loc[year]))
                else:
                    if current_segment:  # Changed from len(current_segment) > 1
                        segments.append(current_segment)
                    current_segment = []
            if current_segment:  # Changed from len(current_segment) > 1
                segments.append(current_segment)

            # Plot each segment
            for segment in segments:
                seg_years, seg_ranks = zip(*segment)
                ax.plot(seg_years, seg_ranks,
                        marker='o',
                        markersize=8,
                        linewidth=2,
                        alpha=0.7,
                        color=color,
                        label=name if segment == segments[0] else None)  # Avoid duplicate labels

        max_rank = int(df["rank"].max())
        # Invert y-axis to show #1 at top
        ax.invert_yaxis()
        ax.set_yticks(range(1, min(51,max_rank+1) ))  # Force show all ticks 1-10
        ax.set_ylim(min(51,max_rank+1), 0)  # Add padding to top/bottom

        ax.set_xticks(df_pivot.index)
        # Add padding around plot content
        ax.margins(x=0.15, y=0.1)  # 10% padding on both axes
        fig.tight_layout(pad=2.0)  # Add padding around figure
        # Customize plot
        ax.set_title(f"Rank Evolution of {self.get_title_statement()} Over Years", fontsize=20, pad=20)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Rank", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        # Add data labels for first and last year

        # Compute offsets to reduce overlapping texts
        y_offset_tracker_first = defaultdict(int)
        y_offset_tracker_last = defaultdict(int)

        for name in df_pivot.columns:
            first_rank = df_pivot[name].iloc[0]
            final_rank = df_pivot[name].iloc[-1]
            first_year = df_pivot.index[0]
            final_year = df_pivot.index[-1]

            # Add small vertical offsets if multiple names share the same rank
            offset_first = y_offset_tracker_first[first_rank] * 0.2
            offset_last = y_offset_tracker_last[final_rank] * 0.2

            ax.text(first_year - 0.1, first_rank + offset_first, name,
                    va='center', ha='right', alpha=0.7, fontsize=9)
            ax.text(final_year + 0.1, final_rank + offset_last, name,
                    va='center', ha='left', alpha=0.7, fontsize=9)

            y_offset_tracker_first[first_rank] += 1
            y_offset_tracker_last[final_rank] += 1
        col_plot.pyplot(fig)

    # Tab-2 plot method (potential alternative using plotly)
    def plot_rank_bump_plotly(self, df, col_plot):
        # 2nd tab: Bump chart using plotly (alternative not used yet)
        # Prepare data in long format (from your pivot table)
        max_rank = int(df["rank"].max())

        # Create figure with custom size
        fig = px.line(
            df.reset_index(),
            x='year',
            y='rank',
            color='name',
            markers=True,
        ).update_layout(
            width=1500,  # Figure width in pixels
            height=800,  # Figure height in pixels
            xaxis_title="Year", yaxis_title="Rank",
            margin=dict(l=50, r=50, t=80, b=50),  # Adjust margins for labels
            legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="right", x=1.15),
            title=dict(text="Rank Evolution Over Years", font=dict(size=50), automargin=True, yref='paper'),
            yaxis_title_font=dict(size=32), xaxis_title_font=dict(size=32)
        ).update_yaxes(
            tickvals=list(range(1, max_rank + 1)),
            range=[max_rank + 0.5, 0.5],
            autorange="reversed",
            dtick=1,
            tickfont=dict(size=32)
        ).update_xaxes(
            showline=True,
            tickfont=dict(size=32)
        ).update_traces(
            line=dict(width=5.5),
            marker=dict(size=30)
        )

        # Add annotations for names at the beginning and ending years
        df_reset = df.reset_index()
        years = df_reset['year'].unique()
        first_year = years.min()
        last_year = years.max()

        for name in df_reset['name'].unique():
            # Data for the first year
            first_year_data = df_reset[(df_reset['name'] == name) & (df_reset['year'] == first_year)]
            if not first_year_data.empty:
                fig.add_annotation(
                    x=first_year,
                    y=first_year_data['rank'].iloc[0],
                    text=name,
                    showarrow=False,
                    xshift=-20,  # Shift left for better positioning
                    font=dict(size=20),
                    xanchor="right"
                )

            # Data for the last year
            last_year_data = df_reset[(df_reset['name'] == name) & (df_reset['year'] == last_year)]
            if not last_year_data.empty:
                fig.add_annotation(
                    x=last_year,
                    y=last_year_data['rank'].iloc[0],
                    text=name,
                    showarrow=False,
                    xshift=20,  # Shift right for better positioning
                    font=dict(size=20),
                    xanchor="left"
                )

        col_plot.plotly_chart(fig)

    # Tab-2.3 & 2.4 plot methods
    def plot_rank_bar(self, df, col_plot):
        # 3rd tab & 4th tab: Bar plot for ranking and custom names
        df = df.reset_index()
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('year:O', title='Year'),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color('name:N', legend=None),  # Remove legend
            column=alt.Column('name:N', title=None)
        ).properties(
            width=150,
        title = {  # Add title configuration
            "text": f"Frequency of {self.get_title_statement()} by Year",  # Title text
            "anchor": "middle",  # Center the title
            "dy": 4,  # Adjust vertical spacing (positive = move down)
            "fontSize": 28  # Adjust font size
        }
        ).configure_header(labelFontSize=16,
            titleFontSize=24,  # Increase font size for column titles (names)
        ).configure_axisX(labelFontSize=16, titleFontSize=16).configure_axisY(labelFontSize=16, titleFontSize=16)
        chart.save("temp/rank_bar.png", scale_factor=3)
        col_plot.altair_chart(chart)
