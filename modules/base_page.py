# Standard Packages
from abc import ABC, abstractmethod
# Data Processing and Analysis
import pandas as pd
import geopandas as gpd
# Visualization
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from clustering.models.factory import get_engine_class
from clustering.models.hierarchical import HierarchicalBaseClusteringEngine
from viz import PCAPlotter, OptimalKPlotter
# Streamlit & Tools
from viz.gui_helpers.clustering_helpers import *
from viz.gui_helpers.ui_base_page import sidebar_controls_basic_setup
from viz.plotters.geo_cluster_plotter import GeoClusterPlotter
from viz.config import COLORS, CLUSTER_COLOR_MAPPING, VA_POSITIONS, HA_POSITIONS
from viz.plotters.network_plotter import plot_cluster_network, plot_clustered_heatmap, plot_umap_tsne, \
    plot_mds_provinces, plot_custom_silhouette


class BasePage(ABC):
    features = None
    page_name = None
    correlation_analysis = False
    animation_available = True
    geo_scales = ["province (ibbs3)", "sub-region (ibbs2)", "region (ibbs1)", "district"]
    top_row_cols = []
    checkbox_group = {}
    data = None  # Class-level data storage
    gdf_centroids = None  # closest provinces to centroids


    @staticmethod
    @st.cache_data
    def load_geo_data():
        gdf = {"district": gpd.read_file("data/preprocessed/gdf_borders_district.geojson"),
               "province": gpd.read_file("data/preprocessed/gdf_borders_ibbs3.geojson")}
        return gdf

    @property
    def gdf(self):
        return BasePage.load_geo_data()

    @abstractmethod
    def get_data(self, geo_scale=None):
        pass # overridden in sub-classes

    @property
    def data(self):
        return self.get_data()

    @property
    def gdf_clusters(self):
        return st.session_state.get("gdf_clusters")

    @gdf_clusters.setter
    def gdf_clusters(self, value):
        st.session_state["gdf_clusters"] = value

    @staticmethod
    def convert_year_index_data_type(df):
        """Not all data years are integers.In particular elections data of 2015 June and 2015 November are not string"""
        return pd.MultiIndex.from_arrays([df.index.get_level_values(0).astype(str),  df.index.get_level_values(1)], names=df.index.names)

    def fun_extras(self, *args):
        pass

    def render(self):
        pass

    def run(self):
        st.session_state["page_name"] = self.page_name
        self.get_data()
        self.render()

    def get_selected_features(self, cols_nom_denom):
        selected_features = {}
        for nom_denom in cols_nom_denom.keys():  # cols_nom_denom is a dict whose keys are "nominator" or "denominator" and values are st.columns
            selected_features[nom_denom] = ()  # tuple type is needed for multiindex columns
            for i, feature in enumerate(self.features[nom_denom]):
                selected_features[nom_denom] = selected_features[nom_denom] + ( self.get_selected_feature_options(cols_nom_denom[nom_denom][i], feature, nom_denom),)
        #    Checkbox_Group.age_group_quick_select()
        return selected_features

    def get_selected_feature_options(self, col, feature_name, nom_denom_key_suffix):
        disabled = not st.session_state["display_percentage"] if nom_denom_key_suffix == "denominator" else False
        if feature_name in ["marital_status", "education", "age", "Party/Alliance", "sex", "month"]:
            self.quick_selection(feature_name, nom_denom_key_suffix)
            self.checkbox_group[feature_name].place_checkboxes(col, nom_denom_key_suffix, disabled, feature_name)
            selected_feature = self.checkbox_group[feature_name].get_checked_keys(nom_denom_key_suffix, feature_name)
        return selected_feature

    def animation_slider_changed(self):
        st.session_state["animate"] = True

    def sidebar_controls(self, *args):  # start_year=2007,end_year=2023
        sidebar_controls_basic_setup(*args)
        self.sidebar_controls_plot_options_setup(*args)

    def sidebar_controls_plot_options_setup(self,*args):
        pass

    def quick_selection(self, feature_name, nom_denom_key_suffix):
        pass

    def set_checkbox_values_for_quick_selection(self, keys_to_check, nom_denom_key_suffix, feature_name):
         for key in self.checkbox_group[feature_name].basic_keys:
            if key in keys_to_check:
                val = True
            else:
                val = False
            st.session_state[self.page_name+"_"+nom_denom_key_suffix+"_"+feature_name+"_"+key] = val
            #cls.checkb ox_group[feature_name].checked_dict[nom_denom_key_suffix][key] = val
       #1 print("$$$",nom_denom_key_suffix,"$$$",cls.checkbox_group[feature_name].checked_dict[nom_denom_key_suffix])

    @abstractmethod
    def preprocess_clustering(self, df, *args):
        pass    # Overriden by sub-classes Base_Page_Names & Base_Page_Common

    def tab_clustering(self, df, save_sub_folder="", *args):
        # 0. Render UI
        scaler, run_optimal_k_analysis, n_seeds, use_consensus, clustering_algorithm, kwargs = gui_clustering_main()
        if not clustering_algorithm:
            return
        engine_class = get_engine_class(clustering_algorithm)
        n_clusters = st.session_state["n_clusters"] = kwargs["n_clusters"]
        df_pivot = self.preprocess_clustering(df, *args)
        engine =  engine_class(**kwargs)  # Single engine object will be initialized later if not optimal_k_analysis or use_consensus_labels
        save_folder = "results/files"
        if save_sub_folder != "":
            save_folder = f"results/files/{save_sub_folder}/{engine_class.__name__}"
        # 1. Run clustering: Preprocess
        # If optimal_k_analysis is selected or use_consensus_labels is checked but it is not present(optimal_k_analysis has not previously run)
        if run_optimal_k_analysis:
            k_values = list(range(2, 11)) if not (engine_class is HierarchicalBaseClusteringEngine) else range(n_clusters, n_clusters + 1)
            random_states = range(st.session_state["number_of_seeds"]) if engine_class.__name__ != "HierarchicalClusteringEngine" else range(1)
            num_seeds_to_plot = 3 if engine_class.__name__ != "HierarchicalClusteringEngine" else 1
            scaler,  year1, year2 = st.session_state['scaler'], st.session_state["year_1"], st.session_state["year_2"]
            saved_file_suffix=f"{scaler}_{year1}_{year2}"
            df_summary, metrics_all, metrics_mean, ari_mean, ari_std, consensus_labels_all = engine_class.optimal_k_analysis(df_pivot, random_states, k_values, kwargs, save_folder, saved_file_suffix)
            df_pivot["clusters"] = consensus_labels_all[n_clusters]
            st.write(f"Running optimal k analysis for {engine_class.__name__}, scaler = {scaler}, year1={year1}, year2={year2}")
            OptimalKPlotter.plot_optimal_k_analysis(engine_class, num_seeds_to_plot, k_values, random_states, metrics_all, metrics_mean, ari_mean, ari_std, kwargs)
            OptimalKPlotter.print_optimal_k_analysis(df_summary)
        elif use_consensus:
            df_pivot["clusters"] = engine_class.load_consensus_labels(kwargs, save_folder)
            st.header("Using previously saved consensus labels")
        else:
            silhouette_analysis=False
            if silhouette_analysis:
                engine_class.silhouette_analysis(df_pivot, kwargs=kwargs)
                return

            labels = engine.fit_predict(df_pivot)
            df_pivot["clusters"] = labels

           # st.dataframe(engine.probabilities(df_pivot.drop(columns=["clusters"])))
            #st.dataframe(df_pivot)
        # PLOT MAP (if geo-clustering tab is selected)
        col_plot, col_df = st.columns([5, 1])
        # Step: Update geodata
        if st.session_state.get("selected_tab_" + self.page_name, "") == "tab_geo_clustering" and engine:
            representatives = engine.get_representatives(df_pivot)
            #df_distances = engine.pairwise(df_pivot,"cosine")
        else:
            representatives = None
        self.gdf_clusters, self.gdf_centroids = engine_class.update_geo_cluster_centers(self.gdf, st.session_state["geo_scale"], df_pivot, representatives)

        if st.session_state.get("selected_tab_" + self.page_name, "") == "tab_geo_clustering":
            # Step-6: Render geo-cluster plots
            self.render_geo_clustering_plots(df_pivot, col_plot, col_df, df)
        with col_plot:
            self.tab_clustering_pca(df_pivot.copy())
            #plot_umap_tsne(df_pivot.copy(), CLUSTER_COLOR_MAPPING)
            #plot_mds_provinces(df_pivot)
    def tab_clustering_pca(self,df_pivot):
        #PLOT PCA
        df_clusters = df_pivot["clusters"]
        df_features = df_pivot.drop(columns=["clusters"])
        total_points = len(df_clusters)
        factor = .1 if self.page_name == "names_surnames" else 1
        dense_threshold = total_points / (10 * factor) if st.session_state.get("selected_tab_" + self.page_name,"no_tab") != "tab_map" else 100  # Define thresholds
        mid_threshold = total_points / (20 * factor) if st.session_state.get("selected_tab_" + self.page_name,"no_tab") != "tab_map" else 100  #
        title= f"PCA of provincial name-distribution profiles (2018–2024)" if self.page_name in ["names_surnames","baby_names"] else f"PCA of feature profiles"
        PCAPlotter().plot_pca(df_features, df_clusters,dense_threshold, mid_threshold, COLORS,title)

    def render_geo_clustering_plots(self, df_pivot, col_plot, col_df, df_original):
        """Tab-1 Step-6:   plot clusters and show clusters dataframe."""
        df_clusters = df_pivot["clusters"]
        # Determine year or year range
        start_year = df_original.index.get_level_values(0).min()
        end_year = df_original.index.get_level_values(0).max()
        if start_year == end_year:
            year_label = f"in {start_year}"
        else:
            year_label = f"between {start_year}-{end_year}"
        # Plot geographic clusters
        with col_plot:
            GeoClusterPlotter(CLUSTER_COLOR_MAPPING, HA_POSITIONS, VA_POSITIONS).plot_cluster_map(self.gdf_clusters, self.gdf_centroids, st.session_state["n_clusters"], year_label)
            #GeoClusterPlotter(self.CLUSTER_COLOR_MAPPING, self.HA_POSITIONS, self.VA_POSITIONS).plot_elections(self.gdf_clusters)
        col_df.dataframe(df_clusters)