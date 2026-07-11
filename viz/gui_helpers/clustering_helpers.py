import streamlit as st
import pandas as pd
import polars as pl


def gui_clustering_up_col1(page_name):
    # First column of upper part in clustering showing scaling options
    if "names" in page_name:
        options = ["Share of Top-n (L1 Norm)",  # Denominator = Sum of the 30 columns
                "Share of Total",  # Denominator = Total births in province (External data)
                "TF-IDF",  # Best for emphasizing unique/rare names
                "L2 Normalization" ] # Best for pure cosine pattern matching
    else:
        options = ["None", "L1 Norm (row based)", "L2 Norm (row based)", "Standard Scaler (column based)", "MinMaxScaler (column based)", "RobustScaler (column based)" ]
    scaler = st.radio("Select scaling option", options=options, key="scaler")
    return scaler

def gui_clustering_up_col2():
    # scaler
    run_optimal_k_analysis = st.checkbox("Run cluster analysis", key="optimal_k_analysis")
    n_seeds = st.number_input("Number of seeds", min_value=3, max_value=100, value=3, key="number_of_seeds")
    use_consensus = st.checkbox("Use consensus labels", False, key="use_consensus_labels")
    return run_optimal_k_analysis, n_seeds, use_consensus
def gui_clustering_bottom():
    # Algorithm Selection
    algos = {
        "kmeans": {"label": "K-means", "gui_func": gui_options_kmeans},
        "gmm": {"label": "GMM", "gui_func": gui_options_gmm},
        "kmedoids": {"label": "K-medoids", "gui_func": gui_options_kmedoids},
        "spectral": {"label": "Spectral", "gui_func": gui_options_spectral},
        "hierarchical": {"label": "Hierarchical", "gui_func": gui_options_hierarchical},
        #  "dbscan": {"label": "DBSCAN", "gui_func": dbscan_gui_options},
    }
    cols = st.columns(len(algos))
    selected_algo, kwargs_for_selected_algo = None, None

    for col, (key, config) in zip(cols, algos.items()):
        with col:
           # with st.form("submit_form_" + key):
               # submitted = st.form_submit_button(config["label"], use_container_width=True)
            clicked = st.button(config["label"], use_container_width=True,key=key)
            kwargs = config["gui_func"]()
            if clicked:
                selected_algo, kwargs_for_selected_algo = key, kwargs
    return selected_algo, kwargs_for_selected_algo

# OPTIONS FOR CLUSTERING ALGORITHMS

def gui_clustering_main(page_name):
    col1, col2,_ = st.columns([1,1, 3])
    with col1:
        scaler = gui_clustering_up_col1(page_name)
    with col2:
        run_optimal_k_analysis, n_seeds, use_consensus = gui_clustering_up_col2()

    selected_algo, kwargs = gui_clustering_bottom()
    return scaler, run_optimal_k_analysis, n_seeds, use_consensus, selected_algo, kwargs

def gui_options_gmm():
    n_clusters = st.number_input("Number of clusters / components", 2, 15, 4, key="n_cluster_gmm")
    n_init = st.number_input("Random restarts (n_init)", 1, 100, 10, key="n_init_gmm")
    covariance_type = st.selectbox("Covariance", options=["spherical", "diag", "tied"],key="gmm_covariance_type")
    return {"n_clusters": n_clusters, "n_init": n_init, "covariance_type": covariance_type}


def gui_options_kmeans():
    n_clusters = st.number_input("Number of clusters", 2, 15, 4, key="n_cluster_kmeans")
    n_init = st.number_input("Random restarts (n_init)", 1, 100, 10, key="n_init_kmeans")
    return {"n_init": n_init, "n_clusters": n_clusters}


def gui_options_kmedoids():
    n_clusters = st.number_input("Number of clusters", 2, 15, 4, key="n_cluster_kmedoids")
    max_iter = st.number_input("Maximum number of iteration", 10, 300, 100, key="max_iter_kmedoids")
    distance_metric = st.selectbox("Select distance metric", options=["euclidean", "cosine"], index=0)

   # metric = st.selectbox("Distance metric", ["cosine"], help="Cosine: profile similarity;", key="distance_metric_pam")
    return {"n_clusters": n_clusters, "metric":distance_metric, "max_iter": max_iter, "method": "pam"}


def gui_options_spectral():
    """
    GUI controls for SpectralClustering parameters.
    """
    n_clusters = st.number_input("Number of clusters", min_value=2, max_value=15, value=4, key="n_cluster_spectral")
    n_neighbors = st.number_input("Number of nearest neighbors", min_value=2, max_value=30, value=10, step=1,
        help="Controls local connectivity in the graph (higher = more global structure)",
        key="n_neighbors_spectral"
    )
    #st.selectbox("Spectral similarity geometry",  options=["euclidean"], index=0, key="spectral_geometry"  )
    affinity = st.selectbox("Affinity", options=["nearest_neighbors", "rbf"], index=0, key="affinity_spectral")
    return {"n_clusters":n_clusters, "n_neighbors": n_neighbors, "affinity": affinity, "assign_labels": "kmeans"}

def gui_options_hierarchical():
    """
    GUI options for Hierarchical Clustering (Agglomerative).
    Designed for structural validation, not k-optimization.
    """
    # ---- Distance metric ----
    metric = st.selectbox("Distance metric", options=["cosine"], index=0, key="distance_metric_hierarchical")

    # ---- Linkage method ----
    linkage_method = st.selectbox("Linkage method", options=["average"], index=0, key="linkage_hierarchical")

    # ---- k for comparison only ----
    use_fixed_k = st.checkbox(
        "Cut dendrogram at fixed number of clusters (for comparison)",
        value=True,
        help=(
            "Hierarchical clustering does not infer the number of clusters. "
            "This option cuts the dendrogram at an externally selected k "
            "to enable comparison with other methods."
        ), key="use_fixed_k_hierarchical"
    )

    k = 15
    if use_fixed_k:
        k = st.slider(
            "Number of clusters (k)",
            min_value=2,
            max_value=15,
            value=4,
            step=1,
            help="Used only for comparison with other clustering methods.",
            key="n_cluster_hierarchical"
        )
    return {"metric": metric, "linkage_method": linkage_method,"n_clusters": k}

def dbscan_gui_options():
    """
            exp1, exp2, exp3 are the three columns you already pass in.
            """
    db_eps = st.number_input("ε (eps)",
                                  min_value=0.05, max_value=2.0,
                                  value=0.25, step=0.05,
                                  help="Max distance for neighbourhood")
    db_min = st.number_input("minPts",
                                  min_value=2, max_value=20,
                                  value=5, step=1,
                                  help="Min points to form a core region")
    # optional metric (keep Euclidean unless you need something else)
    db_metric = st.selectbox("Metric", options=["euclidean", "cosine"])


def render_top_n_selector(max_n):
    col1,col2=st.columns([15,85])
    return col1.number_input("Top-n", min_value=1, max_value=max_n,  value=30,  key="top_n_names")


def render_name_clustering_specific_ui(df,max_rank,tab_selected):
        col1,col2,col3=st.columns([1,2,3])
        n_unique_years = df['year'].n_unique()

        if n_unique_years != 1:
            min_year = df['year'].min()
            max_year = df['year'].max()
        else:
            st.write("Select an interval of years")
        #   render_data_coverage_if_rank_available
        use_data_option = col1.radio("**Data coverage**", ["Use all data","Use top-n names"],index=1)
        top_n_names = render_top_n_selector(max_rank)
        with col2:
            pass
            # TODO:
        return use_data_option, top_n_names
