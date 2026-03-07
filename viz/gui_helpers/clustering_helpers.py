import streamlit as st


def gui_clustering_up_col1():
    # First column of upper part in clustering showing scaling options
    options = ["Share of Top 30 (L1 Norm)",  # Denominator = Sum of the 30 columns
            "Share of Total",  # Denominator = Total births in province (External data)
            "TF-IDF",  # Best for emphasizing unique/rare names
            "L2 Normalization" ] # Best for pure cosine pattern matching
    scaler = st.radio("Select scaling option", options=options, key="scaler")
    return scaler

def gui_clustering_up_col2():
    # scaler
    run_optimal_k_analysis = st.checkbox("Run cluster analysis", key="optimal_k_analysis")
    n_seeds = st.number_input("Number of seeds", min_value=1, max_value=100, value=1, key="number_of_seeds")
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

def gui_clustering_main():
    col1, col2, _ = st.columns([2, 2, 6])
    with col1:
        scaler = gui_clustering_up_col1()
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
   # metric = st.selectbox("Distance metric", ["cosine"], help="Cosine: profile similarity;", key="distance_metric_pam")
    return {"n_clusters": n_clusters, "metric": "cosine", "max_iter": max_iter, "method": "pam"}


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


