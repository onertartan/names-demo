import pandas as pd
import streamlit as st
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform, pdist
#from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def trend_correlation_hierarchical(pivot_df_processed:pd.DataFrame,original_names:list,n_cluster:int):
    corr_heatmap_df = pivot_df_processed.corr(method="pearson")  # corr_df
    if corr_heatmap_df.empty:
        st.warning("Not enough numeric columns to compute Pearson correlation.")
        return
    # Hierarchical clustering on 1 - correlation so similar names are adjacent.
    distance_df = 1 - corr_heatmap_df
    np.fill_diagonal(distance_df.values, 0)

    condensed_distance = squareform(distance_df.values, checks=False)
    linkage_matrix = linkage(condensed_distance, method="average")
    cluster_labels_hierarchical = fcluster(linkage_matrix, t=n_cluster, criterion="maxclust")
    df_cluster_labels_hierarchical = pd.DataFrame(
        {"name": original_names, "cluster": cluster_labels_hierarchical}).sort_values(["cluster", "name"])

    ordered_names = np.array(original_names)[leaves_list(linkage_matrix)]
    heatmap_df = distance_df.loc[ordered_names, ordered_names]
    df_cluster_labels_hierarchical = pd.DataFrame(
        {"name": original_names, "cluster": cluster_labels_hierarchical}).sort_values(["cluster", "name"])

    return ordered_names,heatmap_df,linkage_matrix,df_cluster_labels_hierarchical

def trend_timeserieskmeans_hierarchical(pivot_df_processed:pd.DataFrame,original_names:list,n_cluster:int,years:list):
    # Feature matrix: names=rows, years=columns — correct for pdist
    ts_features = TimeSeriesScalerMeanVariance().fit_transform(pivot_df_processed.T).squeeze()
    # convert ts_features from numpy to pandas
    ts_features_df = pd.DataFrame(ts_features, index=original_names, columns=years)
    # pdist returns condensed distance directly — no squareform needed for linkage
    condensed_distance = pdist(ts_features, metric='euclidean')
    # linkage expects condensed form — correct
    linkage_matrix = linkage(condensed_distance, method="average")
    # Ordered names for heatmap — use Euclidean distance matrix, not corr_df
    square_distance = squareform(condensed_distance)  # (n_names, n_names) —square_distance  for heatmap
    # heatmap_df is distance_matrix_df
    heatmap_df = pd.DataFrame(square_distance, index=original_names, columns=original_names)
    ordered_names = np.array(original_names)[leaves_list(linkage_matrix)]
    heatmap_df = heatmap_df.loc[ordered_names, ordered_names]
    # fcluster labels align with original_names — correct
    cluster_labels_hierarchical = fcluster(linkage_matrix, t=n_cluster, criterion="maxclust")
    df_cluster_labels_hierarchical = pd.DataFrame(
        {"name": original_names, "cluster": cluster_labels_hierarchical}).sort_values(["cluster", "name"])
    return ordered_names,heatmap_df,linkage_matrix,df_cluster_labels_hierarchical,ts_features_df
