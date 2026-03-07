from typing import List
import numpy as np
import pandas as pd
import time
import streamlit as st
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from clustering.base_clustering import BaseClustering
from clustering.evaluation.stability import stability_and_consensus


class SpectralClusteringEngine(BaseClustering):

    def __init__(self, n_clusters: int,
                 n_neighbors: int,
                 random_state: int = 1,
                 affinity: str = "",
                 assign_labels: str = ""):
        self.n_neighbors = n_neighbors
        self.metric_for_silhouette = "euclidean"
        self.affinity = affinity
        self.model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, n_neighbors=n_neighbors, assign_labels=assign_labels, random_state=random_state)

   # Overrides fit_predict method inhertited from Clustering class
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        if self.affinity == "precomputed (cosine)":
            self.metric_for_silhouette="cosine"
            # 1. Full Cosine Similarity Matrix (Non-negative)
            # This ensures A_ij >= 0, which is required for a stable Laplacian.
            df_out = cosine_similarity(df_out)#df_out is affinity matrix
            # 2. Numerical Stability: Ensure diagonal is exactly 1.0
            np.fill_diagonal(df_out, 1.0)
            df_out = np.clip(df_out, 0, 1)

        labels = self.model.fit_predict(df_out) + 1
        return labels

    def get_centroids(self, df_pivot):
        return df_pivot.groupby("clusters").mean()

    @classmethod
    def optimal_k_analysis(cls,
        df: pd.DataFrame,
        random_states: list[int],
        k_values: range,
        model_kwargs: dict,
        save_folder: str,
        saved_file_suffix: str = "",
        model_specific_metrics: list[str] = []
        ):
        affinity = model_kwargs["affinity"]
        saved_file_suffix = f"{affinity}_{saved_file_suffix}"
        #n_range = range(4, 15,2) #if self.affinity == "nearest_neighbors" else range(1, 2)
        n = st.session_state["n_neighbors_spectral"]
        n_range = range(n,n+1)
        for n in n_range:
            model_kwargs["n_neighbors"] = n
            df_summary, metrics_all, metrics_mean, ari_mean, ari_std, consensus_labels_all = super().optimal_k_analysis(
            df, random_states, k_values, model_kwargs, save_folder, saved_file_suffix+f"_{n}", model_specific_metrics)
            st.write(f"Completed optimal k analysis for n_neighbors={n}")
        # returns the results for the last n
        return df_summary, metrics_all, metrics_mean, ari_mean, ari_std, consensus_labels_all