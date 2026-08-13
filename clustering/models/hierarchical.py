from typing import List, Tuple
import numpy as np
import pandas as pd
import time
import streamlit as st
from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score, davies_bouldin_score

from clustering.base_clustering import BaseClustering
from clustering.evaluation.stability import stability_and_consensus


class HierarchicalBaseClusteringEngine(BaseClustering):
    """
    Agglomerative hierarchical clustering engine.
    Deterministic for fixed metric and linkage.
    Used for structural validation / robustness.
    """

    # scipy computes these linkages from whatever condensed distances it is
    # given, silently and wrongly for non-euclidean metrics -- they are only
    # defined in euclidean geometry.
    LINKAGES_REQUIRING_EUCLIDEAN = ("ward", "centroid", "median")

    def __init__(
        self,
        n_clusters: int,
        metric: str,
        linkage_method: str = "average",
        random_state: int = -1 # included for interface compatibility
    ):
        if (linkage_method in self.LINKAGES_REQUIRING_EUCLIDEAN
                and metric != "euclidean"):
            raise ValueError(
                f"linkage_method {linkage_method!r} is only defined for "
                f"euclidean distances; got metric {metric!r}. scipy would "
                f"compute a meaningless hierarchy without warning.")
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage_method = linkage_method
        self.Z = None
        self.metric_for_silhouette = metric
        self.model = self  # for interface compatibility


    # ------------------------------------------------------------------
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # ---- build full hierarchy (deterministic) ----
        D = pdist(df, metric=self.metric)
        self.Z = linkage(D, method=self.linkage_method)
        # ---- cut at externally specified k ----
        labels = fcluster(self.Z, t=self.n_clusters, criterion="maxclust") -1# already 1-based, so we subtract 1, it is incremented again in fit_predict wrapper of BaseClusterer
        df_out = df.copy()
        df_out["clusters"] = labels
        self.plot_dendrogram(df.index)
        return pd.DataFrame(labels)
       # return df_out

    # ------------------------------------------------------------------

    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    def plot_dendrogram(self, provinces, max_d=None):
        fig, ax = plt.subplots(1, figsize=(10, 6))

        dendrogram(
            self.Z,
            labels=provinces,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=max_d
        )

        if max_d is not None:
            plt.axhline(y=max_d, color="red", linestyle="--", linewidth=1)

        ax.set_xlabel("Provinces")
        ax.set_ylabel("Distance")
        ax.set_title("Hierarchical Clustering Dendrogram")
       # ax.tight_layout()
        st.pyplot(fig)
