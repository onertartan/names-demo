"""
KMeans clustering engine for geographic or tabular data.
Decoupled from UI and Streamlit. Easily testable.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, davies_bouldin_score, silhouette_score, silhouette_samples

from clustering.base_clustering import BaseClustering
from clustering.evaluation.stability import stability_and_consensus
import streamlit as st
import time

class KMeansEngine(BaseClustering):
    """
    A clean, UI-independent KMeans clustering module.
    """
    def __init__(self, n_clusters: int,  random_state: int = 1, n_init=-1):
        """
        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        random_state : int
            Random seed for reproducibility
        n_init : int
            Number of random restarts.
        """
        self.model = KMeans(n_clusters=n_clusters, n_init=n_init, init="k-means++", random_state=random_state)
        self.metric_for_silhouette = "euclidean"
    # ------------------------------------------------------------------
    def get_centroids(self, X):
        return self.model.cluster_centers_

    # ------------------------------------------------------------------
    @staticmethod
    def relabel_with_priority(labels: pd.Series, priority_list: List[str]) -> List[int]:
        """
        Optional: deterministic label remapping.

        Useful when you want certain provinces to define cluster order.

        Parameters
        ----------
        labels : pd.Series
            Original labels indexed by province name.
        priority_list : List[str]
            Provinces in the desired label order.

        Returns
        -------
        new_labels : List[int]
        """
        mapping = {}
        next_new = 1  # labels will be 1..K

        # priority first
        for prov in priority_list:
            old_lbl = labels.loc[prov]
            if old_lbl not in mapping:
                mapping[old_lbl] = next_new
                next_new += 1

        # remaining clusters
        for old_lbl in sorted(set(labels) - set(mapping)):
            mapping[old_lbl] = next_new
            next_new += 1

        return labels.map(mapping).tolist()