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
