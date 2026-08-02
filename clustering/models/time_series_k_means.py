import numpy as np
#from tslearn.clustering import TimeSeriesKMeans
#from tslearn.datasets import CachedDatasets
#from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
import pandas as pd

from clustering.base_clustering import BaseClustering


class TimeSeriesKMeansEngine(BaseClustering):
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
        self.model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean",random_state=random_state)
        self.metric_for_silhouette = "euclidean"
    # ------------------------------------------------------------------
    def get_centroids(self, X):
        # X parameters is not needed in KMeans, it is kept for compatibility
        return self.model.cluster_centers_
