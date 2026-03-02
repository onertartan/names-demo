"""
Gaussian Mixture Model (GMM) clustering engine.
Independent from UI and Streamlit. Designed for testability and modularity.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture

from clustering.base_clustering import BaseClustering
from clustering.evaluation.stability import stability_and_consensus
import streamlit as st
import time
class GMMEngine(BaseClustering):
    """
    A clean Gaussian Mixture clustering engine for tabular data.
    """
    def __init__(self, n_clusters: int, n_init: int,  random_state: int = 1,  covariance_type: str = ""):
        """
        Parameters
        ----------
        n_clusters (n_components) : int
            Number of mixture components.
        n_init : int
            Number of restarts.
        covariance_type : str
            'tied', 'diag', or 'spherical'.
        random_state : int
            Random seed for reproducibility.
        """
        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            n_init=n_init,
            random_state=random_state
        )
        self.metric_for_silhouette = "euclidean"

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
        saved_file_suffix = f"{model_kwargs['covariance_type']}_{saved_file_suffix}"
        return super().optimal_k_analysis(df, random_states, k_values, model_kwargs, save_folder,saved_file_suffix, model_specific_metrics)



    def probabilities(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get the posterior probabilities of each sample belonging to each cluster.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_clusters) with posterior probabilities.
        """
        return self.model.predict_proba(df)