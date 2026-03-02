from typing import List
import time
import streamlit as st
from matplotlib import pyplot as plt, cm
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    davies_bouldin_score, pairwise_distances_argmin_min, silhouette_samples
)

from clustering.evaluation.stability import stability_and_consensus

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
from sklearn.metrics import pairwise_distances_argmin_min
class BaseClustering:
    """
    Unified clustering factory.
    Creates the correct engine and executes fit().
    """

    model = None
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        labels = self.model.fit_predict(df) + 1
        return labels



    def get_representatives(self, df_pivot: pd.DataFrame):
        """
        Select cluster representatives in a method-aware way.
        Representatives are used for visualization / interpretation only.

        Parameters
        ----------
        df_pivot : pd.DataFrame
            Feature matrix (rows = observations, columns = features+ 1 column clusters).
        method : str
            One of {"kmeans", "kmedoids", "gmm", "spectral"}.
        metric : str
            Distance metric used for silhouette or medoids ("euclidean", "cosine").
        model : object, optional
            Fitted clustering model (required for GMM).

        Returns
        -------
        representatives : list of indices
        """
        # Ensure alignment
        labels= df_pivot["clusters"]
        df_features= df_pivot.drop(columns="clusters")
        # Convert labels to numpy-friendly form
        unique_clusters = sorted(labels.dropna().unique())
        # 1. K-MEANS: nearest-to-centroid (Euclidean)
        if self.__class__.__name__ == "KMeansEngine":
            # Centroids in feature space
            feature_space_centroids = df_pivot.groupby("clusters").mean()
            closest_idx, _ = pairwise_distances_argmin_min(feature_space_centroids.values, df_features.values)
            representatives = df_pivot.index[closest_idx].tolist()
            return representatives

        # 2. GMM: maximum posterior responsibility (Simplified)
        if self.__class__.__name__ == "GMMEngine":
            probs = self.model.predict_proba(df_features.values)
            representatives = []

            for cid in unique_clusters:
                # Assumes cid=1 maps to component 0, cid=2 to component 1, etc.
                component_idx = int(cid) - 1

                mask = (labels == cid)
                # Find max probability within the cluster for that component
                cluster_probs = probs[mask, component_idx]
                best_local_idx = np.argmax(cluster_probs)

                # Map back to global index
                best_row_in_cluster = df_features[mask].index[best_local_idx]
                representatives.append(best_row_in_cluster)

            return representatives
        # 3. SPECTRAL: highest silhouette
        if self.__class__.__name__ == "SpectralClusteringEngine":
            sil = silhouette_samples(df_features.values,labels.values, metric =  self.metric_for_silhouette  )
            sil_series = pd.Series(sil, index=df_features.index)
            representatives = []
            for cid in unique_clusters:
                idx = sil_series[labels == cid].idxmax()
                representatives.append(idx)
            return representatives

        # 4. K - MEDOIDS
        if self.__class__.__name__ == "KMedoidsEngine":
            return df_pivot.index[self.model.medoid_indices_].tolist()


    @classmethod
    def recompute_centroid_provinces(cls,df_pivot):
        """Recompute centroid provinces after clusters changed due to consensus relabel.
        The last column of df_pivot must be 'clusters', first len(df_pivot.columns-1) columns are features."""
        features = df_pivot.columns[:-1]  # exclude 'clusters'
        centroids = df_pivot.groupby('clusters')[features].mean()
        closest_indices, _ = pairwise_distances_argmin_min(centroids, df_pivot[features])
        return closest_indices

    @classmethod
    def update_geo_cluster_centers(cls, gdf_dict, geo_scale, df_pivot, closest_indices):
        """
        Attach cluster labels to geodata and compute centroids for display.
        Args:
            gdf_dict: Dictionary containing geodataframes (e.g., {'province': gdf...})
            geo_scale: Geographical scale key (province etc. ) to select the appropriate GeoDataFrame
            df_pivot: DataFrame containing the 'clusters' column
            closest_indices: List of indices representing cluster centers
        Returns:
            gdf_clusters: GeoDataFrame merged with clusters (last column is clusters)
            gdf_centroids: GeoDataFrame of the cluster representatives
        """
        # Prepare the base GeoDataFrame
        # Note: We assume gdf_dict has keys matching the values in st.session_state["geo_scale"]
        # or that mapping logic handles it. Based on BasePage logic:
        gdf_centroids = None
        gdf_clusters = gdf_dict[geo_scale].set_index(geo_scale)
        # Merge with clusters
        gdf_clusters = gdf_clusters.merge(df_pivot["clusters"], left_index=True, right_index=True)
        # Compute centroids for representatives
        if closest_indices:
            gdf_centroids = gdf_clusters[gdf_clusters.index.isin(closest_indices)].copy()
            gdf_centroids["centroid"] = gdf_centroids.geometry.centroid
        return gdf_clusters, gdf_centroids


    @classmethod
    def optimal_k_analysis(cls,
        df: pd.DataFrame,
        random_states: list[int],
        k_values: range,
        model_kwargs: dict,
        save_folder: str,
        saved_file_suffix: str = "",
        model_specific_metrics: list[str] = []           # e.g., {"Calinski-Harabasz": calinski_harabasz_score}
    ):
        n_samples = df.shape[0]
        metrics_all = {"Silhouette Score (cosine)": [], "Silhouette Score (euclidean)": [], "Davies-Bouldin Index": []}
        if cls.__name__ == "KMeansEngine":
            metrics_all["Inertia"] = []
        elif cls.__name__ == "GMMEngine":
            metrics_all["AIC"] = []
            metrics_all["BIC"] = []
            metrics_all["NegLogLikelihood"] = []
        elif cls.__name__ == "HierarchicalEngine":
            random_states = range(1)  # Hierarchical is deterministic

        labels_all = {seed: {} for seed in random_states}
        progress_bar = st.progress(0.0)
        status_text = st.empty()  # This will hold the "X / Y completed" message
        total_states = len(random_states)
        start_time = time.time()  # Record overall start time

        # ---- Run Clustering ----
        for idx, random_state in enumerate(random_states):
            silhouettes_cosine, silhouettes_euclidean, db_scores, inertias,aics, bics, nlls = [], [],[], [],[], [], []

            seed_start = time.time()

            for k in k_values:
                if "n_clusters" in model_kwargs:
                    model_kwargs["n_clusters"] = k
                engine = cls(random_state=random_state, **model_kwargs)
                labels = engine.fit_predict(df)
                silhouettes_cosine.append(silhouette_score(df, labels, metric="cosine"))
                silhouettes_euclidean.append(silhouette_score(df, labels, metric="euclidean"))
                db_scores.append(davies_bouldin_score(df, labels))
                labels_all[random_state][k] = labels
                if cls.__name__ == "KMeansEngine":
                    inertias.append(engine.model.inertia_)
                elif cls.__name__ == "GMMEngine":
                    aics.append(engine.model.aic(df))
                    bics.append(engine.model.bic(df))
                    nlls.append(-engine.model.score(df) * n_samples)

            if cls.__name__ == "KMeansEngine":
                metrics_all["Inertia"].append(inertias)
            elif cls.__name__ == "GMMEngine":
                metrics_all["AIC"].append(aics)
                metrics_all["BIC"].append(bics)
                metrics_all["NegLogLikelihood"].append(nlls)

            metrics_all["Silhouette Score (cosine)"].append(silhouettes_cosine)
            metrics_all["Silhouette Score (euclidean)"].append(silhouettes_euclidean)

            metrics_all["Davies-Bouldin Index"].append(db_scores)
            # Update progress and status after loop for one random state is completed
            progress_bar.progress((idx + 1) / total_states)
            elapsed_total = time.time() - start_time
            elapsed_minutes, elapsed_seconds = divmod(int(elapsed_total), 60)
            seed_time = int(time.time() - seed_start)
            status_text.text(f"The last seed took {seed_time}s")
            status_text.text(f"Completed {idx + 1}/{total_states} seeds. Elapsed: {elapsed_minutes}m {elapsed_seconds}s")

        progress_bar.empty()
        status_text.empty()
        # ---- Mean metrics across seeds ----
        metrics_mean = {key: np.mean(metrics_all[key], axis=0) for key in metrics_all}

        # ---- Model-independent evaluation ----
        ari_mean, ari_std, consensus_labels_all = \
            stability_and_consensus(labels_all=labels_all, k_values=k_values, random_states=random_states,
                                    n_samples=n_samples)
        df_summary = cls.summarize(metrics_all, ari_mean, ari_std,  k_values)
        df_summary.to_csv(f"{save_folder}/{saved_file_suffix}.csv")
        pd.DataFrame(consensus_labels_all).to_csv(f"{save_folder}/consensus_labels_all_{saved_file_suffix}.csv")
        return df_summary, metrics_all, metrics_mean, ari_mean, ari_std,  consensus_labels_all

    @staticmethod
    def mean_sd_at_k(metrics_all, metric_name, k_index):
        """
        metrics_all[metric_name] = list of lists
        outer list: seeds
        inner list: k_values
        """
        values = [seed_vals[k_index] for seed_vals in metrics_all[metric_name]]
        return np.mean(values), np.std(values)

    @classmethod
    def summarize(cls, metrics_all, ari_mean, ari_std,  k_values):
        rows = []
        for k in k_values:
            idx = list(k_values).index(k)
            sil_cos_mean, sil_cos_std = cls.mean_sd_at_k(metrics_all, "Silhouette Score (cosine)", idx)
            sil_euc_mean, sil_euc_std = cls.mean_sd_at_k(metrics_all, "Silhouette Score (euclidean)", idx)

            db_m, db_s = cls.mean_sd_at_k(metrics_all, "Davies-Bouldin Index", idx)
            row_dict={
                "Number of clusters": k,
                "Silhouette_mean (cosine)": sil_cos_mean,
                "Silhouette_std (cosine)": sil_cos_std,
                "Silhouette_mean (euclidean)": sil_euc_mean,
                "Silhouette_std (euclidean)": sil_euc_std,
                "DaviesBouldin_mean": db_m,
                "DaviesBouldin_std": db_s,
                "ARI_mean": ari_mean[idx],
                "ARI_std": ari_std[idx]
            }
            if cls.__name__ == "KMeansEngine":
                iner_m, iner_s = cls.mean_sd_at_k(metrics_all, "Inertia", idx)
                row_dict["Inertia_mean"] = iner_m
                row_dict["Inertia_std"] = iner_s
            elif cls.__name__=="GMMEngine":
                bic_m, bic_s = cls.mean_sd_at_k(metrics_all, "BIC", idx)
                aic_m, aic_s = cls.mean_sd_at_k(metrics_all, "AIC", idx)
                row_dict["BIC_mean"] = bic_m
                row_dict["BIC_std"] = bic_s
                row_dict["AIC_mean"] = aic_m
                row_dict["AIC_std"] = aic_s

            rows.append(row_dict)
        return pd.DataFrame(rows).set_index("Number of clusters")

    @classmethod
    def silhouette_analysis(cls,df_pivot, kwargs, k_values=range(2,7)):
        # Create a subplot with 1 row and n columns
        st.dataframe(df_pivot.head())
       # st.header("df_pivot.shape:"+str(df_pivot.shape))
        fig, axs = plt.subplots(1, len(k_values))
        fig.set_size_inches(12, 4)
        for n_clusters, ax1 in zip(k_values, axs.flatten()):

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(df_pivot) + (n_clusters + 1) * 10])
            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            kwargs['n_clusters'] = n_clusters
            clusterer = cls(**kwargs)
            cluster_labels = clusterer.fit_predict(df_pivot)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(df_pivot, cluster_labels, metric=clusterer.metric_for_silhouette)
            st.header("n_clusters="+str(n_clusters))
            st.write("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(df_pivot, cluster_labels)
            y_lower = 10
            for i in range(1, n_clusters+1):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                negative_indices = np.where((cluster_labels == i) & (sample_silhouette_values < 0))[0]
                provinces_with_negative_silhouette = df_pivot.index[negative_indices].tolist()
                end_statement = ""
                if provinces_with_negative_silhouette:
                    end_statement = f"Provinces with negative silhouette values: {provinces_with_negative_silhouette}"
                st.write(f"Cluster {i} has {len(negative_indices)} negative silhouette values."+end_statement)

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx( np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        col1,_=st.columns([9,1])
        col1.pyplot(fig)

    # Not used
    @staticmethod
    def remap_clusters(labels: pd.Series, priority: List[str]) -> pd.Series:
        """
        labels:   pd.Series indexed by province name, values are original kmeans.labels_
        priority: list of province names in the order you want new labels assigned.

        Returns a new pd.Series of same index with relabeled cluster IDs 0,1,2…
        """
        new_label_map = {}
        next_new = 0
        for prov in priority:
            old_lbl = labels.loc[prov]
            if old_lbl not in new_label_map:
                new_label_map[old_lbl] = next_new
                next_new += 1

        # If you have provinces outside your priority list and want to
        # give them labels too, you could continue:
        for old_lbl in sorted(set(labels) - set(new_label_map)):
            new_label_map[old_lbl] = next_new
            next_new += 1
        return labels.map(new_label_map).to_list()

