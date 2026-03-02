import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import geopandas as gpd
import streamlit as st

from viz.color_mapping import create_cluster_color_mapping


class GeoClusterPlotter:
    def __init__(self, cluster_color_mapping_dict, ha_positions, va_positions):
        """
        Initialize with configuration data.
        """
        self.cluster_color_mapping_dict = cluster_color_mapping_dict
        self.ha_positions = ha_positions
        self.va_positions = va_positions

    def create_color_mapping(self, gdf: gpd.GeoDataFrame, n_clusters: int):
        """Generate cluster color mapping with province-based defaults."""
        color_map = {}
        used_colors = set()
        clusters = set(range(1, n_clusters + 1))

        for idx, color in self.cluster_color_mapping_dict.items():
            if idx in gdf.index:
                # Handle potential Series return for index lookup
                val = gdf.loc[idx, "clusters"]
                cluster = val[0] if isinstance(val, pd.Series) else val

                if cluster not in color_map and color not in used_colors:
                    color_map[cluster] = color
                    used_colors.add(color)

        # Assign remaining colors
        remaining_colors = [c for c in self.cluster_color_mapping_dict.values() if c not in used_colors]
        remaining_clusters = clusters - set(color_map.keys())

        for i, cluster in enumerate(remaining_clusters):
            if i < len(remaining_colors):
                color_map[cluster] = remaining_colors[i]
            else:
                color_map[cluster] = "grey"  # Fallback
        return color_map

    def plot_cluster_map(self, gdf_clusters, gdf_centroids, n_clusters, year_label):
        """
        Plots the geographic clusters.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # 1. Create Colors
        ##  color_map = self.create_color_mapping(gdf_clusters, n_clusters)
        color_map = create_cluster_color_mapping(gdf_clusters, self.cluster_color_mapping_dict)

        # 2. Map colors and Plot
        # gdf_clusters = gdf_clusters.copy()
        gdf_clusters["color"] = gdf_clusters["clusters"].map(color_map)

        gdf_clusters.plot(ax=ax, color=gdf_clusters['color'], legend=True, edgecolor="black", linewidth=.2)
        ax.axis("off")
        ax.margins(x=0)

        # 3. Annotations
        bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6)

        for province in gdf_clusters.index:
            # Use configurations passed in __init__
            ha = self.ha_positions.get(province, "center")
            va = self.va_positions.get(province, "center")

            # Centroid safety check
            geom = gdf_clusters.loc[province, "geometry"]
            # Handle cases where index might be duplicated or geom is missing
            if isinstance(geom, gpd.GeoSeries): geom = geom.iloc[0]

            ax.annotate(text=province,
                        xy=(geom.centroid.x, geom.centroid.y),
                        ha=ha, va=va, fontsize=5, color="black", bbox=bbox)

        # 4. Legend & Centroid Markers
        title = f"{n_clusters} Clusters Identified {year_label}"
        ax.set_title(title)
        if gdf_centroids is not None:
            closest_provinces_centroids = gdf_centroids.to_crs("EPSG:4326")#gdf_centroids.to_crs("EPSG:4326").copy()
            closest_provinces_centroids["centroid_geometry"] = closest_provinces_centroids.geometry.centroid

            closest_points = gpd.GeoDataFrame(
                closest_provinces_centroids,
                geometry="centroid_geometry",
                crs=gdf_clusters.crs
            )
            closest_points.plot(ax=ax, facecolor="none", markersize=90,
                                edgecolor="black", linewidth=1.5,
                                label="Cluster representatives")
            ax.legend(loc="upper right", fontsize=6)
        fig.savefig(f"temp/result.png", dpi=300, bbox_inches="tight")
        st.pyplot(fig)

    def plot_elections(self, gdf_clusters):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        n_clusters = st.session_state["n_cluster"]
        # Define a color map for the categories
        # Map the colors to the GeoDataFrame
        file_name = "elections2023.csv"
        gdf_clusters["clusters"] = pd.read_csv(file_name, index_col=0)["cluster"].tolist()  # elections1-->1.figure
        color_map = {1: "darkorange", 2: "red", 3: "purple", 4: "gold"}
        gdf_clusters["color"] = gdf_clusters["clusters"].map(color_map)
        nan_rows = gdf_clusters[gdf_clusters.isna().any(axis=1)]
        print("nan rows:", nan_rows, "n_clusters", n_clusters)
        print(gdf_clusters.index)
        gdf_clusters.plot(ax=ax, color=gdf_clusters['color'], legend=True, edgecolor="black",
                               linewidth=.2)
        ax.axis("off")
        ax.margins(x=0)

        # Add province names (from index) at centroids
        bbox = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.6)
        ha_positions, va_positions = self.ha_positions, self.va_positions
        for province in gdf_clusters.index:
            province_geometry = gdf_clusters.loc[province, "geometry"]
            ha_pos,va_pos =ha_positions.get(province, "center"), va_positions.get(province, "center")
            ax.annotate(text=province,  # Use index (province name) directly
                        xy=(province_geometry.centroid.x, province_geometry.centroid.y),
                        ha=ha_pos, va=va_pos, fontsize=5, color="black", bbox=bbox)


        if file_name == "elections2022.csv":
            legend_handles = [
                Patch(facecolor='darkorange', label='People’s Alliance'),
                Patch(facecolor='red', label="Nation's Alliance"),
                Patch(facecolor='purple', label="Labour's Alliance")
            ]
            title = "2023 Turkish Parliamentary Elections: Provincial Wins by Alliance Blocs"
        else:
            legend_handles = [
                Patch(facecolor='darkorange', label='People’s Alliance (AKP + MHP)'),
                Patch(facecolor='red', label='CHP'),
                Patch(facecolor='purple', label='DEM Party')
            ]
            title = "2024 Turkish Municipal Council Elections: Provincial Wins by Alliance Blocs and Competing Parties"
        ax.legend(
            handles=legend_handles,
            loc=[.55, .87],
            fontsize=6,
            title_fontsize=6,
            frameon=False  # Remove if you want a background
        )
        ax.set_title(title)

        st.pyplot(fig)
