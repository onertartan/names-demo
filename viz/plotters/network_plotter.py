import plotly.graph_objects as go
import numpy as np
from matplotlib import patheffects as pe
from adjustText import adjust_text
from sklearn.decomposition import PCA
import umap
import networkx as nx
from matplotlib import pyplot, pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.metrics import pairwise_distances

from viz.color_mapping import create_cluster_color_mapping
from sklearn.manifold import MDS

from viz.config import CLUSTER_COLOR_MAPPING


def plot_cluster_network( distance_df, threshold=None):
    G = nx.Graph()

    clusters = distance_df.index

    for c in clusters:
        G.add_node(c)

    for i in clusters:
        for j in clusters:
            if i != j:
                dist = distance_df.loc[i, j]

                if threshold is None or dist < threshold:
                    G.add_edge(i, j, weight=dist)

    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(8, 6))  # <-- create explicit figure

    nx.draw(G, pos, with_labels=True, node_size=2000)

    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    st.pyplot(fig)  #

from sklearn.manifold import MDS, TSNE
import pandas as pd

def plot_mds_provinces(df_pivot,
                       metric_mds=False,
                       n_components=2,
                       random_state=42,
                       metric='euclidean',
                       figsize=(12, 10),
                       highlight_provinces=None,
                       title="MDS Projection of Provinces by Name Frequencies",
                       save_path=None):
    """
    Perform MDS on province name frequency data and visualize in 2D.

    Parameters
    ----------
    df_pivot : pd.DataFrame
        DataFrame with provinces as index, name frequencies as columns
        Shape: (81 provinces, ~n name frequencies)
    metric_mds : bool
        If True, perform metric MDS; if False, non-metric MDS
    n_components : int
        Number of dimensions for projection (default 2)
    random_state : int
        Random seed for reproducibility
    metric : str
        'euclidean' or 'precomputed'. If 'precomputed', df_pivot should be distance matrix
    figsize : tuple
        Figure size
    highlight_provinces : list
        List of province names to highlight with different colors/markers
    title : str
        Plot title
    save_path : str
        Path to save figure (optional)

    Returns
    -------
    dict : Dictionary containing:
        - 'coordinates': DataFrame with MDS coordinates
        - 'stress': Final stress value
        - 'distance_matrix': Pairwise distance matrix used
        - 'mds_object': Fitted MDS object
    """

    # Make a copy to avoid modifying original
    df_labels =df_pivot["clusters"]
    df = df_pivot.copy().drop(columns=["clusters"])
    X = df.values
   # n_pca = 80
   # pca = PCA(n_components=n_pca, random_state=336)
   # X = pca.fit_transform(df.values)

    # Compute pairwise distances if not precomputed
    if metric == 'precomputed':
        print("Computing Euclidean distances between provinces...")
        X = pairwise_distances(X, metric='euclidean')


    # MDS'in dokümantasyonuna bak (versiyona göre değişen parametreler)
    print("\nMDS parametreleri:")
    print(help(MDS.__init__))
    # Perform MDS
    print(f"Running {'metric' if metric else 'non-metric'} MDS...")
    mds = MDS(
        n_components=n_components,
        #metric="precomputed",
        metric=metric,
        metric_mds=metric_mds,
        random_state=random_state,
        n_init=10,
        max_iter=500,
        eps=1e-9,
        verbose=0,normalized_stress=True
    )

    # Fit MDS
    # Fit MDS
    coordinates = mds.fit_transform(X)
    st.write(mds.dissimilarity_matrix_)

    # Create DataFrame with coordinates
    coord_df = pd.DataFrame(
        coordinates,
        index=df.index,
        columns=['MDS1', 'MDS2'] if n_components==2 else ["MDS1","MDS2","MDS3"]
    )

    # Print stress value
    stress = mds.stress_
    print(f"✓ MDS completed with stress: {stress:.4f}")

    df_labels = df_pivot["clusters"]
    cluster_color_map = create_cluster_color_mapping(df_pivot, CLUSTER_COLOR_MAPPING)
    for k, v in cluster_color_map.items():
        if v=="darkorange" or v=="peru":
            cluster_color_map[k]="orange"
    # Map each province → its color via its cluster label
    point_colors = [cluster_color_map.get(df_labels.loc[province], "#aaaaaa") for province in coord_df.index]

    # ------------------------------------------------------------------ #
    #  Plot                                                              #
    # ------------------------------------------------------------------ #
    # Define the list of provinces to highlight
    highlight_provinces = {"Adana": "red", "Adıyaman": "orange", "Artvin": "orange", "Bingöl": "orange",
                           "Bitlis": "orange", "Burdur": "orange", "Kars": "orange", "Manisa": "orange",
                           "İstanbul": "orange", "Tunceli": "purple", "Uşak": "orange"}
    annotation_fontsize=4
    if n_components == 2:
        fig, ax = plt.subplots()

        # Scatter — one point per province
        for i, province in enumerate(coord_df.index):
            is_highlighted = province in highlight_provinces
            ax.scatter(
                coord_df.loc[province, "MDS1"],
                coord_df.loc[province, "MDS2"],
                c=point_colors[i],
                s=40 if is_highlighted else 18,
                edgecolors=highlight_provinces.get(province,"black"),
                linewidth=2 if is_highlighted else 0.5,
                zorder=3 if is_highlighted else 2,
                alpha=0.92,
            )

        # Province annotations — adjustText nudges labels apart and draws leader lines
        texts = []
        for province in coord_df.index:
            province_name_shortened = province
            if province=="Afyonkarahisar":
                province_name_shortened="Afyon"
            elif province=="Kahramanmaraş":
                province_name_shortened="K.Maraş"
            x, y = coord_df.loc[province, "MDS1"], coord_df.loc[province, "MDS2"]
            t = ax.text(
                x, y,
                province_name_shortened,
                fontsize=annotation_fontsize,
                fontweight="bold" if province in highlight_provinces else "normal",
                color="#111111",
                #path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=4,
            )
            texts.append(t)

        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5),
            expand_points=(1.2, 1.2),
            expand_text=(1.2, 1.3),
            force_text=(0.4, 0.5),
            force_points=(0.8, 1.2),
        )
        group_legend = True
        if group_legend:
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Group I',
                       markerfacecolor='red', markersize=7, markeredgecolor='black'),
                Line2D([0], [0], marker='o', color='w', label='Group II',
                       markerfacecolor='orange', markersize=7, markeredgecolor='black'),
                Line2D([0], [0], marker='o', color='w', label='Group III',
                       markerfacecolor='purple', markersize=7, markeredgecolor='black')
            ]

            ax.legend(handles=legend_elements, title="Groups")
            # Stress watermark
            ax.text(
                0.01, 0.01,
                f"Stress = {stress:.4f}",
                transform=ax.transAxes,
                fontsize=8,
                color="#555555",
                va="bottom",
            )

            ax.set_title(title, fontsize=13, pad=14)
            ax.set_xlabel("MDS Dimension 1", fontsize=10)
            ax.set_ylabel("MDS Dimension 2", fontsize=10)

            ax.tick_params(labelsize=8)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
            ax.axhline(0, color="#dddddd", linewidth=0.6)
            ax.axvline(0, color="#dddddd", linewidth=0.6)
            fig.tight_layout()

            st.pyplot(fig)
        else:
            # Cluster legend — one patch per unique cluster, sorted by cluster id
            from matplotlib.patches import Patch
            del cluster_color_map[-1]
            legend_handles = [Patch(facecolor=color, edgecolor="#555555",  linewidth=0.6, label=f"Cluster {cluster_id}", )
                for cluster_id, color in sorted(cluster_color_map.items()) ]
            ax.legend(handles=legend_handles, title="Cluster", fontsize=8, title_fontsize=9, loc="best", framealpha=0.88, edgecolor="#cccccc" )
    else:  # n_components == 3

        # ------------------------------------------------------------------ #
        # Depth cues: scale size/opacity/border by Z position                 #
        # ------------------------------------------------------------------ #
        z_vals = coord_df["MDS3"].values
        z_min, z_max = z_vals.min(), z_vals.max()
        z_norm = (z_vals - z_min) / (z_max - z_min + 1e-9)  # 0 = back, 1 = front

        def depth_size(i, base_size):
            return base_size * (0.5 + 0.8 * z_norm[i])  # 0.5x–1.3x base

        def depth_opacity(i):
            return 0.35 + 0.65 * z_norm[i]  # 0.35–1.0

        # ------------------------------------------------------------------ #
        # Hover text and shortened labels                                      #
        # ------------------------------------------------------------------ #
        hover_texts = []
        for province in coord_df.index:
            cluster_id = df_labels.loc[province]
            hover_texts.append(f"<b>{province}</b><br>Cluster: {cluster_id}")

        short_names = []
        for province in coord_df.index:
            if province == "Afyonkarahisar":
                short_names.append("Afyon")
            elif province == "Kahramanmaraş":
                short_names.append("K.Maraş")
            else:
                short_names.append(province)

        # ------------------------------------------------------------------ #
        # Split indices: highlighted vs normal                                 #
        # ------------------------------------------------------------------ #
        normal_idx = [i for i, p in enumerate(coord_df.index) if p not in highlight_provinces]
        highlight_idx = [i for i, p in enumerate(coord_df.index) if p in highlight_provinces]

        # ------------------------------------------------------------------ #
        # Trace factory                                                        #
        # ------------------------------------------------------------------ #
        def make_trace(indices, name, base_size=5, is_highlight=False):
            return go.Scatter3d(
                x=coord_df.iloc[indices]["MDS1"],
                y=coord_df.iloc[indices]["MDS2"],
                z=coord_df.iloc[indices]["MDS3"],
                mode="markers+text",
                name=name,
                marker=dict(
                    size=[depth_size(i, 9 if is_highlight else base_size) for i in indices],
                    color=[point_colors[i] for i in indices],
                    opacity=0.95 if is_highlight else 0.80,
                    line=dict(
                        color=[f"rgba(0,0,0,{depth_opacity(i):.2f})" for i in indices],
                        width=2.5 if is_highlight else 1.2,
                    ),
                ),
                text=[short_names[i] for i in indices],
                textposition="top center",
                textfont=dict(
                    size=9 if is_highlight else 7,
                    color=[f"rgba(0,0,0,{depth_opacity(i):.2f})" for i in indices],
                ),
                hovertext=[hover_texts[i] for i in indices],
                hoverinfo="text",
            )

        traces = [
            make_trace(normal_idx, name="Provinces", base_size=5, is_highlight=False),
            make_trace(highlight_idx, name="Highlighted", base_size=9, is_highlight=True),
        ]

        # Invisible traces for cluster legend
        for cluster_id, color in sorted(cluster_color_map.items()):
            traces.append(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                name=f"Cluster {cluster_id}",
                marker=dict(size=8, color=color, line=dict(color="#555555", width=0.5)),
                showlegend=True,
            ))

        # ------------------------------------------------------------------ #
        # Assemble figure                                                      #
        # ------------------------------------------------------------------ #
        fig_3d = go.Figure(data=traces)
        fig_3d.update_layout(
            title=dict(
                text=f"{title}<br><sup>Stress = {stress:.4f}</sup>",
                font=dict(size=14),
            ),
            scene=dict(
                xaxis=dict(title="MDS1", showgrid=True, gridcolor="#dddddd", backgroundcolor="#f0f0f0"),
                yaxis=dict(title="MDS2", showgrid=True, gridcolor="#dddddd", backgroundcolor="#ebebeb"),
                zaxis=dict(title="MDS3", showgrid=True, gridcolor="#dddddd", backgroundcolor="#e5e5e5"),
                aspectmode="cube",
            ),
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
                eye=dict(x=1.5, y=-1.8, z=0.7),
            ),
            legend=dict(
                title="Legend",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#cccccc",
                borderwidth=1,
            ),
            margin=dict(l=0, r=0, t=60, b=0),
            height=700,
        )

        st.plotly_chart(fig_3d, use_container_width=True)
    D_original = pairwise_distances(X, metric='euclidean')
    plot_shepard(D_original, coord_df, stress)


def plot_clustered_heatmap(distance_df, title="Inter-Cluster Distance Map"):
    BG       = "#0f1117"
    PANEL_BG = "#1a1d27"
    ACCENT   = "#c9a96e"
    CMAP     = "YlOrRd"  # light→dark, high contrast with both black and white text

    n = len(distance_df)
    fig, ax = plt.subplots(figsize=(max(6, n + 2), max(6, n + 2)), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    st.dataframe(distance_df)
    # Draw heatmap without annotations first
    sns.heatmap(
        distance_df,
        ax=ax,
        cmap=CMAP,
        annot=False,
        linewidths=1.2,
        linecolor="#2e3245",
        cbar_kws={"shrink": 0.6},
    )

    # Manually annotate with contrast-aware text color
    data = distance_df.values
    vmin, vmax = data.min(), data.max()
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            normalized = (val - vmin) / (vmax - vmin + 1e-9)
            text_color = "black" if normalized < 0.55 else "white"
            ax.text(j + 0.5, i + 0.5, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    ax.set_title(title, fontsize=16, fontweight="bold",
                 color=ACCENT, fontfamily="serif", pad=16)
    ax.set_xlabel("Cluster", fontsize=13, labelpad=10, color=ACCENT)
    ax.set_ylabel("Cluster", fontsize=13, labelpad=10, color=ACCENT)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12, color="white")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12, color="white")

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white", labelsize=9)
    cbar.set_label("Distance", color="white", fontsize=10)

    fig.tight_layout()
    st.pyplot(fig)

def plot_umap_tsne(df_pivot, CLUSTER_COLOR_MAPPING, methods=["umap"],title="Projection of Provincial Name Profiles (UMAP)"):
    BG       = "#0f1117"
    PANEL_BG = "#1a1d27"
    ACCENT   = "#c9a96e"
    TEXT     = "#e8e3d8"
    # color_map is a dictionary mapping cluster ids to colors
    cluster_color_map = create_cluster_color_mapping(df_pivot, CLUSTER_COLOR_MAPPING)
    cluster_color_map = {k: ("yellow" if v == "darkorange" else v) for k, v in cluster_color_map.items()}
    #df_color = df_pivot["clusters"].map(cluster_color_map)

    df_features = df_pivot.drop(columns=["clusters"])
    labels = df_pivot["clusters"]
    n_clusters = df_pivot["clusters"].max()
    # ── Step 1: PCA to 50 dimensions ─────────────────────────────────
    n_pca = 80
    pca = PCA(n_components=n_pca, random_state=336)
    coords_pca = pca.fit_transform(df_features.values)
    explained = pca.explained_variance_ratio_.sum()
    st.caption(f"PCA retains {explained:.1%} of variance in {n_pca} components")

    # ── Step 2: Compute both embeddings ───────────────────────────────────────
    coords_umap = umap.UMAP(n_components=2,
                        metric="euclidean",  # cosine → euclidean after PCA+L2
                        n_neighbors=10,
                        min_dist=0.3,
                        random_state=336).fit_transform(coords_pca)

    coords_tsne = TSNE(n_components=2, metric="euclidean",
                       perplexity=10, random_state=336).fit_transform(coords_pca)
    if len(methods) == 2:
        coords_list = [coords_umap, coords_tsne]
    elif methods[0]=="t-sne":
        coords_list =[coords_tsne]
    else:
        coords_list =[coords_umap]
    # ── Side by side plot ─────────────────────────────────────────────

    # Define the list of provinces to highlight
    highlight_provinces = {"Adana":"red", "Adıyaman":"yellow", "Artvin":"yellow", "Bingöl":"yellow",
                           "Bitlis":"yellow", "Burdur":"yellow", "Kars":"yellow", "Manisa":"yellow",
                           "İstanbul":"red", "Tunceli":"purple", "Uşak":"yellow"}

    # Create boolean mask for highlighted provinces
    highlight_mask = [province in highlight_provinces for province in df_pivot.index]

    fig, axes = plt.subplots(1, len(methods), figsize=(18, 8), facecolor=BG)
    axes = np.atleast_1d(axes)
    fig.suptitle(title, fontsize=16, fontweight="bold",
                 color=ACCENT, fontfamily="serif", y=1.02)

    for ax, coords, method_title in zip(axes, coords_list, ["UMAP", "t-SNE (perplexity=10)"]):
        ax.set_facecolor(PANEL_BG)

        # Plot all points (normal)
        for cluster_id in range(1, n_clusters + 1):
            mask = labels == cluster_id
            df_cluster = df_pivot[mask]
            if any(mask):
                for province in df_cluster.index:
                        ax.scatter(coords[df_pivot.index.get_loc(province), 0], coords[df_pivot.index.get_loc(province), 1],
                                   s=120, color=cluster_color_map[cluster_id],
                                   edgecolors=highlight_provinces.get(province,ACCENT),    linewidths=2 if province in highlight_provinces else 0.5,
                                    zorder=3)


        # Add labels for all provinces
        for i, province in enumerate(df_pivot.index):
            ax.text(coords[i, 0] + 0.05, coords[i, 1] + 0.05,
                    province, fontsize=6, color=TEXT, alpha=0.75, zorder=5)

        ax.set_title(method_title, fontsize=13, fontweight="bold",
                     color=ACCENT, fontfamily="serif", pad=10)
        ax.set_xlabel(f"{method_title.split()[0]}-1", fontsize=11, color=TEXT)
        ax.set_ylabel(f"{method_title.split()[0]}-2", fontsize=11, color=TEXT)
        ax.tick_params(colors=TEXT)
        ax.legend(fontsize=9, facecolor=PANEL_BG,
                  edgecolor=ACCENT, labelcolor=TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3245")

    fig.tight_layout()
    fig.savefig(f"temp/umap.png", dpi=300, bbox_inches="tight")
    st.pyplot(fig)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_custom_silhouette(df_pivot):
    # 1. Prepare data
    X = df_pivot.drop(columns=['clusters']).values
    cluster_labels = df_pivot['clusters'].values
    n_clusters = max(cluster_labels)

    # 2. Calculate scores
    avg_score = silhouette_score(X, cluster_labels)
    sample_values = silhouette_samples(X, cluster_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_lower = 10

    # 3. Plot each cluster's "silhouette blade"
    for i in range(1, n_clusters + 1):
        # Aggregate scores for cluster i and sort them
        ith_cluster_values = sample_values[cluster_labels == i]
        ith_cluster_values.sort()

        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i - 1) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        # Label the clusters in the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # Gap between blades

    ax.set_title("Silhouette Analysis for Provincial Naming Clusters")
    ax.set_xlabel("Silhouette Coefficient Values")
    ax.set_ylabel("Cluster Label")

    # Draw the vertical line for the average score
    ax.axvline(x=avg_score, color="red", linestyle="--", label=f"Average ({avg_score:.2f})")
    ax.set_yticks([])  # Clear y-axis labels
    ax.legend()
    st.dataframe(pd.DataFrame(sample_values,index=df_pivot.index))

    st.pyplot(fig)

from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np

def plot_shepard(D_original, coord_df, stress, ax=None):
    """
    D_original : (n x n) pairwise distance matrix used as MDS input
    coord_df   : MDS output coordinates (provinces × MDS1/MDS2)
    """
    # Distances in reduced space
    D_mds = pairwise_distances(coord_df.values, metric="euclidean")

    # Extract upper triangle only (avoid duplicates and diagonal)
    idx = np.triu_indices_from(D_original, k=1)
    orig = D_original[idx]
    reduced = D_mds[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(orig, reduced, alpha=0.3, s=8, color="#4C9BE8", edgecolors="none")

    # Reference line (perfect preservation)
    lims = [min(orig.min(), reduced.min()), max(orig.max(), reduced.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")

    ax.set_xlabel("Original dissimilarity", fontsize=10)
    ax.set_ylabel("MDS distance", fontsize=10)
    ax.set_title(f"Shepard Diagram  (stress = {stress:.4f})", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    st.pyplot(fig)