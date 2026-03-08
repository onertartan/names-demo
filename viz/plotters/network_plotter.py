import numpy as np
from sklearn.decomposition import PCA
import umap
import networkx as nx
from matplotlib import pyplot, pyplot as plt
import streamlit as st
import seaborn as sns

from viz.color_mapping import create_cluster_color_mapping


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


def plot_cluster_mds(distance_df, cluster_labels=None):
    from sklearn.manifold import MDS

    # MDS is perfect for 'precomputed' distance matrices
    mds = MDS(dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(distance_df.values)

    fig, ax = plt.subplots(figsize=(10, 8))

    # If you pass your cluster colors/labels, use them here
    scatter = ax.scatter(coords[:, 0], coords[:, 1], s=100, edgecolors='black', alpha=0.7)

    for i, label in enumerate(distance_df.index):
        ax.annotate(label, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    ax.set_title("Inter-Cluster Distance Projection (MDS)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Stress value indicates how well the 2D plot represents original distances
    # A low stress value ( < 0.1) means the plot is very reliable
    st.write(f"MDS Stress: {mds.stress_:.4f}")

    st.pyplot(fig)
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

def plot_umap_tsne(df_pivot, CLUSTER_COLOR_MAPPING, methods=["umap" ],title="Provincial Name Profiles (UMAP)"):
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
    st.header(str(df_features.shape))
    pca = PCA(n_components=n_pca, random_state=42)
    coords_pca = pca.fit_transform(df_features.values)
    explained = pca.explained_variance_ratio_.sum()
    st.caption(f"PCA retains {explained:.1%} of variance in {n_pca} components")

    # ── Step 2: Compute both embeddings ───────────────────────────────────────
    coords_umap = umap.UMAP(n_components=2,
                        metric="euclidean",  # cosine → euclidean after PCA+L2
                        n_neighbors=10,
                        min_dist=0.3,
                        random_state=42).fit_transform(coords_pca)

    coords_tsne = TSNE(n_components=2, metric="euclidean",
                       perplexity=10, random_state=42).fit_transform(coords_pca)
    if len(methods) == 2:
        coords_list = [coords_umap, coords_tsne]
    elif methods[0]=="t-sne":
        coords_list =[coords_tsne]
    else:
        coords_list =[coords_umap]
    # ── Side by side plot ─────────────────────────────────────────────


    fig, axes = plt.subplots(1, len(methods), figsize=(18, 8), facecolor=BG)
    axes = np.atleast_1d(axes)
    fig.suptitle(title, fontsize=16, fontweight="bold",
                 color=ACCENT, fontfamily="serif", y=1.02)

    for ax, coords, method_title in zip(axes,coords_list,["UMAP", "t-SNE (perplexity=10)"] ):
        ax.set_facecolor(PANEL_BG)

        for cluster_id in range(1, n_clusters + 1):
            mask = labels == cluster_id
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=120, color=cluster_color_map[cluster_id],
                       edgecolors=ACCENT, linewidths=0.8,
                       label=f"Cluster {cluster_id}", zorder=3)

        for i, province in enumerate(df_pivot.index):
            ax.text(coords[i, 0] + 0.05, coords[i, 1] + 0.05,
                    province, fontsize=6, color=TEXT, alpha=0.75, zorder=4)

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
