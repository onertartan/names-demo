import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import re

DEFAULT_METRICS = [
    #("Silhouette_mean", "Cluster separation", "Separation"),
    ("ARI_mean", "Clustering stability", "Stability"),
]

DEFAULT_SCALER_ABBR = {
    "Share of Top 30 (L1 Norm)": "L1",
    "Share of Total": "S",
    "TF-IDF": "TF",
    "L2 Normalization": "L2",
}
def _parse_spectral_filename(fname: Path,geometry):
    """
    Expected filename format:
    {affinity}_{scaler}_{year1}_{year2}_{n}.csv
    Examples:
    nearest_neighbors_Share of Total_2018_2024_5.csv
    rbf_TF-IDF_2018_2024_10.csv
    """
    m = re.match(
        r"^(nearest_neighbors|rbf)_(.+?)_(\d{4}_\d{4})_(\d+)\.csv$",
        fname.name
    )
    if m is None:
        return None

    affinity = m.group(1)          # nearest_neighbors | rbf
    scaler = m.group(2)            # may contain spaces
    year_range = m.group(3)        # 2018_2024
    n_neighbors = int(m.group(4))  # kNN size (or gamma index if rbf)
    return affinity, scaler, year_range, n_neighbors


def load_spectral_results(data_dir, geometry):
    data_dir +="/SpectralClusteringEngine"
    DATA_DIR = Path(data_dir)
    files = sorted(f for f in DATA_DIR.glob("*.csv")  if not f.name.startswith("consensus_labels_all_") )

    dfs = []
    for f in files:
        parsed = _parse_spectral_filename(f, geometry)
        if parsed is None:
            continue

        affinity, scaler, year_range, n_nb = parsed
        df = pd.read_csv(f)
        df["Number of clusters"] = df["Number of clusters"].astype(int)
        df.rename(columns={"Number of clusters":"k"}, inplace=True)
        df["geometry"] = "euclidean"
        df["affinity"] = affinity
        df["scaler"] = scaler
        df["n_neighbors"] = n_nb
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No files found for geometry={geometry}")
    print("shape:",pd.concat(dfs, ignore_index=True).shape,"unique ",pd.concat(dfs, ignore_index=True)["geometry"].unique())
    return pd.concat(dfs, ignore_index=True)

def plot_spectral_row(
    ax_row,
    df_all,
    metric="ARI_mean",
    ylabel="Clustering stability",
    title="Stability",
    scaler_abbr=DEFAULT_SCALER_ABBR,
    show_comparison_col=True,
    alpha_thin=0.50,
    linewidth_thin=2.0,
    linewidth_best=3.0,
    marker_best="o",line_style="solid",
    show_annotation=False,
    color_map=None
):
    scalers = list(dict.fromkeys(df_all["scaler"]))
    k_vals = np.sort(df_all["k"].unique())

    # Fixed colors per scaler (stable order)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # ---- per-scaler columns ----
    for j, scaler in enumerate(scalers):
        ax = ax_row[j]
        df_s = df_all[df_all["scaler"] == scaler]

        # ---- Thin lines: all n_neighbors + labels at k = 9 ----
        label_k = 10

        curves = []
        for n_nb, g in df_s.groupby("n_neighbors"):
            g = g.sort_values("k")
            # Plot curve
            ax.plot(
                g["k"],
                g[metric],
                linewidth=linewidth_thin,
                alpha=alpha_thin,
                linestyle=line_style,
                color=(None if color_map is None else color_map.get(n_nb))
            )

            # Only annotate if k=9 exists for this curve
            if label_k in g["k"].values and metric=="ARI_mean":
                y_at_label_k = g.loc[g["k"] == label_k, metric].values[0]
                curves.append((n_nb, y_at_label_k))


        # Sort curves by y-value (to control stacking order)
        curves_sorted = sorted(curves, key=lambda x: x[1])
        # Add vertical spacing using pixel offsets
        for idx, (n_nb, y_val) in enumerate(curves_sorted):
            if metric == "ARI_mean" and scaler == "TF-IDF":
                if n_nb == 8:
                    y_val += .01
                if n_nb == 9:
                    y_val -= 0.005
                if n_nb == 4:
                    y_val += 0.003
                if n_nb == 5:
                    y_val -= 0.003
            x_shift = 0
            if metric == "ARI_mean" and scaler=="L2 Normalization":
                if n_nb > 10:
                    y_val-=0.01
                if n_nb==8:
                    y_val+=0.01
            ax.annotate(
                f"{n_nb}",
                xy=(label_k, y_val),
                xytext=(x_shift, 0),  # small right shift + vertical spacing
                textcoords="offset points",
                fontsize=9,
                ha="left",
                va="center",
                alpha=0.9,
                clip_on=False
            )

        # Best n_neighbors per k
        # ---- select best per k: max ARI_mean, tie → smallest n_neighbors ----
        df_best = (
            df_s
            .sort_values(
                by=["k", metric, "n_neighbors"],
                ascending=[True, False, True]
            )
            .groupby("k", as_index=False)
            .first()
            .sort_values("k")
        )
        if show_annotation:
            for _, row in df_best.iterrows():
                    ax.text(
                        row["k"], row[metric],
                        f"{int(row['n_neighbors'])}",
                        fontsize=10, fontweight="bold",
                        va="bottom", ha="center", color="black"
                    )

        ax.set_title(f"{title} with scaler {scaler}", fontsize=11)
        ax.set_ylabel(f"{ylabel}")
        y_lim_min= 0.7 if metric=="ARI_mean" else  -.1
        ax.set_ylim(y_lim_min, 1.01)
        ax.grid(True, alpha=0.2)
        ax_row[-1].legend(["Euclidean distance based","Cosine distance based"],
                           ncol=1,
                           frameon=True,
                           fancybox=True,
                           framealpha=0.95
                           )
    # ---- last column: comparison across scalers + stacked annotations ----
    if show_comparison_col:
        ax_comp = ax_row[-1]

        # Track per-k labels to stack
        k_data = {k: {"infos": [], "max_y": -np.inf, "min_y": np.inf} for k in k_vals}

        df_hit = df_s[df_s["ARI_mean"] ==1]
        result = (df_hit.groupby("k")["n_neighbors"].unique())
        result_sorted = result.apply(lambda x: sorted(x))

        for sc, scaler in enumerate(scalers):
            df_s = df_all[df_all["scaler"] == scaler]

            # ---- select best per k: max ARI_mean, tie → smallest n_neighbors ----
            df_best = (
                df_s
                .sort_values(
                    by=["k", metric, "n_neighbors"],
                    ascending=[True, False, True]
                )
                .groupby("k", as_index=False)
                .first()
                .sort_values("k")
            )

            abbr = scaler_abbr.get(scaler, scaler)
            color = colors[sc % len(colors)]

            ax_comp.plot(
                df_best["k"], df_best[metric],
                marker=marker_best,
                linewidth=linewidth_best,
                label=abbr,
                color=color,
                alpha=0.7
            )

            for _, row in df_best.iterrows():
                k_val = int(row["k"])
                val = float(row[metric])

                k_data[k_val]["infos"].append(
                    (f"{abbr}-{int(row['n_neighbors'])}", color)
                )
                k_data[k_val]["max_y"] = max(k_data[k_val]["max_y"], val)
                k_data[k_val]["min_y"] = min(k_data[k_val]["min_y"], val)

        # ---- stack annotations (unchanged logic) ----
        for k_val, data in k_data.items():
            peak_y = data["max_y"]
            bottom_y = data["min_y"]

            for rank, (text, color) in enumerate(data["infos"]):
                y_offset = 15 + rank * 8

                if metric == "ARI_mean":
                    y_offset = -15 - abs(1 - bottom_y) * 140 - rank * 10
                elif k_val <= 3:
                    y_offset -= 10
                ax_comp.annotate(
                    text,
                    xy=(k_val, peak_y),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    fontsize=6,
                    fontweight="bold",
                    color=color,
                    ha="center",
                    va="bottom",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0)
                )

        ax_comp.set_title("Best-per-k across scalers", fontsize=11)

        ax_comp.set_ylim(0, 1.05)
        ax_comp.set_ylabel(ylabel)
        ax_comp.set_yticks(np.arange(0, 1.01, 0.1))
        ax_comp.grid(True, alpha=0.2)

        # Legend: show once per row; place inside panel
        ax_comp.legend(fontsize=8, loc="center", title="Scaler")



def plot_spectral_k_analysis_two_geometries(
    geometries=None,
    data_dir=None,
    metrics=["ARI_mean"],
    figsize_per_row=4.0,
    show_comparison_col=False,
    save_path=None,
    dpi=300,
    ylabel=None,
    title_word="Stability"
):
    dfs = {g: load_spectral_results(data_dir, g) for g in geometries}
    scalers = list(dict.fromkeys(dfs[geometries[0]]["scaler"]))
    n_cols = len(scalers) + (1 if show_comparison_col else 0)
    n_rows = len(geometries)

    fig_size=(5 * n_cols, figsize_per_row * n_rows)
    fig, axes = plt.subplots( nrows=n_rows, ncols=n_cols, figsize=fig_size, sharex=True, constrained_layout=True)

    if n_rows == 1:
        axes = np.array([axes])
    MAX_K_TO_PLOT = 10

    for i, geom in enumerate(geometries):
        df_geom = dfs[geom]
        n_neighbors_vals = sorted(df_geom["n_neighbors"].unique())
        cmap = plt.get_cmap("tab10")
        color_map = {
            n_nb: cmap(idx % cmap.N)
            for idx, n_nb in enumerate(n_neighbors_vals)
        }
        for metric in metrics:
            if "(cosine)" in metric:
                line_style = "solid"
            elif "(euclidean)" in metric:
                line_style = "dashed"
            else:
                line_style = "solid"
            df_all = df_geom
            df_all = df_all[df_all["k"] <= MAX_K_TO_PLOT]

            plot_spectral_row(
                ax_row=axes[i],
                df_all=df_all,
                metric=metric,
                ylabel=ylabel,
                title="Seperability" if metric!="ARI_mean" else "Stability",#f"{geom.capitalize()} geometry",
                show_comparison_col=show_comparison_col,
                line_style=line_style,
                color_map=color_map
            )
        style_ax = axes[i][-1]
        if title_word == "Seperability":
            # Legend for line styles (cosine vs euclidean silhouette)
            style_handles = [
                Line2D([0], [0], color="black", lw=2, linestyle="dashed"),
                Line2D([0], [0], color="black", lw=2, linestyle="solid"),
            ]
            style_labels = ["Euclidean distance based", "Cosine distance based"]
            style_legend = style_ax.legend(
                style_handles,
                style_labels,
                loc="upper center",
                ncol=1,
                frameon=True,
                fancybox=True,
                framealpha=0.95,
            )
            style_ax.add_artist(style_legend)

        # Legend for n_neighbors colors
        nn_handles = [
            Line2D([0], [0], color=color_map[n_nb], lw=2, linestyle="solid")
            for n_nb in n_neighbors_vals
        ]
        nn_labels = [str(n_nb) for n_nb in n_neighbors_vals]
        nn_legend = style_ax.legend(
            nn_handles,
            nn_labels,
            loc="best",
            ncol=1,
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            title="n_neighbors"
        )
        style_ax.add_artist(nn_legend)
        #axes[i][0].annotate(geom.capitalize(),xy=(-0.15, 0.5),xycoords="axes fraction", fontsize=11, fontweight="bold", rotation=90,va="center")

        df_first = dfs[geometries[0]] # dfs is a dict, so we take the first geometry's df to get the k values for x-axis ticks
        k_vals = np.sort(df_first[df_first["k"]<= MAX_K_TO_PLOT]["k"].unique())
        for ax in axes.flat:
            ax.set_xticks(k_vals)
            ax.set_xlabel("Number of clusters (k)")
            if title_word=="Seperability":
                ax.set_ylim(0.,0.55)

        fig.suptitle(f"Sensitivity of Spectral Clustering {title_word} to Cluster Resolution, Normalization Scheme and Number of Neighbors", fontsize=13)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig,dfs

def save_max_silhouette_per_geometry(data_dir=None, affinity=None, geometries=None):

    for geom in geometries:
        df = load_spectral_results(data_dir, geom)
        # choose the correct silhouette column
        if geom == "euclidean":
            sort_by_cols = ["k", "Silhouette_mean (euclidean)", "Silhouette_mean (cosine)", "ARI_mean"]
        else:
            sort_by_cols = ["k", "Silhouette_mean (cosine)", "Silhouette_mean (euclidean)", "ARI_mean"]

        df=df[["k","Silhouette_mean (cosine)", "Silhouette_mean (euclidean)", "ARI_mean","scaler","geometry","n_neighbors"]].round(3)
        # ---- max silhouette per k ----
        df_best=df.sort_values(by=sort_by_cols,ascending=[True, False,False,False]).groupby("k", as_index=False).first()
        df_best.T.to_csv(f"{data_dir}/SpectralClustering_{affinity}_{geom}_best.csv")
        df = df.set_index("k")
        df = df.sort_values(by=sort_by_cols, ascending=[True, False,False,False])
        df.to_csv(f"{data_dir}/SpectralClustering_{affinity}_{geom}_all.csv")


def plot_spectral_ari1_regions_two_geometries(
    geometries=("cosine", "euclidean"),
    data_dir=None,
    metric="ARI_mean",
    scaler_abbr=DEFAULT_SCALER_ABBR,
    figsize_per_col=8,
    alpha=0.6,
    point_size=35,
    save_path=None,
    dpi=300,
):
    """
    Plot (k, n_neighbors) points where ARI_mean == 1
    for two geometries, side by side (1×2 layout).
    """

    # ---- load data ----
    dfs = {g: load_spectral_results(data_dir, g) for g in geometries}
    # consistent scaler order & colors
    scalers = list(dict.fromkeys(dfs[geometries[0]]["scaler"]))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_map = {
        scaler: colors[i % len(colors)]
        for i, scaler in enumerate(scalers)
    }

    n_cols = len(geometries)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_cols,
        figsize=(figsize_per_col * n_cols, 5),
        sharey=True,
        constrained_layout=True
    )
    MAX_K_TO_PLOT = 10

    # ensure iterable
    if n_cols == 1:
        axes = [axes]

    # ---- plotting ----
    for ax, geom in zip(axes, geometries):
        df = dfs[geom]
        df=df[df["k"]<=MAX_K_TO_PLOT]
        # filter perfect-stability points
        df_hit = df[df[metric] > .9]

        for i, scaler in enumerate(scalers):
            df_s = df_hit[df_hit["scaler"] == scaler]
            if df_s.empty:
                continue

            # deterministic jitter per scaler
            jitter = (hash(scaler) % 7 - 3) * 0.15  # small, stable offset
            jitter=(i%4-1.5)*0.1
            ax.scatter(
                df_s["k"] + jitter,
                df_s["n_neighbors"],
                s=point_size,
                label="Mean ARI",
                color=color_map[scaler],
                edgecolor="black",
                linewidth=0.4
            )

        #ax.set_title(f"Normalization Schemes ARI_mean>0.9", fontsize=12)
        ax.set_xlabel("Number of clusters (k)")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("n_neighbors")

    # ---- shared legend ----
    fig.legend(scalers,
        title="Scaler",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.89),
        ncol=1,
       frameon=True,
       fancybox=True,
      framealpha=0.95
      )

    fig.suptitle(
        "Neighborhood sizes yielding high clustering stability (ARI_mean > 0.9)",
        fontsize=13
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, dfs



genders= ["both genders"]
affinity = "nearest_neighbors"
for gender in genders:
    data_dir = "files/"
    if gender:
        data_dir += gender
    #geometries= ("cosine", "euclidean")
    geometries = ("euclidean",)
    fig, dfs = plot_spectral_k_analysis_two_geometries(data_dir=data_dir, geometries=geometries,  metrics=["ARI_mean"],ylabel="mean ARI",save_path=data_dir+"/stability_spectral_ari_mean.png")
    plt.show()
    fig, dfs = plot_spectral_ari1_regions_two_geometries(data_dir=data_dir, geometries=geometries, metric="ARI_mean",save_path=data_dir+"/stability_spectral_ari1_regions.png")
    plt.show()
    fig, dfs = plot_spectral_k_analysis_two_geometries(data_dir=data_dir, geometries=geometries,
                                                           metrics=["Silhouette_mean (cosine)","Silhouette_mean (euclidean)"],
                                                           ylabel="Mean Silhouette Score",save_path=data_dir+f"/spectral_silhouette_{affinity}.png",title_word="Seperability")
    plt.show()
    save_max_silhouette_per_geometry(data_dir,affinity,geometries)
