import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


scaler_labels = {
    "Share of Top 30 (L1 Norm)": "L1 (Top 30)",
    "Share of Total": "Share (All)",
    "TF-IDF": "TF–IDF",
    "L2 Normalization": "L2",
}

linestyles = {
    "diag": "-",
    "tied": "--",
    "spherical": ":"
}

METRIC = "ARI_mean"


def _parse_gmm_filename(fname: Path):
    name = fname.name
    if name.startswith("consensus_labels_all_"):
        return None

    m = re.match(r"(diag|tied|spherical)_(.+)_(\d{4}_\d{4})\.csv$", name)
    if m:
        return m.group(1), m.group(2), m.group(3)

    m = re.match(r"(.+)_(diag|tied|spherical)_(\d{4}_\d{4})\.csv$", name)
    if m:
        return m.group(2), m.group(1), m.group(3)

    return None


def load_gmm_results(data_dir):
    data_dir = Path(data_dir)
    dfs = []
    for f in data_dir.glob("*.csv"):
        parsed = _parse_gmm_filename(f)
        if parsed is None:
            continue

      #  scaler, cov, year_range = parsed
        cov, scaler, year_range = parsed
        df = pd.read_csv(f)

        if "Number of clusters" in df.columns:
            df["k"] = df["Number of clusters"].astype(int)
        else:
            df["k"] = df["k"].astype(int)

        df["scaler"] = scaler
        df["covariance"] = cov
        df["year_range"] = year_range
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No matching GMM result files found.")

    return pd.concat(dfs, ignore_index=True)


import numpy as np
import matplotlib.pyplot as plt

def plot_gmm_k_analysis_one_row(
    main_data_dir,
    metric="ARI_mean",
    ylabel="Mean ARI",
    covariances=("diag", "tied", "spherical"),
    colors=None,
    linewidth=2.0,
    marker="o"
    ,title_word=""
):
    """
    Plot GMM stability results in a 1×3 row:
      - Each subplot = one covariance type
      - Each line = one normalization scheme (scaler)

    Parameters
    ----------
    df_all : pd.DataFrame
        Must contain columns: ['k', metric, 'scaler', 'covariance'].
    """
    data_dir = main_data_dir +"/GMMEngine/"
    df_all= load_gmm_results(data_dir)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 4),
        constrained_layout=True,
    )

    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    scalers = list(dict.fromkeys(df_all["scaler"]))
    k_vals = np.sort(df_all["k"].unique())

    # ---- loop over covariance types (panels) ----
    for j, cov in enumerate(covariances):
        ax = axes[j]
        df_cov = df_all[df_all["covariance"] == cov]

        # ---- plot each scaler as a line ----
        for i, scaler in enumerate(scalers):
            df_s = df_cov[df_cov["scaler"] == scaler]

            # best-per-k (max ARI, tie → smallest k handled upstream)
            df_best = (
                df_s
                .sort_values(by=["k", metric], ascending=[True, False])
                .groupby("k", as_index=False)
                .first()
                .sort_values("k")
            )

            ax.plot(
                df_best["k"],
                df_best[metric],
                marker=marker,
                linewidth=linewidth,
                color=colors[i % len(colors)],
                label=scaler,
                alpha=0.8,
            )

        ax.set_title(f"{cov.capitalize()} covariance", fontsize=11)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(k_vals)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # ---- shared legend (only once) ----
    axes[-1].legend(
        title="Normalization scheme",
        fontsize=9,
        title_fontsize=9,
        bbox_to_anchor=(0.95, 0.4),
        frameon=True,
    )

    fig.suptitle(f"Sensitivity of Gaussian Mixture Model {title_word.capitalize()} to Covariance Structure, Cluster Resolution and Data Normalization", fontsize=13)
    fig.savefig(f"{main_data_dir}/stability_gmm.png", dpi=300, bbox_inches="tight")
    return fig


def save_max_silhouette_per_geometry(main_data_dir):

    dfs = load_gmm_results(main_data_dir+ "/GMMEngine")

    for cov in ["diag", "tied", "spherical"]:
        df = dfs[dfs["covariance"] == cov]
        # choose the correct silhouette column
        columns = ["Silhouette_mean (cosine)", "Silhouette_mean (euclidean)"]
        df = df[["k","Silhouette_mean (cosine)", "Silhouette_mean (euclidean)", "ARI_mean","AIC_mean","BIC_mean","scaler"]].round(3)
        # ---- max silhouette per k ----


        df_best=df.sort_values(by=["k", columns[0]],ascending=[True, False]).groupby("k", as_index=False).first()
        columns = ["Silhouette_mean (cosine)", "Silhouette_mean (euclidean)", "ARI_mean","scaler"]
        df_best=df_best[columns].round(3).T
        df_best.to_csv(f"{main_data_dir}/GMM_{cov}_silhouette_best.csv")
        df=df.set_index("k")
        df=df.sort_values(by=["k", "Silhouette_mean (cosine)", "Silhouette_mean (euclidean)","ARI_mean"], ascending=[True, False,False,False])
        df.to_csv(f"{main_data_dir}/GMM_{cov}_silhouette.csv")

    #    df_best = df.sort_values(by=["k", "ARI_mean"], ascending=[True, False]).groupby("k", as_index=False).first()
        df.to_csv(f"{main_data_dir}/GMM_{cov}_ari.csv")

genders = ["both genders"]
for gender in genders:
    main_data_dir = "files/"
    if gender:
        main_data_dir += gender

    save_max_silhouette_per_geometry(main_data_dir)
    fig = plot_gmm_k_analysis_one_row(main_data_dir,title_word="Stability")
    plt.show()
