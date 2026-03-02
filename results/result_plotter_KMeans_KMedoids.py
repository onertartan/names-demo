import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.lines import Line2D
def load_results(gender,engine_name,pattern_map):
    DATA_DIR = Path(f"files/{gender}/{engine_name}")
    # ---- load data ----
    dfs = {}
    for scaler, regex in pattern_map.items():
        matches = [f for f in DATA_DIR.glob("*.csv") if regex.fullmatch(f.name)]

        if not matches:
             raise FileNotFoundError(f"No file found for {scaler} ({engine_name})")
        df = pd.read_csv(matches[0])
        df["Number of clusters"] = df["Number of clusters"].astype(int)
        df.rename(columns={"Number of clusters":"k"}, inplace=True)
        df["scaler"]=scaler
        dfs[scaler] = df
    return dfs
def plot_k_analysis(genders,engines, METRICS, pattern_map, scaler_labels,title_word,fill_std=False):
    n_rows = len(genders)
    n_cols = len(engines)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, constrained_layout=True)
    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    MAX_K_TO_PLOT = 10
    for row_idx, gender in enumerate(genders):
        for col_idx, (engine_name, engine_label) in enumerate(engines):
            dfs = load_results(gender,engine_name,pattern_map)
            # ---- plot ----
            scaler_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
            for metric, ylabel, title, line_style in METRICS:
                ax = axes[0, col_idx]
                for i, (scaler, df) in enumerate(dfs.items()):
                    jitter = 0.005 if (engine_name == "KMedoidsEngine" and scaler != "TF-IDF") else 0.0
                    df=df[df["k"] <= MAX_K_TO_PLOT]
                    ax.plot(
                        df["k"],
                        df[metric] + jitter * (i - 1),
                        marker="o",
                        linewidth=2,
                        linestyle=line_style,
                        color=scaler_colors[i],  # 👈 simple + effective
                        label=scaler_labels[scaler],
                        alpha=0.8,
                    )
                    if fill_std:
                        ax.fill_between(
                            df["Number of clusters"],
                            df[metric] - df[metric.replace("_mean", "_std")],
                            df[metric] + df[metric.replace("_mean", "_std")],
                            alpha=0.15
                        )
                 # Titles only on first row
                ylabel = "Mean ARI" if metric=="ARI_mean" else metric
                ax.set_ylabel(ylabel, fontsize=12)
                # Engine label only on first column
                if gender =="both genders":
                    dataset_name = "combined dataset"
                else:
                    dataset_name = gender + " dataset"

                ax.set_title(f"{engine_label}\n{ylabel} for {dataset_name}")
                ax.set_xlabel("Number of clusters (k)")
                ax.grid(alpha=0.3)
                ax.set_ylim(0, 1.02)
    fig.legend(scaler_labels, title="Normalization",
                   loc="center right" if metric == "ARI_mean" else "upper right",
                   bbox_to_anchor=(1, .31) if metric == "ARI_mean" else (.82, .8), frameon=False)
    if len(METRICS) == 2:
        style_handles = [Line2D([0], [0], color="black", lw=2, linestyle="solid", label="Cosine"),
                         Line2D([0], [0], color="black", lw=2, linestyle="dashed", label="Euclidean")]
        loc = "upper left" if i == 0 else "upper right"
        bbox_to_anchor = (.98, .8)
        style_legend = fig.legend(handles=style_handles, title="Distance metric", loc="upper right",
                                  bbox_to_anchor=bbox_to_anchor, frameon=False)

    # ---- shared legend ----
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Sensitivity of K-Means and K-Medoids {title_word.capitalize()} to Cluster Resolution and Data Normalization", fontsize=14)
    return fig

def save_max_silhouette (gender,engine, pattern_map):
    dfs = load_results(gender,engine[0], pattern_map)
    df = pd.concat(dfs, axis=0, ignore_index=True)

    # choose the correct silhouette column
    columns = ["Silhouette_mean (cosine)", "Silhouette_mean (euclidean)"]
    df = df[["k", "Silhouette_mean (cosine)", "Silhouette_mean (euclidean)", "ARI_mean", "scaler"]].round(3)
    # ---- max silhouette per k ----
    if engine[0]=="KMeansEngine":
        sort_by_cols = ["k", "Silhouette_mean (euclidean)", "Silhouette_mean (cosine)", "ARI_mean"]
    else:
        sort_by_cols= ["k","Silhouette_mean (cosine)", "Silhouette_mean (euclidean)", "ARI_mean"]
    df_best=df.sort_values(by=sort_by_cols, ascending=[True, False, False, False]).groupby("k", as_index=False).first()
    df_best.set_index("k", inplace=True)
    df=df.sort_values(by=sort_by_cols, ascending=[True, False, False, False])
    df.set_index("k", inplace=True)
    df.round(2).to_csv(f"files/{gender}/{engine[0]}.csv", index=True)

    df_best.round(2).T.to_csv(f"files/{gender}/{engine[0]}_best.csv", index=True)
    return df_best

scaler_labels = {"Share of Top 30 (L1 Norm)": "L1 (Top 30)", "Share of Total": "Share (All)", "TF-IDF": "TF–IDF", "L2 Normalization": "L2"}

pattern_map = {"Share of Top 30 (L1 Norm)": re.compile(r"Share of Top 30 \(L1 Norm\)_(\d{4}_\d{4})\.csv"),
    "Share of Total": re.compile(r"Share of Total_(\d{4}_\d{4})\.csv"),
    "TF-IDF": re.compile(r"TF-IDF_(\d{4}_\d{4})\.csv"),
   "L2 Normalization": re.compile(r"L2 Normalization_(\d{4}_\d{4})\.csv")
                }
engines = [("KMeansEngine", "K-Means"), ("KMedoidsEngine", "K-Medoids")]

METRICS = [  ("ARI_mean", "Clustering stability vs number of clusters", "Stability","solid")]
genders= ["both genders"]


fig = plot_k_analysis(genders,engines, METRICS, pattern_map, scaler_labels,"stability")
fig.savefig(f"files/stability_{engines[0][1]}_{engines[1][1]}.png", dpi=300, bbox_inches="tight")

plt.show()
METRICS = [("Silhouette_mean (cosine)", "Clustering seperability vs number of clusters", "Seperability","solid"),
("Silhouette_mean (euclidean)", "Clustering seperability vs number of clusters", "Seperability","dashed")]

fig = plot_k_analysis(genders,engines, METRICS, pattern_map, scaler_labels,"seperability")
fig.savefig(f"files/seperability-{engines[0][1]}-{engines[1][1]}.png", dpi=300, bbox_inches="tight")

plt.show()
for gender in genders:
    for engine in engines:
        print(save_max_silhouette(gender,engine,pattern_map).head())