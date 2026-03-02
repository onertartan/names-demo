from adjustText import adjust_text
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import streamlit as st
import numpy as np
import pandas as pd
class PCAPlotter:
    import numpy as np
    import pandas as pd

    def apply_clr(self,df: pd.DataFrame, pseudocount: float = 1e-10) -> pd.DataFrame:
        """
        Apply centered log-ratio (CLR) transformation row-wise.

        Assumptions (important for your study):
        - Rows = names
        - Columns = provinces
        - Values are non-negative frequencies or proportions
        - Each row represents a compositional profile over provinces

        Parameters
        ----------
        df : pd.DataFrame
            Input data (rows = compositions).
        pseudocount : float
            Small constant added to avoid log(0).

        Returns
        -------
        pd.DataFrame
            CLR-transformed data with same shape and labels.
        """

        # --- safety checks ---
        if (df < 0).any().any():
            raise ValueError("CLR requires non-negative values.")

        # add pseudocount to avoid log(0)
        X = df + pseudocount

        # log-transform
        log_X = np.log(X)

        # row-wise geometric mean (mean of logs)
        row_means = log_X.mean(axis=1)

        # CLR transform: log(x_ij) - mean_j(log(x_i·))
        clr_values = log_X.sub(row_means, axis=0)

        return clr_values

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from adjustText import adjust_text
    import streamlit as st
    from mpl_toolkits.mplot3d import Axes3D

    def plot_pca(
            self,
            df_pivot: pd.DataFrame,
            df_clusters: pd.Series,
            dense_threshold: int,
            mid_threshold: int,
            colors,
            title: str = "",
            raw_label: str = "Share-normalized",
            clr_label: str = "CLR-transformed",
            annotate: bool = True,
            show_loadings: bool = True,
            n_loadings: int = 5,
            point_alpha: float = 0.90,
            same_axis_limits: bool = False,
            n_components: int = 2,  # ← new parameter
    ):
        """
        Flexible PCA visualization: 2D (side-by-side raw vs CLR) or 3D scatter.

        When n_components == 3: shows interactive-ish 3D plot (one panel).
        When n_components == 2: shows classic 1×2 comparison (raw vs CLR).
        """
        # ---- Input validation / fallback ----
        if n_components not in (2, 3):
            st.warning(f"n_components={n_components} not supported → falling back to 2D")
            n_components = 2

        # ---- alignment ----
        if not df_clusters.index.equals(df_pivot.index):
            df_clusters = df_clusters.reindex(df_pivot.index)

        # ---- prepare inputs ----
        X_raw = df_pivot
        X_clr = self.apply_clr(df_pivot) if n_components == 2 else None

        # ---- color mapping (shared) ----
        unique_clusters = sorted(df_clusters.dropna().unique())
        cluster_to_color = {cid: colors[i % len(colors)] for i, cid in enumerate(unique_clusters)}
        point_colors = df_clusters.map(cluster_to_color).fillna("gray")

        # Legend handles (used in both 2D and 3D)
        legend_handles = [
            plt.Line2D([], [], marker="o", linestyle="",
                       color=cluster_to_color[cid], markersize=8,
                       label=f"Cluster {cid}")
            for cid in unique_clusters
        ]

        # ───────────────────────────────────────────────
        #  Case 1:  2 components → classic side-by-side
        # ───────────────────────────────────────────────
        if n_components == 2:

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
            axes = np.atleast_1d(axes)  # make iterable safe

            def _plot_2d_panel(ax, X, panel_title: str):
                pca = PCA(n_components=2)
                scores = pca.fit_transform(X)
                pc1, pc2 = pca.explained_variance_ratio_[:2]

                ax.scatter(
                    scores[:, 0], scores[:, 1],
                    c=point_colors, edgecolors="w", linewidths=0.5,
                    alpha=point_alpha, s=60
                )

                ax.set_xlabel(f"PC1 ({pc1:.2%})")
                ax.set_ylabel(f"PC2 ({pc2:.2%})")
                ax.set_title(f"{panel_title} | PC1+PC2 = {(pc1 + pc2):.2%}", fontsize=11)
                ax.grid(True, alpha=0.3)

                # Annotation logic (sparse)
                if annotate:
                    texts = []
                    cluster_counts = df_clusters.value_counts()
                    for i, name in enumerate(df_pivot.index):
                        cid = df_clusters.loc[name]
                        size = cluster_counts.get(cid, 0)
                        if size >= dense_threshold:
                            continue
                        if size >= mid_threshold and i % 5 != 0:
                            continue
                        texts.append(
                            ax.annotate(
                                name, (scores[i, 0], scores[i, 1]),
                                fontsize=8, alpha=0.7
                            )
                        )
                    if texts:
                        adjust_text(texts, ax=ax)

                # Top loadings (provinces)
                if show_loadings:
                    load = pd.DataFrame(
                        pca.components_.T,
                        columns=["PC1", "PC2"],
                        index=X.columns
                    )
                    for pc in ["PC1", "PC2"]:
                        top = load[pc].abs().nlargest(n_loadings)
                        print(f"\n[{panel_title}] Top |{pc}| loadings:")
                        for feat, val in top.items():
                            print(f"  {feat:20} {val:.3f}")

                return scores, pca

            # ── Plot both panels ──
            scores_raw, pca_raw = _plot_2d_panel(axes[0], X_raw, raw_label)
            scores_clr, pca_clr = _plot_2d_panel(axes[1], X_clr, clr_label)

            # Legend on right panel
            axes[1].legend(handles=legend_handles, title="Clusters", loc="best")

            # Optional: force same axis limits
            if same_axis_limits:
                all_x = np.concatenate([scores_raw[:, 0], scores_clr[:, 0]])
                all_y = np.concatenate([scores_raw[:, 1], scores_clr[:, 1]])
                xpad = 0.05 * (all_x.ptp() + 1e-12)
                ypad = 0.05 * (all_y.ptp() + 1e-12)
                xlim = (all_x.min() - xpad, all_x.max() + xpad)
                ylim = (all_y.min() - ypad, all_y.max() + ypad)
                for ax in axes:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

        # ───────────────────────────────────────────────
        #  Case 2:  3 components → single 3D scatter
        # ───────────────────────────────────────────────
        else:  # n_components == 3

            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            pca = PCA(n_components=3)
            scores = pca.fit_transform(X_raw)  # for simplicity — only raw shown in 3D

            pc_var = pca.explained_variance_ratio_
            total_var = pc_var.sum()

            ax.scatter(
                scores[:, 0], scores[:, 1], scores[:, 2],
                c=point_colors, edgecolors="w", linewidths=0.6,
                alpha=point_alpha, s=70
            )

            ax.set_xlabel(f"PC1 ({pc_var[0]:.2%})")
            ax.set_ylabel(f"PC2 ({pc_var[1]:.2%})")
            ax.set_zlabel(f"PC3 ({pc_var[2]:.2%})")

            ax.set_title(f"3D PCA – {title or 'Provinces'}\n"
                         f"Explained variance: {total_var:.2%}", fontsize=13)

            ax.view_init(elev=20, azim=75)  # nice starting angle — user can rotate in notebook

            # Sparse annotation in 3D (a bit more aggressive thinning)
            if annotate:
                texts = []
                cluster_counts = df_clusters.value_counts()
                for i, name in enumerate(df_pivot.index):
                    cid = df_clusters.loc[name]
                    size = cluster_counts.get(cid, 0)
                    if size >= dense_threshold * 1.3:  # more conservative in 3D
                        continue
                    if size >= mid_threshold and i % 4 != 0:
                        continue
                    texts.append(
                        ax.text(
                            scores[i, 0], scores[i, 1], scores[i, 2],
                            name, fontsize=8, alpha=0.65
                        )
                    )
                # adjust_text works poorly in 3D → optional or use smaller fontsize

            # Legend
            ax.legend(handles=legend_handles, title="Clusters",
                      loc="upper right", bbox_to_anchor=(1.15, 1))

            ax.grid(True, alpha=0.25)

            # For 3D we don't show CLR comparison (too crowded)
            pca_raw = pca
            pca_clr = None

        # ───────────────────────────────────────────────
        #  Common parts: super-title + explained variance plot
        # ───────────────────────────────────────────────
        if title:
            fig.suptitle(title, fontsize=14, y=1.05  )

        # Scree / cumulative variance plot (always shown)
        pca_full = PCA().fit(df_pivot)
        per_var = np.round(pca_full.explained_variance_ratio_ * 100, 1)
        cum_var = per_var.cumsum()

        fig_scree, ax_scree = plt.subplots(figsize=(8, 4))
        ax_scree.plot(range(1, len(per_var) + 1), cum_var, marker="o", linestyle="--", color="teal")
        #ax_scree.axhline(y=90, color="gray", linestyle=":", alpha=0.7, label="90% threshold")
        ax_scree.set_xlabel("Number of Components")
        ax_scree.set_ylabel("Cumulative Explained Variance (%)")
        ax_scree.set_title("Explained Variance by Number of Components")
        ax_scree.grid(True, alpha=0.3)
        ax_scree.legend()
        fig_scree.savefig(f"temp/pca2.png", dpi=300, bbox_inches="tight")
        st.pyplot(fig_scree)
        fig.savefig(f"temp/pca1.png", dpi=300, bbox_inches="tight")

        # Show final figure
        st.pyplot(fig)

        return fig, (pca_raw, pca_clr)
