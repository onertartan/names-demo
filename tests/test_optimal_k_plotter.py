import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from clustering.evaluation.cvi_registry import CVI, CVI_REGISTRY
from clustering.models.gmm import GMMEngine
from clustering.models.kmeans import KMeansEngine
from viz.plotters.optimal_k_plotter import MAX_PANEL_COLUMNS, OptimalKPlotter

K_VALUES = range(2, 5)
N_SEEDS = 2
ROWS = N_SEEDS + 1  # per-seed rows plus the mean row


def _fake_sweep_output(keys):
    metrics_all = {
        key: [[0.5 + 0.01 * k for k in K_VALUES] for _ in range(N_SEEDS)]
        for key in keys
    }
    metrics_mean = {key: np.nanmean(metrics_all[key], axis=0) for key in metrics_all}
    return metrics_all, metrics_mean


def _build(engine_class, metrics_all, metrics_mean):
    return OptimalKPlotter.build_optimal_k_figures(
        engine_class, N_SEEDS, K_VALUES, range(N_SEEDS),
        metrics_all, metrics_mean,
        ari_mean=[0.5] * len(K_VALUES), ari_std=[0.1] * len(K_VALUES))


def _panel_count(figures):
    return sum(len(fig.axes) // ROWS for fig in figures)


def _close(figures):
    for fig in figures:
        plt.close(fig)


def test_panel_count_follows_registry(monkeypatch):
    metrics_all, metrics_mean = _fake_sweep_output(list(CVI_REGISTRY) + ["Inertia"])
    figures = _build(KMeansEngine, metrics_all, metrics_mean)
    base_count = _panel_count(figures)
    # registry entries + the engine's Inertia extra + the ARI panel
    assert base_count == len(CVI_REGISTRY) + 1 + 1
    assert all(len(fig.axes) // ROWS <= MAX_PANEL_COLUMNS for fig in figures)
    _close(figures)

    monkeypatch.setitem(
        CVI_REGISTRY, "Dummy Index",
        CVI("Dummy Index", "Dummy", lambda X, labels: 0.0, True,
            "Dummy_mean", "Dummy_std"))
    metrics_all2, metrics_mean2 = _fake_sweep_output(list(CVI_REGISTRY) + ["Inertia"])
    figures2 = _build(KMeansEngine, metrics_all2, metrics_mean2)
    assert _panel_count(figures2) == base_count + 1
    _close(figures2)


def test_gmm_extras_appended_without_entering_registry():
    keys = list(CVI_REGISTRY) + ["AIC", "BIC", "NegLogLikelihood"]
    metrics_all, metrics_mean = _fake_sweep_output(keys)
    figures = _build(GMMEngine, metrics_all, metrics_mean)
    assert _panel_count(figures) == len(CVI_REGISTRY) + 3 + 1
    assert "AIC" not in CVI_REGISTRY
    _close(figures)


def test_all_nan_metric_gets_labelled_panel(monkeypatch):
    monkeypatch.setitem(
        CVI_REGISTRY, "Dead Index",
        CVI("Dead Index", "Dead", lambda X, labels: float("nan"), True,
            "Dead_mean", "Dead_std"))
    metrics_all, metrics_mean = _fake_sweep_output(list(CVI_REGISTRY) + ["Inertia"])
    metrics_all["Dead Index"] = [[float("nan")] * len(K_VALUES) for _ in range(N_SEEDS)]
    metrics_mean["Dead Index"] = np.full(len(K_VALUES), np.nan)
    figures = _build(KMeansEngine, metrics_all, metrics_mean)
    texts = [text.get_text() for fig in figures for ax in fig.axes for text in ax.texts]
    assert any("unavailable" in text for text in texts)
    _close(figures)


def test_selected_k_reads_polarity():
    assert OptimalKPlotter._selected_k_index([0.2, 0.9, 0.4], True) == 1
    assert OptimalKPlotter._selected_k_index([3.0, 1.0, 2.0], False) == 1
    assert OptimalKPlotter._selected_k_index([np.nan, 0.7, np.nan], True) == 1
    assert OptimalKPlotter._selected_k_index([np.nan, np.nan], True) is None


def test_styled_summary_shows_every_registry_entry():
    columns = {"Number of clusters": list(K_VALUES)}
    for cvi in CVI_REGISTRY.values():
        columns[cvi.mean_column] = [0.5, 0.6, 0.7]
        columns[cvi.std_column] = [0.1, 0.1, 0.1]
    columns["ARI_mean"] = [0.5, 0.6, 0.7]
    columns["ARI_std"] = [0.1, 0.1, 0.1]
    df_summary = pd.DataFrame(columns).set_index("Number of clusters")

    styler = OptimalKPlotter.style_metrics_dataframe(df_summary, using_same_data=True)
    html = styler.to_html()  # forces evaluation of the highlight functions
    for cvi in CVI_REGISTRY.values():
        assert cvi.label in styler.data.columns
    assert "ARI" in styler.data.columns
    assert "d4f7d4" in html  # at least one best-k highlight rendered
