"""AppTest smoke tests covering the seven main app paths.

Each test drives a page headless with real data loads, generators,
clustering (plus the optimal-k sweep where marked), and plotters.
``stx.tab_bar`` is a custom JS component that cannot run headless, so it is
patched to return the wanted tab id -- tab *selection* is simulated, while
everything downstream of it is exercised for real.
"""
from pathlib import Path

import extra_streamlit_components as stx
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from streamlit.delta_generator import DeltaGenerator
from streamlit.testing.v1 import AppTest

from modules.experimental.shape_library import (
    T_YEARS,
    YEARS,
    instance_prototypes,
    instances_to_json,
)
from modules.experimental.shapes import GenConfig, make_dataset_from_prototypes
from viz.plotters.time_series_synthetic_plotter import TimeSeriesSyntheticPlotter

REPO_ROOT = Path(__file__).resolve().parent.parent

CASES = {
    "tr_baby_geo": dict(page="modules/population/baby_names.py",
                        tabs={"tab_main_algorithmic", "tab_geo_clustering"}),
    "tr_names_trend": dict(page="modules/population/names_surnames.py",
                           tabs={"tab_main_algorithmic", "tab_name_trend_clustering"},
                           year_range=True),
    "usa_states": dict(page="modules/usa/baby_names_usa_states.py",
                       tabs={"tab_main_algorithmic", "tab_geo_clustering"}),
    "usa_nation": dict(page="modules/usa/baby_names_usa_nation.py",
                       tabs={"tab_main_algorithmic", "tab_name_trend_clustering"},
                       year_range=True),
    "exp_names": dict(page="modules/experimental/experiment.py",
                      tabs={"tab_geo_clustering"}, sweep=True),
    "exp_blobs": dict(page="modules/experimental/experiment.py",
                      tabs={"tab_synthetic_clustering", "blobs"}),
    "exp_ts": dict(page="modules/experimental/experiment.py",
                   tabs={"tab_synthetic_clustering", "time_series"}, sweep=True),
}


def _failures(at):
    return " | ".join(f"{getattr(e, 'type', '?')}: {getattr(e, 'message', '')}"
                      for e in at.exception)


def make_apptest(case, monkeypatch):
    def fake_tab_bar(data, default=None, return_type=str, key=None):
        for item in data:
            if item.id in case["tabs"]:
                return item.id
        return default

    monkeypatch.setattr(stx, "tab_bar", fake_tab_bar)
    at = AppTest.from_file(str(REPO_ROOT / case["page"]), default_timeout=900)
    # Session defaults normally set by main.py before st.navigation dispatch:
    at.session_state["animate"] = False
    at.session_state["geo_scale"] = "province (ibbs3)"
    at.session_state["colormap"] = {"matplotlib": ["tab10"], "folium": ["Reds"],
                                    "Folium-interactive": ["Reds"], "plotly": ["Viridis"]}
    return at


@pytest.mark.smoke
@pytest.mark.parametrize("name", list(CASES), ids=list(CASES))
def test_page_renders_and_clusters(name, monkeypatch):
    case = CASES[name]
    at = make_apptest(case, monkeypatch)
    at.run()
    assert not at.exception, _failures(at)

    if case.get("year_range"):
        # The range slider's default already spans all years; an identical
        # value would not fire the on_change callback that switches the page
        # out of single-year mode, so pick a strictly different range.
        year_slider = next(s for s in at.select_slider if s.key == "slider_year_2")
        year_slider.set_range(year_slider.options[1], year_slider.options[-1])
        at.run()
        assert not at.exception, _failures(at)

    if case.get("sweep"):
        next(c for c in at.checkbox if c.key == "optimal_k_analysis").check()

    next(b for b in at.button if b.label == "K-means").click()
    at.run()
    assert not at.exception, _failures(at)
    assert len(at.get("imgs")) > 0, "no figure rendered after clustering"


def _preview_pixels(instances, sigma, seed):
    """Render the prototype-overlay preview exactly as the app builds it and
    return the raw RGBA framebuffer for pixel comparison."""
    protos = instance_prototypes(instances)
    samples_x, samples_y = make_dataset_from_prototypes(
        protos, GenConfig(T=T_YEARS, n_per_cluster=5, sigma=float(sigma)),
        np.random.default_rng(int(seed)))
    fig = TimeSeriesSyntheticPlotter().build_prototype_preview_figure(
        YEARS, protos, [inst.label for inst in instances], samples_x, samples_y)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    pixels = bytes(canvas.buffer_rgba())
    plt.close(fig)
    return pixels


@pytest.mark.smoke
def test_ts_json_roundtrip_in_app(monkeypatch):
    """Build a three-class list through the UI, capture the export payload,
    upload it into a fresh session, and require the restored prototype
    overlay to be pixel-identical. st.file_uploader cannot be driven by
    AppTest, so it is patched like stx.tab_bar; the widget clicks, export
    payload, import-once handling, and figures are all real."""
    case = CASES["exp_ts"]
    at = make_apptest(case, monkeypatch)
    at.run()
    assert not at.exception, _failures(at)

    # Clear the three default classes through their per-row delete buttons.
    while True:
        delete_buttons = [b for b in at.button
                          if (b.key or "").startswith("ts_del_Experiment_")]
        if not delete_buttons:
            break
        delete_buttons[0].click()
        at.run()
        assert not at.exception, _failures(at)
    assert list(at.session_state["ts_class_list_Experiment"]) == []

    def pick(kind, key):
        return next(w for w in getattr(at, kind) if w.key == key)

    def add_class(tier, base, position=None, width=None):
        pick("selectbox", "ts_tier_Experiment").select(tier)
        at.run()
        pick("selectbox", "ts_base_Experiment").select(base)
        at.run()
        if position is not None:
            pick("number_input", "ts_position_Experiment").set_value(position)
        if width is not None:
            pick("number_input", f"ts_width_{base}_Experiment").set_value(width)
        next(b for b in at.button if b.key == "ts_add_Experiment").click()
        at.run()
        assert not at.exception, _failures(at)

    add_class("single_turn", "peak", position=1930, width=12)
    add_class("oscillatory", "sine_1", position=1920)
    add_class("monotone", "saturating")

    built = list(at.session_state["ts_class_list_Experiment"])
    assert [inst.key for inst in built] == ["peak@1930w12", "sine_1@1920", "saturating"]

    sigma = at.session_state["ts_sigma_Experiment"]
    seed = at.session_state["ts_seed_Experiment"]
    config = {"sigma": float(sigma),
              "n_per_cluster": int(at.session_state["ts_n_per_cluster_Experiment"]),
              "znorm": True,
              "amplitude_jitter": bool(at.session_state["ts_amp_jitter_Experiment"]),
              "amp_range": [0.5, 2.0], "seed": int(seed)}
    # Exactly the string the download button carries (same pure function,
    # same inputs).
    payload = instances_to_json(built, config)

    class FakeUpload:
        file_id = "roundtrip-upload-1"
        name = "zaman_serisi_siniflari.json"
        size = len(payload)

        @staticmethod
        def getvalue():
            return payload.encode("utf-8")

    monkeypatch.setattr(DeltaGenerator, "file_uploader",
                        lambda self, *args, **kwargs: FakeUpload())
    at2 = make_apptest(case, monkeypatch)
    at2.run()
    assert not at2.exception, _failures(at2)

    restored = list(at2.session_state["ts_class_list_Experiment"])
    assert restored == built

    # The same file object comes back on every rerun; the import must run
    # once only, or this delete would be stomped by a re-import.
    delete_buttons = [b for b in at2.button
                      if (b.key or "").startswith("ts_del_Experiment_")]
    delete_buttons[0].click()
    at2.run()
    assert not at2.exception, _failures(at2)
    assert len(at2.session_state["ts_class_list_Experiment"]) == len(built) - 1

    pixels_before = _preview_pixels(built, sigma, seed)
    pixels_after = _preview_pixels(restored,
                                   at2.session_state["ts_sigma_Experiment"],
                                   at2.session_state["ts_seed_Experiment"])
    assert pixels_before == pixels_after
