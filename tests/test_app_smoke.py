"""AppTest smoke tests covering the seven main app paths.

Each test drives a page headless with real data loads, generators,
clustering (plus the optimal-k sweep where marked), and plotters.
``stx.tab_bar`` is a custom JS component that cannot run headless, so it is
patched to return the wanted tab id -- tab *selection* is simulated, while
everything downstream of it is exercised for real.
"""
from pathlib import Path

import extra_streamlit_components as stx
import pytest
from streamlit.testing.v1 import AppTest

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
