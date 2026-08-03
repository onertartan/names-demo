import extra_streamlit_components as stx
import numpy as np
import streamlit as st

from modules.experimental.shape_library import (
    POSITION_KINDS,
    T_YEARS,
    WIDTH_DEFAULTS,
    YEAR_MAX,
    YEAR_MIN,
    YEARS,
    PositionKind,
    ShapeInstance,
    flag_pairs,
    instance_prototypes,
    instances_from_json,
    instances_to_json,
    separation_matrix,
    suggested_min_gap,
)
from modules.experimental.shapes import (
    DISPLAY_NAMES,
    TIERS,
    GenConfig,
    difficulty_from_prototypes,
    make_dataset_from_prototypes,
)
from utils import SessionAdapter
from utils.session_adapter import PageKeys
from viz.plotters.time_series_synthetic_plotter import TimeSeriesSyntheticPlotter

# module level — built once
GENDER_LABEL_TO_LIST = {
    "Male": ["male"],
    "Female": ["female"],
    "Both genders": ["male", "female"],
}

def render_plot_map_sub_tab(names,page_name):
    # Expression depending on page
    expr = "names or surnames" if page_name == "names_surnames" else "baby names"
    col_1, col_2 = st.columns([3,7])
    choice = col_1.radio("Choose how to display results when multiple years are selected:",
                      options=["Show results for the selected years",
                               "Show accumulated results between the selected years"])
    plotter_engine = col_2.radio("Select plotter engine",
                                 options=["Matplotlib", "Folium", "Plotly", "Altair"],
                                 index=0,
                                 # key=f"bump_engine_{page_name}",
                                 )
    accumulate =choice == "Show accumulated results between the selected years"
    options = list(range(1, 31))  # Options are [1-30]
    btn_col1, btn_col2 = st.columns([1, 1])
    rank = 0
    display_option = None
    button_clicked = btn_col1.button(f"Select {expr} and filter if they are in top-n", use_container_width=True)
    top_n = btn_col1.selectbox('Choose a number top-n to filter', options, index=1, key="top_n_filter")
    btn_col1.multiselect(f"Select {expr}", names,key="names_" + page_name)
    if button_clicked:
        rank = top_n
        display_option = "top-n to filter"
    # Second option for Nth Most Common
    button_clicked = btn_col2.button("Nth Most Common", use_container_width=True)
    include_top_n = btn_col2.checkbox("Include top-n to filter")
    if include_top_n:
        options=list(range(1,6))
    n_most_common = btn_col2.selectbox('Choose a number n for the "nth most common"', options)
    if button_clicked:
        rank = n_most_common
        display_option = "nth most common"
    return rank, display_option, include_top_n,accumulate,plotter_engine



def _render_name_surname_selection(page_name, col):
    """Render the name/surname radio (surname pages only) and return the selection."""
    # data is a dictionary whose keys are names and surnames, values are corresponding dataframes
    if "surnames" not in page_name:
        return "name"
    selection = col.radio("Select name or surname", ["Name", "Surname"],
                          key="name_surname_selection").lower()
    st.session_state["name_surname_rb"] = selection
    return selection


def _initial_gender_label(gender_list_key):
    """Sync the widget's first value to any previously stored gender list."""
    current_list = st.session_state.get(gender_list_key)
    if current_list == ["male", "female"]:
        return "Both genders"
    if current_list == ["female"]:
        return "Female"
    return "Male"


def _render_gender_selection(page_name, col, disabled):
    """Render the gender radio and return the selected list of gender values."""
    gender_list_key = "gender_list_" + page_name
    widget_key = "gender_radio_widget_" + page_name  # used for sub-folder name in saving clustering results

    # One-time initialization: if the widget hasn't been created yet, set its
    # initial value based on the existing list data (if any).
    if widget_key not in st.session_state:
        st.session_state[widget_key] = _initial_gender_label(gender_list_key)

    label = col.radio("Select Gender", list(GENDER_LABEL_TO_LIST), key=widget_key,
                      label_visibility="collapsed", disabled=disabled)
    # Surnames have no gender dimension → force both genders.
    gender_list = ["male", "female"] if disabled else GENDER_LABEL_TO_LIST[label]
    st.session_state[gender_list_key] = gender_list
    return gender_list


def render_gender_name_surname_filters(page_name, cols):
    """
    Configure name/surname selection, temporal filtering, and gender selection state.

    This helper centralizes UI rendering and session state management for
    name-based analyses across both "Names & Surnames" and "Baby Names" pages.
    It ensures consistent handling of:
      (1) name vs. surname selection,
      (2) year range filtering,
      (3) gender selection with persistent session state.

    Returns the selected gender values as a list, not a session-state key.
    """
    name_surname_selection = _render_name_surname_selection(page_name, cols[1])
    selected_years = range(st.session_state["year_1"], st.session_state["year_2"] + 1)
    gender_list = _render_gender_selection(
        page_name, cols[0], disabled=(name_surname_selection == "surname"))
    return name_surname_selection, selected_years, gender_list



def render_synthetic_data():
    st.subheader("Synthetic Data")
    col1, col2 = st.columns([1, 1])

    n_samples = col1.number_input("n_samples", min_value=1, value=100, step=1)
    n_features = col1.number_input("n_features", min_value=1, value=2, step=1)

    centers = col1.number_input("centers", min_value=1, value=3, step=1)
    random_state = col1.number_input("random_state",value=70)
    return {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "centers": centers,
        "random_state": random_state,
    }


# ---------------------------------------------------------------------------
# Time Series synthetic data (Experiment page)
# ---------------------------------------------------------------------------

TIER_DISPLAY = {
    "monotone": "Monoton",
    "single_turn": "Tek dönümlü",
    "piecewise": "Parçalı",
    "oscillatory": "Salınımlı",
    "composite": "Bileşik",
}

DEFAULT_TS_CLASSES = [
    ShapeInstance("peak", 1925, 15),
    ShapeInstance("trough", 1955, 15),
    ShapeInstance("level_shift", 1970),
]


@st.cache_data
def _cached_min_gap(base: str, width):
    return suggested_min_gap(base, width)


@st.cache_data
def _cached_prototypes(spec: tuple) -> np.ndarray:
    # Cache key is a tuple of (base, position, width) primitives; a list of
    # frozen dataclasses would be a poor / unstable cache key.
    return instance_prototypes([ShapeInstance(*item) for item in spec])


def _apply_pending_import(session, keys, page_name):
    """Apply an uploaded class list before any widgets are instantiated."""
    pending = st.session_state.pop(keys.ts_pending_import, None)
    if pending is None:
        return
    instances, config = pending
    session.set(keys.ts_class_list, list(instances))
    widget_map = {
        "sigma": (f"ts_sigma_{page_name}", float),
        "n_per_cluster": (f"ts_n_per_cluster_{page_name}", int),
        "amplitude_jitter": (f"ts_amp_jitter_{page_name}", bool),
        "seed": (f"ts_seed_{page_name}", int),
    }
    for field, (widget_key, cast) in widget_map.items():
        if field in config:
            st.session_state[widget_key] = cast(config[field])
    st.success(f"{len(instances)} sınıf JSON dosyasından geri yüklendi.")


def _render_class_builder(page_name, session, keys, instances):
    col_shape, col_pos, col_width, col_add, col_quick = st.columns([2, 1, 1, 1, 2])
    tier = col_shape.selectbox("Şekil grubu", list(TIERS),
                               format_func=lambda t: TIER_DISPLAY.get(t, t),
                               key=f"ts_tier_{page_name}")
    base = col_shape.selectbox("Temel şekil", TIERS[tier],
                               format_func=lambda s: DISPLAY_NAMES.get(s, s),
                               key=f"ts_base_{page_name}")
    kind = POSITION_KINDS[base]

    position, width = None, None
    if kind is not PositionKind.NONE:
        position_label = {PositionKind.CENTER: "Merkez yılı",
                          PositionKind.ONSET: "Başlangıç yılı",
                          PositionKind.PHASE: "Faz kayması (yıl)"}[kind]
        position = int(col_pos.number_input(position_label, min_value=YEAR_MIN,
                                            max_value=YEAR_MAX, value=1950,
                                            key=f"ts_position_{page_name}"))
        if base in WIDTH_DEFAULTS:
            # Width is keyed per base shape so each shape remembers its own
            # width and starts from its own default.
            width = int(col_width.number_input("Genişlik (yıl)", min_value=2,
                                               max_value=100,
                                               value=WIDTH_DEFAULTS[base],
                                               key=f"ts_width_{base}_{page_name}"))

    if col_add.button("Sınıf ekle", key=f"ts_add_{page_name}", use_container_width=True):
        candidate = ShapeInstance(base, position, width)
        if any(existing.key == candidate.key for existing in instances):
            # The position/width widgets deliberately keep their values after
            # an add (convenient for adding a series of peaks); a double click
            # must therefore warn instead of raising on the duplicate key.
            st.warning(f"{candidate.label} zaten listede — tekrar eklenmedi.")
        else:
            instances.append(candidate)
            session.set(keys.ts_class_list, instances)

    quick_count = int(col_quick.number_input("Hızlı ekle: sınıf sayısı", min_value=1,
                                             max_value=10, value=3,
                                             key=f"ts_quick_n_{page_name}"))
    if col_quick.button("Hızlı ekle", key=f"ts_quick_add_{page_name}",
                        use_container_width=True):
        _quick_add(instances, base, width, quick_count, session, keys)


def _quick_add(instances, base, width, count, session, keys):
    """Insert `count` instances of `base` spread evenly across the window."""
    display = DISPLAY_NAMES.get(base, base)
    if POSITION_KINDS[base] is PositionKind.NONE:
        st.error(f"{display} global bir şekildir; konumlandırılamaz ve tekrarlanamaz.")
        return
    min_gap = _cached_min_gap(base, width)
    if min_gap is None:
        st.error(f"{display} için pencere içinde yeterli ayrışmayı (ρ < 0.4) "
                 f"sağlayan bir yıl aralığı yok — hızlı ekleme yapılmadı.")
        return
    span = YEAR_MAX - YEAR_MIN
    if count == 1:
        positions = [(YEAR_MIN + YEAR_MAX) // 2]
    elif (count - 1) * min_gap > span:
        max_count = span // min_gap + 1
        st.error(f"{count} sınıf için sınıflar arası en az {min_gap} yıl gerekir; "
                 f"{YEAR_MIN}-{YEAR_MAX} penceresine en fazla {max_count} adet "
                 f"{display} sığar — hiçbir sınıf eklenmedi.")
        return
    else:
        positions = [round(YEAR_MIN + i * span / (count - 1)) for i in range(count)]

    candidates = [ShapeInstance(base, int(pos), width) for pos in positions]
    existing_keys = {inst.key for inst in instances}
    clashes = [c.label for c in candidates if c.key in existing_keys]
    if clashes:
        st.error(f"Şu sınıflar zaten listede: {', '.join(clashes)} — hiçbir sınıf eklenmedi.")
        return
    instances.extend(candidates)
    session.set(keys.ts_class_list, instances)


def _render_class_list(page_name, session, keys, instances):
    st.markdown(f"**Sınıflar (k = {len(instances)})**")
    for inst in list(instances):
        col_label, col_del = st.columns([6, 1])
        col_label.markdown(f"- {inst.label} (`{inst.key}`)")
        # Keyed off ShapeInstance.key: index-based keys delete the wrong row
        # once the list shifts under Streamlit's rerun model.
        if col_del.button("Sil", key=f"ts_del_{page_name}_{inst.key}"):
            instances.remove(inst)
            session.set(keys.ts_class_list, instances)
            st.rerun()


def _render_export_import(page_name, session, keys, instances, config):
    col_download, col_upload = st.columns([1, 2])
    if instances:
        col_download.download_button(
            "Sınıf tanımını indir (JSON)",
            data=instances_to_json(instances, config),
            file_name="zaman_serisi_siniflari.json",
            mime="application/json",
            key=f"ts_download_{page_name}",
        )
    uploaded = col_upload.file_uploader("Sınıf tanımı yükle (JSON)", type=["json"],
                                        key=f"ts_upload_{page_name}")
    if uploaded is None:
        return
    # file_uploader returns the same object on every rerun; import exactly once
    # per uploaded file or the import stomps everything done since.
    file_marker = getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"
    if session.get(keys.ts_uploaded_file_id) == file_marker:
        return
    session.set(keys.ts_uploaded_file_id, file_marker)
    try:
        imported = instances_from_json(uploaded.getvalue().decode("utf-8"))
    except (ValueError, KeyError) as err:
        st.error(f"JSON içe aktarılamadı: {err}")
        return
    # Settings widgets are already instantiated this run, so their values can
    # only be restored on the next run: stash and rerun.
    st.session_state[keys.ts_pending_import] = imported
    st.rerun()


def _render_generation_settings(page_name):
    defaults = {
        f"ts_sigma_{page_name}": 0.3,
        f"ts_n_per_cluster_{page_name}": 20,
        f"ts_amp_jitter_{page_name}": False,
        f"ts_seed_{page_name}": 0,
    }
    for widget_key, default in defaults.items():
        if widget_key not in st.session_state:
            st.session_state[widget_key] = default

    col_sigma, col_n, col_jitter, col_seed = st.columns([2, 1, 1, 1])
    sigma = col_sigma.slider("Gürültü düzeyi (sigma)", min_value=0.05, max_value=1.0,
                             step=0.05, key=f"ts_sigma_{page_name}")
    n_per_cluster = int(col_n.number_input("Sınıf başına seri", min_value=5,
                                           max_value=100,
                                           key=f"ts_n_per_cluster_{page_name}"))
    amplitude_jitter = col_jitter.toggle("Genlik değişimi",
                                         key=f"ts_amp_jitter_{page_name}")
    col_jitter.caption("Açıkken kümeler küresel değil, uzamış (ışınsal) olur.")
    seed = int(col_seed.number_input("Tohum (seed)", step=1,
                                     key=f"ts_seed_{page_name}"))
    return float(sigma), n_per_cluster, bool(amplitude_jitter), seed


def _render_diagnostics_and_preview(instances, sigma, n_per_cluster, seed):
    spec = tuple((inst.base, inst.position, inst.width) for inst in instances)
    protos = _cached_prototypes(spec)
    labels = [inst.label for inst in instances]
    k = len(instances)

    st.markdown(f"**k = {k} sınıf · toplam {k * n_per_cluster} seri**")
    if k >= 2:
        diff = difficulty_from_prototypes(protos, labels, sigma=sigma)
        col_rho, col_theta, col_ratio, col_verdict = st.columns(4)
        col_rho.metric("ρ_max", f"{diff.rho_max:.2f}")
        col_theta.metric("θ_min (°)", f"{diff.theta_min_deg:.1f}")
        col_ratio.metric("θ_min / yayılım", f"{diff.ratio:.2f}")
        col_verdict.metric("Zorluk", diff.verdict)
        st.caption(f"En yakın çift: {diff.closest_pair[0]} — {diff.closest_pair[1]}")

        label_by_key = {inst.key: inst.label for inst in instances}
        for key_a, key_b, rho in flag_pairs(instances):
            st.warning(f"{label_by_key[key_a]} ile {label_by_key[key_b]} çok benzer "
                       f"(ρ={rho:.2f}) — bu iki sınıf ayrılamayabilir.")

    plotter = TimeSeriesSyntheticPlotter()
    samples_x, samples_y = make_dataset_from_prototypes(
        protos, GenConfig(T=T_YEARS, n_per_cluster=5, sigma=sigma),
        np.random.default_rng(seed))
    col_preview, col_heat = st.columns([3, 2])
    with col_preview:
        plotter.plot_prototype_preview(YEARS, protos, labels, samples_x, samples_y)
    if k >= 2:
        with col_heat:
            plotter.plot_separation_heatmap(separation_matrix(instances), labels)


def render_time_series_synthetic_data(page_name):
    """Class builder, diagnostics, preview, and JSON export for the
    Time Series sub-tab. Returns kwargs for TimeSeriesSyntheticDataGenerator,
    or None while the class list is empty."""
    session = SessionAdapter(page_name)
    keys = PageKeys(page_name)

    st.subheader("Zaman Serisi Sentetik Verisi")
    _apply_pending_import(session, keys, page_name)

    if session.get(keys.ts_class_list) is None:
        session.set(keys.ts_class_list, list(DEFAULT_TS_CLASSES))
    instances = list(session.get(keys.ts_class_list))

    _render_class_builder(page_name, session, keys, instances)
    instances = list(session.get(keys.ts_class_list))
    if instances:
        _render_class_list(page_name, session, keys, instances)
        instances = list(session.get(keys.ts_class_list))
    else:
        st.info("Henüz sınıf yok. Yukarıdan bir temel şekil seçip ekleyin veya "
                "bir JSON tanımı yükleyin.")

    if not instances:
        _render_export_import(page_name, session, keys, instances, {})
        return None

    sigma, n_per_cluster, amplitude_jitter, seed = _render_generation_settings(page_name)
    config = {"sigma": sigma, "n_per_cluster": n_per_cluster, "znorm": True,
              "amplitude_jitter": amplitude_jitter, "amp_range": [0.5, 2.0],
              "seed": seed}
    _render_export_import(page_name, session, keys, instances, config)
    _render_diagnostics_and_preview(instances, sigma, n_per_cluster, seed)

    return {
        "instances": instances,
        "n_per_cluster": n_per_cluster,
        "sigma": sigma,
        "znorm": True,
        "amplitude_jitter": amplitude_jitter,
        "amp_range": (0.5, 2.0),
        "seed": seed,
    }
