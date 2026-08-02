import numpy as np
import pytest

from modules.experimental.shape_library import (
    POSITION_KINDS,
    T_YEARS,
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
    t_to_year,
    validate_instances,
    year_to_t,
)
from modules.experimental.shapes import ALL_SHAPES
from modules.experimental.synthetic_data_generator import TimeSeriesSyntheticDataGenerator


def test_year_t_roundtrip_and_endpoints():
    for year in (1901, 1950, 2000):
        assert t_to_year(year_to_t(year)) == pytest.approx(year, abs=1e-9)
    assert year_to_t(1901) == 0.0
    assert year_to_t(2000) == 1.0


def test_position_kind_table_covers_all_shapes():
    assert set(POSITION_KINDS) == set(ALL_SHAPES)


# --- Reference values (docs/prompt_02_library_and_ui.md, section 2) ---

REFERENCE_RHO = [
    (ShapeInstance("peak", 1935, 15), ShapeInstance("peak", 1945, 15), +0.406),
    (ShapeInstance("peak", 1935, 15), ShapeInstance("peak", 1950, 15), +0.031),
    (ShapeInstance("peak", 1935, 15), ShapeInstance("peak", 1955, 15), -0.182),
    (ShapeInstance("peak", 1935, 5), ShapeInstance("peak", 1945, 5), -0.077),
    (ShapeInstance("peak", 1935, 30), ShapeInstance("peak", 1965, 30), -0.361),
    (ShapeInstance("impulse", 1930, 6), ShapeInstance("impulse", 1940, 6), -0.064),
    (ShapeInstance("cylinder", 1930, 50), ShapeInstance("cylinder", 1960, 50), -0.200),
    (ShapeInstance("sigmoid", 1930, 20), ShapeInstance("sigmoid", 1960, 20), +0.668),
    (ShapeInstance("level_shift", 1930), ShapeInstance("level_shift", 1960), +0.533),
    (ShapeInstance("level_shift", 1930), ShapeInstance("level_shift", 1980), +0.330),
    (ShapeInstance("peak", 1935, 15), ShapeInstance("trough", 1935, 15), -1.000),
]


@pytest.mark.parametrize(
    "inst_a, inst_b, expected",
    REFERENCE_RHO,
    ids=[f"{a.key}-vs-{b.key}" for a, b, _ in REFERENCE_RHO],
)
def test_reference_rho(inst_a, inst_b, expected):
    rho = separation_matrix([inst_a, inst_b])[0, 1]
    assert rho == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    "base, width, expected",
    [
        ("peak", 5, 4),
        ("peak", 15, 11),
        ("peak", 30, 17),
        ("impulse", 6, 4),
        ("cylinder", 50, 16),
        ("sigmoid", 20, 55),
        ("level_shift", None, 43),
    ],
)
def test_suggested_min_gap_reference(base, width, expected):
    assert suggested_min_gap(base, width) == expected


def test_suggested_min_gap_none_for_global_shapes():
    assert suggested_min_gap("linear_up") is None


def test_suggested_min_gap_computed_for_phase_shapes():
    gap = suggested_min_gap("sine_1")
    assert gap is not None and 1 <= gap < T_YEARS


# --- Validation ---

def test_validation_rejects_out_of_range_years():
    for year in (1900, 2001):
        with pytest.raises(ValueError, match="position"):
            validate_instances([ShapeInstance("peak", year)])


def test_validation_rejects_duplicate_keys():
    # width None resolves to the default 15, so these are the same class
    with pytest.raises(ValueError, match="duplicate"):
        validate_instances(
            [ShapeInstance("peak", 1925, 15), ShapeInstance("peak", 1925)])


def test_validation_rejects_position_on_global_shape():
    with pytest.raises(ValueError, match="linear_up"):
        validate_instances([ShapeInstance("linear_up", 1950)])


def test_validation_rejects_missing_position():
    with pytest.raises(ValueError, match="peak"):
        validate_instances([ShapeInstance("peak", None)])


def test_validation_rejects_bad_width():
    with pytest.raises(ValueError, match="width"):
        validate_instances([ShapeInstance("peak", 1950, 1)])
    with pytest.raises(ValueError, match="width"):
        validate_instances([ShapeInstance("level_shift", 1950, 10)])


def test_validation_rejects_unknown_base():
    with pytest.raises(ValueError, match="unknown"):
        validate_instances([ShapeInstance("mystery", 1950)])


# --- Instance model ---

def test_key_and_label():
    assert ShapeInstance("peak", 1925).key == "peak@1925w15"
    assert ShapeInstance("peak", 1925).label == "Tepe @1925"
    assert ShapeInstance("level_shift", 1930).key == "level_shift@1930"
    assert ShapeInstance("linear_up", None).key == "linear_up"


def test_instance_prototypes_shape_and_normalization():
    instances = [
        ShapeInstance("peak", 1925),
        ShapeInstance("impulse", 1960),
        ShapeInstance("damped_sine", 1930),
        ShapeInstance("saturating", None),
    ]
    protos = instance_prototypes(instances)
    assert protos.shape == (len(instances), T_YEARS)
    assert np.allclose(protos.mean(axis=-1), 0.0, atol=1e-10)
    assert np.allclose(protos.std(axis=-1), 1.0, atol=1e-10)


def test_clipping_truncates_instead_of_wrapping():
    proto = instance_prototypes([ShapeInstance("peak", 1905, 15)])[0]
    assert YEARS[np.argmax(proto)] == 1905
    # The right tail sits at the baseline, i.e. near the series minimum;
    # wrapping would lift the final years back up the peak's far shoulder.
    tail = proto[-5:]
    assert np.all(tail <= proto.min() + 0.05 * (proto.max() - proto.min()))
    assert proto[0] > proto[-1] + 1.0  # left shoulder is on the truncated peak


# --- Separation warnings ---

def test_flag_pairs_flags_close_level_shifts():
    pair = [ShapeInstance("level_shift", 1940), ShapeInstance("level_shift", 1950)]
    flagged = flag_pairs(pair, threshold=0.7)
    assert flagged and flagged[0][2] > 0.7
    assert {flagged[0][0], flagged[0][1]} == {p.key for p in pair}


def test_flag_pairs_empty_when_separated():
    pair = [ShapeInstance("peak", 1920, 15), ShapeInstance("trough", 1980, 15)]
    assert flag_pairs(pair) == []


# --- JSON round-trip ---

def test_json_roundtrip_reproduces_prototypes():
    instances = [
        ShapeInstance("peak", 1925, 15),
        ShapeInstance("level_shift", 1940),
        ShapeInstance("sine_1", 1920),
        ShapeInstance("linear_up", None),
    ]
    config = {"sigma": 0.3, "n_per_cluster": 20, "seed": 7}
    text = instances_to_json(instances, config)
    restored, restored_config = instances_from_json(text)
    assert restored == instances
    assert restored_config == config
    assert np.array_equal(instance_prototypes(restored),
                          instance_prototypes(instances))


def test_json_import_validates():
    bad = '{"version": 1, "instances": [{"base": "peak", "position": 1875}]}'
    with pytest.raises(ValueError, match="position"):
        instances_from_json(bad)


# --- TimeSeriesSyntheticDataGenerator (docs/prompt_02, section 3) ---

def _generator_kwargs(**overrides):
    kwargs = {
        "instances": [ShapeInstance("peak", 1925), ShapeInstance("peak", 1960),
                      ShapeInstance("level_shift", 1940)],
        "n_per_cluster": 5,
        "sigma": 0.3,
        "znorm": True,
        "amplitude_jitter": False,
        "amp_range": (0.5, 2.0),
        "seed": 7,
    }
    kwargs.update(overrides)
    return kwargs


def test_generator_columns_labels_and_contract():
    generator = TimeSeriesSyntheticDataGenerator(_generator_kwargs())
    df, labels = generator.generate()
    assert np.array_equal(df.columns.to_numpy(), YEARS)
    assert df.shape == (3 * 5, T_YEARS)
    assert generator.ground_truth_labels is labels
    assert generator.n_features == T_YEARS
    assert np.array_equal(np.unique(labels), [0, 1, 2])


def test_generator_identical_frame_for_one_seed():
    df_1, labels_1 = TimeSeriesSyntheticDataGenerator(_generator_kwargs()).generate()
    df_2, labels_2 = TimeSeriesSyntheticDataGenerator(_generator_kwargs()).generate()
    assert df_1.equals(df_2)
    assert np.array_equal(labels_1, labels_2)


def test_generator_random_state_overrides_seed_labels_fixed():
    df_a, labels_a = TimeSeriesSyntheticDataGenerator(
        _generator_kwargs(random_state=0)).generate()
    df_b, labels_b = TimeSeriesSyntheticDataGenerator(
        _generator_kwargs(random_state=1)).generate()
    df_c, labels_c = TimeSeriesSyntheticDataGenerator(
        _generator_kwargs(seed=99, random_state=1)).generate()
    assert not df_a.equals(df_b)               # new noise draw per injected seed
    assert np.array_equal(labels_a, labels_b)  # k_true fixed by the instance list
    assert df_b.equals(df_c)                   # random_state wins over seed
    assert np.array_equal(labels_b, labels_c)


def test_generator_ignores_centers():
    df_plain, labels_plain = TimeSeriesSyntheticDataGenerator(
        _generator_kwargs()).generate()
    df_centers, labels_centers = TimeSeriesSyntheticDataGenerator(
        _generator_kwargs(centers=9)).generate()
    assert df_plain.equals(df_centers)
    assert np.array_equal(labels_plain, labels_centers)


def test_generator_absorbs_sweep_style_kwarg_mutation():
    # optimal_k_analysis mutates kwargs in place per (seed, k) and regenerates;
    # for one seed the data must come out identical at every candidate k.
    generator = TimeSeriesSyntheticDataGenerator(_generator_kwargs())
    generator.kwargs["random_state"] = 3
    generator.kwargs["centers"] = 2
    df_k2, labels_k2 = generator.generate()
    generator.kwargs["centers"] = 8
    df_k8, labels_k8 = generator.generate()
    assert df_k2.equals(df_k8)
    assert np.array_equal(labels_k2, labels_k8)
