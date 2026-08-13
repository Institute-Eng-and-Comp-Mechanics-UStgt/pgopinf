from __future__ import annotations

import pytest

from pgopinf.specs.evaluation.evaluation import EvaluationSpec
from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.specs.identification.convex_ph_inference import (
    ConvexPHInferenceSpec,
)
from pgopinf.specs.presets.evaluation_presets import (
    evaluation_presets,
)
from pgopinf.specs.presets.experiment_presets import (
    experiment_preset,
)
from pgopinf.specs.presets.study_presets import study_preset
from pgopinf.specs.study import StudySpec


@pytest.mark.parametrize(
    ("name", "expected_system_kind", "expected_metric_count"),
    [
        ("msd_siso_50mass", "msd", 2),
        ("msd_mimo_50mass", "msd", 2),
        ("cd_player", "cd_player", 2),
        ("earth_atmosphere", "earth_atmosphere", 2),
        ("poro", "poro", 17),
        ("poro_exp3", "poro", 17),
    ],
)
def test_experiment_presets_construct_and_round_trip(
    name: str, expected_system_kind: str, expected_metric_count: int
) -> None:
    spec = experiment_preset(name)

    assert isinstance(spec, ExperimentSpec)
    assert spec.system.kind == expected_system_kind
    assert spec.identification.kind == "opinf"
    assert spec.evaluation is not None
    assert len(spec.evaluation.metrics) == expected_metric_count
    assert ExperimentSpec.from_dict(spec.to_dict()) == spec


def test_experiment_preset_forwards_kwargs() -> None:
    spec = experiment_preset("msd_mimo_50mass", reduce_sample_n_train=8)

    assert spec.data.train.reduce_sample_n == 8


def test_poro_exp3_applies_overrides() -> None:
    spec = experiment_preset("poro_exp3")

    assert spec.data.train.time.n_steps == 500000
    assert spec.data.train.reduce_sample_n is None
    assert spec.reduction.full_basis.full_matrices is False


def test_experiment_spec_preset_applies_overrides_and_kind_switches() -> None:
    spec = ExperimentSpec.preset(
        "msd_mimo_50mass",
        overrides={
            "reduction.r": 7,
            "identification.kind": "convex_ph_inference",
            "identification.solver": "clarabel",
            "identification.return_system_type": "lti",
        },
    )

    assert spec.reduction.r == 7
    assert spec.identification == ConvexPHInferenceSpec(
        solver="clarabel",
        return_system_type="lti",
    )


def test_unknown_experiment_preset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown ExperimentSpec preset"):
        experiment_preset("missing")


@pytest.mark.parametrize(
    ("name", "expected_count"),
    [
        ("metrics_over_time_test", 5),
        ("metrics_over_time_train", 5),
        ("system_error_metrics", 5),
        ("system_error_metrics_high_dimensional", 5),
        ("opinf_theory_metrics", 2),
        ("matrix_comparison", 1),
        ("metrics_over_time_train_and_test", 10),
        ("high_dimensional_system_metrics", 13),
    ],
)
def test_evaluation_preset_factory_matches_class_wrapper(
    name: str, expected_count: int
) -> None:
    spec = evaluation_presets(name)

    assert spec == EvaluationSpec.preset(name)
    assert len(spec.metrics) == expected_count
    assert EvaluationSpec.from_dict(spec.to_dict()) == spec


def test_unknown_evaluation_preset_factory_raises() -> None:
    with pytest.raises(ValueError, match="Unknown evaluation preset name"):
        evaluation_presets("missing")


@pytest.mark.parametrize(
    ("name", "expected_variant_count", "expected_evaluation_count"),
    [
        ("opinf_theory_over_r", 1, 1),
        ("petrov_vs_galerkin", 2, 3),
        ("petrov_vs_galerkin_fine_r", 2, 3),
        ("petrov_galerkin_and_ph", 4, 4),
        ("g_pg_hamcvx_ph", 4, 5),
        ("test_convex_ph_and_g_opinf", 1, 3),
    ],
)
def test_study_presets_construct(
    name: str, expected_variant_count: int, expected_evaluation_count: int
) -> None:
    spec = study_preset(name)

    assert isinstance(spec, StudySpec)
    assert len(spec.variants) == expected_variant_count
    assert len(spec.axes) == 1
    assert spec.axes[0].parameter == "reduction.r"
    assert len(spec.axes[0].values) > 0
    assert len(spec.evaluations) == expected_evaluation_count
    assert spec.to_dict()["name"] == spec.name


def test_study_spec_preset_forwards_kwargs() -> None:
    spec = StudySpec.preset(
        "petrov_vs_galerkin",
        r_sweep=[2, 4],
        title_prefix="demo",
    )

    assert spec.axes[0].values == (2, 4)
    assert spec.evaluations[0].title == "demo_hinf_error_petrov_vs_galerkin"


def test_study_preset_fine_r_sets_name() -> None:
    assert study_preset("petrov_vs_galerkin_fine_r").name == "petrov_vs_galerkin_fine_r"


def test_unknown_study_preset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown study preset name"):
        study_preset("missing")
