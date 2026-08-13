from __future__ import annotations

import importlib

import pytest

from pgopinf.specs.evaluation_study.array_plot import (
    ArrayPlotStudySpec,
)
from pgopinf.specs.evaluation_study.base import (
    STUDY_EVALUATION_REGISTRY,
    study_evaluation_default_by_kind,
    study_evaluation_from_dict,
)
from pgopinf.specs.evaluation_study.scalar_plot import ScalarPlotSpec
from pgopinf.specs.evaluation_study.stability_fraction_by_group import (
    StabilityFractionByGroupSpec,
)


STUDY_EVALUATION_MODULES = (
    "pgopinf.specs.evaluation_study.array_plot",
    "pgopinf.specs.evaluation_study.scalar_plot",
    "pgopinf.specs.evaluation_study.stability_fraction_by_group",
)


@pytest.fixture(scope="module", autouse=True)
def register_study_evaluation_modules() -> None:
    for module_name in STUDY_EVALUATION_MODULES:
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("kind", "spec"),
    [
        (
            "array_plot",
            ArrayPlotStudySpec(
                artifact_name="state_values_over_time",
                artifact_filters={"split": "TRAIN"},
                subplot_indices=[0, 2],
                line_labels="identified",
                variants=["pod", "modal"],
                line_by=["variant", "artifact:target"],
                filters={"reduction.r": 10},
                title="State values",
                line_mode_by="variant",
                line_mode_map={"pod": "line+marker"},
                linestyle_by=["variant"],
                linestyle_map={"modal": "--"},
            ),
        ),
        (
            "scalar_plot",
            ScalarPlotSpec(
                metrics="hinf_error",
                x="reduction.r",
                line_by=["variant", "target"],
                panel_by="metric",
                filters={"target": "identified"},
                title="Hinf error",
                x_label="Reduced order",
                y_labels="Relative error",
                line_mode_by="target",
                line_mode_map={"identified": "line+marker"},
            ),
        ),
        (
            "stability_fraction_by_group",
            StabilityFractionByGroupSpec(
                group_by=["variant", "method"],
                filters={"target": "identified"},
                include_stable=False,
                include_unstable=True,
                title="Stability",
            ),
        ),
    ],
)
def test_study_evaluation_spec_round_trip(kind: str, spec) -> None:
    assert spec.kind == kind
    assert spec.to_dict()["kind"] == kind
    assert study_evaluation_from_dict(spec.to_dict()) == spec


def test_known_study_evaluation_kinds_are_registered() -> None:
    assert STUDY_EVALUATION_REGISTRY.known_kinds() == [
        "array_plot",
        "scalar_plot",
        "stability_fraction_by_group",
    ]


@pytest.mark.parametrize(
    ("kind", "expected_spec"),
    [
        ("array_plot", ArrayPlotStudySpec()),
        ("scalar_plot", ScalarPlotSpec()),
        ("stability_fraction_by_group", StabilityFractionByGroupSpec()),
    ],
)
def test_study_evaluation_default_by_kind(kind: str, expected_spec) -> None:
    assert study_evaluation_default_by_kind(kind) == expected_spec


def test_study_evaluation_from_dict_requires_known_kind() -> None:
    with pytest.raises(KeyError, match="missing required keys"):
        study_evaluation_from_dict({})

    with pytest.raises(ValueError, match="unknown kind"):
        study_evaluation_from_dict({"kind": "missing"})


def test_scalar_plot_spec_normalizes_inputs_and_default_name() -> None:
    spec = ScalarPlotSpec(
        metrics="hinf_error",
        line_by="variant",
        panel_by=["metric"],
        y_labels=["error"],
        line_mode_by="target",
        line_mode_map={"identified": "marker", ("intrusive",): "line"},
    )

    assert spec.metrics == ("hinf_error",)
    assert spec.line_by == ("variant",)
    assert spec.panel_by == ("metric",)
    assert spec.y_labels == ("error",)
    assert spec.line_mode_by == ("target",)
    assert spec.line_mode_map == {
        ("identified",): "marker",
        ("intrusive",): "line",
    }
    assert spec.name == "scalar_plot"


def test_scalar_plot_spec_uses_title_as_default_name() -> None:
    assert ScalarPlotSpec(title="My plot").name == "My plot"


def test_array_plot_study_spec_normalizes_inputs_and_default_name() -> None:
    spec = ArrayPlotStudySpec(
        subplot_indices=[1, 3],
        line_labels="output",
        variants="pod",
        line_by="variant",
        line_mode_by="artifact:target",
        line_mode_map={"identified": "line"},
        linestyle_by="variant",
        linestyle_map={"pod": "--", ("modal",): None},
    )

    assert spec.subplot_indices == (1, 3)
    assert spec.line_labels == ("output",)
    assert spec.variants == ("pod",)
    assert spec.line_by == ("variant",)
    assert spec.line_mode_by == ("artifact:target",)
    assert spec.line_mode_map == {("identified",): "line"}
    assert spec.linestyle_by == ("variant",)
    assert spec.linestyle_map == {("pod",): "--", ("modal",): None}
    assert spec.name == "array_plot"


def test_array_plot_study_spec_uses_title_as_default_name() -> None:
    assert ArrayPlotStudySpec(title="Array plot").name == "Array plot"


def test_stability_fraction_spec_normalizes_inputs_and_sets_metric() -> None:
    spec = StabilityFractionByGroupSpec(group_by="variant", title="Stability")

    assert spec.group_by == ("variant",)
    assert spec.name == "Stability"
    assert spec.metric == "spectral_abscissa_cmp"


@pytest.mark.parametrize(
    ("spec", "runtime_type_name"),
    [
        (ArrayPlotStudySpec(), "ArrayPlotStudyEvaluator"),
        (ScalarPlotSpec(), "ScalarPlotEvaluator"),
        (StabilityFractionByGroupSpec(), "StabilityFractionByGroupEvaluator"),
    ],
)
def test_study_evaluation_specs_build_runtime_evaluators(
    spec, runtime_type_name: str
) -> None:
    assert type(spec.build()).__name__ == runtime_type_name
