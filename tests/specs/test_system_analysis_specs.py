from __future__ import annotations

import importlib

import pytest

from pgopinf.specs.system_analysis.base import (
    SYSTEM_ANALYSIS_TASK_REGISTRY,
    system_analysis_task_default_by_kind,
    system_analysis_task_from_dict,
)
from pgopinf.specs.system_analysis.system_analysis import (
    SystemAnalysisSpec,
)
from pgopinf.specs.system_analysis.tasks import (
    EigenvaluesSpec,
    HInfNormSpec,
    MinimalRealizationSpec,
    SigmaPlotSpec,
)


SYSTEM_ANALYSIS_TASK_MODULES = (
    "pgopinf.specs.system_analysis.tasks",
)


@pytest.fixture(scope="module", autouse=True)
def register_system_analysis_task_modules() -> None:
    for module_name in SYSTEM_ANALYSIS_TASK_MODULES:
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("kind", "spec", "result_key"),
    [
        ("eigenvalues", EigenvaluesSpec(which="all"), "eigenvalues"),
        (
            "minimal_realization",
            MinimalRealizationSpec(trunc_tol=1e-8),
            "minimal_realization",
        ),
        ("sigma_plot", SigmaPlotSpec(), "sigma_plot"),
        ("hinf_norm", HInfNormSpec(), "hinf_norm"),
    ],
)
def test_system_analysis_task_spec_round_trip(
    kind: str, spec, result_key: str
) -> None:
    assert spec.kind == kind
    assert spec.result_key() == result_key
    assert spec.to_dict()["kind"] == kind
    assert system_analysis_task_from_dict(spec.to_dict()) == spec


def test_known_system_analysis_task_kinds_are_registered() -> None:
    assert SYSTEM_ANALYSIS_TASK_REGISTRY.known_kinds() == [
        "eigenvalues",
        "hinf_norm",
        "minimal_realization",
        "sigma_plot",
    ]


@pytest.mark.parametrize(
    ("kind", "expected_spec"),
    [
        ("eigenvalues", EigenvaluesSpec()),
        ("hinf_norm", HInfNormSpec()),
        ("minimal_realization", MinimalRealizationSpec()),
        ("sigma_plot", SigmaPlotSpec()),
    ],
)
def test_system_analysis_task_default_by_kind(kind: str, expected_spec) -> None:
    assert system_analysis_task_default_by_kind(kind) == expected_spec


def test_system_analysis_task_from_dict_requires_known_kind() -> None:
    with pytest.raises(KeyError, match="missing required keys"):
        system_analysis_task_from_dict({})

    with pytest.raises(ValueError, match="unknown kind"):
        system_analysis_task_from_dict({"kind": "missing"})


def test_system_analysis_task_from_dict_uses_spec_specific_parsers() -> None:
    assert system_analysis_task_from_dict(
        {"kind": "minimal_realization", "trunc_tol": "1e-7"}
    ) == MinimalRealizationSpec(trunc_tol=1e-7)

    assert system_analysis_task_from_dict(
        {"kind": "eigenvalues", "which": 12}
    ) == EigenvaluesSpec(which="12")


def test_system_analysis_spec_round_trip_and_tuple_normalization() -> None:
    spec = SystemAnalysisSpec(
        tasks=[
            EigenvaluesSpec(which="all"),
            MinimalRealizationSpec(trunc_tol=1e-9),
            SigmaPlotSpec(),
            HInfNormSpec(),
        ]
    )

    assert isinstance(spec.tasks, tuple)
    assert SystemAnalysisSpec.from_dict(spec.to_dict()) == spec


def test_system_analysis_spec_from_dict_defaults() -> None:
    assert SystemAnalysisSpec.from_dict(None) == SystemAnalysisSpec()
    assert SystemAnalysisSpec.from_dict({}) == SystemAnalysisSpec()


@pytest.mark.parametrize(
    ("spec", "runtime_type_name"),
    [
        (EigenvaluesSpec(), "EigenvaluesTask"),
        (MinimalRealizationSpec(), "MinimalRealizationTask"),
        (SigmaPlotSpec(), "SigmaPlotTask"),
        (HInfNormSpec(), "HInfNormTask"),
    ],
)
def test_system_analysis_task_specs_build_runtime_tasks(
    spec, runtime_type_name: str
) -> None:
    assert type(spec.build()).__name__ == runtime_type_name


def test_system_analysis_task_specs_build_forward_options() -> None:
    eigenvalues = EigenvaluesSpec(which="all").build()
    assert eigenvalues.which == "all"

    minimal_realization = MinimalRealizationSpec(trunc_tol=1e-6).build()
    assert minimal_realization.trunc_tol == pytest.approx(1e-6)

    assert SigmaPlotSpec().build().task_kind == "sigma_plot"
    assert HInfNormSpec().build().task_kind == "hinf_norm"
