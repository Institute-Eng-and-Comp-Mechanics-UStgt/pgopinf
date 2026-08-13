from __future__ import annotations

import importlib

import pytest

from pgopinf.specs.evaluation.base import (
    METRIC_REGISTRY,
    metric_from_dict,
    metric_to_dict,
)
from pgopinf.specs.evaluation.evaluation import EvaluationSpec
from pgopinf.specs.evaluation.metrics.comp_int_ident_matABCD import (
    CompIntIdentMatABCDSpec,
)
from pgopinf.specs.evaluation.metrics.hinf_error import HInfErrorSpec
from pgopinf.specs.evaluation.metrics.input_values_over_time import (
    InputValuesOverTimeSpec,
)
from pgopinf.specs.evaluation.metrics.opinf_theory import (
    OpInfTheorySpec,
)
from pgopinf.specs.evaluation.metrics.output_values_over_time import (
    OutputValuesOverTimeSpec,
)
from pgopinf.specs.evaluation.metrics.singular_value_decay import (
    SingularValueDecaySpec,
)
from pgopinf.specs.evaluation.metrics.spectral_abscissa import (
    SpectralAbscissaSpec,
)
from pgopinf.specs.evaluation.metrics.state_values_over_time import (
    StateValuesOverTimeSpec,
)
from pgopinf.specs.evaluation.reporting import (
    OutputSpec,
    ReportingSpec,
)

METRIC_MODULES = (
    "pgopinf.specs.evaluation.metrics.comp_int_ident_matABCD",
    "pgopinf.specs.evaluation.metrics.hinf_error",
    "pgopinf.specs.evaluation.metrics.input_values_over_time",
    "pgopinf.specs.evaluation.metrics.opinf_theory",
    "pgopinf.specs.evaluation.metrics.output_values_over_time",
    "pgopinf.specs.evaluation.metrics.singular_value_decay",
    "pgopinf.specs.evaluation.metrics.spectral_abscissa",
    "pgopinf.specs.evaluation.metrics.state_values_over_time",
)


@pytest.fixture(scope="module", autouse=True)
def register_metric_modules() -> None:
    for module_name in METRIC_MODULES:
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("kind", "spec"),
    [
        (
            "comp_int_ident_matABCD",
            CompIntIdentMatABCDSpec(relative=False, rel_threshold=1e-6),
        ),
        (
            "hinf_error",
            HInfErrorSpec(
                target="intrusive",
                tol=1e-8,
                stabilized=True,
                use_minimal_realization=True,
                relative=False,
            ),
        ),
        ("input_values_over_time", InputValuesOverTimeSpec(split="TRAIN")),
        ("opinf_theory", OpInfTheorySpec(rcond=1e-10)),
        (
            "output_values_over_time",
            OutputValuesOverTimeSpec(
                split="TRAIN",
                target="intrusive",
                relative=False,
            ),
        ),
        ("singular_value_decay", SingularValueDecaySpec()),
        (
            "spectral_abscissa",
            SpectralAbscissaSpec(target="original", tol=1e-9, stabilized=True),
        ),
        (
            "state_values_over_time",
            StateValuesOverTimeSpec(
                split="TRAIN",
                target="intrusive",
                relative=False,
            ),
        ),
    ],
)
def test_metric_spec_round_trip(kind: str, spec) -> None:
    assert spec.kind == kind
    assert metric_to_dict(spec)["kind"] == kind
    assert metric_from_dict(metric_to_dict(spec)) == spec


def test_known_metric_kinds_are_registered() -> None:
    assert METRIC_REGISTRY.known_kinds() == [
        "comp_int_ident_matABCD",
        "hinf_error",
        "input_values_over_time",
        "opinf_theory",
        "output_values_over_time",
        "singular_value_decay",
        "spectral_abscissa",
        "state_values_over_time",
    ]


@pytest.mark.parametrize(
    ("kind", "expected_spec"),
    [
        ("comp_int_ident_matABCD", CompIntIdentMatABCDSpec()),
        ("hinf_error", HInfErrorSpec()),
        ("input_values_over_time", InputValuesOverTimeSpec()),
        ("opinf_theory", OpInfTheorySpec()),
        ("output_values_over_time", OutputValuesOverTimeSpec()),
        ("singular_value_decay", SingularValueDecaySpec()),
        ("spectral_abscissa", SpectralAbscissaSpec()),
        ("state_values_over_time", StateValuesOverTimeSpec()),
    ],
)
def test_metric_default_by_kind(kind: str, expected_spec) -> None:
    assert METRIC_REGISTRY.default_by_kind(kind) == expected_spec


def test_metric_from_dict_requires_known_kind() -> None:
    with pytest.raises(KeyError, match="missing required keys"):
        metric_from_dict({})

    with pytest.raises(ValueError, match="unknown kind"):
        metric_from_dict({"kind": "missing_metric"})


def test_evaluation_spec_round_trip_and_addition() -> None:
    left = EvaluationSpec.from_dict(
        {
            "metrics": [
                {"kind": "hinf_error", "target": "identified"},
                {"kind": "opinf_theory", "rcond": 1e-11},
            ]
        }
    )
    right = EvaluationSpec.from_dict(
        {"metrics": [{"kind": "state_values_over_time", "split": "TRAIN"}]}
    )

    assert left == EvaluationSpec(
        metrics=(
            HInfErrorSpec(target="identified"),
            OpInfTheorySpec(rcond=1e-11),
        )
    )
    assert EvaluationSpec.from_dict(left.to_dict()) == left
    assert left + right == EvaluationSpec(metrics=left.metrics + right.metrics)


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
def test_evaluation_presets(name: str, expected_count: int) -> None:
    spec = EvaluationSpec.preset(name)

    assert isinstance(spec, EvaluationSpec)
    assert len(spec.metrics) == expected_count
    assert EvaluationSpec.from_dict(spec.to_dict()) == spec


def test_unknown_evaluation_preset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown evaluation preset name"):
        EvaluationSpec.preset("missing_preset")


@pytest.mark.parametrize(
    ("spec", "runtime_type_name"),
    [
        (CompIntIdentMatABCDSpec(), "CompIntIdentMatABCDMetric"),
        (HInfErrorSpec(), "HInfErrorMetric"),
        (InputValuesOverTimeSpec(), "InputValuesOverTimeMetric"),
        (OpInfTheorySpec(), "OpInfTheoryMetric"),
        (OutputValuesOverTimeSpec(), "OutputValuesOverTimeMetric"),
        (SingularValueDecaySpec(), "SingularValueDecayMetric"),
        (SpectralAbscissaSpec(), "SpectralAbscissaMetric"),
        (StateValuesOverTimeSpec(), "StateValuesOverTimeMetric"),
    ],
)
def test_metric_specs_build_runtime_metrics(spec, runtime_type_name: str) -> None:
    assert type(spec.build()).__name__ == runtime_type_name


def test_reporting_spec_defaults() -> None:
    assert OutputSpec() == OutputSpec(
        csv=True,
        png=True,
        pdf=False,
        append_csv=True,
        append_jsonl=False,
        log_to_terminal=True,
    )
    assert ReportingSpec() == ReportingSpec(
        default_outputs=OutputSpec(),
        save_summaries_csv=True,
    )
