from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.metrics import hinf_error as hinf_module
from pgopinf.evaluation.metrics.hinf_error import HInfErrorMetric


class FakeSystem:
    def __init__(self, name: str):
        self.name = name
        self.stable_decomposition_calls = 0
        self.minimal_realization_calls = []
        self.stable_system = None
        self.minimal_system = None

    def stable_decomposition(self):
        self.stable_decomposition_calls += 1
        return self.stable_system

    def minimal_realization(self, *, trunc_tol: float):
        self.minimal_realization_calls.append(trunc_tol)
        return self.minimal_system


def make_context(
    *,
    original_system=None,
    reduced_system=None,
    identified_system=None,
    dataset_artifact_intrusive=None,
    dataset_artifact_identified=None,
    system_analysis=None,
) -> EvaluationContext:
    return EvaluationContext(
        experiment_name="experiment",
        run_id="run",
        original_system=original_system or FakeSystem("original"),
        dataset_artifact=object(),
        reduction_result=SimpleNamespace(reduced_system=reduced_system),
        dataset_artifact_intrusive=dataset_artifact_intrusive,
        identification_result=SimpleNamespace(system_identified=identified_system),
        dataset_artifact_identified=dataset_artifact_identified,
        system_analysis=system_analysis,
    )


def install_fake_hinf(monkeypatch):
    calls = {"hinf_error": [], "hinf_norm": []}

    def fake_hinf_error(**kwargs):
        calls["hinf_error"].append(kwargs)
        return 6.0, -1.0, -2.0

    def fake_hinf_norm(**kwargs):
        calls["hinf_norm"].append(kwargs)
        return 3.0

    monkeypatch.setattr(hinf_module, "hinf_error", fake_hinf_error)
    monkeypatch.setattr(hinf_module, "hinf_norm", fake_hinf_norm)
    return calls


def test_hinf_error_identified_uses_precomputed_analysis_and_relative_norm(
    monkeypatch,
) -> None:
    calls = install_fake_hinf(monkeypatch)
    original = FakeSystem("original")
    identified = FakeSystem("identified")
    eigenvalues = np.array([-1.0, -2.0])
    ctx = make_context(
        original_system=original,
        identified_system=identified,
        dataset_artifact_identified=object(),
        system_analysis=SimpleNamespace(
            values={"eigenvalues": eigenvalues, "hinf_norm": 12.0}
        ),
    )

    output = HInfErrorMetric(tol=1e-5).compute(ctx)

    assert calls["hinf_error"] == [
        {
            "first_system": original,
            "second_system": identified,
            "tol": 1e-5,
            "return_spectral_abscissas": True,
            "first_system_eigvals": eigenvalues,
        }
    ]
    assert calls["hinf_norm"] == []
    assert output.summaries == [
        {
            "metric": "hinf_error",
            "value": 0.5,
            "unit": "-",
            "target": "identified",
            "tol": 1e-5,
            "stabilized": False,
            "relative": True,
        }
    ]
    assert [artifact.name for artifact in output.artifacts] == [
        "hinf_error",
        "spectral_abscissa_orig",
        "spectral_abscissa_cmp",
    ]
    assert output.artifacts[0].value == 0.5
    assert output.artifacts[0].meta == {
        "target": "identified",
        "tol": 1e-5,
        "stabilized": False,
        "relative": True,
    }
    assert output.artifacts[1].value == -1.0
    assert output.artifacts[1].meta == {}
    assert output.artifacts[2].value == -2.0
    assert output.artifacts[2].meta == {
        "target": "identified",
        "stabilized": False,
    }


def test_hinf_error_intrusive_can_stabilize_and_skip_relative_scaling(
    monkeypatch,
) -> None:
    calls = install_fake_hinf(monkeypatch)
    original = FakeSystem("original")
    reduced = FakeSystem("reduced")
    stable_reduced = FakeSystem("stable-reduced")
    reduced.stable_system = stable_reduced
    ctx = make_context(
        original_system=original,
        reduced_system=reduced,
        dataset_artifact_intrusive=object(),
    )

    output = HInfErrorMetric(
        target="intrusive",
        stabilized=True,
        relative=False,
        tol=1e-4,
    ).compute(ctx)

    assert reduced.stable_decomposition_calls == 1
    assert calls["hinf_error"][0]["first_system"] is original
    assert calls["hinf_error"][0]["second_system"] is stable_reduced
    assert calls["hinf_error"][0]["first_system_eigvals"] is None
    assert calls["hinf_error"][0]["tol"] == 1e-4
    assert calls["hinf_norm"] == []
    assert output.summaries[0]["value"] == 6.0
    assert output.summaries[0]["target"] == "intrusive"
    assert output.summaries[0]["stabilized"] is True
    assert output.summaries[0]["relative"] is False
    assert output.artifacts[2].meta == {
        "target": "intrusive",
        "stabilized": True,
    }


def test_hinf_error_relative_metric_computes_hinf_norm_when_not_precomputed(
    monkeypatch,
) -> None:
    calls = install_fake_hinf(monkeypatch)
    original = FakeSystem("original")
    identified = FakeSystem("identified")
    ctx = make_context(
        original_system=original,
        identified_system=identified,
        dataset_artifact_identified=object(),
    )

    output = HInfErrorMetric(relative=True).compute(ctx)

    assert calls["hinf_norm"] == [{"first_system": original}]
    assert output.summaries[0]["value"] == 2.0


def test_hinf_error_uses_precomputed_minimal_realization(monkeypatch) -> None:
    calls = install_fake_hinf(monkeypatch)
    original = FakeSystem("original")
    minimal = FakeSystem("minimal")
    identified = FakeSystem("identified")
    ctx = make_context(
        original_system=original,
        identified_system=identified,
        dataset_artifact_identified=object(),
        system_analysis=SimpleNamespace(
            values={"minimal_realization": minimal, "hinf_norm": 2.0}
        ),
    )

    HInfErrorMetric(use_minimal_realization=True).compute(ctx)

    assert original.minimal_realization_calls == []
    assert calls["hinf_error"][0]["first_system"] is minimal


def test_hinf_error_computes_minimal_realization_when_not_precomputed(
    monkeypatch,
) -> None:
    calls = install_fake_hinf(monkeypatch)
    original = FakeSystem("original")
    minimal = FakeSystem("minimal")
    identified = FakeSystem("identified")
    original.minimal_system = minimal
    ctx = make_context(
        original_system=original,
        identified_system=identified,
        dataset_artifact_identified=object(),
        system_analysis=SimpleNamespace(values={"hinf_norm": 2.0}),
    )

    HInfErrorMetric(use_minimal_realization=True).compute(ctx)

    assert original.minimal_realization_calls == [1e-10]
    assert calls["hinf_error"][0]["first_system"] is minimal


@pytest.mark.parametrize(
    ("metric", "ctx", "message"),
    [
        (
            HInfErrorMetric(target="identified"),
            make_context(identified_system=FakeSystem("identified")),
            "No identified dataset artifact available.",
        ),
        (
            HInfErrorMetric(target="intrusive"),
            make_context(reduced_system=FakeSystem("reduced")),
            "No intrusive dataset artifact available.",
        ),
        (
            HInfErrorMetric(target="unknown"),
            make_context(
                identified_system=FakeSystem("identified"),
                dataset_artifact_identified=object(),
            ),
            "Unknown target 'unknown'",
        ),
    ],
)
def test_hinf_error_rejects_missing_artifacts_and_unknown_targets(
    metric, ctx, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        metric.compute(ctx)
