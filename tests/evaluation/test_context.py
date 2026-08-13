from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pgopinf.evaluation.context import EvaluationContext


def test_evaluation_context_stores_runtime_objects_and_defaults_analysis() -> None:
    original_system = object()
    dataset_artifact = object()
    reduction_result = object()
    identification_result = object()

    ctx = EvaluationContext(
        experiment_name="exp",
        run_id="run-1",
        original_system=original_system,
        dataset_artifact=dataset_artifact,
        reduction_result=reduction_result,
        dataset_artifact_intrusive=None,
        identification_result=identification_result,
        dataset_artifact_identified=None,
    )

    assert ctx.experiment_name == "exp"
    assert ctx.run_id == "run-1"
    assert ctx.original_system is original_system
    assert ctx.dataset_artifact is dataset_artifact
    assert ctx.reduction_result is reduction_result
    assert ctx.dataset_artifact_intrusive is None
    assert ctx.identification_result is identification_result
    assert ctx.dataset_artifact_identified is None
    assert ctx.system_analysis is None


def test_evaluation_context_is_frozen() -> None:
    ctx = EvaluationContext(
        experiment_name="exp",
        run_id="run-1",
        original_system=object(),
        dataset_artifact=object(),
        reduction_result=object(),
        dataset_artifact_intrusive=object(),
        identification_result=object(),
        dataset_artifact_identified=object(),
        system_analysis={"eig": [1.0]},
    )

    with pytest.raises(FrozenInstanceError):
        ctx.run_id = "run-2"  # type: ignore[misc]
