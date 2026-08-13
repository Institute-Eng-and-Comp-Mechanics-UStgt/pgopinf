from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.metrics.singular_value_decay import (
    SingularValueDecayMetric,
)


def make_context(*, full_basis) -> EvaluationContext:
    return EvaluationContext(
        experiment_name="experiment",
        run_id="run",
        original_system=object(),
        dataset_artifact=object(),
        reduction_result=SimpleNamespace(full_basis=full_basis),
        dataset_artifact_intrusive=None,
        identification_result=object(),
        dataset_artifact_identified=None,
    )


def test_singular_value_decay_builds_array_plot_from_full_basis_singular_values() -> None:
    ctx = make_context(full_basis=SimpleNamespace(S=np.array([10.0, 3.0, 1.0])))

    output = SingularValueDecayMetric().compute(ctx)

    assert output.summaries == []
    assert len(output.artifacts) == 1
    artifact = output.artifacts[0]
    assert artifact.name == "singular_value_decay"
    np.testing.assert_allclose(artifact.x, np.array([1, 2, 3]))
    np.testing.assert_allclose(artifact.y[:, 0, 0], np.array([10.0, 3.0, 1.0]))
    assert artifact.y.shape == (3, 1, 1)
    assert artifact.x_label == "reduced order r"
    assert artifact.y_labels == ("singular value",)
    assert artifact.line_labels == ()
    assert artifact.subplot_titles == ("singular value decay over r",)
    assert artifact.meta == {}


def test_singular_value_decay_accepts_list_like_singular_values() -> None:
    ctx = make_context(full_basis=SimpleNamespace(S=[4.0, 2.0]))

    output = SingularValueDecayMetric().compute(ctx)

    np.testing.assert_allclose(output.artifacts[0].x, np.array([1, 2]))
    np.testing.assert_allclose(output.artifacts[0].y[:, 0, 0], np.array([4.0, 2.0]))


def test_singular_value_decay_returns_none_when_full_basis_has_no_singular_values() -> None:
    ctx = make_context(full_basis=SimpleNamespace())

    assert SingularValueDecayMetric().compute(ctx) is None
