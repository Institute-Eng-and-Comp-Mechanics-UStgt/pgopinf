from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.metrics import (
    input_values_over_time as input_values_module,
)
from pgopinf.evaluation.metrics.input_values_over_time import (
    InputValuesOverTimeMetric,
)


def make_context(*, train_u: np.ndarray, test_u: np.ndarray) -> EvaluationContext:
    train = SimpleNamespace(
        U=train_u,
        time=SimpleNamespace(t=np.array([0.0, 0.5, 1.0])),
    )
    test = SimpleNamespace(
        U=test_u,
        time=SimpleNamespace(t=np.array([10.0, 20.0, 30.0])),
    )
    dataset_artifact = SimpleNamespace(
        data=SimpleNamespace(TRAIN=train, TEST=test),
    )
    return EvaluationContext(
        experiment_name="experiment",
        run_id="run",
        original_system=object(),
        dataset_artifact=dataset_artifact,
        reduction_result=object(),
        dataset_artifact_intrusive=None,
        identification_result=object(),
        dataset_artifact_identified=None,
    )


def test_input_values_over_time_builds_array_plot_for_selected_split() -> None:
    train_u = np.zeros((2, 3, 2))
    test_u = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ]
    )
    ctx = make_context(train_u=train_u, test_u=test_u)

    output = InputValuesOverTimeMetric(split="TEST").compute(ctx)

    assert output.summaries == []
    assert len(output.artifacts) == 1
    artifact = output.artifacts[0]
    assert artifact.name == "input_values_over_time"
    np.testing.assert_allclose(artifact.x, np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0]))
    assert artifact.x_label == "time"
    assert artifact.line_labels == ("u",)
    assert artifact.y_labels == ("u_0", "u_1")
    assert artifact.subplot_titles == ("Input 0", "Input 1")
    assert artifact.meta == {"split": "TEST"}
    assert artifact.y.shape == (6, 1, 2)
    np.testing.assert_allclose(
        artifact.y[:, 0, 0],
        np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0]),
    )
    np.testing.assert_allclose(
        artifact.y[:, 0, 1],
        np.array([4.0, 5.0, 6.0, 40.0, 50.0, 60.0]),
    )


def test_input_values_over_time_uses_train_split_when_requested() -> None:
    train_u = np.array([[[7.0], [8.0], [9.0]]])
    test_u = np.array([[[1.0], [2.0], [3.0]]])
    ctx = make_context(train_u=train_u, test_u=test_u)

    output = InputValuesOverTimeMetric(split="TRAIN").compute(ctx)

    artifact = output.artifacts[0]
    np.testing.assert_allclose(artifact.x, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(artifact.y[:, 0, 0], np.array([7.0, 8.0, 9.0]))
    assert artifact.y_labels == ("u_0",)
    assert artifact.meta == {"split": "TRAIN"}


def test_input_values_over_time_limits_to_four_selected_inputs(monkeypatch) -> None:
    calls = []
    selected = np.array([4, 2, 0, 3])

    def fake_pick_trajectories(data, **kwargs):
        calls.append((data, kwargs))
        return np.take(data, selected, axis=0), selected

    monkeypatch.setattr(
        input_values_module,
        "pick_trajectories",
        fake_pick_trajectories,
    )
    train_u = np.zeros((1, 3, 1))
    test_u = np.arange(5 * 3 * 1, dtype=float).reshape(5, 3, 1)
    ctx = make_context(train_u=train_u, test_u=test_u)

    output = InputValuesOverTimeMetric().compute(ctx)

    assert calls[0][1] == {
        "idx_type": "rand",
        "max_size": 4,
        "return_idx": True,
    }
    artifact = output.artifacts[0]
    assert artifact.y.shape == (3, 1, 4)
    assert artifact.y_labels == ("u_4", "u_2", "u_0", "u_3")
    assert artifact.subplot_titles == ("Input 4", "Input 2", "Input 0", "Input 3")
    np.testing.assert_allclose(artifact.y[:, 0, 0], test_u[4, :, 0])
    np.testing.assert_allclose(artifact.y[:, 0, 1], test_u[2, :, 0])
