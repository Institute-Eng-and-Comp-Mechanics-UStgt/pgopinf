from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.metrics import (
    output_values_over_time as output_values_module,
)
from pgopinf.evaluation.metrics.output_values_over_time import (
    OutputValuesOverTimeMetric,
)


def make_artifact(train_y: np.ndarray, test_y: np.ndarray):
    train = SimpleNamespace(
        Y=train_y,
        time=SimpleNamespace(t=np.array([0.0, 0.5, 1.0])),
    )
    test = SimpleNamespace(
        Y=test_y,
        time=SimpleNamespace(t=np.array([10.0, 20.0, 30.0])),
    )
    return SimpleNamespace(data=SimpleNamespace(TRAIN=train, TEST=test))


def make_context(
    *,
    ref_train_y: np.ndarray | None = None,
    ref_test_y: np.ndarray | None = None,
    identified_train_y: np.ndarray | None = None,
    identified_test_y: np.ndarray | None = None,
    intrusive_train_y: np.ndarray | None = None,
    intrusive_test_y: np.ndarray | None = None,
) -> EvaluationContext:
    ref_train_y = np.zeros((1, 3, 1)) if ref_train_y is None else ref_train_y
    ref_test_y = np.zeros((1, 3, 1)) if ref_test_y is None else ref_test_y
    dataset_artifact_identified = None
    if identified_train_y is not None and identified_test_y is not None:
        dataset_artifact_identified = make_artifact(identified_train_y, identified_test_y)
    dataset_artifact_intrusive = None
    if intrusive_train_y is not None and intrusive_test_y is not None:
        dataset_artifact_intrusive = make_artifact(intrusive_train_y, intrusive_test_y)

    return EvaluationContext(
        experiment_name="experiment",
        run_id="run",
        original_system=object(),
        dataset_artifact=make_artifact(ref_train_y, ref_test_y),
        reduction_result=object(),
        dataset_artifact_intrusive=dataset_artifact_intrusive,
        identification_result=object(),
        dataset_artifact_identified=dataset_artifact_identified,
    )


def test_output_values_over_time_builds_identified_relative_artifacts_and_summaries() -> None:
    ref_test_y = np.array(
        [
            [[2.0, 20.0], [4.0, 40.0], [6.0, 60.0]],
            [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]],
        ]
    )
    cmp_test_y = ref_test_y + 1.0
    ctx = make_context(
        ref_test_y=ref_test_y,
        identified_train_y=np.zeros((2, 3, 1)),
        identified_test_y=cmp_test_y,
    )

    output = OutputValuesOverTimeMetric(target="identified", relative=True).compute(ctx)

    assert output.summaries == [
        {
            "metric": "max_error_output",
            "split": "TEST",
            "target": "identified",
            "value": pytest.approx(1.0 / 3.0),
            "unit": "-",
        },
        {
            "metric": "mean_error_output",
            "split": "TEST",
            "target": "identified",
            "value": pytest.approx((1 / 4 + 1 / 40 + 1 / 3 + 1 / 30) / 4),
            "unit": "-",
        },
    ]

    values_plot, error_plot, mean_error, max_error = output.artifacts
    np.testing.assert_allclose(
        values_plot.x,
        np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0]),
    )
    assert values_plot.name == "output_vs_original_over_time"
    assert values_plot.y.shape == (6, 2, 2)
    assert values_plot.y_labels == ("y_0", "y_1")
    assert values_plot.line_labels == ("original", "identified")
    assert values_plot.linestyles == ("-", "--")
    assert values_plot.subplot_titles == ("Output 0", "Output 1")
    assert values_plot.meta == {"split": "TEST", "target": "identified"}
    np.testing.assert_allclose(
        values_plot.y[:, 0, 0],
        np.array([2.0, 4.0, 6.0, 20.0, 40.0, 60.0]),
    )
    np.testing.assert_allclose(
        values_plot.y[:, 1, 0],
        np.array([3.0, 5.0, 7.0, 21.0, 41.0, 61.0]),
    )

    np.testing.assert_allclose(values_plot.x, error_plot.x)
    assert error_plot.name == "error_output_vs_original_over_time"
    assert error_plot.y.shape == (6, 2, 1)
    assert error_plot.y_labels == ("rel. error",)
    assert error_plot.line_labels == ("y_0", "y_1")
    assert error_plot.subplot_titles == ("rel. error TEST outputs",)
    assert error_plot.meta == {
        "split": "TEST",
        "target": "identified",
        "relative": True,
    }
    assert mean_error.name == "mean_error_output"
    assert mean_error.value == pytest.approx(output.summaries[1]["value"])
    assert max_error.name == "max_error_output"
    assert max_error.value == pytest.approx(output.summaries[0]["value"])


def test_output_values_over_time_uses_intrusive_train_split_with_absolute_errors() -> None:
    ref_train_y = np.array([[[1.0], [2.0], [3.0]]])
    cmp_train_y = np.array([[[2.5], [3.5], [4.5]]])
    ctx = make_context(
        ref_train_y=ref_train_y,
        intrusive_train_y=cmp_train_y,
        intrusive_test_y=np.zeros((1, 3, 1)),
    )

    output = OutputValuesOverTimeMetric(
        split="TRAIN",
        target="intrusive",
        relative=False,
    ).compute(ctx)

    values_plot, error_plot, mean_error, max_error = output.artifacts
    np.testing.assert_allclose(values_plot.x, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(values_plot.y[:, 0, 0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(values_plot.y[:, 1, 0], np.array([2.5, 3.5, 4.5]))
    assert values_plot.line_labels == ("original", "intrusive")
    assert error_plot.y_labels == ("abs. error",)
    assert error_plot.line_labels == ("y_0",)
    np.testing.assert_allclose(error_plot.y[:, 0, 0], np.array([1.5, 1.5, 1.5]))
    assert mean_error.value == 1.5
    assert max_error.value == 1.5
    assert output.summaries[0]["target"] == "intrusive"
    assert output.summaries[1]["split"] == "TRAIN"


def test_output_values_over_time_limits_to_four_selected_outputs(monkeypatch) -> None:
    selected = np.array([4, 2, 0, 3])

    def fake_pick_trajectories(data, **kwargs):
        assert kwargs == {
            "idx_type": "rand",
            "max_size": 4,
            "return_idx": True,
        }
        return np.take(data, selected, axis=0), selected

    monkeypatch.setattr(
        output_values_module,
        "pick_trajectories",
        fake_pick_trajectories,
    )
    ref_test_y = np.arange(5 * 3 * 1, dtype=float).reshape(5, 3, 1) + 1.0
    cmp_test_y = ref_test_y + 0.5
    ctx = make_context(
        ref_test_y=ref_test_y,
        identified_train_y=np.zeros((5, 3, 1)),
        identified_test_y=cmp_test_y,
    )

    output = OutputValuesOverTimeMetric().compute(ctx)

    values_plot, error_plot = output.artifacts[:2]
    assert values_plot.y.shape == (3, 2, 4)
    assert values_plot.y_labels == ("y_4", "y_2", "y_0", "y_3")
    assert values_plot.subplot_titles == ("Output 4", "Output 2", "Output 0", "Output 3")
    assert error_plot.y.shape == (3, 4, 1)
    assert error_plot.line_labels == ("y_4", "y_2", "y_0", "y_3")


@pytest.mark.parametrize(
    ("metric", "ctx", "message"),
    [
        (
            OutputValuesOverTimeMetric(target="identified"),
            make_context(),
            "No identified dataset artifact available.",
        ),
        (
            OutputValuesOverTimeMetric(target="intrusive"),
            make_context(),
            "No intrusive dataset artifact available.",
        ),
        (
            OutputValuesOverTimeMetric(target="other"),
            make_context(
                identified_train_y=np.zeros((1, 3, 1)),
                identified_test_y=np.zeros((1, 3, 1)),
            ),
            "Unknown target 'other'",
        ),
    ],
)
def test_output_values_over_time_rejects_missing_artifacts_and_unknown_targets(
    metric, ctx, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        metric.compute(ctx)
