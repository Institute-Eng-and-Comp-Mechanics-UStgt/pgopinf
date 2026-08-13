from __future__ import annotations

import numpy as np
import pytest

from pgopinf.evaluation.results import (
    ArrayPlotResult,
    MetricOutput,
    ScalarMetricResult,
    SeriesMetricResult,
    TableMetricResult,
)


def test_scalar_series_table_and_output_containers_store_values() -> None:
    scalar = ScalarMetricResult(name="error", value=0.1)
    series = SeriesMetricResult(
        name="trajectory",
        x=np.array([0.0, 1.0]),
        y=np.array([2.0, 3.0]),
        meta={"split": "TRAIN"},
    )
    table = TableMetricResult(
        name="values",
        columns=("a", "b"),
        values=np.array([[1.0, 2.0]]),
    )
    output = MetricOutput(summaries=[{"metric": "error"}], artifacts=[scalar])

    assert scalar.unit == "-"
    assert scalar.meta == {}
    assert series.x_label == "x"
    assert series.y_label == "y"
    assert series.unit == "-"
    assert series.meta == {"split": "TRAIN"}
    assert table.columns == ("a", "b")
    assert output.summaries == [{"metric": "error"}]
    assert output.artifacts == [scalar]


def test_result_meta_defaults_are_independent() -> None:
    left = ScalarMetricResult(name="left", value=1.0)
    right = ScalarMetricResult(name="right", value=2.0)

    left.meta["source"] = "identified"

    assert right.meta == {}


def test_array_plot_result_accepts_consistent_shapes_and_labels() -> None:
    result = ArrayPlotResult(
        name="states",
        x=np.array([0.0, 1.0, 2.0]),
        y=np.ones((3, 2, 4)),
        x_label="time",
        y_labels=("x0", "x1", "x2", "x3"),
        line_labels=("truth", "identified"),
        subplot_titles=("state 0", "state 1", "state 2", "state 3"),
        markers=("o", None),
        linestyles=("-", "--"),
        line_modes=("line", "line+marker"),
        markevery=(None, 2),
        colors=("black", "tab:blue"),
        meta={"split": "TEST"},
    )

    assert result.name == "states"
    np.testing.assert_allclose(result.x, np.array([0.0, 1.0, 2.0]))
    assert result.y.shape == (3, 2, 4)
    assert result.x_label == "time"
    assert result.meta == {"split": "TEST"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x": np.ones((3, 1)), "y": np.ones((3, 2, 1))}, "x must be 1D"),
        (
            {"x": np.ones(3), "y": np.ones((3, 2))},
            "y must have shape",
        ),
        (
            {"x": np.ones(3), "y": np.ones((2, 2, 1))},
            "x length 3 does not match y first dim 2",
        ),
        (
            {"x": np.ones(3), "y": np.ones((3, 2, 1)), "line_labels": ("a",)},
            "line_labels has length 1 but n_lines=2",
        ),
        (
            {"x": np.ones(3), "y": np.ones((3, 2, 1)), "y_labels": ("a", "b")},
            "y_labels has length 2 but n_subplots=1",
        ),
        (
            {
                "x": np.ones(3),
                "y": np.ones((3, 2, 1)),
                "subplot_titles": ("a", "b"),
            },
            "subplot_titles has length 2 but n_subplots=1",
        ),
        (
            {"x": np.ones(3), "y": np.ones((3, 2, 1)), "markers": ("o",)},
            "markers must have length 2",
        ),
        (
            {"x": np.ones(3), "y": np.ones((3, 2, 1)), "linestyles": ("-",)},
            "linestyles must have length 2",
        ),
        (
            {"x": np.ones(3), "y": np.ones((3, 2, 1)), "line_modes": ("line",)},
            "line_modes must have length 2",
        ),
        (
            {"x": np.ones(3), "y": np.ones((3, 2, 1)), "markevery": (1,)},
            "markevery must have length 2",
        ),
        (
            {"x": np.ones(3), "y": np.ones((3, 2, 1)), "colors": ("red",)},
            "colors must have length 2",
        ),
    ],
)
def test_array_plot_result_rejects_inconsistent_shapes_and_labels(
    kwargs, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ArrayPlotResult(name="bad", **kwargs)
