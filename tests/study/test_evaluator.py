from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pgopinf.io.results_path import ResultsPath
from pgopinf.study.evaluator import (
    ScalarPlotEvaluator,
    _expand_metadata_json,
    _get_dotted_attr,
    _label_from_cols,
    _line_label_from_key,
    _panel_label_from_key,
    _unique_in_order,
)
from pgopinf.study.result import StudyResult, StudyRunRef


@dataclass(frozen=True)
class FakeSpec:
    name: str
    r: int

    @property
    def reduction(self):
        return SimpleNamespace(r=self.r)

    def to_dict(self):
        return {"name": self.name, "reduction": {"r": self.r}}


def write_scalar_metrics(paths: ResultsPath, spec: FakeSpec, ident_id: str, rows) -> None:
    csv_path = paths.for_run(
        experiment_name=spec.name,
        ident_id=ident_id,
    ).scalar_metrics_csv
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def test_helper_functions_format_and_extract_values() -> None:
    obj = SimpleNamespace(a=SimpleNamespace(b=3))
    row = pd.Series({"metric": "error", "variant": "base"})

    assert _get_dotted_attr(obj, "a.b") == 3
    assert _label_from_cols(row, ()) == "value"
    assert _label_from_cols(row, ("metric", "variant")) == "error | base"
    assert _line_label_from_key(()) == "value"
    assert _line_label_from_key(("error", "base")) == "error | base"
    assert _panel_label_from_key((), ()) == "scalar"
    assert _panel_label_from_key(("TRAIN",), ("split",)) == "split=TRAIN"
    assert _unique_in_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_expand_metadata_json_adds_missing_metadata_columns_only() -> None:
    df = pd.DataFrame(
        {
            "metric": ["m1", "m2", "m3"],
            "split": ["existing", "existing", "existing"],
            "metadata_json": [
                '{"split": "TRAIN", "source": "identified"}',
                '{"source": "intrusive"}',
                "not json",
            ],
        }
    )

    expanded = _expand_metadata_json(df.copy())

    assert expanded["split"].tolist() == ["existing", "existing", "existing"]
    assert expanded["source"].tolist() == ["identified", "intrusive", np.nan]


def test_scalar_plot_evaluator_combines_runs_filters_and_builds_plot(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    spec_r2 = FakeSpec(name="exp-r2", r=2)
    spec_r4 = FakeSpec(name="exp-r4", r=4)
    write_scalar_metrics(
        paths,
        spec_r2,
        "id-r2",
        [
            {
                "metric": "err",
                "value": 0.2,
                "metadata_json": '{"split": "TRAIN", "source": "identified"}',
            },
            {
                "metric": "err",
                "value": 9.9,
                "metadata_json": '{"split": "TEST", "source": "identified"}',
            },
            {
                "metric": "bound",
                "value": 0.4,
                "metadata_json": '{"split": "TRAIN", "source": "intrusive"}',
            },
        ],
    )
    write_scalar_metrics(
        paths,
        spec_r4,
        "id-r4",
        [
            {
                "metric": "err",
                "value": 0.1,
                "metadata_json": '{"split": "TRAIN", "source": "identified"}',
            },
            {
                "metric": "bound",
                "value": 0.3,
                "metadata_json": '{"split": "TRAIN", "source": "intrusive"}',
            },
        ],
    )
    study_result = StudyResult(
        study_name="study",
        run_refs=[
            StudyRunRef(ident_id="id-r2", variant_name="base", spec=spec_r2),
            StudyRunRef(ident_id="id-r4", variant_name="petrov", spec=spec_r4),
        ],
    )
    evaluator = ScalarPlotEvaluator(
        metrics=("err", "bound"),
        x="reduction.r",
        line_by=("metric",),
        panel_by=("source",),
        filters={"split": "TRAIN"},
        title="Error sweep",
        x_label="Reduced order",
    )

    raw_df, plot_result = evaluator.evaluate(study_result=study_result, paths=paths)

    assert raw_df[["metric", "value", "variant", "run_id", "x_value", "source"]].to_dict(
        "records"
    ) == [
        {
            "metric": "err",
            "value": 0.2,
            "variant": "base",
            "run_id": "id-r2",
            "x_value": 2,
            "source": "identified",
        },
        {
            "metric": "bound",
            "value": 0.4,
            "variant": "base",
            "run_id": "id-r2",
            "x_value": 2,
            "source": "intrusive",
        },
        {
            "metric": "err",
            "value": 0.1,
            "variant": "petrov",
            "run_id": "id-r4",
            "x_value": 4,
            "source": "identified",
        },
        {
            "metric": "bound",
            "value": 0.3,
            "variant": "petrov",
            "run_id": "id-r4",
            "x_value": 4,
            "source": "intrusive",
        },
    ]
    np.testing.assert_allclose(plot_result.x, np.array([2, 4]))
    assert plot_result.name == "Error sweep"
    assert plot_result.x_label == "Reduced order"
    assert plot_result.line_labels == ("err", "bound")
    assert plot_result.subplot_titles == ("source=identified", "source=intrusive")
    assert plot_result.y_labels == ("value", "value")
    np.testing.assert_allclose(
        plot_result.y[:, :, 0],
        np.array([[0.2, np.nan], [0.1, np.nan]]),
    )
    np.testing.assert_allclose(
        plot_result.y[:, :, 1],
        np.array([[np.nan, 0.4], [np.nan, 0.3]]),
    )
    assert plot_result.meta == {
        "study_plot_kind": "scalar_plot",
        "metrics": ["err", "bound"],
        "x": "reduction.r",
        "line_by": ["metric"],
        "panel_by": ["source"],
    }


def test_scalar_plot_evaluator_skips_missing_and_empty_scalar_csvs(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    spec_missing = FakeSpec(name="missing", r=2)
    spec_empty = FakeSpec(name="empty", r=4)
    empty_csv = paths.for_run(experiment_name=spec_empty.name, ident_id="empty-id").scalar_metrics_csv
    pd.DataFrame(columns=["metric", "value"]).to_csv(empty_csv, index=False)
    study_result = StudyResult(
        study_name="study",
        run_refs=[
            StudyRunRef(ident_id="missing-id", variant_name="base", spec=spec_missing),
            StudyRunRef(ident_id="empty-id", variant_name="base", spec=spec_empty),
        ],
    )
    evaluator = ScalarPlotEvaluator(
        metrics=("err",),
        x="reduction.r",
        line_by=("metric",),
        panel_by=(),
        filters={},
        title="Empty",
    )

    raw_df, plot_result = evaluator.evaluate(study_result=study_result, paths=paths)

    assert raw_df.empty
    assert plot_result.name == "Empty"
    np.testing.assert_allclose(plot_result.x, np.array([]))
    assert plot_result.y.shape == (0, 0, 1)
    assert plot_result.meta == {"study_plot_kind": "scalar_plot", "empty": True}


def test_scalar_plot_evaluator_requires_metric_column(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    spec = FakeSpec(name="exp", r=2)
    write_scalar_metrics(paths, spec, "run-id", [{"name": "err", "value": 1.0}])
    study_result = StudyResult(
        study_name="study",
        run_refs=[StudyRunRef(ident_id="run-id", variant_name="base", spec=spec)],
    )
    evaluator = ScalarPlotEvaluator(
        metrics=("err",),
        x="reduction.r",
        line_by=("metric",),
        panel_by=(),
        filters={},
    )

    with pytest.raises(ValueError, match="does not contain a 'metric' column"):
        evaluator.evaluate(study_result=study_result, paths=paths)


def test_scalar_plot_evaluator_requires_filter_columns(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    spec = FakeSpec(name="exp", r=2)
    write_scalar_metrics(paths, spec, "run-id", [{"metric": "err", "value": 1.0}])
    study_result = StudyResult(
        study_name="study",
        run_refs=[StudyRunRef(ident_id="run-id", variant_name="base", spec=spec)],
    )
    evaluator = ScalarPlotEvaluator(
        metrics=("err",),
        x="reduction.r",
        line_by=("metric",),
        panel_by=(),
        filters={"split": "TRAIN"},
    )

    with pytest.raises(ValueError, match="filter key 'split' not found"):
        evaluator.evaluate(study_result=study_result, paths=paths)


def test_build_array_plot_result_supports_custom_labels_and_line_modes() -> None:
    evaluator = ScalarPlotEvaluator(
        metrics=("err", "bound"),
        x="reduction.r",
        line_by=("metric",),
        panel_by=(),
        filters={},
        title=None,
        y_labels=("custom y",),
        line_mode_by=("__line_label__",),
        line_mode_map={("err",): "marker", ("bound",): "line+marker"},
    )
    df = pd.DataFrame(
        [
            {"metric": "bound", "x_value": 2, "value": 4.0},
            {"metric": "err", "x_value": 2, "value": 2.0},
            {"metric": "err", "x_value": 1, "value": 1.0},
        ]
    )

    plot_result = evaluator._build_array_plot_result(df)

    assert plot_result.name == "scalar_plot"
    assert plot_result.x_label == "r"
    assert plot_result.y_labels == ("custom y",)
    assert plot_result.line_labels == ("err", "bound")
    assert plot_result.subplot_titles is None
    assert plot_result.line_modes == ["marker", "line+marker"]
    np.testing.assert_allclose(plot_result.x, np.array([1, 2]))
    np.testing.assert_allclose(
        plot_result.y[:, :, 0],
        np.array([[1.0, np.nan], [2.0, 4.0]]),
    )


def test_build_array_plot_result_requires_value_and_x_columns() -> None:
    evaluator = ScalarPlotEvaluator(
        metrics=("err",),
        x="reduction.r",
        line_by=("metric",),
        panel_by=(),
        filters={},
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        evaluator._build_array_plot_result(pd.DataFrame({"metric": ["err"]}))


def test_line_mode_by_requires_columns_to_exist() -> None:
    evaluator = ScalarPlotEvaluator(
        metrics=("err",),
        x="reduction.r",
        line_by=("metric",),
        panel_by=(),
        filters={},
        line_mode_by=("source",),
        line_mode_map={("identified",): "marker"},
    )

    with pytest.raises(ValueError, match="line_mode_by columns"):
        evaluator._build_array_plot_result(
            pd.DataFrame([{"metric": "err", "x_value": 1, "value": 1.0}])
        )
