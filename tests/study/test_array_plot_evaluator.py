from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pgopinf.io.results_path import ResultsPath
from pgopinf.study.array_plot_evaluator import (
    ArrayPlotStudyEvaluator,
    _get_dotted_attr,
    _unique_in_order,
)
from pgopinf.study.result import StudyResult, StudyRunRef


@dataclass(frozen=True)
class FakeSpec:
    name: str
    r: int
    mode: str = "train"

    @property
    def reduction(self):
        return SimpleNamespace(r=self.r)

    def to_dict(self):
        return {"name": self.name, "reduction": {"r": self.r}, "mode": self.mode}


def make_evaluator(**kwargs) -> ArrayPlotStudyEvaluator:
    defaults = {
        "artifact_name": "states",
        "artifact_filters": {},
        "subplot_indices": (0,),
        "line_labels": (),
        "variants": (),
        "line_by": ("variant", "source_line"),
        "filters": {},
        "title": None,
    }
    defaults.update(kwargs)
    return ArrayPlotStudyEvaluator(**defaults)


def write_array_artifact(
    paths: ResultsPath,
    *,
    spec: FakeSpec,
    ident_id: str,
    name: str = "states",
    stem: str = "states_train",
    x_label: str = "time",
    x: tuple[float, ...] = (0.0, 1.0, 2.0),
    line_labels: tuple[str, ...] = ("original", "identified"),
    subplot_titles: tuple[str, ...] = ("x0", "x1"),
    y_by_column: dict[str, tuple[float, ...]] | None = None,
    meta: dict | None = None,
    include_csv: bool = True,
    include_json: bool = True,
) -> None:
    out_dir = paths.for_run(experiment_name=spec.name, ident_id=ident_id).array_plots_dir
    artifact_meta = meta or {"split": "TRAIN"}
    index_path = out_dir / "index.json"
    index_path.write_text(
        json.dumps([{"name": name, "stem": stem, "meta": artifact_meta}]),
        encoding="utf-8",
    )

    if include_csv:
        if y_by_column is None:
            y_by_column = {
                "x0__original": (1.0, 2.0, 3.0),
                "x0__identified": (1.5, 2.5, 3.5),
                "x1__original": (4.0, 5.0, 6.0),
                "x1__identified": (4.5, 5.5, 6.5),
            }
        pd.DataFrame({x_label: x, **y_by_column}).to_csv(out_dir / f"{stem}.csv", index=False)

    if include_json:
        payload = {
            "name": name,
            "x_label": x_label,
            "line_labels": list(line_labels),
            "subplot_titles": list(subplot_titles),
            "y_labels": ["value"] * len(subplot_titles),
            "n_lines": len(line_labels),
            "n_subplots": len(subplot_titles),
            "meta": artifact_meta,
        }
        (out_dir / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_helper_functions_extract_values_and_preserve_unique_order() -> None:
    obj = SimpleNamespace(a=SimpleNamespace(b=3))

    assert _get_dotted_attr(obj, "a.b") == 3
    assert _unique_in_order(["a", "b", "a", "c"]) == ["a", "b", "c"]


def test_collect_needed_spec_paths_includes_filters_and_style_columns() -> None:
    evaluator = make_evaluator(
        filters={"reduction.r": 2},
        line_by=("variant", "spec:mode", "artifact:split"),
        line_mode_by=("spec:reduction.r",),
        linestyle_by=("source_line", "spec:mode"),
    )

    assert evaluator._collect_needed_spec_paths() == {"reduction.r", "mode"}


def test_run_matches_filters_uses_dotted_spec_paths() -> None:
    evaluator = make_evaluator(filters={"reduction.r": 2, "mode": "train"})

    assert evaluator._run_matches_filters(FakeSpec(name="a", r=2, mode="train")) is True
    assert evaluator._run_matches_filters(FakeSpec(name="b", r=3, mode="train")) is False


def test_resolve_matching_artifacts_filters_index_entries_and_requires_files(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    spec = FakeSpec(name="exp", r=2)
    write_array_artifact(paths, spec=spec, ident_id="run-id", meta={"split": "TRAIN"})
    write_array_artifact(
        paths,
        spec=spec,
        ident_id="other",
        stem="missing_json",
        meta={"split": "TRAIN"},
        include_json=False,
    )
    evaluator = make_evaluator(artifact_filters={"split": "TRAIN"})

    matches = evaluator._resolve_matching_artifacts(
        paths.for_run(experiment_name=spec.name, ident_id="run-id").array_plots_dir
    )

    assert len(matches) == 1
    assert matches[0]["stem"] == "states_train"
    assert matches[0]["artifact_meta"] == {"split": "TRAIN"}
    assert make_evaluator(artifact_filters={"split": "TEST"})._resolve_matching_artifacts(
        paths.for_run(experiment_name=spec.name, ident_id="run-id").array_plots_dir
    ) == []


def test_column_name_uses_plain_line_label_for_single_subplot() -> None:
    evaluator = make_evaluator()

    assert (
        evaluator._column_name(
            subplot_title="x0",
            source_line_label="original",
            single_subplot=True,
        )
        == "original"
    )
    assert (
        evaluator._column_name(
            subplot_title="x0",
            source_line_label="original",
            single_subplot=False,
        )
        == "x0__original"
    )


def test_evaluate_combines_selected_artifact_lines_subplots_and_styles(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    spec_r2 = FakeSpec(name="exp-r2", r=2)
    spec_r4 = FakeSpec(name="exp-r4", r=4)
    write_array_artifact(paths, spec=spec_r2, ident_id="id-r2", meta={"split": "TRAIN"})
    write_array_artifact(paths, spec=spec_r4, ident_id="id-r4", meta={"split": "TRAIN"})
    write_array_artifact(
        paths,
        spec=FakeSpec(name="exp-test", r=8),
        ident_id="id-test",
        meta={"split": "TEST"},
    )
    study_result = StudyResult(
        study_name="study",
        run_refs=[
            StudyRunRef(ident_id="id-r2", variant_name="base", spec=spec_r2),
            StudyRunRef(ident_id="id-r4", variant_name="petrov", spec=spec_r4),
            StudyRunRef(
                ident_id="id-test",
                variant_name="base",
                spec=FakeSpec(name="exp-test", r=8),
            ),
        ],
    )
    evaluator = make_evaluator(
        artifact_filters={"split": "TRAIN"},
        subplot_indices=(0, 1),
        line_labels=("identified",),
        variants=("base", "petrov"),
        line_by=("variant", "source_line", "spec:reduction.r", "artifact:split"),
        filters={},
        title="State comparison",
        line_mode_by=("variant",),
        line_mode_map={("base",): "marker", ("petrov",): "line+marker"},
        linestyle_by=("variant",),
        linestyle_map={("base",): "--", ("petrov",): ":"},
    )

    raw_df, plot_result = evaluator.evaluate(study_result=study_result, paths=paths)

    assert raw_df.shape[0] == 4
    assert raw_df["variant"].tolist() == ["base", "base", "petrov", "petrov"]
    assert raw_df["subplot_title"].tolist() == ["x0", "x1", "x0", "x1"]
    assert raw_df["source_line"].tolist() == ["identified"] * 4
    assert raw_df["spec:reduction.r"].tolist() == [2, 2, 4, 4]
    assert raw_df["artifact:split"].tolist() == ["TRAIN"] * 4

    assert plot_result.name == "State comparison"
    np.testing.assert_allclose(plot_result.x, np.array([0.0, 1.0, 2.0]))
    assert plot_result.x_label == "time"
    assert plot_result.line_labels == (
        "base | identified | 2 | TRAIN",
        "petrov | identified | 4 | TRAIN",
    )
    assert plot_result.subplot_titles == ("x0", "x1")
    assert plot_result.line_modes == ("marker", "line+marker")
    assert plot_result.linestyles == ("--", ":")
    np.testing.assert_allclose(
        plot_result.y[:, :, 0],
        np.array([[1.5, 1.5], [2.5, 2.5], [3.5, 3.5]]),
    )
    np.testing.assert_allclose(
        plot_result.y[:, :, 1],
        np.array([[4.5, 4.5], [5.5, 5.5], [6.5, 6.5]]),
    )
    assert plot_result.meta == {
        "study_plot_kind": "array_plot",
        "artifact_name": "states",
        "artifact_filters": {"split": "TRAIN"},
        "line_by": ["variant", "source_line", "spec:reduction.r", "artifact:split"],
        "filters": {},
    }


def test_evaluate_returns_empty_result_when_nothing_matches(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    study_result = StudyResult(
        study_name="study",
        run_refs=[StudyRunRef(ident_id="id", variant_name="base", spec=FakeSpec(name="exp", r=2))],
    )
    evaluator = make_evaluator(title="No matches")

    raw_df, plot_result = evaluator.evaluate(study_result=study_result, paths=paths)

    assert raw_df.empty
    assert plot_result.name == "No matches"
    assert plot_result.y.shape == (0, 0, 1)
    assert plot_result.meta == {"study_plot_kind": "array_plot", "empty": True}


def test_evaluate_supports_single_subplot_artifact_columns(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    spec = FakeSpec(name="exp", r=2)
    write_array_artifact(
        paths,
        spec=spec,
        ident_id="run-id",
        line_labels=("original", "identified"),
        subplot_titles=("only",),
        y_by_column={
            "original": (1.0, 2.0, 3.0),
            "identified": (4.0, 5.0, 6.0),
        },
    )
    study_result = StudyResult(
        study_name="study",
        run_refs=[StudyRunRef(ident_id="run-id", variant_name="base", spec=spec)],
    )
    evaluator = make_evaluator(line_labels=("original",), line_by=("source_line",))

    raw_df, plot_result = evaluator.evaluate(study_result=study_result, paths=paths)

    assert raw_df.shape[0] == 1
    assert raw_df.iloc[0]["subplot_title"] == "only"
    assert plot_result.line_labels == ("original",)
    np.testing.assert_allclose(plot_result.y[:, 0, 0], np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("key_kind", ["line", "style"])
def test_unsupported_line_or_style_keys_raise(key_kind) -> None:
    evaluator = make_evaluator(line_by=("unsupported",))
    row = pd.Series({"variant": "base", "source_line": "original"})

    with pytest.raises(ValueError, match="Unsupported"):
        if key_kind == "line":
            evaluator._line_key(row)
        else:
            evaluator._style_key(row, ("unsupported",))


def test_build_array_plot_result_requires_identical_x_grids() -> None:
    evaluator = make_evaluator(line_by=("variant",))
    df = pd.DataFrame(
        [
            {
                "variant": "base",
                "source_line": "original",
                "subplot_index": 0,
                "subplot_title": "x0",
                "x_label": "time",
                "x": np.array([0.0, 1.0]),
                "y": np.array([1.0, 2.0]),
            },
            {
                "variant": "petrov",
                "source_line": "original",
                "subplot_index": 0,
                "subplot_title": "x0",
                "x_label": "time",
                "x": np.array([0.0, 2.0]),
                "y": np.array([3.0, 4.0]),
            },
        ]
    )

    with pytest.raises(ValueError, match="identical x grids"):
        evaluator._build_array_plot_result(df)


def test_resolve_style_values_returns_none_without_style_map() -> None:
    evaluator = make_evaluator()
    df = pd.DataFrame([{"variant": "base", "source_line": "original"}])

    assert (
        evaluator._resolve_style_values(
            df,
            [("base", "original")],
            style_by=None,
            style_map={("base",): "marker"},
            default="line",
        )
        is None
    )
