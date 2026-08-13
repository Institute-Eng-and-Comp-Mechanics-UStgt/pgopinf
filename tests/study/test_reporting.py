from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from pgopinf.evaluation.results import ArrayPlotResult
from pgopinf.io.results_path import ResultsPath
from pgopinf.study.reporting import StudyReportWriter
from pgopinf.study.result import StudyResult


class FakeEvaluator:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def render(self, **kwargs):
        self.calls.append(kwargs)


def make_plot_result() -> ArrayPlotResult:
    return ArrayPlotResult(
        name="plot",
        x=[0.0, 1.0],
        y=[[[1.0]], [[2.0]]],
        x_label="time",
        y_labels=("value",),
        line_labels=("line",),
        subplot_titles=("subplot",),
    )


def make_writer(default_outputs=object()) -> tuple[StudyReportWriter, FakeRenderer]:
    writer = StudyReportWriter(reporting_spec=SimpleNamespace(default_outputs=default_outputs))
    renderer = FakeRenderer()
    writer.array_plot_renderer = renderer
    return writer, renderer


def test_write_scalar_plot_saves_raw_csv_and_renders_plot(tmp_path) -> None:
    default_outputs = object()
    writer, renderer = make_writer(default_outputs)
    study_paths = ResultsPath(tmp_path).for_study(study_name="study")
    study_result = StudyResult(study_name="study")
    raw_df = pd.DataFrame({"metric": ["err"], "value": [0.25]})
    plot_result = make_plot_result()
    evaluator = FakeEvaluator((raw_df, plot_result))

    writer.write_scalar_plot(
        evaluator=evaluator,
        study_result=study_result,
        study_paths=study_paths,
        evaluation_name="scalar/eval",
    )

    out_dir = study_paths.evaluation_dir("scalar/eval")
    pd.testing.assert_frame_equal(
        pd.read_csv(out_dir / "scalar_plot_table.csv"),
        raw_df,
    )
    assert evaluator.calls == [{"study_result": study_result, "paths": study_paths.root}]
    assert renderer.calls == [
        {"result": plot_result, "out_dir": out_dir, "outputs": default_outputs}
    ]


def test_write_array_plot_saves_raw_pickle_and_renders_plot(tmp_path) -> None:
    default_outputs = object()
    writer, renderer = make_writer(default_outputs)
    study_paths = ResultsPath(tmp_path).for_study(study_name="study")
    study_result = StudyResult(study_name="study")
    raw_df = pd.DataFrame(
        {
            "variant": ["base"],
            "x": [[0.0, 1.0]],
            "y": [[1.0, 2.0]],
        }
    )
    plot_result = make_plot_result()
    evaluator = FakeEvaluator((raw_df, plot_result))

    writer.write_array_plot(
        evaluator=evaluator,
        study_result=study_result,
        study_paths=study_paths,
        evaluation_name="array plot",
    )

    out_dir = study_paths.evaluation_dir("array plot")
    pd.testing.assert_frame_equal(
        pd.read_pickle(out_dir / "array_plot_table.pkl"),
        raw_df,
    )
    assert evaluator.calls == [{"study_result": study_result, "paths": study_paths.root}]
    assert renderer.calls == [
        {"result": plot_result, "out_dir": out_dir, "outputs": default_outputs}
    ]


def test_write_grouped_scalar_table_saves_csv_without_rendering(tmp_path) -> None:
    writer, renderer = make_writer()
    study_paths = ResultsPath(tmp_path).for_study(study_name="study")
    study_result = StudyResult(study_name="study")
    df = pd.DataFrame(
        {
            "variant": ["base"],
            "fraction_type": ["stable"],
            "fraction": [1.0],
        }
    )
    evaluator = FakeEvaluator(df)

    writer.write_grouped_scalar_table(
        evaluator=evaluator,
        study_result=study_result,
        study_paths=study_paths,
        evaluation_name="stability",
    )

    out_dir = study_paths.evaluation_dir("stability")
    pd.testing.assert_frame_equal(
        pd.read_csv(out_dir / "stability_fraction_by_group.csv"),
        df,
    )
    assert evaluator.calls == [{"study_result": study_result, "paths": study_paths.root}]
    assert renderer.calls == []


def test_post_init_creates_default_array_plot_renderer() -> None:
    writer = StudyReportWriter(reporting_spec=SimpleNamespace(default_outputs=object()))

    assert writer.array_plot_renderer.__class__.__name__ == "ArrayPlotRenderer"
