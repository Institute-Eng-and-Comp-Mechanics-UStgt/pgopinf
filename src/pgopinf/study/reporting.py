from __future__ import annotations

from dataclasses import dataclass, field

from pgopinf.evaluation.reporting.render_series import (
    ArrayPlotRenderer,
)
from pgopinf.io.results_path import StudyResultsPath
from pgopinf.specs.evaluation.reporting import ReportingSpec
from pgopinf.study.array_plot_evaluator import ArrayPlotStudyEvaluator
from pgopinf.study.evaluator import StudyEvaluator
from pgopinf.study.result import StudyResult


@dataclass
class StudyReportWriter:
    """Write study-level evaluation tables and plots."""

    reporting_spec: ReportingSpec = field(default_factory=ReportingSpec)

    def __post_init__(self):
        self.array_plot_renderer = ArrayPlotRenderer()

    def write_scalar_plot(
        self,
        *,
        evaluator: StudyEvaluator,
        study_result: StudyResult,
        study_paths: StudyResultsPath,
        evaluation_name: str,
    ) -> None:
        """Write a scalar-plot study evaluation."""
        out_dir = study_paths.evaluation_dir(evaluation_name)

        raw_df, plot_result = evaluator.evaluate(
            study_result=study_result,
            paths=study_paths.root,
        )

        raw_df.to_csv(out_dir / "scalar_plot_table.csv", index=False)

        self.array_plot_renderer.render(
            result=plot_result,
            out_dir=out_dir,
            outputs=self.reporting_spec.default_outputs,
        )

    def write_array_plot(
        self,
        *,
        evaluator: ArrayPlotStudyEvaluator,
        study_result: StudyResult,
        study_paths: StudyResultsPath,
        evaluation_name: str,
    ) -> None:
        """Write an array-plot study evaluation."""
        out_dir = study_paths.evaluation_dir(evaluation_name)

        raw_df, plot_result = evaluator.evaluate(
            study_result=study_result,
            paths=study_paths.root,
        )

        raw_df.to_pickle(out_dir / "array_plot_table.pkl")

        self.array_plot_renderer.render(
            result=plot_result,
            out_dir=out_dir,
            outputs=self.reporting_spec.default_outputs,
        )

    def write_grouped_scalar_table(
        self,
        *,
        evaluator,
        study_result,
        study_paths,
        evaluation_name: str,
    ) -> None:
        """Write a grouped scalar table study evaluation."""
        out_dir = study_paths.evaluation_dir(evaluation_name)
        df = evaluator.evaluate(
            study_result=study_result,
            paths=study_paths.root,
        )
        df.to_csv(out_dir / "stability_fraction_by_group.csv", index=False)
