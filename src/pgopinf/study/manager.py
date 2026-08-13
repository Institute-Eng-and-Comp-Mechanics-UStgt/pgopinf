from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from pgopinf.io.results_path import StudyResultsPath
from pgopinf.specs.study import StudySpec
from pgopinf.study.reporting import StudyReportWriter
from pgopinf.study.result import StudyResult
from pgopinf.study.runner import StudyRunner


@dataclass
class StudyManager:
    """Coordinate study execution, persistence, and study-level evaluation."""

    study_runner: StudyRunner
    report_writer: StudyReportWriter
    study_paths: StudyResultsPath

    def run_and_evaluate(self, *, base_spec, study_spec: StudySpec):
        """Run a study and render configured study-level evaluations.

        Parameters
        ----------
        base_spec
            Base experiment specification.
        study_spec : StudySpec
            Study specification to execute.

        Returns
        -------
        StudyResult
            Completed study result.
        """

        study_result = self.study_runner.run(
            base_spec=base_spec,
            study_spec=study_spec,
        )

        self.save_study_result_spec(study_result=study_result)

        for eval_spec in study_spec.evaluations:
            evaluator = eval_spec.build()

            if eval_spec.kind == "scalar_plot":
                self.report_writer.write_scalar_plot(
                    evaluator=evaluator,
                    study_result=study_result,
                    study_paths=self.study_paths,
                    evaluation_name=eval_spec.name,
                )

            elif eval_spec.kind == "array_plot":
                self.report_writer.write_array_plot(
                    evaluator=evaluator,
                    study_result=study_result,
                    study_paths=self.study_paths,
                    evaluation_name=eval_spec.name,
                )
            elif eval_spec.kind == "stability_fraction_by_group":
                self.report_writer.write_grouped_scalar_table(
                    evaluator=evaluator,
                    study_result=study_result,
                    study_paths=self.study_paths,
                    evaluation_name=eval_spec.name,
                )

        return study_result

    def save_study_result_spec(self, study_result: StudyResult):
        """Write the serialized study result to the study directory."""
        d = self.study_paths.study_dir
        (d / "spec.json").write_text(
            json.dumps(
                study_result.to_dict(),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
