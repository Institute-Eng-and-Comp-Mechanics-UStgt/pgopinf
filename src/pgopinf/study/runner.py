from __future__ import annotations

from dataclasses import dataclass

from pgopinf.core.experiment_runner import ExperimentRunner
from pgopinf.specs.study import StudySpec
from pgopinf.study.design import StudyDesigner
from pgopinf.study.result import StudyResult, StudyRunRef


@dataclass
class StudyRunner:
    """Run all concrete experiments generated from a study specification."""

    experiment_runner: ExperimentRunner
    designer: StudyDesigner

    def run(self, *, base_spec, study_spec: StudySpec) -> StudyResult:
        """Execute all experiments in a study.

        Parameters
        ----------
        base_spec
            Base experiment specification.
        study_spec : StudySpec
            Study specification containing variants and axes.

        Returns
        -------
        StudyResult
            References to all completed runs.
        """
        jobs = self.designer.build_experiments(
            base_spec=base_spec,
            study_spec=study_spec,
        )

        result = StudyResult(study_name=study_spec.name)

        for variant_name, spec_i in jobs:
            out = self.experiment_runner.run(spec_i)
            result.run_refs.append(
                StudyRunRef(
                    ident_id=out["run_id"],
                    spec=spec_i,
                    variant_name=variant_name,
                    run_paths=out.get("run_paths", None),
                )
            )

        return result
