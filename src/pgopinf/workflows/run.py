"""Execution helpers for configured experiments and studies."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pgopinf.io.results_path import ResultsPath
from pgopinf.specs.presets.run_recipes import (
    RunConfig,
    build_experiment_spec,
    build_study_spec,
)

if TYPE_CHECKING:
    from pgopinf.core.experiment_runner import ExperimentRunner
    from pgopinf.specs.experiment import ExperimentSpec
    from pgopinf.specs.study import StudySpec
    from pgopinf.study.manager import StudyManager


def run(config: RunConfig):
    """Run the experiment or study described by a run configuration.

    Parameters
    ----------
    config : RunConfig
        Configuration containing the selected preset names, runtime options,
        and run mode.

    Returns
    -------
    object
        Result returned by the experiment runner or study manager.
    """
    exp_spec = build_experiment_spec(config)

    if config.mode == "single":
        return run_experiment_spec(
            exp_spec,
            results_root=config.results_root,
            force_recompute=config.force_recompute,
        )

    if config.mode == "study":
        study_spec = build_study_spec(config)
        return run_study_specs(
            exp_spec,
            study_spec,
            results_root=config.results_root,
            force_recompute=config.force_recompute,
        )

    raise ValueError(f"Unknown run mode: {config.mode!r}")


def run_experiment_spec(
    exp_spec: "ExperimentSpec",
    *,
    results_root: Path = Path("results"),
    force_recompute: bool = False,
):
    """Run a single experiment specification.

    Parameters
    ----------
    exp_spec : ExperimentSpec
        Experiment specification to execute.
    results_root : pathlib.Path, optional
        Root directory where experiment artifacts are stored.
    force_recompute : bool, optional
        If ``True``, recompute results even when cached artifacts exist.

    Returns
    -------
    object
        Result returned by
        :meth:`pgopinf.core.experiment_runner.ExperimentRunner.run`.
    """
    from pgopinf.core.experiment_runner import ExperimentRunner

    experiment_runner = ExperimentRunner(
        paths=ResultsPath(results_root),
        force_recompute=force_recompute,
    )
    return experiment_runner.run(exp_spec)


def run_study_specs(
    exp_spec: "ExperimentSpec",
    study_spec: "StudySpec",
    *,
    results_root: Path = Path("results"),
    force_recompute: bool = False,
    study_name: str | None = None,
):
    """Run and evaluate a study for a base experiment specification.

    Parameters
    ----------
    exp_spec : ExperimentSpec
        Base experiment specification used by the study.
    study_spec : StudySpec
        Study specification defining parameter variations and evaluation.
    results_root : pathlib.Path, optional
        Root directory where study artifacts are stored.
    force_recompute : bool, optional
        If ``True``, recompute experiment results even when cached artifacts
        exist.
    study_name : str, optional
        Name of the study result directory. If omitted, the experiment and
        study specification names are combined.

    Returns
    -------
    object
        Result returned by
        :meth:`pgopinf.study.manager.StudyManager.run_and_evaluate`.
    """
    from pgopinf.core.experiment_runner import ExperimentRunner

    paths = ResultsPath(results_root)
    experiment_runner = ExperimentRunner(
        paths=paths,
        force_recompute=force_recompute,
    )
    manager = build_study_manager(
        paths=paths,
        experiment_runner=experiment_runner,
        study_name=study_name or f"{exp_spec.name}_{study_spec.name}",
    )
    return manager.run_and_evaluate(base_spec=exp_spec, study_spec=study_spec)


def build_study_manager(
    *,
    paths: ResultsPath,
    experiment_runner: "ExperimentRunner",
    study_name: str,
) -> "StudyManager":
    """Build the default study manager used by run workflows.

    Parameters
    ----------
    paths : ResultsPath
        Result path helper rooted at the selected output directory.
    experiment_runner : ExperimentRunner
        Experiment runner used to execute the base and varied experiments.
    study_name : str
        Name of the study result directory.

    Returns
    -------
    StudyManager
        Study manager with the default designer, runner, and report writer.
    """
    from pgopinf.specs.evaluation.reporting import (
        OutputSpec,
        ReportingSpec,
    )
    from pgopinf.study.design import StudyDesigner
    from pgopinf.study.manager import StudyManager
    from pgopinf.study.reporting import StudyReportWriter
    from pgopinf.study.runner import StudyRunner

    return StudyManager(
        study_runner=StudyRunner(
            experiment_runner=experiment_runner,
            designer=StudyDesigner(),
        ),
        report_writer=StudyReportWriter(
            reporting_spec=ReportingSpec(
                save_summaries_csv=True,
                default_outputs=OutputSpec(),
            )
        ),
        study_paths=paths.for_study(study_name=study_name),
    )
