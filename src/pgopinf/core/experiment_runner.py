from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

# Adjust imports to your package
from pgopinf.core.data_store import (
    DataStore,
    IdentifiedDataStore,
    IntrusiveDataStore,
)
from pgopinf.core.full_basis_store import (
    FullBasisStore,
)
from pgopinf.core.q_matrix_store import QMatrixStore
from pgopinf.core.reduction_store import ReductionStore
from pgopinf.core.identification_store import IdentificationStore

from pgopinf.core.task_store import SystemAnalysisTaskStore
from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.evaluator import MetricEvaluator
from pgopinf.evaluation.reporting.writer import ReportWriter
from pgopinf.io.results_path import ResultsPath
from pgopinf.specs.evaluation.reporting import (
    OutputSpec,
    ReportingSpec,
)
from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.systems.analysis.manager import SystemAnalysisManager

logger = logging.getLogger(__name__)


@dataclass
class RunnerPaths:
    """Filesystem roots used by an experiment runner.

    Attributes
    ----------
    results_root : Path
        Root directory for all generated experiment artifacts.
    """

    results_root: Path

    @property
    def datasets(self) -> Path:
        """Path to generated datasets.

        Returns
        -------
        Path
            Dataset artifact directory below ``results_root``.
        """
        return self.results_root / "datasets"

    @property
    def reductions(self) -> Path:
        """Path to generated reductions.

        Returns
        -------
        Path
            Reduction artifact directory below ``results_root``.
        """
        return self.results_root / "reductions"

    @property
    def runs(self) -> Path:
        """Path to experiment runs.

        Returns
        -------
        Path
            Run artifact directory below ``results_root``.
        """
        return self.results_root / "runs"


class ExperimentRunner:
    """
    Orchestrates:
      (SystemSpec + DataSpec) -> DataSet
      (dataset_id + ReductionSpec) -> ReducedDataSet
      (reduction_id + IdentificationSpec) -> RunResult
      + optional metrics
    """

    def __init__(
        self,
        *,
        paths: ResultsPath = ResultsPath(Path("results")),
        force_recompute: bool = False,
    ):
        """Initialize the experiment runner and its stores.

        Parameters
        ----------
        paths : ResultsPath, optional
            Result-directory manager used by all pipeline stages.
        force_recompute : bool, optional
            If ``True``, recompute cacheable artifacts in stores that support
            forced recomputation.
        """
        self.paths = paths
        self.force_recompute = force_recompute

        self.data_store = DataStore(
            paths=self.paths, force_recompute=self.force_recompute
        )
        self.intrusive_data_store = IntrusiveDataStore(
            paths=self.paths, force_recompute=self.force_recompute
        )
        self.reduction_store = ReductionStore(
            paths=self.paths,
            full_basis_store=FullBasisStore(
                paths=self.paths, force_recompute=self.force_recompute
            ),
            q_matrix_store=QMatrixStore(paths=self.paths),
        )
        self.system_analysis_manager = SystemAnalysisManager(
            task_store=SystemAnalysisTaskStore(
                paths=self.paths
            )  # might add force_recompute here later if needed
        )

        self.identification_store = IdentificationStore(
            paths=self.paths, force_recompute=self.force_recompute
        )
        self.identified_data_store = IdentifiedDataStore(
            paths=self.paths, force_recompute=self.force_recompute
        )
        self.metric_evaluator = MetricEvaluator()
        self.report_writer = ReportWriter(
            reporting_spec=ReportingSpec(
                save_summaries_csv=True,
                default_outputs=OutputSpec(csv=True, png=True, pdf=False),
            )
        )

        # self.ctx = PipelineContext(
        #     self.data_store,
        #     self.reduction_store,
        #     # self.run_store
        # )

    def run(self, spec: ExperimentSpec) -> dict[str, Any]:
        """
        spec: ExperimentSpec with at least:
          - spec.name
          - spec.system (SystemSpec)
          - spec.data (DataSpec)
          - spec.reduction (ReductionSpec)
          - spec.identification (IdentificationSpec)
          - spec.evaluation (EvaluationSpec)

        Returns a summary dict with ids and handles.
        """

        logger.info(f"Running experiment '{spec.name}'")

        # -----------------------
        # Get original system
        # -----------------------
        original_system = spec.system.build()
        # original_system.singular_value_plot(name=spec.system.name + "_sigma_plot.png")

        # -----------------------
        # (Optional) Precompute original system analysis (optional, but often fast and useful for metrics)
        # -----------------------
        system_analysis = None
        if spec.system_analysis is not None and len(spec.system_analysis.tasks) > 0:
            system_analysis = self.system_analysis_manager.get_or_create(
                system_spec=spec.system,
                system=original_system,
                analysis_spec=spec.system_analysis,
            )

        # -----------------------
        # Generate high-dimensional data
        # -----------------------
        dataset_artifact = self.data_store.get_or_create(
            system_spec=spec.system,
            system=original_system,
            data_spec=spec.data,
        )

        # -----------------------
        # Reduction: Projected data and reduced system
        # -----------------------
        reduction_result = self.reduction_store.get_or_create(
            dataset_artifact=dataset_artifact,
            reduction_spec=spec.reduction,
            system=original_system,
        )
        reduction_id = self.reduction_store.compute_id(
            data_train_id=dataset_artifact.train_id,
            reduction_spec=spec.reduction,
        )

        # -----------------------
        # Reduction: Intrusive data from reduced system
        # -----------------------
        dataset_artifact_intrusive = self.intrusive_data_store.get_or_create(
            reduction_id=reduction_id,
            system_spec=spec.system,
            data_spec=spec.data,
            reduction_spec=spec.reduction,
            reduction_result=reduction_result,
        )

        # -----------------------
        # Identification from projected data
        # -----------------------

        # identified system
        ident_id, identification_result = self.identification_store.get_or_create(
            reduction_id=reduction_id,
            reduction_result=reduction_result,
            identification_spec=spec.identification,
            original_system=original_system,
        )
        # data obtained from identified system
        dataset_artifact_identified = self.identified_data_store.get_or_create(
            reduction_id=reduction_id,
            system_spec=spec.system,
            data_spec=spec.data,
            reduction_spec=spec.reduction,
            identification_spec=spec.identification,
            identification_result=identification_result,
            reduction_result=reduction_result,
        )

        # -----------------------
        # (Optional) Metrics
        # -----------------------
        eval_ctx = EvaluationContext(
            experiment_name=spec.name,
            run_id=ident_id,
            original_system=original_system,
            system_analysis=system_analysis,
            dataset_artifact=dataset_artifact,
            reduction_result=reduction_result,
            dataset_artifact_intrusive=dataset_artifact_intrusive,
            identification_result=identification_result,
            dataset_artifact_identified=dataset_artifact_identified,
        )

        metric_outputs = self.metric_evaluator.evaluate(
            eval_ctx=eval_ctx,
            evaluation_spec=spec.evaluation,
        )

        # -----------------------
        # Save results and metrics
        # -----------------------
        run_paths = self.paths.for_run(
            experiment_name=spec.name,
            ident_id=ident_id,
        )
        # save spec to run
        spec_ids = {
            "dataset_train_id": dataset_artifact.train_id,
            "dataset_test_id": dataset_artifact.test_id,
            "dataset_intrusive_train_id": dataset_artifact_intrusive.train.id,
            "dataset_intrusive_test_id": dataset_artifact_intrusive.test.id,
            "dataset_identified_train_id": dataset_artifact_identified.train.id,
            "dataset_identified_test_id": dataset_artifact_identified.test.id,
            "reduction_id": reduction_id,
            "identification_id": ident_id,
        }
        spec.save(run_paths.run_dir, spec_ids=spec_ids)
        # render results and metrics to report
        self.report_writer.write(
            run_paths=run_paths,
            metric_outputs=metric_outputs,
        )

        return {
            "run_id": ident_id,
            "spec": spec,
            "metric_outputs": metric_outputs,
            "run_paths": run_paths,
        }
