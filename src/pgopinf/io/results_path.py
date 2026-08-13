from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _sanitize_name(name: str) -> str:
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class ResultsPath:
    """Path manager for repository result artifacts.

    Attributes
    ----------
    results_root : Path
        Root directory below which datasets, runs, studies, and cached
        artifacts are stored.
    """

    results_root: Path

    @property
    def runs_root(self) -> Path:
        """Directory containing run artifacts."""
        return _ensure_dir(self.results_root / "runs")

    @property
    def datasets_root(self) -> Path:
        """Directory containing full-order dataset artifacts."""
        return _ensure_dir(self.results_root / "datasets")

    @property
    def system_analysis_tasks(self) -> Path:
        """Directory containing cached system-analysis task artifacts."""
        return _ensure_dir(self.results_root / "system_analysis_tasks")

    @property
    def intrusive_datasets_root(self) -> Path:
        """Directory containing intrusive reduced dataset artifacts."""
        return _ensure_dir(self.results_root / "intrusive_datasets")

    @property
    def identified_datasets_root(self) -> Path:
        """Directory containing identified-system dataset artifacts."""
        return _ensure_dir(self.results_root / "identified_datasets")

    @property
    def full_basis_root(self) -> Path:
        """Directory containing full-basis artifacts."""
        return _ensure_dir(self.results_root / "full_bases")

    @property
    def inferred_q_matrix(self) -> Path:
        """Directory containing inferred Q-matrix artifacts."""
        return _ensure_dir(self.results_root / "inferred_q_matrix")

    @property
    def identifications_root(self) -> Path:
        """Directory containing identification artifacts."""
        return _ensure_dir(self.results_root / "identifications")

    @property
    def studies_root(self) -> Path:
        """Directory containing study artifacts."""
        return _ensure_dir(self.results_root / "studies")

    def for_run(self, *, experiment_name: str, ident_id: str) -> "RunResultsPath":
        """Create a path manager for one run."""
        return RunResultsPath(
            root=self,
            experiment_name=experiment_name,
            ident_id=ident_id,
        )

    def for_study(self, *, study_name: str) -> "StudyResultsPath":
        """Create a path manager for one study."""
        return StudyResultsPath(
            root=self,
            study_name=study_name,
        )

    def dataset_dir(self, dataset_id: str) -> Path:
        """Directory for a full-order dataset split."""
        return _ensure_dir(self.datasets_root / f"ds_{dataset_id}")

    def intrusive_dataset_dir(self, dataset_id: str) -> Path:
        """Directory for an intrusive reduced dataset split."""
        return _ensure_dir(self.intrusive_datasets_root / f"ds_{dataset_id}")

    def identified_dataset_dir(self, dataset_id: str) -> Path:
        """Directory for an identified-system dataset split."""
        return _ensure_dir(self.identified_datasets_root / f"ds_{dataset_id}")

    def full_bases_dir(self, full_basis_id: str) -> Path:
        """Directory for a full-basis artifact."""
        return _ensure_dir(self.full_basis_root / f"full_basis_{full_basis_id}")

    def identification_dir(self, identification_id: str) -> Path:
        """Directory for an identification artifact."""
        return _ensure_dir(
            self.identifications_root / f"identification_{identification_id}"
        )


@dataclass(frozen=True)
class RunResultsPath:
    """Path manager for one experiment run."""

    root: ResultsPath
    experiment_name: str
    ident_id: str

    @property
    def safe_experiment_name(self) -> str:
        """Filesystem-safe experiment name."""
        return _sanitize_name(self.experiment_name)

    @property
    def run_dir(self) -> Path:
        """Directory containing all artifacts for this run."""
        return _ensure_dir(
            self.root.runs_root / f"run_{self.safe_experiment_name}_{self.ident_id}"
        )

    @property
    def evaluation_dir(self) -> Path:
        """Directory containing evaluation outputs for this run."""
        return _ensure_dir(self.run_dir / "evaluation")

    @property
    def metric_summaries_csv(self) -> Path:
        """CSV file containing run-level metric summary rows."""
        return self.evaluation_dir / "metric_summaries.csv"

    @property
    def artifacts_dir(self) -> Path:
        """Directory containing rendered metric artifacts."""
        return _ensure_dir(self.evaluation_dir / "artifacts")

    @property
    def series_dir(self) -> Path:
        """Directory containing series metric artifacts."""
        return _ensure_dir(self.artifacts_dir / "series")

    @property
    def array_plots_dir(self) -> Path:
        """Directory containing array plot artifacts."""
        return _ensure_dir(self.artifacts_dir / "array_plots")

    @property
    def scalars_dir(self) -> Path:
        """Directory containing scalar metric artifacts."""
        return _ensure_dir(self.artifacts_dir / "scalars")

    @property
    def scalar_metrics_csv(self) -> Path:
        """CSV file containing scalar metric rows."""
        return self.scalars_dir / "scalar_metrics.csv"

    @property
    def tables_dir(self) -> Path:
        """Directory containing table artifacts."""
        return _ensure_dir(self.artifacts_dir / "tables")

    @property
    def figures_dir(self) -> Path:
        """Directory containing figure artifacts."""
        return _ensure_dir(self.artifacts_dir / "figures")


@dataclass(frozen=True)
class StudyResultsPath:
    """Path manager for one study."""

    root: ResultsPath
    study_name: str

    @property
    def safe_study_name(self) -> str:
        """Filesystem-safe study name."""
        return _sanitize_name(self.study_name)

    @property
    def study_dir(self) -> Path:
        """Directory containing all artifacts for this study."""
        return _ensure_dir(self.root.studies_root / self.safe_study_name)

    def evaluation_dir(self, evaluation_name: str) -> Path:
        """Directory for one study-level evaluation."""
        return _ensure_dir(self.study_dir / _sanitize_name(evaluation_name))

    @property
    def scalar_over_parameter_dir(self) -> Path:
        """Directory for scalar-over-parameter study outputs."""
        return _ensure_dir(self.study_dir / _sanitize_name("scalar_over_parameter"))
