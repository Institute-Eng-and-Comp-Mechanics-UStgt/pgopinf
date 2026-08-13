from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pgopinf.data.dataset import DatasetArtifact
from pgopinf.identification.identifier import IdentificationResult
from pgopinf.reduction.reducer import ReductionResult
from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem


@dataclass(frozen=True)
class EvaluationContext:
    """Runtime context passed to evaluation metrics.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment being evaluated.
    run_id : str
        Identifier of the current run.
    original_system : LTISystem or PHSystem
        Full-order reference system.
    dataset_artifact : DatasetArtifact
        Full-order reference dataset.
    reduction_result : ReductionResult
        Reduction output used by the run.
    dataset_artifact_intrusive : DatasetArtifact or None
        Dataset generated from the intrusive reduced system.
    identification_result : IdentificationResult
        Identified system and diagnostics.
    dataset_artifact_identified : DatasetArtifact or None
        Dataset generated from the identified system.
    system_analysis : Any, optional
        Precomputed system-analysis bundle.
    """

    experiment_name: str
    run_id: str

    original_system: LTISystem | PHSystem

    dataset_artifact: DatasetArtifact
    reduction_result: ReductionResult
    dataset_artifact_intrusive: DatasetArtifact | None
    identification_result: IdentificationResult
    dataset_artifact_identified: DatasetArtifact | None

    system_analysis: Any | None = None
