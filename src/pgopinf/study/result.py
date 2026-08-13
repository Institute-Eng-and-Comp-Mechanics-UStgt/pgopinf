from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pgopinf.io.results_path import RunResultsPath


@dataclass(frozen=True)
class StudyRunRef:
    """Reference to one completed run in a study.

    Attributes
    ----------
    ident_id : str
        Identification/run identifier.
    spec : Any
        Concrete experiment specification used for the run.
    variant_name : str
        Study variant that produced the run.
    run_paths : RunResultsPath, optional
        Paths for the run artifacts.
    """

    ident_id: str
    spec: Any
    variant_name: str
    run_paths: RunResultsPath | None = None


@dataclass
class StudyResult:
    """Collection of completed runs for one study."""

    study_name: str
    run_refs: list[StudyRunRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the study result."""
        return {
            "study_name": self.study_name,
            "run_refs": [
                {
                    "ident_id": run_ref.ident_id,
                    "variant_name": run_ref.variant_name,
                    "spec": run_ref.spec.to_dict(),
                }
                for run_ref in self.run_refs
            ],
        }
