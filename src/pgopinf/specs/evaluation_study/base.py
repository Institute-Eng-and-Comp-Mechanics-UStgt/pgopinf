from __future__ import annotations

from typing import Any, Mapping, Protocol
from dataclasses import replace

from pgopinf.specs.registry import KindRegistry


class StudyEvaluationSpec(Protocol):
    """Protocol for study-level evaluation specifications."""

    kind: str
    name: str | None = None  # evaluation-specific

    def __post_init__(self):
        if not self.name:
            self = replace(self, name=self.kind)

    def build(self):
        """Build the runtime study evaluator."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the study-evaluation spec."""
        ...


STUDY_EVALUATION_REGISTRY: KindRegistry[StudyEvaluationSpec] = KindRegistry(
    "study_evaluation"
)


def register_study_evaluation(kind: str):
    """Register a study-evaluation spec kind."""
    return STUDY_EVALUATION_REGISTRY.register(kind)


def study_evaluation_from_dict(d: Mapping[str, Any]) -> StudyEvaluationSpec:
    """Deserialize a study-evaluation spec."""
    return STUDY_EVALUATION_REGISTRY.from_dict(d)


def study_evaluation_default_by_kind(kind: str) -> StudyEvaluationSpec:
    """Return the default study-evaluation spec for a kind."""
    return STUDY_EVALUATION_REGISTRY.default_by_kind(kind)
