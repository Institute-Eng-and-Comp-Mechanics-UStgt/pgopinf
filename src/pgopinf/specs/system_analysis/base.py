from __future__ import annotations

from typing import Any, Mapping, Protocol

from pgopinf.specs.registry import KindRegistry


class SystemAnalysisTaskSpec(Protocol):
    """Protocol for system-analysis task specifications."""

    kind: str

    def build(self):
        """Build the runtime analysis task."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the task spec."""
        ...

    def result_key(self) -> str:
        """
        Key under which the task result is stored in the bundle.
        Example: 'eigenvalues', 'minimal_realization'
        """
        ...


SYSTEM_ANALYSIS_TASK_REGISTRY: KindRegistry[SystemAnalysisTaskSpec] = KindRegistry(
    "system_analysis_task"
)


def register_system_analysis_task(kind: str):
    """Register a system-analysis task kind."""
    return SYSTEM_ANALYSIS_TASK_REGISTRY.register(kind)


def system_analysis_task_from_dict(d: Mapping[str, Any]) -> SystemAnalysisTaskSpec:
    """Deserialize a system-analysis task spec."""
    return SYSTEM_ANALYSIS_TASK_REGISTRY.from_dict(d)


def system_analysis_task_default_by_kind(kind: str) -> SystemAnalysisTaskSpec:
    """Return the default system-analysis task spec for a kind."""
    return SYSTEM_ANALYSIS_TASK_REGISTRY.default_by_kind(kind)
