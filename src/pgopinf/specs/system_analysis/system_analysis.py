from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.system_analysis.base import (
    system_analysis_task_from_dict,
)


@dataclass(frozen=True)
class SystemAnalysisSpec(SpecBase):
    """Specification for precomputed system-analysis tasks."""

    tasks: tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SystemAnalysisSpec":
        """Deserialize a system-analysis specification."""
        if d is None:
            return cls()
        else:
            tasks = tuple(system_analysis_task_from_dict(t) for t in d.get("tasks", []))
        return cls(tasks=tasks)
