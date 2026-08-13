from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SystemTaskArtifact:
    """
    Cached result of one system-analysis task.
    """

    task_key: str
    task_kind: str
    value: Any
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemAnalysisBundle:
    """
    Collection of individually cached task results.
    """

    task_ids: dict[str, str]
    values: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)
