from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation.base import MetricSpec, metric_from_dict


@dataclass(frozen=True)
class EvaluationSpec(SpecBase):
    """Specification for per-run evaluation metrics."""

    metrics: tuple[MetricSpec, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EvaluationSpec":
        """Deserialize an evaluation specification."""
        metrics = tuple(metric_from_dict(m) for m in d.get("metrics", []))
        return cls(metrics=metrics)

    @classmethod
    def preset(cls, name: str) -> EvaluationSpec:
        """Create an evaluation spec from a named preset."""
        from pgopinf.specs.presets.evaluation_presets import (
            evaluation_presets,
        )

        return evaluation_presets(name)

    def __add__(self, other):
        return EvaluationSpec(metrics=self.metrics + other.metrics)
