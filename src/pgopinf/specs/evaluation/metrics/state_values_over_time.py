from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation.base import register_metric


@register_metric("state_values_over_time")
@dataclass(frozen=True)
class StateValuesOverTimeSpec(SpecBase):
    """Spec for state trajectory comparison."""

    kind: Literal["state_values_over_time"] = "state_values_over_time"
    split: Literal["TRAIN", "TEST"] = "TEST"
    target: Literal["identified", "intrusive"] = "identified"
    relative: bool = True

    def build(self):
        """Build the state trajectory metric."""
        from pgopinf.evaluation.metrics.state_values_over_time import (
            StateValuesOverTimeMetric,
        )

        return StateValuesOverTimeMetric(
            split=self.split,
            target=self.target,
            relative=self.relative,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StateValuesOverTimeSpec":
        """Deserialize the state trajectory metric spec."""
        return cls(
            split=str(d.get("split", "TEST")),  # type: ignore[arg-type]
            target=str(d.get("target", "identified")),  # type: ignore[arg-type]
            relative=bool(d.get("relative", True)),
        )
