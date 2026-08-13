from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation.base import register_metric
from pgopinf.specs.evaluation.metrics.state_values_over_time import (
    StateValuesOverTimeSpec,
)


@register_metric("output_values_over_time")
@dataclass(frozen=True)
class OutputValuesOverTimeSpec(SpecBase):
    """Spec for output trajectory comparison."""

    kind: Literal["output_values_over_time"] = "output_values_over_time"
    split: Literal["TRAIN", "TEST"] = "TEST"
    target: Literal["identified", "intrusive"] = "identified"
    relative: bool = True

    def build(self):
        """Build the output trajectory metric."""
        from pgopinf.evaluation.metrics.output_values_over_time import (
            OutputValuesOverTimeMetric,
        )

        return OutputValuesOverTimeMetric(
            split=self.split,
            target=self.target,
            relative=self.relative,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OutputValuesOverTimeSpec":
        """Deserialize the output trajectory metric spec."""
        return cls(
            split=str(d.get("split", "TEST")),  # type: ignore[arg-type]
            target=str(d.get("target", "identified")),  # type: ignore[arg-type]
            relative=bool(d.get("relative", True)),
        )
