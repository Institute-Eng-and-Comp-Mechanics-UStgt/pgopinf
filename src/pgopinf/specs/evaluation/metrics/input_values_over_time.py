from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation.base import register_metric
from pgopinf.specs.evaluation.metrics.state_values_over_time import (
    StateValuesOverTimeSpec,
)


@register_metric("input_values_over_time")
@dataclass(frozen=True)
class InputValuesOverTimeSpec(SpecBase):
    """Spec for input trajectory plotting."""

    kind: Literal["input_values_over_time"] = "input_values_over_time"
    split: Literal["TRAIN", "TEST"] = "TEST"

    def build(self):
        """Build the input trajectory metric."""
        from pgopinf.evaluation.metrics.input_values_over_time import (
            InputValuesOverTimeMetric,
        )

        return InputValuesOverTimeMetric(
            split=self.split,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "InputValuesOverTimeSpec":
        """Deserialize the input trajectory metric spec."""
        return cls(
            split=str(d.get("split", "TEST")),  # type: ignore[arg-type]
        )
