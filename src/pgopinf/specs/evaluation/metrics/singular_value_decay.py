from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pgopinf.evaluation.metrics.singular_value_decay import (
    SingularValueDecayMetric,
)
from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation.base import register_metric


@register_metric("singular_value_decay")
@dataclass(frozen=True)
class SingularValueDecaySpec(SpecBase):
    """Spec for singular-value decay plotting."""

    kind: Literal["singular_value_decay_over_r"] = "singular_value_decay"

    def build(self):
        """Build the singular-value decay metric."""
        from pgopinf.evaluation.metrics.singular_value_decay import (
            SingularValueDecayMetric,
        )

        return SingularValueDecayMetric()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SingularValueDecayMetric":
        """Deserialize the singular-value decay metric spec."""
        return cls()
