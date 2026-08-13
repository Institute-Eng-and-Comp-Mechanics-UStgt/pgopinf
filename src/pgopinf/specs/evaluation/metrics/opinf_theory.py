from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation.base import register_metric


@register_metric("opinf_theory")
@dataclass(frozen=True)
class OpInfTheorySpec(SpecBase):
    """Spec for operator-inference theory diagnostics."""

    kind: Literal["opinf_theory"] = "opinf_theory"
    rcond: float = 1e-12

    def build(self):
        """Build the operator-inference diagnostics metric."""
        from pgopinf.evaluation.metrics.opinf_theory import (
            OpInfTheoryMetric,
        )

        return OpInfTheoryMetric(
            rcond=self.rcond,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "OpInfTheorySpec":
        """Deserialize the operator-inference diagnostics spec."""
        return cls(
            rcond=float(d.get("rcond", 1e-12)),
        )
