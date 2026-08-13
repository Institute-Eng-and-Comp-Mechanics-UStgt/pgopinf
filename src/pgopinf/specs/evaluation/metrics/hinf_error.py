from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation.base import register_metric


@register_metric("hinf_error")
@dataclass(frozen=True)
class HInfErrorSpec(SpecBase):
    """Spec for the H-infinity error metric."""

    kind: Literal["hinf_error"] = "hinf_error"
    target: Literal["identified", "intrusive"] = "identified"
    tol: float = 1e-10
    stabilized: bool = False
    use_minimal_realization: bool = False
    relative: bool = True

    def build(self):
        """Build the H-infinity error metric."""
        from pgopinf.evaluation.metrics.hinf_error import (
            HInfErrorMetric,
        )

        return HInfErrorMetric(
            target=self.target,
            tol=self.tol,
            stabilized=self.stabilized,
            use_minimal_realization=self.use_minimal_realization,
            relative=self.relative,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HInfErrorSpec":
        """Deserialize the H-infinity error metric spec."""
        return cls(
            target=str(d.get("target", "identified")),
            tol=float(d.get("tol", 1e-10)),
            stabilized=bool(d.get("stabilized", False)),
            use_minimal_realization=bool(d.get("use_minimal_realization", False)),
            relative=bool(d.get("relative", True)),
        )
