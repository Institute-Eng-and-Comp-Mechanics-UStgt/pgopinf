from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation.base import register_metric


@register_metric("spectral_abscissa")
@dataclass(frozen=True)
class SpectralAbscissaSpec(SpecBase):
    """Spec for the spectral abscissa metric."""

    kind: Literal["spectral_abscissa"] = "spectral_abscissa"
    target: Literal["identified", "intrusive", "original"] = "identified"
    tol: float = 1e-10
    stabilized: bool = False

    def build(self):
        """Build the spectral abscissa metric."""
        from pgopinf.evaluation.metrics.spectral_abscissa import (
            SpectralAbscissaMetric,
        )

        return SpectralAbscissaMetric(
            target=self.target,
            tol=self.tol,
            stabilized=self.stabilized,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SpectralAbscissaSpec":
        """Deserialize the spectral abscissa metric spec."""
        return cls(
            target=str(d.get("target", "identified")),
            tol=float(d.get("tol", 1e-10)),
            stabilized=bool(d.get("stabilized", False)),
        )
