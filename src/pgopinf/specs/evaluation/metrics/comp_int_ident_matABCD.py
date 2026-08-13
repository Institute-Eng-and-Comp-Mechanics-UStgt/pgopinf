from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation.base import register_metric


@register_metric("comp_int_ident_matABCD")
@dataclass(frozen=True)
class CompIntIdentMatABCDSpec(SpecBase):
    """Spec for comparing intrusive and identified state-space matrices."""

    kind: Literal["comp_int_ident_matABCD"] = "comp_int_ident_matABCD"
    relative: bool = True
    rel_threshold: float = 1e-8

    def build(self):
        """Build the matrix-comparison metric."""
        from pgopinf.evaluation.metrics.comp_int_ident_matABCD import (
            CompIntIdentMatABCDMetric,
        )

        return CompIntIdentMatABCDMetric(
            relative=self.relative,
            rel_threshold=self.rel_threshold,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CompIntIdentMatABCDSpec":
        """Deserialize the matrix-comparison metric spec."""
        return cls(
            relative=bool(d.get("relative", True)),
            rel_threshold=float(d.get("rel_threshold", 1e-8)),
        )
