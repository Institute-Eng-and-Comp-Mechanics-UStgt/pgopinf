from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Any

from pgopinf.specs.base import SpecBase
from pgopinf.specs.system_analysis.base import (
    register_system_analysis_task,
)


@register_system_analysis_task("eigenvalues")
@dataclass(frozen=True)
class EigenvaluesSpec(SpecBase):
    """Specification for an eigenvalue analysis task."""

    kind: Literal["eigenvalues"] = "eigenvalues"
    which: str = "all"

    def build(self):
        """Build the eigenvalue analysis task."""
        from pgopinf.systems.analysis.eigenvalues import (
            EigenvaluesTask,
        )

        return EigenvaluesTask(which=self.which)

    def result_key(self) -> str:
        """Return the result key used in analysis bundles."""
        return "eigenvalues"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EigenvaluesSpec":
        """Deserialize an eigenvalue task spec."""
        return cls(which=str(d.get("which", "all")))


@register_system_analysis_task("minimal_realization")
@dataclass(frozen=True)
class MinimalRealizationSpec(SpecBase):
    """Specification for a minimal-realization analysis task."""

    kind: Literal["minimal_realization"] = "minimal_realization"
    trunc_tol: float = 1e-10

    def build(self):
        """Build the minimal-realization task."""
        from pgopinf.systems.analysis.minimal_realization import (
            MinimalRealizationTask,
        )

        return MinimalRealizationTask(trunc_tol=self.trunc_tol)

    def result_key(self) -> str:
        """Return the result key used in analysis bundles."""
        return "minimal_realization"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MinimalRealizationSpec":
        """Deserialize a minimal-realization task spec."""
        return cls(trunc_tol=float(d.get("trunc_tol", 1e-10)))


@register_system_analysis_task("sigma_plot")
@dataclass(frozen=True)
class SigmaPlotSpec(SpecBase):
    """Specification for a singular-value plot analysis task."""

    kind: Literal["sigma_plot"] = "sigma_plot"

    def build(self):
        """Build the singular-value plot task."""
        from pgopinf.systems.analysis.sigma_plot import (
            SigmaPlotTask,
        )

        return SigmaPlotTask()

    def result_key(self) -> str:
        """Return the result key used in analysis bundles."""
        return "sigma_plot"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SigmaPlotSpec":
        """Deserialize a singular-value plot task spec."""
        return cls()


@register_system_analysis_task("hinf_norm")
@dataclass(frozen=True)
class HInfNormSpec(SpecBase):
    """Specification for an H-infinity norm analysis task."""

    kind: Literal["hinf_norm"] = "hinf_norm"

    def build(self):
        """Build the H-infinity norm task."""
        from pgopinf.systems.analysis.hinf_norm import (
            HInfNormTask,
        )

        return HInfNormTask()

    def result_key(self) -> str:
        """Return the result key used in analysis bundles."""
        return "hinf_norm"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HInfNormSpec":
        """Deserialize an H-infinity norm task spec."""
        return cls()
