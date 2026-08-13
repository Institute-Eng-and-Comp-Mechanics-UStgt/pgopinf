from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pgopinf.identification.identifier import (
    OperatorInferenceIdentifier,
)
from pgopinf.specs.base import SpecBase
from pgopinf.specs.identification.base import register_identification


@register_identification("convex_ph_inference")
@dataclass(frozen=True)
class ConvexPHInferenceSpec(SpecBase):
    """Specification for convex port-Hamiltonian inference."""

    kind: Literal["convex_ph_inference"] = "convex_ph_inference"
    J_is_known: bool = False
    add_regularization: bool = False
    add_dissipation_inequality_cost: bool = False
    lambdas: float = 1e-6
    no_feedthrough: bool = False
    solver: str = "mosek"
    project_psd: bool = False
    return_system_type: Literal["ph", "lti"] = "ph"
    accept_unknown_mosek: bool = False

    def build(self) -> OperatorInferenceIdentifier:
        """Build the convex pH identifier."""
        from pgopinf.identification.identifier import (
            ConvexPortHamiltonianIdentifier,
        )

        return ConvexPortHamiltonianIdentifier(
            J_is_known=self.J_is_known,
            add_regularization=self.add_regularization,
            add_dissipation_inequality_cost=self.add_dissipation_inequality_cost,
            lambdas=self.lambdas,
            no_feedthrough=self.no_feedthrough,
            solver=self.solver,
            project_psd=self.project_psd,
            return_system_type=self.return_system_type,
            accept_unknown_mosek=self.accept_unknown_mosek,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ConvexPHInferenceSpec":
        """Deserialize a convex pH inference spec."""
        return cls(
            J_is_known=bool(d.get("J_is_known", False)),
            add_regularization=bool(d.get("add_regularization", False)),
            add_dissipation_inequality_cost=bool(
                d.get("add_dissipation_inequality_cost", False)
            ),
            lambdas=float(d.get("lambdas", 1e-6)),
            no_feedthrough=bool(d.get("no_feedthrough", False)),
            solver=str(d.get("solver", "mosek")),
            project_psd=bool(d.get("project_psd", False)),
            return_system_type=str(d.get("return_system_type", "ph")),
            accept_unknown_mosek=bool(d.get("accept_unknown_mosek", False)),
        )
