from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Literal

from pgopinf.specs.base import SpecBase
from pgopinf.specs.reduction.full_basis import (
    PODFullBasisSpec,
    FullBasisSpec,
    full_basis_from_dict,
)
from pgopinf.specs.reduction.test_basis import (
    TestBasisSpec,
    GalerkinTestBasisSpec,
    test_basis_from_dict,
)
from pgopinf.specs.reduction.mor import (
    MORSpec,
    LTIMORSpec,
    mor_from_dict,
)


@dataclass(frozen=True)
class ReductionSpec(SpecBase):
    """Specification for a model-reduction step."""

    kind: Literal["reduction"] = "reduction"

    r: int = 20
    full_basis: FullBasisSpec = field(default_factory=PODFullBasisSpec)
    test_basis: TestBasisSpec = field(default_factory=GalerkinTestBasisSpec)
    mor: MORSpec = field(default_factory=LTIMORSpec)

    project_data: bool = True
    reduce_system: bool = True

    def __post_init__(self):
        if self.r <= 0:
            raise ValueError("ReductionSpec.r must be > 0")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReductionSpec":
        """Deserialize a reduction specification."""
        return cls(
            r=int(d.get("r", 20)),
            full_basis=full_basis_from_dict(dict(d.get("full_basis", {"kind": "pod"}))),
            test_basis=test_basis_from_dict(
                dict(d.get("test_basis", {"kind": "galerkin"}))
            ),
            mor=mor_from_dict(dict(d.get("mor", {"kind": "lti"}))),
            project_data=bool(d.get("project_data", True)),
            reduce_system=bool(d.get("reduce_system", True)),
        )
