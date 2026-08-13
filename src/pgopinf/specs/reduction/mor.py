from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Mapping, Any, Protocol
from pgopinf.reduction.interfaces import MORProjector
from pgopinf.specs.base import SpecBase

from pgopinf.specs.registry import KindRegistry


class MORSpec(Protocol):
    """Protocol for MOR projector specifications."""

    kind: str

    def build(self) -> MORProjector:
        """Build the runtime MOR projector."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the MOR spec."""
        ...


MOR_REGISTRY: KindRegistry[MORSpec] = KindRegistry("mor")


def register_mor(kind: str):
    """Register an MOR spec kind."""
    return MOR_REGISTRY.register(kind)


def mor_default_by_kind(kind: str) -> MORSpec:
    """Return the default MOR spec for a kind."""
    return MOR_REGISTRY.default_by_kind(kind)


def mor_from_dict(d: Mapping[str, Any]) -> MORSpec:
    """Deserialize an MOR spec."""
    return MOR_REGISTRY.from_dict(d)


@register_mor("lti")
@dataclass(frozen=True)
class LTIMORSpec(SpecBase):
    """Specification for generic LTI projection."""

    kind: Literal["lti"] = "lti"

    def build(self):
        """Build the LTI MOR projector."""
        from pgopinf.reduction.mor import LTIMORProjector

        return LTIMORProjector()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LTIMORSpec":
        """Deserialize an LTI MOR spec."""
        return cls()


@register_mor("ph")
@dataclass(frozen=True)
class PHMORSpec(SpecBase):
    """Specification for structure-preserving pH projection."""

    kind: Literal["ph"] = "ph"
    ph_reduction_type: Literal["Berlin", "Gugercin"] = "Berlin"

    def build(self):
        """Build the pH MOR projector."""
        from pgopinf.reduction.mor import PHMORProjector

        return PHMORProjector(ph_reduction_type=self.ph_reduction_type)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PHMORSpec":
        """Deserialize a pH MOR spec."""
        return cls(ph_reduction_type=str(d.get("ph_reduction_type", "Berlin")))
