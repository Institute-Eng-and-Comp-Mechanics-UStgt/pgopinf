from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Literal, Protocol

from pgopinf.specs.base import SpecBase
from pgopinf.specs.registry import KindRegistry


class FullBasisSpec(Protocol):
    """Protocol for full-basis specifications."""

    kind: str

    def build(self) -> Any:
        """Build the runtime full-basis builder."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full-basis spec."""
        ...


FULL_BASIS_REGISTRY: KindRegistry[FullBasisSpec] = KindRegistry("full_basis")


def register_full_basis(kind: str):
    """Register a full-basis spec kind."""
    return FULL_BASIS_REGISTRY.register(kind)


def full_basis_default_by_kind(kind: str) -> FullBasisSpec:
    """Return the default full-basis spec for a kind."""
    return FULL_BASIS_REGISTRY.default_by_kind(kind)


def full_basis_from_dict(d: Mapping[str, Any]) -> FullBasisSpec:
    """Deserialize a full-basis spec."""
    return FULL_BASIS_REGISTRY.from_dict(d)


@register_full_basis("pod")
@dataclass(frozen=True)
class PODFullBasisSpec(SpecBase):
    """Specification for a POD full basis."""

    kind: Literal["pod"] = "pod"
    full_matrices: bool = True

    def build(self):
        """Build a POD full-basis builder."""
        from pgopinf.reduction.full_basis.pod import (
            PODFullBasisBuilder,
        )

        return PODFullBasisBuilder(full_matrices=self.full_matrices)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PODFullBasisSpec":
        """Deserialize a POD full-basis spec."""
        return cls(full_matrices=bool(d.get("full_matrices", True)))


@register_full_basis("modal")
@dataclass(frozen=True)
class ModalFullBasisSpec(SpecBase):
    """Specification for a modal full basis."""

    kind: Literal["modal"] = "modal"
    compute_left: bool = True

    def build(self):
        """Build a modal full-basis builder."""
        from pgopinf.reduction.full_basis.modal import (
            ModalFullBasisBuilder,
        )

        return ModalFullBasisBuilder(compute_left=self.compute_left)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModalFullBasisSpec":
        """Deserialize a modal full-basis spec."""
        return cls(compute_left=bool(d.get("compute_left", True)))
