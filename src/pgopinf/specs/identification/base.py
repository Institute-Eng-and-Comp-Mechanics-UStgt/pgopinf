from __future__ import annotations
from typing import Any, Mapping, Protocol

from pgopinf.identification.identifier import Identifier
from pgopinf.specs.registry import KindRegistry


class IdentificationSpec(Protocol):
    """Protocol for identification method specifications."""

    kind: str

    def build(self) -> Identifier:
        """Build the runtime identifier."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the identification spec."""
        ...


IDENTIFICATION_REGISTRY: KindRegistry[IdentificationSpec] = KindRegistry(
    "identification"
)


def register_identification(kind: str):
    """Register an identification spec kind."""
    return IDENTIFICATION_REGISTRY.register(kind)


def identification_default_by_kind(kind: str) -> IdentificationSpec:
    """Return the default identification spec for a kind."""
    return IDENTIFICATION_REGISTRY.default_by_kind(kind)


def identification_from_dict(d: Mapping[str, Any]) -> IdentificationSpec:
    """Deserialize an identification spec."""
    return IDENTIFICATION_REGISTRY.from_dict(d)
