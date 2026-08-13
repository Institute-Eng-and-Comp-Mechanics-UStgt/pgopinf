from __future__ import annotations

from typing import Any, Mapping, Protocol

from pgopinf.systems.lti_system import LTISystem

from ..base import dataclass_to_dict
from ..registry import KindRegistry


class SystemSpec(Protocol):
    """Protocol for system model specifications."""

    kind: str

    def build(self) -> LTISystem:
        """Build the runtime system."""
        ...


SYSTEM_REGISTRY: KindRegistry[SystemSpec] = KindRegistry("system")


def register_system(kind: str):
    """Register a system spec kind."""
    return SYSTEM_REGISTRY.register(kind)


def system_from_dict(d: Mapping[str, Any]) -> SystemSpec:
    """Deserialize a system spec."""
    return SYSTEM_REGISTRY.from_dict(d)


def system_default_by_kind(kind: str) -> SystemSpec:
    """Return the default system spec for a kind."""
    return SYSTEM_REGISTRY.default_by_kind(kind)


def system_to_dict(spec: SystemSpec) -> dict[str, Any]:
    """Serialize a system spec."""
    return dataclass_to_dict(spec)
