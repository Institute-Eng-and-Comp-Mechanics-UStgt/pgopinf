from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

from pgopinf.systems.models.earth_atmosphere import earth_atmosphere

from .base import register_system

from pgopinf.specs.base import SpecBase


@register_system("earth_atmosphere")
@dataclass(frozen=True)
class EarthAtmosphereSpec(SpecBase):
    """Specification for the earth-atmosphere benchmark system."""

    kind: Literal["earth_atmosphere"] = "earth_atmosphere"

    name: str = "earth_atmosphere"

    def build(self) -> Any:
        """
        Return the runtime system object.
        """
        return earth_atmosphere()
