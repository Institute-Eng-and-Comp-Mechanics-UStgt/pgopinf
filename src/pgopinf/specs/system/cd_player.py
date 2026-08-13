from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

from .base import register_system

from pgopinf.specs.base import SpecBase
from pgopinf.systems.models.cd_player import cd_player


@register_system("cd_player")
@dataclass(frozen=True)
class CDPlayerSpec(SpecBase):
    """Specification for the CD-player benchmark system."""

    kind: Literal["cd_player"] = "cd_player"

    name: str = "cd_player"

    def build(self) -> Any:
        """
        Return the runtime system object.
        """
        return cd_player()
