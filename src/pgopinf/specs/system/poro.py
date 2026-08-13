from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

from .base import register_system

from pgopinf.specs.base import SpecBase
from pgopinf.systems.models.poro import poro


@register_system("poro")
@dataclass(frozen=True)
class PoroSpec(SpecBase):
    """Specification for the poroelastic benchmark system."""

    kind: Literal["poro"] = "poro"

    # parameters
    n_system: int = 980  # must be one of [320, 980, 1805]
    use_Berlin: bool = False  # whether to use the Berlin form with E=I
    use_mimo: bool = (
        True  # whether to use the MIMO version of the system (only for n_system=1805)
    )
    Rshift: float = 1e-3  # denoted eta in the manuscript AltMU21

    name: str = "poro"

    def build(self) -> Any:
        """
        Return the runtime system object.
        """
        return poro(
            n_system=self.n_system,
            use_Berlin=self.use_Berlin,
            use_mimo=self.use_mimo,
            Rshift=self.Rshift,
        )
