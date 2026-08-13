from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

from .base import register_system

from pgopinf.specs.base import SpecBase
from pgopinf.systems.models.msd import mass_spring_damper_system


@register_system("msd")
@dataclass(frozen=True)
class MassSpringDamperSpec(SpecBase):
    """Specification for the mass-spring-damper benchmark system."""

    kind: Literal["msd"] = "msd"

    # parameters
    n_mass: int = 3
    m: float = 4.0
    c: float = 1.0
    k: float = 4.0
    input_vals: tuple[int, ...] = (0, 1)
    form: Literal["lti", "ph"] = "lti"
    use_Berlin: bool = True

    name: str = "mass_spring_damper"

    def build(self) -> Any:
        """
        Return the runtime system object.
        """

        return mass_spring_damper_system(
            n_mass=self.n_mass,
            mass_vals=self.m,
            damp_vals=self.c,
            stiff_vals=self.k,
            input_vals=self.input_vals,
            system_type=self.form,
            use_Berlin=self.use_Berlin,
        )
