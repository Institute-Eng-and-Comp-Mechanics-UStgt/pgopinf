from __future__ import annotations
from dataclasses import dataclass

from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem


@dataclass
class MinimalRealizationTask:
    """System-analysis task that computes a minimal realization."""

    trunc_tol: float = 1e-12

    @property
    def task_kind(self) -> str:
        """Task kind label.

        Returns
        -------
        str
            The string ``"minimal_realization"``.
        """
        return "minimal_realization"

    def compute(self, *, system: PHSystem | LTISystem) -> PHSystem:
        """Compute a minimal realization.

        Parameters
        ----------
        system : PHSystem or LTISystem
            System to reduce.

        Returns
        -------
        PHSystem
            Minimal realization returned by the system.
        """
        return system.minimal_realization(trunc_tol=self.trunc_tol)
