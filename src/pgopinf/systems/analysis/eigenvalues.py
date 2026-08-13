from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem


@dataclass
class EigenvaluesTask:
    """System-analysis task that computes eigenvalues."""

    which: str = "all"

    @property
    def task_kind(self) -> str:
        """Task kind label.

        Returns
        -------
        str
            The string ``"eigenvalues"``.
        """
        return "eigenvalues"

    def compute(self, *, system: PHSystem | LTISystem) -> np.ndarray:
        """Compute eigenvalues of a system.

        Parameters
        ----------
        system : PHSystem or LTISystem
            System to analyze.

        Returns
        -------
        ndarray
            Eigenvalues returned by ``system.eigvals()``.
        """
        if self.which == "all":
            eigvals = system.eigvals()
        else:
            raise ValueError(f"Unsupported option for 'which': {self.which}")
        return eigvals
