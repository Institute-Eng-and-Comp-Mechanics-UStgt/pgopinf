from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from pgopinf.evaluation.metrics.subroutines.hinf import hinf_norm
from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem


@dataclass
class HInfNormTask:
    """System-analysis task that computes the H-infinity norm."""

    @property
    def task_kind(self) -> str:
        """Task kind label.

        Returns
        -------
        str
            The string ``"hinf_norm"``.
        """
        return "hinf_norm"

    def compute(self, *, system: PHSystem | LTISystem) -> np.ndarray:
        """Compute the H-infinity norm of a system.

        Parameters
        ----------
        system : PHSystem or LTISystem
            System to analyze.

        Returns
        -------
        float
            H-infinity norm value.
        """
        hinf_norm_value = hinf_norm(first_system=system)
        return hinf_norm_value
