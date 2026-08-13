from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Literal
import numpy as np

from pgopinf.data.data import Data


@dataclass
class ReducedData(Data):
    """
    Same as Data, but semantically represents reduced state trajectories.

    X: (r, n_t, n_sim)
    U: (n_u, n_t, n_sim) or None
    Y: (n_y, n_t, n_sim) or None
    V: (n, r) or None

    """

    V: np.ndarray | None = None
    meta: Literal["projected", "integrated"] = "projected"

    def __post_init__(self):
        super().__post_init__()  # derivative + validation
        if self.V is None:
            self.r = self.X.shape[0]
            return

        if self.V.ndim != 2:
            raise ValueError(f"V must have shape (n, r), got {self.V.shape}")

        self.r = self.V.shape[1]
        if int(self.r) != int(self.X.shape[0]):
            raise ValueError(
                f"ReducedData.r={self.r} but X has first dim {self.X.shape[0]}"
            )

    @property
    def X_full(self):
        """Lift reduced state trajectories to the full state space.

        Returns
        -------
        ndarray or None, shape (n, n_t, n_sim)
            Full-order state trajectories, or ``None`` if no basis is stored.
        """
        return np.einsum("ij,jtk->itk", self.V, self.X) if self.V is not None else None

    @property
    def dXdt_full(self):
        """Lift reduced state derivatives to the full state space.

        Returns
        -------
        ndarray or None, shape (n, n_t, n_sim)
            Full-order derivative trajectories, or ``None`` if no basis or
            derivative data is stored.
        """
        return (
            np.einsum("ij,jtk->itk", self.V, self.dXdt)
            if self.V is not None and self.dXdt is not None
            else None
        )
