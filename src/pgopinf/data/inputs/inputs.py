# yourpkg/inputs/runtime.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

UTCallable = Callable[
    [np.ndarray | float], np.ndarray
]  # returns shape (n_u,) or (n_u, n_t)


@dataclass
class Input:
    """
    Runtime input: a list of trajectories (simulations), each callable u_i(t).
    Each u_i(t) returns an array of shape (n_u,) for scalar t
    or (n_u, n_t) for array t (recommended).
    """

    u_list: list[UTCallable]

    def n_sim(self) -> int:
        """Number of input trajectories.

        Returns
        -------
        int
            Number of simulations represented by ``u_list``.
        """
        return len(self.u_list)

    def n_u(self) -> int:
        """Input dimension.

        Returns
        -------
        int
            Number of input channels returned by each callable.
        """
        return int(self.u_list[0](0.0).shape[0])

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        """Evaluate all input trajectories.

        Parameters
        ----------
        t : ndarray, shape (n_t,)
            Time points.

        Returns
        -------
        ndarray, shape (n_u, n_t, n_sim)
            Input values for all channels and simulations.

        Raises
        ------
        ValueError
            If a callable returns an incompatible shape.
        """
        t = np.asarray(t)
        n_u = self.n_u()
        n_t = t.size
        n_sim = self.n_sim()
        U = np.empty((n_u, n_t, n_sim), dtype=float)

        for i, u in enumerate(self.u_list):
            Ui = u(t)
            Ui = np.asarray(Ui, dtype=float)

            # Accept (n_u,) for scalar and broadcast, or (n_u, n_t)
            if Ui.ndim == 1:
                if n_t != 1:
                    # user returned vector for array t -> treat as constant in time
                    Ui = Ui[:, None] * np.ones((1, n_t))
                else:
                    Ui = Ui[:, None]
            if Ui.shape != (n_u, n_t):
                raise ValueError(
                    f"Input callable returned shape {Ui.shape}, expected {(n_u, n_t)}"
                )
            U[:, :, i] = Ui

        return U

    def evaluate_midpoints(self, t: np.ndarray) -> np.ndarray:
        """Evaluate inputs at midpoint times.

        Parameters
        ----------
        t : ndarray, shape (n_t,)
            Original time grid.

        Returns
        -------
        ndarray, shape (n_u, n_t - 1, n_sim)
            Input values at pairwise midpoint times.
        """
        t = np.asarray(t)
        t_mid = 0.5 * (t[1:] + t[:-1])
        return self.evaluate(t_mid)

    def plot(self, t: np.ndarray):
        """Plot all input trajectories.

        Parameters
        ----------
        t : ndarray, shape (n_t,)
            Time points at which to evaluate and plot the inputs.
        """
        import matplotlib.pyplot as plt

        U = self.evaluate(t)
        n_u, n_t, n_sim = U.shape
        for i in range(n_u):
            plt.figure()
            for j in range(n_sim):
                plt.plot(t, U[i, :, j], label=f"sim {j}")
            plt.title(f"Input u[{i}]")
            plt.xlabel("time")
            plt.ylabel(f"u[{i}]")
            plt.legend()
        plt.show(block=False)
        plt.savefig("input_plot.png")
