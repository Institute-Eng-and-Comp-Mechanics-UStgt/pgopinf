from __future__ import annotations

import numpy as np
import logging

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from pgopinf.data.time.time import Time

logger = logging.getLogger(__name__)


@dataclass
class Data:
    """
    Runtime data for one split (TRAIN or TEST).

    Shapes:
      time.t: (n_t,)
      ic.x0:  (n_x, n_sim)
      U:      (n_u, n_t, n_sim) or None
      X:      (n_x, n_t, n_sim)
      Y:      (n_y, n_t, n_sim) or None
    """

    time: Any  # your Time runtime object
    X: np.ndarray  # (n_x, n_t, n_sim)
    U: np.ndarray  # (n_u, n_t, n_sim)
    Y: np.ndarray  # (n_y, n_t, n_sim)

    # derived / optional
    dXdt: Optional[np.ndarray] = None  # shape depends on method, see below
    deriv_method: Optional[str] | None = "gradient"  # "gradient" or "midpoints"

    # arbitrary metadata (seed, split name, generator version, etc.)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # calculate dXdt
        if self.dXdt is None:
            self.compute_derivative()

        # Basic validation (cheap but catches many bugs)
        t = np.asarray(self.time.t)
        if t.ndim != 1:
            raise ValueError("time.t must be 1D")

        X = np.asarray(self.X)
        if X.ndim != 3:
            raise ValueError("X must have shape (n_x, n_t, n_sim)")

        n, n_t, n_sim = X.shape
        if t.shape[0] != n_t:
            raise ValueError(f"X has n_t={n_t} but time.t has {t.shape[0]}")

        if self.U is not None:
            U = np.asarray(self.U)
            if U.ndim != 3 or U.shape[1] != n_t or U.shape[2] != n_sim:
                raise ValueError(
                    f"U must have shape (n_u, {n_t}, {n_sim}), got {U.shape}"
                )

        if self.Y is not None:
            Y = np.asarray(self.Y)
            if Y.ndim != 3 or Y.shape[1] != n_t or Y.shape[2] != n_sim:
                raise ValueError(
                    f"Y must have shape (n_y, {n_t}, {n_sim}), got {Y.shape}"
                )

    def compute_derivative(self):
        """Compute or transform state derivatives in place.

        The ``"gradient"`` method computes derivatives on the existing grid.
        The ``"midpoints"`` method computes finite differences between
        neighboring states and shifts all trajectories to midpoint times.

        Raises
        ------
        ValueError
            If ``deriv_method`` is unknown.
        """
        if self.deriv_method == "gradient":
            self.dXdt = np.gradient(self.X, self.dt, axis=1)
        elif self.deriv_method == "midpoints":
            t_mid = 0.5 * (self.time.t[1:] + self.time.t[:-1])
            self.dXdt = (self.X[:, 1:, :] - self.X[:, :-1, :]) / self.dt
            self.X = 1 / 2 * (self.X[:, 1:, :] + self.X[:, :-1, :])
            if self.U is not None:
                self.U = 1 / 2 * (self.U[:, 1:, :] + self.U[:, :-1, :])
            if self.Y is not None:
                self.Y = 1 / 2 * (self.Y[:, 1:, :] + self.Y[:, :-1, :])
            self.time = Time(t_mid)
        else:
            raise ValueError(f"Unknown deriv_method {self.deriv_method}")

    # ----------------------------
    # Convenience properties
    # ----------------------------
    @property
    def n(self) -> int:
        """State dimension.

        Returns
        -------
        int
            Number of state variables.
        """
        return int(self.X.shape[0])

    @property
    def n_t(self) -> int:
        """Number of time points.

        Returns
        -------
        int
            Number of time samples per trajectory.
        """
        return int(self.X.shape[1])

    @property
    def n_sim(self) -> int:
        """Number of simulated trajectories.

        Returns
        -------
        int
            Number of simulations.
        """
        return int(self.X.shape[2])

    @property
    def n_u(self) -> int:
        """Input dimension.

        Returns
        -------
        int
            Number of input channels, or zero if no input data is present.
        """
        return int(self.U.shape[0]) if self.U is not None else 0

    @property
    def n_y(self) -> int:
        """Output dimension.

        Returns
        -------
        int
            Number of output channels, or zero if no output data is present.
        """
        return int(self.Y.shape[0]) if self.Y is not None else 0

    @property
    def dt(self) -> float:
        """Time step.

        Returns
        -------
        float
            Constant spacing of the time grid.
        """
        return float(self.time.dt)

    @property
    def t(self) -> np.ndarray:
        """Time grid.

        Returns
        -------
        ndarray, shape (n_t,)
            Time samples.
        """
        return np.asarray(self.time.t)

    @property
    def x(self) -> np.ndarray:
        """State data in sample format.

        Returns
        -------
        ndarray, shape (n, n_t * n_sim)
            State snapshots with trajectories concatenated.
        """
        x, _, _, _ = self.convert_to_sample_format()
        return x

    @property
    def u(self) -> np.ndarray:
        """Input data in sample format.

        Returns
        -------
        ndarray or None, shape (n_u, n_t * n_sim)
            Input snapshots with trajectories concatenated.
        """
        _, u, _, _ = self.convert_to_sample_format()
        return u

    @property
    def y(self) -> np.ndarray:
        """Output data in sample format.

        Returns
        -------
        ndarray or None, shape (n_y, n_t * n_sim)
            Output snapshots with trajectories concatenated.
        """
        _, _, y, _ = self.convert_to_sample_format()
        return y

    @property
    def dxdt(self) -> np.ndarray:
        """Derivative data in sample format.

        Returns
        -------
        ndarray or None, shape (n, n_t * n_sim)
            State derivative snapshots with trajectories concatenated.
        """
        _, _, _, dxdt = self.convert_to_sample_format()
        return dxdt

    def convert_to_sample_format(self, test_reshape=False):
        """Return trajectory arrays in sample format.

        Parameters
        ----------
        test_reshape : bool, optional
            If ``True``, print a small reshape demonstration.

        Returns
        -------
        tuple
            ``(x, u, y, dxdt)`` with each non-``None`` array reshaped from
            ``(dimension, n_t, n_sim)`` to ``(dimension, n_t * n_sim)`` using
            Fortran order.
        """
        x, u, y, dxdt = self.convert_XUY_to_sample_format(
            self.X, self.U, self.Y, self.dXdt
        )

        if test_reshape:
            # Note the we use Fortran order to keep one trajectory after another. MWE:
            array = np.array([[1, 2, 3], [4, 5, 6]]).T[
                np.newaxis, :
            ]  # n=1,n_t=3,n_sim=2
            array_reshape_fortran = np.reshape(array, (1, 3 * 2), order="F")
            array_reshape_default = np.reshape(array, (1, 3 * 2))
            print("First three entries (should be [1,2,3]):")
            print(f"Fortran order: {array_reshape_fortran[:,:3]}")
            print(f"Default order: {array_reshape_default[:,:3]}")

        return x, u, y, dxdt

    @staticmethod
    def convert_XUY_to_sample_format(X, U, Y, dXdt=None):
        """Convert trajectory arrays to sample format.

        Parameters
        ----------
        X : ndarray, shape (n, n_t, n_sim)
            State trajectories.
        U : ndarray or None, shape (n_u, n_t, n_sim)
            Input trajectories.
        Y : ndarray or None, shape (n_y, n_t, n_sim)
            Output trajectories.
        dXdt : ndarray or None, shape (n, n_t, n_sim), optional
            State derivative trajectories.

        Returns
        -------
        tuple
            ``(x, u, y, dxdt)`` in sample format.
        """
        n = X.shape[0]
        n_t = X.shape[1]
        n_sim = X.shape[2]
        x = np.reshape(X, (n, n_t * n_sim), order="F")
        if U is not None:
            n_u = U.shape[0]
            u = np.reshape(U, (n_u, n_t * n_sim), order="F")
        else:
            u = None
        if Y is not None:
            n_y = Y.shape[0]
            y = np.reshape(Y, (n_y, n_t * n_sim), order="F")
        else:
            y = None
        if dXdt is not None:
            dxdt = np.reshape(dXdt, (n, n_t * n_sim), order="F")
        else:
            dxdt = None
        return x, u, y, dxdt

    def convert_to_time_step_format(self, test_reshape=False):
        """Convert internal arrays from sample format to time-step format.

        Parameters
        ----------
        test_reshape : bool, optional
            If ``True``, print a small reshape demonstration.
        """
        self.X, self.U, self.Y, self.dXdt = self.convert_xuy_to_time_step_format(
            self.x,
            self.u,
            self.y,
            self.dxdt,
            self.n,
            self.n_u,
            self.n_y,
            self.n_t,
            self.n_sim,
        )
        if test_reshape:
            # Note the we use Fortran order to keep one trajectory after another. MWE:
            array = np.array([[1, 2, 3, 4, 5, 6]]).T[np.newaxis, :]  # n=1,n_t=3,n_sim=2
            array_reshape_fortran = np.reshape(array, (1, 3, 2), order="F")
            array_reshape_default = np.reshape(array, (1, 3, 2))
            print("First three entries (should be [1,2,3]):")
            print(f"Fortran order: {array_reshape_fortran[:,:,0]}")
            print(f"Default order: {array_reshape_default[:,:,0]}")

    @staticmethod
    def convert_xuy_to_time_step_format(x, u, y, dxdt, n, n_u, n_y, n_t, n_sim):
        """Convert sample-format arrays to trajectory format.

        Parameters
        ----------
        x : ndarray, shape (n, n_t * n_sim)
            State samples.
        u : ndarray or None, shape (n_u, n_t * n_sim)
            Input samples.
        y : ndarray or None, shape (n_y, n_t * n_sim)
            Output samples.
        dxdt : ndarray or None, shape (n, n_t * n_sim)
            State derivative samples.
        n : int
            State dimension.
        n_u : int
            Input dimension.
        n_y : int
            Output dimension.
        n_t : int
            Number of time points.
        n_sim : int
            Number of simulations.

        Returns
        -------
        tuple
            ``(X, U, Y, dXdt)`` in trajectory format.
        """
        X = np.reshape(x, (n, n_t, n_sim), order="F")
        U = np.reshape(u, (n_u, n_t, n_sim), order="F") if u is not None else None
        Y = np.reshape(y, (n_y, n_t, n_sim), order="F") if y is not None else None
        dXdt = (
            np.reshape(dxdt, (n, n_t, n_sim), order="F") if dxdt is not None else None
        )
        return X, U, Y, dXdt

    def reduce_samples(
        self,
        n: int = 2,
        mode: str = "every_nth",
    ):
        """Subsample trajectories in time.

        Parameters
        ----------
        n : int, optional
            Keep every ``n``-th sample.
        mode : {"every_nth"}, optional
            Subsampling mode.

        Raises
        ------
        ValueError
            If ``mode`` is unknown.
        """
        # if self.samples_reduced == True:
        #     logging.info(f"Samples have already been reduced once. Skipping...")
        #     return
        # else:
        if isinstance(n, float):
            n = int(n)

        if mode == "every_nth":
            logger.info(f"Reduce samples by using only every {n}-th entry of the data.")
            self.X = self.X[:, ::n, :]
            if self.U is not None:
                self.U = self.U[:, ::n, :]
            if self.Y is not None:
                self.Y = self.Y[:, ::n, :]
            if self.dXdt is not None:
                self.dXdt = self.dXdt[:, ::n, :]
            t = self.t[::n]
            self.time = Time(t)

        else:
            raise ValueError(f"Unknown mode {mode} to reduce samples.")
