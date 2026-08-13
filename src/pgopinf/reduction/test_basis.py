from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any, Literal
import numpy as np
import logging
from tqdm import tqdm


from pgopinf.data.hamiltonian_identification import (
    hamiltonian_identification,
    hamiltonian_identification_cvx,
    hamiltonian_identification_lsmr,
)
from pgopinf.identification.subroutines.transformations import (
    solve_Riccati,
)

logger = logging.getLogger(__name__)


class TestBasis(Protocol):
    """Protocol for strategies that construct a test basis."""

    def make_W(
        self, *, V: np.ndarray, system=None, data_train=None, **kwargs
    ) -> np.ndarray:
        """Construct a test basis for a trial basis."""
        ...


def _as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x


def _check_shapes(V: np.ndarray, W: np.ndarray) -> None:
    V = _as2d(V)
    W = _as2d(W)
    if V.shape != W.shape:
        raise ValueError(
            f"V and W must have same shape (n,r). Got V={V.shape}, W={W.shape}"
        )


def biorthonormalize(W: np.ndarray, V: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Adjust W so that W^T V = I (approximately), via:
        W <- W (W^T V)^(-T)

    This is a standard post-processing step for Petrov–Galerkin.

    Assumes (W^T V) is invertible.
    """
    V = _as2d(V)
    W = _as2d(W)
    M = W.T @ V  # (r,r)
    # if np.linalg.cond(M) > 1 / eps:
    #     raise np.linalg.LinAlgError(
    #         f"W^T V is ill-conditioned (cond={np.linalg.cond(M):.2e})."
    #     )
    # W_new = W @ inv(M).T
    return W @ np.linalg.inv(M).T


@dataclass
class GalerkinTestBasis:
    """Galerkin test-basis strategy using ``W = V``."""

    def make_W(
        self, *, V: np.ndarray, system=None, data_train=None, **kwargs
    ) -> np.ndarray:
        """Return the trial basis as the test basis.

        Parameters
        ----------
        V : ndarray, shape (n, r)
            Trial basis.
        system
            Unused; accepted for the test-basis protocol.
        data_train
            Unused; accepted for the test-basis protocol.
        **kwargs
            Additional unused options.

        Returns
        -------
        ndarray, shape (n, r)
            Test basis equal to ``V``.
        """
        return np.asarray(V)


@dataclass
class QVTestBasis:
    """
    Test basis of the form W = Q @ V.

    Supported Q sources:
    - "system_Q": use system.Q
    - "system_E": use system.E
    - "Hamiltonian": identify Q from Hamiltonian values on training data
    - "kyp": compute Q from a Riccati / KYP solve

    Notes
    -----
    - Expensive Q computations are intended to be cached outside this class
      via QMatrixStore.
    - The public protocol remains:
          make_W(V=..., system=..., data_train=..., **kwargs)
      and accepts an optional cached Q via kwargs["Q"].
    """

    source: Literal[
        "system_Q",
        "system_E",
        "Hamiltonian",
        "kyp",
        "Hamiltonian_cvx",
        "Hamiltonian_red",
    ] = "system_Q"
    enforce_WtV_I: bool = False
    eps: float = 1e-12

    # only relevant for source="Hamiltonian"
    project_hamiltonian_Q_spsd: bool = True

    def q_cache_options(self) -> dict[str, Any]:
        """
        Options that affect Q itself.
        Do not include options that only affect the cheap W = QV step.
        """
        return {
            "source": self.source,
            "project_hamiltonian_Q_spsd": self.project_hamiltonian_Q_spsd,
        }

    def compute_Q(self, *, system=None, data_train=None, V=None) -> np.ndarray:
        """
        Compute the full Q matrix.

        This may be expensive depending on `source`.
        Callers should cache the result externally if needed.
        """
        if self.source == "system_Q":
            if system is None or not hasattr(system, "Q"):
                raise ValueError("QVTestBasis with source='system_Q' requires system.Q")

            Q = system.Q

        elif self.source == "system_E":
            if system is None or not hasattr(system, "E"):
                raise ValueError("QVTestBasis with source='system_E' requires system.E")
            Q = system.E

        elif self.source == "Hamiltonian":
            if system is None:
                raise ValueError(
                    "QVTestBasis with source='Hamiltonian' requires system"
                )
            if not hasattr(system, "Q") or not hasattr(system, "E"):
                raise ValueError(
                    "QVTestBasis with source='Hamiltonian' requires system with Q and E"
                )
            if data_train is None:
                raise ValueError(
                    "QVTestBasis with source='Hamiltonian' requires data_train"
                )

            # We need sample-format states data_train.x of shape (n, n_samples)
            if not hasattr(data_train, "x"):
                if hasattr(data_train, "convert_to_sample_format"):
                    data_train.convert_to_sample_format()
                else:
                    raise ValueError(
                        "QVTestBasis with source='Hamiltonian' requires data_train.x "
                        "or a convert_to_sample_format() method"
                    )

            x = data_train.x  # shape (n, n_samples)

            # Hamiltonian values H_i = 1/2 x_i^T E^T Q x_i
            # Here the 'true' Hamiltonian values are computed from the original system.
            # H = np.zeros(x.shape[1])
            # if x.shape[1] > 1e4:
            #     logger.info(f"Build true Hamiltonian...")
            #     # with progress bar for large datasets
            #     for i in tqdm(range(x.shape[1])):
            #         H[i] = 0.5 * x[:, i].T @ system.E.T @ system.Q @ x[:, i]
            # else:
            #     for i in range(x.shape[1]):
            #         H[i] = 0.5 * x[:, i].T @ system.E.T @ system.Q @ x[:, i]

            # Einsum is faster for large datasets
            H = 0.5 * np.einsum("ij,ij->j", x, system.E.T @ system.Q @ x)

            if data_train.x.shape[1] > 1e4:
                # for n_samples > 1e4, the dense least-squares matrix becomes too large to form explicitly;
                # do not precompute dense data matrix; use LSMR algorithm
                logger.info(
                    f"Using LSMR for Hamiltonian identification with {data_train.x.shape[1]} samples"
                )
                Q = hamiltonian_identification_lsmr(
                    x=x,
                    Ham=H,
                    project=self.project_hamiltonian_Q_spsd,
                    damp=0.0,
                    atol=1e-10,
                    btol=1e-10,
                    # maxiter=200,
                )[0]
            else:
                logger.info(
                    f"Using least-squares for Hamiltonian identification with {data_train.x.shape[1]} samples"
                )
                Q = hamiltonian_identification(
                    x,
                    H,
                    project=self.project_hamiltonian_Q_spsd,
                )

        elif self.source.lower() == "hamiltonian_red":
            if system is None:
                raise ValueError(
                    "QVTestBasis with source='Hamiltonian' requires system"
                )
            if not hasattr(system, "Q") or not hasattr(system, "E"):
                raise ValueError(
                    "QVTestBasis with source='Hamiltonian' requires system with Q and E"
                )
            if data_train is None:
                raise ValueError(
                    "QVTestBasis with source='Hamiltonian' requires data_train"
                )
            if V is None:
                raise ValueError(
                    "QVTestBasis with source='Hamiltonian_red' requires V for reduction"
                )
            x = data_train.x  # shape (n, n_samples)
            H = 0.5 * np.einsum("ij,ij->j", x, system.E.T @ system.Q @ x)
            logger.info(
                f"Using least-squares for reducedHamiltonian identification with {data_train.x.shape[1]} samples"
            )
            Q = hamiltonian_identification(
                x,
                H,
                project=self.project_hamiltonian_Q_spsd,
                V=V,
            )

        elif self.source.lower() == "hamiltonian_cvx":
            if system is None:
                raise ValueError(
                    "QVTestBasis with source='Hamiltonian_cvx' requires system"
                )
            if not hasattr(system, "Q") or not hasattr(system, "E"):
                raise ValueError(
                    "QVTestBasis with source='Hamiltonian_cvx' requires system with Q and E"
                )
            if data_train is None:
                raise ValueError(
                    "QVTestBasis with source='Hamiltonian_cvx' requires data_train"
                )
            if V is None:
                raise ValueError(
                    "QVTestBasis with source='Hamiltonian_cvx' requires V for reduction"
                )
            x = data_train.x  # shape (n, n_samples)
            H = 0.5 * np.einsum("ij,ij->j", x, system.E.T @ system.Q @ x)
            logger.info(
                f"Using convex optimization for Hamiltonian identification with {data_train.x.shape[1]} samples"
            )
            Q = hamiltonian_identification_cvx(
                x,
                H,
                project=self.project_hamiltonian_Q_spsd,
            )

            if hasattr(system, "Q"):
                logger.info(
                    f"Relative error in identified Q: {np.linalg.norm(Q - system.Q) / np.linalg.norm(system.Q)}"
                )

        elif self.source == "kyp":
            if system is None:
                raise ValueError("QVTestBasis with source='kyp' requires system")

            Q = solve_Riccati(system)

        else:
            raise ValueError(f"Invalid source {self.source!r} for QVTestBasis")

        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be square 2D matrix, got shape {Q.shape}")
        return Q

    def make_W(
        self,
        *,
        V: np.ndarray,
        system=None,
        data_train=None,
        **kwargs,
    ) -> np.ndarray:
        """
        Build W = QV.

        Optional kwargs
        ---------------
        Q : np.ndarray
            Cached Q matrix. If not provided, compute_Q(...) is called.
        """
        Q = kwargs.get("Q", None)
        if Q is None:
            Q = self.compute_Q(system=system, data_train=data_train, V=V)

        V = _as2d(V)

        W = Q @ V
        _check_shapes(V, W)

        if self.enforce_WtV_I:
            W = biorthonormalize(W, V, eps=self.eps)

        return W


# @dataclass
# class QVTestBasis:
#     """
#     W = Q@V (common when Q defines an inner product/energy matrix)
#     Optionally post-process to enforce W^T V = I.
#     """

#     source: Literal["system_Q", "system_E", "Hamiltonian", "kyp"] = "system_Q"

#     enforce_WtV_I: bool = False
#     eps: float = 1e-12  # for biorthonormalization, if enabled

#     project_hamiltonian_Q_spsd: bool = (
#         True  # only used if source=Hamiltonian, whether to project identified Q to SPSD for stability
#     )

#     def make_W(
#         self, *, V: np.ndarray, system=None, data_train=None, **kwargs
#     ) -> np.ndarray:
#         if self.source == "system_Q":
#             if system is None or not hasattr(system, "Q"):
#                 raise ValueError("QVTestBasis with source=system_Q requires system.Q")
#             Q = np.asarray(system.Q)
#         elif self.source == "system_E":
#             if system is None or not hasattr(system, "E"):
#                 raise ValueError("QVTestBasis with source=system_E requires system.E")
#             Q = np.asarray(system.E)
#         elif self.source == "Hamiltonian":
#             if system is None:
#                 raise ValueError("QVTestBasis with source=Hamiltonian requires system")
#             if not hasattr(system, "Q") or not hasattr(system, "E"):
#                 raise ValueError(
#                     "QVTestBasis with source=Hamiltonian requires system with Q and E"
#                 )
#             if data_train is None:
#                 raise ValueError(
#                     "QVTestBasis with source=Hamiltonian requires data_train for Hamiltonian identification"
#                 )

#             Hamiltonian = 0.5 * np.array(
#                 [
#                     data_train.x[:, i].T @ system.E.T @ system.Q @ data_train.x[:, i]
#                     for i in range(data_train.x.shape[1])
#                 ]
#             )
#             Q = hamiltonian_identification(
#                 data_train.x,
#                 Hamiltonian,
#                 project=self.project_hamiltonian_Q_spsd,
#             )
#             Q = np.asarray(Q)
#         elif self.source == "kyp":
#             # identify Q matrix from data via KYP lemma
#             Q = solve_Riccati(system)
#             Q = np.asarray(Q)
#         else:
#             raise ValueError(f"Invalid source {self.source} for QVTestBasis")

#         V = _as2d(V)

#         W = Q @ V
#         _check_shapes(V, W)
#         if self.enforce_WtV_I:
#             W = biorthonormalize(W, V, eps=self.eps)
#         return W
