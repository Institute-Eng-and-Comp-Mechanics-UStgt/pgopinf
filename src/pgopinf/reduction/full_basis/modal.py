from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np

from pgopinf.reduction.interfaces import BasisArtifact


@dataclass(frozen=True)
class ModalFullBasisArtifact:
    """Full modal basis artifact.

    Attributes
    ----------
    eigvals : ndarray, shape (n,)
        Eigenvalues associated with the modal basis.
    Vright : ndarray, shape (n, n)
        Right eigenvectors.
    Vleft : ndarray or None, shape (n, n)
        Optional left eigenvectors used as a test basis.
    meta : dict of str to Any
        Metadata describing the artifact.
    """

    eigvals: np.ndarray  # (n,)
    Vright: np.ndarray  # (n, n)
    Vleft: np.ndarray | None  # (n, n) optional for Petrov
    meta: dict[str, Any]

    @property
    def r_max(self) -> int:
        """Maximum available reduced dimension.

        Returns
        -------
        int
            Number of right eigenvectors stored.
        """
        return int(self.Vright.shape[1])

    def truncate(self, r: int) -> BasisArtifact:
        """Select ``r`` modal vectors.

        Parameters
        ----------
        r : int
            Reduced dimension.

        Returns
        -------
        BasisArtifact
            Basis artifact containing selected right and optional left modes.
        """
        r = int(r)
        if r <= 0 or r > self.r_max:
            raise ValueError(f"r must be in [1, {self.r_max}], got {r}")

        # selection rule: by smallest |imag| frequency (example)
        idx = np.argsort(np.abs(np.imag(self.eigvals)))[:r]
        V = self.Vright[:, idx]
        W = self.Vleft[:, idx] if self.Vleft is not None else None
        return BasisArtifact(V=V, W=W, meta={**self.meta, "r": r, "idx": idx.tolist()})

    def to_npz(self) -> dict[str, Any]:
        """Serialize the modal artifact to NPZ-compatible data.

        Returns
        -------
        dict of str to Any
            Arrays and JSON metadata suitable for ``numpy.savez``.
        """
        return {
            "kind": "modal",
            "eigvals": self.eigvals,
            "Vright": self.Vright,
            "Vleft": (
                self.Vleft if self.Vleft is not None else np.array([], dtype=complex)
            ),
            "has_Vleft": np.array([self.Vleft is not None], dtype=bool),
            "meta_json": np.array([_json_dumps(self.meta)], dtype=object),
        }

    @classmethod
    def from_npz(cls, npz: dict[str, Any]) -> "ModalFullBasisArtifact":
        """Deserialize a modal artifact from NPZ data.

        Parameters
        ----------
        npz : dict of str to Any
            Data loaded from ``numpy.load``.

        Returns
        -------
        ModalFullBasisArtifact
            Reconstructed modal artifact.
        """
        eigvals = np.asarray(npz["eigvals"])
        Vright = np.asarray(npz["Vright"])
        has_Vleft = bool(np.asarray(npz["has_Vleft"]).ravel()[0])
        Vleft = np.asarray(npz["Vleft"]) if has_Vleft else None
        meta = _json_loads(str(np.asarray(npz["meta_json"]).ravel()[0]))
        return cls(eigvals=eigvals, Vright=Vright, Vleft=Vleft, meta=meta)


def _json_dumps(d: dict) -> str:
    import json

    return json.dumps(d, sort_keys=True)


def _json_loads(s: str) -> dict:
    import json

    return json.loads(s)


@dataclass
class ModalFullBasisBuilder:
    """
    Example for (E,A) generalized eigenproblem.
    You can adapt to your LTISystem class.
    """

    compute_left: bool = True

    def fit(self, *, dataset=None, system=None) -> ModalFullBasisArtifact:
        """Fit a modal basis from a system eigenproblem.

        Parameters
        ----------
        dataset
            Unused; accepted for the full-basis builder protocol.
        system
            System providing ``A`` and ``E`` matrices.

        Returns
        -------
        ModalFullBasisArtifact
            Fitted modal artifact.
        """
        if system is None:
            raise ValueError("ModalFullBasisBuilder.fit requires system")

        E, A = system.E, system.A  # adapt to your runtime API
        # Solve A v = λ E v
        eigvals, Vright = np.linalg.eig(np.linalg.solve(E, A))
        # Note: for general descriptor systems you'd use scipy.linalg.eig(A, E)

        Vleft = None
        if self.compute_left:
            eigvals_l, Vleft = np.linalg.eig(np.linalg.solve(E.T, A.T))
            # left eigenvectors of (A,E) can be computed properly with scipy.linalg.eig(A.T, E.T)

        meta = {"kind": "modal", "compute_left": self.compute_left}
        return ModalFullBasisArtifact(
            eigvals=eigvals, Vright=Vright, Vleft=Vleft, meta=meta
        )
