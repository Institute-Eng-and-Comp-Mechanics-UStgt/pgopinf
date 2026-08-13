from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from pgopinf.data.dataset import DataSet
from pgopinf.reduction.interfaces import BasisArtifact


@dataclass(frozen=True)
class PODFullBasisArtifact:
    """Full POD basis artifact.

    Attributes
    ----------
    U : ndarray, shape (n, k)
        Left singular vectors used as trial basis vectors.
    S : ndarray, shape (k,)
        Singular values.
    Vh : ndarray, shape (k, n_samples)
        Right singular vectors returned by SVD.
    meta : dict of str to Any
        Metadata describing the artifact.
    """

    U: np.ndarray  # (n, k)
    S: np.ndarray  # (k,)
    Vh: np.ndarray  # (k, T)
    meta: dict[str, Any]

    @property
    def r_max(self) -> int:
        """Maximum available reduced dimension.

        Returns
        -------
        int
            Number of POD modes stored in ``U``.
        """
        return int(self.U.shape[1])

    def truncate(self, r: int) -> BasisArtifact:
        """Return the first ``r`` POD modes as a basis artifact.

        Parameters
        ----------
        r : int
            Reduced dimension.

        Returns
        -------
        BasisArtifact
            Truncated trial basis with no test basis.
        """
        r = int(r)
        if r <= 0 or r > self.r_max:
            raise ValueError(f"r must be in [1, {self.r_max}], got {r}")
        V = self.U[:, :r]
        return BasisArtifact(V=V, W=None, meta={**self.meta, "r": r})

    def to_npz(self) -> dict[str, Any]:
        """Serialize the POD artifact to NPZ-compatible data.

        Returns
        -------
        dict of str to Any
            Arrays and JSON metadata suitable for ``numpy.savez``.
        """
        return {
            "kind": "pod",
            "U": self.U,
            "S": self.S,
            "Vh": self.Vh,
            "meta_json": _json_dumps(self.meta),
        }

    @classmethod
    def from_npz(cls, npz: dict[str, Any]) -> "PODFullBasisArtifact":
        """Deserialize a POD artifact from NPZ data.

        Parameters
        ----------
        npz : dict of str to Any
            Data loaded from ``numpy.load``.

        Returns
        -------
        PODFullBasisArtifact
            Reconstructed POD artifact.
        """
        U = np.asarray(npz["U"])
        S = np.asarray(npz["S"])
        Vh = np.asarray(npz["Vh"])
        meta = _json_loads(str(np.asarray(npz["meta_json"]).ravel()[0]))
        return cls(U=U, S=S, Vh=Vh, meta=meta)


def _json_dumps(d: dict) -> str:
    import json

    return json.dumps(d, sort_keys=True)


def _json_loads(s: str) -> dict:
    import json

    return json.loads(s)


@dataclass
class PODFullBasisBuilder:
    """Builder for POD full-basis artifacts."""

    full_matrices: bool = True

    def fit(self, *, dataset: DataSet = None, system=None) -> PODFullBasisArtifact:
        """Fit a POD basis from training snapshots.

        Parameters
        ----------
        dataset : DataSet
            Dataset whose training split provides sample-format states.
        system
            Unused; accepted for the full-basis builder protocol.

        Returns
        -------
        PODFullBasisArtifact
            Fitted POD artifact.
        """
        if dataset is None or dataset.TRAIN is None:
            raise ValueError("PODFullBasisBuilder.fit requires dataset.TRAIN")

        x_train = dataset.TRAIN.x  # (n, T)
        U, S, Vh = pod(x_train, full_matrices=self.full_matrices)

        meta = {
            "kind": "pod",
        }
        return PODFullBasisArtifact(U=U, S=S, Vh=Vh, meta=meta)


def pod(x_train: np.ndarray, full_matrices: bool = True):
    """Compute a singular value decomposition for POD.

    Parameters
    ----------
    x_train : ndarray, shape (n, n_samples)
        Training snapshot matrix.
    full_matrices : bool, optional
        Forwarded to ``numpy.linalg.svd``.

    Returns
    -------
    tuple
        ``(U, S, Vh)`` returned by ``numpy.linalg.svd``.
    """
    V, S, Vh = np.linalg.svd(x_train, full_matrices=full_matrices)
    return V, S, Vh
