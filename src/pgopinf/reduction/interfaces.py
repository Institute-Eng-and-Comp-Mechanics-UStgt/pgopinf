from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any
import numpy as np

from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem


@dataclass(frozen=True)
class BasisArtifact:
    """Reduced trial and optional test basis.

    Attributes
    ----------
    V : ndarray, shape (n, r)
        Trial basis.
    W : ndarray or None, shape (n, r)
        Test basis. If ``None``, it is built later by a test-basis strategy.
    meta : dict of str to Any
        Metadata describing how the basis was constructed.
    """

    V: np.ndarray  # (n, r)
    W: np.ndarray | None  # (n, r) or None -> will be built later
    meta: dict[str, Any]

    @property
    def r(self) -> int:
        """Reduced dimension.

        Returns
        -------
        int
            Number of basis vectors.
        """
        return self.V.shape[1]


class FullBasisArtifact(Protocol):
    """Protocol for full basis artifacts that can be truncated."""

    @property
    def r_max(self) -> int:
        """Maximum available reduced dimension."""
        ...

    def truncate(self, r: int) -> BasisArtifact:
        """Return a truncated basis artifact."""
        ...

    def to_npz(self) -> dict[str, Any]:
        """Serialize the artifact to NPZ-compatible arrays."""
        ...

    @classmethod
    def from_npz(cls, npz: dict[str, Any]) -> "FullBasisArtifact":
        """Deserialize the artifact from NPZ data."""
        ...


class FullBasisBuilder(Protocol):
    """Protocol for builders that compute full basis artifacts."""

    def fit(self, *, dataset=None, system=None) -> FullBasisArtifact:
        """Fit and return a full basis artifact."""
        ...


class MORProjector(Protocol):
    """Protocol for model-order-reduction projectors."""

    def reduce_system(
        self, *, system: LTISystem | PHSystem, basis: BasisArtifact
    ) -> LTISystem:
        """Reduce a system with a basis."""
        ...

    def project_data(self, *, data, basis: BasisArtifact) -> Any:
        """Project data with a basis."""
        ...

    def project_initial_condition(self, ic, basis) -> Any:
        """Project initial conditions with a basis."""
        ...
