from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Mapping, Any, Protocol

from pgopinf.reduction.test_basis import TestBasis
from pgopinf.specs.base import SpecBase
from pgopinf.specs.registry import KindRegistry


class TestBasisSpec(Protocol):
    """Protocol for test-basis specifications."""

    kind: str

    def build(self) -> TestBasis:
        """Build the runtime test-basis strategy."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the test-basis spec."""
        ...


TEST_BASIS_REGISTRY: KindRegistry[TestBasisSpec] = KindRegistry("test_basis")


def register_test_basis(kind: str):
    """Register a test-basis spec kind."""
    return TEST_BASIS_REGISTRY.register(kind)


def test_basis_default_by_kind(kind: str) -> TestBasisSpec:
    """Return the default test-basis spec for a kind."""
    return TEST_BASIS_REGISTRY.default_by_kind(kind)


def test_basis_from_dict(d: Mapping[str, Any]) -> TestBasisSpec:
    """Deserialize a test-basis spec."""
    return TEST_BASIS_REGISTRY.from_dict(d)


@register_test_basis("galerkin")
@dataclass(frozen=True)
class GalerkinTestBasisSpec(SpecBase):
    """Specification for a Galerkin test basis."""

    kind: Literal["galerkin"] = "galerkin"

    def build(self):
        """Build the Galerkin test-basis strategy."""
        from pgopinf.reduction.test_basis import (
            GalerkinTestBasis,
        )

        return GalerkinTestBasis()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GalerkinTestBasisSpec":
        """Deserialize a Galerkin test-basis spec."""
        return cls()


@register_test_basis("vq")
@dataclass(frozen=True)
class QVTestBasisSpec(SpecBase):
    """Specification for a ``W = Q V`` test basis."""

    kind: Literal["vq"] = "vq"
    source: Literal["system_Q", "system_E", "Hamiltonian", "kyp"] = "system_Q"

    enforce_WtV_I: bool = False
    eps: float = 1e-12  # for biorthonormalization, if enabled

    project_hamiltonian_Q_spsd: bool = (
        True  # only used if source=Hamiltonian, whether to project identified Q to SPSD for stability
    )

    def build(self):
        """Build the QV test-basis strategy."""
        from pgopinf.reduction.test_basis import QVTestBasis

        return QVTestBasis(
            source=self.source,
            enforce_WtV_I=self.enforce_WtV_I,
            eps=self.eps,
            project_hamiltonian_Q_spsd=self.project_hamiltonian_Q_spsd,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "QVTestBasisSpec":
        """Deserialize a QV test-basis spec."""
        return cls(
            source=str(d.get("source", "system_Q")),
            enforce_WtV_I=bool(d.get("enforce_WtV_I", False)),
            eps=float(d.get("eps", 1e-12)),
            project_hamiltonian_Q_spsd=bool(d.get("project_hamiltonian_Q_spsd", True)),
        )
