from __future__ import annotations

import pytest

from pgopinf.specs.reduction.base import ReductionSpec
from pgopinf.specs.reduction.full_basis import (
    FULL_BASIS_REGISTRY,
    ModalFullBasisSpec,
    PODFullBasisSpec,
    full_basis_default_by_kind,
    full_basis_from_dict,
)
from pgopinf.specs.reduction.mor import (
    MOR_REGISTRY,
    LTIMORSpec,
    PHMORSpec,
    mor_default_by_kind,
    mor_from_dict,
)
from pgopinf.specs.reduction.test_basis import (
    TEST_BASIS_REGISTRY,
    GalerkinTestBasisSpec,
    QVTestBasisSpec,
    test_basis_default_by_kind as basis_default_spec_by_kind,
    test_basis_from_dict as basis_spec_from_dict,
)


@pytest.mark.parametrize(
    ("kind", "spec"),
    [
        ("pod", PODFullBasisSpec(full_matrices=False)),
        ("modal", ModalFullBasisSpec(compute_left=False)),
    ],
)
def test_full_basis_spec_round_trip(kind: str, spec) -> None:
    assert spec.kind == kind
    assert spec.to_dict()["kind"] == kind
    assert full_basis_from_dict(spec.to_dict()) == spec


def test_full_basis_registry_defaults_and_known_kinds() -> None:
    assert FULL_BASIS_REGISTRY.known_kinds() == ["modal", "pod"]
    assert full_basis_default_by_kind("pod") == PODFullBasisSpec()
    assert full_basis_default_by_kind("modal") == ModalFullBasisSpec()


def test_full_basis_from_dict_requires_known_kind() -> None:
    with pytest.raises(KeyError, match="missing required keys"):
        full_basis_from_dict({})

    with pytest.raises(ValueError, match="unknown kind"):
        full_basis_from_dict({"kind": "missing"})


def test_full_basis_from_dict_uses_spec_specific_parsers() -> None:
    assert full_basis_from_dict({"kind": "pod", "full_matrices": ""}) == (
        PODFullBasisSpec(full_matrices=False)
    )
    assert full_basis_from_dict({"kind": "modal", "compute_left": ""}) == (
        ModalFullBasisSpec(compute_left=False)
    )


@pytest.mark.parametrize(
    ("kind", "spec"),
    [
        ("galerkin", GalerkinTestBasisSpec()),
        (
            "vq",
            QVTestBasisSpec(
                source="Hamiltonian",
                enforce_WtV_I=True,
                eps=1e-10,
                project_hamiltonian_Q_spsd=False,
            ),
        ),
    ],
)
def test_test_basis_spec_round_trip(kind: str, spec) -> None:
    assert spec.kind == kind
    assert spec.to_dict()["kind"] == kind
    assert basis_spec_from_dict(spec.to_dict()) == spec


def test_test_basis_registry_defaults_and_known_kinds() -> None:
    assert TEST_BASIS_REGISTRY.known_kinds() == ["galerkin", "vq"]
    assert basis_default_spec_by_kind("galerkin") == GalerkinTestBasisSpec()
    assert basis_default_spec_by_kind("vq") == QVTestBasisSpec()


def test_test_basis_from_dict_requires_known_kind() -> None:
    with pytest.raises(KeyError, match="missing required keys"):
        basis_spec_from_dict({})

    with pytest.raises(ValueError, match="unknown kind"):
        basis_spec_from_dict({"kind": "missing"})


def test_test_basis_from_dict_uses_spec_specific_parsers() -> None:
    assert basis_spec_from_dict(
        {
            "kind": "vq",
            "source": "system_E",
            "enforce_WtV_I": "1",
            "eps": "1e-9",
            "project_hamiltonian_Q_spsd": "",
        }
    ) == QVTestBasisSpec(
        source="system_E",
        enforce_WtV_I=True,
        eps=1e-9,
        project_hamiltonian_Q_spsd=False,
    )


@pytest.mark.parametrize(
    ("kind", "spec"),
    [
        ("lti", LTIMORSpec()),
        ("ph", PHMORSpec(ph_reduction_type="Gugercin")),
    ],
)
def test_mor_spec_round_trip(kind: str, spec) -> None:
    assert spec.kind == kind
    assert spec.to_dict()["kind"] == kind
    assert mor_from_dict(spec.to_dict()) == spec


def test_mor_registry_defaults_and_known_kinds() -> None:
    assert MOR_REGISTRY.known_kinds() == ["lti", "ph"]
    assert mor_default_by_kind("lti") == LTIMORSpec()
    assert mor_default_by_kind("ph") == PHMORSpec()


def test_mor_from_dict_requires_known_kind() -> None:
    with pytest.raises(KeyError, match="missing required keys"):
        mor_from_dict({})

    with pytest.raises(ValueError, match="unknown kind"):
        mor_from_dict({"kind": "missing"})


def test_mor_from_dict_uses_spec_specific_parsers() -> None:
    assert mor_from_dict({"kind": "ph", "ph_reduction_type": "Gugercin"}) == (
        PHMORSpec(ph_reduction_type="Gugercin")
    )


def test_reduction_spec_round_trip() -> None:
    spec = ReductionSpec(
        r=7,
        full_basis=ModalFullBasisSpec(compute_left=False),
        test_basis=QVTestBasisSpec(
            source="system_E",
            enforce_WtV_I=True,
            eps=1e-9,
            project_hamiltonian_Q_spsd=False,
        ),
        mor=PHMORSpec(ph_reduction_type="Gugercin"),
        project_data=False,
        reduce_system=False,
    )

    assert ReductionSpec.from_dict(spec.to_dict()) == spec


def test_reduction_spec_defaults_and_partial_dict() -> None:
    assert ReductionSpec.from_dict({}) == ReductionSpec()

    spec = ReductionSpec.from_dict(
        {
            "r": "4",
            "full_basis": {"kind": "modal"},
            "test_basis": {"kind": "vq", "source": "system_E"},
            "mor": {"kind": "ph"},
            "project_data": "",
            "reduce_system": "1",
        }
    )

    assert spec == ReductionSpec(
        r=4,
        full_basis=ModalFullBasisSpec(),
        test_basis=QVTestBasisSpec(source="system_E"),
        mor=PHMORSpec(),
        project_data=False,
        reduce_system=True,
    )


def test_reduction_spec_rejects_nonpositive_r() -> None:
    with pytest.raises(ValueError, match="ReductionSpec.r must be > 0"):
        ReductionSpec(r=0)

    with pytest.raises(ValueError, match="ReductionSpec.r must be > 0"):
        ReductionSpec.from_dict({"r": "-1"})


@pytest.mark.parametrize(
    ("spec", "runtime_type_name"),
    [
        (PODFullBasisSpec(), "PODFullBasisBuilder"),
        (ModalFullBasisSpec(), "ModalFullBasisBuilder"),
        (GalerkinTestBasisSpec(), "GalerkinTestBasis"),
        (QVTestBasisSpec(), "QVTestBasis"),
        (LTIMORSpec(), "LTIMORProjector"),
        (PHMORSpec(), "PHMORProjector"),
    ],
)
def test_reduction_specs_build_runtime_objects(spec, runtime_type_name: str) -> None:
    assert type(spec.build()).__name__ == runtime_type_name


def test_reduction_specs_build_forward_options() -> None:
    assert PODFullBasisSpec(full_matrices=False).build().full_matrices is False
    assert ModalFullBasisSpec(compute_left=False).build().compute_left is False

    qv = QVTestBasisSpec(
        source="system_E",
        enforce_WtV_I=True,
        eps=1e-9,
        project_hamiltonian_Q_spsd=False,
    ).build()
    assert qv.source == "system_E"
    assert qv.enforce_WtV_I is True
    assert qv.eps == pytest.approx(1e-9)
    assert qv.project_hamiltonian_Q_spsd is False

    assert PHMORSpec(ph_reduction_type="Gugercin").build().ph_reduction_type == (
        "Gugercin"
    )
