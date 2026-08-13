from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pgopinf.reduction.full_basis.modal import (
    ModalFullBasisArtifact,
    ModalFullBasisBuilder,
)


def make_artifact(*, with_left: bool = True) -> ModalFullBasisArtifact:
    return ModalFullBasisArtifact(
        eigvals=np.array([1.0 + 3.0j, 2.0 + 0.5j, 3.0 + 2.0j]),
        Vright=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        Vleft=(
            np.array(
                [
                    [10.0, 0.0, 0.0],
                    [0.0, 20.0, 0.0],
                    [0.0, 0.0, 30.0],
                ]
            )
            if with_left
            else None
        ),
        meta={"source": "modal"},
    )


def test_modal_full_basis_artifact_truncates_by_smallest_imaginary_magnitude() -> None:
    artifact = make_artifact()

    basis = artifact.truncate(2)

    assert artifact.r_max == 3
    np.testing.assert_allclose(basis.V, artifact.Vright[:, [1, 2]])
    np.testing.assert_allclose(basis.W, artifact.Vleft[:, [1, 2]])
    assert basis.meta == {"source": "modal", "r": 2, "idx": [1, 2]}


def test_modal_full_basis_artifact_truncates_without_left_basis() -> None:
    basis = make_artifact(with_left=False).truncate(1)

    np.testing.assert_allclose(basis.V, np.array([[0.0], [1.0], [0.0]]))
    assert basis.W is None


@pytest.mark.parametrize("r", [0, 4])
def test_modal_full_basis_artifact_rejects_invalid_truncation_rank(r: int) -> None:
    with pytest.raises(ValueError, match="r must be in"):
        make_artifact().truncate(r)


def test_modal_full_basis_artifact_round_trips_npz_payload_with_left_basis() -> None:
    artifact = make_artifact()

    loaded = ModalFullBasisArtifact.from_npz(artifact.to_npz())

    np.testing.assert_allclose(loaded.eigvals, artifact.eigvals)
    np.testing.assert_allclose(loaded.Vright, artifact.Vright)
    np.testing.assert_allclose(loaded.Vleft, artifact.Vleft)
    assert loaded.meta == artifact.meta


def test_modal_full_basis_artifact_round_trips_npz_payload_without_left_basis() -> None:
    artifact = make_artifact(with_left=False)

    loaded = ModalFullBasisArtifact.from_npz(artifact.to_npz())

    assert loaded.Vleft is None
    assert loaded.meta == artifact.meta


def test_modal_full_basis_builder_fits_right_and_left_eigenvectors() -> None:
    system = SimpleNamespace(E=np.eye(2), A=np.diag([1.0, 2.0]))

    artifact = ModalFullBasisBuilder(compute_left=True).fit(system=system)

    assert artifact.meta == {"kind": "modal", "compute_left": True}
    np.testing.assert_allclose(np.sort(artifact.eigvals), np.array([1.0, 2.0]))
    assert artifact.Vright.shape == (2, 2)
    assert artifact.Vleft.shape == (2, 2)


def test_modal_full_basis_builder_can_skip_left_eigenvectors() -> None:
    system = SimpleNamespace(E=np.eye(2), A=np.diag([1.0, 2.0]))

    artifact = ModalFullBasisBuilder(compute_left=False).fit(system=system)

    assert artifact.Vleft is None
    assert artifact.meta == {"kind": "modal", "compute_left": False}


def test_modal_full_basis_builder_requires_system() -> None:
    with pytest.raises(ValueError, match="requires system"):
        ModalFullBasisBuilder().fit(system=None)
