from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pgopinf.reduction.full_basis.pod import (
    PODFullBasisArtifact,
    PODFullBasisBuilder,
    pod,
)


def make_artifact() -> PODFullBasisArtifact:
    return PODFullBasisArtifact(
        U=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        S=np.array([3.0, 1.0]),
        Vh=np.array([[1.0, 0.0], [0.0, 1.0]]),
        meta={"source": "test"},
    )


def test_pod_full_basis_artifact_truncates_to_basis_artifact() -> None:
    artifact = make_artifact()

    basis = artifact.truncate(1)

    assert artifact.r_max == 2
    np.testing.assert_allclose(basis.V, artifact.U[:, :1])
    assert basis.W is None
    assert basis.meta == {"source": "test", "r": 1}


@pytest.mark.parametrize("r", [0, 3])
def test_pod_full_basis_artifact_rejects_invalid_truncation_rank(r: int) -> None:
    with pytest.raises(ValueError, match="r must be in"):
        make_artifact().truncate(r)


def test_pod_full_basis_artifact_round_trips_npz_payload() -> None:
    artifact = make_artifact()

    loaded = PODFullBasisArtifact.from_npz(artifact.to_npz())

    np.testing.assert_allclose(loaded.U, artifact.U)
    np.testing.assert_allclose(loaded.S, artifact.S)
    np.testing.assert_allclose(loaded.Vh, artifact.Vh)
    assert loaded.meta == artifact.meta


def test_pod_function_wraps_numpy_svd() -> None:
    x_train = np.array([[1.0, 0.0], [0.0, 2.0]])

    U, S, Vh = pod(x_train, full_matrices=False)

    np.testing.assert_allclose(U @ np.diag(S) @ Vh, x_train)


def test_pod_full_basis_builder_fits_from_training_data() -> None:
    x_train = np.array([[1.0, 0.0], [0.0, 2.0]])
    dataset = SimpleNamespace(TRAIN=SimpleNamespace(x=x_train))

    artifact = PODFullBasisBuilder(full_matrices=False).fit(dataset=dataset)

    assert artifact.meta == {"kind": "pod"}
    np.testing.assert_allclose(artifact.U @ np.diag(artifact.S) @ artifact.Vh, x_train)


def test_pod_full_basis_builder_requires_training_dataset() -> None:
    with pytest.raises(ValueError, match="requires dataset.TRAIN"):
        PODFullBasisBuilder().fit(dataset=None)

    with pytest.raises(ValueError, match="requires dataset.TRAIN"):
        PODFullBasisBuilder().fit(dataset=SimpleNamespace(TRAIN=None))
