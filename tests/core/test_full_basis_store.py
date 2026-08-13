from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from pgopinf.core.full_basis_store import FullBasisStore
from pgopinf.io.results_path import ResultsPath
from pgopinf.reduction.full_basis.modal import ModalFullBasisArtifact
from pgopinf.reduction.full_basis.pod import PODFullBasisArtifact
from pgopinf.specs.reduction.full_basis import (
    ModalFullBasisSpec,
    PODFullBasisSpec,
)


def make_paths(tmp_path) -> ResultsPath:
    return ResultsPath(results_root=tmp_path / "results")


def make_pod_artifact() -> PODFullBasisArtifact:
    return PODFullBasisArtifact(
        U=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        S=np.array([3.0, 1.0]),
        Vh=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        meta={"kind": "pod", "source": "test"},
    )


def make_modal_artifact(*, with_left: bool = True) -> ModalFullBasisArtifact:
    return ModalFullBasisArtifact(
        eigvals=np.array([1.0 + 0.0j, 2.0 + 1.0j]),
        Vright=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
        Vleft=(
            np.array([[2.0, 0.0], [0.0, 2.0]], dtype=complex)
            if with_left
            else None
        ),
        meta={"kind": "modal", "source": "test"},
    )


def assert_pod_artifact_equal(left: PODFullBasisArtifact, right: PODFullBasisArtifact):
    np.testing.assert_allclose(left.U, right.U)
    np.testing.assert_allclose(left.S, right.S)
    np.testing.assert_allclose(left.Vh, right.Vh)
    assert left.meta == right.meta


def assert_modal_artifact_equal(
    left: ModalFullBasisArtifact, right: ModalFullBasisArtifact
):
    np.testing.assert_allclose(left.eigvals, right.eigvals)
    np.testing.assert_allclose(left.Vright, right.Vright)
    if left.Vleft is None:
        assert right.Vleft is None
    else:
        np.testing.assert_allclose(left.Vleft, right.Vleft)
    assert left.meta == right.meta


@dataclass(frozen=True)
class FakeFullBasisSpec:
    artifact: object
    calls: list
    kind: str = "fake"

    def to_dict(self):
        return {"kind": self.kind}

    def build(self):
        artifact = self.artifact
        calls = self.calls

        class Builder:
            def fit(self, *, dataset=None, system=None):
                calls.append({"dataset": dataset, "system": system})
                return artifact

        return Builder()


def test_compute_id_depends_on_dataset_and_spec(tmp_path) -> None:
    store = FullBasisStore(make_paths(tmp_path))

    pod_id = store.compute_id(
        data_train_id="train-a",
        full_basis_spec=PODFullBasisSpec(full_matrices=True),
    )

    assert pod_id == store.compute_id(
        data_train_id="train-a",
        full_basis_spec=PODFullBasisSpec(full_matrices=True),
    )
    assert pod_id != store.compute_id(
        data_train_id="train-b",
        full_basis_spec=PODFullBasisSpec(full_matrices=True),
    )
    assert pod_id != store.compute_id(
        data_train_id="train-a",
        full_basis_spec=PODFullBasisSpec(full_matrices=False),
    )


def test_save_and_load_pod_artifact_round_trip(tmp_path) -> None:
    paths = make_paths(tmp_path)
    store = FullBasisStore(paths)
    artifact = make_pod_artifact()

    store.save(
        "pod-id",
        data_train_id="train-id",
        full_basis_spec=PODFullBasisSpec(full_matrices=False),
        full_basis=artifact,
    )

    assert store.exists("pod-id")
    assert_pod_artifact_equal(store.load("pod-id"), artifact)
    assert json.loads((paths.full_bases_dir("pod-id") / "spec.json").read_text()) == {
        "data_train_id": "train-id",
        "full_basis": {"kind": "pod", "full_matrices": False},
        "full_basis_id": "pod-id",
    }


@pytest.mark.parametrize("with_left", [True, False])
def test_save_and_load_modal_artifact_round_trip(tmp_path, with_left: bool) -> None:
    store = FullBasisStore(make_paths(tmp_path))
    artifact = make_modal_artifact(with_left=with_left)

    store.save(
        "modal-id",
        data_train_id=None,
        full_basis_spec=ModalFullBasisSpec(compute_left=with_left),
        full_basis=artifact,
    )

    assert store.exists("modal-id")
    assert_modal_artifact_equal(store.load("modal-id"), artifact)


def test_load_rejects_unknown_artifact_kind(tmp_path) -> None:
    store = FullBasisStore(make_paths(tmp_path))
    d = store.paths.full_bases_dir("unknown")
    np.savez_compressed(d / "full_basis.npz", kind="mystery")
    (d / "spec.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown full_basis artifact kind"):
        store.load("unknown")


def test_get_or_create_computes_saves_and_reuses_cached_artifact(tmp_path) -> None:
    store = FullBasisStore(make_paths(tmp_path))
    artifact = make_pod_artifact()
    calls = []
    dataset_artifact = SimpleNamespace(train_id="train-id", data="dataset")
    spec = FakeFullBasisSpec(artifact=artifact, calls=calls)

    fb_id, first = store.get_or_create(
        dataset_artifact=dataset_artifact,
        system="system",
        full_basis_spec=spec,
    )
    second_id, second = store.get_or_create(
        dataset_artifact=dataset_artifact,
        system="system",
        full_basis_spec=spec,
    )

    assert fb_id == second_id
    assert calls == [{"dataset": "dataset", "system": "system"}]
    assert first is artifact
    assert_pod_artifact_equal(second, artifact)


def test_get_or_create_honors_force_recompute(tmp_path) -> None:
    store = FullBasisStore(make_paths(tmp_path), force_recompute=True)
    artifact = make_pod_artifact()
    calls = []
    dataset_artifact = SimpleNamespace(train_id="train-id", data="dataset")
    spec = FakeFullBasisSpec(artifact=artifact, calls=calls)

    store.get_or_create(
        dataset_artifact=dataset_artifact,
        system="system",
        full_basis_spec=spec,
    )
    store.get_or_create(
        dataset_artifact=dataset_artifact,
        system="system",
        full_basis_spec=spec,
    )

    assert calls == [
        {"dataset": "dataset", "system": "system"},
        {"dataset": "dataset", "system": "system"},
    ]


def test_get_or_create_supports_dataset_free_basis_builders(tmp_path) -> None:
    store = FullBasisStore(make_paths(tmp_path))
    artifact = make_modal_artifact()
    calls = []
    spec = FakeFullBasisSpec(artifact=artifact, calls=calls)

    fb_id, loaded = store.get_or_create(
        dataset_artifact=None,
        system="system",
        full_basis_spec=spec,
    )

    assert fb_id == store.compute_id(data_train_id=None, full_basis_spec=spec)
    assert calls == [{"dataset": None, "system": "system"}]
    assert loaded is artifact
    assert json.loads((store.paths.full_bases_dir(fb_id) / "spec.json").read_text())[
        "data_train_id"
    ] is None
