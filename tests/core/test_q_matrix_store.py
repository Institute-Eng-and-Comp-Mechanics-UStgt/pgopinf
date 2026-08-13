from __future__ import annotations

import json

import numpy as np
import pytest

from pgopinf.core.q_matrix_store import QMatrixArtifact, QMatrixStore
from pgopinf.io.results_path import ResultsPath
from pgopinf.specs.reduction.base import ReductionSpec


def make_paths(tmp_path) -> ResultsPath:
    return ResultsPath(results_root=tmp_path / "results")


def make_artifact(scale: float = 1.0) -> QMatrixArtifact:
    return QMatrixArtifact(
        Q=scale * np.array([[2.0, 0.5], [0.5, 1.0]]),
        meta={"source": "test", "scale": scale},
    )


def assert_artifact_equal(left: QMatrixArtifact, right: QMatrixArtifact) -> None:
    np.testing.assert_allclose(left.Q, right.Q)
    assert left.meta == right.meta


def test_compute_id_is_stable_and_depends_on_inputs(tmp_path) -> None:
    store = QMatrixStore(make_paths(tmp_path))

    base_id = store.compute_id(
        source="Hamiltonian",
        data_train_id="train-a",
        options={"b": 2, "a": 1},
    )

    assert base_id == store.compute_id(
        source="Hamiltonian",
        data_train_id="train-a",
        options={"a": 1, "b": 2},
    )
    assert base_id != store.compute_id(
        source="system_Q",
        data_train_id="train-a",
        options={"a": 1, "b": 2},
    )
    assert base_id != store.compute_id(
        source="Hamiltonian",
        data_train_id="train-b",
        options={"a": 1, "b": 2},
    )
    assert base_id != store.compute_id(
        source="Hamiltonian",
        data_train_id="train-a",
        options={"a": 1, "b": 3},
    )


def test_save_and_load_round_trip(tmp_path) -> None:
    store = QMatrixStore(make_paths(tmp_path))
    artifact = make_artifact(scale=2.0)

    store.save("q-id", artifact)

    assert store.exists("q-id")
    assert_artifact_equal(store.load("q-id"), artifact)
    assert json.loads((store._dir("q-id") / "meta.json").read_text()) == artifact.meta


def test_load_missing_artifact_raises(tmp_path) -> None:
    store = QMatrixStore(make_paths(tmp_path))

    with pytest.raises(FileNotFoundError, match="QMatrix artifact missing not found"):
        store.load("missing")


def test_exists_requires_matrix_and_meta_files(tmp_path) -> None:
    store = QMatrixStore(make_paths(tmp_path))
    d = store._dir("partial")
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "Q.npz", Q=np.eye(2))

    assert store.exists("partial") is False

    (d / "meta.json").write_text("{}", encoding="utf-8")
    assert store.exists("partial") is True


def test_get_or_create_computes_saves_and_reuses_cached_artifact(tmp_path) -> None:
    store = QMatrixStore(make_paths(tmp_path))
    calls = []

    def compute_fn():
        calls.append("called")
        return make_artifact()

    q_id, first = store.get_or_create(
        source="Hamiltonian",
        data_train_id="train-id",
        options={"eps": 1e-9},
        compute_fn=compute_fn,
    )
    second_id, second = store.get_or_create(
        source="Hamiltonian",
        data_train_id="train-id",
        options={"eps": 1e-9},
        compute_fn=compute_fn,
    )

    assert q_id == second_id
    assert calls == ["called"]
    assert first.meta == {"source": "test", "scale": 1.0}
    assert_artifact_equal(second, first)


def test_get_or_create_honors_force_recompute(tmp_path) -> None:
    store = QMatrixStore(make_paths(tmp_path), force_recompute=True)
    calls = []

    def compute_fn():
        calls.append(len(calls) + 1)
        return make_artifact(scale=float(calls[-1]))

    q_id, first = store.get_or_create(
        source="Hamiltonian",
        data_train_id="train-id",
        options={},
        compute_fn=compute_fn,
    )
    second_id, second = store.get_or_create(
        source="Hamiltonian",
        data_train_id="train-id",
        options={},
        compute_fn=compute_fn,
    )

    assert q_id == second_id
    assert calls == [1, 2]
    assert first.meta["scale"] == 1.0
    assert second.meta["scale"] == 2.0


def test_hamiltonian_red_cache_key_includes_reduction_spec(tmp_path) -> None:
    store = QMatrixStore(make_paths(tmp_path))

    first_id, _ = store.get_or_create(
        source="Hamiltonian_red",
        data_train_id="train-id",
        options={"source": "reduced"},
        reduction_spec=ReductionSpec(r=2),
        compute_fn=make_artifact,
    )
    second_id, _ = store.get_or_create(
        source="Hamiltonian_red",
        data_train_id="train-id",
        options={"source": "reduced"},
        reduction_spec=ReductionSpec(r=3),
        compute_fn=make_artifact,
    )

    assert first_id != second_id
    assert first_id == store.compute_id(
        source="Hamiltonian_red",
        data_train_id="train-id",
        options={
            "source": "reduced",
            "reduction_spec": ReductionSpec(r=2).to_dict(),
        },
    )


def test_hamiltonian_red_requires_reduction_spec(tmp_path) -> None:
    store = QMatrixStore(make_paths(tmp_path))

    with pytest.raises(ValueError, match="reduction_spec is required"):
        store.get_or_create(
            source="Hamiltonian_red",
            data_train_id="train-id",
            options={},
            reduction_spec=None,
            compute_fn=make_artifact,
        )
