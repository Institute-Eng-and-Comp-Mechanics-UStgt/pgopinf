from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from pgopinf.core import reduction_store as reduction_store_module
from pgopinf.core.q_matrix_store import QMatrixArtifact
from pgopinf.core.reduction_store import ReductionStore
from pgopinf.io.results_path import ResultsPath
from pgopinf.reduction.interfaces import BasisArtifact
from pgopinf.reduction.reducer import ReductionResult
from pgopinf.specs.reduction.base import ReductionSpec
from pgopinf.specs.reduction.test_basis import QVTestBasisSpec


def make_paths(tmp_path) -> ResultsPath:
    return ResultsPath(results_root=tmp_path / "results")


def make_dataset_artifact():
    return SimpleNamespace(
        train_id="train-id",
        data=SimpleNamespace(TRAIN="train-data", TEST="test-data"),
    )


class FakeFullBasis:
    def __init__(self):
        self.truncate_calls = []
        self.basis = BasisArtifact(
            V=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            W=None,
            meta={"kind": "fake"},
        )

    def truncate(self, r: int) -> BasisArtifact:
        self.truncate_calls.append(r)
        return self.basis


class FakeFullBasisStore:
    def __init__(self, full_basis):
        self.full_basis = full_basis
        self.calls = []

    def get_or_create(self, **kwargs):
        self.calls.append(kwargs)
        return "full-basis-id", self.full_basis


class FakeQMatrixStore:
    def __init__(self, *, call_compute_fn: bool = True):
        self.call_compute_fn = call_compute_fn
        self.calls = []

    def get_or_create(self, **kwargs):
        self.calls.append(kwargs)
        if self.call_compute_fn:
            artifact = kwargs["compute_fn"]()
        else:
            artifact = QMatrixArtifact(Q=np.eye(2) * 3.0, meta={"cached": True})
        return "q-id", artifact


@dataclass
class FakeTestBasisNoQ:
    pass


@dataclass
class FakeTestBasisWithQ:
    source: str = "Hamiltonian_red"

    def q_cache_options(self):
        return {"alpha": 1}

    def compute_Q(self, **kwargs):
        self.compute_q_kwargs = kwargs
        return np.eye(2) * 2.0


class FakeReducer:
    instances = []

    def __init__(self, spec):
        self.spec = spec
        self.test_basis = spec.test_basis.build()
        self.reduce_calls = []
        self.result = ReductionResult(
            basis=BasisArtifact(V=np.eye(2), W=np.eye(2), meta={"reduced": True}),
            mor="mor",
            reduced_dataset_projected="projected",
            reduced_system="reduced-system",
            full_basis=None,
        )
        FakeReducer.instances.append(self)

    def reduce(self, **kwargs):
        self.reduce_calls.append(kwargs)
        return self.result


@dataclass(frozen=True)
class FakeTestBasisSpecNoQ:
    kind: str = "fake_no_q"

    def to_dict(self):
        return {"kind": self.kind}

    def build(self):
        return FakeTestBasisNoQ()


@dataclass(frozen=True)
class FakeTestBasisSpecWithQ:
    kind: str = "fake_with_q"

    def to_dict(self):
        return {"kind": self.kind}

    def build(self):
        return FakeTestBasisWithQ()


def make_store(tmp_path, *, full_basis=None, q_store=None) -> ReductionStore:
    full_basis = full_basis or FakeFullBasis()
    return ReductionStore(
        paths=make_paths(tmp_path),
        full_basis_store=FakeFullBasisStore(full_basis),
        q_matrix_store=q_store or FakeQMatrixStore(),
    )


def test_compute_id_is_stable_and_depends_on_inputs(tmp_path) -> None:
    store = make_store(tmp_path)

    base_id = store.compute_id(
        data_train_id="train-a",
        reduction_spec=ReductionSpec(r=2),
    )

    assert base_id == store.compute_id(
        data_train_id="train-a",
        reduction_spec=ReductionSpec(r=2),
    )
    assert base_id != store.compute_id(
        data_train_id="train-b",
        reduction_spec=ReductionSpec(r=2),
    )
    assert base_id != store.compute_id(
        data_train_id="train-a",
        reduction_spec=ReductionSpec(r=3),
    )


def test_get_or_create_fetches_full_basis_and_runs_reducer_without_q_cache(
    tmp_path, monkeypatch
) -> None:
    FakeReducer.instances = []
    monkeypatch.setattr(reduction_store_module, "Reducer", FakeReducer)
    full_basis = FakeFullBasis()
    q_store = FakeQMatrixStore()
    store = make_store(tmp_path, full_basis=full_basis, q_store=q_store)
    spec = ReductionSpec(r=2, test_basis=FakeTestBasisSpecNoQ())
    dataset_artifact = make_dataset_artifact()

    result = store.get_or_create(
        dataset_artifact=dataset_artifact,
        system="system",
        reduction_spec=spec,
    )

    assert store.full_basis_store.calls == [
        {
            "dataset_artifact": dataset_artifact,
            "system": "system",
            "full_basis_spec": spec.full_basis,
        }
    ]
    assert q_store.calls == []
    reducer = FakeReducer.instances[0]
    assert reducer.spec is spec
    assert reducer.reduce_calls == [
        {
            "dataset": dataset_artifact.data,
            "system": "system",
            "full_basis": full_basis,
            "test_basis_kwargs": None,
        }
    ]
    assert result is reducer.result


def test_get_or_create_uses_q_cache_when_test_basis_supports_it(
    tmp_path, monkeypatch
) -> None:
    FakeReducer.instances = []
    monkeypatch.setattr(reduction_store_module, "Reducer", FakeReducer)
    full_basis = FakeFullBasis()
    q_store = FakeQMatrixStore(call_compute_fn=True)
    store = make_store(tmp_path, full_basis=full_basis, q_store=q_store)
    spec = ReductionSpec(r=2, test_basis=FakeTestBasisSpecWithQ())
    dataset_artifact = make_dataset_artifact()

    store.get_or_create(
        dataset_artifact=dataset_artifact,
        system="system",
        reduction_spec=spec,
    )

    assert len(q_store.calls) == 1
    q_call = q_store.calls[0]
    assert q_call["source"] == "Hamiltonian_red"
    assert q_call["data_train_id"] == "train-id"
    assert q_call["reduction_spec"] is spec
    assert q_call["options"] == {"alpha": 1}
    assert full_basis.truncate_calls == [2]

    test_basis = FakeReducer.instances[0].test_basis
    np.testing.assert_allclose(
        test_basis.compute_q_kwargs["V"],
        full_basis.basis.V,
    )
    assert test_basis.compute_q_kwargs["system"] == "system"
    assert test_basis.compute_q_kwargs["data_train"] == "train-data"
    assert FakeReducer.instances[0].reduce_calls[0]["test_basis_kwargs"].keys() == {
        "Q"
    }
    np.testing.assert_allclose(
        FakeReducer.instances[0].reduce_calls[0]["test_basis_kwargs"]["Q"],
        np.eye(2) * 2.0,
    )


def test_get_or_create_uses_cached_q_without_recomputing_q(
    tmp_path, monkeypatch
) -> None:
    FakeReducer.instances = []
    monkeypatch.setattr(reduction_store_module, "Reducer", FakeReducer)
    full_basis = FakeFullBasis()
    q_store = FakeQMatrixStore(call_compute_fn=False)
    store = make_store(tmp_path, full_basis=full_basis, q_store=q_store)
    spec = ReductionSpec(r=2, test_basis=FakeTestBasisSpecWithQ())

    store.get_or_create(
        dataset_artifact=make_dataset_artifact(),
        system="system",
        reduction_spec=spec,
    )

    assert full_basis.truncate_calls == []
    np.testing.assert_allclose(
        FakeReducer.instances[0].reduce_calls[0]["test_basis_kwargs"]["Q"],
        np.eye(2) * 3.0,
    )


def test_get_or_create_with_real_qv_test_basis_hits_q_cache(
    tmp_path, monkeypatch
) -> None:
    FakeReducer.instances = []
    monkeypatch.setattr(reduction_store_module, "Reducer", FakeReducer)
    q_store = FakeQMatrixStore(call_compute_fn=False)
    store = make_store(tmp_path, q_store=q_store)
    spec = ReductionSpec(r=2, test_basis=QVTestBasisSpec(source="system_Q"))

    store.get_or_create(
        dataset_artifact=make_dataset_artifact(),
        system="system",
        reduction_spec=spec,
    )

    assert q_store.calls[0]["source"] == "system_Q"
    assert q_store.calls[0]["options"] == spec.test_basis.build().q_cache_options()
