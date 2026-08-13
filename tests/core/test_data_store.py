from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from pgopinf.core import data_store as data_store_module
from pgopinf.core.data_store import (
    DataStore,
    IdentifiedDataStore,
    IntrusiveDataStore,
)
from pgopinf.data.data import Data
from pgopinf.data.reduced_data import ReducedData
from pgopinf.data.time.time import Time
from pgopinf.identification.identifier import IdentificationResult
from pgopinf.io.results_path import ResultsPath
from pgopinf.reduction.interfaces import BasisArtifact
from pgopinf.reduction.reducer import ReductionResult
from pgopinf.specs.data.base import DataSpec, SplitDataSpec
from pgopinf.specs.data.time.base import TimeSpec
from pgopinf.specs.identification.operator_inference import (
    OperatorInferenceSpec,
)
from pgopinf.specs.reduction.base import ReductionSpec
from pgopinf.specs.system.msd import MassSpringDamperSpec


def make_paths(tmp_path) -> ResultsPath:
    return ResultsPath(results_root=tmp_path / "results")


def make_data(n_t: int = 4, *, with_optional_arrays: bool = True) -> Data:
    t = np.linspace(0.0, float(n_t - 1), n_t)
    X = np.arange(2 * n_t, dtype=float).reshape(2, n_t, 1)
    U = np.arange(n_t, dtype=float).reshape(1, n_t, 1)
    Y = np.arange(n_t, dtype=float).reshape(1, n_t, 1) + 10.0
    dXdt = np.ones_like(X)
    return Data(
        time=Time(t),
        X=X,
        U=U if with_optional_arrays else None,
        Y=Y if with_optional_arrays else None,
        dXdt=dXdt,
        deriv_method="return",
    )


def make_reduced_data(n_t: int = 4) -> ReducedData:
    t = np.linspace(0.0, float(n_t - 1), n_t)
    X = np.arange(2 * n_t, dtype=float).reshape(2, n_t, 1)
    U = np.arange(n_t, dtype=float).reshape(1, n_t, 1)
    Y = np.arange(n_t, dtype=float).reshape(1, n_t, 1) + 20.0
    dXdt = np.ones_like(X)
    return ReducedData(
        time=Time(t),
        X=X,
        U=U,
        Y=Y,
        dXdt=dXdt,
        deriv_method="return",
        V=np.eye(3, 2),
    )


def make_data_spec(*, reduce_sample_n: int | None = None) -> DataSpec:
    split = SplitDataSpec(
        time=TimeSpec(t0=0.0, t_end=3.0, n_steps=4),
        reduce_sample_n=reduce_sample_n,
    )
    return DataSpec(train=split, test=split, seed=10)


def assert_data_equal(left: Data, right: Data) -> None:
    np.testing.assert_allclose(left.time.t, right.time.t)
    np.testing.assert_allclose(left.X, right.X)
    if left.U is None:
        assert right.U is None
    else:
        np.testing.assert_allclose(left.U, right.U)
    if left.Y is None:
        assert right.Y is None
    else:
        np.testing.assert_allclose(left.Y, right.Y)
    np.testing.assert_allclose(left.dXdt, right.dXdt)
    assert left.deriv_method == right.deriv_method


def assert_reduced_data_equal(left: ReducedData, right: ReducedData) -> None:
    assert_data_equal(left, right)
    np.testing.assert_allclose(left.V, right.V)


def test_data_store_split_ids_ignore_reduce_sample_n_only_for_full_cache(
    tmp_path,
) -> None:
    store = DataStore(make_paths(tmp_path))
    system_spec = MassSpringDamperSpec(n_mass=3)
    data_spec_a = make_data_spec(reduce_sample_n=2)
    data_spec_b = make_data_spec(reduce_sample_n=3)

    assert store.compute_split_id_full(
        system_spec=system_spec,
        data_spec=data_spec_a,
        split="train",
    ) == store.compute_split_id_full(
        system_spec=system_spec,
        data_spec=data_spec_b,
        split="train",
    )
    assert store.compute_split_id_explicit(
        system_spec=system_spec,
        data_spec=data_spec_a,
        split="train",
    ) != store.compute_split_id_explicit(
        system_spec=system_spec,
        data_spec=data_spec_b,
        split="train",
    )


def test_data_store_save_and_load_split_round_trip(tmp_path) -> None:
    paths = make_paths(tmp_path)
    store = DataStore(paths)
    system_spec = MassSpringDamperSpec(n_mass=3)
    split_spec = make_data_spec().train
    data = make_data()

    store.save_split(
        "abc",
        system_spec=system_spec,
        split_spec=split_spec,
        split="train",
        seed=10,
        data=data,
    )

    assert store.split_exists("abc")
    assert_data_equal(store.load_split("abc", split_spec), data)
    assert json.loads((paths.dataset_dir("abc") / "meta.json").read_text()) == {
        "artifact_format_version": 1,
        "dataset_id": "abc",
        "split": "train",
    }
    spec_json = json.loads((paths.dataset_dir("abc") / "spec.json").read_text())
    assert spec_json["system"]["kind"] == "msd"
    assert spec_json["materialization"] == {"kind": "full"}


def test_data_store_loads_optional_none_arrays(tmp_path) -> None:
    store = DataStore(make_paths(tmp_path))
    split_spec = make_data_spec().train
    data = make_data(with_optional_arrays=False)

    store.save_split(
        "none-arrays",
        system_spec=MassSpringDamperSpec(),
        split_spec=split_spec,
        split="test",
        seed=11,
        data=data,
    )

    loaded = store.load_split("none-arrays", split_spec)
    assert loaded.U is None
    assert loaded.Y is None
    assert loaded.deriv_method == "return"


def test_data_store_get_or_create_generates_then_reuses_cached_full_split(
    tmp_path, monkeypatch
) -> None:
    store = DataStore(make_paths(tmp_path))
    system_spec = MassSpringDamperSpec(n_mass=3)
    data_spec = make_data_spec(reduce_sample_n=2)
    generated = make_data(n_t=4)
    calls = []

    def fake_generate_split_data(**kwargs):
        calls.append(kwargs)
        return generated

    monkeypatch.setattr(
        data_store_module, "generate_split_data", fake_generate_split_data
    )

    first = store.get_or_create_split(
        system_spec=system_spec,
        system=object(),
        data_spec=data_spec,
        split="TRAIN",
    )
    second = store.get_or_create_split(
        system_spec=system_spec,
        system=object(),
        data_spec=data_spec,
        split="train",
    )

    assert len(calls) == 1
    assert calls[0]["seed"] == 10
    assert calls[0]["split_name"] == "train"
    assert first.id == second.id
    assert first.data.n_t == 2
    assert second.data.n_t == 2


def test_data_store_get_or_create_builds_train_and_test(tmp_path, monkeypatch) -> None:
    store = DataStore(make_paths(tmp_path))
    data_spec = make_data_spec()

    def fake_generate_split_data(**kwargs):
        return make_data()

    monkeypatch.setattr(
        data_store_module, "generate_split_data", fake_generate_split_data
    )

    artifact = store.get_or_create(
        system_spec=MassSpringDamperSpec(),
        system=object(),
        data_spec=data_spec,
    )

    assert artifact.train.split == "train"
    assert artifact.test.split == "test"
    assert artifact.train_id != artifact.test_id
    assert artifact.data.TRAIN is artifact.train.data
    assert artifact.data.TEST is artifact.test.data


def test_intrusive_data_store_save_and_load_split_round_trip(tmp_path) -> None:
    paths = make_paths(tmp_path)
    store = IntrusiveDataStore(paths)
    split_spec = make_data_spec().train
    data = make_reduced_data()

    store.save_split(
        "intrusive",
        system_spec=MassSpringDamperSpec(),
        split_spec=split_spec,
        reduction_spec=ReductionSpec(r=2),
        split="train",
        seed=10,
        data=data,
    )

    assert store.split_exists("intrusive")
    assert_reduced_data_equal(store.load_split("intrusive", split_spec), data)
    spec_json = json.loads(
        (paths.intrusive_dataset_dir("intrusive") / "spec.json").read_text()
    )
    assert spec_json["reduction"]["r"] == 2


def test_intrusive_data_store_get_or_create_reuses_cached_full_split(
    tmp_path, monkeypatch
) -> None:
    store = IntrusiveDataStore(make_paths(tmp_path))
    data_spec = make_data_spec(reduce_sample_n=2)
    reduction_spec = ReductionSpec(r=2)
    reduction_id = "reduction-123"
    calls = []

    def fake_generate_reduced_split_data(**kwargs):
        calls.append(kwargs)
        return make_reduced_data(n_t=4)

    monkeypatch.setattr(
        data_store_module,
        "generate_reduced_split_data",
        fake_generate_reduced_split_data,
    )
    reduction_result = ReductionResult(
        basis=BasisArtifact(V=np.eye(3, 2), W=None, meta={}),
        mor=SimpleNamespace(),
        reduced_dataset_projected=None,
        reduced_system=object(),
    )

    first = store.get_or_create_split(
        reduction_id=reduction_id,
        system_spec=MassSpringDamperSpec(),
        data_spec=data_spec,
        reduction_spec=reduction_spec,
        reduction_result=reduction_result,
        split="train",
    )
    second = store.get_or_create_split(
        reduction_id=reduction_id,
        system_spec=MassSpringDamperSpec(),
        data_spec=data_spec,
        reduction_spec=reduction_spec,
        reduction_result=reduction_result,
        split="train",
    )

    assert len(calls) == 1
    assert (
        first.id
        == second.id
        == store.compute_split_id_explicit(
            reduction_id=reduction_id,
            data_spec=data_spec,
            split="train",
        )
    )
    assert first.data.n_t == 2
    assert second.data.n_t == 2


def test_identified_data_store_save_and_load_split_round_trip(tmp_path) -> None:
    paths = make_paths(tmp_path)
    store = IdentifiedDataStore(paths)
    split_spec = make_data_spec().test
    data = make_reduced_data()

    store.save_split(
        "identified",
        system_spec=MassSpringDamperSpec(),
        split_spec=split_spec,
        reduction_spec=ReductionSpec(r=2),
        identification_spec=OperatorInferenceSpec(lambda_reg=1e-3),
        split="test",
        seed=11,
        data=data,
    )

    assert store.split_exists("identified")
    assert_reduced_data_equal(store.load_split("identified", split_spec), data)
    spec_json = json.loads(
        (paths.identified_dataset_dir("identified") / "spec.json").read_text()
    )
    assert spec_json["identification"]["lambda_reg"] == 1e-3


def test_identified_data_store_get_or_create_reuses_cached_full_split(
    tmp_path, monkeypatch
) -> None:
    store = IdentifiedDataStore(make_paths(tmp_path))
    data_spec = make_data_spec(reduce_sample_n=2)
    reduction_spec = ReductionSpec(r=2)
    identification_spec = OperatorInferenceSpec(lambda_reg=1e-3)
    reduction_id = "reduction-123"
    calls = []

    def fake_generate_reduced_split_data(**kwargs):
        calls.append(kwargs)
        return make_reduced_data(n_t=4)

    monkeypatch.setattr(
        data_store_module,
        "generate_reduced_split_data",
        fake_generate_reduced_split_data,
    )
    reduction_result = ReductionResult(
        basis=BasisArtifact(V=np.eye(3, 2), W=None, meta={}),
        mor=SimpleNamespace(),
        reduced_dataset_projected=None,
        reduced_system=object(),
    )
    identification_result = IdentificationResult(
        system_identified=object(),
        diagnostics={},
        meta={},
    )

    first = store.get_or_create_split(
        reduction_id=reduction_id,
        system_spec=MassSpringDamperSpec(),
        data_spec=data_spec,
        reduction_spec=reduction_spec,
        identification_spec=identification_spec,
        identification_result=identification_result,
        reduction_result=reduction_result,
        split="test",
    )
    second = store.get_or_create_split(
        reduction_id=reduction_id,
        system_spec=MassSpringDamperSpec(),
        data_spec=data_spec,
        reduction_spec=reduction_spec,
        identification_spec=identification_spec,
        identification_result=identification_result,
        reduction_result=reduction_result,
        split="test",
    )

    assert len(calls) == 1
    assert calls[0]["seed"] == 11
    assert calls[0]["split_name"] == "test"
    assert (
        first.id
        == second.id
        == store.compute_split_id_explicit(
            reduction_id=reduction_id,
            data_spec=data_spec,
            identification_spec=identification_spec,
            split="test",
        )
    )
    assert first.data.n_t == 2
    assert second.data.n_t == 2
