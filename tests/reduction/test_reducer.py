from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pgopinf.data.dataset import DataSet
from pgopinf.reduction.interfaces import BasisArtifact
from pgopinf.reduction.reducer import Reducer


class FakeFullBasis:
    def __init__(self, basis):
        self.basis = basis
        self.truncate_calls = []

    def truncate(self, r):
        self.truncate_calls.append(r)
        return self.basis


class FakeTestBasis:
    def __init__(self):
        self.make_w_calls = []

    def make_W(self, **kwargs):
        self.make_w_calls.append(kwargs)
        return np.array([[2.0], [0.0]])


class FakeMOR:
    def __init__(self):
        self.project_data_calls = []
        self.reduce_system_calls = []

    def project_data(self, **kwargs):
        self.project_data_calls.append(kwargs)
        return f"reduced-{kwargs['data']}"

    def reduce_system(self, **kwargs):
        self.reduce_system_calls.append(kwargs)
        return "reduced-system"


@dataclass(frozen=True)
class FakeTestBasisSpec:
    test_basis: FakeTestBasis

    def build(self):
        return self.test_basis


@dataclass(frozen=True)
class FakeMORSpec:
    mor: FakeMOR

    def build(self):
        return self.mor


@dataclass(frozen=True)
class FakeReductionSpec:
    r: int
    test_basis: FakeTestBasisSpec
    mor: FakeMORSpec
    project_data: bool = True
    reduce_system: bool = True


def test_reducer_builds_missing_left_basis_projects_data_and_reduces_system() -> None:
    test_basis = FakeTestBasis()
    mor = FakeMOR()
    spec = FakeReductionSpec(
        r=1,
        test_basis=FakeTestBasisSpec(test_basis),
        mor=FakeMORSpec(mor),
    )
    basis = BasisArtifact(V=np.array([[1.0], [0.0]]), W=None, meta={"basis": True})
    full_basis = FakeFullBasis(basis)
    dataset = DataSet(train="train", test="test")
    system = object()
    reducer = Reducer(spec)
    result = reducer.reduce(
        dataset=dataset,
        system=system,
        full_basis=full_basis,
        test_basis_kwargs={"Q": np.eye(2)},
    )

    assert full_basis.truncate_calls == [1]
    assert test_basis.make_w_calls[0]["V"] is basis.V
    assert test_basis.make_w_calls[0]["data_train"] == "train"
    assert test_basis.make_w_calls[0]["system"] is system
    np.testing.assert_allclose(test_basis.make_w_calls[0]["Q"], np.eye(2))
    np.testing.assert_allclose(result.basis.W, np.array([[2.0], [0.0]]))
    assert result.basis.V is basis.V
    assert result.basis.meta == basis.meta
    assert result.reduced_dataset_projected.TRAIN == "reduced-train"
    assert result.reduced_dataset_projected.TEST == "reduced-test"
    assert result.reduced_system == "reduced-system"
    assert result.full_basis is full_basis
    assert result.mor is mor


def test_reducer_reuses_existing_left_basis_and_can_skip_outputs() -> None:
    test_basis = FakeTestBasis()
    mor = FakeMOR()
    spec = FakeReductionSpec(
        r=1,
        test_basis=FakeTestBasisSpec(test_basis),
        mor=FakeMORSpec(mor),
        project_data=False,
        reduce_system=False,
    )
    basis = BasisArtifact(V=np.eye(2, 1), W=np.ones((2, 1)), meta={})
    full_basis = FakeFullBasis(basis)
    reducer = Reducer(spec)  # type: ignore[arg-type]

    result = reducer.reduce(
        dataset=DataSet(train="train", test="test"),
        system=object(),
        full_basis=full_basis,
    )

    assert test_basis.make_w_calls == []
    assert mor.project_data_calls == []
    assert mor.reduce_system_calls == []
    assert result.basis is basis
    assert result.reduced_dataset_projected is None
    assert result.reduced_system is None
