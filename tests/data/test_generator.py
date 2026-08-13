from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pgopinf.data import generator as generator_module
from pgopinf.data.data import Data
from pgopinf.data.dataset import DataSet
from pgopinf.data.generator import (
    generate_dataset,
    generate_reduced_dataset,
    generate_reduced_split_data,
    generate_split_data,
    simulate_trajectories,
)
from pgopinf.data.reduced_data import ReducedData
from pgopinf.reduction.interfaces import BasisArtifact
from pgopinf.specs.data.base import DataSpec, SplitDataSpec
from pgopinf.specs.data.inputs.base import ExprInputSpec
from pgopinf.specs.data.initial_condition.base import ZerosIC
from pgopinf.specs.data.time.base import TimeSpec


class FakeSystem:
    n = 2
    n_y = 1

    def __init__(self):
        self.solve_calls = []

    def solve(
        self,
        t,
        U,
        x0,
        *,
        integrator,
        decomp_option,
        return_dXdt,
    ):
        self.solve_calls.append(
            {
                "t": np.asarray(t).copy(),
                "U": np.asarray(U).copy(),
                "x0": np.asarray(x0).copy(),
                "integrator": integrator,
                "decomp_option": decomp_option,
                "return_dXdt": return_dXdt,
            }
        )
        t = np.asarray(t)
        X = np.vstack([x0[0] + t, x0[1] + 2.0 * t])
        Y = X[:1, :] + U[:1, :]
        if return_dXdt:
            dXdt = np.vstack([np.ones_like(t), 2.0 * np.ones_like(t)])
            return X, Y, dXdt
        return X, Y


class FakeReducedSystem(FakeSystem):
    n = 1
    n_y = 1

    def solve(
        self,
        t,
        U,
        x0,
        *,
        integrator,
        decomp_option,
        return_dXdt,
    ):
        self.solve_calls.append(
            {
                "t": np.asarray(t).copy(),
                "U": np.asarray(U).copy(),
                "x0": np.asarray(x0).copy(),
                "integrator": integrator,
                "decomp_option": decomp_option,
                "return_dXdt": return_dXdt,
            }
        )
        t = np.asarray(t)
        X = (x0[0] + t)[np.newaxis, :]
        Y = X + U[:1, :]
        if return_dXdt:
            return X, Y, np.ones_like(X)
        return X, Y


class FakeMOR:
    def __init__(self):
        self.project_initial_condition_calls = []

    def project_initial_condition(self, *, ic, basis):
        self.project_initial_condition_calls.append({"ic": ic, "basis": basis})
        return ic.x0[:1, :]


@dataclass
class FakeSystemSpec:
    system: FakeSystem

    def build(self):
        return self.system


def make_split_spec(
    *,
    n_sim: int = 2,
    dXdt_derivation: str = "return",
) -> SplitDataSpec:
    return SplitDataSpec(
        time=TimeSpec(t0=0.0, t_end=2.0, n_steps=3),
        initial_condition=ZerosIC(),
        input=ExprInputSpec(expr={"kind": "constant", "value": 3.0}),
        n_sim=n_sim,
        dXdt_derivation=dXdt_derivation,
    )


def assert_data_shapes(data: Data, *, n: int, n_t: int, n_sim: int) -> None:
    assert data.X.shape == (n, n_t, n_sim)
    assert data.U.shape == (1, n_t, n_sim)
    assert data.Y.shape == (1, n_t, n_sim)


def test_simulate_trajectories_with_returned_derivative() -> None:
    system = FakeSystem()
    split_spec = make_split_spec(n_sim=2, dXdt_derivation="return")
    time = split_spec.time.build()
    U = np.ones((1, time.n_t, 2))
    x0 = np.array([[0.0, 10.0], [1.0, 20.0]])

    X, Y, dXdt, deriv_method = simulate_trajectories(
        system=system,
        time=time,
        U=U,
        x0=x0,
        split_spec=split_spec,
    )

    assert X.shape == (2, 3, 2)
    assert Y.shape == (1, 3, 2)
    assert dXdt.shape == (2, 3, 2)
    assert deriv_method is None
    assert [call["return_dXdt"] for call in system.solve_calls] == [True, True]


def test_simulate_trajectories_without_returned_derivative() -> None:
    system = FakeSystem()
    split_spec = make_split_spec(n_sim=1, dXdt_derivation="finite_difference")
    time = split_spec.time.build()
    U = np.ones((1, time.n_t, 1))
    x0 = np.array([[0.0], [1.0]])

    X, Y, dXdt, deriv_method = simulate_trajectories(
        system=system,
        time=time,
        U=U,
        x0=x0,
        split_spec=split_spec,
    )

    assert X.shape == (2, 3, 1)
    assert Y.shape == (1, 3, 1)
    assert dXdt is None
    assert deriv_method == "finite_difference"
    assert system.solve_calls[0]["return_dXdt"] is False


def test_generate_split_data_returns_data_with_metadata() -> None:
    system = FakeSystem()

    data = generate_split_data(
        system=system,
        split_spec=make_split_spec(n_sim=2),
        seed=123,
        split_name="train",
    )

    assert isinstance(data, Data)
    assert_data_shapes(data, n=2, n_t=3, n_sim=2)
    assert data.meta == {
        "split": "train",
        "seed": 123,
        "dXdt_derivation": "return",
    }
    np.testing.assert_allclose(data.U, np.full((1, 3, 2), 3.0))


def test_generate_reduced_split_data_projects_initial_condition_and_returns_reduced_data() -> None:
    reduced_system = FakeReducedSystem()
    mor = FakeMOR()
    basis = BasisArtifact(V=np.array([[1.0], [0.0]]), W=None, meta={})

    data = generate_reduced_split_data(
        reduced_system=reduced_system,
        split_spec=make_split_spec(n_sim=2),
        seed=123,
        split_name="train",
        mor=mor,
        basis=basis,
    )

    assert isinstance(data, ReducedData)
    assert data.meta == "integrated"
    assert data.V is basis.V
    assert_data_shapes(data, n=1, n_t=3, n_sim=2)
    assert mor.project_initial_condition_calls[0]["basis"] is basis


def test_generate_reduced_split_data_requires_basis_when_mor_is_given() -> None:
    with pytest.raises(ValueError, match="Basis must be provided"):
        generator_module._generate_split(
            system=FakeReducedSystem(),
            split_spec=make_split_spec(n_sim=1),
            seed=0,
            split_name="train",
            mor=FakeMOR(),
            basis=None,
        )


def test_generate_dataset_builds_system_and_uses_train_and_test_seeds(monkeypatch) -> None:
    calls = []
    system = FakeSystem()

    def fake_generate_split_data(**kwargs):
        calls.append(kwargs)
        return f"{kwargs['split_name']}-data"

    monkeypatch.setattr(
        generator_module,
        "generate_split_data",
        fake_generate_split_data,
    )
    data_spec = DataSpec(
        train=make_split_spec(n_sim=1),
        test=make_split_spec(n_sim=1),
        seed=5,
    )

    dataset = generate_dataset(system_spec=FakeSystemSpec(system), data_spec=data_spec)

    assert isinstance(dataset, DataSet)
    assert dataset.TRAIN == "train-data"
    assert dataset.TEST == "test-data"
    assert calls == [
        {
            "system": system,
            "split_spec": data_spec.train,
            "seed": 5,
            "split_name": "train",
        },
        {
            "system": system,
            "split_spec": data_spec.test,
            "seed": 6,
            "split_name": "test",
        },
    ]


def test_generate_reduced_dataset_uses_train_and_test_seeds(monkeypatch) -> None:
    calls = []

    def fake_generate_reduced_split_data(**kwargs):
        calls.append(kwargs)
        return f"{kwargs['split_name']}-reduced"

    monkeypatch.setattr(
        generator_module,
        "generate_reduced_split_data",
        fake_generate_reduced_split_data,
    )
    data_spec = DataSpec(
        train=make_split_spec(n_sim=1),
        test=make_split_spec(n_sim=1),
        seed=7,
    )
    mor = FakeMOR()
    basis = BasisArtifact(V=np.array([[1.0], [0.0]]), W=None, meta={})
    reduced_system = FakeReducedSystem()

    dataset = generate_reduced_dataset(
        reduced_system=reduced_system,
        data_spec=data_spec,
        mor=mor,
        basis=basis,
    )

    assert dataset.TRAIN == "train-reduced"
    assert dataset.TEST == "test-reduced"
    assert calls == [
        {
            "reduced_system": reduced_system,
            "split_spec": data_spec.train,
            "seed": 7,
            "split_name": "train",
            "mor": mor,
            "basis": basis,
        },
        {
            "reduced_system": reduced_system,
            "split_spec": data_spec.test,
            "seed": 8,
            "split_name": "test",
            "mor": mor,
            "basis": basis,
        },
    ]
