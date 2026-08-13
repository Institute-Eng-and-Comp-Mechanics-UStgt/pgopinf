from __future__ import annotations

import numpy as np
import pytest

from pgopinf.data.data import Data
from pgopinf.data.initial_condition.initial_condition import InitialCondition
from pgopinf.data.reduced_data import ReducedData
from pgopinf.data.time.time import Time
from pgopinf.reduction.interfaces import BasisArtifact
from pgopinf.reduction.project_data import (
    define_reduction_projector,
    project_data_linear,
    project_ic_linear,
)


def make_data() -> Data:
    X = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
            [[7.0], [8.0], [9.0]],
        ]
    )
    U = np.array([[[0.0], [1.0], [2.0]]])
    Y = np.array([[[10.0], [11.0], [12.0]]])
    dXdt = np.ones_like(X)
    return Data(
        time=Time(np.array([0.0, 1.0, 2.0])),
        X=X,
        U=U,
        Y=Y,
        dXdt=dXdt,
        deriv_method="return",
    )


def test_define_reduction_projector_uses_transpose_when_biorthonormal() -> None:
    V = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    W = V.copy()

    np.testing.assert_allclose(define_reduction_projector(V, W), W.T)


def test_define_reduction_projector_solves_for_non_biorthonormal_basis() -> None:
    V = np.array([[2.0], [0.0]])
    W = np.array([[1.0], [0.0]])

    projector = define_reduction_projector(V, W)

    np.testing.assert_allclose(projector, np.array([[0.5, 0.0]]))
    np.testing.assert_allclose(projector @ V, np.eye(1))


def test_project_data_linear_projects_state_and_derivative_to_reduced_data() -> None:
    data = make_data()
    V = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    W = V.copy()
    basis = BasisArtifact(V=V, W=W, meta={"kind": "basis"})

    reduced = project_data_linear(data=data, basis=basis)

    assert isinstance(reduced, ReducedData)
    assert reduced.meta == "projected"
    assert reduced.V is V
    np.testing.assert_allclose(reduced.X, data.X[:2])
    np.testing.assert_allclose(reduced.dXdt, data.dXdt[:2])
    np.testing.assert_allclose(reduced.U, data.U)
    np.testing.assert_allclose(reduced.Y, data.Y)


def test_project_data_linear_requires_left_basis() -> None:
    with pytest.raises(ValueError, match="W must be provided"):
        project_data_linear(
            data=make_data(),
            basis=BasisArtifact(V=np.eye(3, 2), W=None, meta={}),
        )


def test_project_ic_linear_projects_initial_condition() -> None:
    ic = InitialCondition(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    basis = BasisArtifact(
        V=np.array([[1.0], [0.0], [0.0]]),
        W=np.array([[1.0], [0.0], [1.0]]),
        meta={},
    )

    x0r = project_ic_linear(ic=ic, basis=basis)

    np.testing.assert_allclose(x0r, np.array([[6.0, 8.0]]))


def test_project_ic_linear_requires_left_basis() -> None:
    with pytest.raises(ValueError, match="W must be provided"):
        project_ic_linear(
            ic=InitialCondition(np.ones((3, 1))),
            basis=BasisArtifact(V=np.eye(3, 1), W=None, meta={}),
        )
