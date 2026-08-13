from __future__ import annotations

import numpy as np
import pytest

from pgopinf.data.reduced_data import ReducedData
from pgopinf.data.time.time import Time


def make_reduced_arrays():
    X = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            [[4.0, 40.0], [5.0, 50.0], [6.0, 60.0]],
        ]
    )
    U = np.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    Y = np.array([[[7.0, 70.0], [8.0, 80.0], [9.0, 90.0]]])
    dXdt = np.ones_like(X)
    return X, U, Y, dXdt


def make_reduced_data(**kwargs) -> ReducedData:
    X, U, Y, dXdt = make_reduced_arrays()
    return ReducedData(
        time=kwargs.pop("time", Time(np.array([0.0, 1.0, 2.0]))),
        X=kwargs.pop("X", X.copy()),
        U=kwargs.pop("U", U.copy()),
        Y=kwargs.pop("Y", Y.copy()),
        dXdt=kwargs.pop("dXdt", dXdt.copy()),
        deriv_method=kwargs.pop("deriv_method", "return"),
        V=kwargs.pop("V", np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])),
        **kwargs,
    )


def test_reduced_data_sets_reduced_dimension_from_basis() -> None:
    data = make_reduced_data(meta="integrated")

    assert data.r == 2
    assert data.n == 2
    assert data.meta == "integrated"


def test_reduced_data_rejects_basis_with_wrong_reduced_dimension() -> None:
    with pytest.raises(ValueError, match="ReducedData.r=3 but X has first dim 2"):
        make_reduced_data(V=np.eye(4, 3))


def test_reduced_data_rejects_non_matrix_basis() -> None:
    with pytest.raises(ValueError, match="V must have shape"):
        make_reduced_data(V=np.ones(2))


def test_reduced_data_allows_missing_basis_and_has_no_full_state_view() -> None:
    data = make_reduced_data(V=None)

    assert data.r == 2
    assert data.X_full is None
    assert data.dXdt_full is None


def test_x_full_lifts_reduced_trajectories_through_basis() -> None:
    data = make_reduced_data()
    expected = np.einsum("ij,jtk->itk", data.V, data.X)

    np.testing.assert_allclose(data.X_full, expected)
    np.testing.assert_allclose(data.X_full[2], data.X[0] + data.X[1])


def test_dxdt_full_lifts_reduced_derivatives_through_basis() -> None:
    data = make_reduced_data()
    expected = np.einsum("ij,jtk->itk", data.V, data.dXdt)

    np.testing.assert_allclose(data.dXdt_full, expected)
    np.testing.assert_allclose(data.dXdt_full[2], 2.0 * np.ones_like(data.X[0]))


def test_dxdt_full_is_none_when_derivative_is_unavailable() -> None:
    data = make_reduced_data(dXdt=np.ones((2, 3, 2)), deriv_method="return")
    data.dXdt = None

    assert data.dXdt_full is None


def test_reduced_data_uses_data_midpoint_derivative_path() -> None:
    X, U, Y, _ = make_reduced_arrays()

    data = ReducedData(
        time=Time(np.array([0.0, 1.0, 2.0])),
        X=X.copy(),
        U=U.copy(),
        Y=Y.copy(),
        dXdt=None,
        deriv_method="midpoints",
        V=np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
    )

    np.testing.assert_allclose(data.time.t, np.array([0.5, 1.5]))
    np.testing.assert_allclose(data.X, 0.5 * (X[:, 1:, :] + X[:, :-1, :]))
    np.testing.assert_allclose(data.dXdt, X[:, 1:, :] - X[:, :-1, :])
    assert data.r == 2
