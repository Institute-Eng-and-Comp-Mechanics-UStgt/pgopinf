from __future__ import annotations

import numpy as np
import pytest

from pgopinf.data.data import Data
from pgopinf.data.time.time import Time


def make_time(n_t: int = 4) -> Time:
    return Time(np.arange(n_t, dtype=float))


def make_arrays():
    X = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]],
        ]
    )
    U = np.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]])
    Y = np.array([[[5.0, 50.0], [6.0, 60.0], [7.0, 70.0], [8.0, 80.0]]])
    dXdt = np.ones_like(X)
    return X, U, Y, dXdt


def make_data(**kwargs) -> Data:
    X, U, Y, dXdt = make_arrays()
    return Data(
        time=kwargs.pop("time", make_time()),
        X=kwargs.pop("X", X.copy()),
        U=kwargs.pop("U", U.copy()),
        Y=kwargs.pop("Y", Y.copy()),
        dXdt=kwargs.pop("dXdt", dXdt.copy()),
        deriv_method=kwargs.pop("deriv_method", "return"),
        **kwargs,
    )


def test_data_properties_and_gradient_derivative() -> None:
    X, U, Y, _ = make_arrays()
    data = Data(time=make_time(), X=X.copy(), U=U.copy(), Y=Y.copy())

    assert data.n == 2
    assert data.n_t == 4
    assert data.n_sim == 2
    assert data.n_u == 1
    assert data.n_y == 1
    assert data.dt == pytest.approx(1.0)
    np.testing.assert_allclose(data.t, np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(data.dXdt, np.gradient(X, 1.0, axis=1))


def test_midpoint_derivative_averages_values_and_uses_midpoint_time_grid() -> None:
    X, U, Y, _ = make_arrays()

    data = Data(
        time=make_time(),
        X=X.copy(),
        U=U.copy(),
        Y=Y.copy(),
        dXdt=None,
        deriv_method="midpoints",
    )

    np.testing.assert_allclose(data.time.t, np.array([0.5, 1.5, 2.5]))
    np.testing.assert_allclose(data.dXdt, X[:, 1:, :] - X[:, :-1, :])
    np.testing.assert_allclose(data.X, 0.5 * (X[:, 1:, :] + X[:, :-1, :]))
    np.testing.assert_allclose(data.U, 0.5 * (U[:, 1:, :] + U[:, :-1, :]))
    np.testing.assert_allclose(data.Y, 0.5 * (Y[:, 1:, :] + Y[:, :-1, :]))


def test_unknown_derivative_method_raises() -> None:
    X, U, Y, _ = make_arrays()

    with pytest.raises(ValueError, match="Unknown deriv_method"):
        Data(time=make_time(), X=X, U=U, Y=Y, dXdt=None, deriv_method="missing")


def test_validation_rejects_incompatible_shapes() -> None:
    X, U, Y, dXdt = make_arrays()

    with pytest.raises(ValueError, match="time.t must be 1D"):
        Data(
            time=type("BadTime", (), {"t": np.zeros((2, 2)), "dt": 1.0})(),
            X=X,
            U=U,
            Y=Y,
            dXdt=dXdt,
        )

    with pytest.raises(ValueError, match="X must have shape"):
        Data(time=make_time(), X=X[:, :, 0], U=U, Y=Y, dXdt=dXdt)

    with pytest.raises(ValueError, match="X has n_t=4 but time.t has 3"):
        Data(time=make_time(3), X=X, U=U, Y=Y, dXdt=dXdt)

    with pytest.raises(ValueError, match="U must have shape"):
        Data(time=make_time(), X=X, U=U[:, :-1, :], Y=Y, dXdt=dXdt)

    with pytest.raises(ValueError, match="Y must have shape"):
        Data(time=make_time(), X=X, U=U, Y=Y[:, :-1, :], dXdt=dXdt)


def test_convert_to_sample_format_uses_fortran_order() -> None:
    data = make_data()

    x, u, y, dxdt = data.convert_to_sample_format()

    np.testing.assert_allclose(
        x,
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
                [5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0],
            ]
        ),
    )
    np.testing.assert_allclose(u, np.array([[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]]))
    np.testing.assert_allclose(y, np.array([[5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0]]))
    np.testing.assert_allclose(dxdt, np.ones((2, 8)))
    np.testing.assert_allclose(data.x, x)
    np.testing.assert_allclose(data.u, u)
    np.testing.assert_allclose(data.y, y)
    np.testing.assert_allclose(data.dxdt, dxdt)


def test_convert_to_time_step_format_round_trips_sample_format() -> None:
    data = make_data()
    original = (data.X.copy(), data.U.copy(), data.Y.copy(), data.dXdt.copy())

    data.convert_to_time_step_format()

    np.testing.assert_allclose(data.X, original[0])
    np.testing.assert_allclose(data.U, original[1])
    np.testing.assert_allclose(data.Y, original[2])
    np.testing.assert_allclose(data.dXdt, original[3])


def test_static_format_converters_support_optional_arrays() -> None:
    X, _, _, dXdt = make_arrays()

    x, u, y, dxdt = Data.convert_XUY_to_sample_format(X, None, None, dXdt)
    assert u is None
    assert y is None
    np.testing.assert_allclose(x, np.reshape(X, (2, 8), order="F"))
    np.testing.assert_allclose(dxdt, np.reshape(dXdt, (2, 8), order="F"))

    X_round, U_round, Y_round, dXdt_round = Data.convert_xuy_to_time_step_format(
        x=x,
        u=None,
        y=None,
        dxdt=dxdt,
        n=2,
        n_u=0,
        n_y=0,
        n_t=4,
        n_sim=2,
    )
    np.testing.assert_allclose(X_round, X)
    assert U_round is None
    assert Y_round is None
    np.testing.assert_allclose(dXdt_round, dXdt)


def test_optional_input_and_output_arrays_are_supported() -> None:
    X, _, _, dXdt = make_arrays()
    data = Data(
        time=make_time(),
        X=X.copy(),
        U=None,
        Y=None,
        dXdt=dXdt.copy(),
        deriv_method="return",
    )

    assert data.n_u == 0
    assert data.n_y == 0
    assert data.u is None
    assert data.y is None
    data.convert_to_time_step_format()
    assert data.U is None
    assert data.Y is None


def test_reduce_samples_every_nth() -> None:
    data = make_data()

    data.reduce_samples(n=2.0)

    assert data.n_t == 2
    np.testing.assert_allclose(data.time.t, np.array([0.0, 2.0]))
    np.testing.assert_allclose(data.X, make_arrays()[0][:, ::2, :])
    np.testing.assert_allclose(data.U, make_arrays()[1][:, ::2, :])
    np.testing.assert_allclose(data.Y, make_arrays()[2][:, ::2, :])
    np.testing.assert_allclose(data.dXdt, make_arrays()[3][:, ::2, :])


def test_reduce_samples_supports_optional_arrays_and_rejects_unknown_mode() -> None:
    X, _, _, dXdt = make_arrays()
    data = Data(time=make_time(), X=X.copy(), U=None, Y=None, dXdt=dXdt.copy())

    data.reduce_samples(n=2)

    assert data.U is None
    assert data.Y is None
    np.testing.assert_allclose(data.X, X[:, ::2, :])

    with pytest.raises(ValueError, match="Unknown mode"):
        data.reduce_samples(mode="missing")
