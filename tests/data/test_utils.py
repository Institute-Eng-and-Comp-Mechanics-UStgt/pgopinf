from __future__ import annotations

import numpy as np

from pgopinf.data.utils.data_manipulating import (
    get_kron_data,
    get_kron_x_dx_data,
    get_kron_x_x_data,
    get_supplied_power,
)


def test_get_supplied_power_computes_samplewise_inner_products() -> None:
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    u = np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])

    supplied_power = get_supplied_power(y, u)

    np.testing.assert_allclose(
        supplied_power,
        np.array([[14.0], [50.0], [108.0]]),
    )


def test_get_kron_data_returns_full_samplewise_kronecker_features() -> None:
    state1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    state2 = np.array([[5.0, 6.0], [7.0, 8.0]])

    features = get_kron_data(state1, state2, symmetric=False)

    np.testing.assert_allclose(
        features,
        np.array(
            [
                [5.0, 7.0, 15.0, 21.0],
                [12.0, 16.0, 24.0, 32.0],
            ]
        ),
    )


def test_get_kron_data_collapses_duplicate_entries_for_symmetric_features() -> None:
    state1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    state2 = np.array([[5.0, 6.0], [7.0, 8.0]])

    features = get_kron_data(state1, state2, symmetric=True)

    np.testing.assert_allclose(
        features,
        np.array(
            [
                [5.0, 22.0, 21.0],
                [12.0, 40.0, 32.0],
            ]
        ),
    )


def test_get_kron_x_x_data_returns_quadratic_upper_triangular_features() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    features = get_kron_x_x_data(x)

    np.testing.assert_allclose(
        features,
        np.array(
            [
                [1.0, 6.0, 9.0],
                [4.0, 16.0, 16.0],
            ]
        ),
    )


def test_get_kron_x_dx_data_delegates_to_symmetric_cross_features() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    dxdt = np.array([[5.0, 6.0], [7.0, 8.0]])

    np.testing.assert_allclose(
        get_kron_x_dx_data(x, dxdt),
        get_kron_data(x, dxdt, symmetric=True),
    )


def test_get_kron_data_supports_one_dimensional_state() -> None:
    state1 = np.array([[2.0, 3.0]])
    state2 = np.array([[5.0, 7.0]])

    np.testing.assert_allclose(
        get_kron_data(state1, state2, symmetric=False),
        np.array([[10.0], [21.0]]),
    )
    np.testing.assert_allclose(
        get_kron_data(state1, state2, symmetric=True),
        np.array([[10.0], [21.0]]),
    )
