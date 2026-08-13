from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from pgopinf.numerics.discretization import integrators
from pgopinf.numerics.discretization.integrators import (
    implicit_midpoint,
    input,
)


@pytest.fixture(autouse=True)
def disable_progress_bar(monkeypatch):
    monkeypatch.setattr(integrators, "tqdm", lambda iterable: iterable)


def test_input_selects_array_column_for_current_time_step() -> None:
    u_mid = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    np.testing.assert_allclose(
        input(u_mid, np.array([0.5, 1.5, 2.5]), 1),
        np.array([2.0, 20.0]),
    )


def test_input_evaluates_callable_at_midpoint() -> None:
    u_mid = lambda t: np.array([t, 2.0 * t])

    np.testing.assert_allclose(
        input(u_mid, np.array([0.5, 1.5]), 0),
        np.array([0.5, 1.0]),
    )


def test_input_rejects_unknown_input_type() -> None:
    with pytest.raises(ValueError, match="Unknown type"):
        input([1.0, 2.0], np.array([0.5]), 0)


def test_implicit_midpoint_keeps_constant_state_for_zero_dynamics() -> None:
    E = np.eye(2)
    A = np.zeros((2, 2))
    t = np.array([0.0, 0.5, 1.0])
    x_init = np.array([1.0, -2.0])

    x, t_out = implicit_midpoint(E, A, t, x_init)

    np.testing.assert_allclose(x, np.array([[1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]]))
    np.testing.assert_allclose(t_out, t[:, np.newaxis])


@pytest.mark.parametrize("decomp_option", ["lu", "linalg_solve"])
def test_implicit_midpoint_matches_scalar_recurrence(decomp_option: str) -> None:
    E = np.array([[1.0]])
    A = np.array([[2.0]])
    t = np.array([0.0, 0.1, 0.2, 0.3])
    x_init = np.array([1.0])
    h = t[1] - t[0]
    recurrence = (1.0 + h) / (1.0 - h)

    x, _ = implicit_midpoint(
        E,
        A,
        t,
        x_init,
        decomp_option=decomp_option,
    )

    np.testing.assert_allclose(
        x,
        np.array([[recurrence**0, recurrence**1, recurrence**2, recurrence**3]]),
    )


def test_implicit_midpoint_uses_array_input_at_midpoints() -> None:
    E = np.array([[1.0]])
    A = np.array([[0.0]])
    B = np.array([[2.0]])
    t = np.array([0.0, 1.0, 2.0])
    x_init = np.array([0.0])
    u_mid = np.array([[1.0, 3.0]])

    x, _ = implicit_midpoint(E, A, t, x_init, B=B, u_mid=u_mid)

    np.testing.assert_allclose(x, np.array([[0.0, 2.0, 8.0]]))


def test_implicit_midpoint_uses_callable_input_at_midpoints() -> None:
    E = np.array([[1.0]])
    A = np.array([[0.0]])
    B = np.array([[1.0]])
    t = np.array([0.0, 1.0, 2.0])
    x_init = np.array([0.0])

    x, _ = implicit_midpoint(E, A, t, x_init, B=B, u_mid=lambda t_mid: np.array([t_mid + 1.0]))

    np.testing.assert_allclose(x, np.array([[0.0, 1.5, 4.0]]))


def test_implicit_midpoint_lu_supports_sparse_left_hand_side() -> None:
    E = scipy.sparse.eye(2, format="csr")
    A = scipy.sparse.csr_matrix((2, 2))
    t = np.array([0.0, 0.5, 1.0])
    x_init = np.array([3.0, -1.0])

    x, _ = implicit_midpoint(E, A, t, x_init, decomp_option="lu")

    np.testing.assert_allclose(x, np.array([[3.0, 3.0, 3.0], [-1.0, -1.0, -1.0]]))


def test_implicit_midpoint_rejects_initial_condition_with_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="Initial condition and number of states"):
        implicit_midpoint(
            np.eye(2),
            np.eye(2),
            np.array([0.0, 1.0]),
            np.array([1.0]),
        )


def test_implicit_midpoint_rejects_unknown_decomposition_option() -> None:
    with pytest.raises(ValueError, match="Decomposition option missing is unknown"):
        implicit_midpoint(
            np.eye(1),
            np.zeros((1, 1)),
            np.array([0.0, 1.0]),
            np.array([1.0]),
            decomp_option="missing",
        )
