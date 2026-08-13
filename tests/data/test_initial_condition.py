from __future__ import annotations

import numpy as np
import pytest

from pgopinf.data.initial_condition.initial_condition import (
    InitialCondition,
)


def test_initial_condition_accepts_two_dimensional_array_like_input() -> None:
    ic = InitialCondition([[1.0, 2.0], [3.0, 4.0]])

    np.testing.assert_allclose(ic.x0, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_initial_condition_rejects_non_matrix_input() -> None:
    with pytest.raises(ValueError, match="x0 must have shape"):
        InitialCondition(np.array([1.0, 2.0]))


def test_repeat_n_sim_times_repeats_vector_as_simulation_columns() -> None:
    ic = InitialCondition.repeat_n_sim_times(np.array([1.0, 2.0, 3.0]), n_sim=4)

    np.testing.assert_allclose(
        ic.x0,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
            ]
        ),
    )


def test_repeat_n_sim_times_repeats_column_initial_condition() -> None:
    ic = InitialCondition.repeat_n_sim_times(np.array([[1.0], [2.0]]), n_sim=3)

    np.testing.assert_allclose(ic.x0, np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))


def test_create_random_initial_conditions_is_seeded_and_scaled() -> None:
    ic_a = InitialCondition.create_random_initial_conditions(
        n=2,
        n_sim=3,
        seed=123,
        scaling=0.25,
    )
    ic_b = InitialCondition.create_random_initial_conditions(
        n=2,
        n_sim=3,
        seed=123,
        scaling=0.25,
    )

    assert ic_a.x0.shape == (2, 3)
    assert np.all((0.0 <= ic_a.x0) & (ic_a.x0 <= 0.25))
    np.testing.assert_allclose(ic_a.x0, ic_b.x0)


def test_reduce_projects_initial_conditions_and_returns_new_instance() -> None:
    ic = InitialCondition(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    projector = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])

    reduced = ic.reduce(projector=projector)

    assert isinstance(reduced, InitialCondition)
    assert reduced is not ic
    np.testing.assert_allclose(reduced.x0, projector @ ic.x0)
