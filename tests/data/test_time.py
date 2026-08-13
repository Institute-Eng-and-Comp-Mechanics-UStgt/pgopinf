from __future__ import annotations

import numpy as np
import pytest

from pgopinf.data.time.time import Time


def test_time_stores_grid_and_spacing() -> None:
    time = Time([0.0, 0.25, 0.5, 0.75])

    np.testing.assert_allclose(time.t, np.array([0.0, 0.25, 0.5, 0.75]))
    assert time.dt == pytest.approx(0.25)
    assert time.n_t == 4


@pytest.mark.parametrize(
    ("t", "message"),
    [
        (np.array([[0.0, 1.0], [2.0, 3.0]]), "one-dimensional"),
        (np.array([0.0]), "at least two"),
        (np.array([-1.0, 0.0, 1.0]), "nonnegative"),
        (np.array([0.0, 1.0, 1.0]), "strictly increasing"),
        (np.array([0.0, 1.0, 3.0]), "equally spaced"),
    ],
)
def test_time_rejects_invalid_grids(t, message) -> None:
    with pytest.raises(ValueError, match=message):
        Time(t)


def test_from_start_end_dt_includes_exact_endpoint() -> None:
    time = Time.from_start_end_dt(start=0.0, end=1.0, dt=0.25)

    np.testing.assert_allclose(time.t, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert time.dt == pytest.approx(0.25)


def test_from_start_end_dt_drops_overshooting_endpoint() -> None:
    time = Time.from_start_end_dt(start=0.0, end=1.0, dt=0.3)

    np.testing.assert_allclose(time.t, np.array([0.0, 0.3, 0.6, 0.9]))
    assert time.dt == pytest.approx(0.3)


def test_from_start_end_dt_rejects_grid_with_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least two"):
        Time.from_start_end_dt(start=0.0, end=0.1, dt=1.0)


def test_from_start_end_timesteps_uses_linspace() -> None:
    time = Time.from_start_end_timesteps(start=1.0, end=2.0, timesteps=5)

    np.testing.assert_allclose(time.t, np.linspace(1.0, 2.0, 5))
    assert time.dt == pytest.approx(0.25)
