from __future__ import annotations

import numpy as np
import pytest

from pgopinf.data.inputs.inputs import Input
from pgopinf.data.inputs.presets import build_preset
from pgopinf.data.inputs.signals.chirp import chirp
from pgopinf.data.inputs.signals.elementary import (
    constant,
    cosine,
    damped_cos,
    damped_sin,
    sawtooth_wave,
    sine,
)
from pgopinf.data.inputs.signals.mimo import stack_channels
from pgopinf.data.inputs.signals.multisine import multisine
from pgopinf.data.inputs.signals.prbs import (
    prbs_sequence,
    prbs_time_signal,
)
from pgopinf.data.time.time import Time


def test_input_evaluate_stacks_simulations_in_time_step_format() -> None:
    inp = Input(
        u_list=[
            lambda t: np.vstack([np.asarray(t), 2.0 * np.asarray(t)]),
            lambda t: np.vstack([10.0 + np.asarray(t), 20.0 + np.asarray(t)]),
        ]
    )
    t = np.array([0.0, 1.0, 2.0])

    U = inp.evaluate(t)

    assert inp.n_sim() == 2
    assert inp.n_u() == 2
    assert U.shape == (2, 3, 2)
    np.testing.assert_allclose(U[:, :, 0], np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]))
    np.testing.assert_allclose(
        U[:, :, 1],
        np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]),
    )


def test_input_evaluate_broadcasts_vector_return_over_time() -> None:
    inp = Input(u_list=[lambda t: np.array([1.0, -1.0])])

    U = inp.evaluate(np.array([0.0, 1.0, 2.0]))

    np.testing.assert_allclose(
        U,
        np.array([[[1.0], [1.0], [1.0]], [[-1.0], [-1.0], [-1.0]]]),
    )


def test_input_evaluate_rejects_wrong_callable_shape() -> None:
    inp = Input(u_list=[lambda t: np.ones((2, 2))])

    with pytest.raises(ValueError, match="Input callable returned shape"):
        inp.evaluate(np.array([0.0, 1.0, 2.0]))


def test_input_evaluate_midpoints_uses_between_time_grid_values() -> None:
    inp = Input(u_list=[lambda t: np.atleast_1d(np.asarray(t))[None, :]])

    np.testing.assert_allclose(
        inp.evaluate_midpoints(np.array([0.0, 1.0, 3.0])),
        np.array([[[0.5], [2.0]]]),
    )


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (constant(value=2.0, n_u=2), np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])),
        (
            sine(freq_hz=0.25, amp=2.0, phase=0.0, n_u=1),
            2.0 * np.sin(0.5 * np.pi * np.array([0.0, 1.0, 2.0]))[None, :],
        ),
        (
            cosine(freq_hz=0.25, amp=2.0, phase=0.0, n_u=1),
            2.0 * np.cos(0.5 * np.pi * np.array([0.0, 1.0, 2.0]))[None, :],
        ),
    ],
)
def test_elementary_signals_return_expected_values(factory, expected) -> None:
    np.testing.assert_allclose(factory(np.array([0.0, 1.0, 2.0])), expected)


def test_damped_signals_support_linear_and_chirped_phase() -> None:
    t = np.array([0.0, 1.0])

    damped_linear = damped_sin(delta=0.1, amp=2.0, freq_hz=0.25, chirp=False)
    damped_chirp = damped_cos(delta=0.1, amp=2.0, freq_hz=0.25, chirp=True)

    np.testing.assert_allclose(
        damped_linear(t),
        (2.0 * np.exp(-0.1 * t) * np.sin(0.5 * np.pi * t))[None, :],
    )
    np.testing.assert_allclose(
        damped_chirp(t),
        (2.0 * np.exp(-0.1 * t) * np.cos(0.5 * np.pi * t**2))[None, :],
    )


def test_sawtooth_wave_repeats_signal_across_channels() -> None:
    u = sawtooth_wave(freq_hz=0.5, amp=3.0, n_u=2)

    values = u(np.array([0.0, 0.5]))

    assert values.shape == (2, 2)
    np.testing.assert_allclose(values[0], values[1])


def test_stack_channels_combines_siso_callables() -> None:
    stacked = stack_channels([constant(1.0), constant(-2.0)])

    np.testing.assert_allclose(
        stacked(np.array([0.0, 1.0])),
        np.array([[1.0, 1.0], [-2.0, -2.0]]),
    )


def test_stack_channels_rejects_non_siso_channel() -> None:
    stacked = stack_channels([constant(1.0, n_u=2)])

    with pytest.raises(ValueError, match="Each channel must be SISO"):
        stacked(np.array([0.0, 1.0]))


def test_chirp_uses_time_span_and_zeroes_values_outside_it() -> None:
    u = chirp(amp=1.0, f0=0.25, f1=1.0, T=2.0, tspan=(1.0, 3.0))
    t = np.array([0.0, 1.0, 2.0, 4.0])

    values = u(t)

    assert values.shape == (1, 4)
    assert values[0, 0] == 0.0
    assert values[0, 3] == 0.0
    np.testing.assert_allclose(values[0, 1], 0.0)


def test_prbs_sequence_values_and_time_signal_indexing() -> None:
    seq = prbs_sequence(order=3, length=5)
    u = prbs_time_signal(dt=0.5, order=3, length=5, amp=2.0)

    assert seq.shape == (5,)
    assert set(np.unique(seq)) <= {-1, 1}
    np.testing.assert_allclose(
        u(np.array([0.0, 0.49, 0.5, 2.5])),
        2.0 * seq[np.array([0, 0, 1, 4])][None, :],
    )


def test_multisine_is_deterministic_with_phase_seed_and_validates_options() -> None:
    t = np.array([0.0, 0.25, 0.5])
    u_a = multisine(f_low=0.1, f_high=1.0, n_freqs=3, phase_seed=123)
    u_b = multisine(f_low=0.1, f_high=1.0, n_freqs=3, phase_seed=123)

    np.testing.assert_allclose(u_a(t), u_b(t))

    with pytest.raises(ValueError, match="Require 0 < f_low < f_high"):
        multisine(f_low=0.0, f_high=1.0, n_freqs=3)

    with pytest.raises(ValueError, match="amplitudes must have shape"):
        multisine(f_low=0.1, f_high=1.0, n_freqs=3, amplitudes=np.ones(2))


def test_multisine_supports_envelope_amplitudes_and_no_random_phase() -> None:
    t = np.array([0.0, 0.25])
    u = multisine(
        f_low=1.0,
        f_high=4.0,
        n_freqs=2,
        amplitudes="envelope",
        random_phase=False,
        overall_scale=2.0,
    )

    freqs = np.array([1.0, 4.0])
    expected = 2.0 * np.sum(
        (1.0 / np.sqrt(freqs))[:, None]
        * np.sin(2.0 * np.pi * freqs[:, None] * t[None, :]),
        axis=0,
    )
    np.testing.assert_allclose(u(t), expected[None, :])


def test_build_preset_creates_named_inputs_with_expected_shapes() -> None:
    time = Time(np.linspace(0.0, 1.0, 5))
    rng = np.random.default_rng(123)

    saw = build_preset("sawtooth", time=time, rng=rng, options={"amp": 2.0})
    mimo = build_preset("mimo_sawtooth_plus_minus", time=time, rng=rng)
    chirp_input = build_preset("chirp_siso", time=time, rng=rng, options={"amp": 0.5})

    assert len(saw) == 1
    assert saw[0](time.t).shape == (1, 5)
    assert mimo[0](time.t).shape == (2, 5)
    np.testing.assert_allclose(mimo[0](time.t)[0], -mimo[0](time.t)[1])
    assert chirp_input[0](time.t).shape == (1, 5)


def test_build_preset_multitrajectory_is_reproducible_from_rng_seed() -> None:
    time = Time(np.linspace(0.0, 1.0, 5))

    first = build_preset(
        "multisine_mimo_mult_traj",
        time=time,
        rng=np.random.default_rng(123),
        options={"n_sim": 2},
    )
    second = build_preset(
        "multisine_mimo_mult_traj",
        time=time,
        rng=np.random.default_rng(123),
        options={"n_sim": 2},
    )

    assert len(first) == 2
    for first_u, second_u in zip(first, second):
        np.testing.assert_allclose(first_u(time.t), second_u(time.t))
        assert first_u(time.t).shape == (2, 5)


def test_build_preset_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown input preset"):
        build_preset("missing", time=Time(np.linspace(0.0, 1.0, 3)), rng=np.random.default_rng(1))
