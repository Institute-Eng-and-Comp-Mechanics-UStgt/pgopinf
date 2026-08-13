# yourpkg/inputs/presets.py
from __future__ import annotations
from typing import Callable
import numpy as np

from .signals.elementary import sawtooth_wave, damped_sin, damped_cos
from .signals.mimo import stack_channels
from .signals.multisine import multisine
from .signals.prbs import prbs_time_signal
from .signals.chirp import chirp


def build_preset(
    name: str, *, time, rng, options: dict | None = None
) -> list[Callable]:
    """
    Returns list[u_i(t)] with length = n_sim.
    `time` is your runtime Time (has t and dt).
    `rng` is np.random.Generator for reproducible multi-trajectory presets.
    """
    options = options or {}
    t = time.t

    if name == "sawtooth":
        freq = float(options.get("freq_hz", 0.5))
        amp = float(options.get("amp", 1.0))
        return [sawtooth_wave(freq_hz=freq, amp=amp, n_u=1)]

    if name == "mimo_sawtooth_plus_minus":
        u1 = sawtooth_wave(
            freq_hz=float(options.get("freq_hz", 0.5)),
            amp=float(options.get("amp", 1.0)),
            n_u=1,
        )
        u2 = sawtooth_wave(
            freq_hz=float(options.get("freq_hz", 0.5)),
            amp=-float(options.get("amp", 1.0)),
            n_u=1,
        )
        return [stack_channels([u1, u2])]

    if name == "mimo_sin_cos":
        amp = float(options.get("amp", 1.0))
        delta = float(options.get("delta", 0.5))
        f = float(options.get("freq_hz", 0.5))
        u_sin = damped_sin(delta=delta, amp=amp, freq_hz=f, chirp=True, n_u=1)
        u_cos = damped_cos(delta=delta, amp=amp, freq_hz=f, chirp=True, n_u=1)
        return [stack_channels([u_sin, u_cos])]

    if name == "prbs_siso":
        order = int(options.get("order", 4))
        amp = float(options.get("amp", 1.0))
        length = len(t)  # consistent with time grid
        return [
            prbs_time_signal(dt=float(time.dt), order=order, length=length, amp=amp)
        ]

    if name == "multisine_siso":
        return [
            multisine(
                f_low=float(options["f_low"]),
                f_high=float(options["f_high"]),
                n_freqs=int(options["n_freqs"]),
                amplitudes=options.get("amplitudes", None),
                overall_scale=float(options.get("overall_scale", 1.0)),
                random_phase=bool(options.get("random_phase", True)),
                phase_seed=int(options.get("phase_seed", 1)),
            )
        ]

    if name == "chirp_siso":
        return [
            chirp(
                amp=float(options.get("amp", 1.0)),
                f0=float(options.get("f0", 1e-2)),
                f1=float(options.get("f1", 1e1)),
                T=float(options.get("T", time.t[-1])),
            )
        ]

    if name == "multisine_mimo_mult_traj":
        # Example: multiple trajectories with different phase seeds
        n_sim = int(options.get("n_sim", 5))
        phase_seeds = rng.integers(0, 1_000_000, size=n_sim)
        u_list = []
        for ps in phase_seeds:
            u1 = multisine(
                f_low=0.1, f_high=1e3, n_freqs=20, random_phase=True, phase_seed=int(ps)
            )
            u2 = multisine(
                f_low=0.1,
                f_high=1e3,
                n_freqs=20,
                random_phase=True,
                phase_seed=int(ps) + 1,
            )
            u_list.append(stack_channels([u1, u2]))
        return u_list

    if name == "mimo_msd_chirp":
        amp = 4
        # start and end frequencies in Hz
        f0_1 = 1
        f1_1 = 1000
        f_0_2 = 0.1
        f1_2 = 10
        T = 10
        u1 = chirp(amp=amp, f0=f0_1, f1=f1_1, T=T)
        u2 = chirp(amp=amp, f0=f_0_2, f1=f1_2, T=T)
        return [stack_channels([u1, u2])]

    raise ValueError(f"Unknown input preset {name!r}")
