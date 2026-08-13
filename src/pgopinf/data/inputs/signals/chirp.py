from __future__ import annotations
from collections.abc import Callable
import numpy as np


def chirp(
    *,
    amp: float,
    f0: float,
    f1: float,
    T: float,
    tspan: tuple[float, float] | None = None,
) -> Callable[[float | np.ndarray], np.ndarray]:
    """Create a scalar linear chirp input signal.

    Parameters
    ----------
    amp : float
        Signal amplitude.
    f0 : float
        Initial frequency in hertz.
    f1 : float
        Final frequency in hertz.
    T : float
        Chirp duration used in the frequency sweep.
    tspan : tuple of float, optional
        Active interval. Values outside the interval are set to zero.

    Returns
    -------
    callable
        Function ``u(t)`` returning an array with shape ``(1, n_t)``.
    """
    def u(t):
        """Evaluate the chirp signal.

        Parameters
        ----------
        t : float or ndarray
            Evaluation time or times.

        Returns
        -------
        ndarray, shape (1, n_t)
            Signal values.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        # use tspan[0] as time offset to start the chirp at the right frequency
        t_offset = tspan[0] if tspan is not None else 0.0
        t_for_freq = t - t_offset
        # A*sin(2π*((f1-f0)/(2T)*t^2 + f0*t))
        sig = amp * np.sin(
            2 * np.pi * (((f1 - f0) / (2 * T)) * t_for_freq**2 + f0 * t_for_freq)
        )
        # set sig to zero if not in tspan
        if tspan is not None:
            # evaluate for each time point whether it is in the tspan
            mask = (t >= tspan[0]) & (t <= tspan[1])
            sig = np.where(mask, sig, 0.0)
        return sig[None, :]

    return u


if __name__ == "__main__":
    # test chirp signal
    f0 = 1e-2
    f1 = 1
    T = 8.4
    t = np.linspace(0, T, 1000)

    import matplotlib.pyplot as plt

    u1 = chirp(amp=1.0, f0=f0, f1=f1, T=T / 2, tspan=(0, T / 2))
    u2 = chirp(amp=1.0, f0=f1, f1=f0, T=T / 2, tspan=(T / 2, T))
    plt.plot(t, u1(t).squeeze(), label="u1")
    plt.plot(t, u2(t).squeeze(), label="u2")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Chirp Signal")
    plt.legend()
    plt.show()
    plt.savefig("chirp_signal.png", dpi=300)
