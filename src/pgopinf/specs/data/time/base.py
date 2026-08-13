from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from pgopinf.data.time.time import Time


@dataclass(frozen=True)
class TimeSpec:
    """Specification for an equally spaced time grid."""

    t0: float = 0.0
    t_end: float = 10.0
    n_steps: int = 100

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimeSpec":
        """Deserialize a time-grid specification."""
        # Accept dt in configs for convenience, but normalize to n_steps
        t0 = float(d.get("t0", 0.0))
        t_end = float(d.get("t_end", 10.0))

        if "n_steps" in d and d["n_steps"] is not None:
            n_steps = int(d["n_steps"])
            return cls(t0=t0, t_end=t_end, n_steps=n_steps)

        if "dt" in d and d["dt"] is not None:
            dt = float(d["dt"])
            n_steps = cls.dt_to_n_steps(
                dt, t0=t0, t_end=t_end
            )  # validate dt before creating instance

            return cls(t0=t0, t_end=t_end, n_steps=n_steps)

        # fall back to default n_steps if neither is given
        return cls(t0=t0, t_end=t_end, n_steps=int(d.get("n_steps", 100)))

    @classmethod
    def from_start_end_timesteps(cls, t0: float, t_end: float, dt: float) -> "TimeSpec":
        """Create a time spec from an interval and time step."""
        n_steps = cls.dt_to_n_steps(dt, t0=t0, t_end=t_end)
        return cls(t0=t0, t_end=t_end, n_steps=n_steps)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the time-grid specification."""
        # Always serialize canonical form
        return {"t0": self.t0, "t_end": self.t_end, "n_steps": self.n_steps}

    def build(self) -> "Time":
        """Build the runtime time-grid object."""
        # Always use deterministic timesteps
        return Time.from_start_end_timesteps(self.t0, self.t_end, self.n_steps)

    def dt_effective(self) -> float:
        """Return the effective time step."""
        return (self.t_end - self.t0) / self.n_steps

    def t_eval(self):
        """Return the evaluation time grid as an array."""
        import numpy as np

        return np.linspace(self.t0, self.t_end, self.n_steps + 1)

    @staticmethod
    def dt_to_n_steps(dt: float, t0: float, t_end: float) -> int:
        """Convert a time step to a number of steps for an interval."""
        span = t_end - t0
        n_steps = int(round(span / dt))
        if n_steps <= 0:
            raise ValueError("dt too large for interval")
        # consistency check:
        dt_eff = span / n_steps
        if abs(dt_eff - dt) > 1e-4 * max(1.0, abs(dt), abs(dt_eff)):
            raise ValueError(
                f"dt={dt} not consistent with t0/t_end (dt_eff={dt_eff}). Difference: {abs(dt_eff - dt)}"
            )
        # add initial conditon
        return n_steps + 1
