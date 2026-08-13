from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from pgopinf.data.inputs.inputs import Input
from pgopinf.data.inputs.presets import build_preset

# composable primitives (Option 1)
from pgopinf.data.inputs.signals.elementary import (
    constant as gen_constant,
    sine as gen_sine,
    cosine as gen_cosine,
    sawtooth_wave as gen_sawtooth,
    damped_sin as gen_damped_sin,
    damped_cos as gen_damped_cos,
)
from pgopinf.data.inputs.signals.prbs import (
    prbs_time_signal as gen_prbs,
)
from pgopinf.data.inputs.signals.multisine import (
    multisine as gen_multisine,
)
from pgopinf.data.inputs.signals.chirp import chirp as gen_chirp
from pgopinf.data.inputs.signals.mimo import stack_channels


class InputSpec(Protocol):
    """Protocol for input-signal specifications."""

    kind: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the input specification."""
        ...

    def build(self, *, time, seed: int, n_sim: int) -> Input:
        """Build runtime input trajectories."""
        ...


# ============================================================
# A) Preset spec
# ============================================================


@dataclass(frozen=True)
class PresetInputSpec:
    """Input specification selecting a named input preset."""

    kind: Literal["preset"] = "preset"
    name: str = "sawtooth"
    options: dict[str, Any] | None = None
    # optionally force number of sims if preset returns 1 signal
    n_sim: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the preset input specification."""
        return {
            "kind": self.kind,
            "name": self.name,
            "options": self.options or {},
            "n_sim": self.n_sim,
        }

    def build(self, *, time, seed: int, n_sim: int) -> Input:
        """Build runtime inputs from the named preset."""
        rng = np.random.default_rng(seed)
        u_list = build_preset(self.name, time=time, rng=rng, options=self.options or {})
        if self.n_sim is not None and len(u_list) == 1:
            u_list = u_list * int(self.n_sim)
        elif len(u_list) == 1 and n_sim != 1:
            # If caller requested many sims but preset returns a single callable,
            # repeat it (common behavior); you can change to "raise" if you prefer.
            u_list = u_list * int(n_sim)
        return Input(u_list=u_list)


# ============================================================
# B) Composable expression spec
# ============================================================
@dataclass(frozen=True)
class ExprInputSpec:
    """
    Expression tree stored as JSON-serializable dict(s).
    Example:
      {"kind":"stack","channels":[{"kind":"sin",...},{"kind":"sawtooth",...}]}
    """

    kind: Literal["expr"] = "expr"
    expr: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the expression input specification."""
        return {
            "kind": self.kind,
            "expr": self.expr or {"kind": "constant", "value": 0.0},
        }

    def build(self, *, time, seed: int, n_sim: int) -> Input:
        """Build runtime inputs from the expression tree."""
        if self.expr is None:
            expr = {"kind": "constant", "value": 0.0}
        else:
            expr = self.expr
        return _build_expr(expr, time=time, seed=seed, n_sim=n_sim)


# ============================================================
# Parsing
# ============================================================


def input_from_dict(d: dict[str, Any]) -> InputSpec:
    """Deserialize an input specification from a dictionary."""
    kind = d.get("kind", "preset")
    if kind == "preset":
        options = d.get("options", None)
        return PresetInputSpec(
            name=str(d.get("name", "sawtooth")),
            options=dict(options) if options else None,
            n_sim=d.get("n_sim", None),
        )
    if kind == "expr":
        return ExprInputSpec(
            expr=dict(d.get("expr", {"kind": "constant", "value": 0.0}))
        )
    raise ValueError(f"Unknown input.kind: {kind!r}")


# ============================================================
# Expression builder (primitives + composition)
# ============================================================


def _build_expr(expr: dict[str, Any], *, time, seed: int, n_sim: int) -> Input:
    """
    Builds an Input from an expression dict. Supports primitives + stack/repeat.
    """
    kind = expr.get("kind", "constant")

    # ---- primitives (SISO) ----
    if kind == "constant":
        u = gen_constant(value=float(expr.get("value", 0.0)), n_u=1)
        return Input(u_list=[u] * n_sim)

    if kind == "sin":
        u = gen_sine(
            freq_hz=float(expr.get("freq_hz", 1.0)),
            amp=float(expr.get("amp", 1.0)),
            phase=float(expr.get("phase", 0.0)),
            n_u=1,
        )
        return Input(u_list=[u] * n_sim)

    if kind == "cos":
        u = gen_cosine(
            freq_hz=float(expr.get("freq_hz", 1.0)),
            amp=float(expr.get("amp", 1.0)),
            phase=float(expr.get("phase", 0.0)),
            n_u=1,
        )
        return Input(u_list=[u] * n_sim)

    if kind == "sawtooth":
        u = gen_sawtooth(
            freq_hz=float(expr.get("freq_hz", 0.5)),
            amp=float(expr.get("amp", 1.0)),
            n_u=1,
        )
        return Input(u_list=[u] * n_sim)

    if kind == "damped_sin":
        u = gen_damped_sin(
            delta=float(expr.get("delta", 0.5)),
            amp=float(expr.get("amp", 1.0)),
            freq_hz=float(expr.get("freq_hz", 1.0)),
            chirp=bool(expr.get("chirp", True)),
            n_u=1,
        )
        return Input(u_list=[u] * n_sim)

    if kind == "chirp":
        T = float(time.t[-1]) if expr.get("T", None) is None else float(expr["T"])
        u = gen_chirp(
            amp=float(expr.get("amp", 1.0)),
            f0=float(expr.get("f0", 1e-2)),
            f1=float(expr.get("f1", 1e1)),
            T=T,
            tspan=expr.get("tspan", None),
        )
        return Input(u_list=[u] * n_sim)

    if kind == "prbs":
        rng = np.random.default_rng(seed)
        dt_hold = expr.get("dt_hold", None)
        dt = float(time.dt) if dt_hold is None else float(dt_hold)
        length = len(time.t)
        order = int(expr.get("order", 4))
        amp = float(expr.get("amp", 1.0))
        base = gen_prbs(dt=dt, order=order, length=length, amp=amp)

        # different sim variants via reproducible roll
        u_list = []
        for _ in range(n_sim):
            shift = int(rng.integers(0, length))

            def u(t, base=base, shift=shift):
                """Evaluate a shifted PRBS trajectory."""
                y = base(t)
                return np.roll(y, shift=shift, axis=1)

            u_list.append(u)
        return Input(u_list=u_list)

    if kind == "multisine":
        rng = np.random.default_rng(seed)
        u_list = []
        for _ in range(n_sim):
            ps = int(rng.integers(0, 1_000_000))
            u_list.append(
                gen_multisine(
                    f_low=float(expr.get("f_low", 0.1)),
                    f_high=float(expr.get("f_high", 10.0)),
                    n_freqs=int(expr.get("n_freqs", 10)),
                    amplitudes=expr.get("amplitudes", None),
                    overall_scale=float(expr.get("overall_scale", 1.0)),
                    random_phase=bool(expr.get("random_phase", True)),
                    phase_seed=int(expr.get("phase_seed", ps)),
                )
            )
        return Input(u_list=u_list)

    # ---- composition ----
    if kind == "repeat":
        base_expr = dict(expr.get("base", {"kind": "constant", "value": 0.0}))
        base_inp = _build_expr(base_expr, time=time, seed=seed, n_sim=1)
        return Input(u_list=base_inp.u_list * n_sim)

    if kind == "stack":
        # channels: list of expressions
        channels = expr.get("channels", [])
        channel_inputs = [
            _build_expr(dict(ch), time=time, seed=seed + 1000 * j, n_sim=n_sim)
            for j, ch in enumerate(channels)
        ]
        u_list = []
        for i in range(n_sim):
            chans_i = [inp.u_list[i] for inp in channel_inputs]
            u_list.append(stack_channels(chans_i))
        return Input(u_list=u_list)

    raise ValueError(f"Unknown expr.kind: {kind!r}")
