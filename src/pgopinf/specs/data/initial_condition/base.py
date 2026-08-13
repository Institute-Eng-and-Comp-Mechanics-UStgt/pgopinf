from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Protocol

# runtime import path: adjust to your project
from pgopinf.data.initial_condition.initial_condition import (
    InitialCondition,
)


class ICSpec(Protocol):
    """Protocol for initial-condition specifications."""

    kind: str

    def build(self, *, n: int, n_sim: int, seed: int | None) -> InitialCondition:
        """Build runtime initial conditions."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the initial-condition specification."""
        ...


@dataclass(frozen=True)
class ZerosIC:
    """Initial-condition spec for zero states."""

    kind: Literal["zeros"] = "zeros"
    # optional: allow specifying a non-zero constant vector later
    # value: float = 0.0

    def build(self, *, n: int, n_sim: int, seed: int | None) -> InitialCondition:
        """Build zero initial conditions."""
        # use repeat method with a zero vector
        import numpy as np

        x0 = np.zeros((n,), dtype=float)
        return InitialCondition.repeat_n_sim_times(x0, n_sim)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the zero initial-condition spec."""
        return {"kind": self.kind}


@dataclass(frozen=True)
class RandomIC:
    """Initial-condition spec for random states."""

    kind: Literal["random"] = "random"
    scaling: float = 1.0

    def build(self, *, n: int, n_sim: int, seed: int | None) -> InitialCondition:
        """Build random initial conditions."""
        # delegates randomness + scaling to your runtime method
        return InitialCondition.create_random_initial_conditions(
            n=n, n_sim=n_sim, seed=seed, scaling=self.scaling
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the random initial-condition spec."""
        return {"kind": self.kind, "scaling": self.scaling}


def ic_from_dict(d: dict[str, Any]) -> ICSpec:
    """Deserialize an initial-condition spec from a dictionary."""
    kind = d.get("kind", "zeros")
    if kind == "zeros":
        return ZerosIC()
    if kind == "random":
        return RandomIC(scaling=float(d.get("scaling", 1.0)))
    raise ValueError(f"Unknown ic.kind: {kind!r}")
