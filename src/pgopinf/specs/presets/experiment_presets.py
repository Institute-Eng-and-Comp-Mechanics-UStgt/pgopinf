from __future__ import annotations

from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.specs.presets.experiments.msd_mimo_50mass import (
    msd_mimo_50mass,
)
from pgopinf.specs.presets.experiments.msd_siso_50mass import (
    msd_siso_50mass,
)
from pgopinf.specs.presets.experiments.cd_player import cd_player
from pgopinf.specs.presets.experiments.earth_atmosphere import (
    earth_atmosphere,
)
from pgopinf.specs.presets.experiments.poro import poro


def experiment_preset(name: str, **kwargs) -> ExperimentSpec:
    """Return a named experiment preset."""
    name = name.lower()

    if name == "msd_siso_50mass":
        return msd_siso_50mass()

    if name == "msd_mimo_50mass":
        return msd_mimo_50mass(**kwargs)

    if name == "cd_player":
        return cd_player()

    if name == "earth_atmosphere":
        return earth_atmosphere()

    if name == "poro":
        return poro()

    if name == "poro_exp3":
        exp_spec = poro()
        exp_spec = exp_spec.with_overrides(
            {
                "data.train.time.n_steps": 500000,
                "data.train.reduce_sample_n": None,
                "reduction.full_basis.full_matrices": False,
            }
        )
        return exp_spec

    else:
        raise ValueError(f"Unknown ExperimentSpec preset: {name!r}")


if __name__ == "__main__":
    name = "poro_exp3"
    exp_spec = experiment_preset(name)
    print(exp_spec)
