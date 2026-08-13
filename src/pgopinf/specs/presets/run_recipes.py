"""Preset run recipes for experiments and studies."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from pgopinf.specs.experiment import ExperimentSpec
    from pgopinf.specs.study import StudySpec

RunMode = Literal["single", "study"]

EXPERIMENT_PRESETS = (
    "msd_siso_50mass",
    "msd_mimo_50mass",
    "cd_player",
    "earth_atmosphere",
    "poro",
)

STUDY_PRESETS = (
    "opinf_theory_over_r",
    "petrov_vs_galerkin",
    "petrov_vs_galerkin_fine_r",
    "petrov_galerkin_and_ph",
    "g_pg_hamcvx_ph",
)


@dataclass(frozen=True)
class RunConfig:
    """Configuration for a preset experiment or study run.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment preset.
    experiment_preset_kwargs : dict, optional
        Additional keyword arguments forwarded to the experiment preset.
    mode : {"single", "study"}, optional
        Whether to run a single experiment or a study.
    study_name : str, optional
        Name of the study preset. Required when ``mode`` is ``"study"``.
    force_recompute : bool, optional
        If ``True``, recompute results even when cached artifacts exist.
    results_root : pathlib.Path, optional
        Root directory where run artifacts are stored.
    n_red : int, optional
        Number of reduction orders used when constructing default sweeps.
    r_sweep : tuple of int, optional
        Explicit reduction-order sweep. If omitted, a preset default is used.
    output_r : int, optional
        Reduction order highlighted in reporting outputs. If omitted, a preset
        default is used when available.
    """

    experiment_name: str
    experiment_preset_kwargs: dict = field(default_factory=dict)
    mode: RunMode = "study"
    study_name: str | None = None
    force_recompute: bool = False
    results_root: Path = Path("results")
    n_red: int = 14
    r_sweep: tuple[int, ...] | None = None
    output_r: int | None = None


@dataclass(frozen=True)
class RunRecipe:
    """Named recipe for reproducing a configured run.

    Parameters
    ----------
    name : str
        Unique recipe name.
    config : RunConfig
        Run configuration associated with the recipe.
    description : str
        Human-readable description of the recipe.
    """

    name: str
    config: RunConfig
    description: str


RUN_RECIPES: tuple[RunRecipe, ...] = (
    # Experiment 1
    RunRecipe(
        name="msd_mimo_50mass_opinf_theory",
        config=RunConfig(
            experiment_name="msd_mimo_50mass",
            study_name="opinf_theory_over_r",
        ),
        description="50-mass MIMO system, Operator Inference theory over reduction order.",
    ),
    RunRecipe(
        name="earth_atmosphere_opinf_theory",
        config=RunConfig(
            experiment_name="earth_atmosphere",
            study_name="opinf_theory_over_r",
        ),
        description="Earth atmosphere system, Operator Inference theory over reduction order.",
    ),
    RunRecipe(
        name="cd_player_opinf_theory",
        config=RunConfig(
            experiment_name="cd_player",
            study_name="opinf_theory_over_r",
        ),
        description="CD player system, Operator Inference theory over reduction order.",
    ),
    RunRecipe(
        name="poro_opinf_theory",
        config=RunConfig(
            experiment_name="poro",
            study_name="opinf_theory_over_r",
        ),
        description="Porelasticity system, Operator Inference theory over reduction order.",
    ),
    # Experiment 2
    RunRecipe(
        name="msd_mimo_50mass_petrov_vs_galerkin",
        config=RunConfig(
            experiment_name="msd_mimo_50mass",
            study_name="petrov_vs_galerkin_fine_r",
        ),
        description="50-mass MIMO system, Petrov-Galerkin vs Galerkin comparison.",
    ),
    # Long version r= 2,8,...200
    RunRecipe(
        name="poro_petrov_vs_galerkin_long",
        config=RunConfig(
            experiment_name="poro",
            study_name="petrov_vs_galerkin_fine_r",
        ),
        description="Porelasticity system, Petrov-Galerkin vs Galerkin comparison.",
    ),
    # Short version r= 2,8,...80
    RunRecipe(
        name="poro_petrov_vs_galerkin",
        config=RunConfig(
            experiment_name="poro",
            study_name="petrov_vs_galerkin",
        ),
        description="Porelasticity system, Petrov-Galerkin vs Galerkin comparison.",
    ),
    # Experiment 3
    RunRecipe(
        name="msd_mimo_50mass_g_pg_hamcvx_ph",
        config=RunConfig(
            experiment_name="msd_mimo_50mass",
            study_name="g_pg_hamcvx_ph",
            experiment_preset_kwargs={
                "reduce_sample_n_train": 3,
            },
        ),
        description="50-mass MIMO system, Galerkin, Petrov-Galerkin, Hamiltonian-cvx, and convex PH methods comparison.",
    ),
)


def get_recipe(name: str) -> RunConfig:
    """Return a run configuration by recipe name.

    Parameters
    ----------
    name : str
        Name of the recipe in :data:`RUN_RECIPES`.

    Returns
    -------
    RunConfig
        Configuration associated with ``name``.

    Raises
    ------
    ValueError
        If no recipe with the given name exists.
    """
    recipe_by_name = {recipe.name: recipe for recipe in RUN_RECIPES}
    try:
        return recipe_by_name[name].config
    except KeyError as exc:
        raise ValueError(f"Unknown run recipe: {name!r}") from exc


def with_runtime_options(
    config: RunConfig,
    *,
    force_recompute: bool | None = None,
    results_root: Path | None = None,
    n_red: int | None = None,
    r_sweep: tuple[int, ...] | None = None,
    output_r: int | None = None,
) -> RunConfig:
    """Return a copy of a run configuration with runtime options updated.

    Parameters
    ----------
    config : RunConfig
        Base configuration.
    force_recompute : bool, optional
        Replacement value for ``config.force_recompute``.
    results_root : pathlib.Path, optional
        Replacement value for ``config.results_root``.
    n_red : int, optional
        Replacement value for ``config.n_red``.
    r_sweep : tuple of int, optional
        Replacement value for ``config.r_sweep``.
    output_r : int, optional
        Replacement value for ``config.output_r``.

    Returns
    -------
    RunConfig
        Updated immutable configuration.
    """
    updates = {}
    if force_recompute is not None:
        updates["force_recompute"] = force_recompute
    if results_root is not None:
        updates["results_root"] = results_root
    if n_red is not None:
        updates["n_red"] = n_red
    if r_sweep is not None:
        updates["r_sweep"] = r_sweep
    if output_r is not None:
        updates["output_r"] = output_r
    return replace(config, **updates)


def build_experiment_spec(config: RunConfig) -> "ExperimentSpec":
    """Build an experiment specification from a run configuration.

    Parameters
    ----------
    config : RunConfig
        Run configuration containing the experiment preset name and keyword
        arguments.

    Returns
    -------
    ExperimentSpec
        Instantiated experiment specification.
    """
    from pgopinf.specs.experiment import ExperimentSpec

    return ExperimentSpec.preset(
        name=experiment_preset_name(config), **config.experiment_preset_kwargs
    )


def build_study_spec(config: RunConfig) -> "StudySpec":
    """Build a study specification from a run configuration.

    Parameters
    ----------
    config : RunConfig
        Run configuration containing the study preset and sweep options.

    Returns
    -------
    StudySpec
        Instantiated study specification.

    Raises
    ------
    ValueError
        If ``config.study_name`` is missing.
    """
    from pgopinf.specs.study import StudySpec

    if config.study_name is None:
        raise ValueError("RunConfig.study_name is required for study mode")

    return StudySpec.preset(
        name=config.study_name,
        **study_preset_kwargs(config),
    )


def experiment_preset_name(config: RunConfig) -> str:
    """Return the experiment preset name for a run configuration.

    Parameters
    ----------
    config : RunConfig
        Run configuration.

    Returns
    -------
    str
        Experiment preset name, including compatibility mappings for special
        study cases.
    """
    if (
        config.experiment_name == "poro"
        and config.study_name == "petrov_galerkin_and_ph"
    ):
        return "poro_exp3"

    return config.experiment_name


def study_preset_kwargs(config: RunConfig) -> dict:
    """Return keyword arguments for the configured study preset.

    Parameters
    ----------
    config : RunConfig
        Run configuration containing study defaults and overrides.

    Returns
    -------
    dict
        Keyword arguments passed to :meth:`StudySpec.preset`.

    Raises
    ------
    ValueError
        If ``config.study_name`` is missing.
    """
    if config.study_name is None:
        raise ValueError("RunConfig.study_name is required for study mode")

    kwargs = {
        "r_sweep": (
            config.r_sweep if config.r_sweep is not None else default_r_sweep(config)
        ),
        "title_prefix": config.experiment_name,
    }
    output_r = (
        config.output_r if config.output_r is not None else default_output_r(config)
    )
    if output_r is not None:
        kwargs["output_r"] = output_r
    return kwargs


def default_output_r(config: RunConfig) -> int | None:
    """Return the default reporting reduction order for a configuration.

    Parameters
    ----------
    config : RunConfig
        Run configuration.

    Returns
    -------
    int or None
        Default output reduction order, or ``None`` when no default is defined.
    """
    if config.experiment_name == "msd_mimo_50mass":
        return 20
    return None


def default_r_sweep(config: RunConfig) -> tuple[int, ...]:
    """Return the default reduction-order sweep for a configuration.

    Parameters
    ----------
    config : RunConfig
        Run configuration containing the experiment and study names.

    Returns
    -------
    tuple of int
        Default reduction orders for the configured experiment and study.

    Raises
    ------
    ValueError
        If no default sweep is defined for the experiment name.
    """
    experiment_name = config.experiment_name
    study_name = config.study_name

    if experiment_name == "msd_mimo_20mass":
        return (2, 10)

    if experiment_name == "msd_siso_50mass":
        return tuple(range(4, 60 + 1, 8))

    if experiment_name == "msd_mimo_50mass":
        if study_name == "petrov_vs_galerkin_fine_r":
            return tuple(range(2, 80 + 1, 2))
        return _linspace_int_tuple(2, 80, config.n_red)

    if experiment_name in {"cd_player", "earth_atmosphere"}:
        return _linspace_int_tuple(2, 80, config.n_red)

    if experiment_name == "poro":
        if study_name == "petrov_vs_galerkin_fine_r":
            return tuple(range(2, 200 + 1, 6))
        return _linspace_int_tuple(2, 80, config.n_red)

    if experiment_name == "guitar":
        return (12, 20, 40, 80)

    raise ValueError(f"Unknown experiment name for study defaults: {experiment_name!r}")


def _linspace_int_tuple(start: int, stop: int, num: int) -> tuple[int, ...]:
    return tuple(int(value) for value in np.linspace(start, stop, num, dtype=int))
