from __future__ import annotations

import pytest

from pgopinf.specs.presets.run_recipes import (
    RUN_RECIPES,
    RunConfig,
    build_experiment_spec,
    build_study_spec,
    default_r_sweep,
    experiment_preset_name,
    study_preset_kwargs,
)


def test_experiment_preset_name_uses_poro_exp3_for_large_ph_study() -> None:
    config = RunConfig(
        experiment_name="poro",
        mode="study",
        study_name="petrov_galerkin_and_ph",
    )

    assert experiment_preset_name(config) == "poro_exp3"
    assert build_experiment_spec(config).data.train.time.n_steps == 500000


def test_default_study_kwargs_for_msd_mimo_50mass_include_output_r() -> None:
    config = RunConfig(
        experiment_name="msd_mimo_50mass",
        mode="study",
        study_name="g_pg_hamcvx_ph",
        n_red=4,
    )

    assert study_preset_kwargs(config) == {
        "r_sweep": (2, 28, 54, 80),
        "title_prefix": "msd_mimo_50mass",
        "output_r": 20,
    }


@pytest.mark.parametrize(
    ("experiment_name", "study_name", "expected"),
    [
        ("msd_mimo_20mass", "petrov_vs_galerkin", (2, 10)),
        ("msd_siso_50mass", "petrov_vs_galerkin", (4, 12, 20, 28, 36, 44, 52, 60)),
        ("msd_mimo_50mass", "petrov_vs_galerkin_fine_r", tuple(range(2, 81, 2))),
        ("poro", "petrov_vs_galerkin_fine_r", tuple(range(2, 201, 6))),
        ("guitar", "petrov_vs_galerkin", (12, 20, 40, 80)),
    ],
)
def test_default_r_sweeps(
    experiment_name: str,
    study_name: str,
    expected: tuple[int, ...],
) -> None:
    config = RunConfig(
        experiment_name=experiment_name,
        mode="study",
        study_name=study_name,
    )

    assert default_r_sweep(config) == expected


def test_explicit_r_sweep_overrides_default() -> None:
    config = RunConfig(
        experiment_name="cd_player",
        mode="study",
        study_name="petrov_vs_galerkin",
        r_sweep=(3, 7),
    )

    assert build_study_spec(config).axes[0].values == (3, 7)


def test_study_name_is_required_for_study_specs() -> None:
    config = RunConfig(experiment_name="guitar", mode="study")

    with pytest.raises(ValueError, match="study_name is required"):
        build_study_spec(config)


def test_recipe_names_are_unique() -> None:
    names = [recipe.name for recipe in RUN_RECIPES]

    assert len(names) == len(set(names))
