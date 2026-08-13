from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from time import perf_counter

from pgopinf.logging_config import setup_logging
from pgopinf.specs.presets.run_recipes import RUN_RECIPES, get_recipe
from pgopinf.workflows.run import run

# This is the public reproduction entry point for the paper results.
#
# Run all linked paper experiments/studies with:
#
#     python scripts/reproduce_paper_results.py
#
# To reproduce only a subset, edit PAPER_RESULT_RECIPE_NAMES below.

RESULTS_ROOT = Path("results")
FORCE_RECOMPUTE = False
TIMINGS_FILE = RESULTS_ROOT / "reproduce_paper_results_timings.txt"

PAPER_RESULT_RECIPE_NAMES = (
    # Experiment 1
    "msd_mimo_50mass_opinf_theory",
    "earth_atmosphere_opinf_theory",
    "cd_player_opinf_theory",
    "poro_opinf_theory",
    # Experiment 2
    "msd_mimo_50mass_petrov_vs_galerkin",
    "poro_petrov_vs_galerkin",
    # Experiment 3
    "msd_mimo_50mass_g_pg_hamcvx_ph",
)


def main(long: bool = False) -> None:
    _check_mosek_license()
    setup_logging()
    _print_catalog()

    timing_rows: list[tuple[str, float]] = []
    total_start = perf_counter()

    for recipe_name in PAPER_RESULT_RECIPE_NAMES:
        recipe_name = _check_for_long(long, recipe_name)
        recipe_start = perf_counter()
        config = get_recipe(recipe_name)
        config = replace(
            config,
            results_root=RESULTS_ROOT,
            force_recompute=FORCE_RECOMPUTE,
        )
        print(f"\nRunning {recipe_name}")
        run(config)
        timing_rows.append((recipe_name, perf_counter() - recipe_start))

    total_duration = perf_counter() - total_start
    _write_timings(timing_rows, total_duration)

    print("\nFinished.")


def _check_mosek_license() -> None:
    """Fail fast when MOSEK cannot check out a license for Experiment 3."""
    try:
        import mosek

        # Creating an environment alone does not check out a license. Optimizing
        # this trivial task exercises the same license checkout as Experiment 3.
        with mosek.Env() as env:
            with env.Task(0, 0) as task:
                task.appendvars(1)
                task.putvarbound(0, mosek.boundkey.fx, 0.0, 0.0)
                task.optimize()
    except Exception as exc:
        raise RuntimeError(
            "A working MOSEK installation and license are required to reproduce "
            "Experiment 3. See also Readme.md."
            " Install MOSEK and place a valid license at "
            "%USERPROFILE%\\mosek\\mosek.lic (Windows) or "
            "$HOME/mosek/mosek.lic (Linux/macOS), or configure your MOSEK "
            "license server."
        ) from exc


def _check_for_long(long: bool, recipe_name: str) -> None:
    if long and recipe_name == "poro_petrov_vs_galerkin":
        return recipe_name.replace(
            "poro_petrov_vs_galerkin", "poro_petrov_vs_galerkin_long"
        )
    return recipe_name


def _print_catalog() -> None:
    recipe_by_name = {recipe.name: recipe for recipe in RUN_RECIPES}
    print("Paper result recipes:")
    for recipe_name in PAPER_RESULT_RECIPE_NAMES:
        recipe = recipe_by_name[recipe_name]
        print(f"  {recipe.name}: {recipe.description}")


def _write_timings(
    timing_rows: list[tuple[str, float]],
    total_duration: float,
) -> None:
    TIMINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    lines = ["recipe,duration_seconds"]
    for recipe_name, duration in timing_rows:
        lines.append(f"{recipe_name},{duration:.6f}")
    lines.append(f"total,{total_duration:.6f}")
    lines.append("")
    lines.append(f"human_readable_total,{timedelta(seconds=total_duration)}")

    TIMINGS_FILE.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--long",
        action="store_true",
        help="Use the long poro Petrov-Galerkin sweep (2,8,...,200).",
    )
    main(**vars(parser.parse_args()))
