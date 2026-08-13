from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pgopinf.specs.presets.run_recipes import RUN_RECIPES


def _load_module():
    module_path = Path("scripts/reproduce_paper_results.py")
    spec = importlib.util.spec_from_file_location(
        "reproduce_paper_results",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paper_result_recipe_names_exist() -> None:
    module = _load_module()

    known_recipe_names = {recipe.name for recipe in RUN_RECIPES}

    assert set(module.PAPER_RESULT_RECIPE_NAMES) <= known_recipe_names
    assert len(module.PAPER_RESULT_RECIPE_NAMES) == len(
        set(module.PAPER_RESULT_RECIPE_NAMES)
    )


def test_main_checks_mosek_license_before_starting(monkeypatch) -> None:
    module = _load_module()
    calls = []

    def unavailable_license() -> None:
        calls.append("license")
        raise RuntimeError("license unavailable")

    monkeypatch.setattr(module, "_check_mosek_license", unavailable_license)
    monkeypatch.setattr(module, "setup_logging", lambda: calls.append("setup"))

    with pytest.raises(RuntimeError, match="license unavailable"):
        module.main()

    assert calls == ["license"]


def test_mosek_license_error_has_setup_instructions(monkeypatch) -> None:
    module = _load_module()

    class BrokenMosek:
        class Env:
            def __init__(self) -> None:
                raise OSError("no license")

    monkeypatch.setitem(__import__("sys").modules, "mosek", BrokenMosek)

    with pytest.raises(RuntimeError, match="valid license") as exc_info:
        module._check_mosek_license()

    assert isinstance(exc_info.value.__cause__, OSError)
