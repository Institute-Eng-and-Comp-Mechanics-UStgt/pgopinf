from __future__ import annotations

import importlib

import pytest

from pgopinf.specs.system.base import (
    SYSTEM_REGISTRY,
    system_default_by_kind,
    system_from_dict,
    system_to_dict,
)
from pgopinf.specs.system.cd_player import CDPlayerSpec
from pgopinf.specs.system.earth_atmosphere import EarthAtmosphereSpec
from pgopinf.specs.system.msd import MassSpringDamperSpec
from pgopinf.specs.system.poro import PoroSpec

SYSTEM_MODULES = (
    "pgopinf.specs.system.msd",
    "pgopinf.specs.system.cd_player",
    "pgopinf.specs.system.poro",
    "pgopinf.specs.system.earth_atmosphere",
)


@pytest.fixture(scope="module", autouse=True)
def register_system_modules() -> None:
    for module_name in SYSTEM_MODULES:
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("kind", "spec"),
    [
        (
            "msd",
            MassSpringDamperSpec(
                n_mass=5,
                m=2.5,
                c=0.75,
                k=1.5,
                input_vals=(0, 2),
                form="ph",
                use_Berlin=False,
            ),
        ),
        ("cd_player", CDPlayerSpec()),
        ("poro", PoroSpec(n_system=320, use_Berlin=True, use_mimo=False, Rshift=2e-3)),
        ("earth_atmosphere", EarthAtmosphereSpec()),
    ],
)
def test_system_spec_round_trip(kind: str, spec) -> None:
    assert spec.kind == kind
    assert system_to_dict(spec)["kind"] == kind
    assert system_from_dict(system_to_dict(spec)) == spec


@pytest.mark.parametrize(
    ("kind", "expected_spec"),
    [
        ("msd", MassSpringDamperSpec()),
        ("cd_player", CDPlayerSpec()),
        ("poro", PoroSpec()),
        ("earth_atmosphere", EarthAtmosphereSpec()),
    ],
)
def test_system_default_by_kind(kind: str, expected_spec) -> None:
    assert system_default_by_kind(kind) == expected_spec


def test_known_system_kinds_are_registered() -> None:
    assert SYSTEM_REGISTRY.known_kinds() == [
        "cd_player",
        "earth_atmosphere",
        "msd",
        "poro",
    ]


def test_unknown_system_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown kind"):
        system_default_by_kind("missing_kind")
