from __future__ import annotations

import importlib

import pytest

from pgopinf.specs.identification.base import (
    IDENTIFICATION_REGISTRY,
    identification_default_by_kind,
    identification_from_dict,
)
from pgopinf.specs.identification.convex_ph_inference import (
    ConvexPHInferenceSpec,
)
from pgopinf.specs.identification.operator_inference import (
    OperatorInferenceSpec,
)


IDENTIFICATION_MODULES = (
    "pgopinf.specs.identification.convex_ph_inference",
    "pgopinf.specs.identification.operator_inference",
)


@pytest.fixture(scope="module", autouse=True)
def register_identification_modules() -> None:
    for module_name in IDENTIFICATION_MODULES:
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("kind", "spec"),
    [
        (
            "convex_ph_inference",
            ConvexPHInferenceSpec(
                J_is_known=True,
                add_regularization=True,
                add_dissipation_inequality_cost=True,
                lambdas=1e-4,
                no_feedthrough=True,
                solver="clarabel",
                project_psd=True,
                return_system_type="lti",
                accept_unknown_mosek=True,
            ),
        ),
        (
            "opinf",
            OperatorInferenceSpec(
                use_E=False,
                convert_to_ph=False,
                seperate_output_inf=True,
                lambda_reg=1e-5,
            ),
        ),
    ],
)
def test_identification_spec_round_trip(kind: str, spec) -> None:
    assert spec.kind == kind
    assert spec.to_dict()["kind"] == kind
    assert identification_from_dict(spec.to_dict()) == spec


def test_known_identification_kinds_are_registered() -> None:
    assert IDENTIFICATION_REGISTRY.known_kinds() == [
        "convex_ph_inference",
        "opinf",
    ]


@pytest.mark.parametrize(
    ("kind", "expected_spec"),
    [
        ("convex_ph_inference", ConvexPHInferenceSpec()),
        ("opinf", OperatorInferenceSpec()),
    ],
)
def test_identification_default_by_kind(kind: str, expected_spec) -> None:
    assert identification_default_by_kind(kind) == expected_spec


def test_identification_from_dict_requires_known_kind() -> None:
    with pytest.raises(KeyError, match="missing required keys"):
        identification_from_dict({})

    with pytest.raises(ValueError, match="unknown kind"):
        identification_from_dict({"kind": "missing"})


def test_identification_from_dict_uses_spec_specific_parsers() -> None:
    assert identification_from_dict(
        {
            "kind": "opinf",
            "lambda_reg": "0.25",
        }
    ) == OperatorInferenceSpec(lambda_reg=0.25)

    assert identification_from_dict(
        {
            "kind": "convex_ph_inference",
            "lambdas": "0.001",
            "solver": "mosek",
        }
    ) == ConvexPHInferenceSpec(lambdas=0.001, solver="mosek")


@pytest.mark.parametrize(
    ("spec", "runtime_type_name"),
    [
        (ConvexPHInferenceSpec(), "ConvexPortHamiltonianIdentifier"),
        (OperatorInferenceSpec(), "OperatorInferenceIdentifier"),
    ],
)
def test_identification_specs_build_runtime_identifiers(
    spec, runtime_type_name: str
) -> None:
    assert type(spec.build()).__name__ == runtime_type_name


def test_operator_inference_build_forwards_options() -> None:
    spec = OperatorInferenceSpec(
        use_E=False,
        convert_to_ph=False,
        seperate_output_inf=True,
        lambda_reg=1e-3,
    )

    identifier = spec.build()

    assert identifier.use_E is False
    assert identifier.convert_to_ph is False
    assert identifier.seperate_output_inf is True
    assert identifier.lambda_reg == pytest.approx(1e-3)


def test_convex_ph_inference_build_forwards_options() -> None:
    spec = ConvexPHInferenceSpec(
        J_is_known=True,
        add_regularization=True,
        add_dissipation_inequality_cost=True,
        lambdas=1e-4,
        no_feedthrough=True,
        solver="clarabel",
        project_psd=True,
        return_system_type="lti",
        accept_unknown_mosek=True,
    )

    identifier = spec.build()

    assert identifier.J_is_known is True
    assert identifier.add_regularization is True
    assert identifier.add_dissipation_inequality_cost is True
    assert identifier.lambdas == pytest.approx(1e-4)
    assert identifier.no_feedthrough is True
    assert identifier.solver == "clarabel"
    assert identifier.project_psd is True
    assert identifier.return_system_type == "lti"
    assert identifier.accept_unknown_mosek is True
