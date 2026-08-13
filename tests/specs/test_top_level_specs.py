from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from pgopinf.specs.base import (
    SpecBase,
    apply_overrides_dict,
    canonicalize,
    dataclass_to_dict,
    require_keys,
    sanitize_name,
    set_in_dict,
    short_id,
    stable_id,
    stable_json_dumps,
)
from pgopinf.specs.data.base import DataSpec
from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.specs.identification.convex_ph_inference import (
    ConvexPHInferenceSpec,
)
from pgopinf.specs.identification.operator_inference import (
    OperatorInferenceSpec,
)
from pgopinf.specs.overrides import (
    UnknownOverrideKey,
    validate_override_paths,
)
from pgopinf.specs.reduction.base import ReductionSpec
from pgopinf.specs.reduction.full_basis import ModalFullBasisSpec
from pgopinf.specs.registry import (
    KindRegistry,
    restore_dataclass_kwargs,
)
from pgopinf.specs.study import StudyAxis, StudySpec, StudyVariant
from pgopinf.specs.system.msd import MassSpringDamperSpec
from pgopinf.specs.system_analysis.system_analysis import (
    SystemAnalysisSpec,
)
from pgopinf.specs.system_analysis.tasks import EigenvaluesSpec


@dataclass(frozen=True)
class NestedSpec(SpecBase):
    values: tuple[int, ...]


@dataclass(frozen=True)
class ExampleSpec(SpecBase):
    name: str
    nested: NestedSpec
    np_value: np.float64


def test_canonicalize_normalizes_nested_values() -> None:
    obj = {
        "tuple": (1, 2),
        "set": {"b", "a"},
        "dataclass": NestedSpec(values=(3, 4)),
        5: np.int64(6),
    }

    assert canonicalize(obj) == {
        "tuple": [1, 2],
        "set": ["a", "b"],
        "dataclass": {"values": [3, 4]},
        "5": 6,
    }


def test_stable_json_and_ids_are_order_independent() -> None:
    left = {"b": 2, "a": (1, 3)}
    right = {"a": [1, 3], "b": 2}

    assert stable_json_dumps(left) == stable_json_dumps(right)
    assert stable_json_dumps(left) == '{"a":[1,3],"b":2}'
    assert stable_id(left) == stable_id(right)
    assert len(stable_id(left)) == 12
    assert short_id("abcdef123456", 6) == "abcdef"


def test_apply_overrides_dict_sets_nested_paths() -> None:
    data = {"a": {"b": 1}}

    assert apply_overrides_dict(data, {"a.b": 2, "a.c.d": 3}) == {
        "a": {"b": 2, "c": {"d": 3}}
    }

    set_in_dict(data, "x.y", 4)
    assert data["x"]["y"] == 4


def test_dataclass_serialization_helpers() -> None:
    spec = ExampleSpec(
        name="example",
        nested=NestedSpec(values=(1, 2)),
        np_value=np.float64(1.5),
    )

    assert spec.to_dict() == {
        "name": "example",
        "nested": {"values": [1, 2]},
        "np_value": 1.5,
    }
    assert dataclass_to_dict(spec) == spec.to_dict()

    with pytest.raises(TypeError, match="dataclass_to_dict expects"):
        dataclass_to_dict({"not": "dataclass"})

    with pytest.raises(TypeError, match="np.ndarray is not JSON-stable"):
        ExampleSpec(
            name="bad",
            nested=NestedSpec(values=(1,)),
            np_value=np.array([1.0]),  # type: ignore[arg-type]
        ).to_dict()


def test_require_keys_and_sanitize_name() -> None:
    require_keys({"a": 1, "b": 2}, ["a"])

    with pytest.raises(KeyError, match="where: missing required keys"):
        require_keys({"a": 1}, ["a", "b"], where="where")

    assert sanitize_name("  My Fancy/Name 2026!! ") == "my_fancy_name_2026"
    assert sanitize_name("abcdefghijklmnopqrstuvwxyz", max_len=5) == "abcde"


@dataclass(frozen=True)
class RegistryTupleSpec(SpecBase):
    kind: str = "tuple"
    values: tuple[int, ...] = ()


@dataclass(frozen=True)
class RegistryParsedSpec(SpecBase):
    kind: str = "parsed"
    value: int = 0

    @classmethod
    def from_dict(cls, d):
        return cls(value=int(d.get("value", 0)))


def test_kind_registry_registers_defaults_and_restores_tuple_fields() -> None:
    registry = KindRegistry("example")
    registry.register("tuple")(RegistryTupleSpec)

    assert registry.known_kinds() == ["tuple"]
    assert registry.default_by_kind("tuple") == RegistryTupleSpec()
    assert registry.from_dict({"kind": "tuple", "values": [1, 2]}) == (
        RegistryTupleSpec(values=(1, 2))
    )


def test_kind_registry_uses_custom_from_dict_and_reports_errors() -> None:
    registry = KindRegistry("example")
    registry.register("parsed")(RegistryParsedSpec)

    assert registry.from_dict({"kind": "parsed", "value": "5"}) == (
        RegistryParsedSpec(value=5)
    )

    with pytest.raises(KeyError, match="example.from_dict"):
        registry.from_dict({})

    with pytest.raises(ValueError, match="example: unknown kind"):
        registry.default_by_kind("missing")


def test_restore_dataclass_kwargs_converts_tuple_lists() -> None:
    assert restore_dataclass_kwargs(
        RegistryTupleSpec, {"kind": "tuple", "values": [1, 2]}
    ) == {"kind": "tuple", "values": (1, 2)}


def make_experiment_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="My Experiment",
        system=MassSpringDamperSpec(n_mass=3),
        data=DataSpec(),
        reduction=ReductionSpec(r=5),
        identification=OperatorInferenceSpec(convert_to_ph=False),
        system_analysis=SystemAnalysisSpec(tasks=(EigenvaluesSpec(),)),
        seed=11,
    )


def test_experiment_spec_round_trip_without_optional_none_fields() -> None:
    spec = make_experiment_spec()
    without_optional = ExperimentSpec(
        name="No Optional",
        system=MassSpringDamperSpec(),
        data=DataSpec(),
        reduction=ReductionSpec(),
        identification=OperatorInferenceSpec(),
        evaluation=None,
        system_analysis=None,
        seed=2,
    )

    assert ExperimentSpec.from_dict(spec.to_dict()) == spec
    assert "evaluation" not in without_optional.to_dict()
    assert "system_analysis" not in without_optional.to_dict()
    assert ExperimentSpec.from_dict(without_optional.to_dict()) == without_optional


def test_experiment_spec_from_dict_requires_keys() -> None:
    with pytest.raises(KeyError, match="missing key 'name'"):
        ExperimentSpec.from_dict({})

    d = make_experiment_spec().to_dict()
    d.pop("system")
    with pytest.raises(KeyError, match="missing key 'system'"):
        ExperimentSpec.from_dict(d)


def test_experiment_default_and_preset_helpers() -> None:
    default = ExperimentSpec.default("msd", name="Custom Name")
    preset = ExperimentSpec.preset("msd_mimo_50mass")

    assert default.name == "Custom Name"
    assert default.system.kind == "msd"
    assert default.safe_name == "custom_name"
    assert len(default.spec_id) == 12
    assert preset.name == "msd_mimo_50mass"


def test_experiment_overrides_simple_fields_and_kind_switches() -> None:
    spec = make_experiment_spec().with_overrides(
        {
            "name": "Changed",
            "identification.kind": "convex_ph_inference",
            "identification.solver": "clarabel",
            "reduction.full_basis.kind": "modal",
            "reduction.full_basis.compute_left": False,
        }
    )

    assert spec.name == "Changed"
    assert spec.identification == ConvexPHInferenceSpec(solver="clarabel")
    assert spec.reduction.full_basis == ModalFullBasisSpec(compute_left=False)


def test_experiment_overrides_reject_unknown_or_impossible_paths() -> None:
    spec = make_experiment_spec()

    with pytest.raises(UnknownOverrideKey, match="Unknown override key"):
        spec.with_overrides({"system.missing": 1})

    with pytest.raises(UnknownOverrideKey, match="is not a dataclass"):
        spec.with_overrides({"name.value": "bad"})

    with pytest.raises(UnknownOverrideKey, match="is None"):
        ExperimentSpec(
            name="No Analysis",
            system=MassSpringDamperSpec(),
            data=DataSpec(),
            reduction=ReductionSpec(),
            identification=OperatorInferenceSpec(),
            system_analysis=None,
        ).with_overrides({"system_analysis.tasks": []})


def test_validate_override_paths_accepts_kind_switch_nested_fields() -> None:
    validate_override_paths(
        make_experiment_spec(),
        {
            "identification.kind": "convex_ph_inference",
            "identification.return_system_type": "lti",
        },
    )


def test_experiment_save_writes_specs(tmp_path) -> None:
    spec = make_experiment_spec()

    spec.save(tmp_path, spec_ids={"a": "b"})

    assert json.loads((tmp_path / "spec.json").read_text()) == spec.to_dict()
    assert json.loads((tmp_path / "spec_ids.json").read_text()) == {"a": "b"}


def test_study_axis_validation() -> None:
    assert StudyAxis(parameter="reduction.r", values=[1, 2]).to_dict() == {
        "parameter": "reduction.r",
        "values": [1, 2],
    }

    with pytest.raises(ValueError, match="parameter must not be empty"):
        StudyAxis(parameter="", values=(1,))

    with pytest.raises(ValueError, match="values must not be empty"):
        StudyAxis(parameter="reduction.r", values=())


def test_study_spec_validation_and_preset() -> None:
    spec = StudySpec(
        name="study",
        variants=(StudyVariant(name="variant", overrides={"reduction.r": 2}),),
        axes=(StudyAxis(parameter="reduction.r", values=(2, 4)),),
    )

    assert spec.to_dict()["variants"][0]["name"] == "variant"
    assert StudySpec.preset("opinf_theory_over_r").name == "opinf_theory_over_r"

    with pytest.raises(ValueError, match="variants must not be empty"):
        StudySpec(variants=())

    with pytest.raises(ValueError, match="Unknown study preset name"):
        StudySpec.preset("missing")
