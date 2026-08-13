from __future__ import annotations

import numpy as np
import pytest

from pgopinf.specs.data.base import DataSpec, SplitDataSpec
from pgopinf.specs.data.initial_condition.base import (
    RandomIC,
    ZerosIC,
    ic_from_dict,
)
from pgopinf.specs.data.inputs.base import (
    ExprInputSpec,
    PresetInputSpec,
    input_from_dict,
)
from pgopinf.specs.data.time.base import TimeSpec


def test_time_spec_round_trip_and_grid_helpers() -> None:
    spec = TimeSpec(t0=1.0, t_end=3.0, n_steps=4)

    assert TimeSpec.from_dict(spec.to_dict()) == spec
    assert spec.dt_effective() == pytest.approx(0.5)
    np.testing.assert_allclose(spec.t_eval(), np.array([1.0, 1.5, 2.0, 2.5, 3.0]))

    time = spec.build()
    np.testing.assert_allclose(time.t, np.linspace(1.0, 3.0, 4))


def test_time_spec_from_dict_accepts_dt_and_rejects_inconsistent_dt() -> None:
    spec = TimeSpec.from_dict({"t0": 0.0, "t_end": 1.0, "dt": 0.25})

    assert spec == TimeSpec(t0=0.0, t_end=1.0, n_steps=5)

    with pytest.raises(ValueError, match="not consistent"):
        TimeSpec.from_dict({"t0": 0.0, "t_end": 1.0, "dt": 0.3})

    with pytest.raises(ValueError, match="dt too large"):
        TimeSpec.from_dict({"t0": 0.0, "t_end": 1.0, "dt": 2.0})


def test_initial_condition_specs_round_trip_and_build() -> None:
    zeros = ic_from_dict({"kind": "zeros"})
    assert zeros == ZerosIC()
    assert zeros.to_dict() == {"kind": "zeros"}
    np.testing.assert_allclose(zeros.build(n=3, n_sim=2, seed=None).x0, np.zeros((3, 2)))

    random = ic_from_dict({"kind": "random", "scaling": "0.5"})
    assert random == RandomIC(scaling=0.5)
    assert random.to_dict() == {"kind": "random", "scaling": 0.5}

    x0_a = random.build(n=2, n_sim=3, seed=123).x0
    x0_b = random.build(n=2, n_sim=3, seed=123).x0
    assert x0_a.shape == (2, 3)
    assert np.all((0.0 <= x0_a) & (x0_a <= 0.5))
    np.testing.assert_allclose(x0_a, x0_b)


def test_unknown_initial_condition_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown ic.kind"):
        ic_from_dict({"kind": "missing"})


def test_input_specs_parse_and_round_trip() -> None:
    preset = input_from_dict(
        {
            "kind": "preset",
            "name": "chirp_siso",
            "options": {"amp": 2.0},
            "n_sim": 3,
        }
    )
    assert preset == PresetInputSpec(name="chirp_siso", options={"amp": 2.0}, n_sim=3)
    assert preset.to_dict() == {
        "kind": "preset",
        "name": "chirp_siso",
        "options": {"amp": 2.0},
        "n_sim": 3,
    }

    expr = input_from_dict({"kind": "expr", "expr": {"kind": "constant", "value": 2}})
    assert expr == ExprInputSpec(expr={"kind": "constant", "value": 2})
    assert expr.to_dict() == {"kind": "expr", "expr": {"kind": "constant", "value": 2}}


def test_unknown_input_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown input.kind"):
        input_from_dict({"kind": "missing"})

    with pytest.raises(ValueError, match="Unknown input.kind"):
        input_from_dict({"kind": "from_file"})


def test_expr_input_builds_constant_and_stack_inputs() -> None:
    time = TimeSpec(t0=0.0, t_end=1.0, n_steps=5).build()

    constant = ExprInputSpec(expr={"kind": "constant", "value": 2.5}).build(
        time=time, seed=0, n_sim=3
    )
    assert constant.n_sim() == 3
    assert constant.n_u() == 1
    np.testing.assert_allclose(constant.evaluate(time.t), np.full((1, 5, 3), 2.5))

    stacked = ExprInputSpec(
        expr={
            "kind": "stack",
            "channels": [
                {"kind": "constant", "value": 1.0},
                {"kind": "constant", "value": -1.0},
            ],
        }
    ).build(time=time, seed=0, n_sim=2)
    expected = np.empty((2, 5, 2))
    expected[0, :, :] = 1.0
    expected[1, :, :] = -1.0
    np.testing.assert_allclose(stacked.evaluate(time.t), expected)


def test_expr_input_unknown_expression_kind_raises() -> None:
    time = TimeSpec(n_steps=3).build()

    with pytest.raises(ValueError, match="Unknown expr.kind"):
        ExprInputSpec(expr={"kind": "missing"}).build(time=time, seed=0, n_sim=1)


def test_preset_input_repeats_single_trajectory_for_requested_simulations() -> None:
    time = TimeSpec(t0=0.0, t_end=1.0, n_steps=5).build()

    built = PresetInputSpec(name="sawtooth").build(time=time, seed=7, n_sim=4)

    assert built.n_sim() == 4
    assert built.evaluate(time.t).shape == (1, 5, 4)


def test_split_data_spec_round_trip_and_exclude() -> None:
    spec = SplitDataSpec.from_dict(
        {
            "time": {"t0": 0.0, "t_end": 2.0, "n_steps": 8},
            "initial_condition": {"kind": "random", "scaling": 0.25},
            "input": {"kind": "expr", "expr": {"kind": "constant", "value": 3.0}},
            "n_sim": "4",
            "dXdt_derivation": "finite_difference",
            "reduce_sample_n": 12,
        }
    )

    assert spec == SplitDataSpec(
        time=TimeSpec(t0=0.0, t_end=2.0, n_steps=8),
        initial_condition=RandomIC(scaling=0.25),
        input=ExprInputSpec(expr={"kind": "constant", "value": 3.0}),
        n_sim=4,
        dXdt_derivation="finite_difference",
        reduce_sample_n=12,
    )
    assert SplitDataSpec.from_dict(spec.to_dict()) == spec

    without_input = spec.to_dict(exclude={"input"})
    assert "input" not in without_input
    assert without_input["n_sim"] == 4


def test_data_spec_round_trip_and_split_helpers() -> None:
    data = DataSpec.from_dict(
        {
            "train": {"n_sim": 3, "initial_condition": {"kind": "random"}},
            "test": {"n_sim": 2, "input": {"kind": "expr"}},
            "seed": "123",
        }
    )

    assert data.seed == 123
    assert data.train.n_sim == 3
    assert data.test.n_sim == 2
    assert DataSpec.from_dict(data.to_dict()) == data
    assert data.get_split_spec("TRAIN") == data.train
    assert data.get_split_spec("test") == data.test
    assert data.split_to_dict("Train", exclude={"input"}) == {
        "split": "train",
        "split_spec": data.train.to_dict(exclude={"input"}),
        "seed": 123,
    }

    with pytest.raises(ValueError, match="Invalid split"):
        data.get_split_spec("validation")
