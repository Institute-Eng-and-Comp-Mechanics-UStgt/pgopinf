from __future__ import annotations

import numpy as np

from pgopinf.specs.data.base import DataSpec, SplitDataSpec
from pgopinf.specs.data.initial_condition.base import ZerosIC
from pgopinf.specs.data.inputs.base import (
    ExprInputSpec,
)
from pgopinf.specs.data.time.base import TimeSpec
from pgopinf.specs.evaluation.evaluation import EvaluationSpec
from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.specs.identification.operator_inference import (
    OperatorInferenceSpec,
)
from pgopinf.specs.reduction.base import ReductionSpec
from pgopinf.specs.reduction.full_basis import PODFullBasisSpec
from pgopinf.specs.reduction.mor import LTIMORSpec
from pgopinf.specs.reduction.test_basis import (
    GalerkinTestBasisSpec,
)
from pgopinf.specs.system.cd_player import CDPlayerSpec


from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.specs.system_analysis.system_analysis import (
    SystemAnalysisSpec,
)


def cd_player() -> ExperimentSpec:
    """Create the CD-player experiment preset."""
    # train
    f_min = 1
    f_max = 1e5
    f_mid = np.sqrt(f_min * f_max)
    n_u = 2
    t_end = 10 / f_mid * n_u  # 10 periods of minimal frequency
    dt = 1 / (2 * f_max)  # 10 samples per period of highest frequency
    reduce_sample_n_train = 10
    tspan_u1 = (0.0, t_end / 2)  # only apply u1 in the first half of the time span
    tspan_u2 = (t_end / 2, t_end)  # only apply u2 in the second half of the time span

    # test
    t_end_test = t_end * 1.2
    freq_hz = f_mid
    amp = 1.0

    return ExperimentSpec(
        name=f"cd_player",
        system=CDPlayerSpec(),
        system_analysis=SystemAnalysisSpec.from_dict(
            {
                "tasks": [
                    {"kind": "sigma_plot"},
                    {"kind": "hinf_norm"},
                ]
            }
        ),
        data=DataSpec(
            train=SplitDataSpec(
                time=TimeSpec.from_start_end_timesteps(t0=0.0, t_end=t_end, dt=dt),
                reduce_sample_n=reduce_sample_n_train,
                initial_condition=ZerosIC(),
                input=ExprInputSpec(
                    expr={
                        "kind": "stack",
                        "channels": [
                            {
                                "kind": "chirp",
                                "amp": 1.0,
                                "f0": f_min,
                                "f1": f_max,
                                "T": t_end / 2,
                                "tspan": tspan_u1,
                            },
                            {
                                "kind": "chirp",
                                "amp": 1.0,
                                "f0": f_min,
                                "f1": f_max,
                                "T": t_end / 2,
                                "tspan": tspan_u2,
                            },
                        ],
                    }
                ),
                n_sim=1,
                dXdt_derivation="return",
            ),
            test=SplitDataSpec(
                time=TimeSpec.from_start_end_timesteps(t0=0.0, t_end=t_end_test, dt=dt),
                reduce_sample_n=None,
                initial_condition=ZerosIC(),
                input=ExprInputSpec(
                    expr={
                        "kind": "stack",
                        "channels": [
                            {
                                "kind": "sawtooth",
                                "freq_hz": freq_hz,
                                "amp": amp,
                            },
                            {
                                "kind": "sawtooth",
                                "freq_hz": freq_hz / 2,
                                "amp": -amp,
                            },
                        ],
                    }
                ),
                n_sim=1,
                dXdt_derivation="return",
            ),
            seed=0,
        ),
        reduction=ReductionSpec(
            r=4,
            full_basis=PODFullBasisSpec(),
            test_basis=GalerkinTestBasisSpec(),
            mor=LTIMORSpec(),
            project_data=True,
            reduce_system=True,
        ),
        identification=OperatorInferenceSpec(convert_to_ph=False),
        evaluation=EvaluationSpec.preset(name="standard_metrics_exp1"),
    )


if __name__ == "__main__":
    exp_spec = cd_player()
    print(f"time n_steps: {exp_spec.data.train.time.n_steps}")
    print(f"reduce sample n: {exp_spec.data.train.reduce_sample_n}")
    red_samp = (
        exp_spec.data.train.reduce_sample_n
        if exp_spec.data.train.reduce_sample_n is not None
        else 1
    )
    print(
        f"Effective time steps after reduction: {exp_spec.data.train.time.n_steps // red_samp}"
    )
    print(exp_spec)
