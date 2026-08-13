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


from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.specs.system.poro import PoroSpec
from pgopinf.specs.system_analysis.system_analysis import (
    SystemAnalysisSpec,
)


def poro() -> ExperimentSpec:
    """Create the poroelastic experiment preset."""
    input_mode = "separated_u1_u2"  # "separated_u1_u2" | "two_freq_ranges" | "jonas_paper_version"
    if input_mode == "separated_u1_u2":
        # train
        f_min = 1e-1
        f_max = 1e3
        f_mid = np.sqrt(f_min * f_max)
        n_u = 2
        t_end = 10 / f_mid * n_u  # 10 periods of minimal frequency
        dt = 1 / (2 * f_max)  # 10 samples per period of highest frequency
        reduce_sample_n_train = 2
        tspan_u1 = (0.0, t_end / 2)  # only apply u1 in the first half of the time span
        tspan_u2 = (
            t_end / 2,
            t_end,
        )  # only apply u2 in the second half of the time span

        f_min_all = f_min
        f_max_all = f_max
    elif input_mode == "two_freq_ranges":
        # train
        # u1
        f_min1 = 1e-3
        f_max1 = 1e1
        # u2
        f_min2 = 1e0
        f_max2 = 1e3
        # sim parameters
        f_min_all = min(f_min1, f_min2)
        f_max_all = max(f_max1, f_max2)
        f_mid = np.sqrt(f_min_all * f_max_all)
        n_u = 2
        t_end = 10 / f_mid * n_u  # 10 periods of minimal frequency
        dt = 1 / (5 * f_max_all)  # 10 samples per period of highest frequency
        reduce_sample_n_train = 20
        # tspan_u1 = (0.0, t_end)  # apply both u1 and u2 over the entire time span
        # tspan_u2 = (0.0, t_end)

    # test
    t_end_test = t_end * 1.2
    freq_hz = f_mid
    # both
    amp = 1.0
    reduce_sample_n_test = None

    use_mimo = True  # whether to use the MIMO version of the system
    if use_mimo:
        if input_mode == "separated_u1_u2" or input_mode == "jonas_paper_version":
            input_spec_train = ExprInputSpec(
                expr={
                    "kind": "stack",
                    "channels": [
                        {
                            "kind": "chirp",
                            "amp": amp,
                            "f0": f_min,
                            "f1": f_max,
                            "T": t_end / 2,
                            "tspan": tspan_u1,
                        },
                        {
                            "kind": "chirp",
                            "amp": amp,
                            "f0": f_min,
                            "f1": f_max,
                            "T": t_end / 2,
                            "tspan": tspan_u2,
                        },
                    ],
                }
            )
        elif input_mode == "two_freq_ranges":
            input_spec_train = ExprInputSpec(
                expr={
                    "kind": "stack",
                    "channels": [
                        {
                            "kind": "chirp",
                            "amp": amp,
                            "f0": f_min1,
                            "f1": f_max1,
                            "T": t_end,
                        },
                        {
                            "kind": "chirp",
                            "amp": amp,
                            "f0": f_min2,
                            "f1": f_max2,
                            "T": t_end,
                        },
                    ],
                }
            )
        input_spec_test = ExprInputSpec(
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
                        "freq_hz": freq_hz,
                        "amp": -amp,
                    },
                ],
            },
        )
    else:
        input_spec_train = ExprInputSpec(
            expr={
                "kind": "stack",
                "channels": [
                    {
                        "kind": "chirp",
                        "amp": 1.0,
                        "f0": f_min,
                        "f1": f_max,
                        "T": t_end,
                    },
                ],
            }
        )
        input_spec_test = ExprInputSpec(
            expr={
                "kind": "sawtooth",
                "freq_hz": freq_hz,
                "amp": amp,
            },
        )

    return ExperimentSpec(
        # name=f"poro_{input_mode}_fmin{f_min_all}_fmax{f_max_all}_tend{t_end:.3f}_dt{dt}_reduce{reduce_sample_n_train}",
        name=f"poro",
        system=PoroSpec(n_system=980, use_Berlin=False, use_mimo=use_mimo, Rshift=1e-3),
        system_analysis=SystemAnalysisSpec.from_dict(
            {
                "tasks": [
                    {"kind": "eigenvalues", "which": "all"},
                    {"kind": "minimal_realization", "trunc_tol": 1e-12},
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
                input=input_spec_train,
                n_sim=1,
                dXdt_derivation="return",
            ),
            test=SplitDataSpec(
                time=TimeSpec.from_start_end_timesteps(t0=0.0, t_end=t_end_test, dt=dt),
                reduce_sample_n=reduce_sample_n_test,
                initial_condition=ZerosIC(),
                input=input_spec_test,
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
        evaluation=EvaluationSpec.preset(name="standard_metrics_exp1_high_dimensional"),
    )


if __name__ == "__main__":
    exp_spec = poro()
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
