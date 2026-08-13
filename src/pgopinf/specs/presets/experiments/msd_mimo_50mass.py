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
from pgopinf.specs.system.msd import MassSpringDamperSpec


from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.specs.system_analysis.system_analysis import (
    SystemAnalysisSpec,
)


def msd_mimo_50mass(**kwargs) -> ExperimentSpec:
    """Create the 50-mass MIMO mass-spring-damper experiment preset."""
    # train
    f_min = 1e-2
    f_max = 1
    f_mid = np.sqrt(f_min * f_max)
    n_u = 2
    t_end = 10 / f_mid * n_u  # 10 periods of minimal frequency
    dt = 1 / (2 * f_max)  # 10 samples per period of highest frequency
    if "reduce_sample_n_train" in kwargs:
        reduce_sample_n_train = kwargs["reduce_sample_n_train"]
    else:
        reduce_sample_n_train = 4
    tspan_u1 = (0.0, t_end / 2)  # only apply u1 in the first half of the time span
    tspan_u2 = (t_end / 2, t_end)  # only apply u2 in the second half of the time span

    # test
    t_end_test = t_end * 1.2
    freq_hz = f_mid
    # both
    amp = 1.0
    reduce_sample_n_test = None

    return ExperimentSpec(
        # name=f"msd_mimo_50mass_fmin{f_min}_fmax{f_max}_tend{t_end:.3f}_dt{dt}_reduce{reduce_sample_n_train}",
        name="msd_mimo_50mass",
        system=MassSpringDamperSpec(
            n_mass=50,
            m=4.0,
            c=1.0,
            k=4.0,
            input_vals=(0, 1),
            form="ph",
            use_Berlin=False,
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
                reduce_sample_n=reduce_sample_n_test,
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
                                "freq_hz": freq_hz,
                                "amp": -amp,
                            },
                        ],
                    },
                ),
                n_sim=1,
                dXdt_derivation="return",
            ),
            seed=0,
        ),
        reduction=ReductionSpec(
            r=6,
            full_basis=PODFullBasisSpec(),
            test_basis=GalerkinTestBasisSpec(),
            mor=LTIMORSpec(),
            project_data=True,
            reduce_system=True,
        ),
        identification=OperatorInferenceSpec(convert_to_ph=False),
        evaluation=EvaluationSpec.preset(name="standard_metrics_exp1"),
        system_analysis=SystemAnalysisSpec.from_dict(
            {
                "tasks": [
                    {"kind": "eigenvalues", "which": "all"},
                    {"kind": "sigma_plot"},
                    {"kind": "hinf_norm"},
                ]
            }
        ),
    )


if __name__ == "__main__":
    exp_spec = msd_mimo_50mass()
    print(exp_spec)
