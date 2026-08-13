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
from pgopinf.specs.system.earth_atmosphere import (
    EarthAtmosphereSpec,
)

from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.specs.system_analysis.system_analysis import (
    SystemAnalysisSpec,
)


def earth_atmosphere() -> ExperimentSpec:
    """Create the earth-atmosphere experiment preset."""

    # train
    f_min = 1e-2
    f_max = 1e1
    f_mid = np.sqrt(f_min * f_max)
    t_end = 10 / f_mid  # 10 periods of minimal frequency
    dt = 1 / (2 * f_max)  # 10 samples per period of highest frequency
    reduce_sample_n_train = None

    # test
    t_end_test = t_end * 1.2
    freq_hz = f_mid
    # both
    amp = 1.0
    reduce_sample_n_test = None

    return ExperimentSpec(
        name=f"earth_atmosphere",
        system=EarthAtmosphereSpec(),
        system_analysis=SystemAnalysisSpec.from_dict(
            {"tasks": [{"kind": "sigma_plot"}, {"kind": "hinf_norm"}]}
        ),
        data=DataSpec(
            train=SplitDataSpec(
                time=TimeSpec.from_start_end_timesteps(t0=0.0, t_end=t_end, dt=dt),
                reduce_sample_n=reduce_sample_n_train,
                initial_condition=ZerosIC(),
                input=ExprInputSpec(
                    expr={
                        "kind": "chirp",
                        "amp": amp,
                        "f0": f_min,
                        "f1": f_max,
                        "T": t_end,
                    },
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
                        "kind": "sawtooth",
                        "freq_hz": freq_hz,
                        "amp": amp,
                    },
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
    exp_spec = earth_atmosphere()
    print(exp_spec)
