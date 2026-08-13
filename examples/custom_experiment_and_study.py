from __future__ import annotations

from pathlib import Path

from pgopinf.logging_config import setup_logging
from pgopinf.specs.evaluation_study.scalar_plot import ScalarPlotSpec
from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.specs.study import StudyAxis, StudySpec, StudyVariant
from pgopinf.workflows.run import run_study_specs


def build_experiment() -> ExperimentSpec:
    return ExperimentSpec.default(
        system="msd",
        name="my_msd_experiment",
        overrides={
            "system.n_mass": 6,
            "reduction.r": 4,
            "identification.kind": "opinf",
            "identification.convert_to_ph": False,
        },
    )


def build_study() -> StudySpec:
    return StudySpec(
        name="my_custom_study",
        include_base_spec=False,
        variants=(
            StudyVariant(
                name="galerkin",
                overrides={
                    "reduction.test_basis.kind": "galerkin",
                    "identification.kind": "opinf",
                    "identification.convert_to_ph": False,
                },
            ),
            StudyVariant(
                name="petrov_galerkin_q",
                overrides={
                    "reduction.test_basis.kind": "vq",
                    "reduction.test_basis.source": "system_Q",
                    "identification.kind": "opinf",
                    "identification.convert_to_ph": False,
                },
            ),
        ),
        axes=(StudyAxis(parameter="reduction.r", values=(2, 4, 6)),),
        evaluations=(
            ScalarPlotSpec(
                metrics=("hinf_error",),
                x="reduction.r",
                line_by=("variant", "target", "stabilized"),
                filters={},
                title="my_hinf_error_over_r",
            ),
        ),
    )


def main() -> None:
    setup_logging()
    exp_spec = build_experiment()
    study_spec = build_study()

    run_study_specs(
        exp_spec,
        study_spec,
        results_root=Path("results_custom"),
        force_recompute=False,
    )


if __name__ == "__main__":
    main()
