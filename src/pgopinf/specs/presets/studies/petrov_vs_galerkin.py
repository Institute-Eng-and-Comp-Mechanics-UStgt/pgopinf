from pgopinf.specs.evaluation.evaluation import EvaluationSpec
from pgopinf.specs.evaluation_study.scalar_plot import ScalarPlotSpec
from pgopinf.specs.evaluation_study.stability_fraction_by_group import (
    StabilityFractionByGroupSpec,
)
from pgopinf.specs.study import StudyAxis, StudySpec, StudyVariant


def petrov_vs_galerkin(**kwargs) -> StudySpec:
    """Create the Petrov-Galerkin versus Galerkin study preset."""
    if "r_sweep" in kwargs:
        r_sweep = tuple(kwargs["r_sweep"])
    else:
        r_sweep = (5, 10, 20)

    if "title_prefix" in kwargs:
        title_prefix = f"{kwargs['title_prefix']}_"
    else:
        title_prefix = ""

    if "name" in kwargs:
        name = kwargs["name"]
    else:
        name = "petrov_vs_galerkin"

    study_spec = StudySpec(
        name=name,
        include_base_spec=False,
        variants=(
            StudyVariant(
                name="pod_opinf_g",
                overrides={
                    "reduction.full_basis.kind": "pod",
                    "identification.kind": "opinf",
                    "identification.convert_to_ph": False,
                    "evaluation": EvaluationSpec.preset(
                        "standard_metrics_exp2"
                    ).to_dict(),
                },
            ),
            StudyVariant(
                name="pod_opinf_pg",
                overrides={
                    "reduction.full_basis.kind": "pod",
                    "reduction.test_basis.kind": "vq",
                    "reduction.test_basis.source": "system_Q",
                    "identification.kind": "opinf",
                    "identification.convert_to_ph": False,
                    "evaluation": EvaluationSpec.preset(
                        "standard_metrics_exp2"
                    ).to_dict(),
                },
            ),
        ),
        axes=(StudyAxis(parameter="reduction.r", values=r_sweep),),
        evaluations=(
            ScalarPlotSpec(
                metrics=("hinf_error",),
                x="reduction.r",
                line_by=("variant", "target", "stabilized"),
                filters={},
                title=f"{title_prefix}hinf_error_petrov_vs_galerkin",
            ),
            ScalarPlotSpec(
                metrics=("spectral_abscissa_cmp",),
                x="reduction.r",
                line_by=("variant", "target", "stabilized"),
                filters={},
                title=f"{title_prefix}spectral_abscissa_cmp_over_r",
            ),
            StabilityFractionByGroupSpec(
                group_by=("variant", "target"),
                filters={"stabilized": False},
                title=f"{title_prefix}stability_fraction_by_variant",
            ),
        ),
    )

    return study_spec
