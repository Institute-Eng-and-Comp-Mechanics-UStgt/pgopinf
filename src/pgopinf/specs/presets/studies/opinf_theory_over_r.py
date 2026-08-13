from pgopinf.specs.evaluation.evaluation import EvaluationSpec
from pgopinf.specs.evaluation_study.scalar_plot import ScalarPlotSpec
from pgopinf.specs.study import StudyAxis, StudySpec, StudyVariant


def opinf_theory_over_r(**kwargs) -> StudySpec:
    """Create the operator-inference theory over reduced order study preset."""

    if "r_sweep" in kwargs:
        r_sweep = tuple(kwargs["r_sweep"])
    else:
        r_sweep = (5, 10, 20)

    if "title_prefix" in kwargs:
        title_prefix = f"{kwargs['title_prefix']}_"
    else:
        title_prefix = ""

    study_spec = StudySpec(
        name="opinf_theory_over_r",
        include_base_spec=False,
        variants=(
            StudyVariant(
                name="pod_opinf",
                overrides={
                    "reduction.full_basis.kind": "pod",
                    "identification.kind": "opinf",
                    "identification.convert_to_ph": False,
                    "evaluation": EvaluationSpec.preset(
                        "standard_metrics_exp1"
                    ).to_dict(),
                },
            ),
        ),
        axes=(StudyAxis(parameter="reduction.r", values=r_sweep),),
        evaluations=(
            ScalarPlotSpec(
                metrics=(
                    "rel_error_operators_combined",
                    "rel_error_bias_term_combined_data_frob_norm",
                    "rel_bound_Xperp_Tr",
                    "rel_error_bound_operators_combined",
                    "singular_value_r",
                    "cond_rhs_data_reduced",
                ),
                x="reduction.r",
                line_by=("metric",),
                filters={},
                title=f"{title_prefix}opinf_metrics_over_r",
                line_mode_by=("metric",),
                line_mode_map={
                    ("rel_error_bias_term_combined_data_frob_norm",): "marker",
                    # ("comp_int_ident_matA",): "marker",
                },
            ),
            # ScalarPlotSpec(
            #     metrics=("hinf_error",),
            #     x="reduction.r",
            #     line_by=("variant", "target", "stabilized"),
            #     filters={},
            #     title="hinf_error_over_r",
            # ),
        ),
    )
    return study_spec
