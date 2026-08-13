from pgopinf.specs.evaluation.evaluation import EvaluationSpec
from pgopinf.specs.evaluation_study.array_plot import (
    ArrayPlotStudySpec,
)
from pgopinf.specs.evaluation_study.scalar_plot import ScalarPlotSpec
from pgopinf.specs.study import StudyAxis, StudySpec, StudyVariant


def g_pg_hamcvx_ph(**kwargs) -> StudySpec:
    """Create the Galerkin/Petrov-Galerkin Hamiltonian-CVX study preset."""
    if "r_sweep" in kwargs:
        r_sweep = tuple(kwargs["r_sweep"])
    else:
        r_sweep = (5, 10, 20)

    if "output_r" in kwargs:
        output_r = kwargs["output_r"]
    else:
        output_r = 20

    if "title_prefix" in kwargs:
        title_prefix = f"{kwargs['title_prefix']}_"
    else:
        title_prefix = ""

    study_spec = StudySpec(
        name="g_pg_hamcvx_ph",
        include_base_spec=False,
        variants=(
            StudyVariant(
                name="pod_opinf_g",
                overrides={
                    "identification.kind": "opinf",
                    "identification.convert_to_ph": False,
                    "evaluation": EvaluationSpec.preset(
                        "standard_metrics_exp3"
                    ).to_dict(),
                },
            ),
            StudyVariant(
                name="pod_opinf_pg_Qknown",
                overrides={
                    "reduction.test_basis.kind": "vq",
                    "reduction.test_basis.source": "system_Q",
                    "identification.kind": "opinf",
                    "identification.convert_to_ph": False,
                    "evaluation": EvaluationSpec.preset(
                        "standard_metrics_exp3"
                    ).to_dict(),
                },
            ),
            StudyVariant(
                name="pod_opinf_pg_Qhamcvx",
                overrides={
                    "reduction.test_basis.kind": "vq",
                    "reduction.test_basis.source": "Hamiltonian_cvx",
                    "identification.kind": "opinf",
                    "identification.convert_to_ph": False,
                    "evaluation": EvaluationSpec.preset(
                        "standard_metrics_exp3"
                    ).to_dict(),
                },
            ),
            StudyVariant(
                name="pod_convex_ph",
                overrides={
                    "reduction.test_basis.kind": "galerkin",
                    "identification.kind": "convex_ph_inference",
                    "identification.return_system_type": "ph",
                    "identification.project_psd": True,
                    "identification.accept_unknown_mosek": False,
                    "evaluation": EvaluationSpec.preset(
                        "standard_metrics_exp3"
                    ).to_dict(),
                },
            ),
        ),
        axes=(StudyAxis(parameter="reduction.r", values=r_sweep),),
        evaluations=(
            # ScalarPlotSpec(
            #     metrics=("hinf_error",),
            #     x="reduction.r",
            #     line_by=("variant", "target"),
            #     filters={"stabilized": False},
            #     title=f"{title_prefix}hinf_error_g_pg_hamcvx_ph",
            #     line_mode_by=("variant", "target"),
            #     line_mode_map={
            #         ("pod_opinf_g", "identified"): "line+marker",
            #         ("pod_opinf_g", "intrusive"): "line+marker",
            #         ("pod_opinf_pg_Qknown", "identified"): "line+marker",
            #         ("pod_opinf_pg_Qknown", "intrusive"): "line+marker",
            #         ("pod_opinf_pg_Qhamcvx", "identified"): "line+marker",
            #         ("pod_opinf_pg_Qhamcvx", "intrusive"): "line+marker",
            #         ("pod_convex_ph", "identified"): "line+marker",
            #         ("pod_convex_ph", "intrusive"): "line+marker",
            #     },
            # ),
            ScalarPlotSpec(
                metrics=("hinf_error",),
                x="reduction.r",
                line_by=("variant", "target", "stabilized"),
                # filters={},
                title=f"{title_prefix}hinf_error_g_pg_hamcvx_ph",
                line_mode_by=("variant", "target"),
                line_mode_map={
                    ("pod_opinf_g", "identified"): "line+marker",
                    ("pod_opinf_g", "intrusive"): "line+marker",
                    ("pod_opinf_pg_Qknown", "identified"): "line+marker",
                    ("pod_opinf_pg_Qknown", "intrusive"): "line+marker",
                    ("pod_opinf_pg_Qhamcvx", "identified"): "line+marker",
                    ("pod_opinf_pg_Qhamcvx", "intrusive"): "line+marker",
                    ("pod_convex_ph", "identified"): "line+marker",
                    ("pod_convex_ph", "intrusive"): "line+marker",
                },
            ),
            ScalarPlotSpec(
                metrics=("spectral_abscissa_cmp", "spectral_abscissa_orig"),
                x="reduction.r",
                line_by=("metric", "variant", "target"),
                filters={"stabilized": False},
                title=f"{title_prefix}spectral_abscissa_cmp_over_r",
            ),
            ScalarPlotSpec(
                metrics=("max_error_output", "mean_error_output"),
                x="reduction.r",
                line_by=("metric", "variant", "target"),
                filters={"target": "identified", "split": "TEST"},
                title=f"{title_prefix}max_mean_error_output_over_r",
            ),
            ArrayPlotStudySpec(
                artifact_name="error_output_vs_original_over_time",
                artifact_filters={
                    "split": "TEST",
                    "target": "identified",
                    "relative": True,
                },
                subplot_indices=(0,),
                line_labels=("y_0",),
                variants=(
                    "pod_opinf_g",
                    "pod_opinf_pg_Qknown",
                    "pod_opinf_pg_Qhamcvx",
                    "pod_convex_ph",
                ),
                line_by=("variant"),
                filters={"reduction.r": output_r},
                title=f"{title_prefix}_error_output0_orig_vs_identified_r{output_r}",
                line_mode_by=("variant",),
                line_mode_map={
                    ("pod_opinf_g",): "line",
                    ("pod_opinf_pg_Qknown",): "line+marker",
                    ("pod_opinf_pg_Qhamcvx",): "line+marker",
                    ("pod_convex_ph",): "line+marker",
                },
                linestyle_by=("variant",),
                linestyle_map={
                    ("pod_opinf_g",): "-",
                    ("pod_opinf_pg_Qknown",): "--",
                    ("pod_opinf_pg_Qhamcvx",): "--",
                    ("pod_convex_ph",): "--",
                },
            ),
            ArrayPlotStudySpec(
                artifact_name="output_vs_original_over_time",
                artifact_filters={
                    "split": "TEST",
                    "target": "identified",
                },
                subplot_indices=(0,),
                line_labels=("identified", "original"),
                variants=(
                    "pod_opinf_g",
                    "pod_opinf_pg_Qknown",
                    "pod_opinf_pg_Qhamcvx",
                    "pod_convex_ph",
                ),
                line_by=("variant", "source_line"),
                filters={"reduction.r": output_r},
                title=f"{title_prefix}output0_orig_vs_identified_r{output_r}",
                line_mode_by=("variant",),
                line_mode_map={
                    ("pod_opinf_g",): "line",
                    ("pod_opinf_pg_Qknown",): "line+marker",
                    ("pod_opinf_pg_Qhamcvx",): "line+marker",
                    ("pod_convex_ph",): "line+marker",
                },
                linestyle_by=("variant",),
                linestyle_map={
                    ("pod_opinf_g",): "-",
                    ("pod_opinf_pg_Qknown",): "--",
                    ("pod_opinf_pg_Qhamcvx",): "--",
                    ("pod_convex_ph",): "--",
                },
            ),
        ),
    )
    return study_spec
