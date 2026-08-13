from pgopinf.specs.evaluation_study.scalar_plot import ScalarPlotSpec
from pgopinf.specs.study import StudyAxis, StudySpec, StudyVariant


def test_convex_ph_and_g_opinf(**kwargs):
    """Create a small convex pH versus Galerkin OpInf test study preset."""
    if "r_sweep" in kwargs:
        r_sweep = tuple(kwargs["r_sweep"])
    else:
        r_sweep = (5, 10, 20)

    if "title_prefix" in kwargs:
        title_prefix = f"{kwargs['title_prefix']}_"
    else:
        title_prefix = ""

    study_spec = StudySpec(
        name="test_convex_ph_and_g_opinf",
        include_base_spec=False,
        variants=(
            # StudyVariant(
            #     name="pod_opinf_g",
            #     overrides={
            #         "identification.kind": "opinf",
            #         "identification.convert_to_ph": False,
            #     },
            # ),
            StudyVariant(
                name="pod_convex_ph_mosek",
                overrides={
                    "reduction.test_basis.kind": "galerkin",
                    "identification.kind": "convex_ph_inference",
                    "identification.return_system_type": "ph",
                    "identification.project_psd": True,
                    "identification.solver": "mosek",
                },
            ),
        ),
        axes=(StudyAxis(parameter="reduction.r", values=r_sweep),),
        evaluations=(
            ScalarPlotSpec(
                metrics=("hinf_error",),
                x="reduction.r",
                line_by=("variant", "target"),
                filters={"stabilized": False},
                title=f"{title_prefix}hinf_error_petrov_galerkin_and_ph",
                line_mode_by=("variant", "target"),
                line_mode_map={
                    # ("pod_opinf_pg_Qknown", "identified"): "line+marker",
                    # ("pod_opinf_pg_Qknown", "intrusive"): "line+marker",
                    # ("pod_opinf_pg_Qham", "identified"): "line+marker",
                    # ("pod_opinf_pg_Qham", "intrusive"): "line+marker",
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
        ),
    )

    return study_spec
