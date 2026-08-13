from pgopinf.specs.evaluation.evaluation import EvaluationSpec


def evaluation_presets(name: str) -> EvaluationSpec:
    """Return a named evaluation preset."""

    if name == "metrics_over_time_test":
        return EvaluationSpec.from_dict(
            {
                "metrics": [
                    {
                        "kind": "state_values_over_time",
                        "split": "TEST",
                        "target": "identified",
                        "relative": True,
                    },
                    {
                        "kind": "state_values_over_time",
                        "split": "TEST",
                        "target": "intrusive",
                        "relative": True,
                    },
                    {
                        "kind": "output_values_over_time",
                        "split": "TEST",
                        "target": "identified",
                        "relative": True,
                    },
                    {
                        "kind": "output_values_over_time",
                        "split": "TEST",
                        "target": "intrusive",
                        "relative": True,
                    },
                    {
                        "kind": "input_values_over_time",
                        "split": "TEST",
                    },
                ]
            }
        )

    elif name == "metrics_over_time_train":
        return EvaluationSpec.from_dict(
            {
                "metrics": [
                    {
                        "kind": "state_values_over_time",
                        "split": "TRAIN",
                        "target": "identified",
                        "relative": True,
                    },
                    {
                        "kind": "state_values_over_time",
                        "split": "TRAIN",
                        "target": "intrusive",
                        "relative": True,
                    },
                    {
                        "kind": "output_values_over_time",
                        "split": "TRAIN",
                        "target": "identified",
                        "relative": True,
                    },
                    {
                        "kind": "output_values_over_time",
                        "split": "TRAIN",
                        "target": "intrusive",
                        "relative": True,
                    },
                    {
                        "kind": "input_values_over_time",
                        "split": "TRAIN",
                    },
                ]
            }
        )

    elif name == "system_error_metrics":
        return EvaluationSpec.from_dict(
            {
                "metrics": [
                    {
                        "kind": "hinf_error",
                        "target": "identified",
                    },
                    {
                        "kind": "hinf_error",
                        "target": "intrusive",
                    },
                    {
                        "kind": "hinf_error",
                        "target": "identified",
                        "stabilized": True,
                    },
                    {
                        "kind": "hinf_error",
                        "target": "intrusive",
                        "stabilized": True,
                    },
                    {
                        "kind": "singular_value_decay",
                    },
                ]
            }
        )
    elif name == "system_error_metrics_high_dimensional":
        return EvaluationSpec.from_dict(
            {
                "metrics": [
                    {
                        "kind": "hinf_error",
                        "target": "identified",
                        "use_minimal_realization": True,
                    },
                    {
                        "kind": "hinf_error",
                        "target": "intrusive",
                        "use_minimal_realization": True,
                    },
                    {
                        "kind": "hinf_error",
                        "target": "identified",
                        "stabilized": True,
                        "use_minimal_realization": True,
                    },
                    {
                        "kind": "hinf_error",
                        "target": "intrusive",
                        "stabilized": True,
                        "use_minimal_realization": True,
                    },
                    {
                        "kind": "singular_value_decay",
                    },
                ]
            }
        )
    elif name == "opinf_theory_metrics":
        return EvaluationSpec.from_dict(
            {
                "metrics": [
                    {
                        "kind": "opinf_theory",
                    },
                    {
                        "kind": "singular_value_decay",
                    },
                ]
            }
        )
    elif name == "matrix_comparison":
        return EvaluationSpec.from_dict(
            {
                "metrics": [
                    {
                        "kind": "comp_int_ident_matABCD",
                    },
                ]
            }
        )
    elif name == "standard_metrics_exp1":
        # eval_spec1 = EvaluationSpec.preset("system_error_metrics")
        eval_spec2 = EvaluationSpec.preset("opinf_theory_metrics")
        # eval_spec3 = EvaluationSpec.preset("metrics_over_time_train")
        # eval_spec4 = EvaluationSpec.preset("metrics_over_time_test")
        # return eval_spec1 + eval_spec2 + eval_spec3 + eval_spec4
        return eval_spec2
    elif name == "standard_metrics_exp2":
        eval_spec1 = EvaluationSpec.preset("system_error_metrics")
        return eval_spec1
    elif name == "standard_metrics_exp3":
        eval_spec1 = EvaluationSpec.preset("system_error_metrics")
        eval_spec2 = EvaluationSpec.preset("metrics_over_time_test")
        return eval_spec1 + eval_spec2
    elif name == "standard_metrics_exp1_high_dimensional":
        eval_spec1 = EvaluationSpec.preset("system_error_metrics_high_dimensional")
        eval_spec2 = EvaluationSpec.preset("opinf_theory_metrics")
        eval_spec3 = EvaluationSpec.preset("metrics_over_time_train")
        eval_spec4 = EvaluationSpec.preset("metrics_over_time_test")
        return eval_spec1 + eval_spec2 + eval_spec3 + eval_spec4
    elif name == "metrics_over_time_train_and_test":
        eval_spec1 = EvaluationSpec.preset("metrics_over_time_train")
        eval_spec2 = EvaluationSpec.preset("metrics_over_time_test")
        return eval_spec1 + eval_spec2
    elif name == "high_dimensional_system_metrics":
        eval_spec1 = EvaluationSpec.preset("metrics_over_time_train")
        eval_spec2 = EvaluationSpec.preset("metrics_over_time_test")
        eval_spec3 = EvaluationSpec.from_dict(
            {
                "metrics": [
                    {
                        "kind": "spectral_abscissa",
                        "target": "identified",
                    },
                    {
                        "kind": "spectral_abscissa",
                        "target": "intrusive",
                    },
                    {
                        "kind": "spectral_abscissa",
                        "target": "original",
                    },
                ]
            }
        )
        return eval_spec1 + eval_spec2 + eval_spec3

    else:
        raise ValueError(f"Unknown evaluation preset name: {name}")
