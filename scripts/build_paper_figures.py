# Path to studies
import os
from pathlib import Path

from matplotlib.ticker import LogLocator
import pandas as pd
import matplotlib.pyplot as plt

STUDIES_ROOT = Path("results/studies/")
RESULTS_PAPER_ROOT = Path("results_paper/")


def main() -> None:
    build_figures_exp1()
    build_figures_exp2()
    build_figures_exp3()


def build_figures_exp1() -> None:
    folder_names_exp1 = [
        "cd_player_opinf_theory_over_r",
        "earth_atmosphere_opinf_theory_over_r",
        "msd_mimo_50mass_opinf_theory_over_r",
        "poro_opinf_theory_over_r",
    ]
    exp_names = [
        "cd_player",
        "earth_atmosphere",
        "msd_mimo_50mass",
        "poro",
    ]

    metric_to_mpl_style = {
        "eps_ref": {
            "label": r"$\varepsilon_{\text{ref}}$",
            "color": "red",
            "linestyle": "-",
            "marker": "o",
        },
        "eps_corr": {
            "label": r"$\varepsilon_{\text{corr}}$",
            "color": "black",
            "linestyle": "None",
            "marker": "x",
        },
        "eps_bound": {
            "label": r"$\varepsilon_{\text{bound}}$",
            "color": "green",
            "linestyle": "-",
            "marker": "^",
        },
        "eps_est": {
            "label": r"$\varepsilon_{\text{est}}$",
            "color": "purple",
            "linestyle": "-",
            "marker": "^",
        },
        "sigma": {
            "label": r"$\sigma$",
            "color": "orange",
            "linestyle": "-",
            "marker": "D",
        },
    }

    data = {}
    # read data from folders
    for exp_name, folder_name in zip(exp_names, folder_names_exp1):
        folder_path = STUDIES_ROOT / folder_name
        # read data from folder_path
        subfolder = os.path.join(folder_path, f"{exp_name}_opinf_metrics_over_r")
        csv_file = os.path.join(subfolder, f"{exp_name}_opinf_metrics_over_r.csv")
        data[exp_name] = pd.read_csv(csv_file)

    # build 2x2 figure
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for i, exp_name in enumerate(exp_names):
        ax = axs[i // 2, i % 2]
        df = data[exp_name]
        for metric, styles in metric_to_mpl_style.items():
            df_metric = df[metric]
            ax.semilogy(
                df["r"],
                df_metric,
                label=styles["label"],
                color=styles["color"],
                linestyle=styles["linestyle"],
                marker=styles.get("marker", None),
            )
        ax.set_title(exp_name)
        ax.set_xlabel(r"Reduced order $r$")
        ax.set_ylabel("Rel. error")

    handles, labels = axs[0, 0].get_legend_handles_labels()

    # First reserve space for the legend
    # fig.subplots_adjust(top=0.85)

    # Then place legend in that reserved top space
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.95),
        # frameon=False,
    )
    # First let matplotlib fix title/xlabel/ylabel overlaps
    fig.tight_layout()

    # Then reserve space for the legend
    fig.subplots_adjust(top=0.85)

    if not RESULTS_PAPER_ROOT.exists():
        RESULTS_PAPER_ROOT.mkdir(parents=True)
    plt.savefig(RESULTS_PAPER_ROOT / "exp1_opinf_theory.pdf")
    print("Finished.")


def build_figures_exp2() -> None:
    folder_name_poro_long = "poro_petrov_vs_galerkin_fine_r"
    folder_name_poro_short = "poro_petrov_vs_galerkin"
    # check which folder exists and choose the most recent one
    folder_name_poro = (
        folder_name_poro_long
        if (STUDIES_ROOT / folder_name_poro_long).exists()
        else folder_name_poro_short
    )

    folder_names_exp2 = [
        "msd_mimo_50mass_petrov_vs_galerkin_fine_r",
        folder_name_poro,
    ]
    exp_names = [
        "msd_mimo_50mass",
        "poro",
    ]

    metric_to_mpl_style = {
        # G-POD (standard)
        "pod_opinf_g | intrusive | False": {
            "label": "G-POD",
            "color": "red",
            "linestyle": "-",
            "marker": "o",
        },
        # G-POD (stabilized)
        "pod_opinf_g | intrusive | True": {
            "label": None,
            "color": "red",
            "linestyle": "--",
            "marker": "o",
        },
        # G-OpInf (standard)
        "pod_opinf_g | identified | False": {
            "label": "G-OpInf",
            "color": "purple",
            "linestyle": "-",
            "marker": "^",
        },
        # G-OpInf (stabilized)
        "pod_opinf_g | identified | True": {
            "label": None,
            "color": "purple",
            "linestyle": "--",
            "marker": "^",
        },
        # PG-POD (standard)
        "pod_opinf_pg | intrusive | False": {
            "label": "PG-POD",
            "color": "green",
            "linestyle": "-",
            "marker": "s",
        },
        # PG-POD (stabilized)
        "pod_opinf_pg | intrusive | True": {
            "label": None,
            "color": "green",
            "linestyle": "--",
            "marker": "s",
        },
        # PG-OpInf (standard)
        "pod_opinf_pg | identified | False": {
            "label": "PG-OpInf",
            "color": "orange",
            "linestyle": "-",
            "marker": "D",
        },
        # PG-OpInf (stabilized)
        "pod_opinf_pg | identified | True": {
            "label": None,
            "color": "orange",
            "linestyle": "--",
            "marker": "D",
        },
    }

    data = {}
    # read data from folders
    for exp_name, folder_name in zip(exp_names, folder_names_exp2):
        folder_path = STUDIES_ROOT / folder_name
        # read data from folder_path
        subfolder = os.path.join(
            folder_path, f"{exp_name}_hinf_error_petrov_vs_galerkin"
        )
        csv_file = os.path.join(
            subfolder, f"{exp_name}_hinf_error_petrov_vs_galerkin.csv"
        )
        data[exp_name] = pd.read_csv(csv_file)

    # build 1x2 figure
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for i, exp_name in enumerate(exp_names):
        ax = axs[i]
        df = data[exp_name]
        if exp_name == "msd_mimo_50mass":
            # use only 2,8,...,80 for r values to avoid clutter
            df = df[df["r"] % 6 == 2]
        else:
            df = df[df["r"] <= 80]
        for metric, styles in metric_to_mpl_style.items():
            df_metric = df[metric]
            ax.semilogy(
                df["r"],
                df_metric,
                label=styles["label"],
                color=styles["color"],
                linestyle=styles["linestyle"],
                marker=styles.get("marker", None),
            )
        ax.set_title(exp_name)
        ax.set_xlabel(r"Reduced order $r$")
        ax.set_ylabel(r"Rel. $\mathcal{H}_{\infty}$ error")
        ax.grid(True, which="major", linestyle="--", linewidth=0.8)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))

    # Then place legend in that reserved top space
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.95),
        # frameon=False,
    )
    # First let matplotlib fix title/xlabel/ylabel overlaps
    fig.tight_layout()

    # Then reserve space for the legend
    fig.subplots_adjust(top=0.75)

    if not RESULTS_PAPER_ROOT.exists():
        RESULTS_PAPER_ROOT.mkdir(parents=True)
    plt.savefig(RESULTS_PAPER_ROOT / "exp2_petrov_vs_galerkin.pdf")

    # build table
    folder_name_spec_abscissa = os.path.join(
        STUDIES_ROOT, folder_name_poro, "poro_spectral_abscissa_cmp_over_r"
    )
    csv_name_path = os.path.join(
        folder_name_spec_abscissa, "poro_spectral_abscissa_cmp_over_r.csv"
    )
    df_spec_abscissa = pd.read_csv(csv_name_path)
    # extract an rename relevant columns
    df_spec_abscissa = df_spec_abscissa[
        [
            "r",
            "pod_opinf_g | intrusive | False",
            "pod_opinf_g | identified | False",
            "pod_opinf_pg | intrusive | False",
            "pod_opinf_pg | identified | False",
        ]
    ]
    df_spec_abscissa.rename(
        columns={
            "pod_opinf_g | intrusive | False": "G-POD",
            "pod_opinf_g | identified | False": "G-OpInf",
            "pod_opinf_pg | intrusive | False": "PG-POD",
            "pod_opinf_pg | identified | False": "PG-OpInf",
        },
        inplace=True,
    )
    df_spec_abscissa.to_csv(
        RESULTS_PAPER_ROOT / "exp2_poro_spectral_abscissa_table.csv", index=False
    )


def build_figure1_exp3() -> None:
    folder_name_exp3 = STUDIES_ROOT / "msd_mimo_50mass_g_pg_hamcvx_ph"
    # Plot 1: H-infinity error over r
    subfolder_hinf = os.path.join(
        folder_name_exp3, "msd_mimo_50mass_hinf_error_g_pg_hamcvx_ph"
    )
    csv_file = os.path.join(
        subfolder_hinf, "msd_mimo_50mass_hinf_error_g_pg_hamcvx_ph.csv"
    )
    df_hinf = pd.read_csv(csv_file)

    metric_to_mpl_style = {
        # G-OpInf (standard)
        "pod_opinf_g | identified | False": {
            "label": "G-OpInf",
            "color": "purple",
            "linestyle": "-",
            "marker": "^",
        },
        # PG-OpInf (standard)
        "pod_opinf_pg_Qknown | identified | False": {
            "label": "PG-OpInf",
            "color": "orange",
            "linestyle": "-",
            "marker": "D",
        },
        # PG-OpInf-Ham
        "pod_opinf_pg_Qhamcvx | identified | False": {
            "label": "PG-OpInf-Ham",
            "color": "red",
            "linestyle": "-",
            "marker": "o",
        },
        # pHOpInf-CVX
        "pod_convex_ph | identified | False": {
            "label": "pHOpInf-CVX",
            "color": "green",
            "linestyle": "-",
            "marker": "s",
        },
    }

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for metric, styles in metric_to_mpl_style.items():
        df_metric = df_hinf[metric]
        ax.semilogy(
            df_hinf["r"],
            df_metric,
            label=styles["label"],
            color=styles["color"],
            linestyle=styles["linestyle"],
            marker=styles.get("marker", None),
        )
    ax.set_title("msd_mimo_50mass")
    ax.set_xlabel(r"Reduced order $r$")
    ax.set_ylabel(r"Rel. $\mathcal{H}_{\infty}$ error")
    ax.grid(True, which="major", linestyle="--", linewidth=0.8)

    # Then place legend in that reserved top space
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.95),
        # frameon=False,
    )
    # First let matplotlib fix title/xlabel/ylabel overlaps
    fig.tight_layout()

    # Then reserve space for the legend
    fig.subplots_adjust(top=0.75)

    if not RESULTS_PAPER_ROOT.exists():
        RESULTS_PAPER_ROOT.mkdir(parents=True)
    plt.savefig(RESULTS_PAPER_ROOT / "exp3_hinf_error.pdf")


def build_figure2_exp3() -> None:
    folder_name_exp3 = STUDIES_ROOT / "msd_mimo_50mass_g_pg_hamcvx_ph"
    # Plot 1: H-infinity error over r
    subfolder_output = os.path.join(
        folder_name_exp3, "msd_mimo_50mass_output0_orig_vs_identified_r20"
    )
    subfolder_output_error = os.path.join(
        folder_name_exp3, "msd_mimo_50mass__error_output0_orig_vs_identified_r20"
    )
    csv_file_output = os.path.join(
        subfolder_output, "msd_mimo_50mass_output0_orig_vs_identified_r20.csv"
    )
    csv_file_output_error = os.path.join(
        subfolder_output_error,
        "msd_mimo_50mass__error_output0_orig_vs_identified_r20.csv",
    )
    df_output = pd.read_csv(csv_file_output)
    df_output_error = pd.read_csv(csv_file_output_error)

    output_metric_to_mpl_style = {
        # Reference
        "pod_opinf_g | original": {
            "label": "reference",
            "color": "black",
            "linestyle": "-",
            "marker": "o",
        },
        # G-OpInf (standard)
        "pod_opinf_g | identified": {
            "label": "G-OpInf",
            "color": "purple",
            "linestyle": "--",
            # "marker": "D",
        },
        # PG-OpInf
        "pod_opinf_pg_Qknown | identified": {
            "label": "PG-OpInf",
            "color": "orange",
            "linestyle": "--",
            # "marker": "o",
        },
        # pHOpInf-Ham
        "pod_opinf_pg_Qhamcvx | identified": {
            "label": "pHOpInf-Ham",
            "color": "red",
            "linestyle": "-.",
            # "marker": "s",
        },
        # pHOpInf-CVX
        "pod_convex_ph | identified": {
            "label": "pHOpInf-CVX",
            "color": "green",
            "linestyle": "-.",
            # "marker": "s",
        },
    }

    output_error_metric_to_mpl_style = {
        # G-OpInf (standard)
        "pod_opinf_g": {
            "label": "G-OpInf",
            "color": "purple",
            "linestyle": "-",
            # "marker": "D",
        },
        # PG-OpInf
        "pod_opinf_pg_Qknown": {
            "label": "PG-OpInf",
            "color": "orange",
            "linestyle": "-",
            # "marker": "o",
        },
        # pHOpInf-Ham
        "pod_opinf_pg_Qhamcvx": {
            "label": "pHOpInf-Ham",
            "color": "red",
            "linestyle": "-",
            # "marker": "s",
        },
        # pHOpInf-CVX
        "pod_convex_ph": {
            "label": "pHOpInf-CVX",
            "color": "green",
            "linestyle": "-",
            # "marker": "s",
        },
    }

    # only in the time interval 50 <= t <= 100
    df_output = df_output[(df_output["time"] >= 50) & (df_output["time"] <= 100)]
    df_output_error = df_output_error[
        (df_output_error["time"] >= 50) & (df_output_error["time"] <= 100)
    ]

    # build 1x2 figure with output and error
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    # output
    ax = axs[0]
    for metric, styles in output_metric_to_mpl_style.items():
        df_metric = df_output[metric]
        ax.plot(
            df_output["time"],
            df_metric,
            label=styles["label"],
            color=styles["color"],
            linestyle=styles["linestyle"],
            marker=styles.get("marker", None),
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Output")
    # error
    ax = axs[1]
    for metric, styles in output_error_metric_to_mpl_style.items():
        df_metric = df_output_error[metric]
        ax.semilogy(
            df_output_error["time"],
            df_metric,
            label=styles["label"],
            color=styles["color"],
            linestyle=styles["linestyle"],
            marker=styles.get("marker", None),
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Rel. output error")

    # Then place legend in that reserved top space
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.95),
        # frameon=False,
    )
    # First let matplotlib fix title/xlabel/ylabel overlaps
    fig.tight_layout()

    # Then reserve space for the legend
    fig.subplots_adjust(top=0.75)

    if not RESULTS_PAPER_ROOT.exists():
        RESULTS_PAPER_ROOT.mkdir(parents=True)
    plt.savefig(RESULTS_PAPER_ROOT / "exp3_output.pdf")


def build_figures_exp3() -> None:
    build_figure1_exp3()
    # build_figure2_exp3()


if __name__ == "__main__":
    main()
    print("Done.")
