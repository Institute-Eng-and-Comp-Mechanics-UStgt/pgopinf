from __future__ import annotations

from pathlib import Path

from pgopinf.io.results_path import (
    ResultsPath,
    RunResultsPath,
    StudyResultsPath,
    _ensure_dir,
    _sanitize_name,
)


def test_sanitize_name_replaces_path_and_separator_characters() -> None:
    assert _sanitize_name("my run/name\\with:parts") == "my_run_name_with_parts"
    assert _sanitize_name(123) == "123"


def test_ensure_dir_creates_directory_and_returns_path(tmp_path) -> None:
    path = tmp_path / "nested" / "dir"

    returned = _ensure_dir(path)

    assert returned == path
    assert path.is_dir()


def test_results_path_roots_create_expected_directories(tmp_path) -> None:
    paths = ResultsPath(tmp_path / "results")

    expected = {
        "runs_root": paths.results_root / "runs",
        "datasets_root": paths.results_root / "datasets",
        "system_analysis_tasks": paths.results_root / "system_analysis_tasks",
        "intrusive_datasets_root": paths.results_root / "intrusive_datasets",
        "identified_datasets_root": paths.results_root / "identified_datasets",
        "full_basis_root": paths.results_root / "full_bases",
        "inferred_q_matrix": paths.results_root / "inferred_q_matrix",
        "identifications_root": paths.results_root / "identifications",
        "studies_root": paths.results_root / "studies",
    }

    for attr, path in expected.items():
        assert getattr(paths, attr) == path
        assert path.is_dir()


def test_results_path_factory_methods_return_runtime_path_objects(tmp_path) -> None:
    paths = ResultsPath(tmp_path / "results")

    run_paths = paths.for_run(experiment_name="exp/name", ident_id="abc")
    study_paths = paths.for_study(study_name="study/name")

    assert run_paths == RunResultsPath(
        root=paths,
        experiment_name="exp/name",
        ident_id="abc",
    )
    assert study_paths == StudyResultsPath(root=paths, study_name="study/name")


def test_results_path_artifact_directories_are_prefixed_and_created(tmp_path) -> None:
    paths = ResultsPath(tmp_path / "results")

    assert paths.dataset_dir("data-id") == paths.datasets_root / "ds_data-id"
    assert (
        paths.intrusive_dataset_dir("data-id")
        == paths.intrusive_datasets_root / "ds_data-id"
    )
    assert (
        paths.identified_dataset_dir("data-id")
        == paths.identified_datasets_root / "ds_data-id"
    )
    assert (
        paths.full_bases_dir("basis-id")
        == paths.full_basis_root / "full_basis_basis-id"
    )
    assert paths.identification_dir("ident-id") == (
        paths.identifications_root / "identification_ident-id"
    )

    for path in (
        paths.dataset_dir("data-id"),
        paths.intrusive_dataset_dir("data-id"),
        paths.identified_dataset_dir("data-id"),
        paths.full_bases_dir("basis-id"),
        paths.identification_dir("ident-id"),
    ):
        assert path.is_dir()


def test_run_results_path_sanitizes_experiment_name_and_creates_output_dirs(
    tmp_path,
) -> None:
    root = ResultsPath(tmp_path / "results")
    run_paths = root.for_run(experiment_name="exp / name:1", ident_id="abc")

    assert run_paths.safe_experiment_name == "exp___name_1"
    assert run_paths.run_dir == root.runs_root / "run_exp___name_1_abc"
    assert run_paths.evaluation_dir == run_paths.run_dir / "evaluation"
    assert (
        run_paths.metric_summaries_csv
        == run_paths.evaluation_dir / "metric_summaries.csv"
    )
    assert run_paths.artifacts_dir == run_paths.evaluation_dir / "artifacts"
    assert run_paths.series_dir == run_paths.artifacts_dir / "series"
    assert run_paths.array_plots_dir == run_paths.artifacts_dir / "array_plots"
    assert run_paths.scalars_dir == run_paths.artifacts_dir / "scalars"
    assert run_paths.scalar_metrics_csv == run_paths.scalars_dir / "scalar_metrics.csv"
    assert run_paths.tables_dir == run_paths.artifacts_dir / "tables"
    assert run_paths.figures_dir == run_paths.artifacts_dir / "figures"

    for path in (
        run_paths.run_dir,
        run_paths.evaluation_dir,
        run_paths.artifacts_dir,
        run_paths.series_dir,
        run_paths.array_plots_dir,
        run_paths.scalars_dir,
        run_paths.tables_dir,
        run_paths.figures_dir,
    ):
        assert path.is_dir()


def test_study_results_path_sanitizes_names_and_creates_dirs(tmp_path) -> None:
    root = ResultsPath(tmp_path / "results")
    study_paths = root.for_study(study_name="study / name:1")

    assert study_paths.safe_study_name == "study___name_1"
    assert study_paths.study_dir == root.studies_root / "study___name_1"
    assert study_paths.evaluation_dir("scalar / plot:1") == (
        study_paths.study_dir / "scalar___plot_1"
    )
    assert study_paths.scalar_over_parameter_dir == (
        study_paths.study_dir / "scalar_over_parameter"
    )
    assert study_paths.study_dir.is_dir()
    assert study_paths.evaluation_dir("scalar / plot:1").is_dir()
    assert study_paths.scalar_over_parameter_dir.is_dir()


def test_results_path_accepts_plain_path_instances(tmp_path) -> None:
    path = Path(tmp_path) / "results"

    paths = ResultsPath(path)

    assert paths.results_root == path
