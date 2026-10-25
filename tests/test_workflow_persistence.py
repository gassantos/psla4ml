"""Testes de persistência dos manifestos P1 de workflow."""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from experiment import persistence
from experiment.workflow import (
    ExperimentDefinition,
    ExperimentRun,
    TaskActivity,
    TaskDefinition,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
)


def test_write_workflow_run_writes_manifest_and_task(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = ExperimentRun(
        experiment_run_id="run-1",
        definition_name="workflow",
        status="success",
        tasks=[TaskRun(
            "train", "Treino", "train", TaskStatus.SUCCEEDED,
            config={"batch_size": 32}, input_signatures={"dataset": "abc123"},
        )],
    )

    run_dir = persistence.write_workflow_run(workflow)

    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        assert json.load(f)["experiment_run_id"] == "run-1"
    with open(run_dir / "tasks" / "train.json", encoding="utf-8") as f:
        task = json.load(f)
    assert task["status"] == "succeeded"
    assert task["config"] == {"batch_size": 32}
    assert task["input_signatures"] == {"dataset": "abc123"}


def test_load_workflow_run_restores_attempt_history(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = ExperimentRun(
        experiment_run_id="run-1",
        definition_name="workflow",
        status="failed",
        tasks=[
            TaskRun(
                "train",
                "Treino",
                "train",
                TaskStatus.FAILED,
                [
                    TaskExecutionAttempt(
                        "attempt-1", 1, TaskStatus.FAILED, error="timeout", error_type="TimeoutError"
                    )
                ],
                config={"epochs": 3},
                input_signatures={"dataset": "abc123"},
            )
        ],
    )

    restored = persistence.load_workflow_run(persistence.write_workflow_run(workflow))

    assert restored.experiment_run_id == "run-1"
    assert restored.tasks[0].attempts[0].error_type == "TimeoutError"
    assert restored.tasks[0].config == {"epochs": 3}
    assert restored.tasks[0].input_signatures == {"dataset": "abc123"}


def test_append_workflow_csv_rows_writes_one_row_per_task_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    definition = ExperimentDefinition(
        "workflow", (TaskDefinition("train", "Treinar", activity=TaskActivity.ADAPTATION),)
    )
    workflow = ExperimentRun(
        "run-1", "workflow", "success", [TaskRun(
            "train", "Treinar", "train", TaskStatus.SUCCEEDED, [
                TaskExecutionAttempt(
                    "attempt-1", 1, TaskStatus.SUCCEEDED,
                    metrics={
                        "resources": {"task_time_sec": 12.5, "energy_kwh": 0.02},
                        "evaluation": {"f1_score": 0.9},
                    },
                    artifacts={"checkpoint": "model.pt"},
                ),
            ],
        )],
    )

    csv_path = persistence.append_workflow_csv_rows(
        workflow,
        device_type="CPU",
        definition=definition,
        context={
            "environment": {"precision": "fp32"},
            "requested_environment": "COLAB",
            "selected_environment": "my-laptop",
        },
    )

    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["task_id"] == "train"
    assert rows[0]["activity"] == "adaptation"
    assert rows[0]["task_time_sec"] == "12.5"
    assert rows[0]["root_error_type"] == ""
    assert rows[0]["selected_environment"] == "my-laptop"
    assert "experiment_context_json" not in rows[0]
    assert rows[0]["configured_environment"] == ""  # chave ausente no contexto -> coluna vazia


def test_append_workflow_csv_rows_expands_context_into_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = _make_workflow_with_attempts([TaskExecutionAttempt("a1", 1, TaskStatus.SUCCEEDED)])

    csv_path = persistence.append_workflow_csv_rows(
        workflow,
        device_type="CPU",
        context={
            "parallel_workers": 2,
            "selected_environment": "LOCAL",
            "configured_environment": "COLAB",
            "resource_machine_name": "host-1",
            "requested_environment": "AWS",
            "grid_params": {"learning_rate": 1e-05, "batch_size": 8, "optimizer": "adamw"},
        },
    )

    with open(csv_path, newline="", encoding="utf-8") as file:
        (row,) = list(csv.DictReader(file))
    assert row["parallel_workers"] == "2"
    assert row["selected_environment"] == "LOCAL"
    assert row["configured_environment"] == "COLAB"
    assert row["resource_machine_name"] == "host-1"
    assert row["grid_learning_rate"] == "1e-05"
    assert row["grid_batch_size"] == "8"
    assert row["grid_optimizer"] == "adamw"
    assert row["grid_dropout"] == ""  # hiperparâmetro não usado nesta combinação
    assert "requested_environment" not in row and "grid_params" not in row


def test_summary_csv_header_is_flat_and_has_no_duplicate_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = _make_workflow_with_attempts([TaskExecutionAttempt("a1", 1, TaskStatus.SUCCEEDED)])

    csv_path = persistence.append_workflow_csv_rows(workflow, device_type="CPU")

    with open(csv_path, newline="", encoding="utf-8") as file:
        header = next(csv.reader(file))
    assert len(header) == len(set(header))
    assert not [name for name in header if name.endswith("_json")]
    expected = {
        "total_gflops",
        *persistence.CONTEXT_COLUMNS, *persistence.GRID_PARAM_COLUMNS,
    }
    assert expected <= set(header)


def test_append_workflow_csv_rows_warns_and_skips_unmapped_context_keys(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = _make_workflow_with_attempts([TaskExecutionAttempt("a1", 1, TaskStatus.SUCCEEDED)])

    with caplog.at_level("WARNING", logger=persistence.logger.name):
        csv_path = persistence.append_workflow_csv_rows(
            workflow, device_type="CPU",
            context={"grid_params": {"weight_decay": 0.01, "batch_size": 8}, "surprise": 1},
        )

    with open(csv_path, newline="", encoding="utf-8") as file:
        (row,) = list(csv.DictReader(file))
    assert row["grid_batch_size"] == "8"
    assert "grid_params.weight_decay" in caplog.text and "surprise" in caplog.text


def test_grid_param_columns_follow_param_mapping():
    mapping_path = Path(__file__).parents[1] / "gridsearch" / "config" / "param_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["mapping"]

    assert set(persistence.GRID_PARAM_NAMES) == set(mapping)


def test_append_workflow_csv_rows_writes_root_error_type_for_infra_error(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    definition = ExperimentDefinition(
        "workflow", (TaskDefinition("train", "Treinar", activity=TaskActivity.ADAPTATION),)
    )
    workflow = ExperimentRun(
        "run-2", "workflow", "failed", [TaskRun(
            "train", "Treinar", "train", TaskStatus.FAILED, [
                TaskExecutionAttempt(
                    "attempt-1", 1, TaskStatus.FAILED,
                    error="hub timeout",
                    error_type="InfraError",
                    root_error_type="ReadTimeout",
                ),
            ],
        )],
    )

    csv_path = persistence.append_workflow_csv_rows(workflow, device_type="CPU", definition=definition)

    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["error_type"] == "InfraError"
    assert rows[0]["root_error_type"] == "ReadTimeout"


def test_append_workflow_csv_rows_writes_failure_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    definition = ExperimentDefinition(
        "workflow", (TaskDefinition("train", "Treinar", activity=TaskActivity.ADAPTATION),)
    )
    workflow = ExperimentRun(
        "run-3", "workflow", "failed", [TaskRun(
            "train", "Treinar", "train", TaskStatus.FAILED, [
                TaskExecutionAttempt(
                    "attempt-1", 1, TaskStatus.FAILED,
                    error="init falhou",
                    error_type="RuntimeError",
                    failure_stage="init",
                ),
            ],
        )],
    )

    csv_path = persistence.append_workflow_csv_rows(workflow, device_type="CPU", definition=definition)

    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["failure_stage"] == "init"


def test_build_result_dict_includes_failure_stage(monkeypatch, tmp_path):
    from experiment.persistence import build_result_dict
    result = build_result_dict(
        experiment_id="exp-1", json_filename="x.json", seed=42, status="failed",
        date_exec="20260930", start_iso="2026-09-30T10:00:00",
        end_iso="2026-09-30T10:01:00", device_type="CPU", device_name="cpu",
        precision="fp32", parallel_workers=1, train_dataset_name="train",
        optimizer="adam", learning_rate=1e-4, avg_gflops_per_batch=0.0,
        batch_size=32, epoch=3, exec_time=60.0, energy_kwh=None,
        emissions_kg=None, cost_usd=None, avg_ram=None, peak_ram=None,
        total_gflops=0.0, eval_metrics={}, stdout="", stderr="init error",
        failure_stage="init",
    )
    assert result["logs"]["failure_stage"] == "init"
    assert result["logs"]["stderr_tail"] == "init error"


# ---------------------------------------------------------------------------
# Item 6 — colunas derivadas: is_infra_error, attempt_count, is_final_attempt
# ---------------------------------------------------------------------------

def _make_workflow_with_attempts(attempts: list[TaskExecutionAttempt]) -> ExperimentRun:
    final_status = attempts[-1].status if attempts else TaskStatus.FAILED
    return ExperimentRun(
        "run-x", "workflow", final_status.value,
        [TaskRun("train", "Treinar", "train", final_status, attempts)],
    )


def test_is_infra_error_true_for_infra_error_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = _make_workflow_with_attempts([
        TaskExecutionAttempt("a1", 1, TaskStatus.FAILED, error_type="InfraError"),
    ])
    csv_path = persistence.append_workflow_csv_rows(workflow, device_type="CPU")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["is_infra_error"] == "True"


def test_is_infra_error_false_for_functional_error(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = _make_workflow_with_attempts([
        TaskExecutionAttempt("a1", 1, TaskStatus.FAILED, error_type="RuntimeError"),
    ])
    csv_path = persistence.append_workflow_csv_rows(workflow, device_type="CPU")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["is_infra_error"] == "False"


def test_is_infra_error_false_for_successful_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = _make_workflow_with_attempts([
        TaskExecutionAttempt("a1", 1, TaskStatus.SUCCEEDED),
    ])
    csv_path = persistence.append_workflow_csv_rows(workflow, device_type="CPU")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["is_infra_error"] == "False"


def test_attempt_count_and_is_final_attempt_with_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    attempts = [
        TaskExecutionAttempt("a1", 1, TaskStatus.FAILED, error_type="InfraError"),
        TaskExecutionAttempt("a2", 2, TaskStatus.FAILED, error_type="InfraError"),
        TaskExecutionAttempt("a3", 3, TaskStatus.SUCCEEDED),
    ]
    workflow = _make_workflow_with_attempts(attempts)
    csv_path = persistence.append_workflow_csv_rows(workflow, device_type="CPU")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert all(r["attempt_count"] == "3" for r in rows)
    assert rows[0]["is_final_attempt"] == "False"
    assert rows[1]["is_final_attempt"] == "False"
    assert rows[2]["is_final_attempt"] == "True"


def test_attempt_count_one_for_single_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = _make_workflow_with_attempts([
        TaskExecutionAttempt("a1", 1, TaskStatus.SUCCEEDED),
    ])
    csv_path = persistence.append_workflow_csv_rows(workflow, device_type="CPU")
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["attempt_count"] == "1"
    assert rows[0]["is_final_attempt"] == "True"


def test_daily_summary_csv_has_expected_filename_and_field_formats(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    definition = ExperimentDefinition(
        "workflow", (TaskDefinition("train", "Treinar", activity=TaskActivity.ADAPTATION),)
    )
    workflow = ExperimentRun(
        "run-format", "workflow", "success", [TaskRun(
            "train", "Treinar", "train", TaskStatus.SUCCEEDED, [
                TaskExecutionAttempt(
                    "attempt-1",
                    1,
                    TaskStatus.SUCCEEDED,
                    started_at="2026-10-17T10:00:00+00:00",
                    completed_at="2026-10-17T10:01:00+00:00",
                    metrics={
                        "resources": {
                            "task_time_sec": 60.0,
                            "avg_ram_mb": 512.25,
                            "peak_ram_mb": 768.5,
                            "vram_mb": 1024.0,
                            "peak_vram_mb": 1536.0,
                            "energy_kwh": 0.0012,
                            "emissions_kg_co2": 0.0003,
                            "cost_usd": 0.00014,
                            "total_gflops": 695.9,
                        },
                        "evaluation": {"f1_score": 0.9},
                    },
                    artifacts={"checkpoint": "model.pt"},
                ),
            ],
        )],
    )

    csv_path = persistence.append_workflow_csv_rows(
        workflow,
        device_type="CPU",
        definition=definition,
        context={"selected_environment": "my-laptop"},
    )

    assert re.match(r"^experiment_summary_CPU-\d{8}\.csv$", csv_path.name)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]

    # Campos numéricos devem ser serializados de forma parseável para float.
    for field in (
        "task_time_sec",
        "avg_ram_mb",
        "peak_ram_mb",
        "vram_mb",
        "peak_vram_mb",
        "energy_kwh",
        "emissions_kg_co2",
        "cost_usd",
        "total_gflops",
    ):
        assert row[field] != ""
        value = float(row[field])
        assert value >= 0

    # Campos inteiros e booleanos derivados.
    assert int(row["attempt_number"]) == 1
    assert int(row["attempt_count"]) == 1
    assert row["is_infra_error"] in {"True", "False"}
    assert row["is_final_attempt"] in {"True", "False"}

    # Datas em ISO 8601.
    datetime.fromisoformat(row["started_at"])
    datetime.fromisoformat(row["completed_at"])

    # O contexto vai em colunas e a coluna JSON antiga não é mais escrita.
    assert row["selected_environment"] == "my-laptop"
    assert "experiment_context_json" not in row


def test_append_workflow_csv_rows_propagates_dataset_statistics_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    definition = ExperimentDefinition(
        "workflow",
        (
            TaskDefinition("ingest_dataset", "Carregar dataset", activity=TaskActivity.INGESTION),
            TaskDefinition("adapt_model", "Adaptar modelo", activity=TaskActivity.ADAPTATION),
        ),
    )
    workflow = ExperimentRun(
        "run-dataset-stats",
        "workflow",
        "success",
        [
            TaskRun(
                "ingest_dataset",
                "Carregar dataset",
                "ingest",
                TaskStatus.SUCCEEDED,
                [
                    TaskExecutionAttempt(
                        "ingest-1",
                        1,
                        TaskStatus.SUCCEEDED,
                        metrics={
                            "dataset": {
                                "dataset_size_mb": 12.34,
                                "tokens_per_document": 87.5,
                                "document_count": 120,
                                "chunk_count": 180,
                                "train_samples": 100,
                                "valid_samples": 10,
                                "test_samples": 10,
                            }
                        },
                    )
                ],
            ),
            TaskRun(
                "adapt_model",
                "Adaptar modelo",
                "train",
                TaskStatus.SUCCEEDED,
                [TaskExecutionAttempt("adapt-1", 1, TaskStatus.SUCCEEDED)],
            ),
        ],
    )

    csv_path = persistence.append_workflow_csv_rows(workflow, device_type="CPU", definition=definition)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    for row in rows:
        assert float(row["dataset_size_mb"]) == pytest.approx(12.34)
        assert float(row["dataset_tokens_per_document"]) == pytest.approx(87.5)
        assert int(row["dataset_document_count"]) == 120
        assert int(row["dataset_chunk_count"]) == 180
        assert int(row["dataset_train_samples"]) == 100
        assert int(row["dataset_valid_samples"]) == 10
        assert int(row["dataset_test_samples"]) == 10

def _legacy_row_kwargs(**overrides):
    kwargs = {
        "experiment_id": "exp-legacy-1", "json_filename": "bert_adam_lr1e5_bs8_ep2.json", "seed": 42,
        "device_type": "CPU", "parallel_workers": 2, "train_dataset_name": "train_task2",
        "optimizer": "adamw", "learning_rate": 1e-05, "batch_size": 8, "epoch": 2, "exec_time": 12.5,
        "energy_kwh": 0.002, "emissions_kg": 0.0004, "cost_usd": 0.0003, "avg_ram": 900.5, "peak_ram": 1024.0,
        "avg_gflops_per_batch": 173.9, "total_gflops": 695.9, "status": "success",
        "end_iso": "2026-10-22T10:01:00+00:00", "eval_metrics": {"f1_score": 0.9, "precision": 0.8, "recall": 1.0},
    }
    return {**kwargs, **overrides}


def test_append_csv_row_writes_same_schema_as_append_workflow_csv_rows(monkeypatch, tmp_path):
    legacy_dir, workflow_dir = tmp_path / "legacy", tmp_path / "workflow"
    legacy_dir.mkdir()
    workflow_dir.mkdir()
    monkeypatch.setattr(persistence, "METRICS_DIR", legacy_dir)
    legacy_path = persistence.append_csv_row(**_legacy_row_kwargs())
    monkeypatch.setattr(persistence, "METRICS_DIR", workflow_dir)
    workflow_path = persistence.append_workflow_csv_rows(
        _make_workflow_with_attempts([TaskExecutionAttempt("a1", 1, TaskStatus.SUCCEEDED)]),
        device_type="CPU",
    )

    def header(path):
        with open(path, newline="", encoding="utf-8") as file:
            return next(csv.reader(file))

    assert header(legacy_path) == header(workflow_path)
    assert legacy_path.name == workflow_path.name


def test_append_csv_row_fills_the_shared_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)

    csv_path = persistence.append_csv_row(**_legacy_row_kwargs(start_iso="2026-10-22T10:00:00+00:00"))

    with open(csv_path, newline="", encoding="utf-8") as file:
        (row,) = list(csv.DictReader(file))
    assert row["experiment_run_id"] == "exp-legacy-1"
    assert row["definition_name"] == "bert_adam_lr1e5_bs8_ep2.json"
    assert row["workflow_status"] == "success"
    assert row["task_id"] == "legacy-main-task"
    assert row["attempt_status"] == "succeeded"
    assert row["started_at"] == "2026-10-22T10:00:00+00:00"
    assert row["completed_at"] == "2026-10-22T10:01:00+00:00"
    assert float(row["task_time_sec"]) == pytest.approx(12.5)
    assert float(row["avg_ram_mb"]) == pytest.approx(900.5)
    assert float(row["peak_ram_mb"]) == pytest.approx(1024.0)
    assert float(row["energy_kwh"]) == pytest.approx(0.002)
    assert float(row["emissions_kg_co2"]) == pytest.approx(0.0004)
    assert float(row["cost_usd"]) == pytest.approx(0.0003)
    assert float(row["total_gflops"]) == pytest.approx(695.9)
    assert row["parallel_workers"] == "2"
    assert row["grid_optimizer"] == "adamw"
    assert float(row["grid_learning_rate"]) == pytest.approx(1e-05)
    assert row["grid_batch_size"] == "8"
    assert row["grid_num_epochs"] == "2"
    assert row["grid_seed"] == "42"


def test_append_csv_row_marks_failed_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)

    csv_path = persistence.append_csv_row(**_legacy_row_kwargs(status="failed", eval_metrics={}))

    with open(csv_path, newline="", encoding="utf-8") as file:
        (row,) = list(csv.DictReader(file))
    assert row["workflow_status"] == "failed"
    assert row["attempt_status"] == "failed"


def test_legacy_result_context_only_uses_columns_the_summary_has(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    result = {
        "experiment": {"seed": 7},
        "execution": {"parallel_workers": 1, "train_dataset": "x"},
        "hyperparameters": {"optimizer": "adam", "epoch": 3, "avg_gflops_per_batch": 1.0},
    }

    context = persistence.legacy_result_context(result)
    with caplog.at_level("WARNING", logger=persistence.logger.name):
        persistence.append_workflow_csv_rows(
            _make_workflow_with_attempts([TaskExecutionAttempt("a1", 1, TaskStatus.SUCCEEDED)]),
            device_type="CPU", context=context,
        )

    hardware = context.pop("cpu_count"), context.pop("gpu_count"), context.pop("gpu_model")
    context.pop("disk_total_gb"), context.pop("disk_free_gb") # pyright: ignore[reportUnusedExpression]
    assert hardware[0] >= 1
    assert context == {
        "parallel_workers": 1,
        "grid_params": {"optimizer": "adam", "num_epochs": 3, "seed": 7},
    }
    assert "sem coluna" not in caplog.text


def test_summary_csv_records_hardware_columns_from_context(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = _make_workflow_with_attempts([TaskExecutionAttempt("a1", 1, TaskStatus.SUCCEEDED)])

    csv_path = persistence.append_workflow_csv_rows(
        workflow, device_type="CPU",
        context={
            "cpu_count": 16, "gpu_count": 1, "gpu_model": "NVIDIA GeForce RTX 2050",
            "disk_total_gb": 252.11, "disk_free_gb": 216.82,
        },
    )

    with open(csv_path, newline="", encoding="utf-8") as file:
        (row,) = list(csv.DictReader(file))
    assert row["cpu_count"] == "16" and row["gpu_count"] == "1"
    assert row["gpu_model"] == "NVIDIA GeForce RTX 2050"
    assert float(row["disk_total_gb"]) == pytest.approx(252.11)
    assert float(row["disk_free_gb"]) == pytest.approx(216.82)
    assert "disk_count" not in row


def test_summary_csv_uses_a_separate_file_when_the_existing_header_differs(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = _make_workflow_with_attempts([TaskExecutionAttempt("a1", 1, TaskStatus.SUCCEEDED)])
    first = persistence.append_workflow_csv_rows(workflow, device_type="CPU")
    old_header = "experiment_run_id,task_id,gpu_count\r\n"
    first.write_bytes((old_header + "run-old,train,1\r\n").encode("utf-8"))

    second = persistence.append_workflow_csv_rows(workflow, device_type="CPU")
    third = persistence.append_workflow_csv_rows(workflow, device_type="CPU")

    assert second != first and second.name.startswith(first.stem + "_schema-")
    assert third == second
    assert first.read_bytes() == (old_header + "run-old,train,1\r\n").encode("utf-8")  # arquivo antigo intacto
    with open(second, newline="", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 2


def test_summary_csv_keeps_appending_to_the_same_file_when_the_header_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    workflow = _make_workflow_with_attempts([TaskExecutionAttempt("a1", 1, TaskStatus.SUCCEEDED)])

    first = persistence.append_workflow_csv_rows(workflow, device_type="CPU")
    second = persistence.append_workflow_csv_rows(workflow, device_type="CPU")

    assert first == second
    assert len(list(tmp_path.glob("experiment_summary_*.csv"))) == 1
