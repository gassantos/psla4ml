"""Testes de persistência dos manifestos P1 de workflow."""

import csv
import json

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
        context={"environment": {"precision": "fp32"}},
    )

    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["task_id"] == "train"
    assert rows[0]["activity"] == "adaptation"
    assert rows[0]["task_time_sec"] == "12.5"
    assert json.loads(rows[0]["evaluation_json"]) == {"f1_score": 0.9}
    assert rows[0]["root_error_type"] == ""
    assert json.loads(rows[0]["experiment_context_json"]) == {
        "environment": {"precision": "fp32"},
    }


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