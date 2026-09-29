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
    assert json.loads(rows[0]["experiment_context_json"]) == {
        "environment": {"precision": "fp32"},
    }