"""Benchmark de consistencia entre experiment_summary e emissions CSV."""

import csv
from pathlib import Path

import pytest

from experiment import persistence
from experiment.task_executor import SequentialWorkflowExecutor
from experiment.task_telemetry import TaskTelemetryCollector
from experiment.workflow import ExperimentDefinition, TaskDefinition


class _BenchmarkTracker:
    def __init__(self):
        self.final_emissions_data = type(
            "Data",
            (),
            {
                "energy_consumed": 0.25,
                "values": {
                    "timestamp": "2026-10-25T10:00:00",
                    "project_name": "workflow-train",
                    "run_id": "tracker-run",
                    "experiment_id": "tracker-exp",
                    "duration": 0.0,
                    "emissions": 0.05,
                    "energy_consumed": 0.25,
                    "tracking_mode": "process",
                },
            },
        )()

    def start(self):
        return None

    def stop(self):
        return 0.05


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _read_emissions_rows(metrics_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(metrics_dir.glob("emissions_*.csv")):
        rows.extend(_read_csv_rows(path))
    return rows


def test_benchmark_matches_keys_and_energy_between_csvs(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    telemetry = TaskTelemetryCollector(
        enable_emissions=True,
        tracker_factory=_BenchmarkTracker,
        emissions_output_dir=tmp_path,
        prewarm_cuda_probe=False,
    )
    definition = ExperimentDefinition("benchmark-workflow", (TaskDefinition("train", "Treinar"),))
    workflow = SequentialWorkflowExecutor({"train": dict}, telemetry=telemetry).execute(definition)

    summary_path = persistence.append_workflow_csv_rows(workflow, device_type="CPU", definition=definition)
    (summary_row,) = _read_csv_rows(summary_path)
    (emissions_row,) = _read_emissions_rows(tmp_path)

    assert summary_row["experiment_run_id"] == emissions_row["experiment_id"]
    assert summary_row["attempt_id"] == emissions_row["run_id"]
    assert summary_row["definition_name"] == emissions_row["experiment_name"]
    assert float(summary_row["energy_kwh"]) == pytest.approx(float(emissions_row["energy_consumed"]))
    assert float(summary_row["emissions_kg_co2"]) == pytest.approx(float(emissions_row["emissions"]))


def test_benchmark_has_one_emissions_row_per_summary_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    telemetry = TaskTelemetryCollector(
        enable_emissions=True,
        tracker_factory=_BenchmarkTracker,
        emissions_output_dir=tmp_path,
        prewarm_cuda_probe=False,
    )
    definition = ExperimentDefinition(
        "benchmark-workflow",
        (TaskDefinition("prepare", "Preparar"), TaskDefinition("train", "Treinar")),
    )
    workflow = SequentialWorkflowExecutor(
        {"prepare": dict, "train": dict},
        telemetry=telemetry,
    ).execute(definition)

    summary_path = persistence.append_workflow_csv_rows(workflow, device_type="CPU", definition=definition)
    summary_rows = _read_csv_rows(summary_path)
    emissions_rows = _read_emissions_rows(tmp_path)

    assert len(summary_rows) == len(emissions_rows) == 2
    summary_keys = {(row["experiment_run_id"], row["attempt_id"]) for row in summary_rows}
    emissions_keys = {(row["experiment_id"], row["run_id"]) for row in emissions_rows}
    assert summary_keys == emissions_keys
