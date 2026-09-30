"""Testes da instrumentacao de telemetria por tentativa."""

import pytest

from experiment.task_executor import SequentialWorkflowExecutor
from experiment.task_telemetry import TaskTelemetryCollector
from experiment.workflow import ExperimentDefinition, TaskDefinition, TaskStatus


class _Tracker:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.final_emissions_data = type("Data", (), {"energy_consumed": 0.25})()

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True
        return 0.05


def test_telemetry_records_standard_resources_and_task_provided_flops():
    tracker = _Tracker()
    telemetry = TaskTelemetryCollector(
        enable_emissions=True, environment_cost_per_hour_usd=3.6, tracker_factory=lambda: tracker
    )
    definition = ExperimentDefinition("workflow", (TaskDefinition("train", "Treinar"),))
    executor = SequentialWorkflowExecutor(
        {"train": lambda: {"metrics": {"resources": {"total_gflops": 12.5}}}}, telemetry=telemetry
    )

    attempt = executor.execute(definition).tasks[0].attempts[0]

    resources = attempt.metrics["resources"]
    assert attempt.status is TaskStatus.SUCCEEDED
    assert tracker.started and tracker.stopped
    assert resources["task_time_sec"] >= 0
    assert resources["peak_ram_mb"] is not None
    assert resources["energy_kwh"] == 0.25
    assert resources["emissions_kg_co2"] == 0.05
    assert resources["cost_usd"] == pytest.approx(resources["task_time_sec"] / 1000)
    assert resources["total_gflops"] == 12.5


def test_telemetry_is_recorded_when_task_fails():
    tracker = _Tracker()
    telemetry = TaskTelemetryCollector(enable_emissions=True, tracker_factory=lambda: tracker)
    definition = ExperimentDefinition("workflow", (TaskDefinition("train", "Treinar"),))

    workflow = SequentialWorkflowExecutor(
        {"train": lambda: (_ for _ in ()).throw(TimeoutError("indisponivel"))}, telemetry=telemetry
    ).execute(definition)

    attempt = workflow.tasks[0].attempts[0]
    assert attempt.status is TaskStatus.FAILED
    assert attempt.error_type == "InfraError"
    assert attempt.root_error_type == "TimeoutError"
    assert tracker.stopped
    assert attempt.metrics["resources"]["energy_kwh"] == 0.25