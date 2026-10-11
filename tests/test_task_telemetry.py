"""Testes da instrumentacao de telemetria por tentativa."""
import time

import pytest

import experiment.task_telemetry as telemetry_module
from experiment.helpers import ENERGY_COST_USD_PER_KWH
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


class _SlowTracker(_Tracker):
    def start(self):
        time.sleep(0.05)
        super().start()

    def stop(self):
        time.sleep(0.05)
        return super().stop()


class _BrokenTracker:
    def start(self):
        raise RuntimeError("tracker indisponivel")

    def stop(self):
        raise RuntimeError("nao deveria ser chamado")


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


def test_task_time_includes_tracker_start_and_stop_overhead():
    tracker = _SlowTracker()
    telemetry = TaskTelemetryCollector(enable_emissions=True, tracker_factory=lambda: tracker)
    definition = ExperimentDefinition("workflow", (TaskDefinition("train", "Treinar"),))

    workflow = SequentialWorkflowExecutor({"train": dict}, telemetry=telemetry).execute(definition)

    attempt = workflow.tasks[0].attempts[0]
    # 0.05s (start) + 0.05s (stop) + overhead pequeno da execução.
    assert attempt.metrics["resources"]["task_time_sec"] >= 0.10


def test_task_time_includes_initial_cuda_sync_overhead(monkeypatch):
    telemetry = TaskTelemetryCollector(enable_emissions=False)
    definition = ExperimentDefinition("workflow", (TaskDefinition("train", "Treinar"),))

    monkeypatch.setattr(
        "experiment.task_telemetry._synchronize_cuda",
        lambda: time.sleep(0.05),
    )

    workflow = SequentialWorkflowExecutor({"train": dict}, telemetry=telemetry).execute(definition)
    attempt = workflow.tasks[0].attempts[0]
    assert attempt.metrics["resources"]["task_time_sec"] >= 0.05


def test_telemetry_without_emissions_keeps_energy_fields_as_none():
    telemetry = TaskTelemetryCollector(enable_emissions=False)
    definition = ExperimentDefinition("workflow", (TaskDefinition("train", "Treinar"),))

    workflow = SequentialWorkflowExecutor({"train": dict}, telemetry=telemetry).execute(definition)
    resources = workflow.tasks[0].attempts[0].metrics["resources"]

    assert resources["energy_kwh"] is None
    assert resources["emissions_kg_co2"] is None
    assert resources["cost_usd"] is None


def test_telemetry_tolerates_tracker_start_failure():
    telemetry = TaskTelemetryCollector(enable_emissions=True, tracker_factory=lambda: _BrokenTracker())
    definition = ExperimentDefinition("workflow", (TaskDefinition("train", "Treinar"),))

    workflow = SequentialWorkflowExecutor({"train": dict}, telemetry=telemetry).execute(definition)
    attempt = workflow.tasks[0].attempts[0]
    resources = attempt.metrics["resources"]

    assert attempt.status is TaskStatus.SUCCEEDED
    assert resources["energy_kwh"] is None
    assert resources["emissions_kg_co2"] is None
    assert resources["cost_usd"] is None


def test_metrics_are_isolated_between_tasks():
    telemetry = TaskTelemetryCollector(enable_emissions=False)
    definition = ExperimentDefinition(
        "workflow",
        (
            TaskDefinition("task_a", "Task A"),
            TaskDefinition("task_b", "Task B"),
        ),
    )
    executor = SequentialWorkflowExecutor(
        {
            "task_a": lambda: {"metrics": {"resources": {"total_gflops": 1.0}}},
            "task_b": lambda: {"metrics": {"resources": {"total_gflops": 2.0}}},
        },
        telemetry=telemetry,
    )

    workflow = executor.execute(definition)
    resources_a = workflow.tasks[0].attempts[0].metrics["resources"]
    resources_b = workflow.tasks[1].attempts[0].metrics["resources"]

    assert resources_a["total_gflops"] == 1.0
    assert resources_b["total_gflops"] == 2.0
    assert resources_a is not resources_b


def test_ram_and_vram_metrics_are_coherent():
    tracker = _Tracker()
    telemetry = TaskTelemetryCollector(enable_emissions=True, tracker_factory=lambda: tracker)
    definition = ExperimentDefinition("workflow", (TaskDefinition("train", "Treinar"),))

    workflow = SequentialWorkflowExecutor({"train": dict}, telemetry=telemetry).execute(definition)
    resources = workflow.tasks[0].attempts[0].metrics["resources"]

    assert resources["avg_ram_mb"] is not None
    assert resources["peak_ram_mb"] is not None
    assert resources["peak_ram_mb"] >= resources["avg_ram_mb"]

    if resources["vram_mb"] is not None and resources["peak_vram_mb"] is not None:
        assert resources["peak_vram_mb"] >= resources["vram_mb"]


def test_cost_uses_energy_tariff_when_hourly_rate_is_not_provided():
    tracker = _Tracker()
    telemetry = TaskTelemetryCollector(enable_emissions=True, tracker_factory=lambda: tracker)
    definition = ExperimentDefinition("workflow", (TaskDefinition("train", "Treinar"),))

    workflow = SequentialWorkflowExecutor({"train": dict}, telemetry=telemetry).execute(definition)
    resources = workflow.tasks[0].attempts[0].metrics["resources"]

    assert resources["energy_kwh"] == 0.25
    assert resources["cost_usd"] == pytest.approx(0.25 * ENERGY_COST_USD_PER_KWH)


def test_cuda_probe_is_cached_and_prewarmed_outside_measure(monkeypatch):
    monkeypatch.setattr(telemetry_module, "_CUDA_AVAILABLE", None)
    calls = {"count": 0}

    def _fake_is_available():
        calls["count"] += 1
        time.sleep(0.05)
        return False

    monkeypatch.setattr("experiment.task_telemetry.torch.cuda.is_available", _fake_is_available)

    telemetry = TaskTelemetryCollector(enable_emissions=False)
    assert calls["count"] == 1

    telemetry.measure(dict)
    telemetry.measure(dict)
    assert calls["count"] == 1