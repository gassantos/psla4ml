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
        self.final_emissions_data = type(
            "Data", (), {"energy_consumed": 0.25, "water_consumed": 1.5}
        )()

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


class _FallbackDataTracker(_Tracker):
    def __init__(self):
        super().__init__()
        self.final_emissions_data = type("Data", (), {"energy_consumed": None, "water_consumed": None})()
        self._total_energy = type("Energy", (), {"kWh": 0.125})()
        self._total_water = type("Water", (), {"litres": 0.75})()


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
    assert resources["water_litter"] == 1.5
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
    assert resources["water_litter"] is None
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
    assert resources["water_litter"] is None
    assert resources["cost_usd"] is None


def test_telemetry_uses_tracker_fallback_energy_and_water_when_final_data_is_empty():
    tracker = _FallbackDataTracker()
    telemetry = TaskTelemetryCollector(enable_emissions=True, tracker_factory=lambda: tracker)
    definition = ExperimentDefinition("workflow", (TaskDefinition("train", "Treinar"),))

    workflow = SequentialWorkflowExecutor({"train": dict}, telemetry=telemetry).execute(definition)
    resources = workflow.tasks[0].attempts[0].metrics["resources"]

    assert resources["energy_kwh"] == pytest.approx(0.125)
    assert resources["water_litter"] == pytest.approx(0.75)


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

class _FakeEmissionsTracker(_Tracker):
    created: list = []

    def __init__(self, **kwargs):
        super().__init__()
        type(self).created.append(kwargs)


def _measure_with_default_tracker(monkeypatch, tmp_path, **collector_kwargs):
    _FakeEmissionsTracker.created = []
    monkeypatch.setattr(telemetry_module, "EmissionsTracker", _FakeEmissionsTracker)
    collector = TaskTelemetryCollector(
        enable_emissions=True, emissions_output_dir=tmp_path, prewarm_cuda_probe=False,
        **collector_kwargs,
    )
    collector.measure(dict, task_id="train", attempt_id="a1", experiment_run_id="r1")
    return _FakeEmissionsTracker.created


def test_default_tracker_isolates_energy_by_process(monkeypatch, tmp_path):
    (kwargs,) = _measure_with_default_tracker(monkeypatch, tmp_path)

    assert kwargs["tracking_mode"] == "process"
    # A linha vai ao CSV consolidado, não a um arquivo por tentativa.
    assert kwargs["save_to_file"] is False
    assert "output_file" not in kwargs
    assert kwargs["experiment_id"] == "r1"


class _RowTracker(_Tracker):
    def __init__(self, experiment_id="tracker-experiment"):
        super().__init__()
        self.final_emissions_data = type(
            "Data", (), {
                "energy_consumed": 0.25,
                "values": {
                    "timestamp": "t", "project_name": "workflow-train", "run_id": "tracker-run",
                    "experiment_id": experiment_id, "energy_consumed": 0.25, "tracking_mode": "process",
                },
            },
        )()


def _read_emissions_rows(directory):
    import csv

    rows = []
    for path in sorted(directory.glob("emissions_*.csv")):
        with open(path, newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
    return rows


def test_emissions_row_carries_summary_keys(tmp_path):
    collector = TaskTelemetryCollector(
        enable_emissions=True, emissions_output_dir=tmp_path,
        tracker_factory=lambda: _RowTracker(), prewarm_cuda_probe=False,
    )
    collector.measure(dict, task_id="train", attempt_id="a1", experiment_run_id="r1", experiment_name="def")
    collector.measure(dict, task_id="train", attempt_id="a2", experiment_run_id="r1", experiment_name="def")

    rows = _read_emissions_rows(tmp_path)
    assert [(r["experiment_id"], r["run_id"], r["experiment_name"]) for r in rows] == [
        ("r1", "a1", "def"), ("r1", "a2", "def"),
    ]
    assert rows[0]["tracking_mode"] == "process"
    assert list(rows[0])[:5] == ["timestamp", "project_name", "run_id", "experiment_id", "experiment_name"]
    assert not list(tmp_path.glob("emissions_train_*.csv"))


def test_emissions_row_keys_match_experiment_summary(tmp_path, isolated_metrics_dir):
    collector = TaskTelemetryCollector(
        enable_emissions=True, emissions_output_dir=tmp_path,
        tracker_factory=lambda: _RowTracker(), prewarm_cuda_probe=False,
    )
    definition = ExperimentDefinition("workflow-x", (TaskDefinition("train", "Treinar"),))
    run = SequentialWorkflowExecutor({"train": dict}, telemetry=collector).execute(definition)

    (row,) = _read_emissions_rows(tmp_path)
    assert row["experiment_id"] == run.experiment_run_id
    assert row["run_id"] == run.tasks[0].attempts[0].attempt_id
    assert row["experiment_name"] == run.definition_name == "workflow-x"


def test_emissions_header_change_goes_to_schema_file(tmp_path):
    from experiment.persistence import append_emissions_row

    first = append_emissions_row({"experiment_id": "a", "energy_consumed": 1}, output_dir=tmp_path)
    second = append_emissions_row({"experiment_id": "b", "energy_consumed": 2, "gpu_count": 1}, output_dir=tmp_path)
    again = append_emissions_row({"experiment_id": "c", "energy_consumed": 3}, output_dir=tmp_path)

    assert second != first and "_schema-" in second.name
    assert again == first
    assert len(_read_emissions_rows(tmp_path)) == 3


def test_concurrent_emissions_writers_lose_no_rows(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from experiment.persistence import append_emissions_row

    def _write(index):
        append_emissions_row({"experiment_id": f"a{index}", "energy_consumed": index}, output_dir=tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_write, range(40)))

    rows = _read_emissions_rows(tmp_path)
    assert sorted(row["experiment_id"] for row in rows) == sorted(f"a{i}" for i in range(40))


def test_emissions_write_failure_does_not_lose_measured_energy(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise OSError("disco cheio")

    monkeypatch.setattr(telemetry_module, "append_emissions_row", _boom)
    collector = TaskTelemetryCollector(
        enable_emissions=True, emissions_output_dir=tmp_path,
        tracker_factory=lambda: _RowTracker(), prewarm_cuda_probe=False,
    )

    _, metrics, error = collector.measure(dict, task_id="train", attempt_id="a1")

    assert error is None
    assert metrics["energy_kwh"] == 0.25


def test_tracker_without_row_does_not_create_emissions_csv(tmp_path):
    collector = TaskTelemetryCollector(
        enable_emissions=True, emissions_output_dir=tmp_path,
        tracker_factory=lambda: _Tracker(), prewarm_cuda_probe=False,
    )
    collector.measure(dict, task_id="train", attempt_id="a1")

    assert not list(tmp_path.glob("emissions_*.csv"))



def test_tracking_mode_machine_can_be_selected(monkeypatch, tmp_path):
    (kwargs,) = _measure_with_default_tracker(monkeypatch, tmp_path, tracking_mode="machine")

    assert kwargs["tracking_mode"] == "machine"


def test_tracking_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="tracking_mode"):
        TaskTelemetryCollector(enable_emissions=False, tracking_mode="gpu")
