"""Coleta de recursos observados para uma tentativa de tarefa."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import psutil
import torch

from .helpers import METRICS_DIR, compute_cost_usd

TrackerFactory = Callable[..., Any]


class TaskTelemetryCollector:
    """Mede tempo, memoria, GPU, energia, emissoes e custo por tentativa."""

    def __init__(
        self,
        *,
        enable_emissions: bool = False,
        environment_cost_per_hour_usd: float | None = None,
        tracker_factory: TrackerFactory | None = None,
        sample_interval_sec: float = 0.1,
        emissions_output_dir: Path | None = None,
    ) -> None:
        self._enable_emissions = enable_emissions
        self._environment_cost_per_hour_usd = environment_cost_per_hour_usd
        self._tracker_factory: TrackerFactory = tracker_factory or _default_tracker
        self._uses_default_tracker = tracker_factory is None
        self._sample_interval_sec = sample_interval_sec
        self._emissions_output_dir = emissions_output_dir or METRICS_DIR

    def measure(
        self,
        task_fn: Callable[[], dict[str, Any] | None],
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, float | None], Exception | None]:
        """Executa a tarefa e retorna sua saida junto das metricas observadas."""
        _synchronize_cuda()
        _reset_peak_cuda()
        process = psutil.Process()
        samples: list[float] = []
        stop_sampling = threading.Event()
        sampler = threading.Thread(
            target=_sample_rss,
            args=(process, samples, stop_sampling, self._sample_interval_sec),
            daemon=True,
        )
        tracker = self._start_tracker(task_id, attempt_id)
        started = time.perf_counter()
        sampler.start()
        output: dict[str, Any] = {}
        error: Exception | None = None
        try:
            output = task_fn() or {}
        except Exception as exc:  # noqa: BLE001
            error = exc
        finally:
            _synchronize_cuda()
            elapsed = time.perf_counter() - started
            stop_sampling.set()
            sampler.join(timeout=2)

        energy_kwh, emissions_kg_co2 = self._stop_tracker(tracker)
        gpu = _gpu_metrics()
        peak_ram_mb = max(samples) if samples else _rss_mb(process)
        metrics = {
            "task_time_sec": elapsed,
            "avg_ram_mb": sum(samples) / len(samples) if samples else peak_ram_mb,
            "peak_ram_mb": peak_ram_mb,
            **gpu,
            "energy_kwh": energy_kwh,
            "emissions_kg_co2": emissions_kg_co2,
            "cost_usd": compute_cost_usd(
                energy_kwh, elapsed, self._environment_cost_per_hour_usd
            ),
        }
        return output, metrics, error

    def _start_tracker(self, task_id: str | None, attempt_id: str | None) -> Any | None:
        if not self._enable_emissions:
            return None
        try:
            tracker = (
                self._tracker_factory(
                    output_dir=self._emissions_output_dir,
                    task_id=task_id,
                    attempt_id=attempt_id,
                )
                if self._uses_default_tracker
                else self._tracker_factory()
            )
            tracker.start()
            return tracker
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _stop_tracker(tracker: Any | None) -> tuple[float | None, float | None]:
        if tracker is None:
            return None, None
        try:
            emissions = tracker.stop()
            data = getattr(tracker, "final_emissions_data", None)
            energy = getattr(data, "energy_consumed", None)
            return _numeric(energy), _numeric(emissions)
        except Exception:  # noqa: BLE001
            return None, None


def _default_tracker(
    *,
    output_dir: Path,
    task_id: str | None,
    attempt_id: str | None,
) -> Any:
    from codecarbon import EmissionsTracker

    safe_task_id = task_id or "task"
    safe_attempt_id = attempt_id or "attempt"
    return EmissionsTracker(
        project_name=f"workflow-{safe_task_id}",
        output_dir=output_dir.as_posix(),
        output_file=f"emissions_{safe_task_id}_{safe_attempt_id}.csv",
        save_to_file=True,
        log_level="error",
    )


def _sample_rss(
    process: psutil.Process,
    samples: list[float],
    stop: threading.Event,
    interval_sec: float,
) -> None:
    while not stop.is_set():
        rss = _rss_mb(process)
        if rss is not None:
            samples.append(rss)
        stop.wait(interval_sec)


def _rss_mb(process: psutil.Process) -> float | None:
    try:
        return process.memory_info().rss / (1024 ** 2)
    except psutil.Error:
        return None


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _gpu_metrics() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"vram_mb": None, "peak_vram_mb": None}
    return {
        "vram_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None