"""Coleta de recursos observados para uma tentativa de tarefa."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import psutil  # type: ignore
import torch
from codecarbon import EmissionsTracker  # type: ignore

from .helpers import METRICS_DIR, compute_cost_usd
from .persistence import append_emissions_row

logger = logging.getLogger(__name__)

TrackerFactory = Callable[..., Any]
_TRACKING_MODES = frozenset({"process", "machine"})

_CUDA_AVAILABLE: bool | None = None
_CUDA_AVAILABLE_LOCK = threading.Lock()


class TaskTelemetryCollector:
    """Mede tempo, memoria, GPU, energia, emissoes e custo por tentativa."""

    def __init__(
        self,
        *,
        enable_emissions: bool = True,
        environment_cost_per_hour_usd: float | None = None,
        tracker_factory: TrackerFactory | None = None,
        sample_interval_sec: float = 0.1,
        emissions_output_dir: Path | None = None,
        prewarm_cuda_probe: bool = True,
        tracking_mode: str = "process",
    ) -> None:
        if tracking_mode not in _TRACKING_MODES:
            raise ValueError(f"tracking_mode deve ser um de {sorted(_TRACKING_MODES)}: {tracking_mode!r}")
        self._enable_emissions = enable_emissions
        self._tracking_mode = tracking_mode
        self._environment_cost_per_hour_usd = environment_cost_per_hour_usd
        self._tracker_factory: TrackerFactory = tracker_factory or _default_tracker
        self._uses_default_tracker = tracker_factory is None
        self._sample_interval_sec = sample_interval_sec
        self._emissions_output_dir = emissions_output_dir or METRICS_DIR
        # Evita inflar a primeira tarefa com custo de inicialização de probe CUDA.
        if prewarm_cuda_probe and _cuda_available():
            _warmup_cuda_runtime()

    def measure(
        self,
        task_fn: Callable[[], dict[str, Any] | None],
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
        experiment_run_id: str | None = None,
        experiment_name: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, float | None], Exception | None]:
        """Executa a tarefa e retorna sua saida junto das metricas observadas."""
        # Mede desde o início da tentativa instrumentada para manter
        # coerência com started_at/completed_at do workflow.
        started = time.perf_counter()
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
        tracker = self._start_tracker(task_id, experiment_run_id)
        sampler.start()
        output: dict[str, Any] = {}
        error: Exception | None = None
        try:
            output = task_fn() or {}
        except Exception as exc:  # noqa: BLE001
            error = exc
        finally:
            _synchronize_cuda()
            stop_sampling.set()
            sampler.join(timeout=2)

        energy_kwh, emissions_kg_co2, water_litter = self._stop_tracker(tracker)
        emissions_row = _label_emissions_row(
            _emissions_row(tracker), experiment_run_id, attempt_id, experiment_name
        )
        gpu = _gpu_metrics()
        elapsed = time.perf_counter() - started
        peak_ram_mb = max(samples) if samples else _rss_mb(process)
        metrics = {
            "task_time_sec": elapsed,
            "avg_ram_mb": sum(samples) / len(samples) if samples else peak_ram_mb,
            "peak_ram_mb": peak_ram_mb,
            **gpu,
            "energy_kwh": energy_kwh,
            "emissions_kg_co2": emissions_kg_co2,
            "water_litter": water_litter,
            "cost_usd": compute_cost_usd(
                energy_kwh, elapsed, self._environment_cost_per_hour_usd
            ),
        }
        # Fora da janela medida: gravar o CSV nem trava o worker nem entra em task_time_sec.
        self._persist_emissions(emissions_row)
        return output, metrics, error

    def _persist_emissions(self, row: Mapping[str, Any] | None) -> None:
        if row is None:
            return
        try:
            append_emissions_row(row, output_dir=self._emissions_output_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Linha do CodeCarbon não gravada no emissions CSV: %s", exc)

    def _start_tracker(self, task_id: str | None, experiment_run_id: str | None) -> Any | None:
        if not self._enable_emissions:
            return None
        try:
            tracker = (
                self._tracker_factory(
                    output_dir=self._emissions_output_dir,
                    task_id=task_id,
                    experiment_run_id=experiment_run_id,
                    tracking_mode=self._tracking_mode,
                )
                if self._uses_default_tracker
                else self._tracker_factory()
            )
            tracker.start()
            return tracker
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _stop_tracker(tracker: Any | None) -> tuple[float | None, float | None, float | None]:
        if tracker is None:
            return None, None, None
        try:
            emissions = tracker.stop()
            data = getattr(tracker, "final_emissions_data", None)
            energy = getattr(data, "energy_consumed", None)
            water = getattr(data, "water_consumed", None)
            if energy is None:
                total_energy = getattr(tracker, "_total_energy", None)
                energy = getattr(total_energy, "kWh", None)
            if water is None:
                total_water = getattr(tracker, "_total_water", None)
                water = getattr(total_water, "litres", None)
            return _numeric(energy), _numeric(emissions), _numeric(water)
        except Exception:  # noqa: BLE001
            return None, None, None


def _emissions_row(tracker: Any | None) -> Mapping[str, Any] | None:
    """Linha padrão (38 colunas) do CodeCarbon; ``None`` se o tracker não a produziu."""
    values = getattr(getattr(tracker, "final_emissions_data", None), "values", None)
    return values if isinstance(values, Mapping) else None


def _label_emissions_row(
    row: Mapping[str, Any] | None,
    experiment_run_id: str | None,
    attempt_id: str | None,
    experiment_name: str | None,
) -> dict[str, Any] | None:
    """Alinha a linha às chaves do experiment_summary; o run_id do CodeCarbon é interno ao tracker."""
    if row is None:
        return None
    labelled: dict[str, Any] = {}
    for key, value in row.items():
        if key == "experiment_id" and experiment_run_id:
            value = experiment_run_id
        elif key == "run_id" and attempt_id:
            value = attempt_id
        labelled[key] = value
        if key == "experiment_id":
            labelled["experiment_name"] = experiment_name
    labelled.setdefault("experiment_name", experiment_name)
    return labelled


def _default_tracker(
    *,
    output_dir: Path,
    task_id: str | None,
    experiment_run_id: str | None,
    tracking_mode: str = "process",
) -> Any:

    safe_task_id = task_id or "task"
    # save_to_file=False: a linha vai ao CSV consolidado (append_emissions_row), com lock entre workers.
    options: dict[str, Any] = {"experiment_id": experiment_run_id} if experiment_run_id else {}
    return EmissionsTracker(
        project_name=f"workflow-{safe_task_id}",
        output_dir=output_dir.as_posix(),
        save_to_file=False,
        log_level="error",
        **options,
        # O CodeCarbon aplica o modo só a CPU e RAM; a GPU segue lendo o dispositivo inteiro (NVML).
        tracking_mode=tracking_mode,
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
    if _cuda_available():
        torch.cuda.synchronize()


def _reset_peak_cuda() -> None:
    if _cuda_available():
        torch.cuda.reset_peak_memory_stats()


def _gpu_metrics() -> dict[str, float | None]:
    if not _cuda_available():
        return {"vram_mb": None, "peak_vram_mb": None}
    return {
        "vram_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    }


def _warmup_cuda_runtime() -> None:
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory_allocated()
        torch.cuda.max_memory_allocated()
    except Exception:  # noqa: BLE001
        # Se o warmup falhar, o coletor continua funcional e faz fallback gracioso.
        return


def _cuda_available() -> bool:
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return _CUDA_AVAILABLE
    with _CUDA_AVAILABLE_LOCK:
        if _CUDA_AVAILABLE is None:
            try:
                _CUDA_AVAILABLE = bool(torch.cuda.is_available())
            except Exception:  # noqa: BLE001
                _CUDA_AVAILABLE = False
    return _CUDA_AVAILABLE


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None