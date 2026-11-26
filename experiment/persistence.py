"""
Persistência de resultados de experimentos
============================================

Construção do dicionário JSON e escrita em CSV acumulado.

Autor: Gustavo Alexandre
"""

import csv
import hashlib
import json
import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .helpers import METRICS_DIR, hardware_info
from .workflow import (
    ExperimentDefinition,
    ExperimentRun,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
    legacy_task_run,
)

logger = logging.getLogger(__name__)

_EMISSIONS_LOCK_TIMEOUT_SEC = 30.0

# Contexto de execução em colunas; grid_* espelha as chaves de gridsearch/config/param_mapping.json.
CONTEXT_COLUMNS = (
    "parallel_workers", "selected_environment",
    "configured_environment", "resource_machine_name",
    "cpu_count", "gpu_count", "gpu_model", "disk_total_gb", "disk_free_gb",
)
GRID_PARAM_NAMES = (
    "learning_rate", "batch_size", "optimizer", "dropout",
    "seed", "environment", "num_epochs", "max_seq_length",
)
GRID_PARAM_COLUMNS = tuple(f"grid_{name}" for name in GRID_PARAM_NAMES)

_HOST_DEVICE_TYPE_CACHE: str | None = None


def _host_device_type() -> str:
    """Retorna o tipo de recurso detectado no host (CPU/GPU/TPU)."""
    global _HOST_DEVICE_TYPE_CACHE
    if _HOST_DEVICE_TYPE_CACHE is not None:
        return _HOST_DEVICE_TYPE_CACHE
    try:
        from utils.device import get_torch_device

        detected = str(get_torch_device().get("type", "")).upper()
        if detected in {"CPU", "GPU", "TPU"}:
            _HOST_DEVICE_TYPE_CACHE = detected
            return detected
    except Exception:
        logger.warning("Falha ao detectar o tipo de dispositivo do host.", exc_info=True)
    _HOST_DEVICE_TYPE_CACHE = "CPU"
    return _HOST_DEVICE_TYPE_CACHE


def _resolve_summary_device_type(requested: str) -> str:
    """Resolve o tipo do CSV de sumário conforme recurso priorizado na execução.

    Regras:
    - usa ``requested`` quando for ``CPU/GPU/TPU`` e compatível com o host;
    - em inconsistências, faz fallback para o device detectado no host.
    """
    requested_norm = str(requested).upper()
    host = _host_device_type()
    if requested_norm == "CPU":
        return "CPU"
    if requested_norm in {"GPU", "TPU"}:
        return requested_norm if requested_norm == host else host
    return host


def build_result_dict(
    *,
    experiment_id: str,
    json_filename: str,
    seed: int,
    status: str,
    date_exec: str,
    start_iso: str,
    end_iso: str,
    device_type: str,
    device_name: str,
    precision: str,
    parallel_workers: int,
    train_dataset_name: str,
    optimizer: str,
    learning_rate: float,
    avg_gflops_per_batch: float,
    batch_size: int,
    epoch: int,
    exec_time: float,
    energy_kwh: float | None,
    emissions_kg: float | None,
    cost_usd: float | None,
    avg_ram: float | None,
    peak_ram: float | None,
    total_gflops: float,
    eval_metrics: dict[str, Any],
    stdout: str,
    stderr: str,
    water_litter: float | None = None,
    tpu_check: Any = None,
    xla_world_size: int = 1,
    failure_stage: str | None = None,
) -> dict[str, Any]:
    """Constrói o dicionário padronizado de resultado de um experimento."""
    result: dict[str, Any] = {
        "experiment": {
            "id": experiment_id,
            "config_name": json_filename,
            "seed": seed,
            "status": status,
            "date": date_exec,
            "timestamp_start": start_iso,
            "timestamp_end": end_iso,
        },
        "environment": {
            "device_type": device_type,
            "device_name": device_name,
            "precision": precision,
        },
        "execution": {
            "parallel_workers": parallel_workers,
            "xla_world_size": xla_world_size,
            "train_dataset": train_dataset_name,
        },
        "hyperparameters": {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "avg_gflops_per_batch": avg_gflops_per_batch,
            "batch_size": batch_size,
            "epoch": epoch,
        },
        "resources": {
            "train_time_sec": f"{exec_time:.2f}",
            "energy_kwh": energy_kwh,
            "emissions_kg_co2": emissions_kg,
            "cost_usd": cost_usd,
            "water_litter": water_litter,
            "avg_ram_mb": avg_ram,
            "peak_ram_mb": peak_ram,
            "total_gflops": total_gflops,
        },
        "evaluation": eval_metrics if eval_metrics else None,
        "logs": {
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-1000:],
            "failure_stage": failure_stage,
        },
    }

    # BL-08: inclui status de ativação do TPU quando disponível
    if tpu_check is not None:
        tpu_dict = (
            tpu_check.to_dict()
            if hasattr(tpu_check, "to_dict")
            else dict(tpu_check)
        )
        result["tpu_acceleration_check"] = tpu_dict
        if tpu_dict.get("warning"):
            result["warnings"] = result.get("warnings", []) + [tpu_dict["warning"]]

    return result


def write_json_result(result: dict[str, Any], json_filename: str) -> Path:
    """Escreve o dicionário de resultado em arquivo JSON."""
    json_path = METRICS_DIR / json_filename
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    return json_path


def write_workflow_run(workflow: ExperimentRun) -> Path:
    """Persiste o manifesto e cada tarefa de uma execução de workflow."""
    run_dir = METRICS_DIR / "workflow_runs" / workflow.experiment_run_id
    tasks_dir = run_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    manifest = workflow.to_dict()
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    for task in manifest["tasks"]:
        with open(tasks_dir / f"{task['task_id']}.json", "w", encoding="utf-8") as f:
            json.dump(task, f, indent=2)

    return run_dir


def append_workflow_csv_rows(
    workflow: ExperimentRun,
    *,
    device_type: str,
    definition: ExperimentDefinition | None = None,
    context: dict[str, Any] | None = None,
) -> Path:
    """Acrescenta uma linha por tentativa ao sumário CSV de experimentos."""
    safe_context = dict(context or {})
    safe_context.pop("environment", None)
    safe_context.pop("requested_environment", None)
    grid_params = safe_context.pop("grid_params", None)
    grid_params = grid_params if isinstance(grid_params, dict) else {}
    unknown = (set(safe_context) - set(CONTEXT_COLUMNS)) | {
        f"grid_params.{name}" for name in set(grid_params) - set(GRID_PARAM_NAMES)
    }
    if unknown:
        logger.warning("Contexto sem coluna no experiment_summary, ignorado: %s", sorted(unknown))
    context_values = {
        **{column: safe_context.get(column) for column in CONTEXT_COLUMNS},
        **{f"grid_{name}": grid_params.get(name) for name in GRID_PARAM_NAMES},
    }
    headers = [
        "experiment_run_id", "definition_name", "workflow_status",
        "task_id", "task_name", "task_type", "activity", "regime", "task_status",
        "attempt_id", "attempt_number", "attempt_status", "started_at", "completed_at",
        "task_time_sec", "avg_ram_mb", "peak_ram_mb", "vram_mb", "peak_vram_mb",
        "energy_kwh", "emissions_kg_co2", "cost_usd", "water_litter",
        "dataset_size_mb", "dataset_tokens_per_document", "dataset_document_count",
        "dataset_chunk_count", "dataset_train_samples", "dataset_valid_samples",
        "dataset_test_samples",
        "total_gflops",
        # "resources_json", "evaluation_json", "artifacts_json",
        "error_type", "root_error_type", "failure_stage", "error",
        "is_infra_error", "attempt_count", "is_final_attempt",
        *CONTEXT_COLUMNS, *GRID_PARAM_COLUMNS,
    ]
    csv_path = _workflow_summary_csv_path(device_type, headers)
    write_header = not csv_path.exists()
    definitions = {task.task_id: task for task in definition.tasks} if definition else {}
    dataset_summary = _extract_dataset_summary_from_workflow(workflow)

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if write_header:
            writer.writeheader()
        for task in workflow.tasks:
            task_definition = definitions.get(task.task_id)
            attempt_count = len(task.attempts)
            for attempt in task.attempts:
                metrics = attempt.metrics if isinstance(attempt.metrics, dict) else {}
                resources = metrics.get("resources", {})
                resources = resources if isinstance(resources, dict) else {}
                writer.writerow({
                    "experiment_run_id": workflow.experiment_run_id,
                    "definition_name": workflow.definition_name,
                    "workflow_status": workflow.status,
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "task_type": task.task_type,
                    "activity": task_definition.activity.value if task_definition else None,
                    "regime": task_definition.regime.value if task_definition else None,
                    "task_status": task.status.value,
                    "attempt_id": attempt.attempt_id,
                    "attempt_number": attempt.attempt_number,
                    "attempt_status": attempt.status.value,
                    "started_at": attempt.started_at,
                    "completed_at": attempt.completed_at,
                    "task_time_sec": resources.get("task_time_sec"),
                    "avg_ram_mb": resources.get("avg_ram_mb"),
                    "peak_ram_mb": resources.get("peak_ram_mb"),
                    "vram_mb": resources.get("vram_mb"),
                    "peak_vram_mb": resources.get("peak_vram_mb"),
                    "energy_kwh": resources.get("energy_kwh"),
                    "emissions_kg_co2": resources.get("emissions_kg_co2"),
                    "water_litter": resources.get("water_litter"),
                    "cost_usd": resources.get("cost_usd"),
                    "dataset_size_mb": dataset_summary.get("dataset_size_mb"),
                    "dataset_tokens_per_document": dataset_summary.get("dataset_tokens_per_document"),
                    "dataset_document_count": dataset_summary.get("dataset_document_count"),
                    "dataset_chunk_count": dataset_summary.get("dataset_chunk_count"),
                    "dataset_train_samples": dataset_summary.get("dataset_train_samples"),
                    "dataset_valid_samples": dataset_summary.get("dataset_valid_samples"),
                    "dataset_test_samples": dataset_summary.get("dataset_test_samples"),
                    "total_gflops": resources.get("total_gflops"),
                    # "resources_json": json.dumps(resources, ensure_ascii=False, sort_keys=True),
                    # "evaluation_json": json.dumps(metrics.get("evaluation", {}), ensure_ascii=False, sort_keys=True),
                    # "artifacts_json": json.dumps(attempt.artifacts, ensure_ascii=False, sort_keys=True),
                    "error_type": attempt.error_type,
                    "root_error_type": attempt.root_error_type,
                    "failure_stage": attempt.failure_stage,
                    "error": attempt.error,
                    "is_infra_error": attempt.error_type == "InfraError" if attempt.error_type else False,
                    "attempt_count": attempt_count,
                    "is_final_attempt": attempt.attempt_number == attempt_count,
                    **context_values,
                })
    return csv_path


def legacy_result_context(result: dict[str, Any]) -> dict[str, Any]:
    """Contexto do sumário a partir do dicionário de resultado legado (``build_result_dict``)."""
    hyperparameters = result.get("hyperparameters", {})
    params = {
        "optimizer": hyperparameters.get("optimizer"),
        "learning_rate": hyperparameters.get("learning_rate"),
        "batch_size": hyperparameters.get("batch_size"),
        "num_epochs": hyperparameters.get("epoch"),
        "seed": result.get("experiment", {}).get("seed"),
    }
    return {
        **hardware_info(),
        "parallel_workers": result.get("execution", {}).get("parallel_workers"),
        "grid_params": {name: value for name, value in params.items() if value is not None},
    }


def _extract_dataset_summary_from_workflow(workflow: ExperimentRun) -> dict[str, Any]:
    """Extrai estatísticas do dataset a partir da tarefa de ingestão do workflow."""
    defaults: dict[str, Any] = {
        "dataset_size_mb": None,
        "dataset_tokens_per_document": None,
        "dataset_document_count": None,
        "dataset_chunk_count": None,
        "dataset_train_samples": None,
        "dataset_valid_samples": None,
        "dataset_test_samples": None,
    }
    ingest_candidates = [
        task
        for task in workflow.tasks
        if task.task_id in {"ingest_dataset", "ingest_data", "ingest"}
    ]
    if not ingest_candidates:
        return defaults

    for task in ingest_candidates:
        for attempt in task.attempts:
            metrics = attempt.metrics if isinstance(attempt.metrics, dict) else {}
            dataset = metrics.get("dataset") if isinstance(metrics, dict) else None
            if not isinstance(dataset, dict):
                continue
            summary = dict(defaults)
            summary["dataset_size_mb"] = dataset.get("dataset_size_mb")
            summary["dataset_tokens_per_document"] = dataset.get("tokens_per_document")
            summary["dataset_document_count"] = dataset.get("document_count")
            summary["dataset_chunk_count"] = dataset.get("chunk_count")
            summary["dataset_train_samples"] = dataset.get("train_samples")
            summary["dataset_valid_samples"] = dataset.get("valid_samples")
            summary["dataset_test_samples"] = dataset.get("test_samples")
            return summary
    return defaults


def _workflow_summary_csv_path(device_type: str, headers: list[str]) -> Path:
    """Arquivo do dia; se o cabeçalho existente for outro, usa um arquivo próprio do esquema."""
    device_type = _resolve_summary_device_type(device_type)
    suffix = datetime.now().astimezone().strftime("%Y%m%d")
    return _schema_csv_path(METRICS_DIR, f"experiment_summary_{device_type}-{suffix}", headers)


def _schema_csv_path(directory: Path, stem: str, headers: list[str]) -> Path:
    """``<stem>.csv`` ou, se o cabeçalho existente diferir, ``<stem>_schema-<hash>.csv``.

    Anexar a um CSV com cabeçalho diferente desalinharia as colunas sem erro.
    """
    csv_path = directory / f"{stem}.csv"
    if _csv_header(csv_path) in (None, headers):
        return csv_path
    schema = hashlib.sha1(",".join(headers).encode("utf-8")).hexdigest()[:8]
    return directory / f"{stem}_schema-{schema}.csv"


def append_emissions_row(
    values: Mapping[str, Any],
    *,
    output_dir: Path | None = None,
    device_type: str = "",
) -> Path:
    """Acrescenta a linha padrão do CodeCarbon ao ``emissions_<DEVICE>-<DATA>.csv`` do dia.

    ``experiment_id`` leva o ``attempt_id`` do workflow, a chave de junção com o
    ``experiment_summary``. A gravação é serializada entre processos porque o
    CodeCarbon não trava o arquivo e workers paralelos perderiam linhas.
    """
    from filelock import FileLock

    directory = output_dir or METRICS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    headers = list(values)
    day = datetime.now().astimezone().strftime("%Y%m%d")
    stem = f"emissions_{_resolve_summary_device_type(device_type)}-{day}"
    with FileLock(str(directory / f".{stem}.lock"), timeout=_EMISSIONS_LOCK_TIMEOUT_SEC):
        csv_path = _schema_csv_path(directory, stem, headers)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerow(values)
    return csv_path


def _csv_header(path: Path) -> list[str] | None:
    """Cabeçalho do CSV, ou ``None`` se o arquivo ainda não existe."""
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as file:
        return next(csv.reader(file), [])


def load_workflow_run(run_dir: Path) -> ExperimentRun:
    """Carrega um manifesto persistido para retomada seletiva do workflow."""
    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    tasks = [
        TaskRun(
            task_id=task["task_id"],
            name=task["name"],
            task_type=task["task_type"],
            status=TaskStatus(task["status"]),
            attempts=[
                TaskExecutionAttempt(
                    attempt_id=attempt["attempt_id"],
                    attempt_number=attempt["attempt_number"],
                    status=TaskStatus(attempt["status"]),
                    started_at=attempt.get("started_at"),
                    completed_at=attempt.get("completed_at"),
                    metrics=attempt.get("metrics", {}),
                    artifacts=attempt.get("artifacts", {}),
                    error=attempt.get("error"),
                    error_type=attempt.get("error_type"),
                )
                for attempt in task.get("attempts", [])
            ],
            config=task.get("config", {}),
            input_signatures=task.get("input_signatures", {}),
        )
        for task in manifest["tasks"]
    ]
    return ExperimentRun(
        experiment_run_id=manifest["experiment_run_id"],
        definition_name=manifest["definition_name"],
        status=manifest["status"],
        tasks=tasks,
        schema_version=manifest.get("schema_version", "1.0"),
    )


def append_csv_row(
    *,
    experiment_id: str,
    json_filename: str,
    seed: int,
    device_type: str,
    parallel_workers: int,
    train_dataset_name: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    epoch: int,
    exec_time: float,
    energy_kwh: float | None,
    emissions_kg: float | None,
    cost_usd: float | None,
    avg_ram: float | None,
    peak_ram: float | None,
    avg_gflops_per_batch: float,
    total_gflops: float,
    status: str,
    end_iso: str,
    eval_metrics: dict[str, Any],
    start_iso: str | None = None,
    water_litter: float | None = None,
) -> Path:
    """Acrescenta ao sumário uma linha no mesmo esquema de ``append_workflow_csv_rows``.

    O experimento legado entra como workflow de uma tarefa implícita
    (:func:`legacy_task_run`), igual ao caminho de ``experiment.runner``.
    """
    result = {
        "experiment": {
            "id": experiment_id,
            "config_name": json_filename,
            "seed": seed,
            "status": status,
            "timestamp_start": start_iso,
            "timestamp_end": end_iso,
        },
        "execution": {"parallel_workers": parallel_workers, "train_dataset": train_dataset_name},
        "hyperparameters": {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "avg_gflops_per_batch": avg_gflops_per_batch,
            "batch_size": batch_size,
            "epoch": epoch,
        },
        "resources": {
            "train_time_sec": f"{exec_time:.2f}",
            "energy_kwh": energy_kwh,
            "emissions_kg_co2": emissions_kg,
            "water_litter": water_litter,
            "cost_usd": cost_usd,
            "avg_ram_mb": avg_ram,
            "peak_ram_mb": peak_ram,
            "total_gflops": total_gflops,
        },
        "evaluation": eval_metrics or None,
    }
    return append_workflow_csv_rows(
        legacy_task_run(result), device_type=device_type, context=legacy_result_context(result),
    )
