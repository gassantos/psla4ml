"""
Persistência de resultados de experimentos
============================================

Construção do dicionário JSON e escrita em CSV acumulado.

Autor: Gustavo Alexandre
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .helpers import METRICS_DIR
from .workflow import (
    ExperimentDefinition,
    ExperimentRun,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
)

logger = logging.getLogger(__name__)


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
    csv_path = _workflow_summary_csv_path(device_type)
    headers = [
        "experiment_run_id", "definition_name", "workflow_status",
        "task_id", "task_name", "task_type", "activity", "regime", "task_status",
        "attempt_id", "attempt_number", "attempt_status", "started_at", "completed_at",
        "task_time_sec", "avg_ram_mb", "peak_ram_mb", "vram_mb", "peak_vram_mb",
        "energy_kwh", "emissions_kg_co2", "cost_usd",
        "resources_json", "evaluation_json", "artifacts_json",
        "error_type", "root_error_type", "failure_stage", "error",
        "experiment_context_json",
    ]
    write_header = not csv_path.exists()
    definitions = {task.task_id: task for task in definition.tasks} if definition else {}

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if write_header:
            writer.writeheader()
        for task in workflow.tasks:
            task_definition = definitions.get(task.task_id)
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
                    "cost_usd": resources.get("cost_usd"),
                    "resources_json": json.dumps(resources, ensure_ascii=False, sort_keys=True),
                    "evaluation_json": json.dumps(metrics.get("evaluation", {}), ensure_ascii=False, sort_keys=True),
                    "artifacts_json": json.dumps(attempt.artifacts, ensure_ascii=False, sort_keys=True),
                    "error_type": attempt.error_type,
                    "root_error_type": attempt.root_error_type,
                    "failure_stage": attempt.failure_stage,
                    "error": attempt.error,
                    "experiment_context_json": json.dumps(context or {}, ensure_ascii=False, sort_keys=True),
                })
    return csv_path


def _workflow_summary_csv_path(device_type: str) -> Path:
    """Evita misturar a telemetria por tarefa com um CSV legado existente."""
    suffix = datetime.now().astimezone().strftime("%Y%m%d")
    csv_path = METRICS_DIR / f"experiment_summary_{device_type}{suffix}.csv"
    if not csv_path.exists():
        return csv_path
    with open(csv_path, newline="", encoding="utf-8") as file:
        header = next(csv.reader(file), [])
    return (
        csv_path
        if "task_id" in header and "attempt_id" in header
        else METRICS_DIR / f"experiment_summary_{device_type}{suffix}_task_telemetry.csv"
    )


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
) -> Path:
    """Acrescenta uma linha no CSV acumulado de sumário."""
    csv_filename = (
        f"experiment_summary_{device_type}"
        f"{datetime.now().astimezone().strftime('%Y%m%d')}.csv"
    )
    csv_path = METRICS_DIR / csv_filename
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "experiment_id",
                "config_name",
                "seed",
                "device_type",
                "parallel_workers",
                "train_dataset",
                "optimizer",
                "learning_rate",
                "batch_size",
                "epoch",
                "train_time_sec",
                "energy_kwh",
                "emissions_kg",
                "cost_usd",
                "avg_ram_mb",
                "peak_ram_mb",
                "avg_gflops_per_batch",
                "total_gflops",
                "status",
                "timestamp",
                "eval_precision",
                "eval_recall",
                "eval_f1",
                "eval_accuracy",
                "eval_source",
            ])

        writer.writerow([
            experiment_id,
            json_filename,
            seed,
            device_type,
            parallel_workers,
            train_dataset_name,
            optimizer,
            learning_rate,
            batch_size,
            epoch,
            f"{exec_time:.2f}",
            energy_kwh,
            emissions_kg,
            f"{cost_usd:.6f}" if cost_usd is not None else None,
            avg_ram,
            peak_ram,
            avg_gflops_per_batch,
            total_gflops,
            status,
            end_iso,
            f"{eval_metrics['precision']:.4f}" if eval_metrics else None,
            f"{eval_metrics['recall']:.4f}" if eval_metrics else None,
            f"{eval_metrics['f1_score']:.4f}" if eval_metrics else None,
            f"{eval_metrics['accuracy']:.4f}" if eval_metrics and 'accuracy' in eval_metrics else None,
            eval_metrics.get("source") if eval_metrics else None,
        ])

    return csv_path
