"""
Definição de workflows conforme tarefas.
============================================

 - Define os contratos de domínio para workflows monitorados por tarefa.
 - Fornece classes e funções para definir, executar e rastrear workflows de experimentos.

"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

WORKFLOW_SCHEMA_VERSION = "1.0"
LEGACY_TASK_ID = "legacy-main-task"

# Pares (módulo, nomes) cujas exceções indicam falha de infraestrutura (rede/timeout).
_INFRA_EXCEPTION_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("requests.exceptions", ("ReadTimeout", "ConnectTimeout", "ConnectionError", "Timeout")),
    ("urllib3.exceptions", ("ReadTimeoutError", "ConnectTimeoutError", "NameResolutionError")),
    ("httpx", ("TimeoutException", "ConnectError", "ReadTimeout")),
)


def classify_error_type(exc: BaseException) -> str:
    """Retorna 'InfraError' para falhas de rede/timeout; caso contrário, o nome da classe."""
    if isinstance(exc, TimeoutError):
        return "InfraError"
    for module_path, names in _INFRA_EXCEPTION_SOURCES:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            continue
        for name in names:
            cls = getattr(mod, name, None)
            if cls is not None and isinstance(exc, cls):
                return "InfraError"
    return exc.__class__.__name__


class TaskStatus(StrEnum):
    """Estados possíveis de uma tarefa ou tentativa de tarefa."""

    CREATED = "created"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    CACHED = "cached"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ArtifactKind(StrEnum):
    """Categorias de artefatos versionados manipulados por workflows."""

    DATA = "data"
    MODEL = "model"
    INDEX = "index"
    INTERACTION = "interaction"


class TaskActivity(StrEnum):
    """Atividades T0--T5 do modelo de workflow de modelos de linguagem."""

    INGESTION = "ingestion"
    PRETRAINING = "pretraining"
    ADAPTATION = "adaptation"
    INDEXING = "indexing"
    RETRIEVAL_GENERATION = "retrieval_generation"
    EVALUATION_MONITORING = "evaluation_monitoring"
    CUSTOM = "custom"


class ExecutionRegime(StrEnum):
    """Regime temporal de uma tarefa."""

    BUILD = "build"
    SERVICE = "service"


@dataclass(frozen=True)
class ArtifactDefinition:
    """Artefato imutável, versionado e rastreável de um workflow."""

    artifact_id: str
    kind: ArtifactKind
    version: str
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id deve ser informado.")
        if not self.version:
            raise ValueError("version deve ser informada.")


@dataclass(frozen=True)
class ResourceRequirements:
    """Requisitos declarados de recursos de uma tarefa."""

    cpu_cores: float | None = None
    memory_mb: float | None = None
    gpu_count: int = 0
    tpu_cores: int = 0
    coupling_degree: float = 0.0

    def __post_init__(self) -> None:
        if self.cpu_cores is not None and self.cpu_cores < 0:
            raise ValueError("cpu_cores não pode ser negativo.")
        if self.memory_mb is not None and self.memory_mb < 0:
            raise ValueError("memory_mb não pode ser negativo.")
        if self.gpu_count < 0:
            raise ValueError("gpu_count não pode ser negativo.")
        if self.tpu_cores < 0:
            raise ValueError("tpu_cores não pode ser negativo.")
        if not 0.0 <= self.coupling_degree <= 1.0:
            raise ValueError("coupling_degree deve estar entre 0 e 1.")


_VALID_TRANSITIONS = {
    TaskStatus.CREATED: {TaskStatus.READY, TaskStatus.CACHED, TaskStatus.CANCELLED, TaskStatus.SKIPPED},
    TaskStatus.READY: {TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.SKIPPED},
    TaskStatus.RUNNING: {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.FAILED: {TaskStatus.READY},
    TaskStatus.SUCCEEDED: set(),
    TaskStatus.CACHED: set(),
    TaskStatus.CANCELLED: set(),
    TaskStatus.SKIPPED: set(),
}


def validate_task_transition(current: TaskStatus, target: TaskStatus) -> None:
    """Valida uma transição de estado antes de ela ser persistida."""
    if target not in _VALID_TRANSITIONS[current]:
        raise ValueError(f"Transição inválida: {current.value} -> {target.value}")


@dataclass(frozen=True)
class RetryPolicy:
    """Política declarativa para novas tentativas após uma falha."""

    max_attempts: int = 1
    retryable_error_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts deve ser maior ou igual a 1.")

    def allows_retry(self, error_type: str) -> bool:
        """Retorna se o tipo de erro está habilitado para nova tentativa.

        Uma lista vazia aceita qualquer exceção; isso permite políticas simples
        de tentativa limitada sem acoplar a definição a classes Python.
        """
        if not self.retryable_error_types:
            return True
        return error_type in self.retryable_error_types


# Política padrão para tarefas de adaptação: recupera apenas falhas de infra
# (timeouts e erros de rede classificados como InfraError), sem mascarar
# erros funcionais de modelagem.
INFRA_RETRY_POLICY = RetryPolicy(max_attempts=3, retryable_error_types=("InfraError",))


@dataclass(frozen=True)
class TaskDefinition:
    """Definição declarativa de uma unidade funcional do workflow."""

    task_id: str
    name: str
    task_type: str = "train"
    depends_on: tuple[str, ...] = ()
    required: bool = True
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    config: dict[str, Any] = field(default_factory=dict)
    input_signatures: dict[str, str] = field(default_factory=dict)
    inputs: tuple[ArtifactDefinition, ...] = ()
    outputs: tuple[ArtifactDefinition, ...] = ()
    activity: TaskActivity = TaskActivity.CUSTOM
    regime: ExecutionRegime = ExecutionRegime.BUILD
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    is_composite: bool = False
    stop_predicate: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id deve ser informado.")
        if not self.name:
            raise ValueError("name deve ser informado.")
        if self.stop_predicate is not None and not self.is_composite:
            raise ValueError("stop_predicate exige uma tarefa composta.")


@dataclass(frozen=True)
class ExperimentDefinition:
    """Definição versionada de um workflow de experimento."""

    name: str
    tasks: tuple[TaskDefinition, ...]
    experiment_type: str = "custom"
    schema_version: str = WORKFLOW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("Um experimento deve conter ao menos uma tarefa.")
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Os identificadores de tarefa devem ser únicos.")


@dataclass
class TaskExecutionAttempt:
    """Tentativa concreta e rastreável de execução de uma tarefa."""

    attempt_id: str
    attempt_number: int
    status: TaskStatus = TaskStatus.CREATED
    started_at: str | None = None
    completed_at: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_type: str | None = None
    root_error_type: str | None = None
    failure_stage: str | None = None

    def transition_to(self, target: TaskStatus) -> None:
        validate_task_transition(self.status, target)
        self.status = target


@dataclass
class TaskRun:
    """Execução lógica de uma tarefa e seu histórico de tentativas."""

    task_id: str
    name: str
    task_type: str
    status: TaskStatus
    attempts: list[TaskExecutionAttempt] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    input_signatures: dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentRun:
    """Agregado de execução que preserva a visão global do experimento."""

    experiment_run_id: str
    definition_name: str
    status: str
    tasks: list[TaskRun]
    schema_version: str = WORKFLOW_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def legacy_task_run(
    result: dict[str, Any],
    *,
    task_id: str = LEGACY_TASK_ID,
) -> ExperimentRun:
    """Projeta o formato histórico como workflow de uma tarefa implícita."""
    experiment = result["experiment"]
    success = experiment["status"] == "success"
    resources = dict(result.get("resources", {}))
    if "task_time_sec" not in resources and "train_time_sec" in resources:
        resources["task_time_sec"] = resources["train_time_sec"]
    attempt = TaskExecutionAttempt(
        attempt_id=experiment["id"],
        attempt_number=1,
        status=TaskStatus.SUCCEEDED if success else TaskStatus.FAILED,
        started_at=experiment.get("timestamp_start"),
        completed_at=experiment.get("timestamp_end"),
        metrics={
            "resources": resources,
            "evaluation": result.get("evaluation"),
        },
        artifacts={"result_config_name": experiment["config_name"]},
        error=result.get("logs", {}).get("stderr_tail") or None,
    )
    task = TaskRun(
        task_id=task_id,
        name=task_id,
        task_type="train",
        status=attempt.status,
        attempts=[attempt],
    )
    return ExperimentRun(
        experiment_run_id=experiment["id"],
        definition_name=experiment["config_name"],
        status=experiment["status"],
        tasks=[task],
    )