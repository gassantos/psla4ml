"""Templates declarativos de etapas basicas por dominio de experimento."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .generic_workflow import SUPPORTED_EXPERIMENT_TYPES
from .workflow import (
    INFRA_RETRY_POLICY,
    ArtifactDefinition,
    ArtifactKind,
    ExecutionRegime,
    ExperimentDefinition,
    ResourceRequirements,
    RetryPolicy,
    TaskActivity,
    TaskDefinition,
)

DatasetProbe = Callable[[], dict[str, Any]]
ExperimentLauncher = Callable[..., dict[str, Any] | None]


@dataclass(frozen=True)
class WorkflowTaskTemplate:
    """Etapa padrao de um tipo de workflow."""

    task_id: str
    name: str
    task_type: str
    depends_on: tuple[str, ...] = ()
    activity: TaskActivity = TaskActivity.CUSTOM
    regime: ExecutionRegime = ExecutionRegime.BUILD
    resources: ResourceRequirements = ResourceRequirements()


@dataclass(frozen=True)
class DomainWorkflowProfile:
    """Perfil declarativo de um domínio, independente de seu runner futuro."""

    experiment_type: str
    framework: str
    tasks: tuple[WorkflowTaskTemplate, ...]


@dataclass(frozen=True)
class HuggingFaceWorkflowConfig:
    """Configuração do template T0 -> T2 -> T5 para Hugging Face."""

    name: str
    dataset_source: str = "local_json"
    dataset_id: str = "train_task2"
    dataset_config: str | None = None
    dataset_version: str = "input"
    dataset_splits: dict[str, str] = field(
        default_factory=lambda: {"train": "train", "valid": "validation", "test": "test"}
    )
    normalization_schema: tuple[str, ...] = ("guid", "text_a", "text_b", "label")
    ingestion_parameters: dict[str, Any] = field(default_factory=dict)
    adaptation_parameters: dict[str, Any] = field(default_factory=dict)
    model_version: str = "pending"
    metrics_version: str = "pending"
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)

    def __post_init__(self) -> None:
        if self.dataset_source not in {"hub", "local_json"}:
            raise ValueError("dataset_source deve ser 'hub' ou 'local_json'.")
        if not self.dataset_id:
            raise ValueError("dataset_id deve ser informado.")
        if not self.name:
            raise ValueError("name deve ser informado.")


@dataclass(frozen=True)
class LauncherWorkflowConfig:
    """Configuração do template T0 -> T2 -> T5 para um config local existente."""

    name: str
    config_path: str
    train_dataset: str
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)


DOMAIN_WORKFLOW_PROFILES: dict[str, DomainWorkflowProfile] = {
    "ml_classic": DomainWorkflowProfile("ml_classic", "scikit-learn", (
        WorkflowTaskTemplate("ingest_data", "Carregar dados", "ingest", activity=TaskActivity.INGESTION),
        WorkflowTaskTemplate("prepare_features", "Preparar atributos", "prepare", ("ingest_data",), TaskActivity.INGESTION),
        WorkflowTaskTemplate("train_model", "Treinar modelo", "train", ("prepare_features",), TaskActivity.ADAPTATION),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("train_model",), TaskActivity.EVALUATION_MONITORING),
    )),
    "deep_learning": DomainWorkflowProfile("deep_learning", "pytorch", (
        WorkflowTaskTemplate("ingest_data", "Carregar dados", "ingest", activity=TaskActivity.INGESTION),
        WorkflowTaskTemplate("prepare_data", "Preparar dados", "prepare", ("ingest_data",), TaskActivity.INGESTION),
        WorkflowTaskTemplate("train_model", "Treinar modelo", "train", ("prepare_data",), TaskActivity.ADAPTATION, resources=ResourceRequirements(gpu_count=1, coupling_degree=0.9)),
        WorkflowTaskTemplate("validate_model", "Validar modelo", "validate", ("train_model",), TaskActivity.EVALUATION_MONITORING, resources=ResourceRequirements(gpu_count=1)),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("validate_model",), TaskActivity.EVALUATION_MONITORING),
    )),
    "nlp": DomainWorkflowProfile("nlp", "spacy", (
        WorkflowTaskTemplate("ingest_data", "Carregar textos", "ingest", activity=TaskActivity.INGESTION),
        WorkflowTaskTemplate("preprocess_text", "Preprocessar textos", "prepare", ("ingest_data",), TaskActivity.INGESTION),
        WorkflowTaskTemplate("train_model", "Treinar modelo", "train", ("preprocess_text",), TaskActivity.ADAPTATION),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("train_model",), TaskActivity.EVALUATION_MONITORING),
    )),
    "llm": DomainWorkflowProfile("llm", "huggingface", (
        WorkflowTaskTemplate("ingest_data", "Carregar dados", "ingest", activity=TaskActivity.INGESTION),
        WorkflowTaskTemplate("prepare_corpus", "Preparar corpus", "prepare", ("ingest_data",), TaskActivity.INGESTION),
        WorkflowTaskTemplate("adapt_model", "Adaptar modelo", "train", ("prepare_corpus",), TaskActivity.ADAPTATION, resources=ResourceRequirements(gpu_count=1, coupling_degree=0.9)),
        WorkflowTaskTemplate("evaluate_model", "Avaliar modelo", "evaluate", ("adapt_model",), TaskActivity.EVALUATION_MONITORING, resources=ResourceRequirements(gpu_count=1)),
        WorkflowTaskTemplate("publish_model", "Publicar modelo", "publish", ("evaluate_model",), TaskActivity.EVALUATION_MONITORING),
    )),
}

DOMAIN_WORKFLOW_TEMPLATES = {
    experiment_type: profile.tasks
    for experiment_type, profile in DOMAIN_WORKFLOW_PROFILES.items()
}


def build_domain_workflow(
    name: str,
    experiment_type: str,
    *,
    task_configs: Mapping[str, Mapping[str, Any]] | None = None,
    task_input_signatures: Mapping[str, Mapping[str, str]] | None = None,
) -> ExperimentDefinition:
    """Cria a DAG base do dominio com perfis opcionais por tarefa.

    Esta funcao define o ciclo de vida comum do experimento, independente de
    biblioteca, modelo ou fonte dos dados. As implementacoes concretas podem
    associar callables, comandos ou pipelines a essas tarefas posteriormente.
    """
    if experiment_type not in SUPPORTED_EXPERIMENT_TYPES:
        supported = ", ".join(sorted(SUPPORTED_EXPERIMENT_TYPES))
        raise ValueError(f"experiment_type invalido: {experiment_type}. Use: {supported}.")
    task_configs = task_configs or {}
    task_input_signatures = task_input_signatures or {}
    template = DOMAIN_WORKFLOW_TEMPLATES[experiment_type]
    return ExperimentDefinition(
        name=name,
        experiment_type=experiment_type,
        tasks=tuple(
            TaskDefinition(
                task_id=task.task_id,
                name=task.name,
                task_type=task.task_type,
                depends_on=task.depends_on,
                config=dict(task_configs.get(task.task_id, {})),
                input_signatures=dict(task_input_signatures.get(task.task_id, {})),
                activity=task.activity,
                regime=task.regime,
                resources=task.resources,
            )
            for task in template
        ),
    )


def build_huggingface_workflow(config: HuggingFaceWorkflowConfig) -> ExperimentDefinition:
    """Cria o workflow de construção T0 -> T2 -> T5 para Hugging Face.

    As dependências são inferidas pelos artefatos versionados. Os adaptadores
    executáveis das tarefas pertencem à próxima etapa de integração do runtime.
    """
    dataset_uri = (
        f"hf://datasets/{config.dataset_id}"
        if config.dataset_source == "hub"
        else f"data/{config.dataset_id}.json"
    )
    dataset_metadata: dict[str, Any] = {
        "source": config.dataset_source,
        "dataset_id": config.dataset_id,
        "splits": dict(config.dataset_splits),
        "normalization_schema": list(config.normalization_schema),
    }
    if config.dataset_config is not None:
        dataset_metadata["dataset_config"] = config.dataset_config

    dataset = ArtifactDefinition(
        artifact_id=f"dataset-{config.dataset_id}",
        kind=ArtifactKind.DATA,
        version=config.dataset_version,
        uri=dataset_uri,
        metadata=dataset_metadata,
    )
    model = ArtifactDefinition(
        artifact_id=f"model-{config.name}",
        kind=ArtifactKind.MODEL,
        version=config.model_version,
    )
    metrics = ArtifactDefinition(
        artifact_id=f"metrics-{config.name}",
        kind=ArtifactKind.DATA,
        version=config.metrics_version,
    )
    dataset_signature = _semantic_signature({
        "source": config.dataset_source,
        "dataset_id": config.dataset_id,
        "dataset_config": config.dataset_config,
        "revision": config.dataset_version,
        "splits": config.dataset_splits,
        "normalization_schema": config.normalization_schema,
        "parameters": config.ingestion_parameters,
    })
    model_signature = _semantic_signature({
        "dataset": dataset_signature,
        "model_version": config.model_version,
        "parameters": config.adaptation_parameters,
    })
    evaluation_signature = _semantic_signature({
        "model": model_signature,
        "metrics_version": config.metrics_version,
    })

    return ExperimentDefinition(
        name=config.name,
        experiment_type="llm",
        tasks=(
            TaskDefinition(
                task_id="ingest_dataset",
                name="Carregar dataset",
                task_type="ingest",
                config={
                    "hf_dataset_source": config.dataset_source,
                    "hf_dataset_id": config.dataset_id,
                    "hf_dataset_config": config.dataset_config,
                    "normalization_schema": list(config.normalization_schema),
                    **config.ingestion_parameters,
                },
                input_signatures={"dataset": dataset_signature},
                outputs=(dataset,),
                activity=TaskActivity.INGESTION,
                regime=ExecutionRegime.BUILD,
            ),
            TaskDefinition(
                task_id="adapt_model",
                name="Adaptar modelo",
                config=dict(config.adaptation_parameters),
                input_signatures={"dataset": dataset_signature, "model": model_signature},
                inputs=(dataset,),
                outputs=(model,),
                activity=TaskActivity.ADAPTATION,
                regime=ExecutionRegime.BUILD,
                resources=config.resources,
                retry_policy=INFRA_RETRY_POLICY,
            ),
            TaskDefinition(
                task_id="evaluate_model",
                name="Avaliar modelo",
                task_type="evaluate",
                input_signatures={"model": model_signature, "evaluation": evaluation_signature},
                inputs=(model,),
                outputs=(metrics,),
                activity=TaskActivity.EVALUATION_MONITORING,
                regime=ExecutionRegime.BUILD,
                resources=config.resources,
                retry_policy=RetryPolicy(max_attempts=2),
            ),
        ),
    )


def build_huggingface_task_functions(
    workflow_config: HuggingFaceWorkflowConfig,
    *,
    config_path: str,
    gpu_list: list[int] | None = None,
    parallel_workers: int = 1,
    train_file: str | None = None,
    environment_overrides: dict[str, str] | None = None,
    environment_cost_per_hour_usd: float | None = None,
    tpu_cores: int = 1,
    dataset_probe: DatasetProbe | None = None,
    experiment_launcher: ExperimentLauncher | None = None,
) -> Mapping[str, Callable[[], dict[str, Any]]]:
    """Cria os adaptadores executáveis do workflow Hugging Face T0 -> T2 -> T5.

    Enquanto o runner legado concentra treino e avaliação em uma única chamada,
    T2 o encapsula e T5 projeta a avaliação retornada como saída explícita da
    tarefa. A separação física da avaliação será feita quando o runner for
    decomposto em tarefas nativas.
    """
    result_holder: dict[str, dict[str, Any]] = {}
    probe = dataset_probe or _build_huggingface_dataset_probe(
        workflow_config, config_path=config_path, train_file=train_file,
    )

    def ingest_dataset() -> dict[str, Any]:
        metadata = probe()
        workflow = build_huggingface_workflow(workflow_config)
        dataset = workflow.tasks[0].outputs[0]
        dataset_metrics = {
            **metadata,
            "version": dataset.version,
            "uri": dataset.uri,
            "splits": dataset.metadata["splits"],
        }
        return {
            "metrics": {"dataset": dataset_metrics},
            "artifacts": {"dataset": _artifact_record(dataset)},
        }

    def adapt_model() -> dict[str, Any]:
        launcher = experiment_launcher or _launch_experiment
        result = launcher(
            config_path=config_path,
            gpu_list=gpu_list,
            parallel_workers=parallel_workers,
            train_file=train_file,
            dataset_overrides=_dataset_overrides(workflow_config),
            environment_overrides=environment_overrides,
            environment_cost_per_hour_usd=environment_cost_per_hour_usd,
            tpu_cores=tpu_cores,
            collect_resource_telemetry=False,
        )
        if result is None:
            raise RuntimeError("O launcher não retornou resultado para a adaptação Hugging Face.")
        if result.get("experiment", {}).get("status") != "success":
            logs = result.get("logs", {})
            exc = RuntimeError(logs.get("stderr_tail") or "Adaptação Hugging Face falhou.")
            exc.failure_stage = logs.get("failure_stage")  # type: ignore[attr-defined]
            raise exc
        result_holder["result"] = result
        workflow = build_huggingface_workflow(workflow_config)
        model = workflow.tasks[1].outputs[0]
        model_record = _artifact_record(model)
        model_record["uri"] = model_record["uri"] or _checkpoint_uri(config_path)
        resources = result.get("resources", {})
        return {
            "metrics": {
                "resources": dict(resources),
                "projected_evaluation": result.get("evaluation") or {},
                "legacy_result": {
                    key: value for key, value in result.items()
                    if key not in {"resources", "evaluation"}
                },
            },
            "artifacts": {"model": model_record, "checkpoint": model_record},
        }

    def evaluate_model() -> dict[str, Any]:
        result = result_holder.get("result")
        if result is None:
            raise RuntimeError("Avaliação Hugging Face requer uma adaptação concluída.")
        workflow = build_huggingface_workflow(workflow_config)
        metrics = workflow.tasks[2].outputs[0]
        return {
            "metrics": {"evaluation": result.get("evaluation") or {}},
            "artifacts": {"metrics": _artifact_record(metrics)},
        }

    def restore_adaptation(metrics: Mapping[str, Any]) -> None:
        evaluation = metrics.get("projected_evaluation")
        if isinstance(evaluation, dict):
            result_holder["result"] = {"evaluation": evaluation}

    setattr(adapt_model, "restore_cached_result", restore_adaptation)

    return {
        "ingest_dataset": ingest_dataset,
        "adapt_model": adapt_model,
        "evaluate_model": evaluate_model,
    }


def build_launcher_workflow(config: LauncherWorkflowConfig) -> ExperimentDefinition:
    """Cria um workflow canônico para o launcher configurado localmente."""
    dataset = ArtifactDefinition(
        artifact_id=f"dataset-{config.name}", kind=ArtifactKind.DATA,
        version="input", uri=f"data/{config.train_dataset}.json",
        metadata={"source": "config", "dataset_id": config.train_dataset},
    )
    model = ArtifactDefinition(
        artifact_id=f"model-{config.name}", kind=ArtifactKind.MODEL, version="pending",
    )
    metrics = ArtifactDefinition(
        artifact_id=f"metrics-{config.name}", kind=ArtifactKind.DATA, version="pending",
    )
    dataset_signature = _semantic_signature({
        "config_path": config.config_path, "train_dataset": config.train_dataset,
    })
    return ExperimentDefinition(
        name=config.name,
        experiment_type="nlp",
        tasks=(
            TaskDefinition(
                "ingest_dataset", "Carregar dataset", task_type="ingest",
                config={"config_path": config.config_path, "train_dataset": config.train_dataset},
                input_signatures={"dataset": dataset_signature}, outputs=(dataset,),
                activity=TaskActivity.INGESTION, regime=ExecutionRegime.BUILD,
            ),
            TaskDefinition(
                "adapt_model", "Adaptar modelo", config={"config_path": config.config_path},
                input_signatures={"dataset": dataset_signature}, inputs=(dataset,), outputs=(model,),
                activity=TaskActivity.ADAPTATION, regime=ExecutionRegime.BUILD,
                resources=config.resources,
            ),
            TaskDefinition(
                "evaluate_model", "Avaliar modelo", task_type="evaluate",
                input_signatures={"dataset": dataset_signature}, inputs=(model,), outputs=(metrics,),
                activity=TaskActivity.EVALUATION_MONITORING, regime=ExecutionRegime.BUILD,
                resources=config.resources,
            ),
        ),
    )


def build_launcher_task_functions(
    config: LauncherWorkflowConfig,
    *,
    gpu_list: list[int] | None = None,
    environment_overrides: dict[str, str] | None = None,
    tpu_cores: int = 1,
    experiment_launcher: ExperimentLauncher | None = None,
) -> Mapping[str, Callable[[], dict[str, Any]]]:
    """Adapta o launcher legado ao mesmo contrato T0 -> T2 -> T5."""
    result_holder: dict[str, dict[str, Any]] = {}

    def ingest_dataset() -> dict[str, Any]:
        workflow = build_launcher_workflow(config)
        dataset = workflow.tasks[0].outputs[0]
        return {"metrics": {"dataset": {**dataset.metadata, "uri": dataset.uri}},
                "artifacts": {"dataset": _artifact_record(dataset)}}

    def adapt_model() -> dict[str, Any]:
        launcher = experiment_launcher or _launch_experiment
        result = launcher(
            config_path=config.config_path, gpu_list=gpu_list, parallel_workers=1,
            train_file=config.train_dataset, dataset_overrides=None,
            environment_overrides=environment_overrides, tpu_cores=tpu_cores,
            collect_resource_telemetry=False,
        )
        if result is None or result.get("experiment", {}).get("status") != "success":
            logs = {} if result is None else result.get("logs", {})
            exc = RuntimeError(logs.get("stderr_tail") or "Adaptação do launcher local falhou.")
            exc.failure_stage = logs.get("failure_stage")  # type: ignore[attr-defined]
            raise exc
        result_holder["result"] = result
        model = build_launcher_workflow(config).tasks[1].outputs[0]
        return {
            "metrics": {"resources": dict(result.get("resources", {})),
                        "projected_evaluation": result.get("evaluation") or {},
                        "legacy_result": {key: value for key, value in result.items()
                                          if key not in {"resources", "evaluation"}}},
            "artifacts": {"model": _artifact_record(model)},
        }

    def evaluate_model() -> dict[str, Any]:
        result = result_holder.get("result")
        if result is None:
            raise RuntimeError("Avaliação requer uma adaptação concluída.")
        metrics = build_launcher_workflow(config).tasks[2].outputs[0]
        return {"metrics": {"evaluation": result.get("evaluation") or {}},
                "artifacts": {"metrics": _artifact_record(metrics)}}

    return {"ingest_dataset": ingest_dataset, "adapt_model": adapt_model, "evaluate_model": evaluate_model}


def _build_huggingface_dataset_probe(
    workflow_config: HuggingFaceWorkflowConfig,
    *,
    config_path: str,
    train_file: str | None,
) -> DatasetProbe:
    def probe() -> dict[str, Any]:
        from dataset.nlp.HuggingFace import HuggingFaceDataset

        from .helpers import load_config

        config = load_config(config_path)
        if not config.has_section("data"):
            config.add_section("data")
        if train_file is not None:
            config.set("data", "train_file_list", f"{train_file}.json")
        for key, value in _dataset_overrides(workflow_config).items():
            config.set("data", key, value)
        dataset = HuggingFaceDataset(config, "train")
        size = len(dataset)
        if size < 1:
            raise ValueError("O dataset Hugging Face não possui exemplos de treino.")
        sample = dataset[0]
        return {"records": size, "fields": sorted(sample), "source": workflow_config.dataset_source}

    return probe


def _dataset_overrides(config: HuggingFaceWorkflowConfig) -> dict[str, str]:
    overrides = {
        "hf_dataset_source": config.dataset_source,
        "hf_dataset_id": config.dataset_id,
        "train_dataset_type": "HuggingFace",
        "valid_dataset_type": "HuggingFace",
        "test_dataset_type": "HuggingFace",
    }
    if config.dataset_config is not None:
        overrides["hf_dataset_config"] = config.dataset_config
    return overrides


def _artifact_record(artifact: ArtifactDefinition) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "kind": artifact.kind.value,
        "version": artifact.version,
        "uri": artifact.uri,
        "metadata": dict(artifact.metadata),
    }


def _semantic_signature(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _checkpoint_uri(config_path: str) -> str | None:
    """Resolve o diretório de checkpoints declarado na configuração de treino."""
    from .helpers import load_config

    try:
        config = load_config(config_path)
    except OSError:
        return None
    if not config.has_section("output"):
        return None
    model_path = config.get("output", "model_path", fallback="").strip()
    model_name = config.get("output", "model_name", fallback="").strip()
    if not model_path or not model_name:
        return None
    return f"{model_path.rstrip('/')}/{model_name}"


def _launch_experiment(**kwargs: Any) -> dict[str, Any] | None:
    from .xla_launcher import launch_experiment

    return launch_experiment(**kwargs)