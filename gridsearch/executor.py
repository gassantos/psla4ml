"""
Grid Search Executor — Orquestração de execução
=================================================

Motor de execução do grid search: gerencia estado, paralelismo,
checkpointing incremental e distribuição de GPUs.

Autor: Gustavo Alexandre
Data: 2026-02-15
"""

import json
import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from experiment.workflow import classify_error_type
from utils.device import get_torch_device
from utils.log_setup import get_log_queue, setup_worker_logging
from utils.paths import PathManager

from .grid import create_config_for_combination, generate_parameter_grid
from .sla_prefilter import prefilter_param_grid_by_execution_sla
from .utils import check_memory_availability, ensure_output_directories

logger = logging.getLogger(__name__)

_TDATE = datetime.now().astimezone().strftime("%Y-%m-%d")
_LOGFILE = PathManager.LOGS_DIR / f"grid_search_{_TDATE}.log"

# Device cache (lazy, evita NVML em CPU-only)
_device_type_cache: "str | None" = None


def _get_device_type() -> str:
    """Retorna o tipo de device (CPU/GPU/TPU), com cache para evitar re-detecção."""
    global _device_type_cache
    if _device_type_cache is None:
        _device_type_cache = get_torch_device()['type']
    return str(_device_type_cache)


def _environment_capacity_registry(grid_config: dict[str, Any]) -> dict[str, int]:
    """Extrai ``parallel_workers`` por ambiente de ``environments.details`` (BL-W2).

    Usado para limitar a alocação de GPU/worker à capacidade declarada de cada
    ambiente computacional (ex.: Colab T4 = 1 GPU, local RTX 3090 = 2), em vez
    de assumir sempre a capacidade global do pool de GPUs físicas detectadas.
    """
    details = grid_config.get("environments", {}).get("details", {})
    if not isinstance(details, dict):
        return {}
    return {
        name: int(info["parallel_workers"])
        for name, info in details.items()
        if isinstance(info, dict) and info.get("parallel_workers") is not None
    }


# ============================================================================
# DIRETÓRIOS — defaults injetáveis (DIP)
# ============================================================================
_DEFAULT_OUTPUT_DIR = PathManager.EXPERIMENTS_DIR / "grid_search"

GRID_OUTPUT_DIR = _DEFAULT_OUTPUT_DIR
GRID_CONFIGS_DIR = GRID_OUTPUT_DIR / "configs"

# Custo energético (USD/kWh)
ENERGY_COST_USD_PER_KWH = float(os.getenv("ENERGY_COST_USD_PER_KWH", "0.12"))

# Criar diretórios na importação
ensure_output_directories()


def _resolve_output_dir(output_dir: Path | None = None) -> Path:
    """Retorna o diretório de saída efetivo, aplicando o default do módulo."""
    return output_dir if output_dir is not None else _DEFAULT_OUTPUT_DIR


def _grid_state_file(output_dir: Path | None = None):
    base = _resolve_output_dir(output_dir)
    return base / f"grid_search_state_{_get_device_type()}_{_TDATE}.json"


def _grid_results_file(output_dir: Path | None = None):
    base = _resolve_output_dir(output_dir)
    return base / f"grid_search_results_{_get_device_type()}_{_TDATE}.json"


def _grid_summary_file(output_dir: Path | None = None):
    base = _resolve_output_dir(output_dir)
    return base / f"grid_search_summary_{_get_device_type()}_{_TDATE}.txt"


def _resource_catalog_file(output_dir: Path | None = None):
    base = _resolve_output_dir(output_dir)
    return base / f"resource_catalog_{_get_device_type()}_{_TDATE}.json"


def _build_workflow_metadata(
    experiment_idx: int,
    params: dict[str, Any],
    *,
    train_dataset: str,
    dataset_overrides: dict[str, str] | None,
    environment_details: dict[str, Any] | None,
    tpu_cores: int,
) -> dict[str, Any]:
    """Declara o workflow já executado pelo runner legado, sem reexecutá-lo."""
    from experiment.workflow_templates import (
        build_huggingface_workflow,
    )

    workflow = build_huggingface_workflow(_build_huggingface_workflow_config(
        experiment_idx,
        params,
        train_dataset=train_dataset,
        dataset_overrides=dataset_overrides,
        environment_details=environment_details,
        tpu_cores=tpu_cores,
    ))
    return asdict(workflow)


def _build_huggingface_workflow_config(
    experiment_idx: int,
    params: dict[str, Any],
    *,
    train_dataset: str,
    dataset_overrides: dict[str, str] | None,
    environment_details: dict[str, Any] | None,
    tpu_cores: int,
):
    """Monta a configuração comum ao metadata e à execução da combinação."""
    from experiment.workflow import ResourceRequirements
    from experiment.workflow_templates import HuggingFaceWorkflowConfig

    overrides = dict(dataset_overrides or {})
    details = dict(environment_details or {})
    source = overrides.get("hf_dataset_source", "local_json")
    dataset_id = overrides.get("hf_dataset_id", train_dataset)
    cores = details.get("cores_by_processor", {})
    cpu_cores = cores.get("CPU") if isinstance(cores, dict) else None
    vram_gb = details.get("vram_gb")
    resources = ResourceRequirements(
        cpu_cores=float(cpu_cores) if cpu_cores is not None else None,
        memory_mb=float(vram_gb) * 1024 if vram_gb is not None else None,
        gpu_count=1 if details.get("gpu") else 0,
        tpu_cores=tpu_cores,
        coupling_degree=0.9 if details.get("gpu") or tpu_cores else 0.0,
    )
    return HuggingFaceWorkflowConfig(
        name=f"grid-experiment-{experiment_idx}",
        dataset_source=source,
        dataset_id=dataset_id,
        dataset_config=overrides.get("hf_dataset_config"),
        ingestion_parameters={"train_dataset": train_dataset, **overrides},
        adaptation_parameters={"train_dataset": train_dataset, **overrides, **params},
        model_version="pending",
        metrics_version="pending",
        resources=resources,
    )


def _project_workflow_result(
    workflow_run,
    definition,
    *,
    workflow_run_dir: Path,
    experiment_idx: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Projeta o workflow no estado do grid sem perder telemetria por tarefa."""
    from experiment.aggregation import aggregate_workflow_run

    adapt = next(task for task in workflow_run.tasks if task.task_id == "adapt_model")
    adapt_metrics = adapt.attempts[-1].metrics if adapt.attempts else {}
    adapt_resources = adapt_metrics.get("resources", {})
    if not isinstance(adapt_resources, dict):
        adapt_resources = {}
    legacy_result = adapt_metrics.get("legacy_result", {})
    result = dict(legacy_result) if isinstance(legacy_result, dict) else {}
    summary = aggregate_workflow_run(workflow_run, definition)
    resources = {**adapt_resources, **summary["resources"]}
    resources["train_time_sec"] = adapt_resources.get(
        "task_time_sec", summary["resources"].get("task_time_sec")
    )
    result["experiment"] = result.get("experiment", {
        "id": workflow_run.experiment_run_id,
        "status": workflow_run.status,
    })
    result["resources"] = resources
    result["evaluation"] = summary["evaluation"]
    result["grid_params"] = params
    result["grid_experiment_idx"] = experiment_idx
    result["workflow"] = asdict(definition)
    result["task_telemetry"] = _task_telemetry(workflow_run, definition)
    result["workflow_run_dir"] = str(workflow_run_dir)
    result["workflow_summary"] = summary
    result["status"] = "success" if workflow_run.status == "success" else "failed"
    return result


def _task_telemetry(workflow_run, definition) -> list[dict[str, Any]]:
    """Serializa tentativas e sua atividade canônica para o estado do grid."""
    definitions = {task.task_id: task for task in definition.tasks}
    telemetry: list[dict[str, Any]] = []
    for task_run in workflow_run.tasks:
        task_definition = definitions[task_run.task_id]
        telemetry.append({
            "task_id": task_run.task_id,
            "name": task_run.name,
            "task_type": task_run.task_type,
            "activity": task_definition.activity.value,
            "regime": task_definition.regime.value,
            "status": task_run.status.value,
            "attempts": [asdict(attempt) for attempt in task_run.attempts],
        })
    return telemetry


# ============================================================================
# EXECUÇÃO DE EXPERIMENTO ÚNICO
# ============================================================================

def run_single_experiment(
    experiment_idx: int,
    config_path: str,
    params: dict[str, Any],
    gpu_list: list[int] | None = None,
    parallel_workers: int = 1,
    dataset_overrides: dict[str, str] | None = None,
    cloud_cost_per_hour_usd: float | None = None,
    tpu_cores: int = 1,
    environment_overrides: dict[str, str] | None = None,
    environment_details: dict[str, Any] | None = None,
    train_dataset: str = "train_task2",
) -> dict[str, Any]:
    """
    Executa um único experimento e retorna os resultados.

    Args:
        experiment_idx: Índice do experimento
        config_path: Caminho do arquivo de configuração
        params: Parâmetros do experimento
        gpu_list: GPUs a utilizar (ex: [0] ou [1]). None = detecta automaticamente.
        parallel_workers: Número de workers paralelos em uso no grid search
            que chamou este experimento (1 = sequencial). Salvo nos resultados
            para rastreabilidade.
        dataset_overrides: Chaves da seção ``[data]`` a sobrescrever no config
            (ex: ``{"hf_dataset_source": "hub", "hf_dataset_id": "nyu-mll/glue"}``).
        cloud_cost_per_hour_usd: Custo horário do ambiente de nuvem selecionado.
            Quando fornecido, ``execute_experiment`` usa a fórmula PSLA4ML:
            ``cost_usd = (train_time_sec / 3600) × cloud_cost_per_hour_usd``.
            ``None`` mantém o cálculo por tarifa flat de energia.
        environment_overrides: Chaves da seção ``[environment]`` a sobrescrever.

    Returns:
        Dicionário com resultados do experimento
    """
    # Imports lazy evitam inicialização de CUDA no processo principal.
    from experiment.helpers import load_config
    from experiment.persistence import append_workflow_csv_rows, write_workflow_run
    from experiment.task_executor import SequentialWorkflowExecutor
    from experiment.task_telemetry import TaskTelemetryCollector
    from experiment.workflow_templates import (
        build_huggingface_task_functions,
        build_huggingface_workflow,
    )

    logger.info(f"[{experiment_idx}] Iniciando experimento com parâmetros: {params}")
    workflow_config = _build_huggingface_workflow_config(
        experiment_idx,
        params,
        train_dataset=train_dataset,
        dataset_overrides=dataset_overrides,
        environment_details=environment_details,
        tpu_cores=tpu_cores,
    )

    try:
        definition = build_huggingface_workflow(workflow_config)
        monitoring = load_config(config_path).getboolean(
            "monitoring", "enable_monitoring", fallback=False,
        )
        workflow_run = SequentialWorkflowExecutor(
            build_huggingface_task_functions(
                workflow_config,
                config_path=config_path,
                gpu_list=gpu_list,
                parallel_workers=parallel_workers,
                train_file=train_dataset,
                environment_overrides=environment_overrides,
                environment_cost_per_hour_usd=cloud_cost_per_hour_usd,
                tpu_cores=tpu_cores,
            ),
            telemetry=TaskTelemetryCollector(
                enable_emissions=monitoring,
                environment_cost_per_hour_usd=cloud_cost_per_hour_usd,
            ),
        ).execute(definition)
        workflow_run_dir = write_workflow_run(workflow_run)
        append_workflow_csv_rows(
            workflow_run,
            device_type=_get_device_type(),
            definition=definition,
            context={
                "grid_experiment_idx": experiment_idx,
                "grid_params": params,
                "parallel_workers": parallel_workers,
                "selected_environment": params.get("environment"),
            },
        )
        result_data = _project_workflow_result(
            workflow_run,
            definition,
            workflow_run_dir=workflow_run_dir,
            experiment_idx=experiment_idx,
            params=params,
        )
        result_data["parallel_workers"] = parallel_workers
        if "environment" in params:
            result_data["selected_environment"] = params["environment"]
        if workflow_run.status != "success":
            raise RuntimeError("Workflow T0-T2-T5 falhou.")

        logger.info(f"[{experiment_idx}] Experimento concluído com sucesso")
        return result_data

    except Exception as e:
        logger.error(f"[{experiment_idx}] Erro no experimento: {e!s}")
        logger.debug(traceback.format_exc())

        return {
            "grid_experiment_idx": experiment_idx,
            "grid_params": params,
            "status": "failed",
            "error_type": classify_error_type(e),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "workflow": asdict(build_huggingface_workflow(workflow_config)),
        }


# ============================================================================
# ORQUESTRAÇÃO DO GRID SEARCH
# ============================================================================

def run_grid_search(
    base_config_path: str,
    grid_config: dict[str, list[Any]],
    resume: bool = False,
    parallel: int = 1,
    gpu_ids: list[int] | None = None,
    execution_sla_constraints: dict[str, float] | None = None,
    train_dataset: str = "train_task2",
    dataset_overrides: dict[str, str] | None = None,
    output_dir: Path | None = None,
    env_cost_registry: dict[str, float] | None = None,
    tpu_cores: int = 1,
    environment_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Executa busca em grade completa.

    Args:
        base_config_path: Caminho da configuração base
        grid_config: Configuração da grade de hiperparâmetros
        resume: Se True, retoma execução anterior
        parallel: Número de processos paralelos (1 = sequencial)
        gpu_ids: Lista explícita de GPUs disponíveis para distribuição
                 round-robin entre workers (ex: [0, 1, 2, 3]).
                 None = detecta automaticamente via torch.cuda.
        execution_sla_constraints: Constraints de SLA avaliáveis antes de
             executar o experimento. Atualmente suporta:
             ``peak_ram_mb`` (estimado por batch_size) e
             ``train_time_sec`` (apenas quando ``_meta`` fornece
             ``per_experiment_train_time_sec``).
        train_dataset: Nome do arquivo de treino sem extensão.
        dataset_overrides: Chaves da seção ``[data]`` a sobrescrever no config.
        output_dir: Diretório de saída para estado e resultados (DIP).
            ``None`` usa o default ``output/experiments/grid_search``.
        env_cost_registry: Mapeamento ``{nome_ambiente: cost_per_hour_usd}``
            extraído de ``environments.details`` do JSON multiambiente.
            Quando um experimento possui ``params["environment"]``, o custo
            horário correspondente é repassado a ``execute_experiment`` para
            usar a fórmula PSLA4ML: ``cost_usd = (train_time_sec/3600) × rate``.
            ``None`` mantém o cálculo por tarifa flat de energia para todos
            os experimentos.
        environment_overrides: Chaves da seção ``[environment]`` a sobrescrever.

    Returns:
        Lista com resultados de todos os experimentos
    """
    output_dir = _resolve_output_dir(output_dir)
    if tpu_cores > 1 and parallel > 1:
        raise ValueError("TPU multicore requer parallel=1 para evitar spawn aninhado.")

    # Carrega estado anterior se existir
    completed_experiments = set()
    all_results = []

    if resume and _grid_state_file(output_dir).exists():
        logger.info("Retomando execução anterior...")
        with open(_grid_state_file(output_dir), 'r', encoding='utf-8') as f:
            state = json.load(f)
            completed_experiments = set(state.get("completed_experiments", []))
            all_results = state.get("results", [])
        logger.info(f"Encontrados {len(completed_experiments)} experimentos já concluídos")

    # Estágio 0 — coleta de recursos computacionais (BL-MILP): detecção local
    # de hardware + catálogo estático de provedores de nuvem, persistidos
    # antes de qualquer experimento ser executado. Alimenta gridsearch.
    # milp_instance como um arquivo já pronto para leitura (sem heurística
    # embutida no leitor da instância MILP).
    env_details = grid_config.get("environments", {}).get("details") # type: ignore
    if env_details:
        from .resource_discovery import collect_and_persist_resource_catalog

        resource_catalog_path = collect_and_persist_resource_catalog(
            env_details, _resource_catalog_file(output_dir), results=all_results or None,
        )
        logger.info(
            "Estágio de coleta de recursos concluído: %s", resource_catalog_path,
        )

    # Gera grade de parâmetros (preserva índice original para compatibilidade
    # com retomada e rastreabilidade dos artefatos).
    param_grid = generate_parameter_grid(grid_config)
    indexed_param_grid = list(enumerate(param_grid))
    original_total_experiments = len(indexed_param_grid)

    # Pré-filtro opcional guiado por SLA antes de disparar workers
    sla_filter_info = {
        "enabled": bool(execution_sla_constraints),
        "constraints": execution_sla_constraints or {},
        "original_total_experiments": original_total_experiments,
        "eligible_total_experiments": original_total_experiments,
        "rejected_total_experiments": 0,
        "rejected_by_metric": {},
        "non_evaluable_constraints": [],
    }
    if execution_sla_constraints:
        indexed_param_grid, sla_filter_info = prefilter_param_grid_by_execution_sla(
            indexed_param_grid=indexed_param_grid,
            constraints=execution_sla_constraints,
            grid_config=grid_config,
        )

    total_experiments = len(indexed_param_grid)
    eligible_idx = {idx for idx, _ in indexed_param_grid}
    completed_eligible = completed_experiments.intersection(eligible_idx)

    logger.info(f"Total de experimentos: {total_experiments}")
    logger.info(f"Já concluídos: {len(completed_eligible)}")
    logger.info(f"Restantes: {total_experiments - len(completed_eligible)}")

    # Prepara experimentos pendentes
    pending_experiments = []
    for idx, params in indexed_param_grid:
        if idx in completed_experiments:
            continue

        config_path = create_config_for_combination(base_config_path, params, idx, train_file=train_dataset)
        pending_experiments.append((idx, config_path, params))

    if not pending_experiments:
        save_state(
            completed_experiments,
            all_results,
            sla_prefilter_info=sla_filter_info,
            output_dir=output_dir,
        )
        logger.info("Todos os experimentos já foram concluídos!")
        return all_results

    # Validação de memória antes de executar
    if parallel > 1:
        max_batch_size = max([p.get('batch_size', 16) for _, _, p in pending_experiments])
        is_safe, mem_message = check_memory_availability(parallel, max_batch_size)
        logger.info(f"\n{mem_message}\n")

        if not is_safe:
            response = input("Deseja continuar mesmo assim? (s/N): ")
            if response.lower() != 's':
                logger.info("Execução cancelada pelo usuário")
                sys.exit(0)

    # Distribui GPUs entre workers em round-robin, respeitando a capacidade
    # declarada por ambiente (parallel_workers em environments.details) — BL-W2.
    import torch as _torch
    _available_gpus: list[int] = (
        gpu_ids
        if gpu_ids is not None
        else list(range(_torch.cuda.device_count()))
    )
    env_capacity_registry = _environment_capacity_registry(grid_config)
    if env_capacity_registry:
        logger.info(
            "Capacidade de workers por ambiente carregada (BL-W2): %s",
            env_capacity_registry,
        )

    def _gpu_for(idx: int, params: dict[str, Any] | None = None) -> list[int] | None:
        """Retorna [gpu_id] para o worker `idx`, limitado à capacidade do ambiente selecionado.

        Quando `params["environment"]` tem `parallel_workers` declarado em
        `environments.details`, o round-robin passa a ciclar apenas sobre essa
        capacidade (ex.: Colab T4 sempre usa o mesmo slot 0), em vez de ciclar
        cegamente sobre todo o pool físico de GPUs disponivel.
        """
        if not _available_gpus:
            return None
        capacity = len(_available_gpus)
        env_name = (params or {}).get("environment")
        if env_name and env_name in env_capacity_registry:
            capacity = max(1, min(capacity, env_capacity_registry[env_name]))
        return [_available_gpus[idx % capacity]]

    def _cost_for_params(params: dict[str, Any]) -> float | None:
        """Retorna cost_per_hour_usd do ambiente selecionado, ou None.

        Usado para aplicar a fórmula PSLA4ML no cálculo de custo:
        ``cost_usd = (train_time_sec / 3600) × cost_per_hour_usd``.
        """
        if env_cost_registry is None:
            return None
        env_name = params.get("environment")
        if not env_name:
            return None
        return env_cost_registry.get(env_name)

    def _details_for_params(params: dict[str, Any]) -> dict[str, Any] | None:
        environment = params.get("environment")
        details = (env_details or {}).get(environment)
        return details if isinstance(details, dict) else None

    # Executa experimentos
    if parallel > 1:
        logger.info(
            "Executando em modo paralelo com %d workers | GPUs disponíveis: %s",
            parallel, _available_gpus or "CPU"
        )
        with ProcessPoolExecutor(
            max_workers=parallel,
            initializer=setup_worker_logging,
            initargs=(get_log_queue(),),
        ) as executor:
            futures = {
                executor.submit(
                    run_single_experiment,
                    idx, cfg, params, _gpu_for(idx, params), parallel,
                    dataset_overrides,
                    _cost_for_params(params),
                    tpu_cores,
                    environment_overrides,
                    _details_for_params(params),
                    train_dataset,
                ): idx
                for idx, cfg, params in pending_experiments
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    completed_experiments.add(idx)

                    # Salva estado incremental
                    save_state(
                        completed_experiments,
                        all_results,
                        sla_prefilter_info=sla_filter_info,
                        output_dir=output_dir,
                    )

                    completed_eligible = len(completed_experiments.intersection(eligible_idx)) # type: ignore
                    logger.info(f"Progresso: {completed_eligible}/{total_experiments}")

                except Exception as e:
                    logger.error(f"Erro ao executar experimento {idx}: {e}")
    else:
        logger.info("Executando em modo sequencial | GPUs disponíveis: %s", _available_gpus or "CPU")
        for idx, config_path, params in pending_experiments:
            result = run_single_experiment(
                idx, config_path, params, _gpu_for(idx, params),
                parallel_workers=parallel,
                dataset_overrides=dataset_overrides,
                cloud_cost_per_hour_usd=_cost_for_params(params),
                tpu_cores=tpu_cores,
                environment_overrides=environment_overrides,
                environment_details=_details_for_params(params),
                train_dataset=train_dataset,
            )
            all_results.append(result)
            completed_experiments.add(idx)

            # Salva estado incremental
            save_state(
                completed_experiments,
                all_results,
                sla_prefilter_info=sla_filter_info,
                output_dir=output_dir,
            )

            completed_eligible = len(completed_experiments.intersection(eligible_idx)) # type: ignore
            logger.info(f"Progresso: {completed_eligible}/{total_experiments}")

    return all_results


# ============================================================================
# PERSISTÊNCIA DE ESTADO
# ============================================================================

def save_state(
    completed_experiments: set,
    results: list[dict[str, Any]],
    sla_prefilter_info: dict[str, Any] | None = None,
    output_dir: Path | None = None,
):
    """Salva estado da execução para permitir retomada.

    Args:
        completed_experiments: Índices dos experimentos já concluídos.
        results: Lista de resultados acumulados.
        sla_prefilter_info: Informações do pré-filtro SLA.
        output_dir: Diretório de saída (Path). None = default do módulo.
    """
    state = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "completed_experiments": list(completed_experiments),
        "results": results,
        "sla_prefilter": sla_prefilter_info or {
            "enabled": False,
            "constraints": {},
        },
    }

    with open(_grid_state_file(output_dir), 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
