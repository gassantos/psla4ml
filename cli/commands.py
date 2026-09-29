"""
Command Pattern — Despacho de modos de execução
=================================================

Classes de comando para o Gridsearch Experiment Runner (Command Pattern).

Autor: Gustavo Alexandre
"""

import argparse
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from experiment.bertpli_workflow import (
    BertPliWorkflowConfig,
    build_bertpli_task_functions,
    build_bertpli_workflow,
)
from experiment.generic_workflow import (
    build_generic_task_functions,
    build_generic_workflow,
    load_generic_workflow_spec,
)
from experiment.helpers import load_config
from experiment.persistence import (
    append_workflow_csv_rows,
    load_workflow_run,
    write_workflow_run,
)
from experiment.task_cache import TaskCache
from experiment.task_executor import SequentialWorkflowExecutor
from experiment.task_telemetry import TaskTelemetryCollector
from experiment.workflow import ResourceRequirements
from experiment.workflow_templates import (
    HuggingFaceWorkflowConfig,
    LauncherWorkflowConfig,
    build_huggingface_task_functions,
    build_huggingface_workflow,
    build_launcher_task_functions,
    build_launcher_workflow,
)
from utils.device import get_torch_device

from .runners import (
    _build_dataset_overrides,
    run_grid_search_experiments,
    run_skyband_analysis,
)


def _workflow_code_version() -> str:
    """Obtém a revisão Git usada para invalidar cache após mudanças de código."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"
    return result.stdout.strip() or "unversioned"


class Command(ABC):
    """Interface para comandos de execução (Command Pattern)."""

    @abstractmethod
    def execute(self, args: argparse.Namespace, sla_dict: dict) -> None:
        """Executa a ação do comando."""


class SkybandOnlyCommand(Command):
    """Executa apenas a análise Skyband sobre estado existente."""

    def execute(self, args: argparse.Namespace, sla_dict: dict) -> None:
        run_skyband_analysis(
            k=args.skyband_k,
            sla_constraints=sla_dict or None,
            sla_profile_name=args.sla_profile,
            metrics=args.skyband_metrics,
            compare=args.skyband_compare,
            state_file=args.skyband_state,
        )


class SingleCommand(Command):
    """Executa um único experimento, opcionalmente seguido de Skyband."""

    def execute(self, args: argparse.Namespace, sla_dict: dict) -> None:
        dataset_overrides = _build_dataset_overrides(args)
        if dataset_overrides:
            self._execute_huggingface_workflow(args, dataset_overrides)
        else:
            self._execute_launcher_workflow(args)
        if not args.no_skyband:
            # require_state=False: modo single não gera estado de grid search;
            # a ausência do arquivo é aviso, não erro.
            run_skyband_analysis(
                k=args.skyband_k,
                sla_constraints=sla_dict or None,
                sla_profile_name=args.sla_profile,
                metrics=args.skyband_metrics,
                compare=args.skyband_compare,
                state_file=args.skyband_state,
                require_state=False,
            )

    @staticmethod
    def _execute_huggingface_workflow(
        args: argparse.Namespace,
        dataset_overrides: dict[str, str],
    ) -> None:
        if args.dataset_source == "hub" and not args.dataset_id:
            raise ValueError("--dataset-id e obrigatorio quando --dataset-source hub.")

        device_type = get_torch_device()["type"]
        gpu_count = len(args.gpu) if args.gpu else (1 if device_type == "GPU" else 0)
        tpu_cores = args.tpu_cores if device_type == "TPU" else 0
        config = HuggingFaceWorkflowConfig(
            name=f"huggingface-single-{args.train_dataset}",
            dataset_source=args.dataset_source,
            dataset_id=args.dataset_id or args.train_dataset,
            dataset_config=args.dataset_config,
            ingestion_parameters={"train_dataset": args.train_dataset, **dataset_overrides},
            adaptation_parameters={"config_path": args.config},
            resources=ResourceRequirements(
                gpu_count=gpu_count,
                tpu_cores=tpu_cores,
                coupling_degree=0.9 if gpu_count or tpu_cores else 0.0,
            ),
        )
        monitoring = load_config(args.config).getboolean(
            "monitoring", "enable_monitoring", fallback=False,
        )
        definition = build_huggingface_workflow(config)
        resume_from = (
            load_workflow_run(Path(args.workflow_resume_run))
            if args.workflow_resume_run else None
        )
        cache = TaskCache(Path(args.workflow_cache_dir)) if args.workflow_cache_dir else None
        workflow = SequentialWorkflowExecutor(
            build_huggingface_task_functions(
                config,
                config_path=args.config,
                gpu_list=args.gpu,
                train_file=args.train_dataset,
                environment_overrides={"precision": args.precision} if args.precision else None,
                tpu_cores=args.tpu_cores,
            ),
            cache=cache,
            code_version=_workflow_code_version() if cache else None,
            telemetry=TaskTelemetryCollector(enable_emissions=monitoring),
        ).execute(definition, resume_from=resume_from)
        run_dir = write_workflow_run(workflow)
        append_workflow_csv_rows(workflow, device_type=device_type, definition=definition)
        if workflow.status != "success":
            raise RuntimeError(f"Workflow Hugging Face falhou. Manifesto: {run_dir}")

    @staticmethod
    def _execute_launcher_workflow(args: argparse.Namespace) -> None:
        """Executa o config local pelo template canônico, com um único worker."""
        device_type = get_torch_device()["type"]
        gpu_count = len(args.gpu) if args.gpu else (1 if device_type == "GPU" else 0)
        tpu_cores = args.tpu_cores if device_type == "TPU" else 0
        config = LauncherWorkflowConfig(
            name=f"single-{args.train_dataset}",
            config_path=args.config,
            train_dataset=args.train_dataset,
            resources=ResourceRequirements(
                gpu_count=gpu_count,
                tpu_cores=tpu_cores,
                coupling_degree=0.9 if gpu_count or tpu_cores else 0.0,
            ),
        )
        monitoring = load_config(args.config).getboolean(
            "monitoring", "enable_monitoring", fallback=False,
        )
        resume_from = (
            load_workflow_run(Path(args.workflow_resume_run))
            if args.workflow_resume_run else None
        )
        cache = TaskCache(Path(args.workflow_cache_dir)) if args.workflow_cache_dir else None
        workflow = SequentialWorkflowExecutor(
            build_launcher_task_functions(
                config,
                gpu_list=args.gpu,
                environment_overrides={"precision": args.precision} if args.precision else None,
                tpu_cores=args.tpu_cores,
            ),
            cache=cache,
            code_version=_workflow_code_version() if cache else None,
            telemetry=TaskTelemetryCollector(enable_emissions=monitoring),
        ).execute(build_launcher_workflow(config), resume_from=resume_from)
        run_dir = write_workflow_run(workflow)
        append_workflow_csv_rows(workflow, device_type=device_type, definition=build_launcher_workflow(config))
        if workflow.status != "success":
            raise RuntimeError(f"Workflow single falhou. Manifesto: {run_dir}")


class GridCommand(Command):
    """Executa grid search, opcionalmente seguido de Skyband."""

    def execute(self, args: argparse.Namespace, sla_dict: dict) -> None:
        run_grid_search_experiments(
            base_config_path=args.config,
            grid_config_path=args.grid_config,
            parallel=args.parallel,
            resume=args.resume,
            sla_profile_name=args.sla_profile,
            sla_constraints=sla_dict or None,
            train_dataset=args.train_dataset,
            dataset_overrides=_build_dataset_overrides(args),
            gpu_ids=args.gpu,
            tpu_cores=args.tpu_cores,
            precision=args.precision,
        )
        if not args.no_skyband:
            run_skyband_analysis(
                k=args.skyband_k,
                sla_constraints=sla_dict or None,
                sla_profile_name=args.sla_profile,
                metrics=args.skyband_metrics,
                compare=args.skyband_compare,
                state_file=args.skyband_state,
            )


class BertPliWorkflowCommand(Command):
    """Executa o pipeline BERT-PLI composto por tarefas rastreáveis."""

    def execute(self, args: argparse.Namespace, sla_dict: dict) -> None:
        del sla_dict
        gpu = ",".join(map(str, args.gpu)) if args.gpu else None
        config = BertPliWorkflowConfig(gpu=gpu)
        definition = build_bertpli_workflow(config)
        commands: list[list[str]] = []
        task_functions = build_bertpli_task_functions(
            config,
            command_runner=commands.append if args.workflow_dry_run else None,
        )
        if args.workflow_dry_run:
            task_functions = {
                **task_functions,
                "evaluate_retrieval": lambda: {
                    "metrics": {"dry_run": True},
                    "artifacts": {"metrics": config.metrics_result},
                },
            }
        monitoring = load_config(config.bert_config).getboolean("monitoring", "enable_monitoring", fallback=False)
        workflow = SequentialWorkflowExecutor(
            task_functions, telemetry=TaskTelemetryCollector(enable_emissions=monitoring)
        ).execute(definition)
        run_dir = write_workflow_run(workflow)
        append_workflow_csv_rows(workflow, device_type=get_torch_device()["type"], definition=definition)
        if args.workflow_dry_run:
            print(f"Workflow BERT-PLI validado sem treinamento: {run_dir}")
            for command in commands:
                print(" ".join(command))
        elif workflow.status != "success":
            raise RuntimeError(f"Workflow BERT-PLI falhou. Manifesto: {run_dir}")


class GenericWorkflowCommand(Command):
    """Executa pipelines ML, DL, NLP ou LLM definidos por JSON."""

    def execute(self, args: argparse.Namespace, sla_dict: dict) -> None:
        del sla_dict
        if not args.workflow_spec:
            raise ValueError("--workflow-spec e obrigatorio para --workflow generic.")
        spec = load_generic_workflow_spec(Path(args.workflow_spec))
        commands: list[list[str]] = []
        functions = build_generic_task_functions(
            spec, command_runner=commands.append if args.workflow_dry_run else None
        )
        definition = build_generic_workflow(spec)
        workflow = SequentialWorkflowExecutor(
            functions,
            telemetry=TaskTelemetryCollector(
                enable_emissions=spec.enable_emissions,
                environment_cost_per_hour_usd=spec.environment_cost_per_hour_usd,
            ),
        ).execute(definition)
        run_dir = write_workflow_run(workflow)
        append_workflow_csv_rows(workflow, device_type=get_torch_device()["type"], definition=definition)
        if args.workflow_dry_run:
            print(f"Workflow generico validado sem executar comandos: {run_dir}")
            for command in commands:
                print(" ".join(command))
        elif workflow.status != "success":
            raise RuntimeError(f"Workflow generico falhou. Manifesto: {run_dir}")


def _resolve_command(args: argparse.Namespace) -> Command:
    """Mapeia os argumentos do CLI para o Command concreto adequado."""
    if args.workflow == "bertpli":
        return BertPliWorkflowCommand()
    if args.workflow == "generic":
        return GenericWorkflowCommand()
    if args.skyband_only:
        return SkybandOnlyCommand()
    if args.mode == "single":
        return SingleCommand()
    if args.mode == "grid":
        return GridCommand()
    raise ValueError(f"Modo inválido: {args.mode}")
