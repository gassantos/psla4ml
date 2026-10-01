"""
Motor de execução de experimentos BERT-PLI
============================================

Executa um experimento rastreável de ponta a ponta:
- Carrega e aplica a configuração em cascata (default → experimento)
- Mede tempo de execução, uso de RAM e emissões de CO₂ (via codecarbon)
- Captura stdout do loop de treino sem bloquear a saída em tempo real
- Persiste artefatos em JSON (por execução) e CSV (histórico acumulado)

Autor: Gustavo Alexandre
"""

import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from warnings import filterwarnings

import psutil  # type: ignore
import torch
from codecarbon import EmissionsTracker  # type: ignore

from gridsearch.protocols import (
    ComputeMetricsFn,
    ConvertTestResultsFn,
    InitFn,
    TrainFn,
)
from tools.eval_tool import (
    compute_metrics as _default_compute_metrics,
)

# Imports concretos de tools/ — usados como defaults quando nenhum
# callable alternativo é injetado via parâmetros (DIP).
from tools.eval_tool import (
    convert_test_results_to_task1 as _default_convert_test_results,
)
from utils.device import get_torch_device
from utils.util import print_system_info

from .bootstrap_lock import BootstrapLock
from .evaluation import extract_eval_metrics
from .helpers import (
    TeeStream,
    compute_cost_usd,
    estimate_bert_flops,
    load_config,
    now_iso,
)
from .persistence import (
    append_workflow_csv_rows,
    build_result_dict,
    write_json_result,
    write_workflow_run,
)
from .tpu_check import check_tpu_acceleration
from .workflow import legacy_task_run

filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


def execute_experiment(
    config_path: str,
    gpu_list: list[int] | None = None,
    parallel_workers: int = 1,
    train_file: str | None = None,
    dataset_overrides: dict | None = None,
    environment_overrides: dict | None = None,
    *,
    environment_cost_per_hour_usd: float | None = None,
    init_fn: InitFn | None = None,
    train_fn: TrainFn | None = None,
    convert_results_fn: ConvertTestResultsFn | None = None,
    compute_metrics_fn: ComputeMetricsFn | None = None,
    xla_rank: int | None = None,
    xla_world_size: int = 1,
    collect_resource_telemetry: bool = True,
) -> dict | None:
    """Executa um experimento completo de forma rastreável.

    Realiza treino in-process com captura de stdout, amostragem contínua
    de RAM, rastreamento de emissões de CO₂ (opcional) e persistência
    de artefatos em JSON e CSV.

    Args:
        config_path: Caminho para o arquivo ``.config`` do experimento.
        gpu_list: Lista de IDs de GPU a utilizar.
        parallel_workers: Número de workers paralelos.
        train_file: Nome do arquivo de treino sem extensão.
        dataset_overrides: Chaves da seção ``[data]`` a sobrescrever.
        environment_overrides: Chaves da seção ``[environment]`` a sobrescrever.
        environment_cost_per_hour_usd: Custo horário do ambiente de nuvem
            (ex: CPU=$0.10/h, GPU=$1.20/h, TPU=$1.50/h). Quando fornecido,
            usa a fórmula PSLA4ML: ``cost_usd = (train_time_sec/3600) × rate``.
            ``None`` mantém o cálculo por tarifa flat de energia.
        init_fn: Callable compatível com ``InitFn`` protocol.
        train_fn: Callable compatível com ``TrainFn`` protocol.
        convert_results_fn: Callable compatível com ``ConvertTestResultsFn``.
        compute_metrics_fn: Callable compatível com ``ComputeMetricsFn``.
        xla_rank: Rank PJRT do worker atual; ``None`` fora do launcher XLA.
        xla_world_size: Quantidade de workers PJRT usados pelo experimento.
        collect_resource_telemetry: Quando ``False``, delega a coleta ao
            ``TaskTelemetryCollector`` da tarefa que encapsula o runner.
    """
    import tempfile as _tempfile

    # Coleta informações do sistema
    print_system_info()
    _torch_device_info = get_torch_device()
    device_name = _torch_device_info['name']

    # Aplica overrides criando um config temporário por worker.
    _temp_config_path: str | None = None
    if train_file is not None or dataset_overrides or environment_overrides:
        _base_cfg = load_config(config_path)
        if not _base_cfg.has_section("data"):
            _base_cfg.add_section("data")
        if train_file is not None:
            _base_cfg.set("data", "train_file_list", f"{train_file}.json")
        if dataset_overrides:
            for key, value in dataset_overrides.items():
                section = "data"
                _base_cfg.set(section, key, str(value))
        if environment_overrides:
            if not _base_cfg.has_section("environment"):
                _base_cfg.add_section("environment")
            for key, value in environment_overrides.items():
                _base_cfg.set("environment", key, str(value))
        _fd, _temp_config_path = _tempfile.mkstemp(suffix=".config")
        os.close(_fd)
        with open(_temp_config_path, "w") as _f:
            _base_cfg.write(_f)
        config_path = _temp_config_path

    cfg = load_config(config_path)

    exp = cfg["experiment"]
    train = cfg["train"]
    env = cfg["environment"]
    mon = cfg["monitoring"]

    # Nome do dataset de treino utilizado (para rastreabilidade)
    _train_dataset_name = (
        cfg.get("data", "train_file_list", fallback="").replace(".json", "")
        or "train_task2"
    )

    experiment_id = str(uuid.uuid4())
    device_type = _torch_device_info['type']
    is_primary_process = xla_rank in (None, 0)

    # Sincroniza o CUDA para garantir que as medições de tempo sejam precisas
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    start_iso = now_iso()
    DATE_EXEC = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    # -------- ENERGY TRACKER --------
    tracker = None
    if (
        collect_resource_telemetry
        and is_primary_process
        and EmissionsTracker
        and mon.getboolean("enable_monitoring")
    ):
        from .helpers import METRICS_DIR
        tracker = EmissionsTracker(
            project_name=exp["name"],
            output_dir=METRICS_DIR.as_posix(),
            log_level="error",
            output_file=f"EmissionsCO2_{device_type}_{datetime.now().astimezone().strftime('%Y%m%d')}.csv"
        )
        tracker.start()

    # -------- EXEC TRAIN (IN-PROCESS) --------
    from utils.config import create_config
    if init_fn is None:
        from tools.init_tool import init_all
        init_fn = init_all
    if train_fn is None:
        from tools.train_tool import train as run_train
        train_fn = run_train

    # Resolve defaults de avaliação
    convert_test_results_to_task1 = convert_results_fn or _default_convert_test_results
    compute_metrics = compute_metrics_fn or _default_compute_metrics

    if gpu_list is None:
        gpu_list = [0] if torch.cuda.is_available() else []

    tee = TeeStream(sys.stdout)
    sys.stdout = tee

    # Amostragem de RAM em thread daemon (1 amostra/segundo).
    ram_samples: list = []
    _stop_ram = threading.Event()

    def _sample_ram() -> None:
        proc = psutil.Process(os.getpid())
        while not _stop_ram.is_set():
            try:
                ram_samples.append(proc.memory_info().rss / (1024 ** 2))
            except psutil.NoSuchProcess:
                break
            _stop_ram.wait(timeout=1.0)

    ram_thread = threading.Thread(target=_sample_ram, daemon=True) if collect_resource_telemetry else None
    if ram_thread:
        ram_thread.start()

    status = "failed"
    stdout = ""
    stderr = ""
    output_lines: list = []
    _failure_stage: str | None = None

    try:
        _failure_stage = "create_config"
        config = create_config(config_path)
        _failure_stage = "init"
        with BootstrapLock():
            parameters = init_fn(config, gpu_list, None, "train")
        _failure_stage = "train"
        train_fn(parameters, config, gpu_list)
        status = "success"
        _failure_stage = None
    except Exception as exc:
        logger.error("Treinamento falhou em %s: %s", _failure_stage, exc, exc_info=True)  # noqa: G201
        stderr = str(exc)
    finally:
        if ram_thread:
            _stop_ram.set()
            ram_thread.join(timeout=5)
        sys.stdout = tee.original
        output_lines = tee.lines
        stdout = "".join(output_lines)
        # Remove config temporário se foi criado
        if _temp_config_path and os.path.exists(_temp_config_path):
            try:
                os.unlink(_temp_config_path)
            except OSError:
                pass

    if xla_world_size > 1:
        from utils.device import xm

        if xm is None:
            raise RuntimeError("Worker PJRT iniciado sem runtime torch_xla disponível.")
        xm.rendezvous("bl08_experiment_complete")
        if not is_primary_process:
            return None

    # -------- STOP ENERGY TRACKER --------
    emissions_kg = None
    energy_kwh = None

    if tracker:
        emissions_kg = tracker.stop()
        try:
            if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data is not None:
                energy_kwh = tracker.final_emissions_data.energy_consumed
            elif hasattr(tracker, '_total_energy') and tracker._total_energy is not None:
                energy_kwh = tracker._total_energy.kWh
        except Exception as e:  # noqa: BLE001
            logger.warning("Não foi possível obter energy_kwh do tracker: %s", e)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    exec_time = time.perf_counter() - start_time
    end_iso = now_iso()

    # -------- TPU ACCELERATION CHECK (BL-08) --------
    # Usa métricas nativas do runtime XLA; CodeCarbon permanece como telemetria.
    tpu_check = check_tpu_acceleration(
        device_type=device_type,
        tracker=tracker,
        exec_time_sec=exec_time,
    )

    # -------- METRICS (proxy / external) --------
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else None
    peak_ram = max(ram_samples) if ram_samples else None

    profiling_path = Path(cfg.get("output", "model_path")) / cfg.get("output", "model_name") / "profiling_metrics.json"
    total_gflops = 0
    avg_gflops_per_batch = 0

    if profiling_path.exists():
        try:
            with open(profiling_path, "r") as f:
                profiling_data = json.load(f)
                total_gflops = profiling_data.get("total_gflops", 0)
                avg_gflops_per_batch = profiling_data.get("avg_gflops_per_batch", 0)
                logger.info(f"Loaded profiling metrics: {avg_gflops_per_batch:.2f} GFLOPs/batch")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not load profiling metrics: {e}")
            total_gflops = estimate_bert_flops(seq_len=256) # type: ignore
    else:
        logger.warning("Profiling metrics not found, using estimation")
        total_gflops = estimate_bert_flops(seq_len=256) # type: ignore

    # -------- EVAL METRICS --------
    eval_metrics = extract_eval_metrics(
        status=status,
        output_lines=output_lines,
        cfg=cfg,
        config_path=config_path,
        convert_test_results_fn=convert_test_results_to_task1,
        compute_metrics_fn=compute_metrics,
    )

    # -------- JSON OUTPUT --------
    exp_id = exp["name"]
    optmzr = train["optimizer"]
    lr = f"lr{train['learning_rate']}".replace('-', '')
    bs = f"bs{train['batch_size']}"
    ep = f"ep{train['epoch']}"
    json_filename = f"{exp_id}_{optmzr}_{lr}_{bs}_{ep}_{DATE_EXEC}.json"

    cost_usd = compute_cost_usd(
        energy_kwh=energy_kwh,
        train_time_sec=exec_time,
        environment_cost_per_hour_usd=environment_cost_per_hour_usd,
    )

    result = build_result_dict(
        experiment_id=experiment_id,
        json_filename=json_filename,
        seed=int(exp["seed"]),
        status=status,
        date_exec=DATE_EXEC,
        start_iso=start_iso,
        end_iso=end_iso,
        device_type=device_type,
        device_name=device_name,
        precision=env["precision"],
        parallel_workers=parallel_workers,
        xla_world_size=xla_world_size,
        train_dataset_name=_train_dataset_name,
        optimizer=train["optimizer"],
        learning_rate=float(train["learning_rate"]),
        avg_gflops_per_batch=avg_gflops_per_batch,
        batch_size=int(train["batch_size"]),
        epoch=int(train["epoch"]),
        exec_time=exec_time,
        energy_kwh=energy_kwh,
        emissions_kg=emissions_kg,
        cost_usd=cost_usd,
        avg_ram=avg_ram,
        peak_ram=peak_ram,
        total_gflops=total_gflops,
        eval_metrics=eval_metrics,
        stdout=stdout,
        stderr=stderr,
        tpu_check=tpu_check,
        failure_stage=_failure_stage,
    )
    if not collect_resource_telemetry:
        result["resources"] = {
            "total_gflops": total_gflops,
            "avg_gflops_per_batch": avg_gflops_per_batch,
        }
        return result

    workflow = legacy_task_run(result)
    result["workflow"] = workflow.to_dict()

    write_json_result(result, json_filename)
    write_workflow_run(workflow)

    append_workflow_csv_rows(
        workflow,
        device_type=device_type,
        context={
            "experiment": result["experiment"],
            "environment": result["environment"],
            "execution": result["execution"],
            "hyperparameters": result["hyperparameters"],
        },
    )

    print(f"[OK] Wrapper finalizou em {exec_time:.2f} segundos - {exp['name']} ({status})")
    return result
