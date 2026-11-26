"""Testes do launcher PJRT multicore do BL-08."""

from unittest.mock import MagicMock

import pytest


def test_single_core_executes_without_xla_spawn(monkeypatch):
    from experiment import runner, xla_launcher

    execute = MagicMock()
    monkeypatch.setattr(runner, "execute_experiment", execute)

    xla_launcher.launch_experiment(config_path="experiment.config", tpu_cores=1)

    execute.assert_called_once_with(config_path="experiment.config")


def test_multicore_spawns_one_process_per_tpu_core(monkeypatch):
    from experiment import xla_launcher

    xmp = MagicMock()
    monkeypatch.setattr(xla_launcher.importlib, "import_module", lambda _: xmp)

    xla_launcher.launch_experiment(
        config_path="experiment.config",
        parallel_workers=1,
        tpu_cores=8,
    )

    xmp.spawn.assert_called_once_with(
        xla_launcher._run_xla_worker,
        args=(8, {"config_path": "experiment.config", "parallel_workers": 1}),
        nprocs=8,
        start_method="spawn",
    )


def test_multicore_requires_torch_xla(monkeypatch):
    from experiment import xla_launcher

    def fail_import(_):
        raise ImportError("missing torch_xla")

    monkeypatch.setattr(xla_launcher.importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="requer torch_xla"):
        xla_launcher.launch_experiment(config_path="experiment.config", tpu_cores=8)


def test_rejects_invalid_tpu_core_count():
    from experiment.xla_launcher import launch_experiment

    with pytest.raises(ValueError, match="maior ou igual a 0"):
        launch_experiment(config_path="experiment.config", tpu_cores=-1)


def test_cli_accepts_tpu_core_count():
    from cli.parser import build_argument_parser

    args = build_argument_parser().parse_args(["--mode", "single", "--tpu-cores", "8"])

    assert args.tpu_cores == 8


def test_cli_accepts_bf16_precision_override():
    from cli.parser import build_argument_parser

    args = build_argument_parser().parse_args(["--precision", "bf16"])

    assert args.precision == "bf16"


def test_grid_rejects_nested_parallelism(tmp_path):
    from gridsearch.executor import run_grid_search

    with pytest.raises(ValueError, match="parallel=1"):
        run_grid_search(
            base_config_path="unused.config",
            grid_config={},
            parallel=2,
            tpu_cores=8,
            output_dir=tmp_path,
        )


def test_worker_forwards_rank_and_world_size(monkeypatch):
    from experiment import runner, xla_launcher

    execute = MagicMock()
    monkeypatch.setattr(runner, "execute_experiment", execute)

    xla_launcher._run_xla_worker(3, 8, {"config_path": "experiment.config"})

    execute.assert_called_once_with(
        config_path="experiment.config",
        xla_rank=3,
        xla_world_size=8,
    )


def test_result_persists_xla_world_size():
    from experiment.persistence import build_result_dict

    result = build_result_dict(
        experiment_id="id",
        json_filename="result.json",
        seed=42,
        status="success",
        date_exec="20260802",
        start_iso="start",
        end_iso="end",
        device_type="TPU",
        device_name="TPU v5e",
        precision="bf16",
        parallel_workers=1,
        train_dataset_name="train_task2",
        optimizer="adamw",
        learning_rate=1e-5,
        avg_gflops_per_batch=0.0,
        batch_size=8,
        epoch=1,
        exec_time=1.0,
        energy_kwh=None,
        emissions_kg=None,
        cost_usd=None,
        avg_ram=None,
        peak_ram=None,
        total_gflops=0.0,
        eval_metrics={},
        stdout="",
        stderr="",
        xla_world_size=8,
    )

    assert result["execution"]["xla_world_size"] == 8


def test_validation_loss_is_reduced_across_xla_workers(monkeypatch):
    from tools import train_tool

    xla_model = MagicMock()
    xla_model.mesh_reduce.return_value = 1.25
    monkeypatch.setattr(train_tool, "xm", xla_model)
    device = MagicMock(type="xla")

    result = train_tool._distributed_validation_loss(2.0, device, epoch=3)

    assert result == 1.25
    tag, value, reducer = xla_model.mesh_reduce.call_args.args
    assert tag == "validation_loss_3"
    assert value == 2.0
    assert reducer([1.0, 2.0, 3.0]) == 2.0