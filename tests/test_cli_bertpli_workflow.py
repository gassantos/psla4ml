"""Integracao dos workflows declarativos com o CLI."""

import json

import pytest

import cli.commands as commands_module
from cli.commands import (
    BertPliWorkflowCommand,
    GenericWorkflowCommand,
    SingleCommand,
    _resolve_command,
)
from cli.parser import build_argument_parser
from cli.runners import run_single_experiment


@pytest.fixture(autouse=True)
def prewarm_calls(monkeypatch):
    """Evita carregar modelo/dataset reais e registra os prewarms do modo single."""
    calls = []

    def fake_prewarm(base_config_path, dataset_overrides=None, *, strict=True):
        calls.append({"config": base_config_path, "overrides": dataset_overrides, "strict": strict})

    monkeypatch.setattr("cli.commands.prewarm_huggingface", fake_prewarm)
    monkeypatch.setattr("cli.runners.prewarm_huggingface", fake_prewarm)
    monkeypatch.setattr(
        "cli.runners._enable_hf_offline_mode",
        lambda: pytest.fail("o modo single n\u00e3o deve ligar HF_HUB_OFFLINE"),
    )
    return calls


def test_parser_resolves_bertpli_workflow_command():
    args = build_argument_parser().parse_args(["--workflow", "bertpli", "--workflow-dry-run"])

    assert args.workflow_dry_run is True
    assert isinstance(_resolve_command(args), BertPliWorkflowCommand)


def test_parser_accepts_huggingface_workflow_cache_and_resume_options():
    args = build_argument_parser().parse_args([
        "--mode", "single", "--dataset-source", "hub", "--dataset-id", "org/data",
        "--workflow-resume-run", "output/experiments/workflow_runs/run-1",
        "--workflow-cache-dir", "output/experiments/workflow_cache",
    ])

    assert args.workflow_resume_run.endswith("workflow_runs/run-1")
    assert args.workflow_cache_dir.endswith("workflow_cache")


def test_bertpli_dry_run_persists_workflow(monkeypatch, tmp_path, capsys):
    workflows = []
    monkeypatch.setattr(
        "cli.commands.write_workflow_run",
        lambda workflow: workflows.append(workflow) or tmp_path / workflow.experiment_run_id,
    )
    args = build_argument_parser().parse_args(["--workflow", "bertpli", "--workflow-dry-run"])

    BertPliWorkflowCommand().execute(args, {})

    assert workflows[0].status == "success"
    assert "Workflow BERT-PLI validado sem treinamento" in capsys.readouterr().out


def test_generic_workflow_dry_run_persists_multidomain_spec(monkeypatch, tmp_path, capsys):
    spec_path = tmp_path / "workflow.json"
    spec_path.write_text(json.dumps({
        "name": "classic", "experiment_type": "ml_classic",
        "tasks": [
            {"task_id": "ingest", "name": "Ingerir", "command": ["python", "ingest.py"], "activity": "ingestion"},
            {"task_id": "train", "name": "Treinar", "command": ["python", "train.py"], "activity": "adaptation"},
            {"task_id": "evaluate", "name": "Avaliar", "command": ["python", "evaluate.py"], "activity": "evaluation_monitoring"},
        ],
    }), encoding="utf-8")
    workflows = []
    monkeypatch.setattr(
        "cli.commands.write_workflow_run",
        lambda workflow: workflows.append(workflow) or tmp_path / workflow.experiment_run_id,
    )
    args = build_argument_parser().parse_args(
        ["--workflow", "generic", "--workflow-spec", str(spec_path), "--workflow-dry-run"]
    )

    assert isinstance(_resolve_command(args), GenericWorkflowCommand)
    GenericWorkflowCommand().execute(args, {})

    assert workflows[0].status == "success"
    assert "Workflow generico validado" in capsys.readouterr().out


def test_generic_workflow_requires_specification_file():
    args = build_argument_parser().parse_args(["--workflow", "generic"])

    with pytest.raises(ValueError, match="workflow-spec"):
        GenericWorkflowCommand().execute(args, {})


def test_single_huggingface_executes_and_persists_workflow(monkeypatch, tmp_path):
    workflows = []
    monkeypatch.setattr(
        "cli.commands.write_workflow_run",
        lambda workflow: workflows.append(workflow) or tmp_path / workflow.experiment_run_id,
    )
    monkeypatch.setattr("cli.commands.load_config", lambda _path: _MonitoringConfig())
    monkeypatch.setattr("cli.commands.get_torch_device", lambda: {"type": "GPU"})
    monkeypatch.setattr(
        "experiment.workflow_templates._build_huggingface_dataset_probe",
        lambda *_args, **_kwargs: lambda: {"records": 1, "fields": ["guid"], "source": "hub"},
    )
    monkeypatch.setattr(
        "experiment.workflow_templates._launch_experiment",
        lambda **_kwargs: {
            "experiment": {"status": "success"},
            "resources": {"train_time_sec": 1.0},
            "evaluation": {"f1_score": 0.9},
        },
    )
    args = build_argument_parser().parse_args([
        "--mode", "single", "--no-skyband",
        "--dataset-source", "hub", "--dataset-id", "nyu-mll/glue", "--dataset-config", "mrpc",
    ])

    assert isinstance(_resolve_command(args), SingleCommand)
    SingleCommand().execute(args, {})

    assert workflows[0].status == "success"
    assert [task.task_id for task in workflows[0].tasks] == [
        "ingest_dataset", "adapt_model", "evaluate_model",
    ]
    assert workflows[0].tasks[1].config["config_path"] == args.config


def test_single_huggingface_requires_dataset_id_for_hub():
    args = build_argument_parser().parse_args([
        "--mode", "single", "--no-skyband", "--dataset-source", "hub",
    ])

    with pytest.raises(ValueError, match="dataset-id"):
        SingleCommand().execute(args, {})


def test_single_local_executes_and_persists_workflow(monkeypatch, tmp_path):
    workflows = []
    monkeypatch.setattr(
        "cli.commands.write_workflow_run",
        lambda workflow: workflows.append(workflow) or tmp_path / workflow.experiment_run_id,
    )
    monkeypatch.setattr("cli.commands.load_config", lambda _path: _MonitoringConfig())
    monkeypatch.setattr("cli.commands.get_torch_device", lambda: {"type": "CPU"})
    monkeypatch.setattr(
        "experiment.workflow_templates._launch_experiment",
        lambda **kwargs: {
            "experiment": {"status": "success"}, "resources": {}, "evaluation": {},
        },
    )
    args = build_argument_parser().parse_args(["--mode", "single", "--no-skyband"])

    SingleCommand().execute(args, {})

    assert workflows[0].status == "success"
    assert [task.task_id for task in workflows[0].tasks] == [
        "ingest_dataset", "adapt_model", "evaluate_model",
    ]


def test_programmatic_single_executes_workflow(monkeypatch, tmp_path):
    workflows = []
    monkeypatch.setattr(
        "experiment.persistence.write_workflow_run",
        lambda workflow: workflows.append(workflow) or tmp_path,
    )
    monkeypatch.setattr("cli.runners.validate_paths", lambda _path: True)
    monkeypatch.setattr(
        "experiment.workflow_templates._launch_experiment",
        lambda **_kwargs: {"experiment": {"status": "success"}, "resources": {}, "evaluation": {}},
    )

    result = run_single_experiment("ignored.config")

    assert result.status == "success"
    assert [task.task_id for task in workflows[0].tasks] == [
        "ingest_dataset", "adapt_model", "evaluate_model",
    ]


def test_single_huggingface_declares_tpu_only_when_detected(monkeypatch, tmp_path):
    workflows = []
    definitions = []
    monkeypatch.setattr(
        "cli.commands.write_workflow_run",
        lambda workflow: workflows.append(workflow) or tmp_path / workflow.experiment_run_id,
    )
    monkeypatch.setattr("cli.commands.load_config", lambda _path: _MonitoringConfig())
    monkeypatch.setattr("cli.commands.get_torch_device", lambda: {"type": "TPU"})
    original_build = commands_module.build_huggingface_workflow
    monkeypatch.setattr(
        "cli.commands.build_huggingface_workflow",
        lambda config: definitions.append(original_build(config)) or definitions[-1],
    )
    monkeypatch.setattr(
        "experiment.workflow_templates._build_huggingface_dataset_probe",
        lambda *_args, **_kwargs: lambda: {"records": 1, "fields": [], "source": "local_json"},
    )
    monkeypatch.setattr(
        "experiment.workflow_templates._launch_experiment",
        lambda **_kwargs: {"experiment": {"status": "success"}, "resources": {}, "evaluation": {}},
    )
    args = build_argument_parser().parse_args([
        "--mode", "single", "--no-skyband", "--dataset-source", "local_json", "--tpu-cores", "8",
    ])

    SingleCommand().execute(args, {})

    resources = definitions[0].tasks[1].resources
    assert resources.gpu_count == 0
    assert resources.tpu_cores == 8
    assert resources.coupling_degree == 0.9


class _MonitoringConfig:
    @staticmethod
    def getboolean(*_args, **_kwargs):
        return False

def test_single_huggingface_prewarms_cache_before_measuring(monkeypatch, tmp_path, prewarm_calls):
    order = []
    monkeypatch.setattr("cli.commands.write_workflow_run", lambda wf: tmp_path / wf.experiment_run_id)
    monkeypatch.setattr("cli.commands.load_config", lambda _path: _MonitoringConfig())
    monkeypatch.setattr("cli.commands.get_torch_device", lambda: {"type": "CPU"})
    monkeypatch.setattr(
        "cli.commands.prewarm_huggingface", lambda *a, **k: order.append("prewarm"),
    )
    monkeypatch.setattr(
        "experiment.workflow_templates._build_huggingface_dataset_probe",
        lambda *_a, **_k: lambda: order.append("measured_ingest") or {"records": 1, "fields": [], "source": "hub"},
    )
    monkeypatch.setattr(
        "experiment.workflow_templates._launch_experiment",
        lambda **_k: {"experiment": {"status": "success"}, "resources": {}, "evaluation": {}},
    )
    args = build_argument_parser().parse_args([
        "--mode", "single", "--no-skyband",
        "--dataset-source", "hub", "--dataset-id", "nyu-mll/glue", "--dataset-config", "mrpc",
    ])

    SingleCommand().execute(args, {})

    assert order == ["prewarm", "measured_ingest"]


def test_single_prewarm_is_not_strict_and_receives_dataset_overrides(monkeypatch, tmp_path, prewarm_calls):
    monkeypatch.setattr("cli.commands.write_workflow_run", lambda wf: tmp_path / wf.experiment_run_id)
    monkeypatch.setattr("cli.commands.load_config", lambda _path: _MonitoringConfig())
    monkeypatch.setattr("cli.commands.get_torch_device", lambda: {"type": "CPU"})
    monkeypatch.setattr(
        "experiment.workflow_templates._build_huggingface_dataset_probe",
        lambda *_a, **_k: lambda: {"records": 1, "fields": [], "source": "hub"},
    )
    monkeypatch.setattr(
        "experiment.workflow_templates._launch_experiment",
        lambda **_k: {"experiment": {"status": "success"}, "resources": {}, "evaluation": {}},
    )
    prewarm_calls.clear()
    args = build_argument_parser().parse_args([
        "--mode", "single", "--no-skyband",
        "--dataset-source", "hub", "--dataset-id", "nyu-mll/glue", "--dataset-config", "mrpc",
    ])

    SingleCommand().execute(args, {})

    (call,) = prewarm_calls
    assert call["strict"] is False
    assert call["overrides"]["hf_dataset_id"] == "nyu-mll/glue"


def test_single_run_context_reads_hyperparameters_through_param_mapping(monkeypatch):
    from cli.runners import single_run_context

    monkeypatch.setenv("PSLA4ML_MACHINE_NAME", "host-1")

    context = single_run_context("config/experiments/BertPLI.config")

    assert context["parallel_workers"] == 1
    assert context["resource_machine_name"] == "host-1"
    expected = {
        "learning_rate": "2e-5", "batch_size": "16", "optimizer": "adamw", "dropout": "0.1",
        "seed": "42", "num_epochs": "4", "max_seq_length": "256",
    }
    assert expected.items() <= context["grid_params"].items()


def test_single_run_context_columns_all_exist_in_the_summary():
    from cli.runners import single_run_context
    from experiment import persistence

    context = single_run_context("config/experiments/BertPLI.config")

    unknown = (set(context) - set(persistence.CONTEXT_COLUMNS) - {"grid_params"}) | (
        set(context["grid_params"]) - set(persistence.GRID_PARAM_NAMES)
    )
    assert not unknown


def test_single_workflow_writes_environment_and_hyperparameters_to_the_summary(monkeypatch, tmp_path):
    import csv

    from experiment import persistence

    monkeypatch.setattr(persistence, "METRICS_DIR", tmp_path)
    monkeypatch.setenv("PSLA4ML_MACHINE_NAME", "host-9")
    monkeypatch.setattr("cli.commands.write_workflow_run", lambda wf: tmp_path / wf.experiment_run_id)
    monkeypatch.setattr("cli.commands.load_config", lambda _path: _MonitoringConfig())
    monkeypatch.setattr("cli.commands.get_torch_device", lambda: {"type": "CPU"})
    monkeypatch.setattr(
        "experiment.workflow_templates._launch_experiment",
        lambda **_k: {"experiment": {"status": "success"}, "resources": {}, "evaluation": {}},
    )
    args = build_argument_parser().parse_args(["--mode", "single", "--no-skyband"])

    SingleCommand().execute(args, {})

    with open(next(tmp_path.glob("experiment_summary_*.csv")), newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows and all(row["resource_machine_name"] == "host-9" for row in rows)
    assert all(row["configured_environment"] for row in rows)
    assert all(row["parallel_workers"] == "1" for row in rows)
    assert all(row["grid_batch_size"] == "16" and row["grid_optimizer"] == "adamw" for row in rows)
