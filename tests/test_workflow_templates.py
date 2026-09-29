"""Testes das definicoes basicas de workflow por dominio."""

import pytest

from experiment.aggregation import aggregate_workflow_run
from experiment.task_cache import TaskCache
from experiment.workflow import ArtifactKind, ResourceRequirements, TaskActivity
from experiment.workflow_planner import WorkflowPlanner
from experiment.task_telemetry import TaskTelemetryCollector
from experiment.workflow_templates import (
    DOMAIN_WORKFLOW_PROFILES,
    HuggingFaceWorkflowConfig,
    build_domain_workflow,
    build_huggingface_task_functions,
    build_huggingface_workflow,
)
from experiment.task_executor import SequentialWorkflowExecutor


@pytest.mark.parametrize(
    ("experiment_type", "task_ids"),
    [
        ("ml_classic", ["ingest_data", "prepare_features", "train_model", "evaluate_model"]),
        ("deep_learning", ["ingest_data", "prepare_data", "train_model", "validate_model", "evaluate_model"]),
        ("nlp", ["ingest_data", "preprocess_text", "train_model", "evaluate_model"]),
        ("llm", ["ingest_data", "prepare_corpus", "adapt_model", "evaluate_model", "publish_model"]),
    ],
)
def test_domain_workflows_define_canonical_task_lifecycle(experiment_type, task_ids):
    workflow = build_domain_workflow("experiment", experiment_type)

    assert workflow.experiment_type == experiment_type
    assert [task.task_id for task in WorkflowPlanner().plan(workflow)] == task_ids


@pytest.mark.parametrize(
    ("experiment_type", "framework"),
    [
        ("ml_classic", "scikit-learn"),
        ("deep_learning", "pytorch"),
        ("nlp", "spacy"),
        ("llm", "huggingface"),
    ],
)
def test_domain_profiles_declare_target_framework_and_workflow_semantics(experiment_type, framework):
    profile = DOMAIN_WORKFLOW_PROFILES[experiment_type]
    workflow = build_domain_workflow("experiment", experiment_type)
    tasks = {task.task_id: task for task in workflow.tasks}

    assert profile.framework == framework
    assert all(task.activity is not TaskActivity.CUSTOM for task in workflow.tasks)
    if experiment_type in {"deep_learning", "llm"}:
        train_task = tasks["train_model"] if experiment_type == "deep_learning" else tasks["adapt_model"]
        assert train_task.resources.gpu_count == 1
        assert train_task.resources.coupling_degree == 0.9


def test_domain_workflow_keeps_model_and_dataset_customization_in_task_profiles():
    workflow = build_domain_workflow(
        "hf-nlp", "nlp",
        task_configs={
            "ingest_data": {"dataset": "nyu-mll/glue", "subset": "mrpc"},
            "train_model": {"model": "bert-base-uncased", "epochs": 3},
        },
        task_input_signatures={"ingest_data": {"dataset": "glue-mrpc-v1"}},
    )
    tasks = {task.task_id: task for task in workflow.tasks}

    assert tasks["ingest_data"].config["dataset"] == "nyu-mll/glue"
    assert tasks["ingest_data"].input_signatures == {"dataset": "glue-mrpc-v1"}
    assert tasks["train_model"].config["model"] == "bert-base-uncased"
    assert tasks["evaluate_model"].depends_on == ("train_model",)


def test_domain_workflow_rejects_unsupported_type():
    with pytest.raises(ValueError, match="experiment_type"):
        build_domain_workflow("invalid", "computer_vision")


def test_huggingface_workflow_connects_t0_t2_t5_by_versioned_artifacts():
    workflow = build_huggingface_workflow(HuggingFaceWorkflowConfig(
        name="hf-mrpc",
        dataset_source="hub",
        dataset_id="nyu-mll/glue",
        dataset_config="mrpc",
        dataset_version="main",
        model_version="train-42",
        metrics_version="eval-42",
        adaptation_parameters={"learning_rate": 2e-5, "seed": 42},
        resources=ResourceRequirements(gpu_count=1, coupling_degree=0.9),
    ))

    tasks = {task.task_id: task for task in workflow.tasks}
    plan = WorkflowPlanner().plan(workflow)

    assert workflow.experiment_type == "llm"
    assert [task.task_id for task in plan] == ["ingest_dataset", "adapt_model", "evaluate_model"]
    assert tasks["ingest_dataset"].activity is TaskActivity.INGESTION
    assert tasks["ingest_dataset"].outputs[0].kind is ArtifactKind.DATA
    assert tasks["ingest_dataset"].outputs[0].uri == "hf://datasets/nyu-mll/glue"
    assert tasks["ingest_dataset"].outputs[0].metadata["dataset_config"] == "mrpc"
    assert tasks["adapt_model"].depends_on == ()
    assert tasks["adapt_model"].inputs == tasks["ingest_dataset"].outputs
    assert tasks["adapt_model"].outputs[0].kind is ArtifactKind.MODEL
    assert tasks["adapt_model"].resources.coupling_degree == 0.9
    assert tasks["evaluate_model"].inputs == tasks["adapt_model"].outputs
    assert tasks["evaluate_model"].outputs[0].version == "eval-42"


def test_huggingface_workflow_rejects_unsupported_dataset_source():
    with pytest.raises(ValueError, match="dataset_source"):
        HuggingFaceWorkflowConfig(name="invalid", dataset_source="filesystem")


def test_huggingface_task_adapters_execute_t0_t2_t5_and_project_legacy_result():
    workflow_config = HuggingFaceWorkflowConfig(
        name="hf-mrpc",
        dataset_source="hub",
        dataset_id="nyu-mll/glue",
        dataset_config="mrpc",
        resources=ResourceRequirements(gpu_count=1, coupling_degree=0.9),
    )
    launch_calls = []

    def probe():
        return {"records": 42, "fields": ["guid", "label", "text_a", "text_b"], "source": "hub"}

    def launcher(**kwargs):
        launch_calls.append(kwargs)
        return {
            "experiment": {"status": "success"},
            "resources": {"total_gflops": 42.0},
            "evaluation": {"f1_score": 0.9, "accuracy": 0.8},
        }

    workflow = build_huggingface_workflow(workflow_config)
    tracker = _Tracker()
    result = SequentialWorkflowExecutor(build_huggingface_task_functions(
        workflow_config,
        config_path="ignored.config",
        train_file="train_task2_v3",
        dataset_probe=probe,
        experiment_launcher=launcher,
    ), telemetry=TaskTelemetryCollector(
        enable_emissions=True,
        environment_cost_per_hour_usd=3.6,
        tracker_factory=lambda: tracker,
    )).execute(workflow)

    ingest, adapt, evaluate = result.tasks
    assert result.status == "success"
    assert ingest.attempts[0].metrics["dataset"]["records"] == 42
    assert ingest.attempts[0].metrics["dataset"]["version"] == "input"
    assert ingest.attempts[0].metrics["dataset"]["splits"] == {
        "train": "train", "valid": "validation", "test": "test",
    }
    assert ingest.attempts[0].artifacts["dataset"]["kind"] == "data"
    assert adapt.attempts[0].artifacts["model"]["kind"] == "model"
    assert adapt.attempts[0].artifacts["checkpoint"]["uri"] is None
    assert adapt.attempts[0].metrics["resources"]["total_gflops"] == 42.0
    assert evaluate.attempts[0].metrics["evaluation"]["f1_score"] == 0.9
    assert evaluate.attempts[0].artifacts["metrics"]["artifact_id"] == "metrics-hf-mrpc"
    assert launch_calls[0]["dataset_overrides"] == {
        "hf_dataset_source": "hub",
        "hf_dataset_id": "nyu-mll/glue",
        "hf_dataset_config": "mrpc",
        "train_dataset_type": "HuggingFace",
        "valid_dataset_type": "HuggingFace",
        "test_dataset_type": "HuggingFace",
    }
    assert launch_calls[0]["collect_resource_telemetry"] is False
    summary = aggregate_workflow_run(result, workflow)
    assert summary["resources"]["task_time_sec"] is not None
    assert summary["resources"]["energy_kwh"] == pytest.approx(0.75)
    assert summary["resources"]["peak_ram_mb"] is not None
    assert summary["evaluation"] == {"accuracy": 0.8, "f1_score": 0.9}


def test_huggingface_task_adapters_skip_t5_after_failed_t2():
    workflow_config = HuggingFaceWorkflowConfig(name="hf-failure", dataset_source="hub", dataset_id="org/data")
    workflow = build_huggingface_workflow(workflow_config)
    result = SequentialWorkflowExecutor(build_huggingface_task_functions(
        workflow_config,
        config_path="ignored.config",
        dataset_probe=lambda: {"records": 1, "fields": [], "source": "hub"},
        experiment_launcher=lambda **_kwargs: {"experiment": {"status": "failed"}, "logs": {"stderr_tail": "train failed"}},
    )).execute(workflow)

    assert result.status == "failed"
    assert result.tasks[1].status.value == "failed"
    assert result.tasks[2].status.value == "skipped"


def test_huggingface_task_adapters_skip_t2_and_t5_after_failed_t0():
    workflow_config = HuggingFaceWorkflowConfig(name="hf-ingest-failure", dataset_source="hub", dataset_id="org/data")
    workflow = build_huggingface_workflow(workflow_config)
    result = SequentialWorkflowExecutor(build_huggingface_task_functions(
        workflow_config,
        config_path="ignored.config",
        dataset_probe=lambda: (_ for _ in ()).throw(ValueError("dataset indisponível")),
        experiment_launcher=lambda **_kwargs: pytest.fail("T2 não deve executar"),
    )).execute(workflow)

    assert result.status == "failed"
    assert [task.status.value for task in result.tasks] == ["failed", "skipped", "skipped"]


def test_huggingface_task_adapters_preserve_t0_t2_when_t5_fails():
    workflow_config = HuggingFaceWorkflowConfig(name="hf-eval-failure", dataset_source="hub", dataset_id="org/data")
    workflow = build_huggingface_workflow(workflow_config)
    task_functions = dict(build_huggingface_task_functions(
        workflow_config,
        config_path="ignored.config",
        dataset_probe=lambda: {"records": 1, "fields": [], "source": "hub"},
        experiment_launcher=lambda **_kwargs: {
            "experiment": {"status": "success"}, "resources": {}, "evaluation": {},
        },
    ))
    task_functions["evaluate_model"] = lambda: (_ for _ in ()).throw(RuntimeError("evaluation failed"))

    result = SequentialWorkflowExecutor(task_functions).execute(workflow)

    assert result.status == "failed"
    assert [task.status.value for task in result.tasks] == ["succeeded", "succeeded", "failed"]


def test_huggingface_t2_records_checkpoint_location_from_training_config(tmp_path):
    config_path = tmp_path / "train.config"
    config_path.write_text(
        "[output]\nmodel_path = output/checkpoints\nmodel_name = hf-model\n",
        encoding="utf-8",
    )
    workflow_config = HuggingFaceWorkflowConfig(name="hf-checkpoint", dataset_source="hub", dataset_id="org/data")
    workflow = build_huggingface_workflow(workflow_config)
    result = SequentialWorkflowExecutor(build_huggingface_task_functions(
        workflow_config,
        config_path=str(config_path),
        dataset_probe=lambda: {"records": 1, "fields": [], "source": "hub"},
        experiment_launcher=lambda **_kwargs: {
            "experiment": {"status": "success"}, "resources": {}, "evaluation": {},
        },
    )).execute(workflow)

    assert result.tasks[1].attempts[0].artifacts["checkpoint"]["uri"] == (
        "output/checkpoints/hf-model"
    )


def test_huggingface_cache_rehydrates_t2_result_when_only_t5_signature_changes(tmp_path):
    cache = TaskCache(tmp_path)
    original_config = HuggingFaceWorkflowConfig(
        name="hf-cache", dataset_source="hub", dataset_id="org/data", metrics_version="v1",
        adaptation_parameters={"pretrained_model": "bert-base", "seed": 42},
    )
    original = build_huggingface_workflow(original_config)
    first = SequentialWorkflowExecutor(build_huggingface_task_functions(
        original_config,
        config_path="ignored.config",
        dataset_probe=lambda: {"records": 1, "fields": [], "source": "hub"},
        experiment_launcher=lambda **_kwargs: {
            "experiment": {"status": "success"}, "resources": {},
            "evaluation": {"f1_score": 0.9},
        },
    ), cache=cache, code_version="commit-a").execute(original)

    changed_config = HuggingFaceWorkflowConfig(
        name="hf-cache", dataset_source="hub", dataset_id="org/data", metrics_version="v2",
        adaptation_parameters={"pretrained_model": "bert-base", "seed": 42},
    )
    changed = build_huggingface_workflow(changed_config)
    second = SequentialWorkflowExecutor(build_huggingface_task_functions(
        changed_config,
        config_path="ignored.config",
        dataset_probe=lambda: pytest.fail("T0 não deve executar"),
        experiment_launcher=lambda **_kwargs: pytest.fail("T2 não deve executar"),
    ), cache=cache, code_version="commit-a").execute(changed)

    assert first.status == "success"
    assert [task.status.value for task in second.tasks] == ["cached", "cached", "succeeded"]
    assert second.tasks[2].attempts[0].metrics["evaluation"] == {"f1_score": 0.9}


def test_huggingface_resume_runs_t5_without_rerunning_valid_t0_and_t2():
    workflow_config = HuggingFaceWorkflowConfig(
        name="hf-resume", dataset_source="hub", dataset_id="org/data",
    )
    definition = build_huggingface_workflow(workflow_config)
    tasks = {task.task_id: task for task in definition.tasks}
    from experiment.workflow import ExperimentRun, TaskExecutionAttempt, TaskRun, TaskStatus

    previous = ExperimentRun(
        "resume-hf", definition.name, "failed", [
            TaskRun(
                **_task_run_fields(tasks["ingest_dataset"], TaskStatus.SUCCEEDED),
                attempts=[TaskExecutionAttempt("t0", 1, TaskStatus.SUCCEEDED)],
            ),
            TaskRun(
                **_task_run_fields(tasks["adapt_model"], TaskStatus.SUCCEEDED),
                attempts=[TaskExecutionAttempt(
                    "t2", 1, TaskStatus.SUCCEEDED,
                    metrics={"projected_evaluation": {"accuracy": 0.8}},
                )],
            ),
            TaskRun(
                **_task_run_fields(tasks["evaluate_model"], TaskStatus.FAILED),
                attempts=[TaskExecutionAttempt("t5", 1, TaskStatus.FAILED, error_type="RuntimeError")],
            ),
        ],
    )
    result = SequentialWorkflowExecutor(build_huggingface_task_functions(
        workflow_config,
        config_path="ignored.config",
        dataset_probe=lambda: pytest.fail("T0 não deve executar"),
        experiment_launcher=lambda **_kwargs: pytest.fail("T2 não deve executar"),
    )).execute(definition, resume_from=previous)

    assert result.status == "success"
    assert [task.status.value for task in result.tasks] == ["succeeded", "succeeded", "succeeded"]
    assert result.tasks[2].attempts[-1].metrics["evaluation"] == {"accuracy": 0.8}


def _task_run_fields(task, status):
    return {
        "task_id": task.task_id,
        "name": task.name,
        "task_type": task.task_type,
        "status": status,
        "config": task.config,
        "input_signatures": task.input_signatures,
    }


class _Tracker:
    final_emissions_data = type("Data", (), {"energy_consumed": 0.25})()

    def start(self):
        return None

    def stop(self):
        return 0.05