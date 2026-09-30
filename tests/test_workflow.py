"""Testes dos contratos P0 para workflows orientados a tarefas."""

import pytest

from experiment.task_cache import TaskCache
from experiment.task_executor import (
    ParallelWorkflowExecutor,
    SequentialWorkflowExecutor,
)
from experiment.workflow import (
    INFRA_RETRY_POLICY,
    ArtifactDefinition,
    ArtifactKind,
    ExecutionRegime,
    ExperimentDefinition,
    ExperimentRun,
    ResourceRequirements,
    RetryPolicy,
    TaskActivity,
    TaskDefinition,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
    classify_error_type,
    legacy_task_run,
)
from experiment.workflow_planner import WorkflowPlanner


def _legacy_result(status: str = "success") -> dict:
    return {
        "experiment": {
            "id": "experiment-1",
            "config_name": "result.json",
            "status": status,
            "timestamp_start": "2026-08-29T10:00:00",
            "timestamp_end": "2026-08-29T10:01:00",
        },
        "resources": {"train_time_sec": "60.00"},
        "evaluation": {"f1_score": 0.9},
        "logs": {"stderr_tail": ""},
    }


def test_experiment_definition_requires_a_task():
    with pytest.raises(ValueError, match="ao menos uma tarefa"):
        ExperimentDefinition(name="empty", tasks=())


def test_task_definition_declares_versioned_artifacts_activity_and_resources():
    dataset = ArtifactDefinition("legal-corpus", ArtifactKind.DATA, "v2", "data/legal-v2.jsonl")
    checkpoint = ArtifactDefinition("bert-legal", ArtifactKind.MODEL, "v1", "models/bert-legal-v1")
    task = TaskDefinition(
        "fine_tune", "Ajustar modelo", inputs=(dataset,), outputs=(checkpoint,),
        activity=TaskActivity.ADAPTATION,
        resources=ResourceRequirements(cpu_cores=4, memory_mb=8192, gpu_count=1, coupling_degree=0.9),
        is_composite=True,
        stop_predicate="validation_loss <= 0.1",
    )

    assert task.inputs == (dataset,)
    assert task.outputs == (checkpoint,)
    assert task.activity is TaskActivity.ADAPTATION
    assert task.regime is ExecutionRegime.BUILD
    assert task.resources.coupling_degree == 0.9


@pytest.mark.parametrize("coupling_degree", [-0.01, 1.01])
def test_resource_requirements_reject_invalid_coupling_degree(coupling_degree):
    with pytest.raises(ValueError, match="entre 0 e 1"):
        ResourceRequirements(coupling_degree=coupling_degree)


def test_non_composite_task_rejects_stop_predicate():
    with pytest.raises(ValueError, match="tarefa composta"):
        TaskDefinition("train", "Treinar", stop_predicate="epoch == 3")


def test_task_attempt_rejects_invalid_transition():
    attempt = TaskExecutionAttempt(attempt_id="attempt-1", attempt_number=1)

    with pytest.raises(ValueError, match="Transição inválida"):
        attempt.transition_to(TaskStatus.SUCCEEDED)


def test_legacy_result_becomes_a_single_task_workflow():
    workflow = legacy_task_run(_legacy_result())

    assert workflow.experiment_run_id == "experiment-1"
    assert len(workflow.tasks) == 1
    assert workflow.tasks[0].task_id == "legacy-main-task"
    assert workflow.tasks[0].attempts[0].status is TaskStatus.SUCCEEDED


def test_definition_rejects_duplicated_task_ids():
    task = TaskDefinition(task_id="train", name="Treino")
    with pytest.raises(ValueError, match="únicos"):
        ExperimentDefinition(name="duplicated", tasks=(task, task))


def test_sequential_executor_records_task_metrics_and_artifacts():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(TaskDefinition(task_id="prepare", name="Preparar dados"),),
    )
    executor = SequentialWorkflowExecutor(
        {"prepare": lambda: {"artifacts": {"dataset": "data.json"}}}
    )

    workflow = executor.execute(definition)

    attempt = workflow.tasks[0].attempts[0]
    assert workflow.status == "success"
    assert attempt.status is TaskStatus.SUCCEEDED
    assert attempt.metrics["resources"]["task_time_sec"] >= 0
    assert attempt.artifacts == {"dataset": "data.json"}


def test_sequential_executor_skips_task_with_failed_dependency():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition(task_id="first", name="Primeira"),
            TaskDefinition(task_id="next", name="Seguinte", depends_on=("first",)),
        ),
    )
    executor = SequentialWorkflowExecutor(
        {"first": lambda: (_ for _ in ()).throw(RuntimeError("falhou")), "next": dict}
    )

    workflow = executor.execute(definition)

    assert workflow.status == "failed"
    assert workflow.tasks[0].status is TaskStatus.FAILED
    assert workflow.tasks[1].status is TaskStatus.SKIPPED


def test_sequential_executor_uses_topological_order():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition(task_id="evaluate", name="Avaliar", depends_on=("train",)),
            TaskDefinition(task_id="prepare", name="Preparar"),
            TaskDefinition(task_id="train", name="Treinar", depends_on=("prepare",)),
        ),
    )
    execution_order: list[str] = []
    executor = SequentialWorkflowExecutor(
        {
            task_id: lambda task_id=task_id: execution_order.append(task_id) or {}
            for task_id in ("prepare", "train", "evaluate")
        }
    )

    workflow = executor.execute(definition)

    assert workflow.status == "success"
    assert execution_order == ["prepare", "train", "evaluate"]


def test_sequential_executor_retries_and_preserves_attempt_history():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition(
                task_id="train",
                name="Treinar",
                retry_policy=RetryPolicy(max_attempts=2),
            ),
        ),
    )
    calls = 0

    def train() -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("erro transitorio")
        return {"artifacts": {"checkpoint": "model.pkl"}}

    workflow = SequentialWorkflowExecutor({"train": train}).execute(definition)

    attempts = workflow.tasks[0].attempts
    assert workflow.status == "success"
    assert [attempt.attempt_number for attempt in attempts] == [1, 2]
    assert [attempt.status for attempt in attempts] == [TaskStatus.FAILED, TaskStatus.SUCCEEDED]
    assert attempts[0].error == "erro transitorio"
    assert attempts[1].artifacts == {"checkpoint": "model.pkl"}


def test_sequential_executor_does_not_retry_non_eligible_error():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition(
                task_id="train",
                name="Treinar",
                retry_policy=RetryPolicy(max_attempts=3, retryable_error_types=("InfraError",)),
            ),
        ),
    )
    calls = 0

    def train() -> dict:
        nonlocal calls
        calls += 1
        raise ValueError("configuracao invalida")

    workflow = SequentialWorkflowExecutor({"train": train}).execute(definition)

    assert workflow.status == "failed"
    assert calls == 1
    assert len(workflow.tasks[0].attempts) == 1


def test_sequential_executor_retries_eligible_error_type():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition(
                task_id="train",
                name="Treinar",
                retry_policy=RetryPolicy(max_attempts=2, retryable_error_types=("InfraError",)),
            ),
        ),
    )
    calls = 0

    def train() -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("tempo esgotado")
        return {}

    workflow = SequentialWorkflowExecutor({"train": train}).execute(definition)

    assert workflow.status == "success"
    assert calls == 2
    assert workflow.tasks[0].attempts[0].error_type == "InfraError"
    assert workflow.tasks[0].attempts[0].root_error_type == "TimeoutError"


def test_retry_policy_requires_positive_max_attempts():
    with pytest.raises(ValueError, match="maior ou igual a 1"):
        RetryPolicy(max_attempts=0)


def test_sequential_executor_resume_keeps_success_and_retries_failed_task():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition(task_id="prepare", name="Preparar"),
            TaskDefinition(
                task_id="train",
                name="Treinar",
                depends_on=("prepare",),
                retry_policy=RetryPolicy(max_attempts=2, retryable_error_types=("TimeoutError",)),
            ),
        ),
    )
    previous = ExperimentRun(
        experiment_run_id="resume-1",
        definition_name="workflow",
        status="failed",
        tasks=[
            TaskRun(
                "prepare", "Preparar", "train", TaskStatus.SUCCEEDED,
                [TaskExecutionAttempt("prepare-1", 1, TaskStatus.SUCCEEDED)],
            ),
            TaskRun(
                "train", "Treinar", "train", TaskStatus.FAILED,
                [TaskExecutionAttempt("train-1", 1, TaskStatus.FAILED, error_type="TimeoutError")],
            ),
        ],
    )
    calls: list[str] = []
    executor = SequentialWorkflowExecutor(
        {
            "prepare": lambda: calls.append("prepare") or {},
            "train": lambda: calls.append("train") or {},
        }
    )

    workflow = executor.execute(definition, resume_from=previous)

    assert workflow.experiment_run_id == "resume-1"
    assert calls == ["train"]
    assert workflow.status == "success"
    assert [attempt.attempt_number for attempt in workflow.tasks[1].attempts] == [1, 2]


def test_sequential_executor_resume_does_not_rerun_exhausted_task():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition("train", "Treinar", retry_policy=RetryPolicy(max_attempts=1)),
        ),
    )
    previous = ExperimentRun(
        "resume-1",
        "workflow",
        "failed",
        [
            TaskRun(
                "train", "Treinar", "train", TaskStatus.FAILED,
                [TaskExecutionAttempt("train-1", 1, TaskStatus.FAILED, error_type="RuntimeError")],
            )
        ],
    )

    workflow = SequentialWorkflowExecutor({"train": lambda: pytest.fail("não deve executar")}).execute(
        definition, resume_from=previous
    )

    assert workflow.status == "failed"
    assert len(workflow.tasks[0].attempts) == 1


def test_sequential_executor_resume_reexecutes_interrupted_task():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(TaskDefinition("train", "Treinar"),),
    )
    previous = ExperimentRun(
        "resume-1",
        "workflow",
        "failed",
        [
            TaskRun(
                "train", "Treinar", "train", TaskStatus.RUNNING,
                [TaskExecutionAttempt("train-1", 1, TaskStatus.RUNNING)],
            )
        ],
    )
    calls = 0

    def train() -> dict:
        nonlocal calls
        calls += 1
        return {}

    workflow = SequentialWorkflowExecutor({"train": train}).execute(
        definition, resume_from=previous
    )

    assert workflow.status == "success"
    assert calls == 1
    assert [attempt.attempt_number for attempt in workflow.tasks[0].attempts] == [1, 2]
    assert workflow.tasks[0].attempts[0].status is TaskStatus.RUNNING
    assert workflow.tasks[0].attempts[1].status is TaskStatus.SUCCEEDED


def test_sequential_executor_reuses_cached_successful_task(tmp_path):
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition(
                "prepare", "Preparar", config={"dataset": "v1"}, input_signatures={"raw": "abc"}
            ),
        ),
    )
    cache = TaskCache(tmp_path)
    calls = 0

    def prepare() -> dict:
        nonlocal calls
        calls += 1
        return {"artifacts": {"dataset": "prepared.json"}}

    executor = SequentialWorkflowExecutor({"prepare": prepare}, cache=cache, code_version="commit-a")
    first = executor.execute(definition)
    second = executor.execute(definition)

    assert first.tasks[0].status is TaskStatus.SUCCEEDED
    assert second.status == "success"
    assert second.tasks[0].status is TaskStatus.CACHED
    assert second.tasks[0].attempts[0].metrics["cache_hit"] is True
    assert calls == 1


def test_task_cache_invalidates_when_input_or_code_version_changes(tmp_path):
    cache = TaskCache(tmp_path)
    calls = 0

    def prepare() -> dict:
        nonlocal calls
        calls += 1
        return {}

    original = ExperimentDefinition(
        "workflow", (TaskDefinition("prepare", "Preparar", input_signatures={"raw": "v1"}),)
    )
    changed_input = ExperimentDefinition(
        "workflow", (TaskDefinition("prepare", "Preparar", input_signatures={"raw": "v2"}),)
    )
    SequentialWorkflowExecutor({"prepare": prepare}, cache=cache, code_version="commit-a").execute(original)
    input_run = SequentialWorkflowExecutor({"prepare": prepare}, cache=cache, code_version="commit-a").execute(changed_input)
    code_run = SequentialWorkflowExecutor({"prepare": prepare}, cache=cache, code_version="commit-b").execute(original)

    assert input_run.tasks[0].status is TaskStatus.SUCCEEDED
    assert code_run.tasks[0].status is TaskStatus.SUCCEEDED
    assert calls == 3


def test_sequential_executor_reexecutes_success_when_resume_signature_changes():
    current = ExperimentDefinition(
        "workflow", (TaskDefinition("prepare", "Preparar", input_signatures={"raw": "v2"}),)
    )
    previous = ExperimentRun(
        "resume-1", "workflow", "success", [
            TaskRun(
                "prepare", "Preparar", "train", TaskStatus.SUCCEEDED,
                [TaskExecutionAttempt("prepare-1", 1, TaskStatus.SUCCEEDED)],
                input_signatures={"raw": "v1"},
            )
        ],
    )
    calls = 0

    def prepare() -> dict:
        nonlocal calls
        calls += 1
        return {}

    workflow = SequentialWorkflowExecutor({"prepare": prepare}).execute(
        current, resume_from=previous
    )

    assert calls == 1
    assert workflow.tasks[0].status is TaskStatus.SUCCEEDED


def test_parallel_executor_runs_independent_tasks_concurrently():
    definition = ExperimentDefinition(
        "workflow",
        (
            TaskDefinition("first", "Primeira"),
            TaskDefinition("second", "Segunda"),
            TaskDefinition("final", "Final", depends_on=("first", "second")),
        ),
    )
    started: list[str] = []
    from threading import Barrier, Thread

    barrier = Barrier(3)

    def independent(task_id: str) -> dict:
        started.append(task_id)
        barrier.wait(timeout=1)
        return {}

    executor = ParallelWorkflowExecutor(
        {
            "first": lambda: independent("first"),
            "second": lambda: independent("second"),
            "final": lambda: {"artifacts": {"done": True}},
        },
        max_workers=2,
    )

    run_thread = Thread(target=lambda: executor.execute(definition))
    run_thread.start()
    barrier.wait(timeout=1)
    run_thread.join(timeout=2)

    assert sorted(started) == ["first", "second"]


def test_parallel_executor_respects_dependencies_and_worker_limit():
    definition = ExperimentDefinition(
        "workflow",
        (
            TaskDefinition("prepare", "Preparar"),
            TaskDefinition("train", "Treinar", depends_on=("prepare",)),
        ),
    )
    calls: list[str] = []
    executor = ParallelWorkflowExecutor(
        {
            "prepare": lambda: calls.append("prepare") or {},
            "train": lambda: calls.append("train") or {},
        },
        max_workers=1,
    )

    workflow = executor.execute(definition)

    assert workflow.status == "success"
    assert calls == ["prepare", "train"]


def test_parallel_executor_skips_dependents_of_failed_task():
    definition = ExperimentDefinition(
        "workflow",
        (
            TaskDefinition("prepare", "Preparar"),
            TaskDefinition("train", "Treinar", depends_on=("prepare",)),
        ),
    )
    executor = ParallelWorkflowExecutor(
        {
            "prepare": lambda: (_ for _ in ()).throw(RuntimeError("falhou")),
            "train": lambda: pytest.fail("não deve executar"),
        },
        max_workers=2,
    )

    workflow = executor.execute(definition)

    assert workflow.status == "failed"
    assert workflow.tasks[0].status is TaskStatus.FAILED
    assert workflow.tasks[1].status is TaskStatus.SKIPPED


def test_parallel_executor_requires_positive_worker_limit():
    with pytest.raises(ValueError, match="maior ou igual a 1"):
        ParallelWorkflowExecutor({}, max_workers=0)


def test_planner_orders_dependencies_before_dependents():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition(task_id="evaluate", name="Avaliar", depends_on=("train",)),
            TaskDefinition(task_id="prepare", name="Preparar"),
            TaskDefinition(task_id="train", name="Treinar", depends_on=("prepare",)),
        ),
    )

    plan = WorkflowPlanner().plan(definition)

    assert [task.task_id for task in plan] == ["prepare", "train", "evaluate"]


def test_planner_derives_dependencies_from_versioned_artifacts():
    dataset = ArtifactDefinition("corpus", ArtifactKind.DATA, "v1")
    checkpoint = ArtifactDefinition("model", ArtifactKind.MODEL, "v1")
    definition = ExperimentDefinition(
        "workflow",
        (
            TaskDefinition("train", "Treinar", inputs=(dataset,), outputs=(checkpoint,)),
            TaskDefinition("prepare", "Preparar", outputs=(dataset,)),
            TaskDefinition("evaluate", "Avaliar", inputs=(checkpoint,)),
        ),
    )

    plan = WorkflowPlanner().plan(definition)

    assert [task.task_id for task in plan] == ["prepare", "train", "evaluate"]
    assert plan[1].depends_on == ("prepare",)
    assert plan[2].depends_on == ("train",)


def test_planner_rejects_multiple_producers_for_same_artifact_version():
    dataset = ArtifactDefinition("corpus", ArtifactKind.DATA, "v1")
    definition = ExperimentDefinition(
        "workflow",
        (
            TaskDefinition("download_a", "Baixar A", outputs=(dataset,)),
            TaskDefinition("download_b", "Baixar B", outputs=(dataset,)),
        ),
    )

    with pytest.raises(ValueError, match="mais de um produtor"):
        WorkflowPlanner().plan(definition)


def test_planner_rejects_input_incompatible_with_internal_producer():
    produced = ArtifactDefinition("model", ArtifactKind.MODEL, "v1")
    expected = ArtifactDefinition("model", ArtifactKind.INDEX, "v2")
    definition = ExperimentDefinition(
        "workflow",
        (
            TaskDefinition("train", "Treinar", outputs=(produced,)),
            TaskDefinition("retrieve", "Recuperar", inputs=(expected,)),
        ),
    )

    with pytest.raises(ValueError, match="artefato incompatível.*model"):
        WorkflowPlanner().plan(definition)


def test_planner_preserves_declaration_order_for_independent_tasks():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition(task_id="second", name="Segunda"),
            TaskDefinition(task_id="first", name="Primeira"),
        ),
    )

    plan = WorkflowPlanner().plan(definition)

    assert [task.task_id for task in plan] == ["second", "first"]


def test_planner_rejects_missing_dependency():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(TaskDefinition(task_id="train", name="Treinar", depends_on=("data",)),),
    )

    with pytest.raises(ValueError, match="tarefa inexistente: data"):
        WorkflowPlanner().plan(definition)


def test_planner_rejects_direct_cycle():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(TaskDefinition(task_id="train", name="Treinar", depends_on=("train",)),),
    )

    with pytest.raises(ValueError, match="ciclo.*train"):
        WorkflowPlanner().plan(definition)


def test_planner_rejects_indirect_cycle():
    definition = ExperimentDefinition(
        name="workflow",
        tasks=(
            TaskDefinition(task_id="first", name="Primeira", depends_on=("third",)),
            TaskDefinition(task_id="second", name="Segunda", depends_on=("first",)),
            TaskDefinition(task_id="third", name="Terceira", depends_on=("second",)),
        ),
    )

    with pytest.raises(ValueError, match="first.*second.*third"):
        WorkflowPlanner().plan(definition)


# ---------------------------------------------------------------------------
# classify_error_type
# ---------------------------------------------------------------------------

class _FakeReadTimeout(Exception):
    """Stub que imita requests.exceptions.ReadTimeout sem importar requests."""


def test_classify_error_type_returns_infra_error_for_builtin_timeout():
    assert classify_error_type(TimeoutError("conn")) == "InfraError"


def test_classify_error_type_returns_class_name_for_generic_exception():
    assert classify_error_type(ValueError("bad value")) == "ValueError"


def test_classify_error_type_returns_class_name_for_unknown_exception():
    class CustomDomainError(Exception):
        pass

    assert classify_error_type(CustomDomainError()) == "CustomDomainError"


def test_classify_error_type_maps_requests_read_timeout(monkeypatch):
    """Verifica mapeamento via isinstance quando requests está disponível."""
    try:
        import requests.exceptions as req_exc  # type: ignore
    except ImportError:
        pytest.skip("requests não disponível")

    exc = req_exc.ReadTimeout("hub timeout")
    assert classify_error_type(exc) == "InfraError"


def test_classify_error_type_maps_requests_connect_timeout(monkeypatch):
    try:
        import requests.exceptions as req_exc
    except ImportError:
        pytest.skip("requests não disponível")

    exc = req_exc.ConnectTimeout("connect timeout")
    assert classify_error_type(exc) == "InfraError"


def test_executor_sets_infra_error_and_root_error_type_on_timeout():
    """Executor preenche error_type=InfraError e root_error_type=ReadTimeout."""
    try:
        import requests.exceptions as req_exc
    except ImportError:
        pytest.skip("requests não disponível")

    def _failing_task():
        raise req_exc.ReadTimeout("hub indisponível")

    definition = ExperimentDefinition(
        name="infra-test",
        tasks=(TaskDefinition(task_id="t1", name="Falha de rede"),),
    )
    executor = SequentialWorkflowExecutor({"t1": _failing_task})
    run = executor.execute(definition)

    attempt = run.tasks[0].attempts[0]
    assert attempt.error_type == "InfraError"
    assert attempt.root_error_type == "ReadTimeout"
    assert run.tasks[0].status == TaskStatus.FAILED


def test_executor_does_not_set_root_error_type_for_non_infra_errors():
    """root_error_type fica None quando não é InfraError."""
    def _failing_task():
        raise RuntimeError("erro funcional")

    definition = ExperimentDefinition(
        name="func-error-test",
        tasks=(TaskDefinition(task_id="t1", name="Erro funcional"),),
    )
    executor = SequentialWorkflowExecutor({"t1": _failing_task})
    run = executor.execute(definition)

    attempt = run.tasks[0].attempts[0]
    assert attempt.error_type == "RuntimeError"
    assert attempt.root_error_type is None


# ---------------------------------------------------------------------------
# INFRA_RETRY_POLICY — retry seletivo para InfraError
# ---------------------------------------------------------------------------

def test_infra_retry_policy_only_retries_infra_error():
    calls = 0

    def _task():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("transitório")
        return {}

    definition = ExperimentDefinition(
        name="retry-infra",
        tasks=(TaskDefinition("t1", "Tarefa", retry_policy=INFRA_RETRY_POLICY),),
    )
    run = SequentialWorkflowExecutor({"t1": _task}).execute(definition)

    assert run.status == "success"
    assert calls == 3
    assert len(run.tasks[0].attempts) == 3
    assert run.tasks[0].attempts[0].error_type == "InfraError"
    assert run.tasks[0].attempts[0].root_error_type == "TimeoutError"


def test_infra_retry_policy_does_not_retry_functional_error():
    calls = 0

    def _task():
        nonlocal calls
        calls += 1
        raise ValueError("configuração inválida")

    definition = ExperimentDefinition(
        name="no-retry-func",
        tasks=(TaskDefinition("t1", "Tarefa", retry_policy=INFRA_RETRY_POLICY),),
    )
    run = SequentialWorkflowExecutor({"t1": _task}).execute(definition)

    assert run.status == "failed"
    assert calls == 1
    assert len(run.tasks[0].attempts) == 1
    assert run.tasks[0].attempts[0].error_type == "ValueError"


def test_infra_retry_policy_exhausts_after_max_attempts():
    calls = 0

    def _task():
        nonlocal calls
        calls += 1
        raise TimeoutError("hub indisponível")

    definition = ExperimentDefinition(
        name="exhaust-retries",
        tasks=(TaskDefinition("t1", "Tarefa", retry_policy=INFRA_RETRY_POLICY),),
    )
    run = SequentialWorkflowExecutor({"t1": _task}).execute(definition)

    assert run.status == "failed"
    assert calls == INFRA_RETRY_POLICY.max_attempts
    assert len(run.tasks[0].attempts) == INFRA_RETRY_POLICY.max_attempts


def test_adapt_model_template_uses_infra_retry_policy():
    """adapt_model declarado pelo template HuggingFace usa INFRA_RETRY_POLICY."""
    from experiment.workflow_templates import (  # type: ignore
        HuggingFaceWorkflowConfig,
        build_huggingface_workflow,
    )

    config = HuggingFaceWorkflowConfig(name="test-wf")
    definition = build_huggingface_workflow(config)

    adapt_task = next(t for t in definition.tasks if t.task_id == "adapt_model")
    assert adapt_task.retry_policy is INFRA_RETRY_POLICY


# ---------------------------------------------------------------------------
# failure_stage — propagação do estágio de falha
# ---------------------------------------------------------------------------

def test_executor_extracts_failure_stage_from_exception():
    """Executor preenche failure_stage quando a exceção carrega o atributo."""
    def _failing_task():
        exc = RuntimeError("init falhou")
        exc.failure_stage = "init"  # type: ignore[attr-defined]
        raise exc

    definition = ExperimentDefinition(
        name="stage-test",
        tasks=(TaskDefinition("t1", "Tarefa"),),
    )
    run = SequentialWorkflowExecutor({"t1": _failing_task}).execute(definition)

    attempt = run.tasks[0].attempts[0]
    assert attempt.failure_stage == "init"
    assert attempt.error_type == "RuntimeError"


def test_executor_failure_stage_is_none_when_not_set():
    """failure_stage fica None quando a exceção não carrega o atributo."""
    def _failing_task():
        raise ValueError("erro sem stage")

    definition = ExperimentDefinition(
        name="no-stage-test",
        tasks=(TaskDefinition("t1", "Tarefa"),),
    )
    run = SequentialWorkflowExecutor({"t1": _failing_task}).execute(definition)

    assert run.tasks[0].attempts[0].failure_stage is None