"""
Executor sequencial de tarefas de um workflow.
============================================

 - Executa tarefas de um workflow de forma sequencial.
 - Esta execução é sequencial, respeitando as dependências entre tarefas.
 - Cada tarefa é executada em uma tentativa única, e os resultados de execução são agregados ao experimento.

"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .helpers import now_iso
from .task_cache import TaskCache
from .task_telemetry import TaskTelemetryCollector
from .workflow import (
    ExperimentDefinition,
    ExperimentRun,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
)
from .workflow_planner import WorkflowPlanner

TaskCallable = Callable[[], dict[str, Any] | None]


class SequentialWorkflowExecutor:
    """Executa tarefas independentes ou encadeadas por ``depends_on`` em série."""

    def __init__(
        self,
        task_functions: Mapping[str, TaskCallable],
        *,
        cache: TaskCache | None = None,
        code_version: str | None = None,
        telemetry: TaskTelemetryCollector | None = None,
    ) -> None:
        self._task_functions = task_functions
        self._cache = cache
        self._code_version = code_version
        self._telemetry = telemetry or TaskTelemetryCollector()

    def execute(
        self,
        definition: ExperimentDefinition,
        resume_from: ExperimentRun | None = None,
    ) -> ExperimentRun:
        """Executa o plano topológico, reutilizando tarefas concluídas quando informado."""
        task_runs: list[TaskRun] = []
        statuses: dict[str, TaskStatus] = {}
        task_plan = WorkflowPlanner().plan(definition)
        previous_tasks = {
            task.task_id: task for task in resume_from.tasks
        } if resume_from else {}

        for task in task_plan:
            blocked = any(
                statuses.get(dependency) not in {TaskStatus.SUCCEEDED, TaskStatus.CACHED}
                for dependency in task.depends_on
            )
            if blocked:
                task_runs.append(
                    TaskRun(task.task_id, task.name, task.task_type, TaskStatus.SKIPPED,
                            config=task.config, input_signatures=task.input_signatures)
                )
                statuses[task.task_id] = TaskStatus.SKIPPED
                continue

            previous = previous_tasks.get(task.task_id)
            if previous and not self._is_resume_compatible(task, previous):
                previous = None
            if previous and previous.status in {TaskStatus.SUCCEEDED, TaskStatus.CACHED}:
                self._restore_task_state(task.task_id, previous)
                task_runs.append(previous)
                statuses[task.task_id] = previous.status
                continue

            if previous and previous.status is TaskStatus.FAILED:
                last_attempt = previous.attempts[-1] if previous.attempts else None
                retry_allowed = (
                    last_attempt is not None
                    and len(previous.attempts) < task.retry_policy.max_attempts
                    and task.retry_policy.allows_retry(last_attempt.error_type or "")
                )
                if not retry_allowed:
                    task_runs.append(previous)
                    statuses[task.task_id] = previous.status
                    continue

            cache = self._cache
            signature = (
                cache.signature(task, self._code_version)
                if cache and self._code_version
                else None
            )
            cached = cache.get(signature) if cache and signature else None
            if cached:
                task_run = self._cached_task_run(task, cached)
                self._restore_task_state(task.task_id, task_run)
            else:
                task_fn = self._task_functions.get(task.task_id)
                if task_fn is None:
                    raise ValueError(f"Nenhuma função registrada para a tarefa '{task.task_id}'.")
                task_run = self._execute_task(task, task_fn, previous)
                if task_run.status is TaskStatus.SUCCEEDED and cache and signature:
                    attempt = task_run.attempts[-1]
                    cache.put(signature, metrics=attempt.metrics, artifacts=attempt.artifacts)
            task_runs.append(task_run)
            statuses[task.task_id] = task_run.status

        required_failed = any(
            task.required and statuses[task.task_id] not in {TaskStatus.SUCCEEDED, TaskStatus.CACHED}
            for task in definition.tasks
        )
        return ExperimentRun(
            experiment_run_id=resume_from.experiment_run_id if resume_from else str(uuid.uuid4()),
            definition_name=definition.name,
            status="failed" if required_failed else "success",
            tasks=task_runs,
        )

    @staticmethod
    def _is_resume_compatible(task, previous: TaskRun) -> bool:
        return (
            task.task_type == previous.task_type
            and task.config == previous.config
            and task.input_signatures == previous.input_signatures
        )

    def _restore_task_state(self, task_id: str, task_run: TaskRun) -> None:
        """Permite que adaptadores reidratem estado ao reutilizar T2 do cache."""
        task_fn = self._task_functions.get(task_id)
        restore = getattr(task_fn, "restore_cached_result", None)
        if restore and task_run.attempts:
            restore(task_run.attempts[-1].metrics)

    @staticmethod
    def _cached_task_run(task, cached: dict[str, Any]) -> TaskRun:
        attempt = TaskExecutionAttempt(
            attempt_id=str(uuid.uuid4()),
            attempt_number=1,
            status=TaskStatus.CACHED,
            started_at=now_iso(),
            completed_at=now_iso(),
            metrics={**cached.get("metrics", {}), "cache_hit": True},
            artifacts=cached.get("artifacts", {}),
        )
        return TaskRun(task.task_id, task.name, task.task_type, TaskStatus.CACHED, [attempt],
                   task.config, task.input_signatures)

    def _execute_task(
        self,
        task,
        task_fn: TaskCallable,
        previous: TaskRun | None = None,
    ) -> TaskRun:
        attempts = list(previous.attempts) if previous else []
        if attempts and attempts[-1].status is TaskStatus.FAILED:  # noqa: SIM102
            if not task.retry_policy.allows_retry(attempts[-1].error_type or ""):
                return previous # type: ignore

        max_attempt_number = task.retry_policy.max_attempts
        if attempts and attempts[-1].status in {
            TaskStatus.CREATED,
            TaskStatus.READY,
            TaskStatus.RUNNING,
            TaskStatus.SKIPPED,
        }:
            max_attempt_number = max(max_attempt_number, len(attempts) + 1)

        for attempt_number in range(len(attempts) + 1, max_attempt_number + 1):
            attempt = self._execute_attempt(
                task.task_id,
                attempt_number,
                task_fn,
            )
            attempts.append(attempt)
            if attempt.status is TaskStatus.SUCCEEDED:
                break
            if not task.retry_policy.allows_retry(attempt.error_type or ""):
                break

        return TaskRun(
            task_id=task.task_id,
            name=task.name,
            task_type=task.task_type,
            status=attempts[-1].status,
            attempts=attempts,
            config=task.config,
            input_signatures=task.input_signatures,
        )

    def _execute_attempt(
        self,
        task_id: str,
        attempt_number: int,
        task_fn: TaskCallable,
    ) -> TaskExecutionAttempt:
        attempt = TaskExecutionAttempt(
            attempt_id=str(uuid.uuid4()),
            attempt_number=attempt_number,
        )
        attempt.transition_to(TaskStatus.READY)
        attempt.transition_to(TaskStatus.RUNNING)
        attempt.started_at = now_iso()
        try:
            output, telemetry_metrics, task_error = self._telemetry.measure(
                task_fn,
                task_id=task_id,
                attempt_id=attempt.attempt_id,
            )
            task_resources = output.get("metrics", {}).get("resources", {})
            attempt.metrics = {
                **output.get("metrics", {}),
                "resources": {**task_resources, **telemetry_metrics},
            }
            attempt.artifacts = output.get("artifacts", {})
            if task_error is not None:
                raise task_error
            attempt.transition_to(TaskStatus.SUCCEEDED)
        except Exception as exc:  # noqa: BLE001
            if not attempt.metrics:
                attempt.metrics = {"resources": {}}
            attempt.error = str(exc)
            attempt.error_type = exc.__class__.__name__
            attempt.transition_to(TaskStatus.FAILED)
        finally:
            attempt.completed_at = now_iso()

        return attempt


class ParallelWorkflowExecutor(SequentialWorkflowExecutor):
    """Executa em paralelo tarefas independentes de um workflow em DAG."""

    def __init__(
        self,
        task_functions: Mapping[str, TaskCallable],
        *,
        max_workers: int,
        cache: TaskCache | None = None,
        code_version: str | None = None,
        telemetry: TaskTelemetryCollector | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers deve ser maior ou igual a 1.")
        super().__init__(task_functions, cache=cache, code_version=code_version, telemetry=telemetry)
        self._max_workers = max_workers

    def execute(
        self,
        definition: ExperimentDefinition,
        resume_from: ExperimentRun | None = None,
    ) -> ExperimentRun:
        """Agenda tarefas prontas até o limite de workers e espera cada onda terminar."""
        task_plan = WorkflowPlanner().plan(definition)
        task_by_id = {task.task_id: task for task in task_plan}
        previous_tasks = {
            task.task_id: task for task in resume_from.tasks
        } if resume_from else {}
        statuses: dict[str, TaskStatus] = {}
        task_runs: dict[str, TaskRun] = {}
        pending = {task.task_id for task in task_plan}

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            while pending:
                progressed = self._reuse_or_skip_tasks(
                    pending, task_plan, previous_tasks, statuses, task_runs
                )
                ready = [
                    task_by_id[task_id]
                    for task_id in pending
                    if all(
                        statuses.get(dependency) in {TaskStatus.SUCCEEDED, TaskStatus.CACHED}
                        for dependency in task_by_id[task_id].depends_on
                    )
                ]
                if not ready:
                    if not progressed:
                        raise RuntimeError("Não foi possível agendar tarefas pendentes do workflow.")
                    continue

                futures: dict[Future[TaskRun], str] = {
                    pool.submit(self._execute_ready_task, task, previous_tasks.get(task.task_id)): task.task_id
                    for task in ready[:self._max_workers]
                }
                for future, task_id in futures.items():
                    task_run = future.result()
                    task_runs[task_id] = task_run
                    statuses[task_id] = task_run.status
                    pending.remove(task_id)

        required_failed = any(
            task.required and statuses[task.task_id] not in {TaskStatus.SUCCEEDED, TaskStatus.CACHED}
            for task in definition.tasks
        )
        return ExperimentRun(
            experiment_run_id=resume_from.experiment_run_id if resume_from else str(uuid.uuid4()),
            definition_name=definition.name,
            status="failed" if required_failed else "success",
            tasks=[task_runs[task.task_id] for task in task_plan],
        )

    def _reuse_or_skip_tasks(
        self,
        pending: set[str],
        task_plan: tuple,
        previous_tasks: dict[str, TaskRun],
        statuses: dict[str, TaskStatus],
        task_runs: dict[str, TaskRun],
    ) -> bool:
        progressed = False
        for task in task_plan:
            if task.task_id not in pending:
                continue
            if any(
                dependency in statuses
                and statuses[dependency] not in {TaskStatus.SUCCEEDED, TaskStatus.CACHED}
                for dependency in task.depends_on
            ):
                task_runs[task.task_id] = TaskRun(
                    task.task_id, task.name, task.task_type, TaskStatus.SKIPPED,
                    config=task.config, input_signatures=task.input_signatures,
                )
            else:
                previous = previous_tasks.get(task.task_id)
                compatible_previous = (
                    previous if previous and self._is_resume_compatible(task, previous) else None
                )
                if compatible_previous and (
                    compatible_previous.status in {TaskStatus.SUCCEEDED, TaskStatus.CACHED}
                    or compatible_previous.status is TaskStatus.FAILED
                    and not self._can_resume(task, compatible_previous)
                ):
                    self._restore_task_state(task.task_id, compatible_previous)
                    task_runs[task.task_id] = compatible_previous
                else:
                    continue
            statuses[task.task_id] = task_runs[task.task_id].status
            pending.remove(task.task_id)
            progressed = True
        return progressed

    @staticmethod
    def _can_resume(task, previous: TaskRun) -> bool:
        last_attempt = previous.attempts[-1] if previous.attempts else None
        return bool(
            last_attempt
            and len(previous.attempts) < task.retry_policy.max_attempts
            and task.retry_policy.allows_retry(last_attempt.error_type or "")
        )

    def _execute_ready_task(self, task, previous: TaskRun | None) -> TaskRun:
        if previous and not self._is_resume_compatible(task, previous):
            previous = None
        cache = self._cache
        signature = (
            cache.signature(task, self._code_version)
            if cache and self._code_version
            else None
        )
        cached = cache.get(signature) if cache and signature else None
        if cached:
            task_run = self._cached_task_run(task, cached)
            self._restore_task_state(task.task_id, task_run)
            return task_run

        task_fn = self._task_functions.get(task.task_id)
        if task_fn is None:
            raise ValueError(f"Nenhuma função registrada para a tarefa '{task.task_id}'.")
        task_run = self._execute_task(task, task_fn, previous)
        if task_run.status is TaskStatus.SUCCEEDED and cache and signature:
            attempt = task_run.attempts[-1]
            cache.put(signature, metrics=attempt.metrics, artifacts=attempt.artifacts)
        return task_run