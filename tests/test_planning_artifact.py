"""Testes do artefato consolidado para planejamento de workflow."""

import json

from experiment.planning_artifact import (
    build_planning_artifact,
    write_planning_artifact,
)
from experiment.workflow import (
    ExperimentDefinition,
    ExperimentRun,
    RetryPolicy,
    TaskDefinition,
    TaskExecutionAttempt,
    TaskRun,
    TaskStatus,
)


def _definition():
    return ExperimentDefinition(
        "pipeline",
        (
            TaskDefinition(
                "prepare", "Preparar", "prepare", config={"source": "v1"},
                input_signatures={"dataset": "abc"}, retry_policy=RetryPolicy(2, ("InfraError",)),
            ),
            TaskDefinition("train", "Treinar", "train", depends_on=("prepare",)),
        ),
        experiment_type="nlp",
    )


def _workflow():
    return ExperimentRun(
        "run-1", "pipeline", "success",
        [
            TaskRun("prepare", "Preparar", "prepare", TaskStatus.SUCCEEDED, [
                TaskExecutionAttempt("prepare-1", 1, TaskStatus.SUCCEEDED, metrics={"resources": {
                    "task_time_sec": 2, "energy_kwh": 0.1,
                }}),
            ], config={"source": "v1"}, input_signatures={"dataset": "abc"}),
            TaskRun("train", "Treinar", "train", TaskStatus.SUCCEEDED, [
                TaskExecutionAttempt("train-1", 1, TaskStatus.SUCCEEDED, metrics={"resources": {
                    "task_time_sec": 5, "cost_usd": 1.5,
                }, "evaluation": {"accuracy": 0.9}}),
            ]),
        ],
    )


def test_planning_artifact_consolidates_dag_profiles_estimates_and_observations():
    definition = _definition()
    artifact = build_planning_artifact(
        definition, [_workflow()], workflow=_workflow(),
        constraints={"cost_usd": 2.0}, preferences={"minimize": ["cost_usd"]},
    )

    prepare, train = artifact["tasks"]
    assert artifact["definition"] == {"name": "pipeline", "experiment_type": "nlp", "schema_version": "1.0"}
    assert prepare["depends_on"] == []
    assert prepare["config"] == {"source": "v1"}
    assert prepare["retry_policy"]["max_attempts"] == 2
    assert train["depends_on"] == ["prepare"]
    assert artifact["estimated_workflow_resources"]["task_time_sec"] == 7.0
    assert artifact["observed_workflow"]["evaluation"] == {"accuracy": 0.9}
    assert artifact["constraints"] == {"cost_usd": 2.0}
    assert artifact["preferences"] == {"minimize": ["cost_usd"]}


def test_planning_artifact_supports_pre_execution_export(tmp_path):
    artifact = build_planning_artifact(_definition(), constraints={"deadline_sec": 60})

    output = write_planning_artifact(artifact, tmp_path / "planning.json")

    with open(output, encoding="utf-8") as file:
        persisted = json.load(file)
    assert persisted["observed_workflow"] is None
    assert persisted["tasks"][0]["estimation_evidence"]["match_level"] == "none"
    assert persisted["constraints"] == {"deadline_sec": 60}