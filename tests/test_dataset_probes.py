"""Testes das estatísticas de dataset coletadas na tarefa de ingestão."""

import pytest

from experiment.workflow_templates import (
    HuggingFaceWorkflowConfig,
    LauncherWorkflowConfig,
    _build_huggingface_dataset_probe,
    _build_launcher_dataset_probe,
    _count_tokens_for_sample,
    _estimate_dataset_size_mb,
    _summarize_dataset_partitions,
    build_launcher_task_functions,
)

MB = 1024 * 1024


def _fake_dataset_class(splits, *, file_list=None, source="hub"):
    """Cria uma classe de dataset em memória; splits ausentes levantam ValueError."""

    class FakeDataset:
        def __init__(self, config, mode, *args, **params):
            if mode not in splits:
                raise ValueError(f"split ausente: {mode}")
            self.rows = splits[mode]
            self.file_list = list(file_list or [])
            self.source = source

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, item):
            return self.rows[item]

    return FakeDataset


class _FakeInfo:
    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


class _FakeBuilder:
    def __init__(self, info):
        self.info = info


def _write_file(path, size_bytes):
    path.write_bytes(b"x" * size_bytes)
    return str(path)


def _config_file(tmp_path, body="[data]\n"):
    path = tmp_path / "train.config"
    path.write_text(body, encoding="utf-8")
    return str(path)


TRAIN_ROWS = [
    {"guid": "t_0", "text_a": "a b c", "text_b": "d e", "label": 1},
    {"guid": "t_1", "text_a": "f", "text_b": "g h", "label": 0},
]
VALID_ROWS = [{"guid": "v_0", "text_a": "x y", "text_b": "", "label": 1}]


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def test_count_tokens_uses_text_pair_columns():
    assert _count_tokens_for_sample({"text_a": "a b c", "text_b": "d e", "label": 1}) == 5


def test_count_tokens_falls_back_to_all_string_fields():
    assert _count_tokens_for_sample({"title": "um dois", "body": "tres", "n": 10}) == 3


def test_count_tokens_ignores_non_dict_samples():
    assert _count_tokens_for_sample("texto solto") == 0


def test_estimate_size_sums_local_files_without_double_counting(tmp_path):
    file_a = _write_file(tmp_path / "a.json", MB)
    file_b = _write_file(tmp_path / "b.json", MB // 2)
    train = _fake_dataset_class({"train": []}, file_list=[file_a, file_b])("cfg", "train")
    valid = _fake_dataset_class({"valid": []}, file_list=[file_a])("cfg", "valid")

    size = _estimate_dataset_size_mb({"train": train, "valid": valid})

    assert size == pytest.approx(1.5)


def test_estimate_size_reads_huggingface_cache_files(tmp_path):
    cache_file = _write_file(tmp_path / "cache.arrow", 2 * MB)

    class HfLike:
        dataset = type("Arrow", (), {"cache_files": [{"filename": cache_file}]})()

    assert _estimate_dataset_size_mb({"train": HfLike()}) == pytest.approx(2.0)


def test_estimate_size_returns_none_without_resolvable_files(tmp_path):
    missing = str(tmp_path / "missing.json")
    dataset = _fake_dataset_class({"train": []}, file_list=[missing])("cfg", "train")

    assert _estimate_dataset_size_mb({"train": dataset}) is None
    assert _estimate_dataset_size_mb({}) is None


def test_summarize_counts_samples_per_partition_and_tokens_per_document():
    cls = _fake_dataset_class({"train": TRAIN_ROWS, "valid": VALID_ROWS})
    partitions = {"train": cls("cfg", "train"), "valid": cls("cfg", "valid")}

    stats = _summarize_dataset_partitions(partitions, dataset_size_mb=3.5)

    assert stats["train_samples"] == 2
    assert stats["valid_samples"] == 1
    assert stats["test_samples"] is None
    assert stats["document_count"] == 3
    assert stats["chunk_count"] == 3
    assert stats["dataset_size_mb"] == 3.5
    # (5 + 3 + 2) tokens em 3 documentos.
    assert stats["tokens_per_document"] == pytest.approx(10 / 3)


def test_summarize_limits_token_sampling_per_split():
    rows = [{"text_a": "a b", "text_b": ""} for _ in range(10)]
    cls = _fake_dataset_class({"train": rows})

    stats = _summarize_dataset_partitions(
        {"train": cls("cfg", "train")}, token_sample_limit_per_split=4,
    )

    assert stats["train_samples"] == 10
    assert stats["tokens_per_document"] == pytest.approx(2.0)


def test_summarize_without_partitions_has_no_statistics():
    stats = _summarize_dataset_partitions({})

    assert stats["tokens_per_document"] is None
    assert stats["dataset_size_mb"] is None
    assert stats["train_samples"] is None
    assert stats["document_count"] == 0


# ---------------------------------------------------------------------------
# Probe Hugging Face
# ---------------------------------------------------------------------------

def test_huggingface_probe_uses_api_size_and_tolerates_missing_test_split(monkeypatch, tmp_path):
    builder_calls = []

    def fake_builder(dataset_id, **kwargs):
        builder_calls.append((dataset_id, kwargs))
        return _FakeBuilder(_FakeInfo(size_in_bytes=2 * MB))

    monkeypatch.setattr("datasets.load_dataset_builder", fake_builder)
    monkeypatch.setattr(
        "dataset.nlp.HuggingFace.HuggingFaceDataset",
        _fake_dataset_class({"train": TRAIN_ROWS, "valid": VALID_ROWS}),
    )
    workflow_config = HuggingFaceWorkflowConfig(
        name="hf", dataset_source="hub", dataset_id="nyu-mll/glue", dataset_config="mrpc",
    )

    stats = _build_huggingface_dataset_probe(
        workflow_config, config_path=_config_file(tmp_path), train_file=None,
    )()

    assert builder_calls[0][0] == "nyu-mll/glue"
    assert builder_calls[0][1]["name"] == "mrpc"
    assert stats["source"] == "hub"
    assert stats["records"] == 2
    assert stats["fields"] == ["guid", "label", "text_a", "text_b"]
    assert stats["dataset_size_mb"] == pytest.approx(2.0)
    assert stats["train_samples"] == 2
    assert stats["valid_samples"] == 1
    assert stats["test_samples"] is None
    assert stats["document_count"] == 3


def test_huggingface_probe_falls_back_to_other_api_size_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "datasets.load_dataset_builder",
        lambda *_a, **_k: _FakeBuilder(_FakeInfo(size_in_bytes=None, dataset_size=MB)),
    )
    monkeypatch.setattr(
        "dataset.nlp.HuggingFace.HuggingFaceDataset", _fake_dataset_class({"train": TRAIN_ROWS}),
    )
    workflow_config = HuggingFaceWorkflowConfig(name="hf", dataset_source="hub", dataset_id="org/data")

    stats = _build_huggingface_dataset_probe(
        workflow_config, config_path=_config_file(tmp_path), train_file=None,
    )()

    assert stats["dataset_size_mb"] == pytest.approx(1.0)


def test_huggingface_probe_local_json_skips_api_and_measures_local_files(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "datasets.load_dataset_builder",
        lambda *_a, **_k: pytest.fail("API do Hub não deve ser consultada para dataset local"),
    )
    data_file = _write_file(tmp_path / "train.json", MB)
    monkeypatch.setattr(
        "dataset.nlp.HuggingFace.HuggingFaceDataset",
        _fake_dataset_class({"train": TRAIN_ROWS}, file_list=[data_file], source="local_json"),
    )
    workflow_config = HuggingFaceWorkflowConfig(
        name="hf-local", dataset_source="local_json", dataset_id="train_task2",
    )

    stats = _build_huggingface_dataset_probe(
        workflow_config, config_path=_config_file(tmp_path), train_file="train_task2",
    )()

    assert stats["source"] == "local_json"
    assert stats["dataset_size_mb"] == pytest.approx(1.0)
    assert stats["train_samples"] == 2
    assert stats["valid_samples"] is None
    assert stats["tokens_per_document"] == pytest.approx(4.0)


def test_huggingface_probe_rejects_empty_training_split(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dataset.nlp.HuggingFace.HuggingFaceDataset", _fake_dataset_class({"train": []}),
    )
    workflow_config = HuggingFaceWorkflowConfig(
        name="hf-empty", dataset_source="local_json", dataset_id="empty",
    )

    probe = _build_huggingface_dataset_probe(
        workflow_config, config_path=_config_file(tmp_path), train_file=None,
    )

    with pytest.raises(ValueError, match="exemplos de treino"):
        probe()


# ---------------------------------------------------------------------------
# Probe do launcher local
# ---------------------------------------------------------------------------

def _launcher_config(tmp_path, dataset_type="FakeLocal"):
    return LauncherWorkflowConfig(
        name="local",
        config_path=_config_file(tmp_path, f"[data]\ndataset_type = {dataset_type}\n"),
        train_dataset="train_task2",
    )


def test_launcher_probe_collects_statistics_from_configured_dataset_class(monkeypatch, tmp_path):
    from dataset import dataset_list

    data_file = _write_file(tmp_path / "train.json", MB)
    monkeypatch.setitem(
        dataset_list,
        "FakeLocal",
        _fake_dataset_class(
            {"train": TRAIN_ROWS, "valid": VALID_ROWS},
            file_list=[data_file],
            source="config",
        ),
    )

    stats = _build_launcher_dataset_probe(_launcher_config(tmp_path))()

    assert stats["source"] == "config"
    assert stats["records"] == 2
    assert stats["fields"] == ["guid", "label", "text_a", "text_b"]
    assert stats["train_samples"] == 2
    assert stats["valid_samples"] == 1
    assert stats["test_samples"] is None
    assert stats["dataset_size_mb"] == pytest.approx(1.0)
    assert stats["tokens_per_document"] == pytest.approx(10 / 3)


def test_launcher_probe_returns_only_source_for_unknown_dataset_type(tmp_path):
    stats = _build_launcher_dataset_probe(_launcher_config(tmp_path, "DoesNotExist"))()

    assert stats == {"source": "config"}


def test_launcher_probe_returns_only_source_when_no_partition_loads(monkeypatch, tmp_path):
    from dataset import dataset_list

    monkeypatch.setitem(dataset_list, "FakeLocal", _fake_dataset_class({}))

    assert _build_launcher_dataset_probe(_launcher_config(tmp_path))() == {"source": "config"}


def test_launcher_ingest_task_exposes_dataset_statistics_in_metrics(monkeypatch, tmp_path):
    from dataset import dataset_list

    monkeypatch.setitem(
        dataset_list, "FakeLocal", _fake_dataset_class({"train": TRAIN_ROWS, "test": VALID_ROWS}),
    )
    config = _launcher_config(tmp_path)

    output = build_launcher_task_functions(config)["ingest_dataset"]()

    dataset = output["metrics"]["dataset"]
    assert dataset["dataset_id"] == "train_task2"
    assert dataset["train_samples"] == 2
    assert dataset["test_samples"] == 1
    assert dataset["valid_samples"] is None
    assert output["artifacts"]["dataset"]["kind"] == "data"
