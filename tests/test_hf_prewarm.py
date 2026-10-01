"""Testes do prewarm de dataset HuggingFace antes do modo offline."""

from unittest.mock import MagicMock, mock_open, patch

from cli.runners import _prewarm_huggingface_dataset

# ---------------------------------------------------------------------------
# _prewarm_huggingface_dataset
# ---------------------------------------------------------------------------

def test_prewarm_calls_load_dataset_for_each_split(tmp_path):
    """load_dataset deve ser chamada uma vez por split."""
    with patch("datasets.load_dataset") as mock_load, \
         patch("utils.paths.PathManager") as mock_pm:
        mock_pm.HF_HUB_CACHE_DIR = tmp_path
        _prewarm_huggingface_dataset("nyu-mll/glue", dataset_config="mrpc")
    assert mock_load.call_count == 3
    splits_called = {c.kwargs["split"] for c in mock_load.call_args_list}
    assert splits_called == {"train", "validation", "test"}


def test_prewarm_passes_dataset_config_as_name(tmp_path):
    """O parâmetro name= deve conter o dataset_config."""
    with patch("datasets.load_dataset") as mock_load, \
         patch("utils.paths.PathManager") as mock_pm:
        mock_pm.HF_HUB_CACHE_DIR = tmp_path
        _prewarm_huggingface_dataset("some/dataset", dataset_config="mrpc")
    for c in mock_load.call_args_list:
        assert c.kwargs.get("name") == "mrpc"


def test_prewarm_omits_name_when_no_config(tmp_path):
    """Sem dataset_config, name= não deve ser passado."""
    with patch("datasets.load_dataset") as mock_load, \
         patch("utils.paths.PathManager") as mock_pm:
        mock_pm.HF_HUB_CACHE_DIR = tmp_path
        _prewarm_huggingface_dataset("some/dataset", dataset_config=None)
    for c in mock_load.call_args_list:
        assert "name" not in c.kwargs


def test_prewarm_tolerates_missing_split(tmp_path):
    """Exceção em um split não deve interromper os demais."""
    call_order: list[str] = []

    def _side_effect(dataset_id, **kwargs):
        split = kwargs["split"]
        call_order.append(split)
        if split == "test":
            raise FileNotFoundError("split ausente")
        return MagicMock()

    with patch("datasets.load_dataset", side_effect=_side_effect), \
         patch("utils.paths.PathManager") as mock_pm:
        mock_pm.HF_HUB_CACHE_DIR = tmp_path
        _prewarm_huggingface_dataset("x/y")

    assert set(call_order) == {"train", "validation", "test"}


def test_prewarm_custom_splits(tmp_path):
    """splits personalizados devem ser respeitados."""
    with patch("datasets.load_dataset") as mock_load, \
         patch("utils.paths.PathManager") as mock_pm:
        mock_pm.HF_HUB_CACHE_DIR = tmp_path
        _prewarm_huggingface_dataset("x/y", splits=("train",))
    assert mock_load.call_count == 1
    assert mock_load.call_args.kwargs["split"] == "train"


# ---------------------------------------------------------------------------
# Integração: prewarm_dataset ocorre ANTES do modo offline
# ---------------------------------------------------------------------------

def test_prewarm_dataset_called_before_offline_mode():
    """_prewarm_huggingface_dataset deve ser invocado antes de _enable_hf_offline_mode."""
    call_sequence: list[str] = []

    def _fake_prewarm_dataset(*a, **kw):
        call_sequence.append("prewarm_dataset")

    def _fake_enable_offline():
        call_sequence.append("enable_offline")

    with patch("cli.runners._resolve_model_id_for_prewarm", return_value="bert-base-uncased"), \
         patch("cli.runners._prewarm_huggingface_assets"), \
         patch("cli.runners._prewarm_huggingface_dataset", side_effect=_fake_prewarm_dataset), \
         patch("cli.runners._enable_hf_offline_mode", side_effect=_fake_enable_offline), \
         patch("cli.runners.validate_paths", return_value=True), \
         patch("cli.runners.run_grid_search", return_value=[]), \
         patch("cli.sla_summary._load_latest_grid_state", return_value={}), \
         patch("cli.sla_summary._emit_sla_execution_summary"), \
         patch("builtins.open", mock_open(read_data="{}")):
        from cli.runners import run_grid_search_experiments
        run_grid_search_experiments(
            base_config_path="config.cfg",
            grid_config_path="grid.json",
            dataset_overrides={
                "hf_dataset_source": "hub",
                "hf_dataset_id": "nyu-mll/glue",
                "hf_dataset_config": "mrpc",
            },
        )

    assert "prewarm_dataset" in call_sequence, "prewarm_dataset não foi chamado"
    assert "enable_offline" in call_sequence, "enable_offline não foi chamado"
    prewarm_idx = call_sequence.index("prewarm_dataset")
    offline_idx = call_sequence.index("enable_offline")
    assert prewarm_idx < offline_idx, (
        f"prewarm_dataset (pos={prewarm_idx}) deve ocorrer ANTES de enable_offline (pos={offline_idx})"
    )
