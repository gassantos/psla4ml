"""
Testes para a configuração do grid search (gridsearch/config/grid_search.json).

Coberturas:
  - Arquivo JSON existe e é válido
  - Chave 'hyperparameters' presente
  - Chave 'optimizer' presente nos hiperparâmetros
  - 'bert_adam' (com underscore) está na lista de otimizadores — fix D1
  - 'bertadam' (sem underscore, nome antigo com bug) NÃO está na lista
  - Todos os valores de optimizer são strings reconhecidas por init_optimizer
  - BL-05: max_seq_length e num_epochs presentes em todos os grids
  - BL-05: param_mapping.json contém os novos mapeamentos
  - BL-05: grid_search_quality.json com 9 combinações de hiperparâmetros (18 com ambientes)
  - BL-05: _estimate_train_time_sec escala com num_epochs e max_seq_length
"""
import json
from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).parents[1] / "gridsearch" / "config"

GRID_SEARCH_CONFIG         = CONFIG_DIR / "grid_search.json"
GRID_SEARCH_TEST_CONFIG    = CONFIG_DIR / "grid_search_test.json"
GRID_SEARCH_QUALITY_CONFIG = CONFIG_DIR / "grid_search_quality.json"
GRID_SEARCH_MULTIENV_CONFIG = CONFIG_DIR / "grid_search_multienv.json"
PARAM_MAPPING_CONFIG       = CONFIG_DIR / "param_mapping.json"

KNOWN_OPTIMIZER_TYPES = {"adam", "adamw", "sgd", "bert_adam"}


@pytest.fixture(scope="module")
def grid_config() -> dict:
    assert GRID_SEARCH_CONFIG.exists(), (
        f"Arquivo de configuração do grid search não encontrado: {GRID_SEARCH_CONFIG}"
    )
    with open(GRID_SEARCH_CONFIG, encoding="utf-8") as f:
        return json.load(f)
@pytest.fixture(scope="module")
def optimizer_values(grid_config) -> list:
    assert "hyperparameters" in grid_config, (
        "Chave 'hyperparameters' não encontrada no grid_search.json"
    )
    assert "optimizer" in grid_config["hyperparameters"], (
        "Chave 'optimizer' não encontrada em hyperparameters"
    )
    return grid_config["hyperparameters"]["optimizer"]


# ---------------------------------------------------------------------------
# Estrutura do arquivo
# ---------------------------------------------------------------------------

class TestGridSearchConfigStructure:
    def test_file_is_valid_json(self, grid_config):
        assert isinstance(grid_config, dict), "grid_search.json deve ser um objeto JSON"

    def test_hyperparameters_key_present(self, grid_config):
        assert "hyperparameters" in grid_config

    def test_optimizer_key_present_in_hyperparameters(self, grid_config):
        assert "optimizer" in grid_config["hyperparameters"]

    def test_optimizer_is_a_list(self, optimizer_values):
        assert isinstance(optimizer_values, list), (
            "O campo 'optimizer' no grid deve ser uma lista de strings"
        )

    def test_optimizer_list_is_non_empty(self, optimizer_values):
        assert len(optimizer_values) > 0


# ---------------------------------------------------------------------------
# Correção do bug de naming (bert_adam vs bertadam)
# ---------------------------------------------------------------------------

class TestOptimizerNamingFix:
    def test_bert_adam_with_underscore_present(self, optimizer_values):
        """Após o fix D1, 'bert_adam' (com underscore) deve estar na lista."""
        assert "bert_adam" in optimizer_values, (
            f"'bert_adam' não encontrado em {optimizer_values}. "
            "Verificar fix aplicado em gridsearch/config/grid_search.json."
        )

    def test_bertadam_without_underscore_absent(self, optimizer_values):
        """O nome com bug ('bertadam', sem underscore) NÃO deve estar na lista."""
        assert "bertadam" not in optimizer_values, (
            f"'bertadam' (nome com bug) ainda presente em {optimizer_values}."
        )

    def test_no_duplicate_bert_adam(self, optimizer_values):
        bert_adam_occurrences = optimizer_values.count("bert_adam")
        assert bert_adam_occurrences == 1, (
            f"'bert_adam' aparece {bert_adam_occurrences} vez(es), esperado 1"
        )


# ---------------------------------------------------------------------------
# Todos os valores de optimizer são reconhecidos
# ---------------------------------------------------------------------------

class TestOptimizerValuesRecognized:
    def test_all_optimizer_values_are_known(self, optimizer_values):
        unknown = [v for v in optimizer_values if v not in KNOWN_OPTIMIZER_TYPES]
        assert not unknown, (
            f"Optimizer(s) desconhecido(s) no grid config: {unknown}. "
            f"Valores conhecidos: {KNOWN_OPTIMIZER_TYPES}"
        )

    @pytest.mark.parametrize("expected_opt", ["adam", "adamw", "sgd", "bert_adam"])
    def test_standard_optimizers_all_present(self, optimizer_values, expected_opt):
        assert expected_opt in optimizer_values, (
            f"Optimizer '{expected_opt}' ausente no grid config"
        )


# ---------------------------------------------------------------------------
# BL-05 — param_mapping.json: novos hiperparâmetros mapeados
# ---------------------------------------------------------------------------

class TestParamMappingBL05:
    @pytest.fixture(scope="class")
    def mapping(self):
        assert PARAM_MAPPING_CONFIG.exists()
        with open(PARAM_MAPPING_CONFIG, encoding="utf-8") as f:
            return json.load(f)["mapping"]

    def test_num_epochs_present(self, mapping):
        assert "num_epochs" in mapping, "num_epochs ausente em param_mapping.json"

    def test_num_epochs_maps_to_train_epoch(self, mapping):
        entry = mapping["num_epochs"]
        assert entry["section"] == "train"
        assert entry["key"] == "epoch"

    def test_max_seq_length_present(self, mapping):
        assert "max_seq_length" in mapping, "max_seq_length ausente em param_mapping.json"

    def test_max_seq_length_maps_to_data_section(self, mapping):
        entry = mapping["max_seq_length"]
        assert entry["section"] == "data"
        assert entry["key"] == "max_seq_length"


# ---------------------------------------------------------------------------
# BL-05 — grid_search.json: espaço expandido
# ---------------------------------------------------------------------------

class TestGridSearchExpandedBL05:
    def test_max_seq_length_present(self, grid_config):
        assert "max_seq_length" in grid_config["hyperparameters"], \
            "max_seq_length ausente em grid_search.json"

    def test_max_seq_length_has_three_values(self, grid_config):
        values = grid_config["hyperparameters"]["max_seq_length"]
        assert set(values) == {128, 256, 512}, f"Esperado [128,256,512], obtido {values}"

    def test_num_epochs_present(self, grid_config):
        assert "num_epochs" in grid_config["hyperparameters"], \
            "num_epochs ausente em grid_search.json"

    def test_num_epochs_has_three_values(self, grid_config):
        values = grid_config["hyperparameters"]["num_epochs"]
        assert set(values) == {2, 3, 5}, f"Esperado [2,3,5], obtido {values}"

    def test_learning_rate_expanded_to_five_values(self, grid_config):
        lr_values = grid_config["hyperparameters"]["learning_rate"]
        assert len(lr_values) == 5, f"Esperado 5 valores de lr, obtido {len(lr_values)}"
        assert 5e-6 in lr_values, "5e-6 ausente nos learning_rates"

    def test_time_estimation_has_reference_num_epochs(self, grid_config):
        te = grid_config["_meta"]["time_estimation"]
        assert "reference_num_epochs" in te

    def test_time_estimation_has_reference_max_seq_length(self, grid_config):
        te = grid_config["_meta"]["time_estimation"]
        assert "reference_max_seq_length" in te

    def test_time_estimation_has_seq_length_factors(self, grid_config):
        te = grid_config["_meta"]["time_estimation"]
        assert "max_seq_length_factors" in te
        factors = te["max_seq_length_factors"]
        assert "128" in factors and "256" in factors and "512" in factors


# ---------------------------------------------------------------------------
# BL-05 — grid_search_test.json: dimensões novas com valores únicos
# ---------------------------------------------------------------------------

class TestGridSearchTestBL05:
    @pytest.fixture(scope="class")
    def test_cfg(self):
        assert GRID_SEARCH_TEST_CONFIG.exists()
        with open(GRID_SEARCH_TEST_CONFIG, encoding="utf-8") as f:
            return json.load(f)

    def test_max_seq_length_present_with_single_value(self, test_cfg):
        values = test_cfg["hyperparameters"].get("max_seq_length", [])
        assert len(values) >= 1, "max_seq_length ausente em grid_search_test.json"

    def test_num_epochs_present_with_single_value(self, test_cfg):
        values = test_cfg["hyperparameters"].get("num_epochs", [])
        assert len(values) >= 1, "num_epochs ausente em grid_search_test.json"

    def test_total_combinations_stays_manageable(self, test_cfg):
        from gridsearch import generate_parameter_grid
        combos = generate_parameter_grid(test_cfg)
        assert len(combos) <= 32, \
            f"grade de teste deve ter ≤32 combinações, obtido {len(combos)}"


# ---------------------------------------------------------------------------
# BL-05 — grid_search_quality.json: 9 combinações exatas
# ---------------------------------------------------------------------------

class TestGridSearchQualityBL05:
    @pytest.fixture(scope="class")
    def quality_cfg(self):
        assert GRID_SEARCH_QUALITY_CONFIG.exists(), \
            "grid_search_quality.json não encontrado"
        with open(GRID_SEARCH_QUALITY_CONFIG, encoding="utf-8") as f:
            return json.load(f)

    def test_quality_config_exists(self, quality_cfg):
        assert isinstance(quality_cfg, dict)

    def test_max_seq_length_three_values(self, quality_cfg):
        values = quality_cfg["hyperparameters"]["max_seq_length"]
        assert set(values) == {128, 256, 512}

    def test_num_epochs_three_values(self, quality_cfg):
        values = quality_cfg["hyperparameters"]["num_epochs"]
        assert set(values) == {2, 3, 5}

    def test_fixed_resource_params(self, quality_cfg):
        hp = quality_cfg["hyperparameters"]
        assert len(hp["learning_rate"]) == 1
        assert len(hp["batch_size"]) == 1
        assert len(hp["optimizer"]) == 1
        assert len(hp["dropout"]) == 1
        assert len(hp["seed"]) == 1

    def test_hyperparameters_only_nine_combinations(self, quality_cfg):
        """BL-W2: sem a dimensao environment, o espaco de hiperparametros permanece com 9 combinacoes."""
        from gridsearch import generate_parameter_grid
        combos = generate_parameter_grid(quality_cfg["hyperparameters"])
        assert len(combos) == 9, f"Esperado 9 combinações de hiperparâmetros, obtido {len(combos)}"

    def test_exactly_nine_combinations(self, quality_cfg):
        """Sem bloco environments, a config completa gera 9 combinações."""
        from gridsearch import generate_parameter_grid
        combos = generate_parameter_grid(quality_cfg)
        assert len(combos) == 9, f"Esperado 9 combinações, obtido {len(combos)}"

    def test_meta_count_matches_actual(self, quality_cfg):
        from gridsearch import generate_parameter_grid
        combos = generate_parameter_grid(quality_cfg)
        expected = quality_cfg["_meta"]["total_combinations"]["with_environments"]
        assert len(combos) == expected


# ---------------------------------------------------------------------------
# BL-05 — _estimate_train_time_sec: fatores de num_epochs e max_seq_length
# ---------------------------------------------------------------------------

class TestTimeEstimationScalingBL05:
    """Verifica que o pré-filtro SLA escala corretamente com as novas dimensões."""

    @pytest.fixture(scope="class")
    def test_cfg(self):
        with open(GRID_SEARCH_TEST_CONFIG, encoding="utf-8") as f:
            return json.load(f)

    def _estimate(self, params, cfg):
        from gridsearch.core import _estimate_train_time_sec
        return _estimate_train_time_sec(params, cfg)

    def test_baseline_without_new_params(self, test_cfg):
        """Sem num_epochs e max_seq_length, retorna baseline × fatores padrão."""
        t = self._estimate({"batch_size": 16, "optimizer": "adam"}, test_cfg)
        assert t is not None and t > 0

    def test_double_epochs_doubles_time(self, test_cfg):
        """num_epochs=6 deve ser 2× mais lento que num_epochs=3."""
        t3 = self._estimate({"batch_size": 16, "optimizer": "adam", "num_epochs": 3}, test_cfg)
        t6 = self._estimate({"batch_size": 16, "optimizer": "adam", "num_epochs": 6}, test_cfg)
        assert t3 is not None and t6 is not None
        assert t6 == pytest.approx(t3 * 2.0, rel=1e-6)

    def test_seq512_is_4x_slower_than_seq256(self, test_cfg):
        """max_seq_length=512 deve ser 4× mais lento que max_seq_length=256 (O(n²))."""
        t256 = self._estimate({"batch_size": 16, "optimizer": "adam", "max_seq_length": 256}, test_cfg)
        t512 = self._estimate({"batch_size": 16, "optimizer": "adam", "max_seq_length": 512}, test_cfg)
        assert t256 is not None and t512 is not None
        assert t512 == pytest.approx(t256 * 4.0, rel=1e-6)

    def test_seq128_is_quarter_of_seq256(self, test_cfg):
        """max_seq_length=128 deve ser 0.25× o tempo de max_seq_length=256."""
        t256 = self._estimate({"batch_size": 16, "optimizer": "adam", "max_seq_length": 256}, test_cfg)
        t128 = self._estimate({"batch_size": 16, "optimizer": "adam", "max_seq_length": 128}, test_cfg)
        assert t256 is not None and t128 is not None
        assert t128 == pytest.approx(t256 * 0.25, rel=1e-6)

    def test_combined_seq512_epochs5_scales_correctly(self, test_cfg):
        """max_seq_length=512 + num_epochs=5 deve ser ~6.67× o baseline (3 épocas, 256 tokens)."""
        t_base = self._estimate(
            {"batch_size": 16, "optimizer": "adam",
             "max_seq_length": 256, "num_epochs": 3}, test_cfg
        )
        t_max = self._estimate(
            {"batch_size": 16, "optimizer": "adam",
             "max_seq_length": 512, "num_epochs": 5}, test_cfg
        )
        assert t_base is not None and t_max is not None
        expected_scale = 4.0 * (5.0 / 3.0)
        assert t_max == pytest.approx(t_base * expected_scale, rel=1e-5)
