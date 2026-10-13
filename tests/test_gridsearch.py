"""
Testes pytest para o módulo gridsearch.

Coberturas:
  - Importações dos símbolos públicos do pacote
  - Geração de grade de hiperparâmetros (generate_parameter_grid)
  - Validação de disponibilidade de memória (check_memory_availability)
  - Estimativa de requisitos de memória (estimate_memory_requirements)
  - Filtragem de metadados de configuração (filter_grid_config)
  - Carregamento dos arquivos JSON de configuração do grid
  - Análise de resultados mock (compute_descriptive_statistics, rank_configurations)
"""

import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Importações opcionais — marcados como skip se o módulo não estiver pronto
# ---------------------------------------------------------------------------

gridsearch = pytest.importorskip("gridsearch", reason="Módulo gridsearch não disponível")

from gridsearch import run_grid_search, generate_parameter_grid, analyze_results  # noqa: E402
from gridsearch.core import (  # noqa: E402
    _prefilter_param_grid_by_execution_sla,
    _estimate_train_time_sec,
    MAX_SLA_REJECTED_SAMPLES,
)
from gridsearch.utils import (  # noqa: E402
    check_memory_availability,
    estimate_memory_requirements,
    filter_grid_config,
)
from gridsearch.analysis import (  # noqa: E402
    compute_descriptive_statistics,
    analyze_correlations,
    rank_configurations,
)

GRID_SEARCH_JSON = Path(__file__).parents[1] / "gridsearch" / "config" / "grid_search.json"
GRID_SEARCH_TEST_JSON = Path(__file__).parents[1] / "gridsearch" / "config" / "grid_search_test.json"
GRID_SEARCH_MINIMAL_JSON = Path(__file__).parents[1] / "gridsearch" / "config" / "grid_search_minimal.json"
GRID_SEARCH_MULTIENV_JSON = Path(__file__).parents[1] / "gridsearch" / "config" / "grid_search_multienv.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_grid():
    return {"learning_rate": [1e-5, 2e-5], "batch_size": [8, 16]}


@pytest.fixture(scope="module")
def mock_results():
    return [
        {
            "grid_experiment_idx": 0,
            "grid_params": {"learning_rate": 1e-5, "batch_size": 8},
            "status": "success",
            "resources": {
                "train_time_sec": 1234.5,
                "energy_kwh": 0.025,
                "peak_ram_mb": 8192.0,
            },
        },
        {
            "grid_experiment_idx": 1,
            "grid_params": {"learning_rate": 2e-5, "batch_size": 16},
            "status": "success",
            "resources": {
                "train_time_sec": 1156.3,
                "energy_kwh": 0.023,
                "peak_ram_mb": 12288.0,
            },
        },
        {
            "grid_experiment_idx": 2,
            "grid_params": {"learning_rate": 1e-5, "batch_size": 16},
            "status": "failed",
            "error": "Out of memory",
        },
    ]


# ---------------------------------------------------------------------------
# Teste 1: Importações
# ---------------------------------------------------------------------------

class TestImports:
    def test_run_grid_search_importable(self):
        assert callable(run_grid_search)

    def test_generate_parameter_grid_importable(self):
        assert callable(generate_parameter_grid)

    def test_analyze_results_importable(self):
        assert callable(analyze_results)

    def test_utils_importable(self):
        assert callable(check_memory_availability)
        assert callable(estimate_memory_requirements)
        assert callable(filter_grid_config)

    def test_analysis_importable(self):
        assert callable(compute_descriptive_statistics)
        assert callable(analyze_correlations)
        assert callable(rank_configurations)


# ---------------------------------------------------------------------------
# Teste 2: Geração de combinações
# ---------------------------------------------------------------------------

class TestGenerateParameterGrid:
    def test_correct_number_of_combinations(self, simple_grid):
        combos = generate_parameter_grid(simple_grid)
        assert len(combos) == 4, f"Esperado 4 combinações (2x2), obtido {len(combos)}"

    def test_combinations_are_dicts(self, simple_grid):
        combos = generate_parameter_grid(simple_grid)
        for combo in combos:
            assert isinstance(combo, dict)

    def test_all_keys_present_in_each_combo(self, simple_grid):
        combos = generate_parameter_grid(simple_grid)
        for combo in combos:
            assert "learning_rate" in combo
            assert "batch_size" in combo

    def test_single_param_returns_all_values(self):
        grid = {"learning_rate": [1e-5, 2e-5, 3e-5]}
        combos = generate_parameter_grid(grid)
        assert len(combos) == 3

    def test_three_params_cartesian_product(self):
        grid = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
        combos = generate_parameter_grid(grid)
        assert len(combos) == 8  # 2 * 2 * 2

    @pytest.mark.skipif(not GRID_SEARCH_MULTIENV_JSON.exists(), reason="grid_search_multienv.json não encontrado")
    def test_multienv_grid_expands_environment_dimension(self):
        with open(GRID_SEARCH_MULTIENV_JSON, encoding="utf-8") as f:
            config = json.load(f)

        combos = generate_parameter_grid(config)

        expected = config["_meta"]["total_combinations"]["with_environments"]
        assert len(combos) == expected
        assert all("environment" not in combo for combo in combos)


# ---------------------------------------------------------------------------
# Teste 3: Validação de memória
# ---------------------------------------------------------------------------

class TestMemoryValidation:
    def test_check_memory_returns_tuple(self):
        result = check_memory_availability(parallel=2, batch_size=16)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_check_memory_first_element_is_bool(self):
        is_safe, _ = check_memory_availability(parallel=2, batch_size=16)
        assert isinstance(is_safe, bool)

    def test_check_memory_second_element_is_str(self):
        _, message = check_memory_availability(parallel=2, batch_size=16)
        assert isinstance(message, str)

    def test_estimate_memory_returns_float(self):
        estimated_gb = estimate_memory_requirements(parallel=2, batch_size=16)
        assert isinstance(estimated_gb, (int, float))

    def test_estimate_memory_positive(self):
        estimated_gb = estimate_memory_requirements(parallel=1, batch_size=8)
        assert estimated_gb > 0


# ---------------------------------------------------------------------------
# Teste 4: Filtragem de configuração
# ---------------------------------------------------------------------------

class TestFilterGridConfig:
    def test_metadata_keys_removed(self):
        config = {
            "description": "Test grid",
            "notes": ["test note"],
            "learning_rate": [1e-5],
            "batch_size": [16],
        }
        filtered = filter_grid_config(config)
        assert "description" not in filtered
        assert "notes" not in filtered

    def test_hyperparameter_keys_preserved(self):
        config = {
            "description": "To be removed",
            "learning_rate": [1e-5],
            "batch_size": [16],
        }
        filtered = filter_grid_config(config)
        assert "learning_rate" in filtered
        assert "batch_size" in filtered

    def test_empty_config_returns_empty(self):
        filtered = filter_grid_config({})
        assert isinstance(filtered, dict)


# ---------------------------------------------------------------------------
# Teste 5: Filtro de SLA
# ---------------------------------------------------------------------------

class TestExecutionSlaPrefilter:
    def test_prefilter_rejects_by_peak_ram_mb(self):
        indexed = [
            (0, {"batch_size": 8}),
            (1, {"batch_size": 32}),
        ]
        eligible, info = _prefilter_param_grid_by_execution_sla(
            indexed_param_grid=indexed,
            constraints={"peak_ram_mb": 3000},
            grid_config={},
        )

        assert len(eligible) == 0
        assert info["rejected_total_experiments"] == 2
        assert info["rejected_by_metric"]["peak_ram_mb"] == 2
        assert len(info["rejected_samples"]) == 2
        sample = info["rejected_samples"][0]
        assert sample["metric"] == "peak_ram_mb"
        assert sample["estimated_value"] > sample["threshold"]
        assert "params" in sample

    def test_prefilter_keeps_when_within_peak_ram_mb(self):
        indexed = [
            (0, {"batch_size": 8}),
            (1, {"batch_size": 16}),
        ]
        eligible, info = _prefilter_param_grid_by_execution_sla(
            indexed_param_grid=indexed,
            constraints={"peak_ram_mb": 5200},
            grid_config={},
        )

        assert len(eligible) == 2
        assert info["rejected_total_experiments"] == 0

    def test_train_time_constraint_is_non_evaluable_without_metadata(self):
        indexed = [(0, {"batch_size": 16})]
        eligible, info = _prefilter_param_grid_by_execution_sla(
            indexed_param_grid=indexed,
            constraints={"train_time_sec": 1200},
            grid_config={},
        )

        assert len(eligible) == 1
        assert "train_time_sec" in info["non_evaluable_constraints"]

    def test_estimate_train_time_uses_metadata_and_batch_size_scaling(self):
        grid_config = {
            "_meta": {
                "per_experiment_train_time_sec": 1800,
                "time_estimation": {
                    "baseline_train_time_sec": 1800,
                    "reference_batch_size": 16,
                },
            }
        }
        t_bs8 = _estimate_train_time_sec({"batch_size": 8}, grid_config)
        t_bs16 = _estimate_train_time_sec({"batch_size": 16}, grid_config)
        t_bs32 = _estimate_train_time_sec({"batch_size": 32}, grid_config)

        assert t_bs8 == pytest.approx(3600.0)
        assert t_bs16 == pytest.approx(1800.0)
        assert t_bs32 == pytest.approx(900.0)

    def test_estimate_train_time_applies_optimizer_factor(self):
        grid_config = {
            "_meta": {
                "time_estimation": {
                    "baseline_train_time_sec": 1800,
                    "reference_batch_size": 16,
                    "optimizer_factors": {
                        "adamw": 1.0,
                        "bert_adam": 1.1,
                    },
                }
            }
        }

        adamw = _estimate_train_time_sec(
            {"batch_size": 16, "optimizer": "adamw"},
            grid_config,
        )
        bert_adam = _estimate_train_time_sec(
            {"batch_size": 16, "optimizer": "bert_adam"},
            grid_config,
        )

        assert adamw == pytest.approx(1800.0)
        assert bert_adam == pytest.approx(1980.0)

    def test_estimate_train_time_prefers_environment_baseline(self):
        grid_config = {
            "_meta": {
                "time_estimation": {
                    "baseline_train_time_sec": 1800,
                    "reference_batch_size": 16,
                }
            },
            "environments": {
                "details": {
                    "azure": {
                        "estimated_time_hours": {
                            "per_experiment": 0.5,
                        }
                    },
                    "gcp": {
                        "estimated_time_hours": {
                            "per_experiment": 0.75,
                        }
                    },
                }
            },
        }

        azure = _estimate_train_time_sec(
            {"batch_size": 16, "environment": "azure"},
            grid_config,
        )
        gcp = _estimate_train_time_sec(
            {"batch_size": 16, "environment": "gcp"},
            grid_config,
        )

        assert azure == pytest.approx(1800.0)
        assert gcp == pytest.approx(2700.0)

    def test_prefilter_rejected_samples_truncated_with_limit(self):
        indexed = [(i, {"batch_size": 32}) for i in range(MAX_SLA_REJECTED_SAMPLES + 10)]
        eligible, info = _prefilter_param_grid_by_execution_sla(
            indexed_param_grid=indexed,
            constraints={"peak_ram_mb": 1000},
            grid_config={},
        )

        assert len(eligible) == 0
        assert len(info["rejected_samples"]) == MAX_SLA_REJECTED_SAMPLES
        assert info["rejected_samples_truncated"] == 10

    def test_prefilter_logs_ranking_and_samples(self, caplog):
        indexed = [
            (0, {"batch_size": 32}),
            (1, {"batch_size": 32}),
            (2, {"batch_size": 8}),
        ]

        with caplog.at_level("INFO"):
            _prefilter_param_grid_by_execution_sla(
                indexed_param_grid=indexed,
                constraints={"peak_ram_mb": 1000},
                grid_config={},
            )

        assert "ranking de rejeicoes por metrica" in caplog.text
        assert "exemplo rejeitado idx=" in caplog.text


# ---------------------------------------------------------------------------
# Teste 6: Carregamento de arquivos JSON
# ---------------------------------------------------------------------------

class TestJsonConfigLoading:
    @pytest.mark.skipif(not GRID_SEARCH_TEST_JSON.exists(), reason="grid_search_test.json não encontrado")
    def test_test_json_generates_combinations(self):
        with open(GRID_SEARCH_TEST_JSON, encoding="utf-8") as f:
            config = json.load(f)
        assert "hyperparameters" in config
        combos = generate_parameter_grid(config["hyperparameters"])
        assert len(combos) > 0

    @pytest.mark.skipif(not GRID_SEARCH_JSON.exists(), reason="grid_search.json não encontrado")
    def test_full_json_generates_combinations(self):
        with open(GRID_SEARCH_JSON, encoding="utf-8") as f:
            config = json.load(f)
        assert "hyperparameters" in config
        combos = generate_parameter_grid(config["hyperparameters"])
        assert len(combos) > 0

    @pytest.mark.parametrize(
        "config_path",
        [GRID_SEARCH_TEST_JSON, GRID_SEARCH_JSON, GRID_SEARCH_MINIMAL_JSON],
    )
    def test_grid_configs_define_train_time_baseline(self, config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        assert "_meta" in config
        assert "per_experiment_train_time_sec" in config["_meta"]
        assert float(config["_meta"]["per_experiment_train_time_sec"]) > 0
        assert "time_estimation" in config["_meta"]
        assert float(config["_meta"]["time_estimation"]["baseline_train_time_sec"]) > 0


# ---------------------------------------------------------------------------
# Teste 7: Análise de resultados mock
# ---------------------------------------------------------------------------

class TestAnalyzeResults:
    def test_descriptive_stats_returns_dict(self, mock_results):
        stats = compute_descriptive_statistics(mock_results)
        assert isinstance(stats, dict)

    def test_descriptive_stats_has_train_time(self, mock_results):
        stats = compute_descriptive_statistics(mock_results)
        assert "train_time" in stats

    def test_descriptive_stats_counts_success_and_failure(self, mock_results):
        stats = compute_descriptive_statistics(mock_results)
        assert stats.get("successful_experiments") == 2
        assert stats.get("failed_experiments") == 1

    def test_rank_configurations_returns_list(self, mock_results):
        ranked = rank_configurations(mock_results, metrics=["train_time_sec"])
        assert isinstance(ranked, list)

    def test_rank_configurations_non_empty(self, mock_results):
        ranked = rank_configurations(mock_results, metrics=["train_time_sec"])
        assert len(ranked) > 0

    def test_rank_configurations_has_params_key(self, mock_results):
        ranked = rank_configurations(mock_results, metrics=["train_time_sec"])
        for item in ranked:
            assert "params" in item


# ---------------------------------------------------------------------------
# BL-06 — detect_collinear_metrics e metric_correlation_matrix
# ---------------------------------------------------------------------------

class TestCollinearityDetectionBL06:
    """Testes para detect_collinear_metrics (BL-06 / Seção 4 do artigo PSLA4ML)."""

    def _make(self, idx, time, energy, co2, cost):
        return {
            "status": "success",
            "grid_experiment_idx": idx,
            "grid_params": {"lr": 1e-5, "bs": 8},
            "resources": {
                "train_time_sec":   time,
                "energy_kwh":       energy,
                "emissions_kg_co2": co2,
                "cost_usd":         cost,
            },
        }

    @pytest.fixture
    def perfectly_collinear(self):
        """4 métricas perfeitamente correlacionadas (r = 1.0)."""
        return [self._make(i, t, t * 0.001, t * 0.0001, t * 0.0002)
                for i, t in enumerate([100, 200, 300, 400, 500, 600])]

    @pytest.fixture
    def uncorrelated(self):
        """Métricas não correlacionadas entre si."""
        import math
        return [
            self._make(0, 100, 0.9,  0.09, 0.1),
            self._make(1, 200, 0.1,  0.80, 0.9),
            self._make(2, 300, 0.7,  0.02, 0.5),
            self._make(3, 400, 0.3,  0.60, 0.2),
            self._make(4, 500, 0.6,  0.04, 0.8),
            self._make(5, 600, 0.2,  0.70, 0.3),
        ]

    def test_import_detect_collinear_metrics(self):
        from gridsearch.analysis.correlations import detect_collinear_metrics
        assert callable(detect_collinear_metrics)

    def test_import_via_analysis_package(self):
        from gridsearch.analysis import detect_collinear_metrics, CollinearityReport
        assert callable(detect_collinear_metrics)
        assert CollinearityReport is not None

    def test_collinearity_report_dataclass(self):
        from gridsearch.analysis import CollinearityReport
        r = CollinearityReport()
        assert r.has_collinearity is False
        assert r.collinear_pairs == []
        assert r.correlation_matrix == {}

    def test_detects_perfect_collinearity(self, perfectly_collinear):
        from gridsearch.analysis.correlations import detect_collinear_metrics
        metrics = ["train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"]
        report = detect_collinear_metrics(perfectly_collinear, metrics=metrics, threshold=0.95)
        assert report.has_collinearity is True
        assert len(report.collinear_pairs) == 6  # C(4,2) = 6 pares

    def test_no_collinearity_below_threshold(self, uncorrelated):
        from gridsearch.analysis.correlations import detect_collinear_metrics
        metrics = ["train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"]
        report = detect_collinear_metrics(uncorrelated, metrics=metrics, threshold=0.95)
        assert report.has_collinearity is False
        assert report.collinear_pairs == []

    def test_pairs_sorted_by_abs_r_descending(self, perfectly_collinear):
        from gridsearch.analysis.correlations import detect_collinear_metrics
        metrics = ["train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"]
        report = detect_collinear_metrics(perfectly_collinear, metrics=metrics, threshold=0.9)
        if len(report.collinear_pairs) > 1:
            rs = [abs(r) for _, _, r in report.collinear_pairs]
            assert rs == sorted(rs, reverse=True)

    def test_default_metrics_are_paper_four(self):
        from gridsearch.analysis.correlations import detect_collinear_metrics
        report = detect_collinear_metrics([])
        assert set(report.metrics) == {
            "train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"
        }

    def test_report_contains_n_samples(self, perfectly_collinear):
        from gridsearch.analysis.correlations import detect_collinear_metrics
        report = detect_collinear_metrics(perfectly_collinear, metrics=["train_time_sec", "energy_kwh"])
        assert report.n_samples == len(perfectly_collinear)

    def test_ignores_failed_results(self, perfectly_collinear):
        from gridsearch.analysis.correlations import detect_collinear_metrics
        with_failed = perfectly_collinear + [
            {"status": "failed", "resources": {"train_time_sec": 9999, "energy_kwh": 9999}}
        ]
        report_clean  = detect_collinear_metrics(perfectly_collinear, metrics=["train_time_sec", "energy_kwh"])
        report_failed = detect_collinear_metrics(with_failed,         metrics=["train_time_sec", "energy_kwh"])
        assert report_clean.n_samples == report_failed.n_samples

    def test_empty_results_returns_no_collinearity(self):
        from gridsearch.analysis.correlations import detect_collinear_metrics
        report = detect_collinear_metrics([], metrics=["train_time_sec", "energy_kwh"])
        assert report.has_collinearity is False
        assert report.n_samples == 0

    def test_threshold_parameter_respected(self, perfectly_collinear):
        from gridsearch.analysis.correlations import detect_collinear_metrics
        metrics = ["train_time_sec", "energy_kwh"]
        report_95 = detect_collinear_metrics(perfectly_collinear, metrics=metrics, threshold=0.95)
        report_99 = detect_collinear_metrics(perfectly_collinear, metrics=metrics, threshold=0.999)
        # threshold menor inclui mais pares (ou igual); threshold maior é mais restritivo
        assert len(report_95.collinear_pairs) >= len(report_99.collinear_pairs)

    def test_diagonal_of_matrix_is_one(self, perfectly_collinear):
        from gridsearch.analysis.correlations import metric_correlation_matrix
        metrics = ["train_time_sec", "energy_kwh"]
        matrix = metric_correlation_matrix(perfectly_collinear, metrics)
        for m in metrics:
            assert matrix[m][m] == pytest.approx(1.0)

    def test_matrix_is_symmetric(self, perfectly_collinear):
        from gridsearch.analysis.correlations import metric_correlation_matrix
        metrics = ["train_time_sec", "energy_kwh", "cost_usd"]
        matrix = metric_correlation_matrix(perfectly_collinear, metrics)
        for m_a in metrics:
            for m_b in metrics:
                assert matrix[m_a][m_b] == pytest.approx(
                    matrix[m_b][m_a], rel=1e-9
                )

    def test_skyband_report_includes_collinearity_warning(self, perfectly_collinear):
        """skyband_report deve incluir bloco de aviso quando há multicolinearidade."""
        from gridsearch.skyband import skyband_report
        metrics = ["train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"]
        report = skyband_report(
            perfectly_collinear, k=1, metrics=metrics,
            collinearity_threshold=0.95,
        )
        assert "AVISO DE MULTICOLINEARIDADE" in report
        assert "↔" in report

    def test_skyband_report_no_warning_when_uncorrelated(self, uncorrelated):
        """skyband_report NÃO deve incluir aviso quando não há colinearidade."""
        from gridsearch.skyband import skyband_report
        metrics = ["train_time_sec", "energy_kwh", "emissions_kg_co2", "cost_usd"]
        report = skyband_report(
            uncorrelated, k=1, metrics=metrics,
            collinearity_threshold=0.95,
        )
        assert "AVISO DE MULTICOLINEARIDADE" not in report

    def test_skyband_report_paper_scenario(self):
        """Reproduz o cenário do artigo: métricas r>0.99 em hardware fixo."""
        from gridsearch.analysis.correlations import detect_collinear_metrics
        # Simula execuções em CPU fixo: energia ∝ tempo, CO2 ∝ energia, custo ∝ energia
        results = [
            self._make(i, t,
                       t * 4.64e-5,           # energy = t × taxa_cpu
                       t * 4.64e-5 * 0.2677,  # CO2 = energy × fator_region
                       t / 3600 * 0.10)        # cost = (t/3600) × $0.10/h
            for i, t in enumerate([5410, 6805, 6793, 3701, 3662, 3634, 30, 32, 36])
        ]
        report = detect_collinear_metrics(results, threshold=0.95)
        assert report.has_collinearity is True
        # No artigo, todos os 6 pares têm r > 0.99
        assert len(report.collinear_pairs) == 6


# ---------------------------------------------------------------------------
# Ponto de entrada manual (não executado pelo pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], check=True)
