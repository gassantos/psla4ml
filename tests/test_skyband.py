"""
Testes pytest para o módulo gridsearch.skyband.

Coberturas:
  - Importações de todos os símbolos públicos (inclui BL-07)
  - sla_filter: admissão, rejeição, filtro vazio, sem resultados válidos
  - dominates: casos de dominância estrita, igualdade, não-dominância
  - pareto_front (k=1): frente correta, sem duplicatas, campo domination_count=0
  - skyband_query: k=1, k>1, com SLA, métricas customizadas, k inválido
  - compare_skyband_vs_ranking: Jaccard, conjuntos only_in_skyband / only_in_scalar
  - skyband_report: formato textual, conteúdo esperado
  - Integração com sla_profiles.json e grid_search_multienv.json
  - BL-07: QUALITY_METRICS, _get_minimize_flag, _extract_metric_value com evaluation,
           skyband_query com include_quality_metrics
"""

import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Importação do módulo
# ---------------------------------------------------------------------------

skyband_mod = pytest.importorskip("gridsearch.skyband", reason="gridsearch.skyband não disponível")

from gridsearch.skyband import (   # noqa: E402
    sla_filter,
    dominates,
    pareto_front,
    skyband_query,
    compare_skyband_vs_ranking,
    skyband_report,
    DEFAULT_METRICS,
    DEFAULT_MINIMIZE,
)

# Caminhos dos arquivos de configuração
CONFIG_DIR = Path(__file__).parents[1] / "gridsearch" / "config"
SLA_PROFILES_PATH   = CONFIG_DIR / "sla_profiles.json"
MULTIENV_PATH       = CONFIG_DIR / "grid_search_multienv.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_results():
    """
    7 experimentos sintéticos cobrindo os 5 critérios padrão.

    Relações de dominância (train_time_sec × cost_usd):
      Exp 2 (1800s, $2.00) é dominado por Exp 0 (3600s, $1.00)?
        → Não: Exp 0 tem tempo maior. São não-comparáveis.
      Exp 4 (3600s, $1.50) é dominado por Exp 0 (3600s, $1.00)?
        → Sim: tempo igual, custo maior → Exp 0 domina Exp 4.
      Exp 5 (2400s, $3.00) é dominado por Exp 0 (3600s, $1.00)?
        → Não: tempo menor em Exp 5. São não-comparáveis.
    """
    return [
        # idx  lr     bs  time   cost  energy  ram    co2
        _make(0, 1e-5,  8, 3600, 1.00, 0.050,  4096, 0.0020),
        _make(1, 2e-5, 16, 7200, 0.50, 0.030,  8192, 0.0010),
        _make(2, 3e-5, 32, 1800, 2.00, 0.080,  2048, 0.0040),
        _make(3, 1e-5, 32, 9000, 0.20, 0.020, 16384, 0.0005),
        _make(4, 2e-5,  8, 3600, 1.50, 0.060,  4096, 0.0030),
        _make(5, 3e-5, 16, 2400, 3.00, 0.100,  6144, 0.0060),
        _make(6, 1e-5, 16, 4800, 0.80, 0.040, 10240, 0.0015),
    ]


@pytest.fixture(scope="module")
def mock_with_failed(mock_results):
    """Adiciona um experimento com status 'failed'."""
    failed = {
        "status": "failed",
        "grid_experiment_idx": 99,
        "grid_params": {"lr": 1e-5, "bs": 8},
        "resources": {},
        "error": "CUDA OOM",
    }
    return list(mock_results) + [failed]


@pytest.fixture(scope="module")
def sla_profiles():
    assert SLA_PROFILES_PATH.exists(), f"Arquivo não encontrado: {SLA_PROFILES_PATH}"
    with open(SLA_PROFILES_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def multienv_config():
    assert MULTIENV_PATH.exists(), f"Arquivo não encontrado: {MULTIENV_PATH}"
    with open(MULTIENV_PATH, encoding="utf-8") as f:
        return json.load(f)


def _make(idx, lr, bs, time, cost, energy, ram, co2):
    return {
        "status": "success",
        "grid_experiment_idx": idx,
        "grid_params": {"lr": lr, "bs": bs},
        "resources": {
            "train_time_sec": time,
            "cost_usd":       cost,
            "energy_kwh":     energy,
            "total_gflops":   ram,
            "emissions_kg_co2": co2,
        },
    }


# ---------------------------------------------------------------------------
# Importações públicas
# ---------------------------------------------------------------------------

class TestImports:
    def test_default_metrics_is_list(self):
        assert isinstance(DEFAULT_METRICS, list)

    def test_default_metrics_has_five_entries(self):
        assert len(DEFAULT_METRICS) == 5

    def test_default_minimize_all_true(self):
        assert all(DEFAULT_MINIMIZE)

    def test_default_metrics_and_minimize_same_length(self):
        assert len(DEFAULT_METRICS) == len(DEFAULT_MINIMIZE)

    def test_expected_metric_names_present(self):
        expected = {"train_time_sec", "energy_kwh", "total_gflops",
                    "emissions_kg_co2", "cost_usd"}
        assert expected == set(DEFAULT_METRICS)


# ---------------------------------------------------------------------------
# sla_filter
# ---------------------------------------------------------------------------

class TestSlaFilter:
    def test_no_constraints_returns_all_successful(self, mock_results):
        result = sla_filter(mock_results, {})
        assert len(result) == len(mock_results)

    def test_filters_by_cost(self, mock_results):
        admitted = sla_filter(mock_results, {"cost_usd": 1.0})
        assert all(r["resources"]["cost_usd"] <= 1.0 for r in admitted)

    def test_filters_by_time(self, mock_results):
        admitted = sla_filter(mock_results, {"train_time_sec": 3600})
        assert all(r["resources"]["train_time_sec"] <= 3600 for r in admitted)

    def test_multiple_constraints_all_must_hold(self, mock_results):
        admitted = sla_filter(mock_results, {"cost_usd": 1.0, "train_time_sec": 4000})
        for r in admitted:
            assert r["resources"]["cost_usd"]       <= 1.0
            assert r["resources"]["train_time_sec"] <= 4000

    def test_impossible_constraint_returns_empty(self, mock_results):
        admitted = sla_filter(mock_results, {"cost_usd": 0.0001})
        assert admitted == []

    def test_failed_experiments_excluded(self, mock_with_failed):
        admitted = sla_filter(mock_with_failed, {})
        indices = [r["grid_experiment_idx"] for r in admitted]
        assert 99 not in indices

    def test_returns_list(self, mock_results):
        assert isinstance(sla_filter(mock_results, {}), list)


# ---------------------------------------------------------------------------
# dominates
# ---------------------------------------------------------------------------

class TestDominates:
    def test_strictly_better_in_all(self):
        assert dominates([1.0, 1.0], [2.0, 2.0]) is True

    def test_better_in_one_equal_in_other(self):
        assert dominates([1.0, 2.0], [2.0, 2.0]) is True

    def test_equal_in_all_does_not_dominate(self):
        assert dominates([1.0, 1.0], [1.0, 1.0]) is False

    def test_worse_in_one_does_not_dominate(self):
        assert dominates([1.0, 3.0], [2.0, 2.0]) is False

    def test_worse_in_all_does_not_dominate(self):
        assert dominates([3.0, 3.0], [1.0, 1.0]) is False

    def test_three_criteria(self):
        assert dominates([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) is True
        assert dominates([1.0, 2.0, 1.0], [1.0, 1.0, 2.0]) is False

    def test_asymmetry(self):
        vi = [1.0, 3.0]
        vj = [2.0, 2.0]
        assert dominates(vi, vj) is False
        assert dominates(vj, vi) is False


# ---------------------------------------------------------------------------
# pareto_front
# ---------------------------------------------------------------------------

class TestParetoFront:
    def test_returns_list(self, mock_results):
        assert isinstance(pareto_front(mock_results), list)

    def test_all_have_domination_count_zero(self, mock_results):
        front = pareto_front(mock_results)
        assert all(r["domination_count"] == 0 for r in front)

    def test_front_subset_of_results(self, mock_results):
        front = pareto_front(mock_results)
        all_ids = {r["grid_experiment_idx"] for r in mock_results}
        front_ids = {r["grid_experiment_idx"] for r in front}
        assert front_ids.issubset(all_ids)

    def test_no_duplicates(self, mock_results):
        front = pareto_front(mock_results)
        ids = [r["grid_experiment_idx"] for r in front]
        assert len(ids) == len(set(ids))

    def test_non_dominated_points_not_in_front_are_dominated(self, mock_results):
        """Exp 4 (3600s, $1.50) deve ser dominado por Exp 0 (3600s, $1.00)."""
        front = pareto_front(mock_results, metrics=["train_time_sec", "cost_usd"])
        front_ids = {r["grid_experiment_idx"] for r in front}
        # Exp 4 é dominado por Exp 0 em critérios time × cost → não deve estar na frente
        assert 4 not in front_ids

    def test_pareto_equals_skyband_k1(self, mock_results):
        front = pareto_front(mock_results)
        sb1   = skyband_query(mock_results, k=1)
        front_ids = {r["grid_experiment_idx"] for r in front}
        sb1_ids   = {r["grid_experiment_idx"] for r in sb1}
        assert front_ids == sb1_ids

    def test_sla_filter_applied_before_pareto(self, mock_results):
        front_no_sla = pareto_front(mock_results)
        front_sla    = pareto_front(mock_results, sla_constraints={"cost_usd": 1.0})
        # Com SLA mais restritivo, tamanho pode ser menor ou igual
        assert len(front_sla) <= len(front_no_sla)


# ---------------------------------------------------------------------------
# skyband_query
# ---------------------------------------------------------------------------

class TestSkybandQuery:
    def test_returns_list(self, mock_results):
        assert isinstance(skyband_query(mock_results, k=1), list)

    def test_k1_is_pareto_front(self, mock_results):
        sb = skyband_query(mock_results, k=1)
        assert all(r["domination_count"] == 0 for r in sb)

    def test_k_greater_includes_more_points(self, mock_results):
        sb1 = skyband_query(mock_results, k=1)
        sb3 = skyband_query(mock_results, k=3)
        assert len(sb3) >= len(sb1)

    def test_skyband_k_n_includes_all(self, mock_results):
        """Para k suficientemente grande, todos os pontos são incluídos."""
        sb = skyband_query(mock_results, k=len(mock_results) + 1)
        assert len(sb) == len(mock_results)

    def test_domination_count_less_than_k(self, mock_results):
        k = 3
        sb = skyband_query(mock_results, k=k)
        assert all(r["domination_count"] < k for r in sb)

    def test_skyband_rank_is_sequential(self, mock_results):
        sb = skyband_query(mock_results, k=3)
        ranks = [r["skyband_rank"] for r in sb]
        assert ranks == list(range(len(sb)))

    def test_sorted_by_domination_count(self, mock_results):
        sb = skyband_query(mock_results, k=5)
        counts = [r["domination_count"] for r in sb]
        assert counts == sorted(counts)

    def test_sla_filter_reduces_candidates(self, mock_results):
        sb_no_sla = skyband_query(mock_results, k=2)
        sb_sla    = skyband_query(mock_results, k=2, sla_constraints={"cost_usd": 1.0})
        assert len(sb_sla) <= len(sb_no_sla)

    def test_impossible_sla_returns_empty(self, mock_results):
        sb = skyband_query(mock_results, k=2, sla_constraints={"cost_usd": 0.0001})
        assert sb == []

    def test_invalid_k_raises_value_error(self, mock_results):
        with pytest.raises(ValueError, match="k deve ser >= 1"):
            skyband_query(mock_results, k=0)

    def test_metrics_minimize_mismatch_raises_value_error(self, mock_results):
        with pytest.raises(ValueError):
            skyband_query(
                mock_results,
                metrics=["train_time_sec", "cost_usd"],
                minimize=[True],        # tamanho incorreto
            )

    def test_custom_two_metrics(self, mock_results):
        sb = skyband_query(
            mock_results, k=1,
            metrics=["train_time_sec", "cost_usd"],
            minimize=[True, True],
        )
        assert isinstance(sb, list)
        assert all("domination_count" in r for r in sb)

    def test_maximization_metric(self, mock_results):
        """Testa minimize=False: inverte o sinal da métrica."""
        sb_min = skyband_query(mock_results, k=1,
                               metrics=["cost_usd"], minimize=[True])
        sb_max = skyband_query(mock_results, k=1,
                               metrics=["cost_usd"], minimize=[False])
        # O ponto com menor custo fica na frente de minimização;
        # o ponto com maior custo fica na frente de maximização.
        min_ids = {r["grid_experiment_idx"] for r in sb_min}
        max_ids = {r["grid_experiment_idx"] for r in sb_max}
        assert min_ids != max_ids

    def test_result_enriched_with_domination_count(self, mock_results):
        sb = skyband_query(mock_results, k=2)
        for r in sb:
            assert "domination_count" in r
            assert "skyband_rank" in r

    def test_original_result_not_mutated(self, mock_results):
        """skyband_query não deve modificar os dicts originais."""
        original_keys = set(mock_results[0].keys())
        _ = skyband_query(mock_results, k=2)
        assert set(mock_results[0].keys()) == original_keys


# ---------------------------------------------------------------------------
# compare_skyband_vs_ranking
# ---------------------------------------------------------------------------

class TestCompareSkybandVsRanking:
    def test_returns_dict(self, mock_results):
        r = compare_skyband_vs_ranking(mock_results, k=2)
        assert isinstance(r, dict)

    def test_required_keys_present(self, mock_results):
        r = compare_skyband_vs_ranking(mock_results, k=2)
        required = {
            "skyband", "scalar_top", "only_in_skyband", "only_in_scalar",
            "intersection", "jaccard_similarity", "skyband_size",
            "scalar_top_size", "k", "sla", "metrics",
        }
        assert required.issubset(r.keys())

    def test_jaccard_between_0_and_1(self, mock_results):
        r = compare_skyband_vs_ranking(mock_results, k=2)
        assert 0.0 <= r["jaccard_similarity"] <= 1.0

    def test_intersection_subset_of_both(self, mock_results):
        r = compare_skyband_vs_ranking(mock_results, k=2)
        sb_ids     = {x["grid_experiment_idx"] for x in r["skyband"]}
        scalar_ids = {x["grid_experiment_idx"] for x in r["scalar_top"]}
        inter_set  = set(r["intersection"])
        assert inter_set.issubset(sb_ids)
        assert inter_set.issubset(scalar_ids)

    def test_only_in_skyband_not_in_scalar(self, mock_results):
        r = compare_skyband_vs_ranking(mock_results, k=2)
        scalar_ids = {x["grid_experiment_idx"] for x in r["scalar_top"]}
        for idx in r["only_in_skyband"]:
            assert idx not in scalar_ids

    def test_only_in_scalar_not_in_skyband(self, mock_results):
        r = compare_skyband_vs_ranking(mock_results, k=2)
        sb_ids = {x["grid_experiment_idx"] for x in r["skyband"]}
        for idx in r["only_in_scalar"]:
            assert idx not in sb_ids

    def test_skyband_size_matches(self, mock_results):
        r = compare_skyband_vs_ranking(mock_results, k=2)
        assert r["skyband_size"] == len(r["skyband"])

    def test_k_preserved_in_report(self, mock_results):
        r = compare_skyband_vs_ranking(mock_results, k=3)
        assert r["k"] == 3

    def test_sla_preserved_in_report(self, mock_results):
        sla = {"cost_usd": 2.0}
        r = compare_skyband_vs_ranking(mock_results, sla=sla, k=2)
        assert r["sla"] == sla

    def test_identical_for_large_k(self, mock_results):
        """Para k suficientemente grande, ambas as abordagens retornam todos os pontos."""
        big_k = len(mock_results) + 5
        r = compare_skyband_vs_ranking(mock_results, k=big_k)
        assert r["jaccard_similarity"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# skyband_report
# ---------------------------------------------------------------------------

class TestSkybandReport:
    def test_returns_string(self, mock_results):
        assert isinstance(skyband_report(mock_results, k=2), str)

    def test_contains_header(self, mock_results):
        report = skyband_report(mock_results, k=2)
        assert "RELATÓRIO SKYBAND" in report

    def test_contains_k_value(self, mock_results):
        report = skyband_report(mock_results, k=3)
        assert "k=3" in report

    def test_contains_experiment_indices(self, mock_results):
        sb     = skyband_query(mock_results, k=2)
        report = skyband_report(mock_results, k=2)
        for r in sb:
            assert str(r["grid_experiment_idx"]) in report

    def test_contains_metric_names(self, mock_results):
        report = skyband_report(mock_results, k=1,
                                metrics=["train_time_sec", "cost_usd"])
        assert "train_time_sec" in report
        assert "cost_usd" in report

    def test_sla_none_shows_nenhuma(self, mock_results):
        report = skyband_report(mock_results, k=1)
        assert "nenhuma" in report

    def test_non_empty_report_lines(self, mock_results):
        report = skyband_report(mock_results, k=2)
        lines = [l for l in report.splitlines() if l.strip()]
        assert len(lines) > 5


# ---------------------------------------------------------------------------
# Integração com sla_profiles.json
# ---------------------------------------------------------------------------

class TestSlaProfilesIntegration:
    def test_file_exists(self):
        assert SLA_PROFILES_PATH.exists()

    def test_all_profiles_work_with_skyband_query(self, mock_results, sla_profiles):
        for name, profile in sla_profiles["profiles"].items():
            active = {m: v for m, v in profile["constraints"].items() if v is not None}
            sb = skyband_query(
                mock_results,
                k=profile["skyband_k"],
                sla_constraints=active or None,
                metrics=profile["metrics"],
                minimize=profile["minimize"],
            )
            assert isinstance(sb, list), f"Perfil '{name}' falhou"

    def test_profile_economico_rejects_high_cost(self, mock_results, sla_profiles):
        profile = sla_profiles["profiles"]["economico"]
        threshold = profile["constraints"]["cost_usd"]
        active = {m: v for m, v in profile["constraints"].items() if v is not None}
        admitted = sla_filter(mock_results, active)
        assert all(r["resources"]["cost_usd"] <= threshold for r in admitted)

    def test_profile_sustentavel_rejects_high_co2(self, mock_results, sla_profiles):
        profile = sla_profiles["profiles"]["sustentavel"]
        threshold = profile["constraints"]["emissions_kg_co2"]
        active = {m: v for m, v in profile["constraints"].items() if v is not None}
        admitted = sla_filter(mock_results, active)
        assert all(r["resources"]["emissions_kg_co2"] <= threshold for r in admitted)

    def test_all_recommended_envs_exist_in_sla_profiles(self, sla_profiles):
        env_keys = set(sla_profiles["environments"].keys())
        for name, profile in sla_profiles["profiles"].items():
            for env in profile["recommended_environments"]:
                assert env in env_keys, (
                    f"Perfil '{name}': ambiente recomendado '{env}' não existe em environments"
                )

    def test_environment_names_are_short_aliases(self, sla_profiles):
        expected = {"local", "colab", "gcp", "aws", "azure"}
        assert set(sla_profiles["environments"].keys()) == expected


# ---------------------------------------------------------------------------
# Integração com grid_search_multienv.json
# ---------------------------------------------------------------------------

class TestMultienvConfigIntegration:
    def test_file_exists(self):
        assert MULTIENV_PATH.exists()

    def test_active_envs_match_sla_profiles(self, multienv_config, sla_profiles):
        if "environments" not in multienv_config:
            pytest.skip("grid_search_multienv.json padronizado sem bloco environments")
        active = set(multienv_config["environments"]["active"])
        sla_envs = set(sla_profiles["environments"].keys())
        assert active == sla_envs

    def test_total_combinations_correct(self, multienv_config):
        import itertools
        hp   = multienv_config["hyperparameters"]
        keys = [k for k in hp if not k.startswith("_")]
        vals = [hp[k] for k in keys]
        n_hyper = len(list(itertools.product(*vals)))
        n_envs  = len(multienv_config.get("environments", {}).get("active", [])) or 1
        stated  = multienv_config["_meta"]["total_combinations"]["with_environments"]
        assert n_hyper * n_envs == stated

    def test_default_metrics_match_skyband_module(self, multienv_config):
        json_metrics = multienv_config["skyband"]["default_metrics"]
        assert set(json_metrics) == set(DEFAULT_METRICS)

    def test_analysis_k_values_all_positive(self, multienv_config):
        for k in multienv_config["skyband"]["analysis_k_values"]:
            assert k >= 1

    def test_cost_per_hour_consistent_between_files(self, multienv_config, sla_profiles):
        if "environments" not in multienv_config:
            pytest.skip("grid_search_multienv.json padronizado sem bloco environments")
        for env in multienv_config["environments"]["active"]:
            mg_cost  = multienv_config["environments"]["details"][env]["cost_per_hour_usd"]
            sla_cost = sla_profiles["environments"][env]["cost_per_hour_usd"]
            assert mg_cost == sla_cost, (
                f"cost_per_hour diverge para '{env}': multienv={mg_cost} sla={sla_cost}"
            )


# ---------------------------------------------------------------------------
# BL-07 — QUALITY_METRICS, _get_minimize_flag, _extract_metric_value e
#          skyband_query com include_quality_metrics
# ---------------------------------------------------------------------------

class TestQualityMetricsBL07:
    """Testes para métricas de qualidade preditiva (f1_score, accuracy)."""

    def _make_with_eval(self, idx, time, cost, f1, acc, status="success"):
        """Trace com métricas de avaliação em result['evaluation']."""
        return {
            "status": status,
            "grid_experiment_idx": idx,
            "grid_params": {"lr": 1e-5, "bs": 8},
            "resources": {"train_time_sec": time, "cost_usd": cost,
                          "energy_kwh": cost * 0.001, "emissions_kg_co2": cost * 0.0005,
                          "total_gflops": 100.0},
            "evaluation": {"f1_score": f1, "accuracy": acc, "source": "pool_out_training"},
        }

    @pytest.fixture
    def traces_with_quality(self):
        return [
            # fast/expensive/low-quality
            self._make_with_eval(0, 30.0,  1.20, 0.50, 0.55),
            # fast/expensive/high-quality
            self._make_with_eval(1, 35.0,  1.25, 0.85, 0.87),
            # slow/cheap/medium-quality
            self._make_with_eval(2, 6000.0, 0.60, 0.70, 0.72),
            # slow/cheap/low-quality
            self._make_with_eval(3, 6500.0, 0.65, 0.50, 0.52),
        ]

    # — QUALITY_METRICS e constantes —

    def test_quality_metrics_exported(self):
        from gridsearch.skyband import QUALITY_METRICS, QUALITY_MINIMIZE
        assert "f1_score" in QUALITY_METRICS
        assert "accuracy" in QUALITY_METRICS

    def test_quality_metrics_minimize_is_false(self):
        from gridsearch.skyband import QUALITY_METRICS, QUALITY_MINIMIZE
        for m, minimize in zip(QUALITY_METRICS, QUALITY_MINIMIZE):
            assert minimize is False, f"{m} deveria ter minimize=False"

    def test_default_metrics_minimize_is_true(self):
        from gridsearch.skyband import DEFAULT_METRICS, DEFAULT_MINIMIZE
        for m, minimize in zip(DEFAULT_METRICS, DEFAULT_MINIMIZE):
            assert minimize is True, f"{m} deveria ter minimize=True"

    # — _get_minimize_flag —

    def test_get_minimize_flag_resource_metric(self):
        from gridsearch.dominance import _get_minimize_flag
        assert _get_minimize_flag("train_time_sec") is True
        assert _get_minimize_flag("cost_usd") is True
        assert _get_minimize_flag("energy_kwh") is True

    def test_get_minimize_flag_quality_metric(self):
        from gridsearch.dominance import _get_minimize_flag
        assert _get_minimize_flag("f1_score") is False
        assert _get_minimize_flag("accuracy") is False

    def test_get_minimize_flag_unknown_metric_defaults_true(self):
        from gridsearch.dominance import _get_minimize_flag
        assert _get_minimize_flag("unknown_metric_xyz") is True

    # — _extract_metric_value com result["evaluation"] —

    def test_extract_f1_from_evaluation(self):
        from gridsearch.dominance import _extract_metric_value
        result = {"status": "success",
                  "resources": {},
                  "evaluation": {"f1_score": 0.85}}
        assert _extract_metric_value(result, "f1_score") == pytest.approx(0.85)

    def test_extract_accuracy_from_evaluation(self):
        from gridsearch.dominance import _extract_metric_value
        result = {"status": "success",
                  "resources": {},
                  "evaluation": {"accuracy": 0.90}}
        assert _extract_metric_value(result, "accuracy") == pytest.approx(0.90)

    def test_extract_resource_metric_still_works(self):
        from gridsearch.dominance import _extract_metric_value
        result = {"status": "success",
                  "resources": {"train_time_sec": 33.0},
                  "evaluation": {"f1_score": 0.85}}
        assert _extract_metric_value(result, "train_time_sec") == pytest.approx(33.0)

    def test_extract_resources_takes_priority_over_evaluation(self):
        """Se o mesmo campo existir em resources E evaluation, resources vence."""
        from gridsearch.dominance import _extract_metric_value
        result = {"resources": {"f1_score": 0.5}, "evaluation": {"f1_score": 0.9}}
        assert _extract_metric_value(result, "f1_score") == pytest.approx(0.5)

    def test_extract_missing_quality_metric_returns_inf(self):
        from gridsearch.dominance import _extract_metric_value
        result = {"status": "success", "resources": {}, "evaluation": {}}
        assert _extract_metric_value(result, "f1_score") == float("inf")

    # — skyband_query com include_quality_metrics —

    def test_include_quality_metrics_expands_criteria(self, traces_with_quality):
        from gridsearch.skyband import skyband_query, DEFAULT_METRICS, QUALITY_METRICS
        # Sem qualidade: 5 métricas de recurso
        sb_default = skyband_query(traces_with_quality, k=1)
        # Com qualidade: 5 recurso + 2 qualidade = 7
        sb_quality = skyband_query(traces_with_quality, k=1,
                                   include_quality_metrics=True)
        # A fronteira pode mudar ao incluir dimensões de qualidade
        assert isinstance(sb_quality, list)

    def test_high_quality_trace_in_skyband_with_quality_metrics(self, traces_with_quality):
        """O trace com melhor f1 (idx=1) deve estar no Skyband quando qualidade importa."""
        from gridsearch.skyband import skyband_query
        sb = skyband_query(traces_with_quality, k=2, include_quality_metrics=True)
        indices = {r["grid_experiment_idx"] for r in sb}
        assert 1 in indices, "Trace com melhor f1_score deveria estar no Skyband"

    def test_quality_minimize_direction_is_respected(self, traces_with_quality):
        """f1_score=0.85 (alto) deve ser preferível a f1_score=0.50 (baixo)."""
        from gridsearch.skyband import skyband_query
        # Trace 1 (f1=0.85) domina Trace 0 (f1=0.50) se ambos têm recursos similares
        metrics  = ["f1_score", "accuracy"]
        minimize = [False, False]  # maximizar ambos
        sb = skyband_query(traces_with_quality, k=1,
                           metrics=metrics, minimize=minimize)
        indices = {r["grid_experiment_idx"] for r in sb}
        # Trace com f1=0.50 e acc=0.55 é dominado por f1=0.85 e acc=0.87
        assert 1 in indices

    def test_include_quality_metrics_backward_compatible(self, mock_results):
        """include_quality_metrics=False mantém comportamento original."""
        sb_old = skyband_query(mock_results, k=1)
        sb_new = skyband_query(mock_results, k=1, include_quality_metrics=False)
        assert [r["grid_experiment_idx"] for r in sb_old] == \
               [r["grid_experiment_idx"] for r in sb_new]

    # — generate_psla4ml com include_quality_metrics —

    def test_generate_psla4ml_with_quality(self, traces_with_quality):
        from gridsearch.tiers import generate_psla4ml
        tiers = generate_psla4ml(
            traces_with_quality, k=2, include_quality_metrics=True,
        )
        assert isinstance(tiers, list)
        # Os tiers devem incluir f1_score e accuracy em raw_metrics
        for t in tiers:
            assert "f1_score" in t.raw_metrics or "accuracy" in t.raw_metrics

    def test_generate_psla4ml_minimize_flags_correct(self, traces_with_quality):
        """Verifica que generate_psla4ml não aplica mais minimize=True para quality."""
        from gridsearch.tiers import generate_psla4ml
        # Com quality: trace de alta qualidade (f1=0.85) não deve ser excluído
        # da fronteira em favor do de baixa qualidade (f1=0.50)
        tiers = generate_psla4ml(
            traces_with_quality, k=1,
            metrics=["f1_score", "accuracy"],
            include_quality_metrics=False,  # métricas explícitas
        )
        assert any(t.raw_metrics.get("f1_score", 0) >= 0.80 for t in tiers), \
            "Tier com f1_score alto deveria estar na fronteira"
