"""Testes para gridsearch.milp_instance — leitor da instância PSLA4MLData (Gurobi)."""

import json
import math

import pytest

from gridsearch.milp_instance import (
    DEFAULT_COST_USD,
    DEFAULT_DISK_GB,
    DEFAULT_ENERGY_KWH,
    DEFAULT_PEAK_RAM_GB,
    DEFAULT_TRAIN_TIME_HOURS,
    PSLA4MLData,
    _extract_alpha_weights,
    _find_result_by_idx,
    build_psla4ml_data,
    load_psla4ml_data,
    read_environments_details,
    read_grid_search_state,
    read_resource_catalog,
)

ENVIRONMENTS_DETAILS = {
    "local": {"gpu": "NVIDIA RTX 3090", "cost_per_hour_usd": 0.04, "vram_gb": 24},
    "colab": {"gpu": "NVIDIA T4", "cost_per_hour_usd": 0.0, "vram_gb": 16},
}

RESULTS = [
    {
        "status": "success", "grid_experiment_idx": 0, "selected_environment": "local",
        "resources": {"energy_kwh": 0.0018, "train_time_sec": 3600, "total_gflops": 521.93},
    },
    {
        "status": "success", "grid_experiment_idx": 1, "selected_environment": "colab",
        "resources": {"energy_kwh": 0.0005, "train_time_sec": 3600, "total_gflops": 300.0},
    },
]

SLA_PROFILE_BALANCEADO = {
    "constraints": {
        "cost_usd": 5.00,
        "train_time_sec": 7200,
        "energy_kwh": 0.100,
        "peak_ram_mb": None,
        "emissions_kg_co2": 0.020,
        "disk_gb": 30.0,
    },
    "metrics": ["train_time_sec", "cost_usd", "energy_kwh", "emissions_kg_co2"],
    "weights_scalar": [0.30, 0.30, 0.20, 0.20],
}

RESOURCE_CATALOG = {
    "resources": {
        "local": {"c_i": 0.04, "d_i": 119.4, "m_i": 3.99, "e_i": 0.09,
                  "n_i": {"CPU": 16, "GPU": 16}, "g_i": {"CPU": 5.0, "GPU": 1375.0}},
        "colab": {"c_i": 0.0, "d_i": 100.0, "m_i": 16.0, "e_i": None,
                  "n_i": {"CPU": 2, "GPU": 40}, "g_i": {"CPU": 5.0, "GPU": 1625.0}},
    },
    "communication_costs": [
        {"resource_i": "local", "resource_j": "colab", "cost_usd": 0.0528},
    ],
}


class TestFindResultByIdx:
    def test_finds_matching_experiment(self):
        assert _find_result_by_idx(RESULTS, 1)["selected_environment"] == "colab"

    def test_returns_none_when_absent(self):
        assert _find_result_by_idx(RESULTS, 999) is None


class TestExtractAlphaWeights:
    def test_no_profile_returns_equal_weights(self):
        alphas = _extract_alpha_weights(None)
        assert alphas == pytest.approx((1 / 3, 1 / 3, 1 / 3))
        assert sum(alphas) == pytest.approx(1.0)

    def test_profile_with_all_three_metrics_normalizes_to_one(self):
        alphas = _extract_alpha_weights(SLA_PROFILE_BALANCEADO)
        total = 0.30 + 0.30 + 0.20
        assert alphas == pytest.approx((0.30 / total, 0.30 / total, 0.20 / total))
        assert sum(alphas) == pytest.approx(1.0)

    def test_profile_missing_all_three_metrics_falls_back_to_equal(self):
        profile = {"metrics": ["emissions_kg_co2"], "weights_scalar": [1.0]}
        assert _extract_alpha_weights(profile) == pytest.approx((1 / 3, 1 / 3, 1 / 3))


class TestBuildPsla4mlData:
    def test_returns_dataclass_with_expected_shape(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert isinstance(data, PSLA4MLData)
        assert data.R == ["local", "colab"]
        assert data.P == ["CPU", "GPU", "TPU"]
        assert data.Gf == pytest.approx(521.93)

    def test_only_persisted_fields_are_populated_no_invented_data(self):
        """Sem cores_by_processor/gflops_per_core em environments_details, n/g ficam vazios."""
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert data.n == {}
        assert data.g == {}
        assert data.d == {}  # nenhum dos dois ambientes declara disk_gb

    def test_n_and_g_populated_when_present_in_config(self):
        environments_with_cores = {
            "local": {
                "cost_per_hour_usd": 0.04, "vram_gb": 24, "disk_gb": 200.0,
                "cores_by_processor": {"CPU": 8, "GPU": 82},
                "gflops_per_core": {"CPU": 5.0, "GPU": 865.0},
            },
        }
        data = build_psla4ml_data(
            environments_with_cores, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert data.n[("local", "GPU")] == 82
        assert data.g[("local", "GPU")] == pytest.approx(865.0)
        assert data.d["local"] == pytest.approx(200.0)
        assert ("local", "TPU") not in data.n  # nao declarado -> ausente, nao inventado

    def test_cost_per_resource_reflects_environment_tariff(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert data.c["local"] == pytest.approx(0.04)
        assert data.c["colab"] == pytest.approx(0.0)

    def test_memory_mapped_from_environments_details(self):
        """m_i: nunca coberto antes -- vram_gb declarado em environments.details."""
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert data.m["local"] == pytest.approx(24)
        assert data.m["colab"] == pytest.approx(16)

    def test_memory_falls_back_to_resource_catalog_when_absent_in_config(self):
        environments_sem_vram = {"local": {"cost_per_hour_usd": 0.04}}
        data = build_psla4ml_data(
            environments_sem_vram, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
            resource_catalog=RESOURCE_CATALOG,
        )
        assert data.m["local"] == pytest.approx(3.99)  # m_i do catalogo, nao do config

    def test_memory_absent_when_not_in_config_nor_catalog(self):
        environments_sem_vram = {"local": {"cost_per_hour_usd": 0.04}}
        data = build_psla4ml_data(
            environments_sem_vram, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert "local" not in data.m  # sem fonte alguma -> ausente, nao inventado

    def test_energy_rate_derived_from_persisted_results(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert data.e["local"] == pytest.approx(0.0018)
        assert data.e["colab"] == pytest.approx(0.0005)

    def test_environment_without_history_has_no_energy_entry(self):
        environments = {**ENVIRONMENTS_DETAILS, "gcp": {"cost_per_hour_usd": 0.35}}
        data = build_psla4ml_data(
            environments, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert "gcp" not in data.e

    def test_alpha_weights_sum_to_one(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert data.alpha1 + data.alpha2 + data.alpha3 == pytest.approx(1.0)

    def test_sla_constraints_mapped_correctly(self):
        profile = {
            **SLA_PROFILE_BALANCEADO,
            "constraints": {
                **SLA_PROFILE_BALANCEADO["constraints"],
                "peak_ram_mb": 8192,
            },
        }
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=profile, target_grid_experiment_idx=0,
        )
        assert data.CM == pytest.approx(5.00)
        assert data.Eref == pytest.approx(0.100)
        assert data.TM == 2
        assert data.DS == pytest.approx(30.0)
        assert data.MC == pytest.approx(8.0)

    def test_missing_sla_profile_uses_documented_defaults(self):
        data = build_psla4ml_data(ENVIRONMENTS_DETAILS, RESULTS, target_grid_experiment_idx=0)
        assert data.CM == DEFAULT_COST_USD
        assert data.Eref == DEFAULT_ENERGY_KWH
        assert data.MC == DEFAULT_PEAK_RAM_GB
        assert data.DS == DEFAULT_DISK_GB
        assert data.TM == math.ceil(DEFAULT_TRAIN_TIME_HOURS)  # TM tambem cai no fallback

    def test_time_period_hours_changes_tm(self):
        """TM = ceil(train_time_sec/3600 / time_period_hours) — periodo de 2h reduz TM pela metade."""
        data_1h = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        data_2h = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
            time_period_hours=2.0,
        )
        assert data_1h.TM == 2  # 7200s / 3600s por periodo de 1h
        assert data_2h.TM == 1  # 7200s / 3600s = 2h -> 1 periodo de 2h

    def test_communication_cost_matrix_present_for_all_pairs(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert data.c_comm == {("local", "colab"): 0.0}

    def test_nm_defaults_to_number_of_resources(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert data.NM == len(ENVIRONMENTS_DETAILS)

    def test_explicit_nm_overrides_default(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0, NM=1,
        )
        assert data.NM == 1

    def test_explicit_ds_overrides_profile_and_default(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0, DS=500.0,
        )
        assert data.DS == pytest.approx(500.0)

    def test_custom_processor_types_restricts_p(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
            resource_catalog=RESOURCE_CATALOG, processor_types=("GPU",),
        )
        assert data.P == ["GPU"]
        assert data.n[("local", "GPU")] == 16
        assert all(p == "GPU" for (_, p) in data.n)
        assert all(p == "GPU" for (_, p) in data.g)

    def test_gf_from_target_grid_experiment_idx(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=1,
        )
        assert data.Gf == pytest.approx(300.0)

    def test_gf_from_target_result_overrides_idx_lookup(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO,
            target_result={"resources": {"total_gflops": 999.0}},
        )
        assert data.Gf == pytest.approx(999.0)

    def test_explicit_gf_overrides_everything(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0, Gf=42.0,
        )
        assert data.Gf == pytest.approx(42.0)

    def test_raises_without_gf_source(self):
        with pytest.raises(ValueError, match="Gf"):
            build_psla4ml_data(ENVIRONMENTS_DETAILS, RESULTS)

    def test_raises_on_empty_environments(self):
        with pytest.raises(ValueError, match="environments_details"):
            build_psla4ml_data({}, RESULTS, target_grid_experiment_idx=0)


class TestResourceCatalogFallback:
    def test_resource_catalog_fills_d_n_g_when_config_lacks_them(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
            resource_catalog=RESOURCE_CATALOG,
        )
        assert data.d["local"] == pytest.approx(119.4)
        assert data.n[("local", "GPU")] == 16
        assert data.g[("local", "GPU")] == pytest.approx(1375.0)
        assert data.n[("local", "CPU")] == 16
        assert ("local", "TPU") not in data.n  # catalogo nao declara TPU -> ausente, nao inventado

    def test_environments_details_takes_priority_over_resource_catalog(self):
        environments = {"local": {**ENVIRONMENTS_DETAILS["local"], "disk_gb": 999.0}}
        data = build_psla4ml_data(
            environments, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
            resource_catalog=RESOURCE_CATALOG,
        )
        assert data.d["local"] == pytest.approx(999.0)  # declarado explicitamente vence o catalogo

    def test_communication_costs_come_from_persisted_catalog(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
            resource_catalog=RESOURCE_CATALOG,
        )
        assert data.c_comm[("local", "colab")] == pytest.approx(0.0528)

    def test_communication_costs_default_to_zero_without_catalog(self):
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert data.c_comm == {("local", "colab"): 0.0}

    def test_resource_catalog_energy_used_only_without_history(self):
        # colab tem e_i=None no catalogo, mas ha historico real em RESULTS -> usa o historico
        data = build_psla4ml_data(
            ENVIRONMENTS_DETAILS, RESULTS,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
            resource_catalog=RESOURCE_CATALOG,
        )
        assert data.e["colab"] == pytest.approx(0.0005)  # do historico, nao do catalogo

    def test_load_psla4ml_data_with_resource_catalog_path(self, tmp_path):
        grid_config_path = tmp_path / "grid.json"
        grid_config_path.write_text(json.dumps({
            "environments": {"details": ENVIRONMENTS_DETAILS},
        }), encoding="utf-8")
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({"results": RESULTS}), encoding="utf-8")
        catalog_path = tmp_path / "resource_catalog.json"
        catalog_path.write_text(json.dumps(RESOURCE_CATALOG), encoding="utf-8")

        data = load_psla4ml_data(
            grid_config_path, state_path,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
            resource_catalog_path=catalog_path,
        )
        assert data.n[("local", "GPU")] == 16
        assert data.d["colab"] == pytest.approx(100.0)


class TestReadResourceCatalog:
    def test_reads_persisted_catalog(self, tmp_path):
        path = tmp_path / "resource_catalog.json"
        path.write_text(json.dumps(RESOURCE_CATALOG), encoding="utf-8")
        assert read_resource_catalog(path) == RESOURCE_CATALOG


class TestReadPersistedFiles:
    def test_read_environments_details_from_real_config(self):
        with pytest.raises(ValueError, match="environments.details"):
            read_environments_details("gridsearch/config/grid_search_multienv.json")

    def test_read_environments_details_raises_when_absent(self, tmp_path):
        path = tmp_path / "no_env.json"
        path.write_text(json.dumps({"hyperparameters": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="environments.details"):
            read_environments_details(path)

    def test_read_grid_search_state(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text(json.dumps({"results": RESULTS, "completed_experiments": [0, 1]}), encoding="utf-8")
        state = read_grid_search_state(path)
        assert state["results"] == RESULTS

    def test_load_psla4ml_data_end_to_end(self, tmp_path):
        grid_config_path = tmp_path / "grid.json"
        grid_config_path.write_text(json.dumps({
            "environments": {"details": ENVIRONMENTS_DETAILS},
        }), encoding="utf-8")
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({"results": RESULTS}), encoding="utf-8")

        data = load_psla4ml_data(
            grid_config_path, state_path,
            sla_profile=SLA_PROFILE_BALANCEADO, target_grid_experiment_idx=0,
        )
        assert isinstance(data, PSLA4MLData)
        assert data.R == ["local", "colab"]
        assert data.Gf == pytest.approx(521.93)

