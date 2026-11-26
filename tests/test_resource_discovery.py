"""Testes para gridsearch.resource_discovery — parâmetros de recurso do MILP PSLA4ML."""

import json

import pytest

from gridsearch.resource_discovery import (
    DEFAULT_CLOUD_CPU_CORES,
    DEFAULT_CLOUD_DISK_GB,
    DEFAULT_EGRESS_COST_USD_PER_GB,
    EGRESS_COST_USD_PER_GB,
    GPU_CATALOG,
    REFERENCE_TRANSFER_GB,
    TPU_CATALOG,
    LocalHardwareSpec,
    benchmark_cpu_gflops_per_core,
    build_resource_catalog,
    build_resource_spec,
    collect_and_persist_machine_baseline,
    collect_and_persist_resource_catalog,
    communication_cost_matrix,
    detect_local_hardware,
    detect_local_tpu,
    detect_local_tpu_version,
    estimate_energy_rate_kwh_per_hour,
    extract_machine_baseline,
    persist_resource_catalog,
)


class TestDetectLocalHardware:
    def test_returns_positive_cpu_ram_disk(self):
        spec = detect_local_hardware()
        assert isinstance(spec, LocalHardwareSpec)
        assert spec.cpu_logical_cores >= 1
        assert spec.ram_total_gb > 0
        assert spec.disk_total_gb > 0
        assert spec.disk_free_gb >= 0

    def test_gpu_fields_are_none_or_consistent(self):
        spec = detect_local_hardware()
        # Sem CUDA: todos None. Com CUDA: todos preenchidos.
        gpu_fields = (spec.gpu_name, spec.gpu_vram_gb, spec.gpu_multiprocessor_count)
        assert all(f is None for f in gpu_fields) or all(f is not None for f in gpu_fields)

    def test_cpu_gflops_per_core_is_measured(self):
        spec = detect_local_hardware()
        assert spec.cpu_gflops_per_core is None or spec.cpu_gflops_per_core > 0

    def test_tpu_core_count_is_non_negative(self):
        spec = detect_local_hardware()
        assert spec.tpu_core_count >= 0


class TestBenchmarkCpuGflops:
    def test_returns_positive_measurement(self):
        gflops = benchmark_cpu_gflops_per_core(matrix_size=128, repeats=1)
        assert gflops > 0


class TestDetectLocalTpu:
    def test_returns_non_negative_int_without_torch_xla(self):
        # Nesta maquina de desenvolvimento nao ha torch_xla instalado/TPU real.
        assert detect_local_tpu() >= 0

    def test_detect_version_from_env(self, monkeypatch):
        monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v6e-1")
        assert detect_local_tpu_version() == "v6e"

    def test_detect_version_from_shell_probe_output(self, monkeypatch):
        monkeypatch.delenv("TPU_ACCELERATOR_TYPE", raising=False)
        monkeypatch.delenv("TPU_NAME", raising=False)
        monkeypatch.setattr(
            "gridsearch.resource_discovery._probe_tpu_environment_from_shell",
            lambda: ["TPU_ACCELERATOR_TYPE=v5e-1"],
        )
        assert detect_local_tpu_version() == "v5e"


class TestGpuCatalog:
    @pytest.mark.parametrize("name", ["NVIDIA RTX 3090", "NVIDIA T4", "NVIDIA V100", "NVIDIA A100"])
    def test_known_gpus_present(self, name):
        assert name in GPU_CATALOG
        assert GPU_CATALOG[name]["fp16_tflops"] > 0
        assert GPU_CATALOG[name]["sm_count"] > 0


class TestTpuCatalog:
    @pytest.mark.parametrize("version", ["v2", "v3", "v4", "v5e", "v6e"])
    def test_known_tpu_versions_present(self, version):
        assert version in TPU_CATALOG
        assert TPU_CATALOG[version]["bf16_tflops_per_chip"] > 0
        assert TPU_CATALOG[version]["cores_per_chip"] > 0


class TestBuildResourceSpec:
    def test_local_gpu_name_with_geforce_prefix_still_matches_catalog(self):
        """Regressao: torch reporta 'NVIDIA GeForce RTX 2050', catalogo usa 'NVIDIA RTX 2050'."""
        hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=16.0, disk_total_gb=200.0, disk_free_gb=100.0,
            gpu_name="NVIDIA GeForce RTX 2050", gpu_vram_gb=4.0, gpu_multiprocessor_count=16,
            cpu_gflops_per_core=50.0,
        )
        spec = build_resource_spec("local", {"gpu": "NVIDIA RTX 3090", "cost_per_hour_usd": 0.04}, local_hardware=hw)
        assert spec["n_i"]["GPU"] == 16
        assert spec["g_i"]["GPU"] > 0  # nao deve cair para 0.0 por falha de matching

    def test_cloud_resource_uses_static_catalog(self):
        spec = build_resource_spec("colab", {"gpu": "NVIDIA T4", "cost_per_hour_usd": 0.0, "vram_gb": 16})
        assert spec["resource_id"] == "colab"
        assert spec["c_i"] == 0.0
        assert spec["m_i"] == 16
        assert spec["n_i"]["GPU"] == GPU_CATALOG["NVIDIA T4"]["sm_count"]
        assert spec["g_i"]["GPU"] > 0
        assert spec["source"] == "static_catalog"

    def test_cloud_resource_without_local_hardware_has_no_cpu_gflops(self):
        """Sem proxy de benchmark local disponivel, g_i['CPU'] fica ausente (nao inventado)."""
        spec = build_resource_spec("colab", {"gpu": "NVIDIA T4", "cost_per_hour_usd": 0.0})
        assert spec["n_i"]["CPU"] > 0  # contagem de nucleos ainda vem do catalogo por label
        assert "CPU" not in spec["g_i"]
        assert spec["cpu_gflops_source"] is None

    def test_cloud_resource_uses_local_benchmark_as_cpu_proxy(self):
        hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=16.0, disk_total_gb=200.0, disk_free_gb=100.0,
            cpu_gflops_per_core=42.0,
        )
        spec = build_resource_spec("colab", {"gpu": "NVIDIA T4"}, local_hardware=hw)
        assert spec["g_i"]["CPU"] == pytest.approx(42.0)
        assert spec["cpu_gflops_source"] == "local_benchmark_proxy"

    def test_cloud_resource_unknown_gpu_falls_back_to_default(self):
        spec = build_resource_spec("mystery_cloud", {"gpu": "Some Unknown GPU", "cost_per_hour_usd": 1.0})
        assert spec["source"] == "default"
        assert spec["d_i"] == DEFAULT_CLOUD_DISK_GB
        assert "GPU" not in spec["n_i"]  # sem GPU reconhecida, nao inventa
        assert "GPU" not in spec["g_i"]
        # CPU: contagem por catalogo de label continua presente:
        assert spec["n_i"]["CPU"] == DEFAULT_CLOUD_CPU_CORES
        # TPU: nao detectado nem declarado -> ausente (nao fabricado como 0):
        assert "TPU" not in spec["n_i"]
        assert "TPU" not in spec["g_i"]

    def test_processor_types_param_restricts_output(self):
        spec = build_resource_spec(
            "colab", {"gpu": "NVIDIA T4", "cost_per_hour_usd": 0.0},
            processor_types=("GPU",),
        )
        assert set(spec["n_i"].keys()) == {"GPU"}
        assert set(spec["g_i"].keys()) == {"GPU"}

    def test_local_resource_uses_detected_hardware(self):
        hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=32.0, disk_total_gb=500.0, disk_free_gb=200.0,
            gpu_name="NVIDIA RTX 3090", gpu_vram_gb=24.0, gpu_multiprocessor_count=82,
            cpu_gflops_per_core=30.0,
        )
        spec = build_resource_spec("local", {"gpu": "NVIDIA RTX 3090", "cost_per_hour_usd": 0.04}, local_hardware=hw)
        assert spec["source"] == "detected_local"
        assert spec["m_i"] == 24.0
        assert spec["d_i"] == 200.0
        assert spec["n_i"]["GPU"] == 82
        assert spec["g_i"]["CPU"] == pytest.approx(30.0)
        assert spec["cpu_gflops_source"] == "benchmarked_local"
        assert "TPU" not in spec["n_i"]  # tpu_core_count=0 (padrao), sem deteccao

    def test_local_resource_with_detected_tpu_cores(self):
        hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=32.0, disk_total_gb=500.0, disk_free_gb=200.0,
            tpu_core_count=8,
        )
        spec = build_resource_spec("local", {}, local_hardware=hw)
        assert spec["n_i"]["TPU"] == 8
        assert "TPU" not in spec["g_i"]  # nucleos reais, mas sem versao conhecida -> GFLOPS nao inventado

    def test_declared_tpu_version_populates_catalog_specs(self):
        spec = build_resource_spec("gcp_tpu", {"tpu_version": "v3", "cost_per_hour_usd": 4.0})
        assert spec["n_i"]["TPU"] == TPU_CATALOG["v3"]["cores_per_chip"]
        assert spec["g_i"]["TPU"] > 0

    def test_declared_tpu_v6e_uses_918_tflops(self):
        spec = build_resource_spec("gcp_tpu", {"tpu_version": "v6e-1", "cost_per_hour_usd": 4.0})
        assert spec["n_i"]["TPU"] == 1
        assert spec["g_i"]["TPU"] == pytest.approx(918000.0)

    def test_declared_tpu_v5e_uses_197_tflops(self):
        spec = build_resource_spec("gcp_tpu", {"tpu_version": "v5e-1", "cost_per_hour_usd": 4.0})
        assert spec["n_i"]["TPU"] == 1
        assert spec["g_i"]["TPU"] == pytest.approx(197000.0)


class TestExtractMachineBaseline:
    def test_reports_non_extractable_fields_when_inputs_missing(self, monkeypatch):
        fake_hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=32.0, disk_total_gb=500.0, disk_free_gb=200.0,
            gpu_name="NVIDIA RTX 3090", gpu_vram_gb=24.0, gpu_multiprocessor_count=82,
            cpu_gflops_per_core=30.0,
        )
        monkeypatch.setattr(
            "gridsearch.resource_discovery.detect_local_hardware", lambda disk_path=".": fake_hw,
        )
        baseline = extract_machine_baseline("host-a")
        assert baseline["source"] == "detected_host"
        assert "c_i" in baseline["unavailable_fields"]
        assert "e_i" in baseline["unavailable_fields"]
        assert baseline["n_i"]["CPU"] == 8
        assert baseline["n_i"]["GPU"] == 82

    def test_uses_historical_results_for_energy_rate(self, monkeypatch):
        fake_hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=32.0, disk_total_gb=500.0, disk_free_gb=200.0,
            cpu_gflops_per_core=30.0,
        )
        monkeypatch.setattr(
            "gridsearch.resource_discovery.detect_local_hardware", lambda disk_path=".": fake_hw,
        )
        baseline = extract_machine_baseline(
            "host-b",
            historical_results=[
                {
                    "status": "success",
                    "selected_environment": "host-b",
                    "resources": {"energy_kwh": 0.2, "train_time_sec": 3600},
                },
            ],
        )
        assert baseline["e_i"] == pytest.approx(0.2)
        assert "e_i" not in baseline["unavailable_fields"]

    def test_collect_and_persist_machine_baseline(self, monkeypatch, tmp_path):
        fake_hw = LocalHardwareSpec(
            cpu_logical_cores=4, ram_total_gb=8.0, disk_total_gb=100.0, disk_free_gb=40.0,
            cpu_gflops_per_core=12.0,
        )
        monkeypatch.setattr(
            "gridsearch.resource_discovery.detect_local_hardware", lambda disk_path=".": fake_hw,
        )
        output_path = tmp_path / "machine_catalog.json"
        path = collect_and_persist_machine_baseline("edge-node", output_path)
        assert path.exists()
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert "edge-node" in loaded["resources"]
        assert loaded["resources"]["edge-node"]["source"] == "detected_host"


class TestEstimateEnergyRate:
    def test_computes_average_rate_for_environment(self):
        results = [
            {"status": "success", "selected_environment": "local",
             "resources": {"energy_kwh": 0.001, "train_time_sec": 3600}},
            {"status": "success", "selected_environment": "local",
             "resources": {"energy_kwh": 0.002, "train_time_sec": 3600}},
            {"status": "success", "selected_environment": "colab",
             "resources": {"energy_kwh": 0.0005, "train_time_sec": 3600}},
            {"status": "failed", "selected_environment": "local",
             "resources": {}},
        ]
        assert estimate_energy_rate_kwh_per_hour(results, "local") == pytest.approx(0.0015)
        assert estimate_energy_rate_kwh_per_hour(results, "colab") == pytest.approx(0.0005)

    def test_returns_none_without_samples(self):
        assert estimate_energy_rate_kwh_per_hour([], "local") is None
        assert estimate_energy_rate_kwh_per_hour(
            [{"status": "success", "selected_environment": "gcp", "resources": {"energy_kwh": 0.1, "train_time_sec": 3600}}],
            "local",
        ) is None


class TestBuildResourceCatalog:
    def test_builds_entry_per_environment(self):
        environments_details = {
            "local": {"gpu": "NVIDIA RTX 3090", "cost_per_hour_usd": 0.04},
            "colab": {"gpu": "NVIDIA T4", "cost_per_hour_usd": 0.0, "vram_gb": 16},
        }
        fake_hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=32.0, disk_total_gb=500.0, disk_free_gb=200.0,
            gpu_name="NVIDIA RTX 3090", gpu_vram_gb=24.0, gpu_multiprocessor_count=82,
        )
        catalog = build_resource_catalog(environments_details, local_hardware=fake_hw)

        assert set(catalog.keys()) == {"local", "colab"}
        assert catalog["local"]["source"] == "detected_local"
        assert catalog["colab"]["source"] == "static_catalog"
        assert catalog["local"]["e_i"] is None  # sem histórico fornecido

    def test_populates_e_i_from_results_history(self):
        environments_details = {"local": {"gpu": "NVIDIA RTX 3090", "cost_per_hour_usd": 0.04}}
        results = [
            {"status": "success", "selected_environment": "local",
             "resources": {"energy_kwh": 0.0018, "train_time_sec": 3600}},
        ]
        fake_hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=32.0, disk_total_gb=500.0, disk_free_gb=200.0,
            gpu_name="NVIDIA RTX 3090", gpu_vram_gb=24.0, gpu_multiprocessor_count=82,
        )
        catalog = build_resource_catalog(environments_details, results=results, local_hardware=fake_hw)
        assert catalog["local"]["e_i"] == pytest.approx(0.0018)


class TestCommunicationCostMatrix:
    def test_same_provider_pairs_are_free(self):
        """colab e gcp mapeiam para o mesmo provedor (GCP) -> custo zero."""
        matrix = communication_cost_matrix(["colab", "gcp"])
        assert matrix[("colab", "gcp")] == 0.0

    def test_different_providers_use_max_egress_rate(self):
        matrix = communication_cost_matrix(["gcp", "aws"])
        expected = max(EGRESS_COST_USD_PER_GB["gcp"], EGRESS_COST_USD_PER_GB["aws"]) * REFERENCE_TRANSFER_GB
        assert matrix[("gcp", "aws")] == pytest.approx(expected)

    def test_local_to_cloud_uses_cloud_egress_rate(self):
        matrix = communication_cost_matrix(["local", "azure"])
        expected = EGRESS_COST_USD_PER_GB["azure"] * REFERENCE_TRANSFER_GB
        assert matrix[("local", "azure")] == pytest.approx(expected)

    def test_unknown_resource_name_uses_default_egress_rate(self):
        matrix = communication_cost_matrix(["mystery1", "mystery2"])
        expected = DEFAULT_EGRESS_COST_USD_PER_GB * REFERENCE_TRANSFER_GB
        assert matrix[("mystery1", "mystery2")] == pytest.approx(expected)

    def test_explicit_provider_field_overrides_name_hint(self):
        environments_details = {
            "custom_a": {"provider": "aws"},
            "custom_b": {"provider": "aws"},
        }
        matrix = communication_cost_matrix(
            ["custom_a", "custom_b"], environments_details=environments_details,
        )
        assert matrix[("custom_a", "custom_b")] == 0.0  # mesmo provedor declarado -> gratis

    def test_custom_reference_transfer_scales_cost(self):
        matrix = communication_cost_matrix(["gcp", "aws"], reference_transfer_gb=2.0)
        expected = max(EGRESS_COST_USD_PER_GB["gcp"], EGRESS_COST_USD_PER_GB["aws"]) * 2.0
        assert matrix[("gcp", "aws")] == pytest.approx(expected)

    def test_single_resource_has_no_pairs(self):
        assert communication_cost_matrix(["only_one"]) == {}


class TestPersistResourceCatalog:
    def test_persist_resource_catalog_writes_json(self, tmp_path):
        catalog = {"local": {"c_i": 0.04, "n_i": {"GPU": 82}, "g_i": {"GPU": 865.0}}}
        output_path = tmp_path / "sub" / "resource_catalog.json"
        result_path = persist_resource_catalog(catalog, output_path)
        assert result_path == output_path
        assert output_path.exists()
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        assert loaded == catalog

    def test_collect_and_persist_resource_catalog_end_to_end(self, tmp_path):
        environments_details = {
            "local": {"gpu": "NVIDIA RTX 3090", "cost_per_hour_usd": 0.04},
            "colab": {"gpu": "NVIDIA T4", "cost_per_hour_usd": 0.0, "vram_gb": 16},
        }
        fake_hw = LocalHardwareSpec(
            cpu_logical_cores=8, ram_total_gb=32.0, disk_total_gb=500.0, disk_free_gb=200.0,
            gpu_name="NVIDIA RTX 3090", gpu_vram_gb=24.0, gpu_multiprocessor_count=82,
        )
        output_path = tmp_path / "resource_catalog.json"
        result_path = collect_and_persist_resource_catalog(
            environments_details, output_path, local_hardware=fake_hw,
        )
        assert result_path.exists()
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        assert set(loaded.keys()) == {"resources", "communication_costs"}
        assert set(loaded["resources"].keys()) == {"local", "colab"}
        assert loaded["resources"]["local"]["n_i"]["GPU"] == 82
        assert loaded["resources"]["colab"]["source"] == "static_catalog"
        # local (provider "local") x colab (provider "gcp") -> custo de egress > 0
        comm = {(r["resource_i"], r["resource_j"]): r["cost_usd"] for r in loaded["communication_costs"]}
        assert comm[("local", "colab")] > 0.0
