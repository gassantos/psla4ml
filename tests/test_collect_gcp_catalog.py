from pathlib import Path

import pytest

from scripts.collect_gcp_catalog import (
    NA_VALUE,
    _build_sku_index,
    _collect_adaptation_stats,
    _communication_rows,
    _extract_machine_type,
    _extract_price_from_sku,
    _is_gcp_environment,
    _resource_rows,
)


def test_is_gcp_environment_by_provider_or_name() -> None:
    assert _is_gcp_environment("A100", {"provider": "gcp"}) is True
    assert _is_gcp_environment("COLAB", {}) is True
    assert _is_gcp_environment("AWS", {"provider": "aws"}) is False


def test_collect_adaptation_stats_reads_final_rows(tmp_path: Path) -> None:
    file_path = tmp_path / "experiment_summary_GPU-20261121.csv"
    file_path.write_text(
        "activity,is_final_attempt,configured_environment,total_gflops,task_time_sec,cost_usd"
        + "\n"
        + "adaptation,True,COLAB,100.0,10.0,0.2\n"
        + "adaptation,True,COLAB,300.0,30.0,0.4\n"
        + "adaptation,False,COLAB,999.0,999.0,999.0\n"
        + "train,True,COLAB,10.0,10.0,0.1\n",
        encoding="utf-8",
    )

    stats = _collect_adaptation_stats([file_path])

    assert "COLAB" in stats
    assert stats["COLAB"]["samples"] == 2
    assert stats["COLAB"]["q_period_gflops_avg"] == 200.0
    assert stats["COLAB"]["period_seconds_avg"] == 20.0
    assert stats["COLAB"]["cost_usd_period_avg"] == pytest.approx(0.3)


def test_resource_rows_merge_catalog_and_summary() -> None:
    resources = {
        "COLAB": {
            "c_i": 0.5,
            "m_i": 40.0,
            "d_i": 100.0,
            "e_i": 0.1,
            "n_i": {"CPU": 8, "GPU": 1},
            "g_i": {"CPU": 12.0, "GPU": 400.0},
            "source": "catalog",
            "cpu_gflops_source": "benchmark",
        }
    }
    env_details = {"COLAB": {"provider": "gcp", "label": "Colab (a2-highgpu-1g)"}}
    adaptation_stats = {
        "COLAB": {
            "q_period_gflops_avg": 123.0,
            "period_seconds_avg": 45.0,
            "cost_usd_period_avg": 0.7,
            "samples": 3,
        }
    }

    rows = _resource_rows("2026-11-22T10:00:00+00:00", resources, env_details, adaptation_stats)

    assert len(rows) == 1
    row = rows[0]
    assert row["resource_id"] == "COLAB"
    assert row["provider"] == NA_VALUE
    assert row["n_gpu"] == NA_VALUE
    assert row["g_gpu_gflops_per_core"] == NA_VALUE
    assert row["q_period_gflops_avg"] == 123.0
    assert row["samples"] == 3


def test_communication_rows_include_pairs() -> None:
    env_details = {
        "COLAB": {"provider": "gcp"},
        "GCP": {"provider": "gcp"},
    }
    rows = _communication_rows("ts", ["COLAB", "GCP"], env_details, 0.5)

    assert rows
    target = next(r for r in rows if r["resource_i"] == "COLAB" and r["resource_j"] == "GCP")
    assert target["reference_transfer_gb"] == NA_VALUE
    assert target["comm_cost_usd_period"] == NA_VALUE


def test_extract_machine_type_from_label() -> None:
    assert _extract_machine_type("Google Colab (a2-highgpu-1g)") == "a2-highgpu-1g"
    assert _extract_machine_type("Google Colab") is None


def test_resource_rows_keep_machine_measured_fields_for_detected_local() -> None:
    resources = {
        "LOCAL": {
            "c_i": 0.1,
            "m_i": 52.0,
            "d_i": 400.0,
            "e_i": 0.2,
            "n_i": {"CPU": 12, "GPU": 1},
            "g_i": {"CPU": 55.0, "GPU": 300.0},
            "source": "detected_local",
            "cpu_gflops_source": "benchmarked_local",
        }
    }
    rows = _resource_rows("ts", resources, {"LOCAL": {}}, {})
    row = rows[0]

    assert row["m_i_gb"] == 52.0
    assert row["d_i_gb"] == 400.0
    assert row["n_cpu"] == 12
    assert row["g_cpu_gflops_per_core"] == 55.0
    assert row["c_i_usd_per_hour"] == NA_VALUE


def test_extract_price_from_sku_parses_units_and_nanos() -> None:
    sku = {
        "pricingInfo": [
            {
                "pricingExpression": {
                    "tieredRates": [
                        {"unitPrice": {"units": "1", "nanos": 250000000}},
                    ]
                }
            }
        ]
    }
    assert _extract_price_from_sku(sku) == pytest.approx(1.25)


def test_build_sku_index_groups_by_region_resource_and_usage() -> None:
    skus = [
        {
            "category": {"resourceGroup": "CPU", "usageType": "OnDemand"},
            "serviceRegions": ["us-central1"],
        },
        {
            "category": {"resourceGroup": "RAM", "usageType": "OnDemand"},
            "serviceRegions": ["us-central1", "us-east1"],
        },
    ]
    index = _build_sku_index(skus)
    assert ("us-central1", "CPU", "ONDEMAND") in index
    assert ("us-east1", "RAM", "ONDEMAND") in index


def test_resource_rows_uses_api_enriched_values_when_available() -> None:
    resources = {
        "COLAB": {
            "c_i": 99.0,
            "m_i": 1.0,
            "d_i": 2.0,
            "e_i": 0.2,
            "n_i": {"CPU": 2, "GPU": 0},
            "g_i": {"CPU": 10.0},
            "source": "catalog",
            "cpu_gflops_source": "benchmark",
        }
    }
    api_enriched = {
        "COLAB": {
            "source": "gcp_api_compute+billing",
            "c_i_usd_per_hour": 0.42,
            "m_i_gb": 30.0,
            "d_i_gb": 120.0,
            "n_cpu": 8,
            "n_gpu": 1,
            "gpu_name": "nvidia-l4",
        }
    }

    rows = _resource_rows("ts", resources, {"COLAB": {}}, {}, api_enriched=api_enriched)
    row = rows[0]

    assert row["source"] == "gcp_api_compute+billing"
    assert row["c_i_usd_per_hour"] == pytest.approx(0.42)
    assert row["m_i_gb"] == pytest.approx(30.0)
    assert row["n_cpu"] == 8
    assert row["n_gpu"] == 1
    assert row["gpu_name"] == "nvidia-l4"
