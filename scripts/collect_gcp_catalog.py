"""
scripts/collect_gcp_catalog.py — coleta catálogo de recursos GCP/Colab em CSV
==============================================================================

Gera catálogos em CSV no diretório ``output/experiments/metrics`` para apoiar
o pré-gate de viabilidade do MILP (custos/capacidade/comunicação), com foco em
ambientes GCP/Colab declarados no grid config.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment.helpers import METRICS_DIR, normalize_environment_label
from gridsearch.milp_instance import read_environments_details
from gridsearch.resource_discovery import (
    REFERENCE_TRANSFER_GB,
    build_resource_catalog,
)

logger = logging.getLogger(__name__)

API_TIMEOUT_SEC = 10
NA_VALUE = "N/A"
COMPUTE_SERVICE_DISPLAY_NAME = "Compute Engine"
METADATA_BASE_URL = "http://metadata.google.internal/computeMetadata/v1"


def _is_gcp_environment(env_name: str, env_details: dict[str, Any]) -> bool:
    provider = str(env_details.get("provider", "")).lower()
    if provider in {"gcp", "google", "google-cloud"}:
        return True
    normalized_name = normalize_environment_label(env_name)
    return normalized_name in {"GCP", "COLAB"}


def _summary_files(summary_path: str | None) -> list[Path]:
    if summary_path:
        candidate = Path(summary_path)
        if candidate.is_file():
            return [candidate]
        if candidate.is_dir():
            return sorted(candidate.glob("experiment_summary_*.csv"))
        return []
    return sorted(METRICS_DIR.glob("experiment_summary_*.csv"))


def _collect_adaptation_stats(
    files: Iterable[Path],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for file in files:
        with open(file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("activity") != "adaptation":
                    continue
                if row.get("is_final_attempt") != "True":
                    continue
                env = (
                    row.get("configured_environment")
                    or row.get("selected_environment")
                    or row.get("grid_environment")
                    or ""
                )
                normalized = normalize_environment_label(env)
                if not normalized:
                    continue
                slot = grouped.setdefault(
                    normalized,
                    {"q_period_gflops": [], "task_time_sec": [], "cost_usd": []},
                )
                for key, raw in (
                    ("q_period_gflops", row.get("total_gflops")),
                    ("task_time_sec", row.get("task_time_sec")),
                    ("cost_usd", row.get("cost_usd")),
                ):
                    try:
                        if raw not in (None, ""):
                            slot[key].append(float(raw))
                    except (TypeError, ValueError):
                        continue

    stats: dict[str, dict[str, float | int]] = {}
    for env, values in grouped.items():
        stats[env] = {
            "samples": len(values["task_time_sec"]),
            "q_period_gflops_avg": mean(values["q_period_gflops"]) if values["q_period_gflops"] else 0.0,
            "period_seconds_avg": mean(values["task_time_sec"]) if values["task_time_sec"] else 0.0,
            "cost_usd_period_avg": mean(values["cost_usd"]) if values["cost_usd"] else 0.0,
        }
    return stats


def _resource_rows(
    timestamp: str,
    resources: dict[str, dict[str, Any]],
    env_details: dict[str, dict[str, Any]],
    adaptation_stats: dict[str, dict[str, float | int]],
    api_enriched: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    def machine_or_na(value: Any, *, allowed: bool) -> Any:
        if not allowed:
            return NA_VALUE
        if value is None:
            return NA_VALUE
        return value

    rows: list[dict[str, Any]] = []
    for env_name, spec in resources.items():
        details = env_details.get(env_name, {})
        api_info = (api_enriched or {}).get(env_name, {})
        normalized = normalize_environment_label(env_name)
        stats = adaptation_stats.get(normalized, {})
        n_i = spec.get("n_i", {})
        g_i = spec.get("g_i", {})
        source = str(spec.get("source", "")).lower()
        cpu_gflops_source = str(spec.get("cpu_gflops_source", "")).lower()
        is_detected_local = source == "detected_local"
        has_benchmarked_cpu = cpu_gflops_source == "benchmarked_local"
        has_api = bool(api_info)

        rows.append({
            "timestamp": timestamp,
            "resource_id": env_name,
            "provider": machine_or_na(details.get("provider"), allowed=False),
            "label": machine_or_na(details.get("label"), allowed=False),
            "gpu_name": machine_or_na(api_info.get("gpu_name") or spec.get("gpu_name") or details.get("gpu"), allowed=has_api),
            "c_i_usd_per_hour": machine_or_na(api_info.get("c_i_usd_per_hour"), allowed=has_api),
            "m_i_gb": machine_or_na(api_info.get("m_i_gb") if has_api else spec.get("m_i"), allowed=is_detected_local or has_api),
            "d_i_gb": machine_or_na(api_info.get("d_i_gb") if has_api else spec.get("d_i"), allowed=is_detected_local or has_api),
            "e_i_kwh_per_hour": machine_or_na(spec.get("e_i"), allowed=is_detected_local),
            "n_cpu": machine_or_na(api_info.get("n_cpu") if has_api else n_i.get("CPU"), allowed=is_detected_local or has_api),
            "n_gpu": machine_or_na(api_info.get("n_gpu"), allowed=has_api),
            "n_tpu": machine_or_na(n_i.get("TPU"), allowed=False),
            "g_cpu_gflops_per_core": machine_or_na(g_i.get("CPU"), allowed=has_benchmarked_cpu),
            "g_gpu_gflops_per_core": machine_or_na(g_i.get("GPU"), allowed=False),
            "g_tpu_gflops_per_core": machine_or_na(g_i.get("TPU"), allowed=False),
            "q_period_gflops_avg": stats.get("q_period_gflops_avg"),
            "period_seconds_avg": stats.get("period_seconds_avg"),
            "cost_usd_period_avg": stats.get("cost_usd_period_avg"),
            "samples": stats.get("samples", 0),
            "source": machine_or_na(api_info.get("source") or spec.get("source"), allowed=has_api),
            "cpu_gflops_source": machine_or_na(spec.get("cpu_gflops_source"), allowed=False),
        })
    return rows


def _communication_rows(
    timestamp: str,
    resource_ids: list[str],
    env_details: dict[str, dict[str, Any]],
    reference_transfer_gb: float,
) -> list[dict[str, Any]]:
    del env_details, reference_transfer_gb
    rows: list[dict[str, Any]] = []
    for idx, resource_i in enumerate(resource_ids):
        for resource_j in resource_ids[idx + 1:]:
            rows.append({
                "timestamp": timestamp,
                "resource_i": resource_i,
                "resource_j": resource_j,
                "reference_transfer_gb": NA_VALUE,
                "comm_cost_usd_period": NA_VALUE,
            })
    return rows


def _extract_machine_type(label: str | None) -> str | None:
    if not label:
        return None
    match = re.search(r"\(([^()]+)\)", label)
    if not match:
        return None
    machine = match.group(1).strip().lower()
    return machine or None


def _region_from_zone(zone: str | None) -> str | None:
    if not zone:
        return None
    value = zone.strip().split("/")[-1]
    if value.count("-") < 2:
        return None
    return value.rsplit("-", 1)[0]


def _machine_family(machine_type: str) -> str:
    return machine_type.split("-", 1)[0].upper()


def _resolve_access_token() -> str | None:
    token = os.environ.get("GCP_ACCESS_TOKEN")
    if token:
        return token
    try:
        completed = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    token = completed.stdout.strip()
    return token or None


def _http_get_json(
    url: str,
    *,
    token: str | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_url = url
    if params:
        encoded = urllib.parse.urlencode({k: v for k, v in params.items() if v not in (None, "")})
        if encoded:
            separator = "&" if "?" in final_url else "?"
            final_url = f"{final_url}{separator}{encoded}"

    request_headers = {"Accept": "application/json"}
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(
        final_url,
        headers=request_headers,
    )
    with urllib.request.urlopen(request, timeout=API_TIMEOUT_SEC) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _fetch_metadata_value(path: str) -> str | None:
    try:
        payload = _http_get_json(
            f"{METADATA_BASE_URL}/{path}",
            headers={"Metadata-Flavor": "Google"},
        )
        if isinstance(payload, dict):
            return None
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        try:
            request = urllib.request.Request(
                f"{METADATA_BASE_URL}/{path}",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(request, timeout=API_TIMEOUT_SEC) as response:
                raw = response.read().decode("utf-8")
            return raw.strip() or None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            return None
    return str(payload).strip() if payload else None


def _resolve_project_and_zone(
    gcp_project: str | None,
    gcp_zone: str | None,
) -> tuple[str | None, str | None]:
    project = gcp_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    zone = gcp_zone
    if not project:
        project = _fetch_metadata_value("project/project-id")
    if not zone:
        zone_path = _fetch_metadata_value("instance/zone")
        zone = zone_path.split("/")[-1] if zone_path else None
    return project, zone


def _extract_price_from_sku(sku: dict[str, Any]) -> float | None:
    pricing_info = sku.get("pricingInfo", [])
    if not pricing_info:
        return None
    expression = pricing_info[0].get("pricingExpression", {})
    rates = expression.get("tieredRates", [])
    if not rates:
        return None
    unit_price = rates[0].get("unitPrice", {})
    units = unit_price.get("units", "0")
    nanos = unit_price.get("nanos", 0)
    try:
        return float(units) + (float(nanos) / 1_000_000_000.0)
    except (TypeError, ValueError):
        return None


def _build_sku_index(skus: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for sku in skus:
        category = sku.get("category", {})
        resource_group = str(category.get("resourceGroup", "")).upper()
        usage_type = str(category.get("usageType", "")).upper()
        if not resource_group or not usage_type:
            continue
        for region in sku.get("serviceRegions", []):
            key = (str(region), resource_group, usage_type)
            index.setdefault(key, []).append(sku)
    return index


def _pick_single_sku(
    skus: list[dict[str, Any]],
    *,
    matcher: re.Pattern[str] | None = None,
) -> dict[str, Any] | None:
    if matcher:
        filtered = [sku for sku in skus if matcher.search(str(sku.get("description", "")))]
    else:
        filtered = skus
    if len(filtered) == 1:
        return filtered[0]
    return None


def _normalize_gpu_name(raw_name: str | None) -> str | None:
    if not raw_name:
        return None
    value = raw_name.split("/")[-1].replace("_", "-").lower().strip()
    return value or None


def _fetch_compute_services(token: str) -> list[dict[str, Any]]:
    services: list[dict[str, Any]] = []
    next_token: str | None = None
    while True:
        response = _http_get_json(
            "https://cloudbilling.googleapis.com/v1/services",
            token=token,
            params={"pageToken": next_token},
        )
        services.extend(response.get("services", []))
        next_token = response.get("nextPageToken")
        if not next_token:
            break
    return services


def _fetch_service_skus(token: str, service_name: str) -> list[dict[str, Any]]:
    skus: list[dict[str, Any]] = []
    next_token: str | None = None
    while True:
        response = _http_get_json(
            f"https://cloudbilling.googleapis.com/v1/{service_name}/skus",
            token=token,
            params={"pageToken": next_token, "currencyCode": "USD"},
        )
        skus.extend(response.get("skus", []))
        next_token = response.get("nextPageToken")
        if not next_token:
            break
    return skus


def _fetch_machine_types_aggregated(project: str, token: str) -> dict[str, dict[str, Any]]:
    machine_types: dict[str, dict[str, Any]] = {}
    next_token: str | None = None
    while True:
        response = _http_get_json(
            f"https://compute.googleapis.com/compute/v1/projects/{project}/aggregated/machineTypes",
            token=token,
            params={"pageToken": next_token, "returnPartialSuccess": "true"},
        )
        for scope_data in response.get("items", {}).values():
            for machine in scope_data.get("machineTypes", []) or []:
                key = f"{machine.get('zone', '').split('/')[-1]}::{machine.get('name')}"
                machine_types[key] = machine
        next_token = response.get("nextPageToken")
        if not next_token:
            break
    return machine_types


def _fetch_accelerator_types_aggregated(project: str, token: str) -> dict[str, dict[str, Any]]:
    accelerators: dict[str, dict[str, Any]] = {}
    next_token: str | None = None
    while True:
        response = _http_get_json(
            f"https://compute.googleapis.com/compute/v1/projects/{project}/aggregated/acceleratorTypes",
            token=token,
            params={"pageToken": next_token, "returnPartialSuccess": "true"},
        )
        for scope_data in response.get("items", {}).values():
            for accelerator in scope_data.get("acceleratorTypes", []) or []:
                key = f"{accelerator.get('zone', '').split('/')[-1]}::{accelerator.get('name')}"
                accelerators[key] = accelerator
        next_token = response.get("nextPageToken")
        if not next_token:
            break
    return accelerators


def _fetch_machine_type_with_gcloud(project: str, zone: str, machine_type: str) -> dict[str, Any] | None:
    command = [
        "gcloud",
        "compute",
        "machine-types",
        "describe",
        machine_type,
        f"--project={project}",
        f"--zone={zone}",
        "--format=json",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_machine_payload(
    machine_types_index: dict[str, dict[str, Any]],
    *,
    zone: str,
    machine_type: str,
    project: str,
    gcloud_fallback: bool,
) -> dict[str, Any] | None:
    key = f"{zone}::{machine_type}"
    payload = machine_types_index.get(key)
    if payload:
        return payload
    if gcloud_fallback:
        return _fetch_machine_type_with_gcloud(project, zone, machine_type)
    return None


def _enrich_from_gcp_api(
    *,
    gcp_env_details: dict[str, dict[str, Any]],
    token: str,
    project: str,
    default_zone: str,
    gcloud_fallback: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    machine_types_index = _fetch_machine_types_aggregated(project, token)
    accelerator_index = _fetch_accelerator_types_aggregated(project, token)

    services = _fetch_compute_services(token)
    compute_service = next(
        (service for service in services if service.get("displayName") == COMPUTE_SERVICE_DISPLAY_NAME),
        None,
    )
    compute_skus = _fetch_service_skus(token, str(compute_service.get("name"))) if compute_service else []
    sku_index = _build_sku_index(compute_skus)

    enriched: dict[str, dict[str, Any]] = {}
    machine_rows: list[dict[str, Any]] = []

    for env_name, details in gcp_env_details.items():
        machine_type = _extract_machine_type(str(details.get("label")))
        if not machine_type:
            continue
        zone = str(details.get("zone") or default_zone)
        region = _region_from_zone(zone)
        if not region:
            continue

        machine = _resolve_machine_payload(
            machine_types_index,
            zone=zone,
            machine_type=machine_type,
            project=project,
            gcloud_fallback=gcloud_fallback,
        )
        if not machine:
            continue

        cpu = machine.get("guestCpus")
        memory_gib = float(machine.get("memoryMb", 0.0)) / 1024.0 if machine.get("memoryMb") is not None else None
        family = _machine_family(machine_type)
        usage_type = "ON_DEMAND"

        cpu_skus = sku_index.get((region, "CPU", usage_type), [])
        ram_skus = sku_index.get((region, "RAM", usage_type), [])
        local_ssd_skus = sku_index.get((region, "LOCAL_SSD", usage_type), [])
        gpu_skus = sku_index.get((region, "GPU", usage_type), [])

        family_cpu = _pick_single_sku(
            cpu_skus,
            matcher=re.compile(rf"\b{re.escape(family)}\b.*\bCore\b", re.IGNORECASE),
        )
        family_ram = _pick_single_sku(
            ram_skus,
            matcher=re.compile(rf"\b{re.escape(family)}\b.*\bRam\b", re.IGNORECASE),
        )
        local_ssd_sku = _pick_single_sku(local_ssd_skus)

        accelerators = machine.get("accelerators", []) or []
        gpu_count = 0
        gpu_model: str | None = None
        gpu_price_per_unit: float | None = None
        if accelerators:
            first_acc = accelerators[0]
            gpu_count = int(first_acc.get("guestAcceleratorCount", 0))
            gpu_type_name = str(first_acc.get("guestAcceleratorType", "")).split("/")[-1]
            gpu_model = _normalize_gpu_name(gpu_type_name)
            accelerator_info = accelerator_index.get(f"{zone}::{gpu_type_name}", {})
            if accelerator_info:
                gpu_model = _normalize_gpu_name(str(accelerator_info.get("name"))) or gpu_model
            if gpu_model:
                gpu_sku = _pick_single_sku(
                    gpu_skus,
                    matcher=re.compile(re.escape(gpu_model.replace("-", " ")), re.IGNORECASE),
                )
                gpu_price_per_unit = _extract_price_from_sku(gpu_sku) if gpu_sku else None

        cpu_price_per_vcpu = _extract_price_from_sku(family_cpu) if family_cpu else None
        ram_price_per_gib = _extract_price_from_sku(family_ram) if family_ram else None
        local_ssd_price_per_gib = _extract_price_from_sku(local_ssd_sku) if local_ssd_sku else None

        local_ssd_total_gib = None
        bundled_ssds = machine.get("scratchDisks", []) or machine.get("bundledLocalSsds", []) or []
        if isinstance(bundled_ssds, list) and bundled_ssds:
            size_mb = 0.0
            for disk in bundled_ssds:
                try:
                    size_mb += float(disk.get("diskSizeGb", 0.0)) * 1024.0
                except (TypeError, ValueError):
                    continue
            local_ssd_total_gib = size_mb / 1024.0 if size_mb else None

        total_cost = None
        if cpu is not None and memory_gib is not None and cpu_price_per_vcpu is not None and ram_price_per_gib is not None:
            total = float(cpu) * cpu_price_per_vcpu + float(memory_gib) * ram_price_per_gib
            if gpu_count and gpu_price_per_unit is not None:
                total += gpu_count * gpu_price_per_unit
            elif gpu_count:
                total = None
            if total is not None and local_ssd_total_gib and local_ssd_price_per_gib is not None:
                total += local_ssd_total_gib * local_ssd_price_per_gib
            total_cost = total

        machine_rows.append(
            {
                "resource_id": env_name,
                "project": project,
                "zone": zone,
                "region": region,
                "machine_type": machine_type,
                "guest_cpus": cpu,
                "memory_mb": machine.get("memoryMb"),
                "gpu_model": gpu_model or NA_VALUE,
                "gpu_count": gpu_count,
                "cpu_price_per_vcpu": cpu_price_per_vcpu if cpu_price_per_vcpu is not None else NA_VALUE,
                "ram_price_per_gib": ram_price_per_gib if ram_price_per_gib is not None else NA_VALUE,
                "gpu_price_per_unit": gpu_price_per_unit if gpu_price_per_unit is not None else NA_VALUE,
                "local_ssd_price_per_gib": local_ssd_price_per_gib if local_ssd_price_per_gib is not None else NA_VALUE,
                "estimated_total_cost_usd_hour": total_cost if total_cost is not None else NA_VALUE,
            }
        )
        enriched[env_name] = {
            "source": "gcp_api_compute+billing",
            "zone": zone,
            "region": region,
            "machine_type": machine_type,
            "m_i_gb": memory_gib,
            "d_i_gb": local_ssd_total_gib,
            "n_cpu": cpu,
            "n_gpu": gpu_count,
            "gpu_name": gpu_model,
            "c_i_usd_per_hour": total_cost,
            "price_components": {
                "cpu_price_per_vcpu": cpu_price_per_vcpu,
                "ram_price_per_gib": ram_price_per_gib,
                "gpu_price_per_unit": gpu_price_per_unit,
                "local_ssd_price_per_gib": local_ssd_price_per_gib,
            },
        }
    return enriched, machine_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coleta catálogo de recursos GCP/Colab e persiste CSVs em output/experiments/metrics.",
    )
    parser.add_argument(
        "--grid-config",
        default="gridsearch/config/grid_search_multienv.json",
        help="Caminho do grid config com environments.details.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Arquivo ou diretório com experiment_summary_*.csv (opcional).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(METRICS_DIR),
        help="Diretório de saída dos CSVs.",
    )
    parser.add_argument(
        "--device-type",
        default="GPU",
        choices=("CPU", "GPU", "TPU"),
        help="Tipo de dispositivo para o nome do arquivo.",
    )
    parser.add_argument(
        "--date",
        default=datetime.now().astimezone().strftime("%Y%m%d"),
        help="Sufixo de data (YYYYMMDD).",
    )
    parser.add_argument(
        "--reference-transfer-gb",
        type=float,
        default=REFERENCE_TRANSFER_GB,
        help="Volume de referência para custo de comunicação por período.",
    )
    parser.add_argument(
        "--fetch-gcp-api",
        action="store_true",
        help="Consulta opcional da API GCP Compute para machine types.",
    )
    parser.add_argument("--gcp-project", default=None, help="Projeto GCP (opcional; autodetecção por metadata).")
    parser.add_argument("--gcp-zone", default=None, help="Zona GCP (opcional; autodetecção por metadata).")
    parser.add_argument(
        "--gcloud-fallback",
        action="store_true",
        help="Habilita fallback com 'gcloud compute machine-types describe' quando necessário.",
    )
    return parser


def collect_gcp_catalog(
    *,
    grid_config: str,
    summary_path: str | None = None,
    output_dir: str | Path = METRICS_DIR,
    device_type: str = "GPU",
    date_suffix: str | None = None,
    reference_transfer_gb: float = REFERENCE_TRANSFER_GB,
    fetch_gcp_api: bool = False,
    gcp_project: str | None = None,
    gcp_zone: str | None = None,
    gcloud_fallback: bool = False,
    strict_no_gcp: bool = True,
) -> dict[str, Any]:
    """Coleta e persiste catálogos GCP/Colab no padrão CSV de métricas.

    Retorna metadados da coleta para permitir integração não bloqueante em
    fluxos de execução (single/grid).
    """
    date = date_suffix or datetime.now().astimezone().strftime("%Y%m%d")
    environments_details = read_environments_details(grid_config)
    gcp_env_details = {
        name: details
        for name, details in environments_details.items()
        if isinstance(details, dict) and _is_gcp_environment(name, details)
    }
    if not gcp_env_details:
        message = f"Nenhum ambiente GCP/Colab encontrado em {grid_config}."
        if strict_no_gcp:
            logger.error(message)
            raise ValueError(message)
        logger.info("%s Coleta de catálogo será ignorada.", message)
        return {
            "collected": False,
            "reason": "no_gcp_environment",
            "resource_path": None,
            "communication_path": None,
            "machine_types_path": None,
        }

    summary_stats = _collect_adaptation_stats(_summary_files(summary_path))
    resource_catalog = build_resource_catalog(gcp_env_details)
    timestamp = datetime.now().astimezone().isoformat()
    api_enriched: dict[str, dict[str, Any]] = {}
    machine_type_rows: list[dict[str, Any]] = []

    output_dir = Path(output_dir)
    resource_path = output_dir / f"gcp_resource_catalog_{device_type}-{date}.csv"
    communication_path = output_dir / f"gcp_communication_catalog_{device_type}-{date}.csv"

    if fetch_gcp_api:
        project, zone = _resolve_project_and_zone(gcp_project, gcp_zone)
        token = _resolve_access_token()
        if not project or not zone or not token:
            logger.warning(
                "Coleta via APIs oficiais do Google não pôde ser inicializada (project=%s zone=%s token=%s).",
                bool(project),
                bool(zone),
                bool(token),
            )
        else:
            try:
                api_enriched, machine_type_rows = _enrich_from_gcp_api(
                    gcp_env_details=gcp_env_details,
                    token=token,
                    project=project,
                    default_zone=zone,
                    gcloud_fallback=gcloud_fallback,
                )
            except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
                logger.warning("Falha ao coletar dados de APIs oficiais do Google: %s", exc)

    resource_rows = _resource_rows(
        timestamp,
        resource_catalog,
        gcp_env_details,
        summary_stats,
        api_enriched=api_enriched,
    )
    _write_csv(
        resource_path,
        resource_rows,
        [
            "timestamp", "resource_id", "provider", "label", "gpu_name",
            "c_i_usd_per_hour", "m_i_gb", "d_i_gb", "e_i_kwh_per_hour",
            "n_cpu", "n_gpu", "n_tpu",
            "g_cpu_gflops_per_core", "g_gpu_gflops_per_core", "g_tpu_gflops_per_core",
            "q_period_gflops_avg", "period_seconds_avg", "cost_usd_period_avg", "samples",
            "source", "cpu_gflops_source",
        ],
    )

    communication_rows = _communication_rows(
        timestamp,
        list(gcp_env_details.keys()),
        gcp_env_details,
        reference_transfer_gb,
    )
    _write_csv(
        communication_path,
        communication_rows,
        ["timestamp", "resource_i", "resource_j", "reference_transfer_gb", "comm_cost_usd_period"],
    )

    logger.info("Catálogo de recursos GCP salvo em %s", resource_path)
    logger.info("Catálogo de comunicação GCP salvo em %s", communication_path)

    machine_types_path: Path | None = None
    if machine_type_rows:
        api_rows = []
        for row in machine_type_rows:
            api_rows.append({"timestamp": timestamp, **row})
        api_path = output_dir / f"gcp_machine_types_{device_type}-{date}.csv"
        _write_csv(
            api_path,
            api_rows,
            [
                "timestamp", "resource_id", "project", "zone", "region",
                "machine_type", "guest_cpus", "memory_mb", "gpu_model", "gpu_count",
                "cpu_price_per_vcpu", "ram_price_per_gib",
                "gpu_price_per_unit", "local_ssd_price_per_gib",
                "estimated_total_cost_usd_hour",
            ],
        )
        logger.info("Catálogo de machine types GCP salvo em %s", api_path)
        machine_types_path = api_path

    return {
        "collected": True,
        "reason": None,
        "resource_path": str(resource_path),
        "communication_path": str(communication_path),
        "machine_types_path": str(machine_types_path) if machine_types_path else None,
        "resource_rows": len(resource_rows),
        "communication_rows": len(communication_rows),
    }


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    args = _build_parser().parse_args()

    try:
        collect_gcp_catalog(
            grid_config=args.grid_config,
            summary_path=args.summary_path,
            output_dir=args.output_dir,
            device_type=args.device_type,
            date_suffix=args.date,
            reference_transfer_gb=args.reference_transfer_gb,
            fetch_gcp_api=args.fetch_gcp_api,
            gcp_project=args.gcp_project,
            gcp_zone=args.gcp_zone,
            gcloud_fallback=args.gcloud_fallback,
            strict_no_gcp=True,
        )
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        logger.error("Falha na coleta de catálogo GCP: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
