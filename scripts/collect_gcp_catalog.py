"""Cria, para e exclui uma instância GCP para validar integração com Compute API."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from google.api_core import exceptions as gcp_exceptions
from google.cloud import billing_v1, compute_v1

from config.config import settings

DEFAULT_COMPUTE_ENGINE_SERVICE_ID = "6F81-5844-456A"
COMPUTE_ENGINE_SERVICE_DISPLAY_NAME = "Compute Engine"

_SERVICE_ID_CACHE: dict[str, str] = {}

SHARED_CORE_BILLABLE_VCPUS = {
    "e2-micro": 0.25,
    "e2-small": 0.5,
    "e2-medium": 1.0,
}

DISK_SKU_PREFIXES = {
    "pd-standard": "Storage PD Capacity",
    "pd-balanced": "Balanced PD Capacity",
    "pd-ssd": "SSD backed PD Capacity",
}

GPU_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "l4": ("L4", "NVIDIA L4", "TESLA L4"),
    "t4": ("T4", "NVIDIA T4", "TESLA T4"),
    "a100": ("A100", "NVIDIA A100", "TESLA A100"),
    "h100": ("H100", "NVIDIA H100"),
    "v100": ("V100", "NVIDIA V100", "TESLA V100"),
    "p100": ("P100", "NVIDIA P100", "TESLA P100"),
    "p4": ("P4", "NVIDIA P4", "TESLA P4"),
}

LLM_PROFILES: dict[str, dict[str, Any]] = {
    "none": {},
    "llm-small": {
        "machine_type": "n1-standard-16",
        "boot_disk_size_gb": 500,
        "boot_disk_type": "pd-ssd",
        "enable_gpu": True,
        "gpu_type": "nvidia-tesla-l4",
        "gpu_count": 1,
    },
    "llm-medium": {
        "machine_type": "n1-standard-32",
        "boot_disk_size_gb": 1000,
        "boot_disk_type": "pd-ssd",
        "enable_gpu": True,
        "gpu_type": "nvidia-tesla-a100",
        "gpu_count": 1,
    },
    "llm-large": {
        "machine_type": "n1-standard-64",
        "boot_disk_size_gb": 2000,
        "boot_disk_type": "pd-ssd",
        "enable_gpu": True,
        "gpu_type": "nvidia-tesla-a100",
        "gpu_count": 4,
    },
}


@dataclass
class ProvisioningOptions:
    machine_type: str
    boot_disk_size_gb: int
    boot_disk_type: str
    use_spot: bool
    enable_gpu: bool
    gpu_type: str | None
    gpu_count: int


def _print_provisioning_plan(
    cfg: dict[str, str],
    *,
    action: str,
    options: ProvisioningOptions,
) -> None:
    print("=== DRY RUN: plano de execução (sem alterações) ===")
    print(f"Projeto/Zona: {cfg['project']}/{cfg['zone']}")
    print(f"Instância alvo: {cfg['instance']}")
    print(f"Ação solicitada: {action}")
    print(f"Machine type efetivo: {options.machine_type}")
    print(f"Disco de boot: {options.boot_disk_size_gb} GB ({options.boot_disk_type})")
    print(f"Spot: {'sim' if options.use_spot else 'não'}")
    if options.enable_gpu:
        print(f"GPU: {options.gpu_count} x {options.gpu_type}")
    else:
        print("GPU: desabilitada")


def _gpu_type_short_name(gpu_type: str) -> str:
    short = gpu_type.lower().split("/")[-1]
    short = short.replace("nvidia-tesla-", "")
    short = short.replace("nvidia-", "")
    short = short.replace("tesla-", "")
    return short.upper().replace("-", "_")


def _gpu_quota_metric_candidates(gpu_type: str, use_spot: bool) -> list[str]:
    model = _gpu_type_short_name(gpu_type)
    candidates = []
    if use_spot:
        candidates.append(f"PREEMPTIBLE_NVIDIA_{model}_GPUS")
    candidates.append(f"NVIDIA_{model}_GPUS")
    candidates.append("GPUS_ALL_REGIONS")
    return candidates


def _suggest_alternative_zones_for_gpu(
    cfg: dict[str, str],
    *,
    accelerator_name: str,
    gpu_type: str,
    limit: int = 8,
) -> list[str]:
    accelerator_client = compute_v1.AcceleratorTypesClient()
    zones_found: list[str] = []
    normalized_model = _normalize_gpu_model(gpu_type) or ""

    def _matches(acc_name: str) -> bool:
        if acc_name == accelerator_name:
            return True
        if normalized_model:
            compact_model = normalized_model.replace("-", "")
            acc_norm = acc_name.lower().replace("-", "")
            return compact_model in acc_norm
        return False

    try:
        for zone_key, scoped_list in accelerator_client.aggregated_list(project=cfg["project"]):
            if not scoped_list.accelerator_types:
                continue
            zone_name = zone_key.split("/")[-1]
            for acc in scoped_list.accelerator_types:
                if _matches(acc.name):
                    zones_found.append(zone_name)
                    break
    except gcp_exceptions.GoogleAPICallError:
        return []

    if not zones_found:
        return []

    preferred_region = _zone_to_region(cfg["zone"])

    def _zone_rank(zone: str) -> tuple[int, str]:
        same_region = 0 if zone.startswith(f"{preferred_region}-") else 1
        return (same_region, zone)

    ranked = sorted(set(zones_found), key=_zone_rank)
    ranked = [zone for zone in ranked if zone != cfg["zone"]]
    return ranked[:limit]


def _precheck_gpu_zone_and_quota(
    cfg: dict[str, str],
    options: ProvisioningOptions,
    *,
    strict: bool,
) -> None:
    if not options.enable_gpu or not options.gpu_type:
        return

    accelerator_name = options.gpu_type.split("/")[-1]

    accelerator_client = compute_v1.AcceleratorTypesClient()
    try:
        accelerator_client.get(
            project=cfg["project"],
            zone=cfg["zone"],
            accelerator_type=accelerator_name,
        )
        print(
            "Pre-check GPU: tipo de acelerador disponível na zona "
            f"({accelerator_name} em {cfg['zone']})."
        )
    except gcp_exceptions.NotFound as exc:
        message = f"GPU '{accelerator_name}' não está disponível na zona {cfg['zone']}."
        suggestions = _suggest_alternative_zones_for_gpu(
            cfg,
            accelerator_name=accelerator_name,
            gpu_type=options.gpu_type,
        )
        if suggestions:
            message += (
                " Zonas alternativas sugeridas: "
                + ", ".join(suggestions)
                + "."
            )
        if strict:
            raise RuntimeError(message) from exc
        print(f"Aviso (dry-run): {message}")
        return
    except gcp_exceptions.GoogleAPICallError as exc:
        print(
            "Aviso: não foi possível validar disponibilidade zonal do acelerador via API. "
            "A execução continuará sem esta garantia."
        )
        print(f"Detalhe do pre-check de zona: {exc}")
        return

    region = _zone_to_region(cfg["zone"])
    region_client = compute_v1.RegionsClient()
    try:
        region_info = region_client.get(project=cfg["project"], region=region)
    except gcp_exceptions.GoogleAPICallError as exc:
        print(
            "Aviso: não foi possível consultar quota regional de GPU. "
            "A execução continuará sem validação de quota."
        )
        print(f"Detalhe do pre-check de quota: {exc}")
        return

    quotas = {q.metric: q for q in (region_info.quotas or [])}
    metric = None
    for candidate in _gpu_quota_metric_candidates(options.gpu_type, options.use_spot):
        if candidate in quotas:
            metric = candidate
            break

    if metric is None:
        print(
            "Aviso: nenhuma métrica de quota específica de GPU foi encontrada para validação "
            "(verifique quotas manualmente no projeto)."
        )
        return

    quota = quotas[metric]
    available = float(quota.limit) - float(quota.usage)
    print(
        f"Pre-check GPU quota ({metric}): limite={quota.limit}, "
        f"uso={quota.usage}, disponível={available}."
    )
    if available < options.gpu_count:
        message = (
            "Quota insuficiente para GPU. "
            f"Necessário={options.gpu_count}, disponível={available} "
            f"(métrica {metric}, região {region})."
        )
        if strict:
            raise RuntimeError(message)
        print(f"Aviso (dry-run): {message}")


def _required_setting(name: str) -> str:
    value = settings.get(name)
    if not value:
        raise RuntimeError(f"Configuração obrigatória ausente: {name}")
    return str(value)


def _load_config() -> dict[str, str]:
    return {
        "project": _required_setting("PROJECT_ID"),
        "zone": _required_setting("ZONE_NAME"),
        "instance": _required_setting("INSTANCE_NAME"),
        "machine_type": _required_setting("MACHINE_TYPE"),
        "source_image": _required_setting("SRC_IMG"),
        "network": _required_setting("INTERFACE_NETWORK"),
        "nat_name": _required_setting("NAT_NAME"),
        "nat_type": _required_setting("NAT_TYPE"),
    }


def _machine_type_uri(project: str, zone: str, machine_type: str) -> str:
    if "/" in machine_type:
        return machine_type
    return f"projects/{project}/zones/{zone}/machineTypes/{machine_type}"


def _disk_type_uri(project: str, zone: str, disk_type: str) -> str:
    mapping = {
        "pd-standard": "pd-standard",
        "pd-balanced": "pd-balanced",
        "pd-ssd": "pd-ssd",
    }
    if disk_type not in mapping:
        raise ValueError(
            f"Tipo de disco não suportado: {disk_type}. Use: {sorted(mapping.keys())}"
        )
    return f"projects/{project}/zones/{zone}/diskTypes/{mapping[disk_type]}"


def _build_instance_payload(cfg: dict[str, str], options: ProvisioningOptions) -> compute_v1.Instance:
    disk = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=compute_v1.AttachedDiskInitializeParams(
            source_image=cfg["source_image"],
            disk_size_gb=options.boot_disk_size_gb,
            disk_type=_disk_type_uri(cfg["project"], cfg["zone"], options.boot_disk_type),
        ),
    )
    network_interface = compute_v1.NetworkInterface(
        network=cfg["network"],
        access_configs=[
            compute_v1.AccessConfig(name=cfg["nat_name"], type_=cfg["nat_type"]),
        ],
    )

    scheduling = compute_v1.Scheduling(
        provisioning_model="SPOT" if options.use_spot else "STANDARD",
        on_host_maintenance="TERMINATE" if options.enable_gpu else "MIGRATE",
        automatic_restart=(not options.use_spot and not options.enable_gpu),
    )

    guest_accelerators: list[compute_v1.AcceleratorConfig] = []
    if options.enable_gpu and options.gpu_type:
        accelerator_type = (
            options.gpu_type
            if "/" in options.gpu_type
            else (
                f"projects/{cfg['project']}/zones/{cfg['zone']}/"
                f"acceleratorTypes/{options.gpu_type}"
            )
        )
        guest_accelerators = [
            compute_v1.AcceleratorConfig(
                accelerator_type=accelerator_type,
                accelerator_count=options.gpu_count,
            )
        ]

    return compute_v1.Instance(
        name=cfg["instance"],
        machine_type=_machine_type_uri(cfg["project"], cfg["zone"], options.machine_type),
        disks=[disk],
        network_interfaces=[network_interface],
        scheduling=scheduling,
        guest_accelerators=guest_accelerators,
    )


def _get_instance(
    client: compute_v1.InstancesClient,
    *,
    project: str,
    zone: str,
    instance_name: str,
) -> Any | None:
    try:
        return client.get(project=project, zone=zone, instance=instance_name)
    except gcp_exceptions.NotFound:
        return None


def _create_if_missing(
    client: compute_v1.InstancesClient,
    cfg: dict[str, str],
    options: ProvisioningOptions,
) -> None:
    existing = _get_instance(
        client,
        project=cfg["project"],
        zone=cfg["zone"],
        instance_name=cfg["instance"],
    )
    if existing is not None:
        print(f"Instance {cfg['instance']} already exists in zone {cfg['zone']}.")
        return

    print(f"Iniciando a criação da instância '{cfg['instance']}' com Debian 12...")
    operation = client.insert(
        project=cfg["project"],
        zone=cfg["zone"],
        instance_resource=_build_instance_payload(cfg, options),
    )
    operation.result()
    print(f"Instância '{cfg['instance']}' criada com sucesso!")


def _print_instances(client: compute_v1.InstancesClient, cfg: dict[str, str]) -> None:
    for instance in client.list(project=cfg["project"], zone=cfg["zone"]):
        print(f"Instâncias encontradas: {instance.name} ({instance.status})")


def _stop_if_exists(client: compute_v1.InstancesClient, cfg: dict[str, str]) -> None:
    instance = _get_instance(
        client,
        project=cfg["project"],
        zone=cfg["zone"],
        instance_name=cfg["instance"],
    )
    if instance is None:
        print(f"Instância '{cfg['instance']}' não encontrada para parada.")
        return
    if instance.status == "TERMINATED":
        print(f"Instância '{cfg['instance']}' já está parada.")
        return

    print(f"Parando a instância '{cfg['instance']}'...")
    operation = client.stop(project=cfg["project"], zone=cfg["zone"], instance=cfg["instance"])
    operation.result()
    print(f"Instância '{cfg['instance']}' parada com sucesso!")


def _delete_if_exists(client: compute_v1.InstancesClient, cfg: dict[str, str]) -> None:
    instance = _get_instance(
        client,
        project=cfg["project"],
        zone=cfg["zone"],
        instance_name=cfg["instance"],
    )
    if instance is None:
        print(f"Instância '{cfg['instance']}' não encontrada para exclusão.")
        return

    print(f"Excluindo a instância '{cfg['instance']}'...")
    operation = client.delete(project=cfg["project"], zone=cfg["zone"], instance=cfg["instance"])
    operation.result()
    print(f"Instância '{cfg['instance']}' excluída com sucesso!")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa operações de ciclo de vida de instância no Google Compute Engine."
    )
    parser.add_argument(
        "--action",
        choices=["list", "create", "stop", "delete", "cycle"],
        default="cycle",
        help="Ação a executar. Padrão: cycle (create -> list -> stop -> delete).",
    )
    parser.add_argument(
        "--safe",
        action="store_true",
        help="Modo seguro: exige ação explícita e impede execução do ciclo completo.",
    )
    parser.add_argument(
        "--confirm-delete",
        action="store_true",
        help="Confirma a exclusão sem prompt interativo (use apenas em automação controlada).",
    )
    parser.add_argument(
        "--fetch-catalog",
        action="store_true",
        help="Coleta detalhes de recursos da machine type e SKUs da Cloud Billing.",
    )
    parser.add_argument(
        "--catalog-limit",
        type=int,
        default=5,
        metavar="N",
        help="Limite de SKUs exibidos na consulta de catálogo (padrão: 5).",
    )
    parser.add_argument(
        "--disk-type",
        choices=sorted(DISK_SKU_PREFIXES.keys()),
        default="pd-balanced",
        help="Tipo de disco para estimativa de custo (padrão: pd-balanced).",
    )
    parser.add_argument(
        "--disk-size-gb",
        type=int,
        default=10,
        help="Tamanho do disco em GB para estimativa de custo (padrão: 10).",
    )
    parser.add_argument(
        "--spot",
        action="store_true",
        help="Usa SKUs preemptibles/spot para estimativa de custo de compute.",
    )
    parser.add_argument(
        "--llm-profile",
        choices=sorted(LLM_PROFILES.keys()),
        default="none",
        help=(
            "Perfil pré-definido para treino de LLM com recursos altos "
            "(none, llm-small, llm-medium, llm-large)."
        ),
    )
    parser.add_argument(
        "--machine-type",
        help="Sobrescreve o machine type de configuração.",
    )
    parser.add_argument(
        "--boot-disk-size-gb",
        type=int,
        help="Sobrescreve tamanho do disco de boot (GB) para provisionamento.",
    )
    parser.add_argument(
        "--boot-disk-type",
        choices=sorted(DISK_SKU_PREFIXES.keys()),
        help="Sobrescreve tipo do disco de boot para provisionamento.",
    )
    parser.add_argument(
        "--enable-gpu",
        action="store_true",
        help="Habilita GPU na criação da VM.",
    )
    parser.add_argument(
        "--gpu-type",
        help=(
            "Tipo da GPU (ex: nvidia-tesla-l4, nvidia-tesla-a100) "
            "ou URI completa do accelerator type."
        ),
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Quantidade de GPUs a anexar quando --enable-gpu estiver ativo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o plano e custos sem executar ações mutáveis na instância.",
    )
    return parser


def _resolve_provisioning_options(args: argparse.Namespace, cfg: dict[str, str]) -> ProvisioningOptions:
    profile = LLM_PROFILES[args.llm_profile]

    machine_type = args.machine_type or profile.get("machine_type") or _machine_type_name(
        cfg["machine_type"]
    )
    boot_disk_size_gb = (
        args.boot_disk_size_gb
        if args.boot_disk_size_gb is not None
        else int(profile.get("boot_disk_size_gb", args.disk_size_gb))
    )
    boot_disk_type = args.boot_disk_type or profile.get("boot_disk_type") or args.disk_type

    enable_gpu = bool(args.enable_gpu or profile.get("enable_gpu", False))
    gpu_type = args.gpu_type or profile.get("gpu_type")
    gpu_count = int(profile.get("gpu_count", args.gpu_count)) if profile else int(args.gpu_count)

    if enable_gpu and not gpu_type:
        raise RuntimeError("--enable-gpu requer --gpu-type (ou um --llm-profile com GPU).")
    if gpu_count < 1:
        raise RuntimeError("--gpu-count deve ser maior ou igual a 1.")
    if boot_disk_size_gb < 1:
        raise RuntimeError("--boot-disk-size-gb deve ser maior ou igual a 1.")

    return ProvisioningOptions(
        machine_type=machine_type,
        boot_disk_size_gb=boot_disk_size_gb,
        boot_disk_type=boot_disk_type,
        use_spot=bool(args.spot),
        enable_gpu=enable_gpu,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
    )


def _confirm_delete(cfg: dict[str, str]) -> bool:
    prompt = (
        f"Confirma exclusão da instância '{cfg['instance']}' em "
        f"{cfg['project']}/{cfg['zone']}? [y/N]: "
    )
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _run_action(
    client: compute_v1.InstancesClient,
    cfg: dict[str, str],
    options: ProvisioningOptions,
    action: str,
    *,
    confirm_delete: bool,
) -> None:
    if action == "list":
        _print_instances(client, cfg)
        return
    if action == "create":
        _create_if_missing(client, cfg, options)
        return
    if action == "stop":
        _stop_if_exists(client, cfg)
        return
    if action == "delete":
        if not confirm_delete and not _confirm_delete(cfg):
            print("Exclusão cancelada pelo usuário.")
            return
        _delete_if_exists(client, cfg)
        return
    if action == "cycle":
        _create_if_missing(client, cfg, options)
        _print_instances(client, cfg)
        _stop_if_exists(client, cfg)
        if not confirm_delete and not _confirm_delete(cfg):
            print("Exclusão cancelada pelo usuário.")
            return
        _delete_if_exists(client, cfg)
        return
    raise RuntimeError(f"Ação inválida: {action}")


def _machine_type_name(machine_type: str) -> str:
    return machine_type.split("/")[-1]


def _print_machine_type_details(
    cfg: dict[str, str],
    *,
    machine_type_name: str | None = None,
) -> compute_v1.MachineType:
    machine_type_name = machine_type_name or _machine_type_name(cfg["machine_type"])
    machine_types_client = compute_v1.MachineTypesClient()

    machine_type_info = machine_types_client.get(
        project=cfg["project"],
        zone=cfg["zone"],
        machine_type=machine_type_name,
    )

    print(f"=== Detalhes de Recursos para o Tipo de Máquina: {machine_type_name} ===")
    print(f"Descrição: {machine_type_info.description}")
    print(f"Quantidade de vCPUs: {machine_type_info.guest_cpus}")
    print(
        "Memória RAM: "
        f"{machine_type_info.memory_mb} MB "
        f"(aprox. {machine_type_info.memory_mb / 1024:.2f} GB)"
    )

    if machine_type_info.accelerators:
        for acc in machine_type_info.accelerators:
            print(
                "Acelerador (GPU): "
                f"{acc.guest_accelerator_type} "
                f"(Quantidade: {acc.guest_accelerator_count})"
            )
    else:
        print("Aceleradores/GPUs: Não possui (tipo de máquina compartilhado/básico)")

    print(f"Limite de Discos Persistentes: {machine_type_info.maximum_persistent_disks} discos")
    print(
        "Tamanho Máximo Total de Discos: "
        f"{machine_type_info.maximum_persistent_disks_size_gb} GB"
    )
    return machine_type_info


def _extract_unit_price(sku: billing_v1.Sku) -> float:
    if not sku.pricing_info:
        return 0.0
    tiered_rates = sku.pricing_info[0].pricing_expression.tiered_rates
    for rate in tiered_rates:
        price = rate.unit_price.units + (rate.unit_price.nanos * 1e-9)
        if price > 0:
            return price
    return 0.0


def _normalize_gpu_model(gpu_type: str | None) -> str | None:
    if not gpu_type:
        return None
    raw = gpu_type.lower().split("/")[-1]
    raw = raw.replace("nvidia-tesla-", "")
    raw = raw.replace("nvidia-", "")
    raw = raw.replace("tesla-", "")
    return raw.strip()


def _gpu_match_tokens(gpu_type: str | None) -> tuple[str, ...]:
    model = _normalize_gpu_model(gpu_type)
    if not model:
        return tuple()
    aliases = GPU_MODEL_ALIASES.get(model, (model.upper(),))
    normalized_aliases = list(aliases)
    normalized_aliases.append(model.replace("-", "").upper())
    return tuple(dict.fromkeys(normalized_aliases))


def _resolve_service_id(
    billing_client: billing_v1.CloudCatalogClient,
    *,
    display_name: str,
    fallback_service_id: str,
) -> str:
    cache_key = display_name.lower()
    cached = _SERVICE_ID_CACHE.get(cache_key)
    if cached:
        return cached

    try:
        for service in billing_client.list_services():
            current_display_name = (service.display_name or "").strip().lower()
            if current_display_name != cache_key:
                continue

            service_id = service.service_id or service.name.split("/")[-1]
            if service_id:
                _SERVICE_ID_CACHE[cache_key] = service_id
                return service_id
    except (
        gcp_exceptions.PermissionDenied,
        gcp_exceptions.FailedPrecondition,
        gcp_exceptions.Unauthenticated,
        gcp_exceptions.GoogleAPICallError,
    ):
        return fallback_service_id

    return fallback_service_id


def _zone_to_region(zone: str) -> str:
    parts = zone.split("-")
    if len(parts) < 2:
        raise ValueError(f"Zona inválida para derivar região: {zone}")
    return "-".join(parts[:-1])


def _estimate_instance_cost_e2(
    *,
    machine_type_name: str,
    vcpus_api: int,
    ram_gb: float,
    zone: str,
    disk_type: str,
    disk_size_gb: int,
    is_spot: bool,
    gpu_type: str | None,
    gpu_count: int,
) -> dict[str, float | str]:
    if disk_type not in DISK_SKU_PREFIXES:
        raise ValueError(
            f"Tipo de disco não suportado: {disk_type}. Use: {sorted(DISK_SKU_PREFIXES.keys())}"
        )

    region = _zone_to_region(zone)
    compute_usage_type = "Preemptible" if is_spot else "OnDemand"
    billable_vcpus = SHARED_CORE_BILLABLE_VCPUS.get(machine_type_name, float(vcpus_api))
    disk_prefix = DISK_SKU_PREFIXES[disk_type]

    billing_client = billing_v1.CloudCatalogClient()
    service_id = _resolve_service_id(
        billing_client,
        display_name=COMPUTE_ENGINE_SERVICE_DISPLAY_NAME,
        fallback_service_id=DEFAULT_COMPUTE_ENGINE_SERVICE_ID,
    )
    skus_pager = billing_client.list_skus(parent=f"services/{service_id}")

    price_core = None
    price_ram = None
    price_disk_gb_month = None
    price_gpu_hour = 0.0

    gpu_tokens = _gpu_match_tokens(gpu_type)
    best_gpu_score = -1

    for sku in skus_pager:
        if region not in sku.service_regions:
            continue

        family = sku.category.resource_family
        usage = sku.category.usage_type
        description = sku.description

        if family == "Compute" and usage == compute_usage_type:
            if price_core is None and (
                description.startswith(("E2 Instance Core running in", "Spot Preemptible E2 Instance Core running in"))
            ):
                price_core = _extract_unit_price(sku)
            elif price_ram is None and (
                description.startswith(("E2 Instance Ram running in", "Spot Preemptible E2 Instance Ram running in"))
            ):
                price_ram = _extract_unit_price(sku)
        elif family == "Storage" and usage == "OnDemand":  # noqa: SIM102
            if price_disk_gb_month is None and description.startswith(disk_prefix):
                price_disk_gb_month = _extract_unit_price(sku)
        elif family in {"Compute", "GPU"} and gpu_tokens:
            desc_upper = description.upper()
            group_upper = (sku.category.resource_group or "").upper()
            if "GPU" not in desc_upper and "GPU" not in group_upper:
                continue
            if not any(token in desc_upper for token in gpu_tokens):
                continue

            candidate_price = _extract_unit_price(sku)
            if candidate_price <= 0:
                continue

            # Prefere SKU com o usage exato (OnDemand vs Preemptible),
            # mas aceita fallback quando o catálogo não separa por usage_type.
            score = 2 if usage == compute_usage_type else 1
            if score > best_gpu_score:
                best_gpu_score = score
                price_gpu_hour = candidate_price

        if None not in (price_core, price_ram, price_disk_gb_month):
            break

    if None in (price_core, price_ram, price_disk_gb_month):
        raise ValueError(
            "SKUs incompletos para cálculo de custo "
            f"(region={region}, core={price_core}, ram={price_ram}, disco={price_disk_gb_month})."
        )

    cost_compute_hour = (price_core * billable_vcpus) + (price_ram * ram_gb)
    cost_gpu_hour = price_gpu_hour * gpu_count
    cost_compute_month = cost_compute_hour * 730
    cost_gpu_month = cost_gpu_hour * 730

    cost_disk_month = price_disk_gb_month * disk_size_gb
    cost_disk_hour = cost_disk_month / 730

    return {
        "region": region,
        "machine_type": machine_type_name,
        "billable_vcpus": billable_vcpus,
        "ram_gb": ram_gb,
        "disk_type": disk_type,
        "disk_size_gb": float(disk_size_gb),
        "cost_compute_hour_usd": cost_compute_hour,
        "cost_gpu_hour_usd": cost_gpu_hour,
        "cost_disk_hour_usd": cost_disk_hour,
        "cost_total_hour_running_usd": cost_compute_hour + cost_gpu_hour + cost_disk_hour,
        "cost_total_month_running_usd": cost_compute_month + cost_gpu_month + cost_disk_month,
        "cost_month_terminated_usd": cost_disk_month,
    }


def _print_cost_estimate(
    cfg: dict[str, str],
    *,
    machine_type_name: str,
    vcpus_api: int,
    ram_gb: float,
    disk_type: str,
    disk_size_gb: int,
    is_spot: bool,
    gpu_type: str | None,
    gpu_count: int,
) -> None:
    result = _estimate_instance_cost_e2(
        machine_type_name=machine_type_name,
        vcpus_api=vcpus_api,
        ram_gb=ram_gb,
        zone=cfg["zone"],
        disk_type=disk_type,
        disk_size_gb=disk_size_gb,
        is_spot=is_spot,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
    )

    print("=== Estimativa de Custo da VM (Catalog API) ===")
    print(f"Instância: {result['machine_type']} ({result['region']})")
    print(f"vCPUs faturáveis: {result['billable_vcpus']:.2f}")
    print(f"RAM: {result['ram_gb']:.2f} GB")
    print(f"Disco: {int(result['disk_size_gb'])} GB ({result['disk_type']})")
    print(f"Custo compute/hora: ${result['cost_compute_hour_usd']:.6f} USD")
    print(f"Custo GPU/hora:     ${result['cost_gpu_hour_usd']:.6f} USD")
    print(f"Custo disco/hora:   ${result['cost_disk_hour_usd']:.6f} USD")
    print(f"Custo total/hora (RUNNING): ${result['cost_total_hour_running_usd']:.6f} USD")
    print(f"Custo total/mês (RUNNING):  ${result['cost_total_month_running_usd']:.2f} USD")
    print(f"Custo/mês (TERMINATED):     ${result['cost_month_terminated_usd']:.2f} USD")
    if gpu_type and gpu_count > 0 and float(result["cost_gpu_hour_usd"]) == 0.0:
        print(
            "Aviso: GPU habilitada, mas SKU de GPU não encontrado no filtro atual "
            "(o custo de GPU pode estar subestimado)."
        )


def _print_billing_catalog(cfg: dict[str, str], limit: int) -> None:
    billing_client = billing_v1.CloudCatalogClient()
    region = _zone_to_region(cfg["zone"])
    print("Consultando a Cloud Billing Catalog API para listar SKUs do Compute Engine...")
    try:
        service_id = _resolve_service_id(
            billing_client,
            display_name=COMPUTE_ENGINE_SERVICE_DISPLAY_NAME,
            fallback_service_id=DEFAULT_COMPUTE_ENGINE_SERVICE_ID,
        )
        skus_pager = billing_client.list_skus(parent=f"services/{service_id}")
        count = 0
        for sku in skus_pager:
            if region not in sku.service_regions:
                continue
            if "E2 Instance" not in sku.description and "Micro" not in sku.description:
                continue

            print(f"=== SKU: {sku.description} ===")
            print(f"ID do SKU: {sku.sku_id}")
            print(f"Regiões aplicáveis: {list(sku.service_regions)}")

            pricing_info = sku.pricing_info[0] if sku.pricing_info else None
            if pricing_info:
                for rate in pricing_info.pricing_expression.tiered_rates:
                    unit_price = rate.unit_price
                    real_price = unit_price.units + (unit_price.nanos / 1e9)
                    print(
                        f"Preço base: ${real_price:.6f} "
                        f"por {pricing_info.pricing_expression.usage_unit}"
                    )
            print("-" * 40)
            count += 1
            if count >= limit:
                break
    except (
        gcp_exceptions.PermissionDenied,
        gcp_exceptions.FailedPrecondition,
        gcp_exceptions.Unauthenticated,
        gcp_exceptions.GoogleAPICallError,
    ) as exc:
        print(
            "Aviso: não foi possível consultar a Cloud Billing Catalog API. "
            "A execução continuará sem dados de preço de catálogo."
        )
        print(f"Detalhe da API Billing: {exc}")


def _collect_gcp_catalog(
    cfg: dict[str, str],
    *,
    limit: int,
    disk_type: str,
    disk_size_gb: int,
    is_spot: bool,
    machine_type_name: str,
    gpu_type: str | None,
    gpu_count: int,
) -> None:
    machine_type_info = _print_machine_type_details(cfg, machine_type_name=machine_type_name)
    _print_billing_catalog(cfg, limit)
    try:
        _print_cost_estimate(
            cfg,
            machine_type_name=machine_type_name,
            vcpus_api=machine_type_info.guest_cpus,
            ram_gb=machine_type_info.memory_mb / 1024,
            disk_type=disk_type,
            disk_size_gb=disk_size_gb,
            is_spot=is_spot,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
        )
    except (
        ValueError,
        gcp_exceptions.PermissionDenied,
        gcp_exceptions.FailedPrecondition,
        gcp_exceptions.Unauthenticated,
        gcp_exceptions.GoogleAPICallError,
    ) as exc:
        print(
            "Aviso: não foi possível calcular a estimativa completa de custo da VM. "
            "A execução continuará sem a seção de custo."
        )
        print(f"Detalhe da estimativa: {exc}")


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        if args.safe and args.action == "cycle":
            raise RuntimeError("Modo seguro requer --action explícita (list/create/stop/delete).")
        if args.catalog_limit < 1:
            raise RuntimeError("--catalog-limit deve ser maior ou igual a 1.")
        if args.disk_size_gb < 1:
            raise RuntimeError("--disk-size-gb deve ser maior ou igual a 1.")
        cfg = _load_config()
        options = _resolve_provisioning_options(args, cfg)
        if args.action in {"create", "cycle"}:
            _precheck_gpu_zone_and_quota(cfg, options, strict=not args.dry_run)
        client = compute_v1.InstancesClient()
        if args.dry_run:
            _print_provisioning_plan(cfg, action=args.action, options=options)
        else:
            _run_action(client, cfg, options, args.action, confirm_delete=args.confirm_delete)
        if args.fetch_catalog:
            _collect_gcp_catalog(
                cfg,
                limit=args.catalog_limit,
                disk_type=options.boot_disk_type,
                disk_size_gb=options.boot_disk_size_gb,
                is_spot=args.spot,
                machine_type_name=options.machine_type,
                gpu_type=options.gpu_type if options.enable_gpu else None,
                gpu_count=options.gpu_count if options.enable_gpu else 0,
            )
        return 0
    except (RuntimeError, gcp_exceptions.GoogleAPICallError) as exc:
        print(f"Erro ao executar ciclo de instância GCP: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())