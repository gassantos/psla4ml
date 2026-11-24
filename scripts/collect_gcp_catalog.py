"""Cria, para e exclui uma instância GCP para validar integração com Compute API."""

from __future__ import annotations

import argparse
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


def _build_instance_payload(cfg: dict[str, str]) -> compute_v1.Instance:
    disk = compute_v1.AttachedDisk(
        boot=True,
        auto_delete=True,
        initialize_params=compute_v1.AttachedDiskInitializeParams(
            source_image=cfg["source_image"],
            disk_size_gb=10,
        ),
    )
    network_interface = compute_v1.NetworkInterface(
        network=cfg["network"],
        access_configs=[
            compute_v1.AccessConfig(name=cfg["nat_name"], type_=cfg["nat_type"]),
        ],
    )
    return compute_v1.Instance(
        name=cfg["instance"],
        machine_type=cfg["machine_type"],
        disks=[disk],
        network_interfaces=[network_interface],
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


def _create_if_missing(client: compute_v1.InstancesClient, cfg: dict[str, str]) -> None:
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
        instance_resource=_build_instance_payload(cfg),
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
    return parser


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
    action: str,
    *,
    confirm_delete: bool,
) -> None:
    if action == "list":
        _print_instances(client, cfg)
        return
    if action == "create":
        _create_if_missing(client, cfg)
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
        _create_if_missing(client, cfg)
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


def _print_machine_type_details(cfg: dict[str, str]) -> compute_v1.MachineType:
    machine_type_name = _machine_type_name(cfg["machine_type"])
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

        if None not in (price_core, price_ram, price_disk_gb_month):
            break

    if None in (price_core, price_ram, price_disk_gb_month):
        raise ValueError(
            "SKUs incompletos para cálculo de custo "
            f"(region={region}, core={price_core}, ram={price_ram}, disco={price_disk_gb_month})."
        )

    cost_compute_hour = (price_core * billable_vcpus) + (price_ram * ram_gb)
    cost_compute_month = cost_compute_hour * 730

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
        "cost_disk_hour_usd": cost_disk_hour,
        "cost_total_hour_running_usd": cost_compute_hour + cost_disk_hour,
        "cost_total_month_running_usd": cost_compute_month + cost_disk_month,
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
) -> None:
    result = _estimate_instance_cost_e2(
        machine_type_name=machine_type_name,
        vcpus_api=vcpus_api,
        ram_gb=ram_gb,
        zone=cfg["zone"],
        disk_type=disk_type,
        disk_size_gb=disk_size_gb,
        is_spot=is_spot,
    )

    print("=== Estimativa de Custo da VM (Catalog API) ===")
    print(f"Instância: {result['machine_type']} ({result['region']})")
    print(f"vCPUs faturáveis: {result['billable_vcpus']:.2f}")
    print(f"RAM: {result['ram_gb']:.2f} GB")
    print(f"Disco: {int(result['disk_size_gb'])} GB ({result['disk_type']})")
    print(f"Custo compute/hora: ${result['cost_compute_hour_usd']:.6f} USD")
    print(f"Custo disco/hora:   ${result['cost_disk_hour_usd']:.6f} USD")
    print(f"Custo total/hora (RUNNING): ${result['cost_total_hour_running_usd']:.6f} USD")
    print(f"Custo total/mês (RUNNING):  ${result['cost_total_month_running_usd']:.2f} USD")
    print(f"Custo/mês (TERMINATED):     ${result['cost_month_terminated_usd']:.2f} USD")


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
) -> None:
    machine_type_info = _print_machine_type_details(cfg)
    _print_billing_catalog(cfg, limit)
    try:
        _print_cost_estimate(
            cfg,
            machine_type_name=_machine_type_name(cfg["machine_type"]),
            vcpus_api=machine_type_info.guest_cpus,
            ram_gb=machine_type_info.memory_mb / 1024,
            disk_type=disk_type,
            disk_size_gb=disk_size_gb,
            is_spot=is_spot,
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
        client = compute_v1.InstancesClient()
        _run_action(client, cfg, args.action, confirm_delete=args.confirm_delete)
        if args.fetch_catalog:
            _collect_gcp_catalog(
                cfg,
                limit=args.catalog_limit,
                disk_type=args.disk_type,
                disk_size_gb=args.disk_size_gb,
                is_spot=args.spot,
            )
        return 0
    except (RuntimeError, gcp_exceptions.GoogleAPICallError) as exc:
        print(f"Erro ao executar ciclo de instância GCP: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())