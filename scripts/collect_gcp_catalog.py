"""Cria, para e exclui uma instância GCP para validar integração com Compute API."""

from __future__ import annotations

from typing import Any

from google.api_core import exceptions as gcp_exceptions
from google.cloud import compute_v1

from config.config import settings


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


def main() -> int:
    try:
        cfg = _load_config()
        client = compute_v1.InstancesClient()
        _create_if_missing(client, cfg)
        _print_instances(client, cfg)
        _stop_if_exists(client, cfg)
        _delete_if_exists(client, cfg)
        return 0
    except (RuntimeError, gcp_exceptions.GoogleAPICallError) as exc:
        print(f"Erro ao executar ciclo de instância GCP: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())