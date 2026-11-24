"""Cria, para e exclui uma instância GCP para validar integração com Compute API."""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        if args.safe and args.action == "cycle":
            raise RuntimeError("Modo seguro requer --action explícita (list/create/stop/delete).")
        cfg = _load_config()
        client = compute_v1.InstancesClient()
        _run_action(client, cfg, args.action, confirm_delete=args.confirm_delete)
        return 0
    except (RuntimeError, gcp_exceptions.GoogleAPICallError) as exc:
        print(f"Erro ao executar ciclo de instância GCP: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())