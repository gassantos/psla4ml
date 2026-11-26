"""
Resource Discovery — Preenchimento de parâmetros de recurso para o MILP PSLA4ML
=================================================================================

Preenche os parâmetros de recurso do modelo de otimização CC-IP
(Yuri Frota, 2026) — ``d_i`` (disco), ``m_i`` (memória), ``n_i^p``
(núcleos por tipo de processador), ``g_i^p`` (poder de processamento por
núcleo) e ``c_ij`` (custo de comunicação) — a partir de três fontes:

1. **Detecção/benchmark local**: para o recurso ``local``, hardware real
   da máquina de execução via ``psutil``/``torch.cuda`` (disco, RAM, GPU) e
   um benchmark real de multiplicação de matrizes para GFLOPS/núcleo de
   CPU. TPU é detectada via ``torch_xla`` quando instalado (extra ``tpu``).
2. **Catálogo estático curado**: para recursos de nuvem (``colab``, ``gcp``,
   ``aws``, ``azure``), especificações públicas de datasheet por modelo de
   GPU (casadas pelo campo ``gpu``) e vCPUs por tipo de instância (campo
   ``label``) — valores publicados pelos próprios provedores.
3. **Preço público de egress**: para ``c_ij``, tarifa de saída de dados por
   provedor (GCP/AWS/Azure), aplicada a um volume de referência quando os
   recursos pertencem a provedores diferentes; pares do mesmo provedor
   recebem custo zero (transferência intra-provedor assumida gratuita).

Por que não consultar as APIs de billing/catálogo ao vivo (GCP/AWS/Azure)?
    GCP (Cloud Billing Catalog) e AWS (Price List API) exigem credenciais
    de conta que não estão disponíveis neste ambiente. Uma integração ao
    vivo tornaria o pipeline de pesquisa dependente de rede e não
    determinístico entre execuções — inadequado para reprodutibilidade
    científica. O catálogo estático é versionado e auditável; pode ser
    substituído por uma consulta real quando credenciais estiverem
    disponíveis (ver ``fetch_live`` como ponto de extensão futuro).

Autor: Gustavo Alexandre
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from experiment.helpers import normalize_environment_label

logger = logging.getLogger(__name__)

DEFAULT_PROCESSOR_TYPES = ("CPU", "GPU", "TPU")

# Nome padronizado do artefato persistido pelo estágio de coleta de recursos.
RESOURCE_CATALOG_FILENAME_TEMPLATE = "resource_catalog_{device_type}_{date}.json"


# ============================================================================
# DETECÇÃO LOCAL DE HARDWARE
# ============================================================================

@dataclass(frozen=True)
class LocalHardwareSpec:
    """Especificação de hardware detectada na máquina de execução local."""

    cpu_logical_cores: int
    ram_total_gb: float
    disk_total_gb: float
    disk_free_gb: float
    gpu_name: str | None = None
    gpu_vram_gb: float | None = None
    gpu_multiprocessor_count: int | None = None
    cpu_gflops_per_core: float | None = None
    tpu_core_count: int = 0


def benchmark_cpu_gflops_per_core(matrix_size: int = 512, repeats: int = 3) -> float:
    """Mede o poder de processamento REAL da CPU local via multiplicação de matrizes.

    Executa ``repeats`` multiplicações de matrizes quadradas
    ``matrix_size x matrix_size`` (numpy/BLAS) e calcula o throughput médio
    em GFLOPS, dividido pelo número de núcleos físicos — uma medição real
    (não uma constante de catálogo). BLAS pode paralelizar entre núcleos;
    dividir pelo total de núcleos físicos dá uma estimativa por núcleo sob
    utilização plena.

    Returns:
        GFLOPS por núcleo medidos empiricamente nesta máquina.
    """
    import numpy as np

    a = np.random.rand(matrix_size, matrix_size)
    b = np.random.rand(matrix_size, matrix_size)
    np.dot(a, b)  # warm-up (alocação/cache)

    elapsed = 0.0
    for _ in range(repeats):
        start = time.perf_counter()
        np.dot(a, b)
        elapsed += time.perf_counter() - start
    avg_seconds = elapsed / repeats

    flops = 2 * (matrix_size ** 3)  # multiplicação de matrizes n×n: 2n³ FLOPs
    gflops_total = (flops / avg_seconds) / 1e9
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    return gflops_total / cores


def detect_local_tpu() -> int:
    """Detecta núcleos de TPU realmente disponíveis via ``torch_xla``.

    Retorna ``0`` quando ``torch_xla`` não está instalado (extra opcional
    ``tpu`` do projeto) ou nenhuma TPU está disponível — nunca fabrica um
    valor quando a detecção real não é possível.

    Returns:
        Número de núcleos de TPU detectados (0 se ausente).
    """
    try:
        import torch_xla.runtime as xr  # type: ignore

        world_size = xr.world_size()
        return int(world_size) if world_size else 0
    except Exception as exc:  # pragma: no cover - defensivo, sem torch_xla instalado
        logger.debug("Detecção de TPU local indisponível: %s", exc)
        return 0


def detect_local_hardware(disk_path: str = ".") -> LocalHardwareSpec:
    """Detecta as especificações reais de hardware da máquina local.

    Usa ``psutil`` para CPU/RAM/disco (sem dependências novas), ``torch.cuda``
    para GPU e um benchmark real de multiplicação de matrizes para GFLOPS/
    núcleo de CPU. Detecta TPU via ``torch_xla`` quando disponível. Nunca
    acessa a rede.

    Args:
        disk_path: Caminho usado para medir o disco (padrão: diretório atual).

    Returns:
        ``LocalHardwareSpec`` preenchido com os valores detectados/medidos.
        Campos de GPU ficam ``None`` quando CUDA não está disponível;
        ``cpu_gflops_per_core`` fica ``None`` se o benchmark falhar;
        ``tpu_core_count`` fica ``0`` quando não há TPU disponível.
    """
    disk = shutil.disk_usage(disk_path)
    spec = {
        "cpu_logical_cores": psutil.cpu_count(logical=True) or 1,
        "ram_total_gb": psutil.virtual_memory().total / (1024 ** 3),
        "disk_total_gb": disk.total / (1024 ** 3),
        "disk_free_gb": disk.free / (1024 ** 3),
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpu_multiprocessor_count": None,
        "cpu_gflops_per_core": None,
        "tpu_core_count": 0,
    }

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            spec["gpu_name"] = props.name
            spec["gpu_vram_gb"] = props.total_memory / (1024 ** 3)
            spec["gpu_multiprocessor_count"] = props.multi_processor_count
    except Exception as exc:  # pragma: no cover - defensivo, ambiente sem CUDA
        logger.debug("Detecção de GPU local indisponível: %s", exc)

    try:
        spec["cpu_gflops_per_core"] = benchmark_cpu_gflops_per_core()
    except Exception as exc:  # pragma: no cover - defensivo, numpy indisponível
        logger.debug("Benchmark de CPU local indisponível: %s", exc)

    spec["tpu_core_count"] = detect_local_tpu()

    return LocalHardwareSpec(**spec)


# ============================================================================
# CATÁLOGO ESTÁTICO DE GPUs (especificações públicas de datasheet)
# ============================================================================

# Valores de referência pública (datasheets NVIDIA, FP16/Tensor peak TFLOPS
# e contagem de Streaming Multiprocessors). Usados como estimativa quando
# não há benchmark local disponível — não substituem medição real.
GPU_CATALOG: dict[str, dict[str, float]] = {
    "NVIDIA RTX 3090":  {"fp16_tflops": 71.0,  "sm_count": 82, "vram_gb": 24.0},
    "NVIDIA T4":        {"fp16_tflops": 65.0,  "sm_count": 40, "vram_gb": 16.0},
    "NVIDIA V100":      {"fp16_tflops": 125.0, "sm_count": 80, "vram_gb": 16.0},
    "NVIDIA A100":      {"fp16_tflops": 312.0, "sm_count": 108, "vram_gb": 40.0},
    "NVIDIA RTX 2050":  {"fp16_tflops": 22.0,  "sm_count": 16, "vram_gb": 4.0},
}

# Disco padrão assumido para VMs de nuvem quando não declarado explicitamente
# em ``environments.details`` (não obtido de API ao vivo — valor configurável).
DEFAULT_CLOUD_DISK_GB = 100.0

# vCPUs por instância de nuvem, inferidos do texto de `label` em
# environments.details — valores publicados pelos próprios provedores
# (GCP n1-standard-4 = 4 vCPUs, AWS g4dn.xlarge = 4 vCPUs, Azure NC6s v3 =
# 6 vCPUs, Colab free tier = 2 vCPUs). Fallback: DEFAULT_CLOUD_CPU_CORES.
CLOUD_CPU_CORES_BY_LABEL_HINT: dict[str, int] = {
    "n1-standard-4": 4,
    "g4dn.xlarge": 4,
    "nc6s v3": 6,
    "colab": 2,
}
DEFAULT_CLOUD_CPU_CORES = 4


def _estimate_cloud_cpu_cores(label: str | None) -> int:
    """Estima vCPUs de um recurso de nuvem a partir do texto de `label` (catálogo do provedor)."""
    if not label:
        return DEFAULT_CLOUD_CPU_CORES
    normalized = label.lower()
    for hint, cores in CLOUD_CPU_CORES_BY_LABEL_HINT.items():
        if hint in normalized:
            return cores
    return DEFAULT_CLOUD_CPU_CORES


# ============================================================================
# CATÁLOGO ESTÁTICO DE TPUs (especificações públicas Google)
# ============================================================================

# Peak bf16 TFLOPS por chip e núcleos por chip (especificações públicas
# documentadas pelo Google Cloud). Só é aplicado quando o recurso declara
# explicitamente ``tpu_version`` (ex.: "v3") em environments.details, ou
# quando detectado localmente via torch_xla — nunca inventado por padrão.
TPU_CATALOG: dict[str, dict[str, float]] = {
    "v2": {"bf16_tflops_per_chip": 45.0, "cores_per_chip": 2},
    "v3": {"bf16_tflops_per_chip": 123.0, "cores_per_chip": 2},
    "v4": {"bf16_tflops_per_chip": 275.0, "cores_per_chip": 2},
}


def _match_gpu_catalog(gpu_name: str | None) -> dict[str, float] | None:
    """Casa o nome de GPU declarado/detectado com uma entrada do catálogo estático.

    Tolera variantes de nome (ex.: ``torch`` reporta "NVIDIA GeForce RTX 2050",
    enquanto o catálogo usa a chave "NVIDIA RTX 2050") via correspondência por
    substring do modelo, após remover o prefixo "NVIDIA"/"GeForce".
    """
    if not gpu_name:
        return None
    if gpu_name in GPU_CATALOG:
        return GPU_CATALOG[gpu_name]
    normalized = gpu_name.strip().lower()
    for name, spec in GPU_CATALOG.items():
        if name.lower() == normalized:
            return spec
    for name, spec in GPU_CATALOG.items():
        model = name.lower().replace("nvidia", "").strip()
        if model and model in normalized.replace("geforce", "").strip():
            return spec
    return None


# ============================================================================
# MONTAGEM DO CATÁLOGO DE RECURSOS (parâmetros do MILP por recurso i ∈ R)
# ============================================================================

def build_resource_spec(
    env_name: str,
    env_details: dict[str, Any],
    *,
    local_hardware: LocalHardwareSpec | None = None,
    processor_types: tuple[str, ...] = DEFAULT_PROCESSOR_TYPES,
) -> dict[str, Any]:
    """Monta a especificação de recurso ``i`` para o modelo MILP PSLA4ML.

    Args:
        env_name: Nome do ambiente (ex.: "local", "colab", "gcp").
        env_details: Entrada correspondente em ``environments.details``.
        local_hardware: Hardware detectado via :func:`detect_local_hardware`.
            Obrigatório apenas quando ``env_name == "local"``.
        processor_types: Tipos de processador ``P`` a preencher em ``n_i``/``g_i``.

    Returns:
        Dicionário com os parâmetros do modelo: ``c_i`` (custo/hora),
        ``m_i`` (memória em GB), ``d_i`` (disco em GB), ``n_i`` (núcleos
        por tipo de processador), ``g_i`` (GFLOPS por núcleo), e ``source``
        (rastreabilidade: "detected_local" | "static_catalog" | "default").
        ``g_i["CPU"]``/``n_i["TPU"]``/``g_i["TPU"]`` ficam ausentes quando
        não há medição/detecção/catálogo aplicável — nunca são fabricados.
    """
    c_i = float(env_details.get("cost_per_hour_usd", 0.0))
    gpu_name = env_details.get("gpu")
    label = env_details.get("label")
    tpu_version = env_details.get("tpu_version")

    if env_name == "local" and local_hardware is not None:
        m_i = local_hardware.gpu_vram_gb or local_hardware.ram_total_gb
        d_i = local_hardware.disk_free_gb
        sm_count = local_hardware.gpu_multiprocessor_count
        catalog = _match_gpu_catalog(local_hardware.gpu_name) or _match_gpu_catalog(gpu_name)
        g_i_gpu = (catalog["fp16_tflops"] * 1000 / catalog["sm_count"]) if catalog else None
        n_i_gpu = sm_count or (catalog["sm_count"] if catalog else None)
        cpu_cores = local_hardware.cpu_logical_cores
        cpu_gflops = local_hardware.cpu_gflops_per_core
        cpu_gflops_source = "benchmarked_local" if cpu_gflops is not None else None
        tpu_cores = local_hardware.tpu_core_count
        source = "detected_local"
    else:
        catalog = _match_gpu_catalog(gpu_name)
        m_i = float(env_details.get("vram_gb", catalog["vram_gb"] if catalog else 0.0))
        d_i = float(env_details.get("disk_gb", DEFAULT_CLOUD_DISK_GB))
        n_i_gpu = catalog["sm_count"] if catalog else None
        g_i_gpu = (catalog["fp16_tflops"] * 1000 / catalog["sm_count"]) if catalog else None
        cpu_cores = _estimate_cloud_cpu_cores(label)
        # Sem acesso remoto para medir a CPU da nuvem: usa o benchmark local
        # como proxy documentado (nao e medicao daquela maquina especifica).
        cpu_gflops = local_hardware.cpu_gflops_per_core if local_hardware else None
        cpu_gflops_source = "local_benchmark_proxy" if cpu_gflops is not None else None
        tpu_cores = 0  # sem deteccao remota de TPU de nuvem
        source = "static_catalog" if catalog else "default"

    n_i: dict[str, float] = {}
    g_i: dict[str, float] = {}
    for p in processor_types:
        if p == "GPU" and n_i_gpu is not None:
            n_i["GPU"] = n_i_gpu
            if g_i_gpu is not None:
                g_i["GPU"] = g_i_gpu
        elif p == "CPU":
            n_i["CPU"] = cpu_cores
            if cpu_gflops is not None:
                g_i["CPU"] = cpu_gflops
        elif p == "TPU":
            tpu_spec = TPU_CATALOG.get(tpu_version) if tpu_version else None
            if tpu_cores > 0:
                n_i["TPU"] = tpu_cores
                if tpu_spec:
                    g_i["TPU"] = tpu_spec["bf16_tflops_per_chip"] * 1000 / tpu_spec["cores_per_chip"]
            elif tpu_spec:
                # TPU declarada explicitamente na config (ex.: nuvem), sem deteccao local
                n_i["TPU"] = tpu_spec["cores_per_chip"]
                g_i["TPU"] = tpu_spec["bf16_tflops_per_chip"] * 1000 / tpu_spec["cores_per_chip"]
            # senao: TPU ausente de n_i/g_i (nao fabricada como 0)

    return {
        "resource_id": env_name,
        "c_i": c_i,
        "m_i": m_i,
        "d_i": d_i,
        "n_i": n_i,
        "g_i": g_i,
        "gpu_name": gpu_name,
        "cpu_gflops_source": cpu_gflops_source,
        "source": source,
    }


def estimate_energy_rate_kwh_per_hour(
    results: list[dict[str, Any]],
    environment: str,
) -> float | None:
    """Estima ``e_i`` (consumo energético por hora) empiricamente a partir do histórico.

    Usa a média de ``energy_kwh / (train_time_sec / 3600)`` entre os
    experimentos concluídos com sucesso no ambiente informado. Mais preciso
    que uma estimativa de catálogo, pois reflete o consumo real medido pelo
    CodeCarbon durante execuções passadas.

    Args:
        results: Lista de resultados de grid search (``status``,
            ``selected_environment`` ou ``grid_params.environment``,
            ``resources``).
        environment: Nome do ambiente a filtrar.

    Returns:
        Taxa média em kWh/hora, ou ``None`` se não houver amostras válidas.
    """
    rates = []
    for r in results:
        if r.get("status") != "success":
            continue
        env = (
            r.get("configured_environment")
            or r.get("grid_params", {}).get("environment")
            or r.get("selected_environment")
        )
        if normalize_environment_label(str(env)) != normalize_environment_label(str(environment)):
            continue
        resources = r.get("resources", {})
        try:
            energy_kwh = float(resources.get("energy_kwh"))
            train_time_hours = float(resources.get("train_time_sec")) / 3600.0
        except (TypeError, ValueError):
            continue
        if train_time_hours <= 0:
            continue
        rates.append(energy_kwh / train_time_hours)

    if not rates:
        return None
    return sum(rates) / len(rates)


def build_resource_catalog(
    environments_details: dict[str, dict[str, Any]],
    *,
    results: list[dict[str, Any]] | None = None,
    local_hardware: LocalHardwareSpec | None = None,
    processor_types: tuple[str, ...] = DEFAULT_PROCESSOR_TYPES,
) -> dict[str, dict[str, Any]]:
    """Monta o catálogo completo de recursos ``R`` para o modelo MILP PSLA4ML.

    Args:
        environments_details: ``environments.details`` do grid config
            multiambiente (ex.: ``grid_search_multienv.json``).
        results: Histórico de resultados de grid search, usado para estimar
            ``e_i`` empiricamente via :func:`estimate_energy_rate_kwh_per_hour`.
            ``None`` deixa ``e_i`` como ``None`` (não estimável sem histórico).
        local_hardware: Hardware local pré-detectado. ``None`` chama
            :func:`detect_local_hardware` automaticamente uma única vez —
            usado tanto para o recurso ``"local"`` quanto como proxy de
            ``g_i["CPU"]`` para recursos de nuvem (sem acesso remoto para
            benchmark real nesses casos).
        processor_types: Tipos de processador ``P`` a preencher em ``n_i``/``g_i``.

    Returns:
        Dicionário ``{nome_ambiente: especificação_de_recurso}``.
    """
    hw = local_hardware if local_hardware is not None else detect_local_hardware()
    catalog: dict[str, dict[str, Any]] = {}
    for env_name, env_details in environments_details.items():
        if not isinstance(env_details, dict):
            continue
        spec = build_resource_spec(
            env_name, env_details, local_hardware=hw, processor_types=processor_types,
        )
        if results is not None:
            spec["e_i"] = estimate_energy_rate_kwh_per_hour(results, env_name)
        else:
            spec["e_i"] = None
        catalog[env_name] = spec
    return catalog


def persist_resource_catalog(
    catalog: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Persiste um catálogo de recursos (ou estrutura combinada) já construído como JSON.

    Args:
        catalog: Saída de :func:`build_resource_catalog`, ou a estrutura
            combinada ``{"resources": ..., "communication_costs": ...}``
            montada por :func:`collect_and_persist_resource_catalog`.
        output_path: Caminho de destino do arquivo JSON.

    Returns:
        O ``Path`` do arquivo escrito.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    return output_path


# ============================================================================
# CUSTO DE COMUNICAÇÃO ENTRE RECURSOS (c_ij) — egress público por provedor
# ============================================================================

# Preço público de saída de dados (egress) para a internet, USD/GB —
# valores de referência publicados pelos provedores (tarifas variam por
# região/volume; usar como estimativa, não substitui o price sheet oficial).
EGRESS_COST_USD_PER_GB: dict[str, float] = {
    "gcp": 0.12,
    "aws": 0.09,
    "azure": 0.087,
    "local": 0.0,
}
DEFAULT_EGRESS_COST_USD_PER_GB = 0.10

# Provedor de infraestrutura por nome de recurso, usado quando
# environments.details.<env> não declara "provider" explicitamente.
# Colab roda sobre infraestrutura GCP.
PROVIDER_BY_ENV_NAME_HINT: dict[str, str] = {
    "local": "local",
    "colab": "gcp",
    "gcp": "gcp",
    "aws": "aws",
    "azure": "azure",
}

# Volume de referência assumido por transferência entre recursos (ex.:
# tamanho aproximado de um checkpoint BERT-base). Documentado, não medido —
# ajustável via parâmetro ``reference_transfer_gb``.
REFERENCE_TRANSFER_GB = 0.44


def _infer_provider(env_name: str, env_details: dict[str, Any] | None = None) -> str:
    """Infere o provedor de infraestrutura de um recurso.

    Prioriza ``env_details["provider"]`` quando declarado explicitamente;
    senão usa uma tabela de referência por nome de ambiente.
    """
    if env_details and env_details.get("provider"):
        return str(env_details["provider"])
    return PROVIDER_BY_ENV_NAME_HINT.get(env_name, "unknown")


def communication_cost_matrix(
    resource_ids: list[str],
    *,
    environments_details: dict[str, dict[str, Any]] | None = None,
    reference_transfer_gb: float = REFERENCE_TRANSFER_GB,
) -> dict[tuple[str, str], float]:
    """Monta a matriz ``c_ij`` de custo de comunicação entre recursos.

    Estima o custo de transferir um volume de referência de dados
    (``reference_transfer_gb``) entre cada par de recursos, usando o preço
    público de egress do provedor mais caro envolvido no par. Pares do
    mesmo provedor (ex.: dois recursos GCP) recebem custo ``0.0``
    (transferência intra-provedor assumida gratuita/negligenciável).

    Args:
        resource_ids: Lista de identificadores de recursos (``R``).
        environments_details: ``environments.details``, usado para checar
            um campo ``provider`` explícito por recurso (tem prioridade
            sobre a tabela de nomes conhecidos). ``None`` usa apenas a
            tabela de nomes.
        reference_transfer_gb: Volume de dados assumido por transferência.

    Returns:
        Dicionário ``{(i, j): custo_usd}`` para todos os pares ``i < j``.
    """
    pairs: dict[tuple[str, str], float] = {}
    for idx, i in enumerate(resource_ids):
        provider_i = _infer_provider(i, (environments_details or {}).get(i))
        for j in resource_ids[idx + 1:]:
            provider_j = _infer_provider(j, (environments_details or {}).get(j))
            if provider_i == provider_j and provider_i != "unknown":
                cost = 0.0
            else:
                rate = max(
                    EGRESS_COST_USD_PER_GB.get(provider_i, DEFAULT_EGRESS_COST_USD_PER_GB),
                    EGRESS_COST_USD_PER_GB.get(provider_j, DEFAULT_EGRESS_COST_USD_PER_GB),
                )
                cost = rate * reference_transfer_gb
            pairs[(i, j)] = cost
    return pairs


def collect_and_persist_resource_catalog(
    environments_details: dict[str, dict[str, Any]],
    output_path: str | Path,
    *,
    results: list[dict[str, Any]] | None = None,
    local_hardware: LocalHardwareSpec | None = None,
    processor_types: tuple[str, ...] = DEFAULT_PROCESSOR_TYPES,
    reference_transfer_gb: float = REFERENCE_TRANSFER_GB,
) -> Path:
    """Executa o estágio de coleta de recursos e persiste o resultado em JSON.

    Pensado para rodar uma única vez, no início de uma execução de grid
    search (``gridsearch.executor.run_grid_search``), coletando por detecção/
    benchmark local e catálogo estático de provedores de nuvem todos os
    parâmetros de recurso ``c_i``/``d_i``/``m_i``/``n_i^p``/``g_i^p``/``e_i``
    e a matriz de custo de comunicação ``c_ij`` necessários ao modelo MILP
    PSLA4ML, antes de qualquer experimento ser executado.

    O arquivo persistido tem o formato::

        {
          "resources": {"<env>": {...}, ...},
          "communication_costs": [
            {"resource_i": ..., "resource_j": ..., "cost_usd": ...}, ...
          ]
        }

    Returns:
        O ``Path`` do arquivo JSON persistido.
    """
    resources = build_resource_catalog(
        environments_details, results=results, local_hardware=local_hardware,
        processor_types=processor_types,
    )
    comm_costs = communication_cost_matrix(
        list(environments_details.keys()),
        environments_details=environments_details,
        reference_transfer_gb=reference_transfer_gb,
    )
    combined = {
        "resources": resources,
        "communication_costs": [
            {"resource_i": i, "resource_j": j, "cost_usd": cost}
            for (i, j), cost in comm_costs.items()
        ],
    }
    return persist_resource_catalog(combined, output_path)

