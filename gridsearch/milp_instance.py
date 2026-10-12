""" MILP Instance Reader — Monta ``PSLA4MLData`` a partir de arquivos persistidos
==================================================================================

Lê os arquivos JSON já persistidos pelo pipeline ``gridsearch`` — o config
multiambiente (``environments.details``, ex.: ``grid_search_multienv.json``),
o estado/resultados de uma execução (``grid_search_state_*.json``, produzido
por ``gridsearch.executor.save_state``) e um perfil de ``sla_profiles.json``
— e monta a instância de entrada do modelo MILP CC-IP (PSLA4ML, Yuri Frota,
2026), no formato ``PSLA4MLData``.

Este módulo **apenas lê e mapeia** dados já persistidos; não detecta
hardware, não consulta catálogos externos e não estima valores por
heurística. Campos do modelo sem correspondente já persistido em algum dos
arquivos (``d_i``, ``n_i^p``, ``g_i^p`` por recurso) ficam ausentes do
dicionário resultante — cabe a quem mantém ``environments.details``/
``sla_profiles.json`` declará-los explicitamente quando disponíveis.

Autor: Gustavo Alexandre
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiment.helpers import normalize_environment_label

logger = logging.getLogger(__name__)

DEFAULT_PROCESSOR_TYPES = ("CPU", "GPU", "TPU")

# Fallbacks aplicados apenas quando uma constraint do perfil de SLA é `null`
# ou o perfil não é informado. Documentados e sobrescrevíveis pelo chamador.
DEFAULT_COST_USD = 1000.0
DEFAULT_ENERGY_KWH = 500.0
DEFAULT_PEAK_RAM_GB = 64.0
DEFAULT_TRAIN_TIME_HOURS = 24.0
DEFAULT_DISK_GB = 20.0


@dataclass
class PSLA4MLData:
    """Instância de entrada do modelo CC-IP (PSLA4ML), pronta para Gurobi.

    Espelha exatamente os campos usados pela implementação de referência do
    modelo — ver Tabela 1 do documento "Modelo de Otimização PSLA4ML"
    (Yuri Frota, 2026):

        R       — conjunto de recursos (ambientes computacionais).
        P       — conjunto de tipos de processador ({"CPU", "GPU", "TPU"}).
        NM      — nº máximo de recursos alocáveis por período de tempo.
        CM      — orçamento financeiro máximo do usuário.
        TM      — nº de períodos de tempo (T = {1, ..., TM}).
        Eref    — consumo energético de referência (teto).
        DS      — espaço em disco requerido pelo usuário.
        MC      — memória requerida pelo usuário.
        Gf      — trabalho computacional total requerido (GFLOPS).
        alpha1/2/3 — pesos do objetivo (custo/tempo/energia); soma = 1.
        c       — {recurso: custo por período}.
        d       — {recurso: disco disponível} (apenas recursos com o dado persistido).
        m       — {recurso: memória disponível}.
        e       — {recurso: consumo energético por período} (apenas recursos com histórico).
        n       — {(recurso, processador): nº de núcleos} (apenas pares com o dado persistido).
        g       — {(recurso, processador): poder de processamento por núcleo} (idem).
        c_comm  — {(recurso_i, recurso_j): custo médio de comunicação}, i < j.
    """

    R: list[str]
    P: list[str]
    NM: int
    CM: float
    TM: int
    Eref: float
    DS: float
    MC: float
    Gf: float
    alpha1: float
    alpha2: float
    alpha3: float
    c: dict[str, float] = field(default_factory=dict)
    d: dict[str, float] = field(default_factory=dict)
    m: dict[str, float] = field(default_factory=dict)
    e: dict[str, float] = field(default_factory=dict)
    n: dict[tuple[str, str], int] = field(default_factory=dict)
    g: dict[tuple[str, str], float] = field(default_factory=dict)
    c_comm: dict[tuple[str, str], float] = field(default_factory=dict)


# ============================================================================
# LEITURA DOS ARQUIVOS PERSISTIDOS
# ============================================================================

def read_environments_details(grid_config_path: str | Path) -> dict[str, dict[str, Any]]:
    """Lê ``environments.details`` de um grid config JSON já persistido.

    Args:
        grid_config_path: Caminho do grid config (ex.:
            ``gridsearch/config/grid_search_multienv.json``).

    Returns:
        Dicionário ``{nome_ambiente: detalhes}`` tal como persistido no
        arquivo, sem nenhuma transformação.

    Raises:
        ValueError: Se o arquivo não tiver a chave ``environments.details``.
    """
    with open(grid_config_path, encoding="utf-8") as f:
        grid_config = json.load(f)
    details = grid_config.get("environments", {}).get("details")
    if not details:
        raise ValueError(
            f"{grid_config_path} não possui 'environments.details' — "
            "use um grid config multiambiente (ex.: grid_search_multienv.json)."
        )
    return details


def read_grid_search_state(state_path: str | Path) -> dict[str, Any]:
    """Lê um arquivo ``grid_search_state_*.json`` persistido por ``save_state``.

    Args:
        state_path: Caminho do arquivo de estado.

    Returns:
        Dicionário completo do estado (``timestamp``, ``completed_experiments``,
        ``results``, ``sla_prefilter``), tal como persistido.
    """
    with open(state_path, encoding="utf-8") as f:
        return json.load(f)


def read_resource_catalog(resource_catalog_path: str | Path) -> dict[str, dict[str, Any]]:
    """Lê um ``resource_catalog_*.json`` persistido por
    ``gridsearch.resource_discovery.collect_and_persist_resource_catalog``
    (estágio inicial de coleta do ``gridsearch.executor.run_grid_search``).

    Args:
        resource_catalog_path: Caminho do arquivo de catálogo de recursos.

    Returns:
        Dicionário ``{nome_ambiente: especificação_de_recurso}`` tal como
        persistido (``c_i``, ``d_i``, ``m_i``, ``n_i``, ``g_i``, ``e_i``).
    """
    with open(resource_catalog_path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# EXTRAÇÃO DE PARÂMETROS JÁ PERSISTIDOS (sem detecção/catálogo/heurística)
# ============================================================================

def _energy_rate_from_results(
    results: list[dict[str, Any]],
    environment: str,
) -> float | None:
    """Deriva ``e_i`` (energia/hora) da média aritmética dos resultados já persistidos.

    Não estima nem consulta catálogo — apenas agrega números já presentes em
    ``resources.energy_kwh``/``resources.train_time_sec`` dos resultados do
    ambiente informado. Retorna ``None`` se não houver amostras válidas.
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


def _find_result_by_idx(
    results: list[dict[str, Any]],
    grid_experiment_idx: int,
) -> dict[str, Any] | None:
    """Localiza, nos resultados já persistidos, o experimento de índice informado."""
    for r in results:
        if r.get("grid_experiment_idx") == grid_experiment_idx:
            return r
    return None


def _extract_alpha_weights(
    sla_profile: dict[str, Any] | None,
) -> tuple[float, float, float]:
    """Deriva (alpha1, alpha2, alpha3) — pesos custo/tempo/energia — de um perfil de SLA.

    Busca ``cost_usd``, ``train_time_sec`` e ``energy_kwh`` em
    ``profile["metrics"]``/``profile["weights_scalar"]`` por nome (não por
    posição, pois a ordem varia entre perfis) e normaliza para somar 1,
    conforme exigido pelo modelo (``alpha1 + alpha2 + alpha3 = 1``).
    Sem perfil, ou sem nenhuma das três métricas presentes, cai para pesos
    iguais (1/3 cada).
    """
    if sla_profile is None:
        return (1 / 3, 1 / 3, 1 / 3)

    metrics = sla_profile.get("metrics", [])
    weights = sla_profile.get("weights_scalar", [])
    metric_weight = dict(zip(metrics, weights))

    raw = (
        metric_weight.get("cost_usd", 0.0),
        metric_weight.get("train_time_sec", 0.0),
        metric_weight.get("energy_kwh", 0.0),
    )
    total = sum(raw)
    if total <= 0:
        logger.warning(
            "Perfil de SLA sem pesos para cost_usd/train_time_sec/energy_kwh; "
            "usando pesos iguais (1/3 cada) para alpha1/alpha2/alpha3."
        )
        return (1 / 3, 1 / 3, 1 / 3)
    return tuple(w / total for w in raw)  # type: ignore[return-value]


# ============================================================================
# MONTAGEM DA INSTÂNCIA (a partir de dados já carregados em memória)
# ============================================================================

def build_psla4ml_data(
    environments_details: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    sla_profile: dict[str, Any] | None = None,
    target_grid_experiment_idx: int | None = None,
    target_result: dict[str, Any] | None = None,
    resource_catalog: dict[str, Any] | None = None,
    processor_types: tuple[str, ...] = DEFAULT_PROCESSOR_TYPES,
    NM: int | None = None,
    DS: float | None = None,
    Gf: float | None = None,
    time_period_hours: float = 1.0,
) -> PSLA4MLData:
    """Monta uma :class:`PSLA4MLData` apenas a partir de dados já persistidos.

    Não detecta hardware, não consulta catálogos e não estima valores por
    heurística — todo campo vem de ``environments_details`` (config),
    ``results`` (estado de execução persistido), ``resource_catalog``
    (estágio de coleta já persistido por
    ``resource_discovery.collect_and_persist_resource_catalog``) ou
    ``sla_profile``.

    Args:
        environments_details: ``environments.details`` já lido de um grid
            config (via :func:`read_environments_details`) — define ``R``.
        results: Lista ``results`` já lida de um ``grid_search_state_*.json``
            (via :func:`read_grid_search_state`) — usada para ``e`` e ``Gf``.
        sla_profile: Perfil de ``sla_profiles.json`` (dict de um profile).
            Fornece ``CM``/``TM``/``Eref``/``MC``/``DS`` via ``constraints``
            e ``alpha1``/``alpha2``/``alpha3`` via ``metrics``/``weights_scalar``.
        target_grid_experiment_idx: Índice do experimento (em ``results``)
            usado para extrair ``Gf`` de ``resources.total_gflops``.
        target_result: Alternativa a ``target_grid_experiment_idx`` — um
            registro de resultado já selecionado externamente (ex.: pelo
            Skyband), usado apenas para ``Gf`` quando informado.
        resource_catalog: Catálogo persistido (via
            :func:`read_resource_catalog`) pelo estágio de coleta do
            ``gridsearch.executor.run_grid_search``: dict no formato
            ``{"resources": {env: spec}, "communication_costs": [...]}``.
            Preenche ``d``/``n``/``g`` (e ``c``/``m``/``e`` como fallback)
            quando ``environments_details`` não os declara explicitamente, e
            preenche ``c_comm`` com os custos de comunicação reais
            (egress público) em vez do zero padrão. ``None`` mantém esses
            campos ausentes/zerados quando não persistidos.
        processor_types: Conjunto ``P``. Padrão ``("CPU", "GPU", "TPU")``.
        NM: Nº máximo de recursos alocáveis por período. ``None`` usa ``len(R)``.
        DS: Disco requerido pelo usuário. ``None`` lê ``disk_gb`` do perfil
            de SLA; se também ausente, cai no fallback documentado.
        Gf: Sobrescreve o ``total_gflops`` do resultado-alvo.
        time_period_hours: Duração de cada período ``t ∈ T`` em horas.

    Returns:
        Instância ``PSLA4MLData``. Os dicionários ``d``, ``n`` e ``g`` só
        incluem entradas para os campos efetivamente presentes em
        ``environments_details``/``resource_catalog`` — nenhum valor é
        inventado neste módulo.

    Raises:
        ValueError: Se ``environments_details`` estiver vazio, ou se ``Gf``
            não puder ser determinado.
    """
    if not environments_details:
        raise ValueError("environments_details não pode ser vazio (R ficaria vazio).")

    R = list(environments_details.keys())
    P = list(processor_types)

    resources_catalog_map = (resource_catalog or {}).get("resources", {})

    c: dict[str, float] = {}
    d: dict[str, float] = {}
    m: dict[str, float] = {}
    e: dict[str, float] = {}
    n: dict[tuple[str, str], int] = {}
    g: dict[tuple[str, str], float] = {}

    for r in R:
        details = environments_details[r]
        cat = resources_catalog_map.get(r, {})

        if "cost_per_hour_usd" in details:
            c[r] = float(details["cost_per_hour_usd"])
        elif cat.get("c_i") is not None:
            c[r] = float(cat["c_i"])

        if "vram_gb" in details:
            m[r] = float(details["vram_gb"])
        elif cat.get("m_i") is not None:
            m[r] = float(cat["m_i"])

        if "disk_gb" in details:
            d[r] = float(details["disk_gb"])
        elif cat.get("d_i") is not None:
            d[r] = float(cat["d_i"])

        rate = _energy_rate_from_results(results, r)
        if rate is not None:
            e[r] = rate
        elif cat.get("e_i") is not None:
            e[r] = float(cat["e_i"])

        cores_by_processor = details.get("cores_by_processor", {})
        gflops_by_processor = details.get("gflops_per_core", {})
        cat_n = cat.get("n_i", {})
        cat_g = cat.get("g_i", {})
        for p in P:
            if p in cores_by_processor:
                n[(r, p)] = int(cores_by_processor[p])
            elif p in cat_n:
                n[(r, p)] = int(cat_n[p])
            if p in gflops_by_processor:
                g[(r, p)] = float(gflops_by_processor[p])
            elif p in cat_g:
                g[(r, p)] = float(cat_g[p])

    persisted_comm_costs: list[dict[str, Any]] | None = (resource_catalog or {}).get("communication_costs")
    if persisted_comm_costs:
        c_comm = {
            (rec["resource_i"], rec["resource_j"]): float(rec["cost_usd"])
            for rec in persisted_comm_costs
            if rec["resource_i"] in R and rec["resource_j"] in R
        }
    else:
        c_comm = {
            (i, j): 0.0
            for idx, i in enumerate(R)
            for j in R[idx + 1:]
        }

    constraints = (sla_profile or {}).get("constraints", {})
    cost_usd = constraints.get("cost_usd")
    energy_kwh = constraints.get("energy_kwh")
    peak_ram_mb = constraints.get("peak_ram_mb")
    train_time_sec = constraints.get("train_time_sec")
    disk_gb = constraints.get("disk_gb")

    fallbacks_used: list[str] = []

    CM = float(cost_usd) if cost_usd is not None else DEFAULT_COST_USD
    if cost_usd is None:
        fallbacks_used.append(f"CM={DEFAULT_COST_USD} (cost_usd ausente no perfil)")

    Eref = float(energy_kwh) if energy_kwh is not None else DEFAULT_ENERGY_KWH
    if energy_kwh is None:
        fallbacks_used.append(f"Eref={DEFAULT_ENERGY_KWH} (energy_kwh ausente no perfil)")

    MC = float(peak_ram_mb) / 1024.0 if peak_ram_mb is not None else DEFAULT_PEAK_RAM_GB
    if peak_ram_mb is None:
        fallbacks_used.append(f"MC={DEFAULT_PEAK_RAM_GB} (peak_ram_mb ausente no perfil)")

    train_time_hours = (
        float(train_time_sec) / 3600.0 if train_time_sec is not None else DEFAULT_TRAIN_TIME_HOURS
    )
    if train_time_sec is None:
        fallbacks_used.append(f"TM baseado em {DEFAULT_TRAIN_TIME_HOURS}h (train_time_sec ausente no perfil)")
    TM = max(1, math.ceil(train_time_hours / time_period_hours))

    if DS is not None:
        DS_value = DS
    elif disk_gb is not None:
        DS_value = float(disk_gb)
    else:
        DS_value = DEFAULT_DISK_GB
        fallbacks_used.append(f"DS={DEFAULT_DISK_GB} (disk_gb ausente no perfil e não informado)")

    if NM is None:
        fallbacks_used.append(f"NM=len(R)={len(R)} (nenhum limite de concorrência informado)")

    if fallbacks_used:
        logger.info("Instância PSLA4ML usou fallbacks documentados: %s", "; ".join(fallbacks_used))

    alpha1, alpha2, alpha3 = _extract_alpha_weights(sla_profile)

    if Gf is None:
        record = target_result
        if record is None and target_grid_experiment_idx is not None:
            record = _find_result_by_idx(results, target_grid_experiment_idx)
        if record is not None:
            Gf = float(record.get("resources", {}).get("total_gflops", 0.0))
        else:
            raise ValueError(
                "Gf não pôde ser determinado: informe `Gf`, `target_result` ou "
                "`target_grid_experiment_idx` presente em `results`."
            )

    return PSLA4MLData(
        R=R,
        P=P,
        NM=NM if NM is not None else len(R),
        CM=CM,
        TM=TM,
        Eref=Eref,
        DS=DS_value,
        MC=MC,
        Gf=Gf,
        alpha1=alpha1,
        alpha2=alpha2,
        alpha3=alpha3,
        c=c,
        d=d,
        m=m,
        e=e,
        n=n,
        g=g,
        c_comm=c_comm,
    )


# ============================================================================
# PONTO DE ENTRADA — leitura direta dos arquivos JSON persistidos
# ============================================================================

def load_psla4ml_data(
    grid_config_path: str | Path,
    state_path: str | Path,
    *,
    sla_profile: dict[str, Any] | None = None,
    target_grid_experiment_idx: int | None = None,
    resource_catalog_path: str | Path | None = None,
    **kwargs: Any,
) -> PSLA4MLData:
    """Lê os arquivos JSON persistidos e monta a instância ``PSLA4MLData``.

    Args:
        grid_config_path: Caminho do grid config multiambiente (ex.:
            ``gridsearch/config/grid_search_multienv.json``).
        state_path: Caminho do ``grid_search_state_*.json`` já persistido
            pela execução (``gridsearch.executor.save_state``).
        sla_profile: Perfil de ``sla_profiles.json`` (dict de um profile).
        target_grid_experiment_idx: Índice do experimento em ``results``
            usado para ``Gf``.
        resource_catalog_path: Caminho do ``resource_catalog_*.json``
            persistido pelo estágio de coleta de
            ``gridsearch.executor.run_grid_search``. ``None`` deixa
            ``d``/``n``/``g`` ausentes quando não declarados em
            ``environments.details``.
        **kwargs: Repassado para :func:`build_psla4ml_data` (``NM``, ``DS``,
            ``Gf``, ``target_result``, ``processor_types``, ``time_period_hours``).

    Returns:
        Instância ``PSLA4MLData`` para uso no modelo Gurobi.
    """
    environments_details = read_environments_details(grid_config_path)
    state = read_grid_search_state(state_path)
    results = state.get("results", [])
    resource_catalog = (
        read_resource_catalog(resource_catalog_path) if resource_catalog_path is not None else None
    )
    return build_psla4ml_data(
        environments_details,
        results,
        sla_profile=sla_profile,
        target_grid_experiment_idx=target_grid_experiment_idx,
        resource_catalog=resource_catalog,
        **kwargs,
    )

