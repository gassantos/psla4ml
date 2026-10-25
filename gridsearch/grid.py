"""
Grid Generation — Geração de grade de hiperparâmetros
=====================================================

Módulo responsável por gerar combinações de hiperparâmetros e criar
arquivos de configuração para cada experimento do grid search.

Porta de configuração (Hexagonal / OCP):
    O mapeamento ``param → (section, key)`` é lido de
    ``gridsearch/config/param_mapping.json`` — novos hiperparâmetros
    podem ser adicionados sem alterar código.

Autor: Gustavo Alexandre
Data: 2026-02-15
"""

import configparser
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# PORTA DE CONFIGURAÇÃO — Mapeamento de hiperparâmetros (Hexagonal / OCP)
# ============================================================================
_PARAM_MAPPING_PATH = Path(__file__).parent / "config" / "param_mapping.json"


def _load_param_mapping() -> Dict[str, Dict[str, str]]:
    """Carrega o mapeamento param_name → (section, key) de arquivo externo.

    O mapeamento define como cada hiperparâmetro do grid search é traduzido
    para uma seção/chave no ``ConfigParser``.  Novos hiperparâmetros podem
    ser adicionados editando ``gridsearch/config/param_mapping.json`` —
    sem alterar código (OCP).

    Returns:
        Dicionário ``{param_name: {"section": ..., "key": ...}}``.
    """
    with open(_PARAM_MAPPING_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("mapping", {})


def read_config_hyperparameters(config_path: str | Path) -> Dict[str, str]:
    """Valores efetivos dos hiperparâmetros mapeados, lidos de um config.

    Inclui também os que não variam na grade (ex.: ``seed``, ``dropout``); os ausentes
    no config ficam de fora. Os valores são texto, como no ``ConfigParser``.
    """
    from experiment.helpers import load_config

    config = load_config(str(config_path))
    values: Dict[str, str] = {}
    for name, target in _load_param_mapping().items():
        value = config.get(target["section"], target.get("key", name), fallback="").strip()
        if value:
            values[name] = value
    return values


# ============================================================================
# FILTRAGEM E VALIDAÇÃO DE CONFIGURAÇÃO DE GRID
# ============================================================================


def filter_grid_config(config: dict) -> dict:
    """
    Filtra configuração de grid search removendo campos de metadados.

    Suporta dois formatos:
    1. Hiperparâmetros em campo "hyperparameters": {"hyperparameters": {"lr": [...]}}
    2. Hiperparâmetros no nível raiz: {"lr": [...], "batch": [...]}

    Args:
        config: Dicionário de configuração completo

    Returns:
        Dicionário apenas com parâmetros de busca
    """
    # Se existe campo "hyperparameters", filtra metadados
    # e mantém apenas listas válidas de busca.
    if "hyperparameters" in config:
        return {
            k: v for k, v in config["hyperparameters"].items()
            if not str(k).startswith("_") and isinstance(v, list)
        }

    # Caso contrário, filtra metadados do nível raiz
    metadata_fields = {
        "description",
        "experiment_base",
        "output_dir",
        "parallel_workers",
        "notes",
        "recommendations"
    }

    # Remove metadados e mantém apenas listas (hiperparâmetros válidos)
    filtered = {
        k: v for k, v in config.items()
        if k not in metadata_fields and isinstance(v, list)
    }

    return filtered


def validate_grid_config(config: dict) -> Tuple[bool, str]:
    """
    Valida configuração de grid search.

    Args:
        config: Dicionário de configuração

    Returns:
        Tupla (is_valid, message)
    """
    filtered = filter_grid_config(config)

    if not filtered:
        return False, "Nenhum parâmetro de busca encontrado na configuração"

    # Verifica se todos os valores são listas
    for key, value in filtered.items():
        if not isinstance(value, list):
            return False, f"Parâmetro '{key}' deve ser uma lista de valores"

        if len(value) == 0:
            return False, f"Parâmetro '{key}' não pode ter lista vazia"

    # Calcula total de experimentos
    total = 1
    for value in filtered.values():
        total *= len(value)

    if total > 1000:
        message = (
            f"⚠️  AVISO: Grid search gerará {total} experimentos.\n"
            f"  Isso pode levar muito tempo. Considere reduzir o espaço de busca."
        )
        logger.warning(message)

    return True, f"Configuração válida: {total} experimentos serão gerados"


# ============================================================================
# GERAÇÃO DE COMBINAÇÕES
# ============================================================================

def generate_parameter_grid(grid_config: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Gera todas as combinações possíveis de hiperparâmetros.

    Args:
        grid_config: Dicionário com listas de valores para cada hiperparâmetro

    Returns:
        Lista de dicionários, cada um representando uma combinação única

    Exemplo:
        >>> grid = {
        ...     "learning_rate": [1e-5, 2e-5],
        ...     "batch_size": [8, 16]
        ... }
        >>> generate_parameter_grid(grid)
        [
            {"learning_rate": 1e-5, "batch_size": 8},
            {"learning_rate": 1e-5, "batch_size": 16},
            {"learning_rate": 2e-5, "batch_size": 8},
            {"learning_rate": 2e-5, "batch_size": 16}
        ]
    """
    # Filtra metadados da configuração
    full_config = grid_config if isinstance(grid_config, dict) else {}
    grid_config = filter_grid_config(full_config)

    # Quando a configuração define ambientes ativos, adiciona a dimensão
    # "environment" para expandir hiperparâmetros x ambientes.
    active_envs = (
        full_config.get("environments", {})
        .get("active", [])
        if isinstance(full_config.get("environments", {}), dict)
        else []
    )
    if isinstance(active_envs, list) and active_envs:
        grid_config = {**grid_config, "environment": active_envs}

    keys = list(grid_config.keys())
    values = list(grid_config.values())

    # Gera produto cartesiano
    combinations = list(itertools.product(*values))

    # Converte para lista de dicionários
    param_grid = []
    for combo in combinations:
        param_dict = dict(zip(keys, combo))
        param_grid.append(param_dict)

    logger.info(f"Geradas {len(param_grid)} combinações de hiperparâmetros")
    return param_grid


def create_config_for_combination(
    base_config_path: str,
    params: Dict[str, Any],
    experiment_idx: int,
    train_file: str = "train_task2",
    *,
    configs_dir: Path | None = None,
) -> str:
    """
    Cria um arquivo de configuração específico para uma combinação de parâmetros.

    O mapeamento ``param_name → (section, key)`` é carregado de
    ``gridsearch/config/param_mapping.json`` (OCP): para suportar novos
    hiperparâmetros basta adicionar uma entrada no JSON, sem alterar código.

    Args:
        base_config_path: Caminho do arquivo de configuração base
        params: Dicionário com os parâmetros a serem modificados
        experiment_idx: Índice do experimento na grade
        train_file: Nome do arquivo de treino sem extensão (ex:
            ``"train_task2_v2"``). Substitui ``train_file_list`` na seção
            ``[data]`` do config gerado.
        configs_dir: Diretório onde salvar os configs gerados. None = default.

    Returns:
        Caminho do novo arquivo de configuração criado
    """
    # Import tardio para evitar dependência circular
    from .executor import GRID_CONFIGS_DIR

    effective_configs_dir = configs_dir if configs_dir is not None else GRID_CONFIGS_DIR

    config = configparser.ConfigParser()
    config.read(base_config_path)

    # Carrega mapeamento externalizado (porta de configuração)
    mapping = _load_param_mapping()

    # Aplica cada hiperparâmetro via mapeamento genérico
    for param_name, value in params.items():
        entry = mapping.get(param_name)
        if entry is None:
            logger.warning(
                "Hiperparâmetro '%s' sem mapeamento em param_mapping.json — ignorado",
                param_name,
            )
            continue
        section = entry["section"]
        key = entry.get("key", param_name)
        if not config.has_section(section):
            config.add_section(section)
        config.set(section, key, str(value))

    # Atualiza dataset de treino
    if not config.has_section("data"):
        config.add_section("data")
    config.set("data", "train_file_list", f"{train_file}.json")

    # Atualiza nome do experimento
    base_name = config.get("experiment", "name")

    # Gera nome descritivo
    param_suffix = "_".join([
        f"{k}{v}".replace(".", "").replace("-", "")
        for k, v in params.items()
    ])

    new_name = f"{base_name}_grid{experiment_idx:03d}_{param_suffix}"
    config.set("experiment", "name", new_name)

    # Atualiza descrição
    description = f"Grid Search Experiment {experiment_idx}\n"
    description += "Hyperparameters:\n"
    for k, v in params.items():
        description += f"  - {k}: {v}\n"
    config.set("experiment", "description", description)

    # Salva nova configuração
    new_config_path = effective_configs_dir / f"grid_exp_{experiment_idx:03d}.config"
    with open(new_config_path, 'w') as f:
        config.write(f)

    logger.debug(f"Config criada: {new_config_path}")
    return str(new_config_path)
