"""
Helpers de execução de experimentos
====================================

Funções utilitárias, constantes e classes auxiliares usadas pelo
motor de execução de experimentos.

Autor: Gustavo Alexandre
"""

import configparser
import logging
import os
import platform
from datetime import datetime

from utils.paths import PathManager

logger = logging.getLogger(__name__)

# Constantes de diretórios e arquivos
METRICS_DIR = PathManager.EXPERIMENTS_DIR / "metrics"
PathManager.ensure_dir(METRICS_DIR)

# Tarifa de energia (USD/kWh) — configurável via variável de ambiente
ENERGY_COST_USD_PER_KWH = float(os.getenv("ENERGY_COST_USD_PER_KWH", "0.12"))


def compute_cost_usd(
    energy_kwh: "float | None",
    train_time_sec: "float | None" = None,
    environment_cost_per_hour_usd: "float | None" = None,
) -> "float | None":
    """Calcula o custo estimado em USD do experimento.

    Quando ``environment_cost_per_hour_usd`` é fornecido usa a fórmula
    do artigo PSLA4ML (Seção 3.2), baseada no custo horário de nuvem::

        cost_usd = (train_time_sec / 3600) × environment_cost_per_hour_usd

    Caso contrário usa a tarifa de energia flat configurada via
    ``ENERGY_COST_USD_PER_KWH``::

        cost_usd = energy_kwh × ENERGY_COST_USD_PER_KWH

    Args:
        energy_kwh: Energia consumida em kWh. Pode ser ``None`` quando o
            CodeCarbon não está habilitado.
        train_time_sec: Duração do treinamento em segundos. Necessário
            quando ``environment_cost_per_hour_usd`` é fornecido.
        environment_cost_per_hour_usd: Custo horário do ambiente de nuvem
            (ex: CPU=0.10, GPU=1.20, TPU=1.50 segundo o artigo PSLA4ML).
            ``None`` ativa o fallback para tarifa flat.

    Returns:
        Custo estimado em USD, ou ``None`` quando os dados de entrada
        são insuficientes.
    """
    if environment_cost_per_hour_usd is not None:
        if train_time_sec is None:
            return None
        return (train_time_sec / 3600.0) * environment_cost_per_hour_usd
    if energy_kwh is None:
        return None
    return energy_kwh * ENERGY_COST_USD_PER_KWH


def now_iso() -> str:
    """Retorna o instante atual em formato ISO 8601 com timezone UTC."""
    return datetime.now().astimezone().isoformat()


def get_machine_name() -> str:
    """Retorna o identificador da máquina de execução de forma estável.

    Prioriza variáveis de ambiente comuns em cloud e desktop e faz fallback
    para ``platform.node()`` quando necessário.
    """
    for key in ("PSLA4ML_MACHINE_NAME", "COMPUTERNAME", "HOSTNAME"):
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    node = platform.node().strip()
    return node or "unknown-machine"


def normalize_environment_label(value: str | None) -> str:
    """Normaliza um rótulo de ambiente para o conjunto canônico do projeto.

    Valores suportados: ``LOCAL``, ``COLAB``, ``AWS``, ``AZURE``, ``GCP``.
    """
    raw = (value or "").strip().upper()
    if raw in {"LOCAL", "COLAB", "AWS", "AZURE", "GCP"}:
        return raw
    return "LOCAL"


def detect_execution_environment() -> str:
    """Detecta o ambiente real de execução da máquina atual.

    Regras:
    - ``COLAB``: execução no Google Colab.
    - ``AZURE``: variáveis típicas de App Service/Functions/identidade Azure.
    - ``GCP``: variáveis típicas de Cloud Run/Functions/App Engine/GCE.
    - ``AWS``: variáveis típicas de Lambda/ECS/EC2.
    - fallback ``LOCAL``.
    """
    env = os.environ
    if any(env.get(key) for key in ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_BACKEND_VERSION")):
        return "COLAB"
    if any(env.get(key) for key in ("WEBSITE_SITE_NAME", "FUNCTIONS_WORKER_RUNTIME", "AZURE_HTTP_USER_AGENT", "IDENTITY_ENDPOINT")):
        return "AZURE"
    if any(env.get(key) for key in ("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "K_SERVICE", "FUNCTION_TARGET", "GAE_ENV")):
        return "GCP"
    if any(env.get(key) for key in ("AWS_EXECUTION_ENV", "AWS_REGION", "AWS_DEFAULT_REGION", "LAMBDA_TASK_ROOT", "ECS_CONTAINER_METADATA_URI", "ECS_CONTAINER_METADATA_URI_V4")):
        return "AWS"
    return "LOCAL"


def load_config(path: str) -> configparser.ConfigParser:
    """Lê e retorna um ConfigParser padrão da stdlib a partir do arquivo.

    Args:
        path: Caminho para o arquivo ``.config``.

    Returns:
        ``configparser.ConfigParser`` com as seções e chaves carregadas.
    """
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def estimate_bert_flops(
    seq_len: int,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
) -> float:
    """Estima os GFLOPs de uma passagem forward pelo BERT.

    Args:
        seq_len: Comprimento da sequência de tokens.
        hidden_size: Dimensão oculta do modelo. Padrão: 768 (BERT-base).
        num_layers: Número de camadas Transformer. Padrão: 12.
        num_heads: Número de cabeças de atenção. Padrão: 12.

    Returns:
        GFLOPs estimados por forward pass.
    """
    attention = (
        4 * seq_len * hidden_size * hidden_size +
        2 * num_heads * seq_len * seq_len * (hidden_size // num_heads)
    )
    ffn = 8 * seq_len * hidden_size * hidden_size
    return num_layers * (attention + ffn) / 1e9  # GFLOPs


class TeeStream:
    """Stream que escreve simultaneamente no terminal e em um buffer em memória.

    Permite capturar o stdout do loop de treino sem perder a saída em
    tempo real. Implementa a interface mínima exigida pelo Python e pelo
    Transformers >= 5.x (``write``, ``flush``, ``isatty``, ``fileno``).
    """

    def __init__(self, original) -> None:
        self.original = original
        self.lines: list = []

    def write(self, text: str) -> None:
        """Escreve ``text`` no stream original e acrescenta à lista ``lines``."""
        self.original.write(text)
        self.lines.append(text)

    def flush(self) -> None:
        """Propaga flush para o stream original."""
        self.original.flush()

    def isatty(self) -> bool:
        """Delega ``isatty()`` ao stream original; retorna ``False`` se não implementado."""
        return getattr(self.original, "isatty", lambda: False)()

    def fileno(self) -> int:
        """Retorna o descritor de arquivo do stream original."""
        return self.original.fileno()
