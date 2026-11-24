"""
Construtor do argparse para Gridsearch Experiment Runner
=======================================================

Centraliza a definição de todos os argumentos CLI em uma função
de construção ``build_argument_parser()``.

Autor: Gustavo Alexandre
"""

import argparse

from .constants import (
    DEFAULT_CONFIG,
    DEFAULT_GRID_CONFIG,
    DEFAULT_MODE,
    DEFAULT_PARALLEL,
    DEFAULT_SKYBAND_K,
    DEFAULT_TRAIN_DATASET,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Constrói e retorna o ``ArgumentParser`` completo do CLI."""
    parser = argparse.ArgumentParser(
        description="Gridsearch Experiment Runner - Execução centralizada de experimentos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemplos de uso:

  # Execução padrão — grid search + Skyband k={DEFAULT_SKYBAND_K} automático
  python -m main

  # Experimento único (Skyband também roda ao final por padrão)
  python -m main --mode single

  # Experimento único com seleção explícita de GPU
  python -m main --mode single --gpu 0
  python -m main --mode single --gpu 0 1

  # Grid search com configuração específica
  python -m main --mode grid --grid-config gridsearch/config/grid_search_test.json

  # Grid search paralelo com 4 workers
  python -m main --mode grid --parallel 4

    # Experimento usando todos os 8 cores de uma TPU no Colab
    python -m main --mode single --tpu-cores 8

  # Retomar grid search interrompido (Skyband roda ao final)
  python -m main --mode grid --resume

  # Desativar análise Skyband (somente execução dos experimentos)
  python -m main --mode grid --no-skyband

  # Skyband com perfil de SLA sustentável (k={DEFAULT_SKYBAND_K} default)
  python -m main --mode grid --sla-profile sustentavel

  # Skyband com k personalizado
  python -m main --mode grid --skyband-k 5 --sla-profile balanceado

  # Apenas análise Skyband sobre estado existente (sem novo treino)
  python -m main --skyband-only

  # Skyband-only com k=2 e constraints de SLA customizadas
  python -m main --skyband-only --skyband-k 2 \\
      --sla-constraint cost_usd=5.0 \\
      --sla-constraint train_time_sec=7200

  # Skyband-only com perfil predefinido + comparação vs ranking escalar
  python -m main --skyband-only --sla-profile balanceado --skyband-compare

  # Skyband sobre arquivo de estado específico
  python -m main --skyband-only \\
      --skyband-state output/experiments/grid_search/grid_search_state_GPU_2026-03-01.json \\
      --skyband-k 2 --skyband-metrics train_time_sec cost_usd energy_kwh

  # Skyband com métricas customizadas (2 critérios: tempo e custo)
  python -m main --skyband-only --skyband-metrics train_time_sec cost_usd

  # Usar dataset público do HuggingFace Hub (glue/mrpc)
  python -m main --mode single \\
      --dataset-source hub --dataset-id nyu-mll/glue --dataset-config mrpc

  # Usar dataset local JSONL via HuggingFace Datasets (substitui config)
  python -m main --mode single \\
      --dataset-source local_json

  # Retomar um workflow Hugging Face persistido e usar cache semântico
  python -m main --mode single --dataset-source hub --dataset-id nyu-mll/glue \
      --workflow-resume-run output/experiments/workflow_runs/<run-id> \
      --workflow-cache-dir output/experiments/workflow_cache

  # Grid search com dataset do Hub + SLA
  python -m main --mode grid --sla-profile dev \\
      --dataset-source hub --dataset-id nyu-mll/glue --dataset-config mrpc

Perfis de SLA disponíveis (--sla-profile):
  economico    — custo <= $2.00
  sustentavel  — energia <= 0.05 kWh, CO2 <= 0.01 kg
  tempo        — treino <= 3600 s
  balanceado   — custo <= $5.00, tempo <= 7200 s, energia <= 0.1 kWh
  dev          — tempo <= 1800 s, RAM <= 8192 MB
  producao     — custo <= $20.00, tempo <= 1800 s, RAM <= 16384 MB

Métricas para --sla-constraint (filtro de admissibilidade, checagem de execução):
  train_time_sec   energy_kwh   peak_ram_mb   emissions_kg_co2   cost_usd

Métricas para --skyband-metrics (critérios de dominância Skyband):
  train_time_sec   energy_kwh   total_gflops   emissions_kg_co2   cost_usd

Configurações padrão:
  - Modo: {DEFAULT_MODE}
  - Config: {DEFAULT_CONFIG}
  - Grid config: {DEFAULT_GRID_CONFIG}
  - Parallel: {DEFAULT_PARALLEL}
  - Dataset: {DEFAULT_TRAIN_DATASET}
  - Skyband k: {DEFAULT_SKYBAND_K}
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "grid"],
        default=DEFAULT_MODE,
        help=f"Modo de execução: 'single' para um único experimento, 'grid' para grid search (padrão: {DEFAULT_MODE})"
    )

    parser.add_argument(
        "--workflow",
        choices=["bertpli", "generic"],
        default=None,
        help="Executa um workflow declarativo em vez dos modos single/grid.",
    )

    parser.add_argument(
        "--workflow-dry-run",
        action="store_true",
        help="Valida o encadeamento do workflow sem executar os treinamentos.",
    )

    parser.add_argument(
        "--workflow-spec",
        type=str,
        default=None,
        metavar="ARQUIVO",
        help="Especificacao JSON obrigatoria para --workflow generic.",
    )

    parser.add_argument(
        "--workflow-resume-run",
        type=str,
        default=None,
        metavar="DIRETORIO",
        help="Retoma um manifesto persistido no workflow Hugging Face do modo single.",
    )

    parser.add_argument(
        "--workflow-cache-dir",
        type=str,
        default=None,
        metavar="DIRETORIO",
        help="Diretório opcional para cache semântico no workflow Hugging Face do modo single.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Caminho do arquivo de configuração base (padrão: {DEFAULT_CONFIG})"
    )

    parser.add_argument(
        "--grid-config",
        type=str,
        default=DEFAULT_GRID_CONFIG,
        help=f"Caminho do arquivo JSON com grade de hiperparâmetros (padrão: {DEFAULT_GRID_CONFIG})"
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help=f"Número de processos paralelos para grid search (padrão: {DEFAULT_PARALLEL})"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma execução anterior de grid search usando estado salvo"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        nargs="*",
        default=None,
        metavar="ID",
        help=(
            "IDs das GPUs a utilizar. "
            "Sem valor = detecção automática. "
            "Ex: --gpu 0  (single GPU), --gpu 0 1  (multi-GPU)"
        ),
    )

    parser.add_argument(
        "--tpu-cores",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Número de cores TPU para execução PJRT (padrão: 0 = desabilitado). "
            "Use 1 ou 8 no Colab TPU; requer --parallel 1 no modo grid."
        ),
    )

    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default=None,
        help="Sobrescreve environment.precision do arquivo de configuração.",
    )

    parser.add_argument(
        "--train-dataset",
        type=str,
        choices=["train_task2", "train_task2_v2", "train_task2_v3"],
        default=DEFAULT_TRAIN_DATASET,
        dest="train_dataset",
        help=(
            f"Arquivo de treino a utilizar (sem extensão). "
            f"Padrão: {DEFAULT_TRAIN_DATASET}. "
            "Opções: train_task2 | train_task2_v2 | train_task2_v3"
        ),
    )

    # ── Grupo: dataset HuggingFace ───────────────────────────────────────────
    hf_group = parser.add_argument_group(
        "Dataset HuggingFace",
        "Sobrescreve as chaves [data] do config para usar HuggingFaceDataset.",
    )

    hf_group.add_argument(
        "--dataset-source",
        type=str,
        choices=["hub", "local_json"],
        default=None,
        dest="dataset_source",
        metavar="FONTE",
        help=(
            "Fonte do dataset: 'hub' (HuggingFace Hub) ou 'local_json' (JSONL local). "
            "Quando informado, ativa automaticamente train/valid/test_dataset_type=HuggingFace."
        ),
    )

    hf_group.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        dest="dataset_id",
        metavar="ID",
        help=(
            "ID do dataset no HuggingFace Hub (ex: 'nyu-mll/glue') ou "
            "caminho local ao usar --dataset-source local_json."
        ),
    )

    hf_group.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        dest="dataset_config",
        metavar="CONFIG",
        help=(
            "Subconfiguração do dataset no Hub (ex: 'mrpc' para glue). "
            "Corresponde ao parâmetro 'name' do load_dataset."
        ),
    )

    # ── Grupo: análise Skyband ───────────────────────────────────────────────
    skyband_group = parser.add_argument_group(
        "Skyband",
        "Análise multicriterio por dominância de Pareto (Skyband Query Engine)",
    )

    skyband_group.add_argument(
        "--no-skyband",
        action="store_true",
        dest="no_skyband",
        help="Desativa a análise Skyband automática após a execução dos experimentos",
    )

    skyband_group.add_argument(
        "--skyband-only",
        action="store_true",
        help=(
            "Carrega estado existente e executa apenas a análise Skyband, "
            "sem disparar novos experimentos"
        ),
    )

    skyband_group.add_argument(
        "--skyband-k",
        type=int,
        default=DEFAULT_SKYBAND_K,
        metavar="K",
        help=(
            "Ordem do Skyband: retorna experimentos dominados por menos de K outros. "
            "k=1 = frente de Pareto pura. "
            f"k=2 inclui o segundo nível de dominância, etc. (padrão: {DEFAULT_SKYBAND_K})"
        ),
    )

    skyband_group.add_argument(
        "--sla-profile",
        type=str,
        default=None,
        metavar="PERFIL",
        choices=["economico", "sustentavel", "tempo", "balanceado", "dev", "producao"],
        help=(
            "Perfil de SLA predefinido em gridsearch/config/sla_profiles.json. "
            "Sobrescreve --skyband-k, --skyband-metrics e --sla-constraint quando informado. "
            "Opções: economico | sustentavel | tempo | balanceado | dev | producao"
        ),
    )

    skyband_group.add_argument(
        "--sla-constraint",
        action="append",
        metavar="METRICA=VALOR",
        dest="sla_constraints",
        help=(
            "Restrição de SLA no formato metrica=valor_maximo (pode repetir). "
            "Métricas disponíveis (filtro de admissibilidade): train_time_sec, energy_kwh, "
            "peak_ram_mb, emissions_kg_co2, cost_usd. "
            "Ex: --sla-constraint peak_ram_mb=8192 --sla-constraint cost_usd=5.0"
        ),
    )

    skyband_group.add_argument(
        "--skyband-metrics",
        nargs="+",
        metavar="METRICA",
        default=None,
        help=(
            "Lista de métricas a usar na dominância de Pareto (critérios Skyband). "
            "Padrão: train_time_sec energy_kwh total_gflops emissions_kg_co2 cost_usd "
            "(todos os 5 critérios). "
            "Ex: --skyband-metrics train_time_sec cost_usd total_gflops"
        ),
    )

    skyband_group.add_argument(
        "--skyband-compare",
        action="store_true",
        help="Exibe comparação entre Skyband e ranking escalar ponderado (Jaccard + diferenças)",
    )

    skyband_group.add_argument(
        "--skyband-state",
        type=str,
        default=None,
        metavar="ARQUIVO",
        help=(
            "Caminho direto para o arquivo JSON de estado do grid search a ser analisado. "
            "Padrão: detecta automaticamente o arquivo mais recente em "
            "output/experiments/grid_search/"
        ),
    )

    return parser
