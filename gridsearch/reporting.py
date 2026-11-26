"""
Grid Search Reporting — Análise e relatórios
=============================================

Análise de resultados do grid search e geração de relatórios
textuais (CLI e programático).

Autor: Gustavo Alexandre
Data: 2026-02-15
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

from .executor import (
    ENERGY_COST_USD_PER_KWH,
    GRID_OUTPUT_DIR,
    _grid_results_file,
    _grid_summary_file,
    run_grid_search,
)

logger = logging.getLogger(__name__)


# ============================================================================
# ANÁLISE DE RESULTADOS
# ============================================================================

def analyze_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analisa resultados e identifica as melhores configurações por múltiplos critérios.

    Critérios de análise:
        - Tempo de treinamento (train_time_sec)
        - Eficiência energética (energy_kwh)
        - Uso de memória RAM (peak_ram_mb)
        - Emissão de carbono (emissions_kg_co2)
        - Custo financeiro (cost_usd, calculado a partir de energy_kwh)

    Args:
        results: Lista com resultados de todos os experimentos

    Returns:
        Dicionário com análise dos resultados incluindo:
        - best_by_time: Melhor configuração por tempo
        - best_by_energy: Melhor configuração por energia
        - best_by_ram: Melhor configuração por memória
        - best_by_carbon: Melhor configuração por emissão de CO2
        - best_by_cost: Melhor configuração por custo financeiro
    """
    logger.info("Analisando resultados...")

    # Filtra experimentos bem-sucedidos
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    infra_failed = [r for r in failed if r.get("error_type") == "InfraError"]
    functional_failed = [r for r in failed if r.get("error_type") != "InfraError"]

    logger.info(f"Experimentos bem-sucedidos: {len(successful)}")
    logger.info(f"Experimentos falhos por InfraError: {len(infra_failed)}")
    logger.info(f"Experimentos falhos por erro funcional: {len(functional_failed)}")

    if not successful:
        logger.warning("Nenhum experimento foi concluído com sucesso!")
        return {
            "total_experiments": len(results),
            "successful": 0,
            "failed": len(failed),
            "infra_failed": len(infra_failed),
            "functional_failed": len(functional_failed),
            "best_config": None
        }

    # Ordena por tempo de treinamento (menor é melhor)
    sorted_by_time = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("train_time_sec", float('inf')))
    )

    # Ordena por eficiência energética (menor é melhor)
    sorted_by_energy = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("energy_kwh", float('inf')))
        if x.get("resources", {}).get("energy_kwh") is not None else float('inf')
    )

    # Ordena por uso de RAM (menor é melhor)
    sorted_by_ram = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("peak_ram_mb", float('inf')))
        if x.get("resources", {}).get("peak_ram_mb") is not None else float('inf')
    )

    # Ordena por emissão de CO2 (menor é melhor)
    sorted_by_carbon = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("emissions_kg_co2", float('inf')))
        if x.get("resources", {}).get("emissions_kg_co2") is not None else float('inf')
    )

    # Calcula custo financeiro e ordena (menor é melhor)
    for result in successful:
        energy_kwh = result.get("resources", {}).get("energy_kwh")
        if energy_kwh is not None:
            cost_usd = float(energy_kwh) * ENERGY_COST_USD_PER_KWH
            result["resources"]["cost_usd"] = cost_usd
        else:
            result["resources"]["cost_usd"] = None

    sorted_by_cost = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("cost_usd", float('inf')))
        if x.get("resources", {}).get("cost_usd") is not None else float('inf')
    )

    analysis = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "total_experiments": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "infra_failed": len(infra_failed),
        "functional_failed": len(functional_failed),
        "energy_cost_usd_per_kwh": ENERGY_COST_USD_PER_KWH,

        "best_by_time": {
            "experiment_idx": sorted_by_time[0]["grid_experiment_idx"],
            "params": sorted_by_time[0]["grid_params"],
            "train_time_sec": sorted_by_time[0]["resources"]["train_time_sec"]
        } if sorted_by_time else None,

        "best_by_energy": {
            "experiment_idx": sorted_by_energy[0]["grid_experiment_idx"],
            "params": sorted_by_energy[0]["grid_params"],
            "energy_kwh": sorted_by_energy[0]["resources"].get("energy_kwh")
        } if sorted_by_energy and sorted_by_energy[0]["resources"].get("energy_kwh") else None,

        "best_by_ram": {
            "experiment_idx": sorted_by_ram[0]["grid_experiment_idx"],
            "params": sorted_by_ram[0]["grid_params"],
            "peak_ram_mb": sorted_by_ram[0]["resources"].get("peak_ram_mb")
        } if sorted_by_ram and sorted_by_ram[0]["resources"].get("peak_ram_mb") else None,

        "best_by_carbon": {
            "experiment_idx": sorted_by_carbon[0]["grid_experiment_idx"],
            "params": sorted_by_carbon[0]["grid_params"],
            "emissions_kg_co2": sorted_by_carbon[0]["resources"].get("emissions_kg_co2")
        } if sorted_by_carbon and sorted_by_carbon[0]["resources"].get("emissions_kg_co2") else None,

        "best_by_cost": {
            "experiment_idx": sorted_by_cost[0]["grid_experiment_idx"],
            "params": sorted_by_cost[0]["grid_params"],
            "cost_usd": sorted_by_cost[0]["resources"].get("cost_usd")
        } if sorted_by_cost and sorted_by_cost[0]["resources"].get("cost_usd") else None,

        "all_results": results
    }

    return analysis


def generate_summary_report(analysis: dict[str, Any]) -> str:
    """
    Gera relatório textual resumido dos resultados.

    Args:
        analysis: Dicionário com análise dos resultados

    Returns:
        String formatada com o relatório
    """
    report = []
    report.append("=" * 80)
    report.append("GRID SEARCH - RELATÓRIO DE RESULTADOS")
    report.append("=" * 80)
    report.append(f"Data: {datetime.now().astimezone().strftime('%d/%m/%Y %H:%M:%S')}")
    report.append("")

    report.append("RESUMO GERAL:")
    report.append(f"  Total de experimentos: {analysis['total_experiments']}")
    report.append(f"  Bem-sucedidos: {analysis['successful']}")
    report.append(f"  Falhos: {analysis['failed']}")
    report.append("")

    if analysis.get("best_by_time"):
        report.append("MELHOR CONFIGURAÇÃO (Tempo de Treinamento):")
        best = analysis["best_by_time"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Tempo: {best['train_time_sec']} segundos")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")

    if analysis.get("best_by_energy"):
        report.append("MELHOR CONFIGURAÇÃO (Eficiência Energética):")
        best = analysis["best_by_energy"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Energia: {best['energy_kwh']} kWh")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")

    if analysis.get("best_by_ram"):
        report.append("MELHOR CONFIGURAÇÃO (Uso de Memória RAM):")
        best = analysis["best_by_ram"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  RAM Pico: {best['peak_ram_mb']} MB")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")

    if analysis.get("best_by_carbon"):
        report.append("MELHOR CONFIGURAÇÃO (Menor Emissão de Carbono):")
        best = analysis["best_by_carbon"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Emissão CO2: {best['emissions_kg_co2']:.6f} kg")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")

    if analysis.get("best_by_cost"):
        report.append("MELHOR CONFIGURAÇÃO (Menor Custo Financeiro):")
        best = analysis["best_by_cost"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Custo: ${best['cost_usd']:.4f} USD")
        report.append(f"  (Tarifa: ${analysis['energy_cost_usd_per_kwh']:.4f}/kWh)")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")

    report.append("=" * 80)

    # Adiciona estatísticas gerais
    if analysis['successful'] > 0:
        report.append("")
        report.append("ESTATÍSTICAS GERAIS DOS EXPERIMENTOS BEM-SUCEDIDOS:")
        report.append("")

        # Calcula estatísticas agregadas
        all_successful = [r for r in analysis['all_results'] if r.get('status') == 'success']

        # Tempo total
        total_time = sum(
            float(r.get('resources', {}).get('train_time_sec', 0))
            for r in all_successful
        )
        report.append(f"  Tempo total de treinamento: {total_time:.2f} segundos ({total_time/3600:.2f} horas)")

        # Energia total
        total_energy = sum(
            float(r.get('resources', {}).get('energy_kwh', 0) or 0)
            for r in all_successful
        )
        if total_energy > 0:
            report.append(f"  Energia total consumida: {total_energy:.4f} kWh")

        # CO2 total
        total_co2 = sum(
            float(r.get('resources', {}).get('emissions_kg_co2', 0) or 0)
            for r in all_successful
        )
        if total_co2 > 0:
            report.append(f"  Emissão total de CO2: {total_co2:.6f} kg ({total_co2*1000:.2f} g)")

        # Custo total
        total_cost = sum(
            float(r.get('resources', {}).get('cost_usd', 0) or 0)
            for r in all_successful
        )
        if total_cost > 0:
            report.append(f"  Custo financeiro total: ${total_cost:.4f} USD")

        report.append("")
        report.append("=" * 80)

    return "\n".join(report)


# ============================================================================
# INTERFACE CLI
# ============================================================================

def main():
    """Ponto de entrada CLI para execução de grid search via ``python -m gridsearch.core``.

    Analisa argumentos de linha de comando e despacha para uma das operações:

    - Busca em grade completa (``--config`` + ``--search-config``)
    - Retomada de execução interrompida (``--resume``)
    - Análise de resultados existentes (``--analyze-only``)
    """
    parser = argparse.ArgumentParser(
        description="Grid Search para hiperparâmetros do BERT-PLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Busca completa com configuração JSON
  python -m gridsearch.core --config config/experiments/BertPLI.config \\
                            --search-config gridsearch/config/grid_search.json

  # Execução paralela com 4 processos
  python -m gridsearch.core --config config/experiments/BertPLI.config \\
                            --search-config gridsearch/config/grid_search.json \\
                            --parallel 4

  # Retomar execução interrompida
  python -m gridsearch.core --resume

  # Analisar resultados existentes
  python -m gridsearch.core --analyze-only
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Caminho do arquivo de configuração base"
    )

    parser.add_argument(
        "--search-config",
        type=str,
        help="Caminho do arquivo JSON com grade de hiperparâmetros"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma execução anterior usando estado salvo"
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Número de processos paralelos (padrão: 1 = sequencial)"
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Apenas analisa resultados existentes sem executar novos experimentos"
    )

    args = parser.parse_args()

    # Modo: apenas análise
    if args.analyze_only:
        results_file = None
        if _grid_results_file().exists():
            results_file = _grid_results_file()
        else:
            fallback_no_date = GRID_OUTPUT_DIR / "grid_search_results.json"
            if fallback_no_date.exists():
                results_file = fallback_no_date
                logger.warning(f"Arquivo do dia não encontrado. Usando: {results_file}")
            else:
                candidates = sorted(GRID_OUTPUT_DIR.glob("grid_search_results_*.json"), reverse=True)
                if candidates:
                    results_file = candidates[0]
                    logger.warning(f"Arquivo do dia não encontrado. Usando o mais recente: {results_file}")

        if results_file is None:
            logger.error(f"Nenhum arquivo de resultados encontrado em: {GRID_OUTPUT_DIR}")
            sys.exit(1)

        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        analysis = analyze_results(results)
        report = generate_summary_report(analysis)

        print("\n" + report)

        with open(_grid_summary_file(), 'w', encoding='utf-8') as f:
            f.write(report)

        canonical_summary = GRID_OUTPUT_DIR / "grid_search_summary.txt"
        with open(canonical_summary, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Relatório salvo em: {_grid_summary_file()}")
        logger.info(f"Relatório canônico salvo em: {canonical_summary}")
        return

    # Modo: retomar execução
    if args.resume:
        from .executor import _grid_state_file
        if not _grid_state_file().exists():
            logger.error(f"Arquivo de estado não encontrado: {_grid_state_file()}")
            sys.exit(1)

        logger.info("Retomando execução...")

    else:
        # Modo: nova execução
        if not args.config or not args.search_config:
            parser.error("--config e --search-config são obrigatórios para nova execução")

        if not os.path.exists(args.config):
            logger.error(f"Arquivo de configuração não encontrado: {args.config}")
            sys.exit(1)

        if not os.path.exists(args.search_config):
            logger.error(f"Arquivo de busca não encontrado: {args.search_config}")
            sys.exit(1)

        # Carrega configuração da grade
        with open(args.search_config, 'r', encoding='utf-8') as f:
            grid_config = json.load(f)

        logger.info(f"Configuração base: {args.config}")
        logger.info(f"Grade de hiperparâmetros: {args.search_config}")
        logger.info(f"Modo de execução: {'Paralelo (' + str(args.parallel) + ' workers)' if args.parallel > 1 else 'Sequencial'}")

        # Executa grid search
        results = run_grid_search(
            base_config_path=args.config,
            grid_config=grid_config,
            resume=args.resume,
            parallel=args.parallel
        )

        # Salva resultados completos
        with open(_grid_results_file(), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Resultados completos salvos em: {_grid_results_file()}")

        # Analisa e gera relatório
        analysis = analyze_results(results)
        report = generate_summary_report(analysis)

        print("\n" + report)

        with open(_grid_summary_file(), 'w', encoding='utf-8') as f:
            f.write(report)

        canonical_summary = GRID_OUTPUT_DIR / "grid_search_summary.txt"
        with open(canonical_summary, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Relatório salvo em: {_grid_summary_file()}")
        logger.info(f"Relatório canônico salvo em: {canonical_summary}")


if __name__ == "__main__":
    main()
