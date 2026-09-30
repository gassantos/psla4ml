"""
Resumo SLA de execução
=======================

Funções para montar e exibir o bloco de resumo SLA de execução,
incluindo KPIs agregados do grid search.

Autor: Gustavo Alexandre
"""

import json
import logging
from typing import Any

from gridsearch.core import GRID_OUTPUT_DIR

logger = logging.getLogger(__name__)


def _load_latest_grid_state() -> dict[str, Any] | None:
    """Carrega o arquivo de estado mais recente do grid search."""
    candidates = sorted(GRID_OUTPUT_DIR.glob("grid_search_state_*.json"), reverse=True)
    if not candidates:
        return None

    state_file = candidates[0]
    try:
        with open(state_file, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning("Não foi possível ler estado para resumo SLA: %s", state_file)
        return None

    return state


def _emit_sla_execution_summary(
    sla_prefilter: dict[str, Any] | None,
    sla_profile_name: str | None,
    results: list[dict[str, Any]] | None = None,
) -> None:
    """Imprime e registra no logger o resumo final da triagem SLA de execução."""
    if not sla_prefilter or not sla_prefilter.get("enabled"):
        return

    lines = _build_sla_execution_summary_lines(
        sla_prefilter=sla_prefilter,
        sla_profile_name=sla_profile_name,
        results=results,
    )

    print()
    for line in lines:
        print(line)

    for line in lines:
        if line.strip():
            logger.info(line)


def _build_sla_execution_summary_lines(
    sla_prefilter: dict[str, Any],
    sla_profile_name: str | None,
    results: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Monta linhas do bloco de resumo SLA de execução para print/log."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("RESUMO SLA (EXECUÇÃO)")
    lines.append("=" * 72)
    lines.append(f"  Perfil SLA          : {sla_profile_name or 'custom/constraints'}")
    lines.append(f"  Constraints         : {sla_prefilter.get('constraints', {})}")
    lines.append(
        "  Experimentos        : "
        f"originais={sla_prefilter.get('original_total_experiments', 0)} | "
        f"elegíveis={sla_prefilter.get('eligible_total_experiments', 0)} | "
        f"rejeitados={sla_prefilter.get('rejected_total_experiments', 0)}"
    )

    lines.extend(_build_execution_kpi_lines(results or []))

    ranked = sorted(
        (sla_prefilter.get("rejected_by_metric") or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )
    ranked = [(metric, count) for metric, count in ranked if count > 0]
    if ranked:
        ranking_text = ", ".join(f"{metric}={count}" for metric, count in ranked)
        lines.append(f"  Ranking rejeições   : {ranking_text}")

    non_eval = sla_prefilter.get("non_evaluable_constraints") or []
    if non_eval:
        lines.append(f"  Não avaliáveis      : {non_eval}")

    sample_list = (sla_prefilter.get("rejected_samples") or [])[:3]
    if sample_list:
        lines.append("  Exemplos rejeitados :")
        for sample in sample_list:
            lines.append(
                "    - "
                f"idx={sample.get('grid_experiment_idx')} "
                f"metric={sample.get('metric')} "
                f"estimated={float(sample.get('estimated_value', 0.0)):.4f} "
                f"threshold={float(sample.get('threshold', 0.0)):.4f}"
            )

    truncated = int(sla_prefilter.get("rejected_samples_truncated", 0) or 0)
    if truncated > 0:
        lines.append(
            "  Amostra truncada    : "
            f"{truncated} rejeições omitidas (limite={sla_prefilter.get('rejected_samples_limit')})"
        )

    lines.append("=" * 72)
    return lines


def _build_infra_waste_lines(infra_failed: list[dict[str, Any]]) -> list[str]:
    """Monta linhas de desperdício operacional causado por falhas de infraestrutura."""
    lines: list[str] = []
    lines.append(f"  Desperdício infra    : {len(infra_failed)} experimento(s) perdido(s) por InfraError")

    def _sum_resource(key: str) -> float | None:
        vals = []
        for r in infra_failed:
            val = (r.get("resources") or {}).get(key)
            if val is not None:
                try:
                    vals.append(float(val))
                except (TypeError, ValueError):
                    pass
        return sum(vals) if vals else None

    wasted_time = _sum_resource("train_time_sec")
    if wasted_time is not None:
        lines.append(f"  Tempo desperdiçado  : {wasted_time:.2f}s")

    wasted_energy = _sum_resource("energy_kwh")
    if wasted_energy is not None:
        lines.append(f"  Energia desperdiçada: {wasted_energy:.6f} kWh")

    wasted_cost = _sum_resource("cost_usd")
    if wasted_cost is not None:
        lines.append(f"  Custo desperdiçado  : ${wasted_cost:.6f} USD")

    return lines


def _build_execution_kpi_lines(results: list[dict[str, Any]]) -> list[str]:
    """Monta linhas de KPIs agregados da execução real dos experimentos."""
    lines: list[str] = []

    executed = [r for r in results if isinstance(r, dict)]
    successful = [r for r in executed if r.get("status") == "success"]
    failed = [r for r in executed if r.get("status") == "failed"]
    infra_failed = [r for r in failed if r.get("error_type") == "InfraError"]
    functional_failed = [r for r in failed if r.get("error_type") != "InfraError"]

    lines.append(
        "  Execução real       : "
        f"rodados={len(executed)} | sucesso={len(successful)} | "
        f"falha_infra={len(infra_failed)} | falha_funcional={len(functional_failed)}"
    )

    if infra_failed:
        lines.extend(_build_infra_waste_lines(infra_failed))

    def _values(path1: str, path2: str) -> list[float]:
        vals: list[float] = []
        for item in successful:
            sub = item.get(path1, {}) if isinstance(item.get(path1), dict) else {}
            val = sub.get(path2)
            if val is None:
                continue
            try:
                vals.append(float(val))
            except (TypeError, ValueError):
                continue
        return vals

    time_vals = _values("resources", "train_time_sec")
    if time_vals:
        total_time = sum(time_vals)
        lines.append(
            "  KPI tempo           : "
            f"media={total_time / len(time_vals):.2f}s | total={total_time:.2f}s"
        )

    energy_vals = _values("resources", "energy_kwh")
    if energy_vals:
        lines.append(f"  KPI energia         : total={sum(energy_vals):.6f} kWh")

    co2_vals = _values("resources", "emissions_kg_co2")
    if co2_vals:
        lines.append(f"  KPI CO2             : total={sum(co2_vals):.6f} kg")

    cost_vals = _values("resources", "cost_usd")
    if cost_vals:
        lines.append(f"  KPI custo           : total=${sum(cost_vals):.6f} USD")

    f1_vals: list[float] = []
    for item in successful:
        evaluation = item.get("evaluation", {}) if isinstance(item.get("evaluation"), dict) else {}
        raw_f1 = evaluation.get("f1_score")
        if raw_f1 is None:
            continue
        try:
            f1_vals.append(float(raw_f1))
        except (TypeError, ValueError):
            continue
    if f1_vals:
        lines.append(
            "  KPI F1              : "
            f"melhor={max(f1_vals):.4f} | media={sum(f1_vals)/len(f1_vals):.4f}"
        )

    return lines
