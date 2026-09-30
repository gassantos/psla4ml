"""Testes de relatório de infraestrutura — Categorização de experimentos falhos."""

from cli.sla_summary import _build_execution_kpi_lines, _build_infra_waste_lines
from gridsearch.reporting import analyze_results

# ---------------------------------------------------------------------------
# Fixtures de resultados
# ---------------------------------------------------------------------------

def _success(idx: int, time_sec: float = 60.0, energy: float = 0.01) -> dict:
    return {
        "grid_experiment_idx": idx,
        "grid_params": {"lr": 1e-4},
        "status": "success",
        "resources": {"train_time_sec": time_sec, "energy_kwh": energy,
                      "emissions_kg_co2": 0.001, "peak_ram_mb": 512.0},
        "evaluation": {"f1_score": 0.9},
    }


def _infra_fail(idx: int) -> dict:
    return {
        "grid_experiment_idx": idx,
        "grid_params": {"lr": 1e-4},
        "status": "failed",
        "error_type": "InfraError",
        "error": "ReadTimeout: hub indisponível",
    }


def _func_fail(idx: int) -> dict:
    return {
        "grid_experiment_idx": idx,
        "grid_params": {"lr": 99.0},
        "status": "failed",
        "error_type": "RuntimeError",
        "error": "NaN em loss",
    }


# ---------------------------------------------------------------------------
# analyze_results — separação de falhas
# ---------------------------------------------------------------------------

def test_analyze_results_counts_infra_and_functional_separately():
    results = [_success(0), _infra_fail(1), _func_fail(2)]
    analysis = analyze_results(results)

    assert analysis["successful"] == 1
    assert analysis["failed"] == 2
    assert analysis["infra_failed"] == 1
    assert analysis["functional_failed"] == 1


def test_analyze_results_only_success_returns_zero_infra_failed():
    results = [_success(0), _success(1)]
    analysis = analyze_results(results)

    assert analysis["infra_failed"] == 0
    assert analysis["functional_failed"] == 0


def test_analyze_results_all_infra_fail_has_no_best_config():
    results = [_infra_fail(0), _infra_fail(1)]
    analysis = analyze_results(results)

    assert analysis["successful"] == 0
    assert analysis["infra_failed"] == 2
    assert analysis["functional_failed"] == 0
    assert analysis["best_config"] is None


def test_analyze_results_failed_without_error_type_counts_as_functional():
    """Resultados legados sem error_type são tratados como erro funcional."""
    legacy_fail = {"grid_experiment_idx": 0, "grid_params": {}, "status": "failed"}
    analysis = analyze_results([_success(1), legacy_fail])

    assert analysis["infra_failed"] == 0
    assert analysis["functional_failed"] == 1


# ---------------------------------------------------------------------------
# _build_execution_kpi_lines — contagem separada no relatório
# ---------------------------------------------------------------------------

def test_kpi_lines_separates_infra_and_functional_in_count_line():
    results = [_success(0), _infra_fail(1), _func_fail(2)]
    lines = _build_execution_kpi_lines(results)

    count_line = next(l for l in lines if "Execução real" in l)
    assert "falha_infra=1" in count_line
    assert "falha_funcional=1" in count_line


def test_kpi_lines_includes_waste_section_when_infra_errors_exist():
    results = [_success(0), _infra_fail(1)]
    lines = _build_execution_kpi_lines(results)

    assert any("Desperdício infra" in l for l in lines)


def test_kpi_lines_no_waste_section_when_no_infra_errors():
    results = [_success(0), _func_fail(1)]
    lines = _build_execution_kpi_lines(results)

    assert not any("Desperdício infra" in l for l in lines)


# ---------------------------------------------------------------------------
# _build_infra_waste_lines — recursos desperdiçados
# ---------------------------------------------------------------------------

def test_infra_waste_lines_shows_wasted_resources():
    infra_fail_with_resources = {
        "grid_experiment_idx": 0,
        "grid_params": {},
        "status": "failed",
        "error_type": "InfraError",
        "resources": {"train_time_sec": 45.0, "energy_kwh": 0.005, "cost_usd": 0.002},
    }
    lines = _build_infra_waste_lines([infra_fail_with_resources])

    assert any("45.00s" in l for l in lines)
    assert any("0.005000" in l for l in lines)
    assert any("0.002000" in l for l in lines)


def test_infra_waste_lines_omits_unavailable_resources():
    """Quando o experimento falhou antes de medir recursos, omite linhas de KPI."""
    lines = _build_infra_waste_lines([_infra_fail(0)])

    assert any("Desperdício infra" in l for l in lines)
    assert not any("Tempo" in l for l in lines)
    assert not any("Energia" in l for l in lines)


# ---------------------------------------------------------------------------
# executor — error_type no dict de falha (integração leve)
# ---------------------------------------------------------------------------

def test_executor_failed_result_includes_error_type():
    """Verifica que o campo error_type é classificado corretamente no resultado de falha."""
    from experiment.workflow import classify_error_type

    exc = TimeoutError("hub timeout")
    assert classify_error_type(exc) == "InfraError"

    exc2 = ValueError("config inválida")
    assert classify_error_type(exc2) == "ValueError"
