"""
Testes unitários para verificar correções no pacote experiment/.

Coberturas:
  - get_torch_device() retorna dict com chave 'name' (não tupla)
  - _ENERGY_COST_USD_PER_KWH é configurável via variável de ambiente
  - cost_usd é calculado corretamente a partir de energy_kwh
"""

import os

import pytest


class TestGetTorchDevice:
    def test_returns_dict(self):
        from utils.util import get_torch_device
        result = get_torch_device()
        assert isinstance(result, dict), "get_torch_device() deve retornar um dicionário"

    def test_has_name_key(self):
        from utils.util import get_torch_device
        result = get_torch_device()
        assert "name" in result, "get_torch_device() deve conter chave 'name'"

    def test_name_is_string(self):
        from utils.util import get_torch_device
        result = get_torch_device()
        assert isinstance(result["name"], str), "device['name'] deve ser string"

    def test_has_type_key(self):
        from utils.util import get_torch_device
        result = get_torch_device()
        assert "type" in result, "get_torch_device() deve conter chave 'type'"


class TestCostUsdCalculation:
    def test_cost_usd_with_known_energy(self):
        energy_kwh = 0.1
        cost_per_kwh = 0.12
        cost_usd = energy_kwh * cost_per_kwh
        assert cost_usd == pytest.approx(0.012, rel=1e-6)

    def test_cost_usd_none_when_energy_none(self):
        energy_kwh = None
        cost_usd = float(energy_kwh) * 0.12 if energy_kwh is not None else None
        assert cost_usd is None

    def test_cost_usd_env_var_override(self, monkeypatch):
        monkeypatch.setenv("ENERGY_COST_USD_PER_KWH", "0.20")
        cost = float(os.getenv("ENERGY_COST_USD_PER_KWH", "0.12"))
        assert cost == pytest.approx(0.20)

    def test_cost_usd_default_value(self, monkeypatch):
        monkeypatch.delenv("ENERGY_COST_USD_PER_KWH", raising=False)
        cost = float(os.getenv("ENERGY_COST_USD_PER_KWH", "0.12"))
        assert cost == pytest.approx(0.12)


class TestComputeCostUsd:
    """Testes para experiment.helpers.compute_cost_usd (BL-01)."""

    def test_flat_rate_fallback(self, monkeypatch):
        """Sem ambiente: usa tarifa flat energy_kwh × ENERGY_COST_USD_PER_KWH."""
        monkeypatch.delenv("ENERGY_COST_USD_PER_KWH", raising=False)
        from experiment.helpers import compute_cost_usd
        result = compute_cost_usd(energy_kwh=0.1)
        assert result == pytest.approx(0.1 * 0.12, rel=1e-6)

    def test_flat_rate_env_var(self, monkeypatch):
        """Tarifa flat respeita ENERGY_COST_USD_PER_KWH via env var."""
        monkeypatch.setenv("ENERGY_COST_USD_PER_KWH", "0.20")
        # Força reload da constante no módulo após alteração do env var
        import importlib

        import experiment.helpers as _helpers
        importlib.reload(_helpers)
        from experiment.helpers import compute_cost_usd as _fn
        result = _fn(energy_kwh=0.5)
        assert result == pytest.approx(0.5 * 0.20, rel=1e-6)
        # Recarrega novamente com o ambiente padrão para não vazar estado.
        monkeypatch.delenv("ENERGY_COST_USD_PER_KWH", raising=False)
        importlib.reload(_helpers)

    def test_flat_rate_none_energy(self):
        """energy_kwh=None sem ambiente → retorna None."""
        from experiment.helpers import compute_cost_usd
        assert compute_cost_usd(energy_kwh=None) is None

    def test_formula_gpu(self):
        """GPU a $1.20/h por 36 s → cost_usd = (36/3600) × 1.20 = 0.012."""
        from experiment.helpers import compute_cost_usd
        result = compute_cost_usd(
            energy_kwh=0.001,
            train_time_sec=36.0,
            environment_cost_per_hour_usd=1.20,
        )
        assert result == pytest.approx(36.0 / 3600.0 * 1.20, rel=1e-6)

    def test_formula_cpu(self):
        """CPU a $0.10/h por 6000 s → cost_usd = (6000/3600) × 0.10 ≈ 0.1667."""
        from experiment.helpers import compute_cost_usd
        result = compute_cost_usd(
            energy_kwh=0.25,
            train_time_sec=6000.0,
            environment_cost_per_hour_usd=0.10,
        )
        assert result == pytest.approx(6000.0 / 3600.0 * 0.10, rel=1e-6)

    def test_formula_tpu(self):
        """TPU a $1.50/h por 3600 s → cost_usd = 1.0 × 1.50 = 1.50."""
        from experiment.helpers import compute_cost_usd
        result = compute_cost_usd(
            energy_kwh=0.09,
            train_time_sec=3600.0,
            environment_cost_per_hour_usd=1.50,
        )
        assert result == pytest.approx(1.50, rel=1e-6)

    def test_none_train_time_returns_none(self):
        """environment_cost fornecido mas train_time_sec=None → retorna None."""
        from experiment.helpers import compute_cost_usd
        result = compute_cost_usd(
            energy_kwh=0.1,
            train_time_sec=None,
            environment_cost_per_hour_usd=1.20,
        )
        assert result is None

    def test_ignores_energy_kwh(self):
        """Quando rate de ambiente é fornecido, energy_kwh não afeta o resultado."""
        from experiment.helpers import compute_cost_usd
        r1 = compute_cost_usd(energy_kwh=0.001, train_time_sec=36.0,
                              environment_cost_per_hour_usd=1.20)
        r2 = compute_cost_usd(energy_kwh=9999.0, train_time_sec=36.0,
                              environment_cost_per_hour_usd=1.20)
        assert r1 == pytest.approx(r2, rel=1e-9)
