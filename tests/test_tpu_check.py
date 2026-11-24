"""
Testes para experiment.tpu_check — BL-08.

Coberturas:
  - TpuAccelerationStatus: dataclass, campos padrão, to_dict
  - _xla_available: retorna bool
  - _get_gpu_energy_from_tracker: extrai energia de tracker mock, None quando ausente
  - check_tpu_acceleration:
    - Dispositivo não-TPU → retorna sem aviso
    - TPU com gpu_energy > 0 → acelerador ativo, sem aviso
    - TPU com gpu_energy ≈ 0 → acelerador inativo, gera aviso e recomendações
    - TPU sem tracker (gpu_energy=None) → usa xla_available como fallback
    - Campos corretos no resultado serializado
  - build_result_dict com tpu_check:
    - tpu_acceleration_check presente no resultado
    - warnings preenchido quando há problema
    - backward compat: tpu_check=None não altera o resultado
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mocks de tracker CodeCarbon
# ---------------------------------------------------------------------------

def _mock_tracker(gpu_energy_kwh: None | float) -> Any:
    """Cria um mock do EmissionsTracker com final_emissions_data.gpu_energy."""
    tracker = MagicMock()
    if gpu_energy_kwh is None:
        tracker.final_emissions_data = None
    else:
        emissions_data = MagicMock()
        emissions_data.gpu_energy = gpu_energy_kwh
        tracker.final_emissions_data = emissions_data
    return tracker


# ---------------------------------------------------------------------------
# TpuAccelerationStatus
# ---------------------------------------------------------------------------

class TestTpuAccelerationStatus:
    def test_import(self):
        from experiment.tpu_check import TpuAccelerationStatus
        assert TpuAccelerationStatus is not None

    def test_default_construction(self):
        from experiment.tpu_check import TpuAccelerationStatus
        s = TpuAccelerationStatus()
        assert s.device_type == "CPU"
        assert s.is_tpu_environment is False
        assert s.xla_available is False
        assert s.gpu_energy_kwh is None
        assert s.accelerator_active is True
        assert s.warning is None
        assert s.recommendations == []

    def test_to_dict_keys(self):
        from experiment.tpu_check import TpuAccelerationStatus
        d = TpuAccelerationStatus().to_dict()
        expected = {
            "device_type", "is_tpu_environment", "xla_available",
            "gpu_energy_kwh", "xla_runtime_metrics", "accelerator_active",
            "warning", "recommendations",
        }
        assert set(d.keys()) == expected

    def test_to_dict_values(self):
        from experiment.tpu_check import TpuAccelerationStatus
        s = TpuAccelerationStatus(device_type="TPU", is_tpu_environment=True,
                                  accelerator_active=False,
                                  warning="test warning",
                                  recommendations=["step 1"])
        d = s.to_dict()
        assert d["device_type"] == "TPU"
        assert d["is_tpu_environment"] is True
        assert d["accelerator_active"] is False
        assert d["warning"] == "test warning"
        assert d["recommendations"] == ["step 1"]


# ---------------------------------------------------------------------------
# _xla_available
# ---------------------------------------------------------------------------

class TestXlaAvailable:
    def test_returns_bool(self):
        from experiment.tpu_check import _xla_available
        result = _xla_available()
        assert isinstance(result, bool)

    def test_false_when_torch_xla_not_installed(self, monkeypatch):
        """Em ambiente Windows/CPU sem torch_xla, deve retornar False."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "torch_xla" in name:
                raise ImportError("torch_xla not available")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from experiment import tpu_check as tc
        result = tc._xla_available()
        assert result is False


class TestXlaRuntimeMetrics:
        def test_collects_compile_execute_and_mark_step_counts(self, monkeypatch):
                from experiment import tpu_check

                report = """
Metric: CompileTime
    TotalSamples: 2
Metric: ExecuteTime
    TotalSamples: 12
Counter: MarkStep
    Value: 10
"""
                metrics_module = MagicMock()
                metrics_module.metrics_report.return_value = report
                monkeypatch.setattr(tpu_check.importlib, "import_module", lambda _: metrics_module)

                result = tpu_check.collect_xla_runtime_metrics()

                assert result["available"] is True
                assert result["compile_count"] == 2
                assert result["execute_count"] == 12
                assert result["mark_step_count"] == 10

        def test_returns_unavailable_when_xla_metrics_cannot_be_imported(self, monkeypatch):
                from experiment import tpu_check

                def raise_import_error(_):
                        raise ImportError("torch_xla unavailable")

                monkeypatch.setattr(tpu_check.importlib, "import_module", raise_import_error)

                result = tpu_check.collect_xla_runtime_metrics()

                assert result["available"] is False
                assert result["execute_count"] == 0


# ---------------------------------------------------------------------------
# _get_gpu_energy_from_tracker
# ---------------------------------------------------------------------------

class TestGetGpuEnergyFromTracker:
    def test_returns_none_when_tracker_is_none(self):
        from experiment.tpu_check import _get_gpu_energy_from_tracker
        assert _get_gpu_energy_from_tracker(None) is None

    def test_returns_gpu_energy_from_final_data(self):
        from experiment.tpu_check import _get_gpu_energy_from_tracker
        tracker = _mock_tracker(gpu_energy_kwh=0.05)
        result = _get_gpu_energy_from_tracker(tracker)
        assert result == pytest.approx(0.05)

    def test_returns_zero_when_no_gpu_energy(self):
        from experiment.tpu_check import _get_gpu_energy_from_tracker
        tracker = _mock_tracker(gpu_energy_kwh=0.0)
        result = _get_gpu_energy_from_tracker(tracker)
        assert result == pytest.approx(0.0)

    def test_returns_none_when_final_data_is_none(self):
        from experiment.tpu_check import _get_gpu_energy_from_tracker
        tracker = _mock_tracker(gpu_energy_kwh=None)
        assert _get_gpu_energy_from_tracker(tracker) is None

    def test_handles_attribute_error_gracefully(self):
        from experiment.tpu_check import _get_gpu_energy_from_tracker
        bad_tracker = MagicMock(spec=[])  # sem atributos
        result = _get_gpu_energy_from_tracker(bad_tracker)
        assert result is None


# ---------------------------------------------------------------------------
# check_tpu_acceleration — casos principais
# ---------------------------------------------------------------------------

class TestCheckTpuAcceleration:
    def test_non_tpu_device_returns_no_warning(self):
        from experiment.tpu_check import check_tpu_acceleration
        status = check_tpu_acceleration("GPU")
        assert status.is_tpu_environment is False
        assert status.accelerator_active is True
        assert status.warning is None

    def test_cpu_device_returns_no_warning(self):
        from experiment.tpu_check import check_tpu_acceleration
        status = check_tpu_acceleration("CPU")
        assert status.warning is None
        assert status.accelerator_active is True

    def test_tpu_with_xla_execution_no_warning(self, monkeypatch):
        """ExecuteTime > 0 comprova XLA mesmo quando energia não é conclusiva."""
        from experiment import tpu_check
        monkeypatch.setattr(tpu_check, "_xla_available", lambda: True)
        tracker = _mock_tracker(gpu_energy_kwh=0.09)
        status = tpu_check.check_tpu_acceleration(
            "TPU",
            tracker=tracker,
            exec_time_sec=3600.0,
            xla_runtime_metrics={"available": True, "compile_count": 1, "execute_count": 10},
        )
        assert status.is_tpu_environment is True
        assert status.accelerator_active is True
        assert status.warning is None
        assert status.gpu_energy_kwh == pytest.approx(0.09)

    def test_tpu_with_zero_gpu_energy_generates_warning(self):
        """TPU com gpu_energy ≈ 0 → execução na host CPU detectada (cenário do artigo)."""
        from experiment.tpu_check import check_tpu_acceleration
        tracker = _mock_tracker(gpu_energy_kwh=0.0)
        status = check_tpu_acceleration("TPU", tracker=tracker, exec_time_sec=3634.0)
        assert status.is_tpu_environment is True
        assert status.accelerator_active is False
        assert status.warning is not None
        assert "BL-08" in status.warning
        assert "runtime XLA" in status.warning

    def test_positive_energy_without_xla_execution_is_not_active(self, monkeypatch):
        from experiment import tpu_check
        monkeypatch.setattr(tpu_check, "_xla_available", lambda: True)

        status = tpu_check.check_tpu_acceleration(
            "TPU",
            tracker=_mock_tracker(gpu_energy_kwh=0.5),
            xla_runtime_metrics={"available": True, "compile_count": 0, "execute_count": 0},
        )

        assert status.gpu_energy_kwh == pytest.approx(0.5)
        assert status.accelerator_active is False

    def test_zero_energy_with_xla_execution_is_active(self, monkeypatch):
        from experiment import tpu_check
        monkeypatch.setattr(tpu_check, "_xla_available", lambda: True)

        status = tpu_check.check_tpu_acceleration(
            "TPU",
            tracker=_mock_tracker(gpu_energy_kwh=0.0),
            xla_runtime_metrics={"available": True, "compile_count": 1, "execute_count": 5},
        )

        assert status.gpu_energy_kwh == pytest.approx(0.0)
        assert status.accelerator_active is True

    def test_tpu_warning_contains_exec_time(self):
        from experiment.tpu_check import check_tpu_acceleration
        tracker = _mock_tracker(gpu_energy_kwh=0.0)
        status = check_tpu_acceleration("TPU", tracker=tracker, exec_time_sec=3634.15)
        # Verifica que o exec_time aparece no aviso (arredondado para 1 casa decimal)
        assert "3634" in status.warning # type: ignore

    def test_tpu_warning_has_recommendations(self):
        from experiment.tpu_check import check_tpu_acceleration
        tracker = _mock_tracker(gpu_energy_kwh=0.0)
        status = check_tpu_acceleration("TPU", tracker=tracker)
        assert len(status.recommendations) > 0
        # Deve mencionar torch_xla e padding estático
        recs = " ".join(status.recommendations)
        assert "torch_xla" in recs
        assert "padding" in recs.lower()

    def test_tpu_without_tracker_uses_xla_fallback(self):
        """Sem tracker (MonitoringDisabled), usa xla_available como fallback."""
        from experiment.tpu_check import check_tpu_acceleration
        # Quando tracker=None e device=TPU, gpu_energy=None
        # Se XLA não está disponível (ambiente Windows) → accelerator_active=False
        # O warning só é gerado se accelerator_active=False
        status = check_tpu_acceleration("TPU", tracker=None)
        assert status.is_tpu_environment is True
        # gpu_energy_kwh é None pois não há tracker
        assert status.gpu_energy_kwh is None
        # accelerator_active depende de xla_available — não testamos o valor exato
        assert isinstance(status.accelerator_active, bool)
        # Se warning existe, deve ser serializável
        if status.warning:
            import json
            json.dumps(status.to_dict())  # não lança

    def test_device_type_case_insensitive(self):
        """'tpu', 'TPU', 'Tpu' devem todos ser reconhecidos."""
        from experiment.tpu_check import check_tpu_acceleration
        for dt in ["tpu", "TPU", "Tpu"]:
            status = check_tpu_acceleration(dt, tracker=_mock_tracker(0.0))
            assert status.is_tpu_environment is True

    def test_result_is_serializable(self):
        """TpuAccelerationStatus.to_dict() deve ser JSON-serializável."""
        import json

        from experiment.tpu_check import check_tpu_acceleration
        tracker = _mock_tracker(gpu_energy_kwh=0.0)
        status = check_tpu_acceleration("TPU", tracker=tracker, exec_time_sec=100.0)
        d = status.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_paper_scenario_reproduction(self):
        """Reproduz o cenário da Seção 4: TPU com gpu_power = 0.0 W."""
        from experiment.tpu_check import check_tpu_acceleration
        # Simula os dados reportados no artigo:
        # exec_time ≈ 3634s, gpu_power = 0.0 W
        tracker = _mock_tracker(gpu_energy_kwh=0.0)
        status = check_tpu_acceleration("TPU", tracker=tracker, exec_time_sec=3634.15)

        assert status.is_tpu_environment is True
        assert status.accelerator_active is False
        assert status.warning is not None
        assert len(status.recommendations) >= 4  # pelo menos 4 passos de correção


# ---------------------------------------------------------------------------
# Integração: build_result_dict com tpu_check
# ---------------------------------------------------------------------------

class TestBuildResultDictWithTpuCheck:
    def _base_kwargs(self):
        return {
            "experiment_id": "test-uuid",
            "json_filename": "test.json",
            "seed": 42,
            "status": "success",
            "date_exec": "20260715_120000",
            "start_iso": "2026-07-15T12:00:00+00:00",
            "end_iso": "2026-07-15T13:00:00+00:00",
            "device_type": "CPU",
            "device_name": "AMD EPYC",
            "precision": "fp16",
            "parallel_workers": 1,
            "train_dataset_name": "train_task2",
            "optimizer": "adam",
            "learning_rate": 1e-5,
            "avg_gflops_per_batch": 100.0,
            "batch_size": 8,
            "epoch": 3,
            "exec_time": 100.0,
            "energy_kwh": 0.01,
            "emissions_kg": 0.001,
            "cost_usd": 0.001,
            "avg_ram": 4096.0,
            "peak_ram": 5120.0,
            "total_gflops": 1000.0,
            "eval_metrics": {},
            "stdout": "output",
            "stderr": "",
        }

    def test_without_tpu_check_no_extra_key(self):
        from experiment.persistence import build_result_dict
        result = build_result_dict(**self._base_kwargs())
        assert "tpu_acceleration_check" not in result

    def test_with_tpu_check_none_no_extra_key(self):
        from experiment.persistence import build_result_dict
        result = build_result_dict(**self._base_kwargs(), tpu_check=None)
        assert "tpu_acceleration_check" not in result

    def test_with_active_tpu_check_adds_key(self):
        from experiment.persistence import build_result_dict
        from experiment.tpu_check import TpuAccelerationStatus
        tpu = TpuAccelerationStatus(device_type="TPU", is_tpu_environment=True,
                                    accelerator_active=True)
        result = build_result_dict(**self._base_kwargs(), tpu_check=tpu)
        assert "tpu_acceleration_check" in result
        assert result["tpu_acceleration_check"]["device_type"] == "TPU"

    def test_with_failed_tpu_check_adds_warning(self):
        from experiment.persistence import build_result_dict
        from experiment.tpu_check import TpuAccelerationStatus
        tpu = TpuAccelerationStatus(
            device_type="TPU",
            is_tpu_environment=True,
            accelerator_active=False,
            warning="TPU acelerador NÃO ativo",
            recommendations=["instale torch_xla"],
        )
        result = build_result_dict(**self._base_kwargs(), tpu_check=tpu)
        assert "tpu_acceleration_check" in result
        assert "warnings" in result
        assert len(result["warnings"]) == 1
        assert "TPU acelerador NÃO ativo" in result["warnings"][0]

    def test_no_warnings_key_when_no_issue(self):
        from experiment.persistence import build_result_dict
        from experiment.tpu_check import TpuAccelerationStatus
        tpu = TpuAccelerationStatus(
            device_type="GPU", is_tpu_environment=False, accelerator_active=True
        )
        result = build_result_dict(**self._base_kwargs(), tpu_check=tpu)
        assert "warnings" not in result


def test_launch_experiment_accepts_zero_tpu_cores(monkeypatch):
    """Regressão: --tpu-cores padrão (0) não deve lançar ValueError."""
    from experiment import xla_launcher

    calls = []
    monkeypatch.setattr(
        "experiment.runner.execute_experiment",
        lambda **kw: calls.append(kw) or {"ok": True},
    )
    assert xla_launcher.launch_experiment(tpu_cores=0, config_path="x") == {"ok": True}
    assert xla_launcher.launch_experiment(tpu_cores=1, config_path="x") == {"ok": True}
    assert len(calls) == 2


def test_launch_experiment_rejects_negative_tpu_cores():
    import pytest
    from experiment import xla_launcher

    with pytest.raises(ValueError):
        xla_launcher.launch_experiment(tpu_cores=-1)


def test_effective_tpu_cores_defaults_to_one_on_tpu():
    import argparse
    from cli.commands import _effective_tpu_cores

    assert _effective_tpu_cores(argparse.Namespace(tpu_cores=0), "TPU") == 1
    assert _effective_tpu_cores(argparse.Namespace(tpu_cores=8), "TPU") == 8
    assert _effective_tpu_cores(argparse.Namespace(tpu_cores=8), "GPU") == 0


def test_tpu_unavailable_error_explains_cause(monkeypatch):
    import argparse
    import pytest
    from cli import commands

    monkeypatch.setattr(commands, "get_torch_device", lambda: {"type": "CPU"})
    monkeypatch.setattr(commands, "describe_tpu_unavailable", lambda: "torch_xla não pôde ser importado (X)")
    with pytest.raises(RuntimeError, match=r"--extra tpu.*PJRT_DEVICE|PJRT_DEVICE.*--extra tpu") as exc:
        commands._ensure_tpu_available(argparse.Namespace(tpu_cores=1))
    assert "torch_xla não pôde ser importado" in str(exc.value)
