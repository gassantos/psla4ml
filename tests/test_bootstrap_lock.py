"""Testes do BootstrapLock — serialização de init entre workers (Item 8)."""

import threading
import time

import pytest

from experiment.bootstrap_lock import BootstrapLock

# ---------------------------------------------------------------------------
# Uso básico
# ---------------------------------------------------------------------------

def test_bootstrap_lock_acquires_and_releases(tmp_path):
    lock_file = tmp_path / "test.lock"
    with BootstrapLock(lock_file=lock_file):
        assert lock_file.exists()


def test_bootstrap_lock_file_created_in_missing_parent(tmp_path):
    lock_file = tmp_path / "subdir" / "nested" / "test.lock"
    with BootstrapLock(lock_file=lock_file):
        assert lock_file.exists()


def test_bootstrap_lock_timeout_raises_timeout_error(tmp_path):
    """Timeout expira como TimeoutError (→ InfraError pelo classify_error_type)."""
    lock_file = tmp_path / "timeout.lock"
    from filelock import FileLock
    outer = FileLock(str(lock_file))
    outer.acquire()
    try:
        with pytest.raises(TimeoutError, match="Bootstrap lock não adquirido"):  # noqa: SIM117
            with BootstrapLock(lock_file=lock_file, timeout=0.1):
                pass
    finally:
        outer.release()


def test_bootstrap_lock_timeout_error_is_classified_as_infra_error(tmp_path):
    """TimeoutError do lock é classificado como InfraError pelo classify_error_type."""
    from experiment.workflow import classify_error_type

    exc = TimeoutError("Bootstrap lock não adquirido em 0.1s")
    assert classify_error_type(exc) == "InfraError"


# ---------------------------------------------------------------------------
# Serialização entre threads (proxy de processo)
# ---------------------------------------------------------------------------

def test_bootstrap_lock_serializes_concurrent_threads(tmp_path):
    """Apenas uma thread por vez executa o bloco protegido."""
    lock_file = tmp_path / "concurrent.lock"
    concurrency_log: list[int] = []
    errors: list[Exception] = []

    def _worker(worker_id: int) -> None:
        try:
            with BootstrapLock(lock_file=lock_file, timeout=10):
                concurrency_log.append(1)
                time.sleep(0.05)
                concurrency_log.append(-1)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # O máximo de workers simultâneos dentro do bloco deve ser 1.
    # Verificamos via log de entrada/saída: nunca dois +1 sem -1 entre eles.
    active = 0
    for delta in concurrency_log:
        active += delta
        assert active <= 1, f"Concorrência detectada: {concurrency_log}"


def test_bootstrap_lock_released_on_exception(tmp_path):
    """Lock é liberado mesmo quando o bloco interno lança exceção."""
    lock_file = tmp_path / "exc.lock"

    with pytest.raises(RuntimeError), BootstrapLock(lock_file=lock_file):
        raise RuntimeError("falha interna")

    # Deve ser possível adquirir novamente sem timeout
    with BootstrapLock(lock_file=lock_file, timeout=1):
        pass
