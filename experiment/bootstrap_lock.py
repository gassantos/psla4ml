"""Lock de disco para serializar o bootstrap de ativos HuggingFace entre workers."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from utils.paths import PathManager

_DEFAULT_LOCK_FILE = PathManager.HF_HUB_CACHE_DIR / ".psla4ml_init.lock"
_DEFAULT_TIMEOUT_SEC: float = 120.0


class BootstrapLock:
    """Serializa o bootstrap de tokenizer/modelo entre workers concorrentes.

    Usa filelock (disponível via huggingface_hub) para garantir que apenas um
    worker por vez execute init_all durante o cold start. Com cache ativo, o
    lock é adquirido e liberado em microssegundos, sem impacto no throughput.

    Em caso de timeout eleva TimeoutError — classificado como InfraError pelo
    executor, ativando o retry seletivo do item 4.
    """

    def __init__(
        self,
        lock_file: Path | str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self._lock_file = Path(lock_file) if lock_file else _DEFAULT_LOCK_FILE
        self._timeout = timeout
        self._lock: object = None

    def __enter__(self) -> Self:
        from filelock import (  # lazy: evita import em contextos sem filelock
            FileLock,
            Timeout,
        )

        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self._lock_file), timeout=self._timeout)
        try:
            lock.acquire()
        except Timeout as exc:
            raise TimeoutError(
                f"Bootstrap lock não adquirido em {self._timeout}s — "
                f"arquivo: {self._lock_file}"
            ) from exc
        self._lock = lock
        return self

    def __exit__(self, *_: object) -> None:
        if self._lock is not None:
            self._lock.release()
            self._lock = None
