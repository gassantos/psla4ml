"""
Módulo para detecção automática de dispositivo de computação.
Suporta CUDA (NVIDIA), TPU (TorchXLA), MPS (Apple Silicon) e CPU.
"""
from __future__ import annotations

import contextlib
import logging
import os
import platform
import sys

import psutil
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_nvml_stderr():
    """Suprime a mensagem C-level do NVML no stderr durante detecção de GPU.

    Em ambientes CPU-only (ex.: Google Colab sem runtime de GPU), o NVML
    escreve diretamente no file-descriptor 2 (stderr) a mensagem::

        gpuGetDeviceCount failed with code 35

    Essa mensagem é gerada em nível C, não capturável via ``logging`` do
    Python. O context manager redireciona o FD-2 para ``/dev/null`` apenas
    durante a chamada, restaurando-o imediatamente depois.

    Seguro em Linux/macOS. Em Windows não faz nada (NVML geralmente não
    emite essa mensagem no stderr nessa plataforma).
    """
    if sys.platform == "win32":
        yield
        return
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)

# ---------------------------------------------------------------------------
# XLA availability — exposto como atributo de módulo para permitir mocking
# ---------------------------------------------------------------------------
_XLA_IMPORT_ERROR: str | None = None
try:
    import torch_xla.core.xla_model as xm  # type: ignore[import]
    import torch_xla.distributed.parallel_loader as xla_parallel_loader  # type: ignore[import]
    import torch_xla.runtime as xr  # type: ignore[import]
    _XLA_AVAILABLE = True
except ImportError as _exc:
    xm = None  # type: ignore[assignment]
    xla_parallel_loader = None  # type: ignore[assignment]
    xr = None  # type: ignore[assignment]
    _XLA_AVAILABLE = False
    _XLA_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


def describe_tpu_unavailable() -> str:
    """Explica por que o runtime TPU/XLA não foi detectado (para mensagens de erro)."""
    if not _XLA_AVAILABLE:
        return f"torch_xla não pôde ser importado ({_XLA_IMPORT_ERROR})"
    return (
        "torch_xla foi importado, mas nenhum device XLA foi encontrado "
        "(verifique PJRT_DEVICE=TPU e o runtime TPU do Colab)"
    )


def _shard_data_loader(data_loader):
    if xr is None or xr.world_size() <= 1:
        return data_loader, None

    sampler = DistributedSampler(
        data_loader.dataset,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
        shuffle=isinstance(data_loader.sampler, RandomSampler),
        drop_last=data_loader.drop_last,
    )
    sharded_loader = DataLoader(
        dataset=data_loader.dataset,
        batch_size=data_loader.batch_size,
        sampler=sampler,
        num_workers=data_loader.num_workers,
        collate_fn=data_loader.collate_fn,
        drop_last=data_loader.drop_last,
        pin_memory=data_loader.pin_memory,
        timeout=data_loader.timeout,
        worker_init_fn=data_loader.worker_init_fn,
        persistent_workers=data_loader.persistent_workers,
    )
    return sharded_loader, sampler


def prepare_data_loader(data_loader, device):
    """Prepara um DataLoader para transferência eficiente ao dispositivo XLA."""
    if device is None or device.type != "xla":
        return data_loader
    if xla_parallel_loader is None:
        raise RuntimeError("Dispositivo XLA selecionado, mas MpDeviceLoader não está disponível.")
    data_loader, sampler = _shard_data_loader(data_loader)
    device_loader = xla_parallel_loader.MpDeviceLoader(data_loader, device)
    if sampler is not None:
        device_loader._xla_distributed_sampler = sampler
    return device_loader


def set_data_loader_epoch(data_loader, epoch):
    """Atualiza o embaralhamento determinístico do sampler distribuído XLA."""
    sampler = getattr(data_loader, "_xla_distributed_sampler", None)
    if sampler is not None:
        sampler.set_epoch(epoch)


def move_batch_to_device(batch, device):
    """Move os tensores de um batch para o dispositivo selecionado."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def validate_xla_batch_shape(batch, expected_signature=None):
    """Valida que os tensores dos batches XLA mantêm formas e tipos estáticos."""
    signature = tuple(
        sorted(
            (key, tuple(value.shape), str(value.dtype))
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        )
    )
    if expected_signature is not None and signature != expected_signature:
        raise RuntimeError(
            "Forma dinâmica detectada no treino XLA. Use padding até max_seq_length, "
            "limites fixos de parágrafos e drop_last=True."
        )
    return signature


def get_device(prefer_cpu: bool = False):
    """
    Detecta o melhor dispositivo disponível de forma multiplataforma.

    Args:
        prefer_cpu: Se True, força uso de CPU mesmo com GPU disponível

    Returns:
        torch.device: Device otimizado para a plataforma atual, ou None se torch indisponível
    """
    if torch is None:
        logger.warning("PyTorch não disponível. Retornando device=None.")
        return None

    if prefer_cpu:
        logger.info("CPU mode forced by user")
        return torch.device("cpu")

    system = platform.system()

    # Windows/Linux com CUDA
    with _suppress_nvml_stderr():
        cuda_available = torch.cuda.is_available()

    if cuda_available:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using CUDA GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        return device

    # Linux com TPU via XLA
    if sys.platform == "linux" and _XLA_AVAILABLE and xm is not None:
        logger.info("Using TPU via Torch_XLA")
        return xm.xla_device()

    if sys.platform == "linux" and not _XLA_AVAILABLE:
        logger.warning("Torch_XLA não disponível, usando CPU fallback")

    # macOS com Apple Silicon (MPS)
    if system == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
        return device

    # CPU fallback
    device = torch.device("cpu")
    logger.info(f"Using CPU on {system} system")
    if system == "Darwin":
        logger.warning("MPS not available. Ensure PyTorch version supports Apple Silicon.")
    elif system in ["Windows", "Linux"]:
        logger.warning("CUDA not available. Install CUDA toolkit for GPU acceleration.")
    return device


def get_device_info():
    """
    Retorna informações detalhadas sobre o dispositivo.

    Returns:
        dict: Informações sobre dispositivo, memória e capacidade
    """
    if torch is None:
        return {
            "device_type": "unavailable",
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "pytorch_version": None,
            "error": "PyTorch não disponível neste ambiente"
        }
    device = get_device()
    assert device is not None  # torch is not None, portanto get_device() sempre retorna um device
    info = {
        "device_type": device.type,
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
    }

    if device.type == "cuda":
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        })
    elif device.type == "mps":
        info.update({
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        })

    return info


def set_device_optimization(device):
    """
    Configura otimizações específicas do dispositivo.

    Args:
        device: torch.device para otimizar
    """
    if torch is None or device is None:
        logger.warning("PyTorch não disponível. Otimizações de device ignoradas.")
        return
    if device.type == "cuda":
        # Habilita TF32 para Ampere GPUs (melhor performance)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("CUDA optimizations enabled (TF32)")
    elif device.type == "mps":
        # MPS ainda é experimental, sem otimizações específicas por enquanto
        logger.info("MPS device set (experimental support)")
    else:
        logger.info("CPU mode - no specific optimizations applied")


def get_torch_device() -> dict:
    """Retorna o dispositivo PyTorch disponível (CPU, GPU ou TPU)."""
    if _XLA_AVAILABLE and xm is not None:
        if len(xm.get_xla_supported_devices()) > 0:
            return {
                'type': 'TPU',
                'name': xm.xla_device_kind(),
                'device': xm.xla_device()
            }

    if torch is not None:
        with _suppress_nvml_stderr():
            cuda_available = torch.cuda.is_available()
        if cuda_available:
            return {
                'type': 'GPU',
                'name': torch.cuda.get_device_name(0),
                'device': torch.device('cuda')
            }
    if torch is not None:
        return {
            'type': 'CPU',
            'name': platform.processor(),
            'device': torch.device('cpu')
        }
    return {'type': 'unavailable', 'name': None, 'device': None}


def set_cpu_parallelism():
    """Otimiza o paralelismo para execução em CPU.

    Define ``torch.set_num_threads`` para o número de cores físicos (não lógicos)
    para evitar overhead de context switch e contenção de cache L3, o que é
    uma melhor prática para modelos Transformer em CPU.
    Também habilita o OneDNN (MKL-DNN) para aceleração de kernels.
    """
    try:
        # Núcleos físicos são preferíveis para deep learning (evita hyperthreading noise)
        phy_cores = psutil.cpu_count(logical=False) or 1
        torch.set_num_threads(phy_cores)

        # Habilita OneDNN se disponível (aceleração para Intel CPUs)
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True

        logger.info(f"Otimizações de CPU aplicadas: threads={phy_cores}, oneDNN=True")
    except Exception as e:
        logger.warning(f"Não foi possível aplicar otimizações de afinidade de CPU: {e}")