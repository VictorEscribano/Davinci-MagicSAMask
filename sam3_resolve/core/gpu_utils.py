"""GPU detection, VRAM accounting, and device selection utilities."""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Backend(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass(frozen=True)
class GPUInfo:
    backend: Backend
    device_name: str
    vram_gb: float
    driver_version: str
    cuda_version: str  # empty string for non-CUDA backends


def _cuda_driver_version() -> str:
    """Return CUDA driver version string from nvidia-smi, or empty string."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        )
        return out.strip().splitlines()[0]
    except Exception:
        return ""


def detect_gpu() -> GPUInfo:
    """
    Probe the available compute backend and return a GPUInfo descriptor.

    Returns:
        GPUInfo with backend, device name, VRAM, and version strings.

    Detection order:
        1. CUDA (NVIDIA)
        2. MPS  (Apple Silicon)
        3. CPU  fallback
    """
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("torch not installed; falling back to CPU info")
        return GPUInfo(
            backend=Backend.CPU,
            device_name=platform.processor() or "CPU",
            vram_gb=0.0,
            driver_version="",
            cuda_version="",
        )

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        cuda_ver = torch.version.cuda or ""
        driver_ver = _cuda_driver_version()
        logger.info(
            "CUDA device detected: %s · %.1f GB VRAM · CUDA %s",
            props.name,
            vram_gb,
            cuda_ver,
        )
        return GPUInfo(
            backend=Backend.CUDA,
            device_name=props.name,
            vram_gb=round(vram_gb, 1),
            driver_version=driver_ver,
            cuda_version=cuda_ver,
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS backend detected")
        return GPUInfo(
            backend=Backend.MPS,
            device_name="Apple Silicon",
            vram_gb=0.0,  # shared memory; not queryable
            driver_version="",
            cuda_version="",
        )

    logger.warning("No GPU detected; using CPU (expect slow inference)")
    return GPUInfo(
        backend=Backend.CPU,
        device_name=platform.processor() or "CPU",
        vram_gb=0.0,
        driver_version="",
        cuda_version="",
    )


def recommended_dtype(backend: Backend) -> str:
    """
    Return the torch dtype string appropriate for SAM3 inference.

    Args:
        backend: The active compute backend.

    Returns:
        One of 'float16', 'bfloat16', 'float32'.
    """
    mapping = {
        Backend.CUDA: "float16",
        Backend.MPS: "bfloat16",
        Backend.CPU: "float32",
    }
    return mapping[backend]


def vram_sufficient_for_large(gpu: GPUInfo) -> bool:
    """Return True if the GPU has enough VRAM for SAM3-Large (8 GB threshold)."""
    return gpu.backend == Backend.CUDA and gpu.vram_gb >= 8.0


def torch_device_string(backend: Backend) -> str:
    """
    Return the torch device string for the given backend.

    Args:
        backend: Detected backend enum value.

    Returns:
        E.g. 'cuda:0', 'mps', or 'cpu'.
    """
    mapping = {
        Backend.CUDA: "cuda:0",
        Backend.MPS: "mps",
        Backend.CPU: "cpu",
    }
    return mapping[backend]


def release_gpu_memory() -> None:
    """Free unused CUDA memory. Safe to call on non-CUDA systems."""
    try:
        import torch  # type: ignore[import-untyped]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    except ImportError:
        pass


def estimate_cpu_time_minutes(total_frames: int) -> float:
    """
    Rough estimate of SAM3 CPU propagation time.

    Uses a conservative constant of 8 seconds per frame on a modern CPU.

    Args:
        total_frames: Number of frames to process.

    Returns:
        Estimated processing time in minutes.
    """
    SECONDS_PER_FRAME_CPU = 8.0
    return (total_frames * SECONDS_PER_FRAME_CPU) / 60.0
