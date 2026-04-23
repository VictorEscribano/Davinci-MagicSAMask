#!/usr/bin/env python3
"""
SAM3 Resolve Plugin — one-command installer.

Usage:
    python install.py [--model {large,base}] [--device {auto,cuda,mps,cpu}]

Steps:
    1.  Python version check (≥3.10 required)
    2.  GPU / backend detection
    3.  Create venv at ~/.sam3_resolve_env/
    4.  Install pip dependencies (torch with matching CUDA wheels + extras)
    5.  Download SAM3 model checkpoint with progress, SHA-256 validation,
        and resumable retry logic
    6.  Copy plugin scripts to DaVinci Resolve script folder
    7.  Write config.json with detected settings
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap: resolve the package root so we can import our own constants
# before the venv exists.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from sam3_resolve.constants import (  # noqa: E402
    BASE_DEPS,
    CACHE_DIR,
    CONFIG_PATH,
    DOWNLOAD_CHUNK_BYTES,
    DOWNLOAD_MAX_RETRIES,
    DOWNLOAD_TIMEOUT_S,
    MODELS_DIR,
    PACKAGE_ROOT,
    PLUGIN_VERSION,
    RESOLVE_SCRIPTS_DIR,
    SAM3_BASE_FILENAME,
    SAM3_BASE_SHA256,
    SAM3_BASE_SIZE_BYTES,
    SAM3_BASE_URL,
    SAM3_GITHUB_URL,
    SAM3_LARGE_FILENAME,
    SAM3_LARGE_SHA256,
    SAM3_LARGE_SIZE_BYTES,
    SAM3_LARGE_URL,
    SAM3_MIN_PYTHON,
    TORCH_DEPS_CUDA,
    TORCH_INDEX_CPU,
    VENV_DIR,
    VRAM_THRESHOLD_LARGE_GB,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("install")

# ANSI helpers
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET}  {msg}")


def _err(msg: str) -> None:
    print(f"  {_RED}✗{_RESET}  {msg}")


def _section(title: str) -> None:
    print(f"\n{_BOLD}── {title} {'─' * max(0, 50 - len(title))}{_RESET}")


# ---------------------------------------------------------------------------
# Step 1 — Python version
# ---------------------------------------------------------------------------

def check_python_version() -> None:
    """
    Assert Python is ≥3.10.

    Raises:
        SystemExit: If version is too old.
    """
    _section("Python version check")
    major, minor = sys.version_info[:2]
    if (major, minor) < SAM3_MIN_PYTHON:
        _err(
            f"Python {major}.{minor} detected. "
            f"SAM3 plugin requires Python ≥{SAM3_MIN_PYTHON[0]}.{SAM3_MIN_PYTHON[1]}."
        )
        sys.exit(1)
    _ok(f"Python {major}.{minor} — OK")


# ---------------------------------------------------------------------------
# Step 2 — GPU detection (without importing torch yet)
# ---------------------------------------------------------------------------

def _run_silent(cmd: list[str]) -> tuple[int, str]:
    """Run a subprocess and return (returncode, stdout)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15
        )
        return result.returncode, result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 1, ""


def detect_gpu_pre_torch() -> dict:
    """
    Detect GPU before torch is installed.

    Returns a dict with keys:
        backend (str): 'cuda' | 'mps' | 'cpu'
        cuda_version (str): e.g. '12.1' or ''
        device_name (str)
        vram_gb (float)
    """
    _section("GPU detection")
    system = platform.system()

    # NVIDIA via nvidia-smi
    rc, smi_out = _run_silent(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader,nounits"]
    )
    if rc == 0 and smi_out:
        parts = [p.strip() for p in smi_out.splitlines()[0].split(",")]
        name = parts[0] if len(parts) > 0 else "NVIDIA GPU"
        vram_mib = float(parts[1]) if len(parts) > 1 else 0.0

        # Detect installed CUDA toolkit version
        _, nvcc_out = _run_silent(["nvcc", "--version"])
        cuda_ver = ""
        for line in nvcc_out.splitlines():
            if "release" in line.lower():
                cuda_ver = line.split("release")[-1].strip().split(",")[0].strip()
                break
        if not cuda_ver:
            # Fallback: parse from nvidia-smi log
            _, smi_cuda = _run_silent(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
            )

        vram_gb = round(vram_mib / 1024, 1)
        _ok(f"NVIDIA {name} · {vram_gb} GB VRAM · CUDA {cuda_ver or 'unknown'}")
        return {
            "backend": "cuda",
            "cuda_version": cuda_ver,
            "device_name": name,
            "vram_gb": vram_gb,
        }

    # Apple Silicon
    if system == "Darwin":
        _, uname = _run_silent(["uname", "-m"])
        if "arm" in uname.lower():
            _ok("Apple Silicon — MPS backend will be used")
            return {"backend": "mps", "cuda_version": "", "device_name": "Apple Silicon", "vram_gb": 0.0}

    _warn("No GPU detected. SAM3 will run on CPU — expect slow processing (~8 s/frame).")
    cpu_name = platform.processor() or "CPU"
    return {"backend": "cpu", "cuda_version": "", "device_name": cpu_name, "vram_gb": 0.0}


# ---------------------------------------------------------------------------
# Step 3 — Virtual environment
# ---------------------------------------------------------------------------

def create_venv() -> Path:
    """
    Create the plugin venv at VENV_DIR if it does not already exist.

    Returns:
        Path to the venv's Python binary.

    Raises:
        SystemExit: On creation failure.
    """
    _section("Virtual environment")
    VENV_DIR.mkdir(parents=True, exist_ok=True)

    if (VENV_DIR / "pyvenv.cfg").exists():
        _ok(f"Existing venv found at {VENV_DIR}")
    else:
        _ok(f"Creating venv at {VENV_DIR} …")
        venv.create(str(VENV_DIR), with_pip=True, clear=False, upgrade_deps=True)
        _ok("venv created")

    python_bin = _venv_python()
    _ok(f"venv Python: {python_bin}")
    return python_bin


def _venv_python() -> Path:
    """Return the Python binary path inside the plugin venv."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _venv_pip() -> Path:
    """Return the pip binary path inside the plugin venv."""
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


# ---------------------------------------------------------------------------
# Step 4 — Pip dependencies
# ---------------------------------------------------------------------------

def install_deps(gpu_info: dict) -> None:
    """
    Install all plugin dependencies into the plugin venv.

    Args:
        gpu_info: Dict returned by detect_gpu_pre_torch().

    Raises:
        SystemExit: If pip install fails.
    """
    _section("Installing dependencies")
    pip = str(_venv_pip())

    # Upgrade pip first
    _pip_run([pip, "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Torch — pick wheels matching the detected backend
    backend = gpu_info["backend"]
    if backend == "cuda":
        cuda_ver = gpu_info.get("cuda_version", "")
        index_url = _torch_index_url(cuda_ver)
        _ok(f"Installing torch (CUDA wheels) from {index_url}")
        _pip_run([pip, "install"] + TORCH_DEPS_CUDA + ["--index-url", index_url])
    elif backend == "mps":
        # Apple Silicon: standard PyPI wheels already include MPS support
        _ok("Installing torch (standard wheels, MPS support included)")
        _pip_run([pip, "install"] + TORCH_DEPS_CUDA)
    else:
        # CPU-only: use the dedicated CPU index to avoid downloading CUDA wheels
        _ok(f"Installing torch (CPU-only wheels) from {TORCH_INDEX_CPU}")
        _pip_run([pip, "install"] + TORCH_DEPS_CUDA + ["--index-url", TORCH_INDEX_CPU])

    # SAM3 from GitHub
    _ok("Installing SAM3 from GitHub …")
    _pip_run([pip, "install", SAM3_GITHUB_URL])

    # Remaining deps
    _ok("Installing remaining dependencies …")
    _pip_run([pip, "install"] + BASE_DEPS)

    _ok("All dependencies installed")


def _torch_index_url(cuda_version: str) -> str:
    """
    Map a CUDA toolkit version string to the appropriate PyTorch wheel index.

    Args:
        cuda_version: Version string like '12.1', '11.8', etc.

    Returns:
        PyTorch wheel index URL.
    """
    # Map major.minor → cu tag
    mapping = {
        "12": "cu124",
        "11": "cu118",
    }
    major = cuda_version.split(".")[0] if cuda_version else ""
    tag = mapping.get(major, "cu124")   # default to latest CUDA if unknown
    return f"https://download.pytorch.org/whl/{tag}"


def _pip_run(cmd: list[str]) -> None:
    """
    Execute a pip command, streaming output to stdout.

    Raises:
        SystemExit: On non-zero return code.
    """
    result = subprocess.run(cmd)
    if result.returncode != 0:
        _err(f"pip command failed: {' '.join(cmd)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 5 — Model download
# ---------------------------------------------------------------------------

def download_model(model: str, gpu_info: dict) -> Path:
    """
    Download the SAM3 checkpoint with progress display, SHA-256 validation,
    and resumable retry logic.

    Args:
        model: 'large' or 'base'.
        gpu_info: GPU info dict used to auto-downgrade if VRAM insufficient.

    Returns:
        Path to the downloaded model file.

    Raises:
        SystemExit: After DOWNLOAD_MAX_RETRIES failed attempts.
    """
    _section("Model checkpoint download")

    # Auto-downgrade if VRAM is insufficient for Large
    if model == "large" and gpu_info["backend"] == "cuda":
        if gpu_info["vram_gb"] < VRAM_THRESHOLD_LARGE_GB:
            _warn(
                f"Only {gpu_info['vram_gb']} GB VRAM detected. "
                f"SAM3-Large needs ≥{VRAM_THRESHOLD_LARGE_GB:.0f} GB. "
                "Auto-switching to SAM3-Base."
            )
            model = "base"

    if model == "large":
        url = SAM3_LARGE_URL
        filename = SAM3_LARGE_FILENAME
        expected_size = SAM3_LARGE_SIZE_BYTES
        expected_sha = SAM3_LARGE_SHA256
    else:
        url = SAM3_BASE_URL
        filename = SAM3_BASE_FILENAME
        expected_size = SAM3_BASE_SIZE_BYTES
        expected_sha = SAM3_BASE_SHA256

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / filename

    if dest.exists():
        if _validate_checkpoint(dest, expected_sha, expected_size):
            _ok(f"Model already downloaded and valid: {dest}")
            return dest
        else:
            _warn("Existing file is invalid or incomplete — re-downloading …")

    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            _download_with_progress(url, dest, attempt)
            if _validate_checkpoint(dest, expected_sha, expected_size):
                _ok(f"Model downloaded and verified: {dest}")
                return dest
            else:
                _warn("Checksum mismatch after download — retrying …")
                dest.unlink(missing_ok=True)
        except KeyboardInterrupt:
            _warn("Download cancelled by user.")
            sys.exit(0)
        except Exception as exc:  # noqa: BLE001
            _warn(f"Attempt {attempt}/{DOWNLOAD_MAX_RETRIES} failed: {exc}")
            if attempt < DOWNLOAD_MAX_RETRIES:
                time.sleep(2 ** attempt)

    _err(
        f"Failed to download model after {DOWNLOAD_MAX_RETRIES} attempts.\n"
        f"  Download manually from:\n    {url}\n"
        f"  Place the file at:\n    {dest}"
    )
    sys.exit(1)


def _download_with_progress(url: str, dest: Path, attempt: int) -> None:
    """
    Stream-download *url* to *dest* with a live progress bar.

    Supports resuming an interrupted download via the HTTP Range header.

    Args:
        url: Remote URL to fetch.
        dest: Local destination path.
        attempt: Current attempt number (displayed to user).
    """
    import requests  # available on host Python for installer bootstrap

    existing_bytes = dest.stat().st_size if dest.exists() else 0
    headers: dict[str, str] = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        _ok(f"Resuming from byte {existing_bytes:,} (attempt {attempt})")
    else:
        _ok(f"Starting download (attempt {attempt}) …")

    mode = "ab" if existing_bytes else "wb"

    with requests.get(
        url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT_S
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) + existing_bytes
        downloaded = existing_bytes
        start_time = time.monotonic()

        with dest.open(mode) as fh:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                _print_progress(downloaded, total, start_time)

    print()  # newline after progress bar


def _print_progress(downloaded: int, total: int, start: float) -> None:
    """Print a one-line progress indicator that overwrites itself."""
    elapsed = max(time.monotonic() - start, 0.001)
    speed_mb = (downloaded / elapsed) / (1024 * 1024)
    pct = (downloaded / total * 100) if total else 0.0
    remaining = ((total - downloaded) / (downloaded / elapsed)) if downloaded else 0
    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(
        f"\r  [{bar}] {pct:5.1f}%  "
        f"{downloaded / 1e6:7.1f} / {total / 1e6:.1f} MB  "
        f"{speed_mb:5.1f} MB/s  ETA {remaining:.0f}s   ",
        end="",
        flush=True,
    )


def _validate_checkpoint(path: Path, expected_sha: str, expected_size: int) -> bool:
    """
    Validate a model checkpoint by size and optionally by SHA-256.

    Args:
        path: Path to the file.
        expected_sha: Expected SHA-256 hex digest; skip if empty.
        expected_size: Expected file size in bytes; used as quick-check.

    Returns:
        True if validation passes.
    """
    if not path.exists():
        return False

    actual_size = path.stat().st_size
    # Allow a small tolerance for servers that add metadata
    if expected_size and abs(actual_size - expected_size) > 10_000_000:
        logger.warning(
            "Size mismatch: expected ~%d bytes, got %d bytes",
            expected_size,
            actual_size,
        )
        return False

    if not expected_sha:
        return True  # no checksum on file, size check passed

    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(DOWNLOAD_CHUNK_BYTES), b""):
            sha256.update(chunk)
    actual_sha = sha256.hexdigest()
    if actual_sha != expected_sha:
        logger.warning("SHA-256 mismatch: expected %s, got %s", expected_sha, actual_sha)
        return False

    return True


# ---------------------------------------------------------------------------
# Step 6 — Copy plugin to Resolve scripts folder
# ---------------------------------------------------------------------------

def install_plugin_scripts() -> None:
    """
    Copy the sam3_resolve package to DaVinci Resolve's Comp Scripts folder.

    Raises:
        SystemExit: If the target directory cannot be created.
    """
    _section("Installing plugin scripts into Resolve")

    dest_dir = RESOLVE_SCRIPTS_DIR / "sam3_resolve"
    try:
        RESOLVE_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(str(PACKAGE_ROOT), str(dest_dir))
        # Also copy plugin_main.py one level up so Resolve can find it
        plugin_main = REPO_ROOT / "sam3_resolve" / "plugin_main.py"
        if plugin_main.exists():
            shutil.copy2(str(plugin_main), str(RESOLVE_SCRIPTS_DIR / "SAM3_MaskTracker.py"))
        _ok(f"Plugin scripts copied to:\n    {RESOLVE_SCRIPTS_DIR}")
    except OSError as exc:
        _err(f"Could not copy scripts: {exc}")
        _warn("You can run the plugin directly from this folder instead.")


# ---------------------------------------------------------------------------
# Step 7 — Write config
# ---------------------------------------------------------------------------

def write_config(gpu_info: dict, model: str, venv_path: Path, dest_path: Path) -> None:
    """
    Persist detected settings to config.json.

    Args:
        gpu_info: GPU info dict.
        model: Active model name ('large' or 'base').
        venv_path: Absolute path to the plugin venv.
        dest_path: Path where the model file was saved.
    """
    _section("Writing config")

    # Load existing config template
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        data = {}

    data["installed"] = True
    data["plugin_version"] = PLUGIN_VERSION
    data["venv_path"] = str(venv_path)
    data["models_dir"] = str(MODELS_DIR)
    data["active_model"] = f"sam3_{model}"
    data["device"] = gpu_info["backend"]
    data["gpu_profile"] = {
        "backend": gpu_info["backend"],
        "device_name": gpu_info["device_name"],
        "vram_gb": gpu_info["vram_gb"],
        "cuda_version": gpu_info.get("cuda_version", ""),
        "driver_version": "",
    }

    CONFIG_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _ok(f"Config saved to {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install SAM3 Resolve Plugin"
    )
    parser.add_argument(
        "--model",
        choices=["large", "base"],
        default="large",
        help="SAM3 model size (default: large)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute backend override (default: auto-detect)",
    )
    parser.add_argument(
        "--skip-scripts",
        action="store_true",
        help="Skip copying scripts to Resolve folder (useful for dev)",
    )
    args = parser.parse_args()

    print(f"\n{_BOLD}SAM3 Resolve Plugin — Installer v{PLUGIN_VERSION}{_RESET}")
    print("=" * 60)

    check_python_version()
    gpu_info = detect_gpu_pre_torch()
    python_bin = create_venv()
    install_deps(gpu_info)
    model_path = download_model(args.model, gpu_info)
    if not args.skip_scripts:
        install_plugin_scripts()
    write_config(gpu_info, args.model, VENV_DIR, model_path)

    _section("Done")
    print(
        f"\n{_GREEN}{_BOLD}Installation complete!{_RESET}\n\n"
        "  How to open the plugin in DaVinci Resolve:\n"
        "  1. Open DaVinci Resolve 20.\n"
        "  2. Enable External Scripting:\n"
        "       Preferences → System → General → External scripting using\n"
        "       → select 'Local' and restart Resolve.\n"
        "  3. Open the Comp Scripts menu:\n"
        "       Workspace → Scripts → SAM3_MaskTracker\n\n"
        "  To repair the installation at any time:\n"
        "       python install.py\n"
    )


if __name__ == "__main__":
    main()
