#!/usr/bin/env python3
"""
SAM3 Resolve Plugin — one-command installer.

Usage:
    python install.py [--model {large,base,both}] [--device {auto,cuda,mps,cpu}]
                      [--skip-scripts] [--skip-venv]

Steps:
    1.  Python version check (≥3.10 required)
    2.  GPU / backend detection (nvidia-smi, driver CUDA version)
    3.  Install pip dependencies into the current Python environment
        (torch with matching CUDA wheels, sam2, opencv, numpy<2, etc.)
    4.  [Optional] Create venv at ~/.sam3_resolve_env/ for Resolve integration
    5.  Download SAM3 model checkpoint(s) with progress + validation
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
import time
import venv
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap: resolve the package root so we can import constants
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
# Logging / display helpers
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("install")

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _ok(msg: str)   -> None: print(f"  {_GREEN}✓{_RESET}  {msg}")
def _warn(msg: str) -> None: print(f"  {_YELLOW}⚠{_RESET}  {msg}")
def _err(msg: str)  -> None: print(f"  {_RED}✗{_RESET}  {msg}")

def _section(title: str) -> None:
    print(f"\n{_BOLD}── {title} {'─' * max(0, 50 - len(title))}{_RESET}")


# ---------------------------------------------------------------------------
# Step 1 — Python version
# ---------------------------------------------------------------------------

def check_python_version() -> None:
    _section("Python version check")
    major, minor = sys.version_info[:2]
    if (major, minor) < SAM3_MIN_PYTHON:
        _err(
            f"Python {major}.{minor} detected. "
            f"SAM3 requires Python ≥{SAM3_MIN_PYTHON[0]}.{SAM3_MIN_PYTHON[1]}."
        )
        sys.exit(1)
    _ok(f"Python {major}.{minor} — OK")


# ---------------------------------------------------------------------------
# Step 2 — GPU detection (without torch)
# ---------------------------------------------------------------------------

def _run_silent(cmd: list[str]) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return r.returncode, r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 1, ""


def detect_gpu_pre_torch() -> dict:
    """
    Detect GPU and CUDA version before torch is installed.

    Uses nvidia-smi driver output for the CUDA version (not nvcc), because
    the driver supports all CUDA toolkit versions up to the displayed one.
    Returns a dict: {backend, cuda_version, cuda_tag, device_name, vram_gb}.
    """
    _section("GPU detection")
    system = platform.system()

    # ── NVIDIA via nvidia-smi ──────────────────────────────────────────────
    rc, smi_out = _run_silent([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if rc == 0 and smi_out:
        parts = [p.strip() for p in smi_out.splitlines()[0].split(",")]
        name     = parts[0] if len(parts) > 0 else "NVIDIA GPU"
        vram_mib = float(parts[1]) if len(parts) > 1 else 0.0
        vram_gb  = round(vram_mib / 1024, 1)

        # Use nvidia-smi full output to find "CUDA Version: X.Y" (driver-reported)
        _, smi_full = _run_silent(["nvidia-smi"])
        cuda_ver = _parse_driver_cuda(smi_full)

        cuda_tag = _cuda_tag_for_version(cuda_ver)
        _ok(f"NVIDIA {name} · {vram_gb} GB VRAM · driver CUDA {cuda_ver or 'unknown'} → wheels: {cuda_tag}")
        return {
            "backend":      "cuda",
            "cuda_version": cuda_ver,
            "cuda_tag":     cuda_tag,
            "device_name":  name,
            "vram_gb":      vram_gb,
        }

    # ── Apple Silicon ──────────────────────────────────────────────────────
    if system == "Darwin":
        _, uname = _run_silent(["uname", "-m"])
        if "arm" in uname.lower():
            _ok("Apple Silicon — MPS backend")
            return {"backend": "mps", "cuda_version": "", "cuda_tag": "", "device_name": "Apple Silicon", "vram_gb": 0.0}

    # ── CPU fallback ───────────────────────────────────────────────────────
    _warn("No GPU detected — SAM3 will run on CPU (slow, ~8 s/frame).")
    return {"backend": "cpu", "cuda_version": "", "cuda_tag": "", "device_name": platform.processor() or "CPU", "vram_gb": 0.0}


def _parse_driver_cuda(smi_text: str) -> str:
    """Extract 'CUDA Version: X.Y' from nvidia-smi full output."""
    for line in smi_text.splitlines():
        if "CUDA Version" in line:
            # e.g. "| CUDA Version: 12.4     |"  or  "CUDA Version: 13.0"
            parts = line.replace("|", "").strip().split(":")
            if len(parts) >= 2:
                return parts[-1].strip().split()[0]
    return ""


def _cuda_tag_for_version(cuda_ver: str) -> str:
    """
    Map a driver CUDA version string to the best available PyTorch cu-tag.

    Driver CUDA is always backwards-compatible, so we pick the newest
    supported torch wheel that fits under the driver version.
    """
    if not cuda_ver:
        return "cu124"   # safe default

    try:
        major, minor = (int(x) for x in cuda_ver.split(".")[:2])
    except ValueError:
        return "cu124"

    # Ordered newest-first; pick first that fits ≤ driver version
    _WHEEL_MAP = [
        ((12, 4), "cu124"),
        ((12, 1), "cu121"),
        ((11, 8), "cu118"),
        ((11, 7), "cu117"),
    ]
    for (req_major, req_minor), tag in _WHEEL_MAP:
        if (major, minor) >= (req_major, req_minor):
            return tag

    return "cu118"   # oldest supported


# ---------------------------------------------------------------------------
# Step 3 — Install dependencies into current Python
# ---------------------------------------------------------------------------

def install_deps(gpu_info: dict) -> None:
    """
    Install all plugin dependencies into the current Python environment.

    Handles externally-managed environments (Debian/Ubuntu) by retrying
    with --break-system-packages when the plain install is rejected.
    """
    _section("Installing dependencies")

    pip_base = [sys.executable, "-m", "pip"]

    # Upgrade pip first
    _pip_run(pip_base + ["install", "--upgrade", "pip", "setuptools", "wheel"])

    # ── Torch ─────────────────────────────────────────────────────────────
    backend = gpu_info["backend"]
    if backend == "cuda":
        index_url = f"https://download.pytorch.org/whl/{gpu_info['cuda_tag']}"
        _ok(f"Installing torch (CUDA wheels, {gpu_info['cuda_tag']}) …")
        _pip_run(pip_base + ["install"] + TORCH_DEPS_CUDA + ["--index-url", index_url])
    elif backend == "mps":
        _ok("Installing torch (standard wheels, MPS included) …")
        _pip_run(pip_base + ["install"] + TORCH_DEPS_CUDA)
    else:
        _ok(f"Installing torch (CPU-only) …")
        _pip_run(pip_base + ["install"] + TORCH_DEPS_CUDA + ["--index-url", TORCH_INDEX_CPU])

    # ── SAM2 from GitHub ───────────────────────────────────────────────────
    _ok("Installing SAM2 from GitHub …")
    _pip_run(pip_base + ["install", SAM3_GITHUB_URL])

    # ── numpy<2 BEFORE opencv to avoid ABI mismatch ────────────────────────
    # opencv pre-built wheels are compiled against NumPy 1.x; NumPy 2 breaks them.
    _ok("Installing numpy<2 (required for opencv ABI compatibility) …")
    _pip_run(pip_base + ["install", "numpy<2"])

    # ── Remaining deps ─────────────────────────────────────────────────────
    _ok("Installing remaining dependencies …")
    _pip_run(pip_base + ["install"] + BASE_DEPS)

    _ok("All dependencies installed")


def _pip_run(cmd: list[str]) -> None:
    """
    Run a pip command, streaming output.

    If the environment is externally managed (Debian/Ubuntu PEP 668),
    retries automatically with --break-system-packages.
    """
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        return

    # Check if it failed due to externally-managed environment
    probe = subprocess.run(cmd, capture_output=True, text=True)
    if "externally-managed-environment" in probe.stderr:
        _warn("Externally-managed environment detected — retrying with --break-system-packages")
        cmd_bsp = cmd + ["--break-system-packages"]
        result2 = subprocess.run(cmd_bsp)
        if result2.returncode == 0:
            return
        _err(f"pip command failed even with --break-system-packages:\n  {' '.join(cmd_bsp)}")
    else:
        _err(f"pip command failed: {' '.join(cmd)}")

    sys.exit(1)


# ---------------------------------------------------------------------------
# Step 4 — Optional venv for Resolve integration
# ---------------------------------------------------------------------------

def create_venv_for_resolve(gpu_info: dict) -> Optional[Path]:
    """
    Create a dedicated venv that Resolve activates at runtime via plugin_main.py.
    Installs the same deps as step 3 so Resolve's bundled Python can import them.

    Returns the venv Python path, or None if creation was skipped/failed.
    """
    _section("Resolve venv (for Workspace → Scripts integration)")

    VENV_DIR.mkdir(parents=True, exist_ok=True)

    if (VENV_DIR / "pyvenv.cfg").exists():
        _ok(f"Existing venv at {VENV_DIR}")
    else:
        _ok(f"Creating venv at {VENV_DIR} …")
        try:
            venv.create(str(VENV_DIR), with_pip=True, clear=False, upgrade_deps=True)
        except Exception as exc:  # noqa: BLE001
            _warn(f"Could not create venv: {exc}. Skipping.")
            return None
        _ok("venv created")

    venv_pip = _venv_pip()
    pip_base = [str(venv_pip)]

    # Install into venv — no externally-managed issue inside a venv
    backend = gpu_info["backend"]
    if backend == "cuda":
        index_url = f"https://download.pytorch.org/whl/{gpu_info['cuda_tag']}"
        subprocess.run([str(venv_pip), "install"] + TORCH_DEPS_CUDA + ["--index-url", index_url])
    elif backend == "mps":
        subprocess.run([str(venv_pip), "install"] + TORCH_DEPS_CUDA)
    else:
        subprocess.run([str(venv_pip), "install"] + TORCH_DEPS_CUDA + ["--index-url", TORCH_INDEX_CPU])

    subprocess.run([str(venv_pip), "install", SAM3_GITHUB_URL])
    subprocess.run([str(venv_pip), "install", "numpy<2"])
    subprocess.run([str(venv_pip), "install"] + BASE_DEPS)

    _ok(f"Resolve venv ready: {VENV_DIR}")
    return _venv_python()


def _venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _venv_pip() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


# ---------------------------------------------------------------------------
# Step 5 — Model download
# ---------------------------------------------------------------------------

def download_model(model: str, gpu_info: dict) -> Path:
    """
    Download the requested SAM3 checkpoint.

    model: 'large' | 'base' | 'both'
    Auto-selects 'base' if VRAM < VRAM_THRESHOLD_LARGE_GB and model == 'large'.
    """
    _section("Model checkpoint download")

    # Auto-downgrade Large → Base on low-VRAM machines
    if model == "large" and gpu_info["backend"] == "cuda":
        if gpu_info["vram_gb"] < VRAM_THRESHOLD_LARGE_GB:
            _warn(
                f"Only {gpu_info['vram_gb']} GB VRAM — SAM3-Large needs "
                f"≥{VRAM_THRESHOLD_LARGE_GB:.0f} GB. Downloading Base instead."
            )
            model = "base"

    models_to_download = []
    if model in ("large", "both"):
        models_to_download.append(("large", SAM3_LARGE_URL, SAM3_LARGE_FILENAME,
                                   SAM3_LARGE_SIZE_BYTES, SAM3_LARGE_SHA256))
    if model in ("base", "both"):
        models_to_download.append(("base", SAM3_BASE_URL, SAM3_BASE_FILENAME,
                                   SAM3_BASE_SIZE_BYTES, SAM3_BASE_SHA256))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    last_path = None

    for name, url, filename, expected_size, expected_sha in models_to_download:
        dest = MODELS_DIR / filename
        _ok(f"Checking SAM3-{name.capitalize()} ({filename}) …")

        if dest.exists() and _validate_checkpoint(dest, expected_sha, expected_size):
            _ok(f"Already downloaded and valid: {dest}")
            last_path = dest
            continue

        if dest.exists():
            _warn("Existing file invalid or incomplete — re-downloading …")
            dest.unlink(missing_ok=True)

        success = False
        for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
            try:
                _download_with_progress(url, dest, attempt)
                if _validate_checkpoint(dest, expected_sha, expected_size):
                    _ok(f"SAM3-{name.capitalize()} downloaded and verified: {dest}")
                    last_path = dest
                    success = True
                    break
                else:
                    _warn("Validation failed — retrying …")
                    dest.unlink(missing_ok=True)
            except KeyboardInterrupt:
                _warn("Download cancelled.")
                sys.exit(0)
            except Exception as exc:  # noqa: BLE001
                _warn(f"Attempt {attempt}/{DOWNLOAD_MAX_RETRIES} failed: {exc}")
                if attempt < DOWNLOAD_MAX_RETRIES:
                    time.sleep(2 ** attempt)

        if not success:
            _err(
                f"Failed to download SAM3-{name.capitalize()} after {DOWNLOAD_MAX_RETRIES} attempts.\n"
                f"  Download manually:\n    {url}\n"
                f"  Place at:\n    {dest}"
            )
            sys.exit(1)

    return last_path or (MODELS_DIR / SAM3_LARGE_FILENAME)


def _download_with_progress(url: str, dest: Path, attempt: int) -> None:
    import requests

    existing_bytes = dest.stat().st_size if dest.exists() else 0
    headers: dict[str, str] = {}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        _ok(f"Resuming from byte {existing_bytes:,} (attempt {attempt})")
    else:
        _ok(f"Starting download (attempt {attempt}) …")

    mode = "ab" if existing_bytes else "wb"
    with requests.get(url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as resp:
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
    print()


def _print_progress(downloaded: int, total: int, start: float) -> None:
    elapsed  = max(time.monotonic() - start, 0.001)
    speed_mb = (downloaded / elapsed) / (1024 * 1024)
    pct      = (downloaded / total * 100) if total else 0.0
    eta      = ((total - downloaded) / (downloaded / elapsed)) if downloaded else 0
    bar_w    = 30
    filled   = int(bar_w * pct / 100)
    bar      = "█" * filled + "░" * (bar_w - filled)
    print(
        f"\r  [{bar}] {pct:5.1f}%  "
        f"{downloaded / 1e6:7.1f} / {total / 1e6:.1f} MB  "
        f"{speed_mb:5.1f} MB/s  ETA {eta:.0f}s   ",
        end="", flush=True,
    )


def _validate_checkpoint(path: Path, expected_sha: str, expected_size: int) -> bool:
    if not path.exists():
        return False

    actual_size = path.stat().st_size
    if expected_size and abs(actual_size - expected_size) > 10_000_000:
        logger.warning(
            "Size mismatch for %s: expected ~%d bytes, got %d bytes",
            path.name, expected_size, actual_size,
        )
        return False

    if not expected_sha:
        return True   # no SHA configured — size check is enough

    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(DOWNLOAD_CHUNK_BYTES), b""):
            sha256.update(chunk)
    if sha256.hexdigest() != expected_sha:
        logger.warning("SHA-256 mismatch for %s", path.name)
        return False
    return True


# ---------------------------------------------------------------------------
# Step 6 — Copy plugin to Resolve scripts folder
# ---------------------------------------------------------------------------

def install_plugin_scripts() -> None:
    _section("Installing plugin scripts into Resolve")
    dest_dir = RESOLVE_SCRIPTS_DIR / "sam3_resolve"
    try:
        RESOLVE_SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(str(PACKAGE_ROOT), str(dest_dir))
        # Top-level entry point so Resolve's Scripts menu can find it
        plugin_main = REPO_ROOT / "sam3_resolve" / "plugin_main.py"
        if plugin_main.exists():
            shutil.copy2(str(plugin_main), str(RESOLVE_SCRIPTS_DIR / "SAM3_MaskTracker.py"))
        _ok(f"Plugin scripts copied to:\n    {RESOLVE_SCRIPTS_DIR}")
    except OSError as exc:
        _err(f"Could not copy scripts: {exc}")
        _warn("You can open the plugin directly: python -m sam3_resolve.plugin_main")


# ---------------------------------------------------------------------------
# Step 7 — Write config
# ---------------------------------------------------------------------------

def write_config(gpu_info: dict, model: str, venv_path: Optional[Path], model_path: Path) -> None:
    _section("Writing config")

    data: dict = {}
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}

    # model_name is used by create_runner() to decide which checkpoint to load
    model_name = "base" if model == "base" else "large"

    data["installed"]      = True
    data["plugin_version"] = PLUGIN_VERSION
    data["model_name"]     = model_name
    data["models_dir"]     = str(MODELS_DIR)
    data["device"]         = gpu_info["backend"]
    if venv_path:
        data["venv_path"]  = str(venv_path.parent.parent)   # venv root, not bin/

    data["gpu_profile"] = {
        "backend":      gpu_info["backend"],
        "device_name":  gpu_info["device_name"],
        "vram_gb":      gpu_info["vram_gb"],
        "cuda_version": gpu_info.get("cuda_version", ""),
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    _ok(f"Config saved to {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Verify installation
# ---------------------------------------------------------------------------

def verify_install() -> None:
    _section("Verifying installation")
    ok = True

    # torch + CUDA
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            _ok(result.stdout.strip())
        else:
            _warn(f"torch import failed: {result.stderr.strip()[:120]}")
            ok = False
    except Exception as exc:  # noqa: BLE001
        _warn(f"Could not verify torch: {exc}")
        ok = False

    # sam2
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import sam2; print('sam2 OK')"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            _ok(result.stdout.strip())
        else:
            _warn(f"sam2 import failed: {result.stderr.strip()[:120]}")
            ok = False
    except Exception as exc:  # noqa: BLE001
        _warn(f"Could not verify sam2: {exc}")
        ok = False

    # opencv / numpy ABI
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import cv2; print('cv2', cv2.__version__)"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            _ok(result.stdout.strip())
        else:
            _warn(f"cv2 import failed (numpy ABI mismatch?): {result.stderr.strip()[:200]}")
            ok = False
    except Exception as exc:  # noqa: BLE001
        _warn(f"Could not verify cv2: {exc}")
        ok = False

    # Model files
    for filename in (SAM3_LARGE_FILENAME, SAM3_BASE_FILENAME):
        p = MODELS_DIR / filename
        if p.exists():
            _ok(f"Model present: {p.name} ({p.stat().st_size / 1e6:.0f} MB)")

    if not ok:
        _warn("Some checks failed — run 'python install.py' again to repair.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Install SAM3 Resolve Plugin")
    parser.add_argument(
        "--model",
        choices=["large", "base", "both"],
        default="large",
        help=(
            "Which SAM3 checkpoint to download. "
            "'both' downloads large + base (useful when VRAM may be tight). "
            "Default: large"
        ),
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Override compute backend (default: auto-detect)",
    )
    parser.add_argument(
        "--skip-scripts",
        action="store_true",
        help="Skip copying scripts to Resolve (useful during development)",
    )
    parser.add_argument(
        "--skip-venv",
        action="store_true",
        help="Skip creating the Resolve venv (install only into current Python)",
    )
    args = parser.parse_args()

    print(f"\n{_BOLD}SAM3 Resolve Plugin — Installer v{PLUGIN_VERSION}{_RESET}")
    print("=" * 60)

    check_python_version()
    gpu_info = detect_gpu_pre_torch()

    # Allow device override
    if args.device != "auto":
        gpu_info["backend"] = args.device
        if args.device == "cuda" and not gpu_info.get("cuda_tag"):
            gpu_info["cuda_tag"] = "cu124"

    install_deps(gpu_info)

    venv_python = None
    if not args.skip_venv:
        venv_python = create_venv_for_resolve(gpu_info)

    model_path = download_model(args.model, gpu_info)

    if not args.skip_scripts:
        install_plugin_scripts()

    write_config(gpu_info, args.model, venv_python, model_path)
    verify_install()

    # Determine which model is active for the summary
    active_model = args.model if args.model != "both" else "large"
    if (
        active_model == "large"
        and gpu_info["backend"] == "cuda"
        and gpu_info["vram_gb"] < VRAM_THRESHOLD_LARGE_GB
    ):
        active_model = "base"

    _section("Done")
    print(
        f"\n{_GREEN}{_BOLD}Installation complete!{_RESET}\n"
        f"\n  Active model : SAM3-{active_model.capitalize()}"
        f"\n  Device       : {gpu_info['backend'].upper()}"
        f"\n  Models dir   : {MODELS_DIR}"
        "\n"
        "\n  How to use:"
        "\n    Debug mode (no Resolve needed):"
        "\n      python -m sam3_resolve.plugin_main --debug"
        "\n"
        "\n    From DaVinci Resolve:"
        "\n      1. Preferences → System → General → External scripting → Local"
        "\n      2. Restart Resolve"
        "\n      3. Workspace → Scripts → SAM3_MaskTracker"
        "\n"
        "\n    Download Base model (for low-VRAM / OOM fallback):"
        "\n      python install.py --model base"
        "\n    Download both models:"
        "\n      python install.py --model both"
        "\n"
        "\n  To repair at any time:"
        "\n      python install.py"
        "\n"
    )


if __name__ == "__main__":
    main()
