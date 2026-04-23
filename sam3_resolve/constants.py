"""All magic-number constants for the SAM3 Resolve plugin.

No numeric or string literals should appear elsewhere in the codebase;
import them from here.
"""

from __future__ import annotations

import pathlib
import platform
import sys

# ── Version ────────────────────────────────────────────────────────────────

PLUGIN_VERSION = "1.0.0"
SAM3_MIN_PYTHON = (3, 10)

# ── Paths ──────────────────────────────────────────────────────────────────

HOME = pathlib.Path.home()
VENV_DIR = HOME / ".sam3_resolve_env"
CACHE_DIR = HOME / ".sam3_resolve_cache"
MODELS_DIR = HOME / ".sam3_resolve_models"
RECOVERY_DIR = CACHE_DIR / "recovery"

# Config file lives next to this constants.py, inside the package root.
PACKAGE_ROOT = pathlib.Path(__file__).parent
CONFIG_PATH = PACKAGE_ROOT / "config.json"

# Resolve script installation targets per OS
_RESOLVE_SCRIPTS: dict[str, pathlib.Path] = {
    "Windows": pathlib.Path(
        pathlib.Path.home(),
        "AppData",
        "Roaming",
        "Blackmagic Design",
        "DaVinci Resolve",
        "Support",
        "Scripts",
        "Comp",
    ),
    "Darwin": pathlib.Path(
        pathlib.Path.home(),
        "Library",
        "Application Support",
        "Blackmagic Design",
        "DaVinci Resolve",
        "Fusion",
        "Scripts",
        "Comp",
    ),
    "Linux": pathlib.Path(
        pathlib.Path.home(),
        ".local",
        "share",
        "DaVinciResolve",
        "Fusion",
        "Scripts",
        "Comp",
    ),
}

RESOLVE_SCRIPTS_DIR: pathlib.Path = _RESOLVE_SCRIPTS.get(
    platform.system(), _RESOLVE_SCRIPTS["Linux"]
)

# Resolve Python API paths to probe (in order)
RESOLVE_API_SEARCH_PATHS: list[pathlib.Path] = [
    pathlib.Path(
        "/opt/resolve/libs/Fusion/fusionscript.so"  # Linux typical
    ),
    pathlib.Path(
        "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
    ),
    pathlib.Path(
        "C:/Program Files/Blackmagic Design/DaVinci Resolve/fusionscript.dll"
    ),
]

# ── Model checkpoints ──────────────────────────────────────────────────────

SAM3_LARGE_FILENAME = "sam3_large.pt"
SAM3_BASE_FILENAME = "sam3_base.pt"

# Official Meta / HuggingFace CDN URLs — update when upstream moves
SAM3_LARGE_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
)
SAM3_BASE_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
)

SAM3_LARGE_SHA256 = ""   # populated at first successful download; validated on re-use
SAM3_BASE_SHA256 = ""

SAM3_LARGE_SIZE_BYTES = 898_100_000     # ~898 MB (sam2.1_hiera_large.pt)
SAM3_BASE_SIZE_BYTES = 378_000_000      # ~360 MB

VRAM_THRESHOLD_LARGE_GB = 7.0           # switch to Base below this (8 GB cards report ~7.7 GB free)

# ── Pip dependencies ───────────────────────────────────────────────────────

BASE_DEPS = [
    "opencv-python-headless",
    "numpy<2",
    "Pillow",
    "ffmpeg-python",
    "PyQt6",
    "requests",
    "tqdm",
]

TORCH_PACKAGES = [
    "torch",
    "torchvision",
    "torchaudio",
]

# Keep the old name as an alias so existing imports don't break
TORCH_DEPS_CUDA = TORCH_PACKAGES

TORCH_INDEX_CPU = "https://download.pytorch.org/whl/cpu"

# SAM3 / SAM2 from GitHub
SAM3_GITHUB_URL = "git+https://github.com/facebookresearch/sam2.git"

# ── Inference ──────────────────────────────────────────────────────────────

LIVE_INFERENCE_DEBOUNCE_MS = 80         # debounce for single-frame preview
LIVE_INFERENCE_TARGET_MS = 200          # UI warning threshold

SAM3_LOGIT_THRESHOLD = 0.0              # binarise logits above this value
MAX_OBJECTS = 8
DEFAULT_FEATHER_PX = 0
DEFAULT_OPACITY = 70                    # percent

# ── Proxy / cache ──────────────────────────────────────────────────────────

PROXY_SCALE_QUARTER = 0.25
PROXY_SCALE_HALF = 0.5
PROXY_SCALE_FULL = 1.0

PROXY_CRF = 18
PROXY_PRESET = "fast"

FRAME_JPEG_QUALITY = 92
MASK_UPSCALE_BLUR_SIGMA = 0.5           # soften edge artifacts after upscale
MASK_UPSCALE_MAX_CROP_PX = 2            # tolerated ffmpeg rounding difference

# ── Export ────────────────────────────────────────────────────────────────

EXPORT_PNG_BIT_DEPTH = 16               # single-channel alpha PNG
EXPORT_WORKERS = 4                      # multiprocessing pool size

# ── Scene-cut detection ────────────────────────────────────────────────────

SCENE_CUT_HIST_DIFF_THRESHOLD = 0.35    # histogram L1 distance triggering cut

# ── Memory / performance ──────────────────────────────────────────────────

OOM_RETRY_BATCH_SIZE = 1
LOG_MAX_LINES = 500
THUMBNAIL_COUNT = 9

# ── UI colours (also in styles.qss — kept in sync here for Python use) ────

COLOR_BG = "#1C1C1C"
COLOR_PANEL = "#242424"
COLOR_RAISED = "#2E2E2E"
COLOR_HOVER = "#383838"
COLOR_BORDER = "#3A3A3A"
COLOR_ACCENT = "#4A9EFF"
COLOR_ACCENT_TINT = "#1A3A5E"
COLOR_TEXT_PRIMARY = "#E8E8E8"
COLOR_TEXT_SECONDARY = "#909090"
COLOR_TEXT_DISABLED = "#555555"
COLOR_SUCCESS = "#5AB85A"
COLOR_WARNING = "#E8A030"
COLOR_DANGER = "#E85050"

COLOR_CONFIDENCE_HIGH = COLOR_SUCCESS
COLOR_CONFIDENCE_MED = COLOR_WARNING
COLOR_CONFIDENCE_LOW = COLOR_DANGER
COLOR_CONFIDENCE_NONE = "#444444"

# ── Download ───────────────────────────────────────────────────────────────

DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MB streaming chunks
DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_TIMEOUT_S = 30                 # per-request connect/read timeout
