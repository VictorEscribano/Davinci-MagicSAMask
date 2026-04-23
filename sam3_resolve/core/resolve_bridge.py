"""
DaVinci Resolve API bridge.

Provides a single ResolveBridge abstraction used by the rest of the plugin.
On machines without Resolve, MockResolveBridge is returned automatically so
the full UI and inference pipeline can be developed and tested normally.

Only the final Fusion node injection step requires a real Resolve instance.
"""

from __future__ import annotations

import logging
import platform
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Data transfer objects ──────────────────────────────────────────────────

class ClipFormat(str, Enum):
    """Video format categories that affect how frames are extracted."""
    DIRECT = "direct"       # OpenCV/ffmpeg can read directly
    NEEDS_PROXY = "needs_proxy"  # BRAW, R3D, ARRIRAW
    GENERATOR = "generator"     # Titles, adjustments — no video data
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClipInfo:
    """
    All metadata extracted from a Resolve timeline clip.

    Frozen so it can be safely shared across QThread workers.
    """
    name: str
    file_path: str
    proxy_path: str          # empty string if no proxy available
    media_pool_uuid: str
    color_label: str
    width: int
    height: int
    fps: float
    duration_frames: int
    start_frame: int         # absolute frame index of clip start in source file
    end_frame: int           # absolute frame index of clip end
    in_point_frame: int      # in-point relative to source file
    out_point_frame: int     # out-point relative to source file
    start_timecode: str
    clip_format: ClipFormat
    track_index: int = 1

    @property
    def resolution_label(self) -> str:
        return f"{self.width}×{self.height}"

    @property
    def duration_seconds(self) -> float:
        return self.duration_frames / self.fps if self.fps else 0.0

    @property
    def needs_proxy(self) -> bool:
        return self.clip_format == ClipFormat.NEEDS_PROXY


@dataclass
class FusionImportResult:
    """Result of importing mask sequence nodes into Fusion."""
    success: bool
    clip_name: str
    node_names: list[str] = field(default_factory=list)
    error: str = ""


# ── Bridge interface ───────────────────────────────────────────────────────

class ResolveBridgeBase(ABC):
    """Abstract interface for all DaVinci Resolve operations."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if connected to a running Resolve instance."""

    @abstractmethod
    def get_current_clip(self) -> ClipInfo:
        """
        Return metadata for the clip currently selected in the timeline.

        Raises:
            ResolveNotRunningError: If Resolve is not reachable.
            NoClipSelectedError: If no clip is selected.
            UnsupportedClipError: If the clip type is not supported (generator, etc.)
        """

    @abstractmethod
    def get_selected_clips(self) -> list[ClipInfo]:
        """
        Return all clips currently selected in the timeline.

        Returns an empty list if nothing is selected.
        """

    @abstractmethod
    def import_mask_to_fusion(
        self,
        clip_info: ClipInfo,
        mask_folder: Path,
        object_name: str,
        object_index: int,
    ) -> FusionImportResult:
        """
        Add a PNG alpha mask sequence as Loader + MatteControl nodes in Fusion.

        Never overwrites existing SAM3 nodes — appends only.
        Asks for confirmation if SAM3 nodes already exist.

        Args:
            clip_info:     The clip to operate on.
            mask_folder:   Folder containing mask_XXXXXX.png files.
            object_name:   Human-readable object label (used in node name).
            object_index:  Object number, used to build node label "SAM3_Mask_ObjN".

        Returns:
            FusionImportResult with success flag and node names.
        """

    @abstractmethod
    def render_proxy_via_resolve(
        self,
        clip_info: ClipInfo,
        scale: float,
        output_path: Path,
    ) -> bool:
        """
        Queue and execute a proxy render via Resolve's render queue.

        Args:
            clip_info:   Source clip.
            scale:       Output scale factor (e.g. 0.25 for 1/4 res).
            output_path: Destination file path.

        Returns:
            True on success.
        """


# ── Exceptions ─────────────────────────────────────────────────────────────

class ResolveNotRunningError(RuntimeError):
    """Resolve is not running or External Scripting is disabled."""
    HINT = (
        "Enable External Scripting in Resolve → Preferences → System → General "
        "→ External scripting using → 'Local', then restart Resolve."
    )


class NoClipSelectedError(RuntimeError):
    """No clip is selected in the timeline."""
    HINT = "Select a clip in the timeline and try again."


class ClipOfflineError(RuntimeError):
    """The clip file is offline (missing media)."""
    HINT = "Relink the clip in the Media Pool first."


class UnsupportedClipError(RuntimeError):
    """Clip type is not supported (generator, title, adjustment layer)."""
    HINT = "SAM3 requires a video clip. Generator clips have no video media."


class ProxyRequiredError(RuntimeError):
    """Format requires a proxy (BRAW, R3D, ARRIRAW)."""
    HINT = "Click 'Generate proxy' to create a working copy of this clip."


# ── Real implementation ────────────────────────────────────────────────────

def _candidate_resolve_api_paths() -> list[Path]:
    """
    Return all candidate directories for the Resolve scripting module,
    ordered from most-authoritative to least.

    Priority:
      1. RESOLVE_INSTALL_DIR env var — set by Resolve itself when launching
         external scripts; guaranteed correct when running inside Resolve.
      2. resolve_api_path in config.json — manually configured by the user.
      3. Standard install locations per OS.
      4. Glob search under common parent directories (non-standard installs).
    """
    import os
    candidates: list[Path] = []
    system = platform.system()

    # 1. Env var that Resolve sets in its own scripting environment
    env_dir = os.environ.get("RESOLVE_INSTALL_DIR", "")
    if env_dir:
        p = Path(env_dir)
        candidates += [
            p / "Developer" / "Scripting" / "Modules",
            p / "libs" / "Fusion",
            p,
        ]

    # 2. User-configured path saved after a previous successful connection
    try:
        from sam3_resolve.config import Config
        stored = Config.instance().get("resolve_api_path", "")
        if stored:
            candidates.append(Path(stored))
    except Exception:  # noqa: BLE001
        pass

    # 3. Standard install paths per OS
    if system == "Linux":
        candidates += [
            Path("/opt/resolve/Developer/Scripting/Modules"),
            Path("/opt/resolve/libs/Fusion"),
            Path("/usr/local/resolve/Developer/Scripting/Modules"),
            Path(Path.home() / "resolve" / "Developer" / "Scripting" / "Modules"),
        ]
    elif system == "Darwin":
        app_base = Path("/Applications/DaVinci Resolve")
        # Resolve can also be installed in /Applications/DaVinci Resolve Studio
        for app_name in ("DaVinci Resolve", "DaVinci Resolve Studio"):
            bundle = app_base.parent / app_name / f"{app_name}.app"
            candidates += [
                bundle / "Contents" / "Libraries" / "Fusion",
                bundle / "Contents" / "MacOS",
            ]
    elif system == "Windows":
        candidates += [
            Path(r"C:\Program Files\Blackmagic Design\DaVinci Resolve"),
            Path(r"C:\Program Files\Blackmagic Design\DaVinci Resolve Studio"),
        ]

    # 4. Glob fallback — search common parent dirs for fusionscript.so/.dll
    glob_parents: list[Path] = []
    if system == "Linux":
        glob_parents = [Path("/opt"), Path("/usr/local"), Path.home()]
    elif system == "Darwin":
        glob_parents = [Path("/Applications")]
    elif system == "Windows":
        glob_parents = [Path(r"C:\Program Files\Blackmagic Design")]

    lib_name = "fusionscript.so" if system != "Windows" else "fusionscript.dll"
    for parent in glob_parents:
        if not parent.exists():
            continue
        try:
            for found in parent.glob(f"**/{lib_name}"):
                candidates.append(found.parent)
        except (PermissionError, OSError):
            pass

    return candidates


def _inject_resolve_api_path() -> None:
    """
    Prepend the Resolve scripting module directory to sys.path.

    Searches in priority order: RESOLVE_INSTALL_DIR env var → config.json
    stored path → standard OS paths → glob fallback.  Saves the found path
    back to config.json so future launches skip the glob.
    """
    for candidate in _candidate_resolve_api_paths():
        if not candidate.exists():
            continue
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            logger.debug("Added Resolve API path: %s", candidate)
        # Persist the found path so we skip the glob next time
        try:
            from sam3_resolve.config import Config
            cfg = Config.instance()
            if not cfg.get("resolve_api_path"):
                cfg.set("resolve_api_path", path_str)
                cfg.save()
        except Exception:  # noqa: BLE001
            pass
        return   # stop at first valid candidate

    logger.debug(
        "No Resolve scripting path found. "
        "Set resolve_api_path in config.json or RESOLVE_INSTALL_DIR env var."
    )


_NEEDS_PROXY_EXTENSIONS = frozenset({".braw", ".r3d", ".ari", ".arx", ".mxf"})
_DIRECT_EXTENSIONS = frozenset({".mov", ".mp4", ".mkv", ".avi", ".mxf", ".m4v"})


def _detect_format(file_path: str) -> ClipFormat:
    suffix = Path(file_path).suffix.lower()
    if suffix in {".braw", ".r3d", ".ari", ".arx"}:
        return ClipFormat.NEEDS_PROXY
    if suffix in _DIRECT_EXTENSIONS or suffix in {".mxf"}:
        return ClipFormat.DIRECT
    return ClipFormat.UNKNOWN


class RealResolveBridge(ResolveBridgeBase):
    """
    Bridge backed by the live DaVinci Resolve scripting API.

    Instantiation will raise ResolveNotRunningError if Resolve is not
    reachable — callers should catch this and fall back to MockResolveBridge
    or display the error card.
    """

    def __init__(self) -> None:
        _inject_resolve_api_path()
        self._resolve = self._connect()

    def _connect(self):
        """
        Connect to the running Resolve instance.

        Returns:
            The Resolve scripting object.

        Raises:
            ResolveNotRunningError: On any connection failure.
        """
        try:
            import DaVinciResolveScript as dvr  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ResolveNotRunningError(
                f"DaVinciResolveScript module not found: {exc}. "
                + ResolveNotRunningError.HINT
            ) from exc

        resolve = dvr.scriptapp("Resolve")
        if resolve is None:
            raise ResolveNotRunningError(
                "Could not connect to Resolve. Is it running? "
                + ResolveNotRunningError.HINT
            )

        logger.info("Connected to DaVinci Resolve")
        return resolve

    def is_connected(self) -> bool:
        try:
            return self._resolve is not None and self._resolve.GetProductName() is not None
        except Exception:
            return False

    def get_current_clip(self) -> ClipInfo:
        pm = self._resolve.GetProjectManager()
        project = pm.GetCurrentProject()
        if project is None:
            raise ResolveNotRunningError("No project open in Resolve.")

        timeline = project.GetCurrentTimeline()
        if timeline is None:
            raise NoClipSelectedError("No timeline is open. " + NoClipSelectedError.HINT)

        item = timeline.GetCurrentVideoItem()
        if item is None:
            raise NoClipSelectedError(NoClipSelectedError.HINT)

        return self._clip_info_from_item(item)

    def get_selected_clips(self) -> list[ClipInfo]:
        pm = self._resolve.GetProjectManager()
        project = pm.GetCurrentProject()
        if not project:
            return []
        timeline = project.GetCurrentTimeline()
        if not timeline:
            return []

        clips: list[ClipInfo] = []
        track_count = timeline.GetTrackCount("video")
        for track_idx in range(1, track_count + 1):
            for item in timeline.GetItemListInTrack("video", track_idx):
                if item.GetProperty("Selected"):
                    try:
                        clips.append(self._clip_info_from_item(item, track_idx))
                    except (UnsupportedClipError, ClipOfflineError) as exc:
                        logger.warning("Skipping clip: %s", exc)
        return clips

    def _clip_info_from_item(self, item, track_index: int = 1) -> ClipInfo:
        """Extract ClipInfo from a Resolve timeline item."""
        media_item = item.GetMediaPoolItem()
        if media_item is None:
            raise UnsupportedClipError(UnsupportedClipError.HINT)

        props = media_item.GetClipProperty()
        file_path: str = props.get("File Path", "")
        if not file_path:
            raise ClipOfflineError(ClipOfflineError.HINT)

        clip_format = _detect_format(file_path)
        if clip_format == ClipFormat.GENERATOR:
            raise UnsupportedClipError(UnsupportedClipError.HINT)

        # Proxy
        proxy_path: str = media_item.GetClipProperty().get("Proxy Media Path", "")

        fps_str: str = props.get("FPS", "24")
        try:
            fps = float(fps_str)
        except ValueError:
            fps = 24.0

        width = int(props.get("Resolution", "1920x1080").split("x")[0])
        height = int(props.get("Resolution", "1920x1080").split("x")[1])
        duration_frames = int(props.get("Frames", 0))
        start_tc: str = props.get("Start TC", "00:00:00:00")

        in_point = item.GetLeftOffset()
        out_point = duration_frames - item.GetRightOffset()

        return ClipInfo(
            name=media_item.GetName(),
            file_path=file_path,
            proxy_path=proxy_path,
            media_pool_uuid=str(media_item.GetUniqueId()),
            color_label=str(media_item.GetClipColor()),
            width=width,
            height=height,
            fps=fps,
            duration_frames=duration_frames,
            start_frame=0,
            end_frame=duration_frames,
            in_point_frame=in_point,
            out_point_frame=out_point,
            start_timecode=start_tc,
            clip_format=clip_format,
            track_index=track_index,
        )

    def import_mask_to_fusion(
        self,
        clip_info: ClipInfo,
        mask_folder: Path,
        object_name: str,
        object_index: int,
    ) -> FusionImportResult:
        node_label = f"SAM3_Mask_Obj{object_index}"
        try:
            pm = self._resolve.GetProjectManager()
            project = pm.GetCurrentProject()
            timeline = project.GetCurrentTimeline()

            item = self._find_timeline_item(timeline, clip_info)
            if item is None:
                return FusionImportResult(
                    success=False,
                    clip_name=clip_info.name,
                    error="Could not locate clip in timeline.",
                )

            comp = item.GetFusionCompByIndex(1)
            if comp is None:
                comp = item.AddFusionComp()

            # Guard: never overwrite existing SAM3 nodes
            existing = [
                name for name in (comp.GetToolList() or {})
                if name.startswith("SAM3_")
            ]
            if existing:
                logger.warning(
                    "SAM3 nodes already exist on %s: %s", clip_info.name, existing
                )

            # PNG sequence pattern: mask_000001.png → mask_[000001].png
            png_pattern = str(mask_folder / "mask_[000000-999999].png")
            loader = comp.AddTool("Loader")
            loader.Clip[1] = png_pattern
            loader.SetAttrs({"TOOLS_Name": node_label + "_Loader"})

            matte = comp.AddTool("MatteControl")
            matte.SetAttrs({"TOOLS_Name": node_label + "_Matte"})
            matte.Foreground.ConnectTo(loader.Output)

            logger.info("Fusion nodes added to %s: %s", clip_info.name, node_label)
            return FusionImportResult(
                success=True,
                clip_name=clip_info.name,
                node_names=[node_label + "_Loader", node_label + "_Matte"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Fusion import failed: %s", exc)
            return FusionImportResult(
                success=False,
                clip_name=clip_info.name,
                error=str(exc),
            )

    def _find_timeline_item(self, timeline, clip_info: ClipInfo):
        """Locate the timeline item matching clip_info by UUID."""
        track_count = timeline.GetTrackCount("video")
        for t in range(1, track_count + 1):
            for item in timeline.GetItemListInTrack("video", t):
                mp_item = item.GetMediaPoolItem()
                if mp_item and str(mp_item.GetUniqueId()) == clip_info.media_pool_uuid:
                    return item
        return None

    def render_proxy_via_resolve(
        self,
        clip_info: ClipInfo,
        scale: float,
        output_path: Path,
    ) -> bool:
        # Resolve render queue integration is highly version-specific.
        # This implementation targets Resolve 20's render queue API.
        try:
            pm = self._resolve.GetProjectManager()
            project = pm.GetCurrentProject()
            render_settings = {
                "SelectAllFrames": True,
                "TargetDir": str(output_path.parent),
                "CustomName": output_path.stem,
                "FormatWidth": int(clip_info.width * scale),
                "FormatHeight": int(clip_info.height * scale),
                "VideoQuality": 18,
            }
            project.SetRenderSettings(render_settings)
            job_id = project.AddRenderJob()
            project.StartRendering(job_id)
            # Poll until done (blocking — caller should run in QThread)
            while project.IsRenderingInProgress():
                import time
                time.sleep(0.5)
            return output_path.exists()
        except Exception as exc:  # noqa: BLE001
            logger.error("Resolve proxy render failed: %s", exc)
            return False


# ── Mock implementation ────────────────────────────────────────────────────

class MockResolveBridge(ResolveBridgeBase):
    """
    Drop-in replacement for RealResolveBridge when Resolve is not installed.

    Returns realistic synthetic clip metadata so the full UI pipeline can be
    developed and tested without DaVinci Resolve.

    All state is mutable so tests can override clip properties via
    `bridge.mock_clip = dataclasses.replace(bridge.mock_clip, width=3840)`.
    """

    DEFAULT_CLIP = ClipInfo(
        name="A001C001_240315.mov",
        file_path="/media/footage/A001C001_240315.mov",
        proxy_path="",
        media_pool_uuid="mock-uuid-0001",
        color_label="Orange",
        width=1920,
        height=1080,
        fps=23.976,
        duration_frames=576,    # ~24s at 23.976
        start_frame=0,
        end_frame=576,
        in_point_frame=0,
        out_point_frame=576,
        start_timecode="01:00:00:00",
        clip_format=ClipFormat.DIRECT,
        track_index=1,
    )

    def __init__(self, clip: Optional[ClipInfo] = None) -> None:
        self.mock_clip: ClipInfo = clip or self.DEFAULT_CLIP
        self.simulate_no_clip = False
        self.simulate_offline = False
        self.imported_masks: list[dict] = []   # records calls to import_mask_to_fusion
        logger.info("MockResolveBridge active — Resolve not installed on this machine")

    def is_connected(self) -> bool:
        return True

    def get_current_clip(self) -> ClipInfo:
        if self.simulate_no_clip:
            raise NoClipSelectedError(NoClipSelectedError.HINT)
        if self.simulate_offline:
            raise ClipOfflineError(ClipOfflineError.HINT)
        return self.mock_clip

    def get_selected_clips(self) -> list[ClipInfo]:
        if self.simulate_no_clip:
            return []
        return [self.mock_clip]

    def import_mask_to_fusion(
        self,
        clip_info: ClipInfo,
        mask_folder: Path,
        object_name: str,
        object_index: int,
    ) -> FusionImportResult:
        node_label = f"SAM3_Mask_Obj{object_index}"
        record = {
            "clip": clip_info.name,
            "folder": str(mask_folder),
            "object_name": object_name,
            "object_index": object_index,
            "node_label": node_label,
        }
        self.imported_masks.append(record)
        logger.info("[MOCK] Fusion import recorded: %s", record)
        return FusionImportResult(
            success=True,
            clip_name=clip_info.name,
            node_names=[node_label + "_Loader", node_label + "_Matte"],
        )

    def render_proxy_via_resolve(
        self,
        clip_info: ClipInfo,
        scale: float,
        output_path: Path,
    ) -> bool:
        logger.info(
            "[MOCK] Proxy render requested: %s → %s (scale=%.2f)",
            clip_info.name,
            output_path,
            scale,
        )
        return True


# ── Factory ────────────────────────────────────────────────────────────────

def create_bridge(force_mock: bool = False) -> ResolveBridgeBase:
    """
    Return the appropriate bridge for the current environment.

    Tries RealResolveBridge first. Falls back to MockResolveBridge if:
      - force_mock is True
      - DaVinciResolveScript cannot be imported
      - Resolve is not running

    Args:
        force_mock: Skip the real bridge entirely (useful in tests).

    Returns:
        A connected ResolveBridgeBase instance.
    """
    if force_mock:
        return MockResolveBridge()

    try:
        bridge = RealResolveBridge()
        logger.info("Using RealResolveBridge")
        return bridge
    except Exception as exc:  # noqa: BLE001 — catches ImportError, ResolveNotRunningError, etc.
        logger.warning(
            "RealResolveBridge unavailable (%s). Falling back to MockResolveBridge.",
            exc,
        )
        return MockResolveBridge()
