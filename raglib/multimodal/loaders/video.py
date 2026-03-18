"""
Video loader for multimodal RAG.

Supports: .mp4, .avi, .mov, .mkv, .webm, .flv, .wmv

Dependencies (lazy imports):
  - cv2 (opencv-python) — frame extraction
  - decord — fast video decoding (preferred over cv2)
  - moviepy — audio extraction
  - scenedetect — scene boundary detection

Install: pip install raglib[video]
"""
from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path
from typing import Optional

from raglib.multimodal.models.media import VideoChunk, VideoFrame

_SUPPORTED_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]


def _encode_frame_cv2(frame) -> str:
    """Encode a BGR numpy frame (from cv2) to base64-encoded JPEG bytes."""
    import cv2  # type: ignore

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("cv2.imencode failed for frame")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_frames_decord(
    path: str,
    fps_sample: float,
    native_fps: float,
) -> list[tuple[int, float, str, int, int]]:
    """Extract frames using decord (preferred fast path).

    Returns a list of ``(frame_index, timestamp_ms, image_bytes_b64, width, height)``
    tuples sampled at *fps_sample* frames per second.
    """
    import cv2  # type: ignore  # used only for JPEG encoding
    import decord  # type: ignore

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    total_frames = len(vr)
    step = max(1, int(round(native_fps / fps_sample)))
    indices = list(range(0, total_frames, step))

    results: list[tuple[int, float, float, str, int, int]] = []
    for local_idx, frame_idx in enumerate(indices):
        frame = vr[frame_idx].asnumpy()  # RGB
        h, w = frame.shape[:2]
        # Convert RGB -> BGR for cv2 JPEG encoding
        import numpy as np  # type: ignore

        bgr = frame[:, :, ::-1]
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        ts_ms = (frame_idx / native_fps) * 1000.0
        results.append((local_idx, ts_ms, b64, w, h))

    return results


def _extract_frames_cv2(
    path: str,
    fps_sample: float,
    native_fps: float,
) -> list[tuple[int, float, str, int, int]]:
    """Extract frames using OpenCV (fallback path).

    Returns a list of ``(frame_index, timestamp_ms, image_bytes_b64, width, height)``
    tuples sampled at *fps_sample* frames per second.
    """
    import cv2  # type: ignore

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cv2 could not open video: {path}")

    step = max(1, int(round(native_fps / fps_sample)))
    results: list[tuple[int, float, str, int, int]] = []
    frame_idx = 0
    local_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            h, w = frame.shape[:2]
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                ts_ms = (frame_idx / native_fps) * 1000.0
                results.append((local_idx, ts_ms, b64, w, h))
                local_idx += 1
        frame_idx += 1

    cap.release()
    return results


def _get_video_metadata_cv2(path: str) -> dict:
    """Read basic video metadata using OpenCV."""
    import cv2  # type: ignore

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cv2 could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # CAP_PROP_CODEC_PIXEL_FORMAT is not the codec name; use fourcc instead
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc_int >> (i * 8)) & 0xFF) for i in range(4)]).strip("\x00")
    cap.release()

    duration_ms = (frame_count / fps) * 1000.0 if fps > 0 else 0.0
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "codec": codec,
        "duration_ms": duration_ms,
    }


def _detect_scenes(path: str, threshold: float) -> dict[int, int]:
    """Run PySceneDetect ContentDetector and return a mapping frame_idx -> scene_id.

    Returns an empty dict if scenedetect is not installed (caller must handle fallback).
    """
    try:
        from scenedetect import VideoManager, SceneManager  # type: ignore
        from scenedetect.detectors import ContentDetector  # type: ignore
    except ImportError:
        return {}

    video_manager = VideoManager([path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()

    # Build a simple list of (start_frame, end_frame, scene_id) tuples
    frame_to_scene: dict[int, int] = {}
    for scene_id, (start_tc, end_tc) in enumerate(scene_list):
        start_frame = start_tc.get_frames()
        end_frame = end_tc.get_frames()
        for f in range(start_frame, end_frame):
            frame_to_scene[f] = scene_id

    return frame_to_scene


def _extract_audio_moviepy(path: str) -> Optional[bytes]:
    """Extract the audio track from *path* to raw WAV bytes using moviepy.

    Returns ``None`` if moviepy is not installed or the video has no audio.
    """
    try:
        from moviepy.editor import VideoFileClip  # type: ignore
    except ImportError:
        return None

    buf = io.BytesIO()
    try:
        clip = VideoFileClip(path)
        if clip.audio is None:
            clip.close()
            return None
        # Write audio to an in-memory WAV buffer
        clip.audio.write_audiofile(
            buf,
            fps=16000,
            nbytes=2,
            codec="pcm_s16le",
            logger=None,
        )
        clip.close()
    except Exception:
        return None

    return buf.getvalue()


class VideoLoader:
    """Load video files into :class:`~raglib.multimodal.models.media.VideoChunk` objects.

    Frame extraction uses *decord* when available (faster), falling back to
    *opencv-python*.  Scene detection uses *PySceneDetect* when available.
    Audio extraction uses *moviepy* when available.

    Parameters
    ----------
    fps_sample:
        Number of frames to sample per second of video.  Lower values produce
        smaller outputs; the default of ``1.0`` samples one frame per second.
    max_frames_per_scene:
        Maximum number of frames to retain per detected scene.  Frames are
        taken evenly from within each scene's time window.
    extract_audio:
        If ``True`` (default), attempt to extract the audio track as raw WAV
        bytes and store it in :attr:`VideoChunk.audio_bytes`.
    scene_detection:
        If ``True`` (default), run PySceneDetect to identify scene boundaries
        and group frames by scene.  Falls back to a single scene when
        PySceneDetect is not installed.
    scene_threshold:
        Content-detector threshold passed to PySceneDetect.  Higher values
        result in fewer detected scenes.
    """

    supported_extensions: list[str] = _SUPPORTED_EXTENSIONS

    def __init__(
        self,
        fps_sample: float = 1.0,
        max_frames_per_scene: int = 10,
        extract_audio: bool = True,
        scene_detection: bool = True,
        scene_threshold: float = 27.0,
    ) -> None:
        self.fps_sample = fps_sample
        self.max_frames_per_scene = max_frames_per_scene
        self.extract_audio = extract_audio
        self.scene_detection = scene_detection
        self.scene_threshold = scene_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_load(self, path: str) -> bool:
        """Return ``True`` if *path* has a supported video extension."""
        return Path(path).suffix.lower() in self.supported_extensions

    def load(self, path: str) -> VideoChunk:
        """Load a video file and return a :class:`VideoChunk`.

        Parameters
        ----------
        path:
            Absolute or relative filesystem path to the video file.

        Returns
        -------
        VideoChunk
            Contains extracted frames (with base64-encoded JPEG bytes),
            optional audio bytes, and video metadata.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file extension is not supported.
        ImportError
            If neither *decord* nor *opencv-python* is installed.
        """
        p = Path(path).resolve()

        # --- 1. Validate path and extension -----------------------------------
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {p}")
        ext = p.suffix.lower()
        if ext not in self.supported_extensions:
            raise ValueError(
                f"Unsupported video extension {ext!r}. "
                f"Supported: {self.supported_extensions}"
            )

        # --- 2. Extract video metadata ----------------------------------------
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required to load videos. "
                "Install with: pip install raglib[video]"
            ) from exc

        meta = _get_video_metadata_cv2(str(p))
        native_fps: float = meta["fps"]
        duration_ms: float = meta["duration_ms"]
        width: int = meta["width"]
        height: int = meta["height"]
        codec: str = meta["codec"]

        # --- 3. Extract frames ------------------------------------------------
        raw_frames: list[tuple[int, float, str, int, int]]
        try:
            import decord  # noqa: F401

            raw_frames = _extract_frames_decord(str(p), self.fps_sample, native_fps)
        except ImportError:
            raw_frames = _extract_frames_cv2(str(p), self.fps_sample, native_fps)

        # --- 4. Scene detection and grouping ----------------------------------
        frame_to_scene: dict[int, int] = {}
        scene_count = 1

        if self.scene_detection:
            # frame indices here refer to the *native* video frame indices;
            # we need to map sampled frame timestamps back to native indices.
            frame_to_scene = _detect_scenes(str(p), self.scene_threshold)

        # Assign scene ids to sampled frames.
        # When scene detection is unavailable/disabled, every frame gets scene 0.
        frames_with_scene: list[tuple[int, float, str, int, int, int]] = []
        for local_idx, ts_ms, b64, fw, fh in raw_frames:
            native_idx = int(round((ts_ms / 1000.0) * native_fps))
            scene_id = frame_to_scene.get(native_idx, 0)
            frames_with_scene.append((local_idx, ts_ms, b64, fw, fh, scene_id))

        if frame_to_scene:
            scene_count = len(set(frame_to_scene.values()))

        # Group by scene and cap max_frames_per_scene
        from collections import defaultdict

        scene_groups: dict[int, list] = defaultdict(list)
        for entry in frames_with_scene:
            scene_groups[entry[5]].append(entry)

        kept_frames: list[tuple[int, float, str, int, int, int]] = []
        for scene_id in sorted(scene_groups.keys()):
            group = scene_groups[scene_id]
            if len(group) > self.max_frames_per_scene:
                # Evenly downsample within the scene
                step = len(group) / self.max_frames_per_scene
                group = [group[int(i * step)] for i in range(self.max_frames_per_scene)]
            kept_frames.extend(group)

        # Re-assign sequential frame_index values after filtering
        video_frames: list[VideoFrame] = []
        for new_idx, (_, ts_ms, b64, fw, fh, scene_id) in enumerate(kept_frames):
            video_id = p.stem
            video_frames.append(
                VideoFrame(
                    frame_id=f"{video_id}_f{new_idx}_{uuid.uuid4().hex[:6]}",
                    video_id=video_id,
                    frame_index=new_idx,
                    timestamp_ms=ts_ms,
                    image_bytes_b64=b64,
                    scene_id=str(scene_id),
                    metadata={"width": fw, "height": fh},
                )
            )

        # --- 5. Extract audio -------------------------------------------------
        audio_bytes: Optional[bytes] = None
        if self.extract_audio:
            audio_bytes = _extract_audio_moviepy(str(p))

        # --- 6. Build and return VideoChunk -----------------------------------
        video_id = p.stem + "_" + uuid.uuid4().hex[:8]
        return VideoChunk(
            id=video_id,
            video_id=video_id,
            frames=video_frames,
            start_ms=0.0,
            end_ms=duration_ms,
            metadata={
                "source_path": str(p),
                "fps": native_fps,
                "width": width,
                "height": height,
                "codec": codec,
                "duration_ms": duration_ms,
                "scene_count": scene_count,
                "audio_extracted": audio_bytes is not None,
                # Store audio as base64 in metadata to avoid breaking Pydantic
                # serialization (bytes fields are not JSON-serializable by default).
                "audio_bytes_b64": (
                    base64.b64encode(audio_bytes).decode("ascii")
                    if audio_bytes is not None
                    else None
                ),
            },
        )
