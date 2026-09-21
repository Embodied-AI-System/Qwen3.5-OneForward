"""Validated image/video decoding for multimodal OneForward requests."""

from __future__ import annotations

import base64
import binascii
import io
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError
from transformers.video_utils import VideoMetadata


MAX_ATTACHMENTS = 8
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_VIDEO_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_VIDEO_SECONDS = 180.0
IMAGE_MIME_TYPES = {"image/gif", "image/jpeg", "image/png", "image/webp"}
VIDEO_MIME_SUFFIXES = {
    "video/avi": ".avi",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
}


class MediaInputError(ValueError):
    """A safe client-facing error for invalid or unsupported media."""


@dataclass(frozen=True)
class PreparedMedia:
    kind: str
    name: str
    mime_type: str
    content: Any
    size_bytes: int
    width: int
    height: int
    frames: int | None = None
    duration_seconds: float | None = None
    video_metadata: VideoMetadata | None = None

    def content_block(self) -> dict[str, Any]:
        return {"type": self.kind, self.kind: self.content}

    def public_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.kind,
            "name": self.name,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
        }
        if self.frames is not None:
            result["sampled_frames"] = self.frames
        if self.duration_seconds is not None:
            result["duration_seconds"] = round(self.duration_seconds, 3)
        return result


def _field(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def decode_data_url(data_url: str, *, expected_kind: str) -> tuple[str, bytes]:
    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise MediaInputError("media must use a base64 data URL")
    header, separator, encoded = data_url.partition(",")
    if not separator or not header.endswith(";base64"):
        raise MediaInputError("media data URL must use base64 encoding")
    mime_type = header[5:-7].lower()
    allowed = IMAGE_MIME_TYPES if expected_kind == "image" else set(VIDEO_MIME_SUFFIXES)
    if mime_type not in allowed:
        raise MediaInputError(f"unsupported {expected_kind} media type: {mime_type or 'missing'}")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaInputError("media data URL contains invalid base64") from exc
    limit = MAX_IMAGE_BYTES if expected_kind == "image" else MAX_VIDEO_BYTES
    if not payload:
        raise MediaInputError("media file is empty")
    if len(payload) > limit:
        raise MediaInputError(f"{expected_kind} is too large ({len(payload)} bytes; limit {limit})")
    return mime_type, payload


def prepare_image(data: bytes, *, name: str, mime_type: str) -> PreparedMedia:
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.seek(0)
            image = source.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MediaInputError(f"could not decode image {name!r}") from exc
    width, height = image.size
    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
        raise MediaInputError(f"image {name!r} has unsupported dimensions {width}x{height}")
    return PreparedMedia("image", name, mime_type, image, len(data), width, height)


def prepare_video(
    data: bytes,
    *,
    name: str,
    mime_type: str,
    sample_fps: float,
    max_frames: int,
) -> PreparedMedia:
    if sample_fps <= 0 or max_frames < 1:
        raise ValueError("video sampling configuration must be positive")
    try:
        import cv2
    except ImportError as exc:
        raise MediaInputError("video input requires OpenCV") from exc
    path = ""
    capture = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="oneforward-video-", suffix=VIDEO_MIME_SUFFIXES[mime_type], delete=False
        ) as temporary:
            temporary.write(data)
            path = temporary.name
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise MediaInputError(f"could not decode video {name!r}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if total_frames < 1 or source_fps <= 0 or width < 1 or height < 1:
            raise MediaInputError(f"video {name!r} has invalid stream metadata")
        duration = total_frames / source_fps
        if duration > MAX_VIDEO_SECONDS:
            raise MediaInputError(
                f"video {name!r} is {duration:.1f}s; limit is {MAX_VIDEO_SECONDS:.0f}s"
            )
        if width * height > MAX_IMAGE_PIXELS:
            raise MediaInputError(f"video {name!r} has unsupported dimensions {width}x{height}")

        step = max(source_fps / sample_fps, 1.0)
        requested = np.arange(0, total_frames, step, dtype=np.float64).round().astype(int)
        requested = np.unique(np.clip(requested, 0, total_frames - 1))
        if len(requested) > max_frames:
            positions = np.linspace(0, len(requested) - 1, max_frames).round().astype(int)
            requested = requested[positions]
        frames: list[np.ndarray] = []
        actual_indices: list[int] = []
        for index in requested.tolist():
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                actual_indices.append(index)
        if not frames:
            raise MediaInputError(f"video {name!r} contains no readable frames")
        metadata = VideoMetadata(
            total_num_frames=total_frames,
            fps=source_fps,
            width=width,
            height=height,
            duration=duration,
            video_backend="opencv",
            frames_indices=actual_indices,
        )
        return PreparedMedia(
            "video",
            name,
            mime_type,
            np.stack(frames),
            len(data),
            width,
            height,
            len(frames),
            duration,
            metadata,
        )
    finally:
        if capture is not None:
            capture.release()
        if path:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def prepare_media(
    items: Sequence[Any], *, video_fps: float, max_video_frames: int
) -> tuple[PreparedMedia, ...]:
    if len(items) > MAX_ATTACHMENTS:
        raise MediaInputError(f"at most {MAX_ATTACHMENTS} media attachments are allowed")
    if sum(_field(item, "type") == "video" for item in items) > 1:
        raise MediaInputError("at most one video is allowed per request")
    prepared: list[PreparedMedia] = []
    for index, item in enumerate(items):
        kind = _field(item, "type")
        if kind not in {"image", "video"}:
            raise MediaInputError(f"media item {index} has invalid type")
        name = str(_field(item, "name") or f"{kind}-{index + 1}")[:200]
        mime_type, data = decode_data_url(_field(item, "data_url"), expected_kind=kind)
        if kind == "image":
            prepared.append(prepare_image(data, name=name, mime_type=mime_type))
        else:
            prepared.append(
                prepare_video(
                    data,
                    name=name,
                    mime_type=mime_type,
                    sample_fps=video_fps,
                    max_frames=max_video_frames,
                )
            )
    return tuple(prepared)
