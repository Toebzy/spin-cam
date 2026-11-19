from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.config import (
    BACKGROUND,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SPIN_TURNS,
    TILT_DEGREES,
)

_VHS_SCANLINES = np.ones((OUTPUT_HEIGHT, 1, 1), dtype=np.float32)
_VHS_SCANLINES[::4] = 0.72
_SEPIA_MATRIX = np.array(
    [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]],
    dtype=np.float32,
)


@dataclass
class PingPongMotionState:
    x: float = 0.5
    y: float = 0.5
    velocity_x: float = 0.28
    velocity_y: float = 0.18

    def advance(self, delta_seconds: float) -> None:
        self.x, self.velocity_x = self._advance_axis(
            self.x, self.velocity_x, delta_seconds
        )
        self.y, self.velocity_y = self._advance_axis(
            self.y, self.velocity_y, delta_seconds
        )

    @staticmethod
    def _advance_axis(
        position: float,
        velocity: float,
        delta_seconds: float,
    ) -> tuple[float, float]:
        speed = abs(velocity)
        unfolded = position if velocity >= 0 else 2.0 - position
        unfolded = (unfolded + speed * max(0.0, delta_seconds)) % 2.0
        if unfolded <= 1.0:
            return unfolded, speed
        return 2.0 - unfolded, -speed


def create_app_icon() -> Image.Image:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, 62, 62), radius=14, fill="#7557e8")
    draw.polygon([(17, 22), (22, 15), (39, 15), (45, 22)], fill="#f7f8ff")
    draw.rounded_rectangle((8, 21, 56, 52), radius=6, fill="#f7f8ff")
    draw.ellipse((19, 26, 44, 50), fill="#212846", outline="#54d8ed", width=3)
    draw.ellipse((27, 34, 36, 43), fill="#54d8ed")
    return image


def fit_to_output(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    target_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT
    source_ratio = width / height
    if source_ratio > target_ratio:
        cropped_width = int(height * target_ratio)
        left = (width - cropped_width) // 2
        frame = frame[:, left : left + cropped_width]
    else:
        cropped_height = int(width / target_ratio)
        top = (height - cropped_height) // 2
        frame = frame[top : top + cropped_height, :]
    interpolation = (
        cv2.INTER_AREA
        if frame.shape[1] >= OUTPUT_WIDTH and frame.shape[0] >= OUTPUT_HEIGHT
        else cv2.INTER_LINEAR
    )
    return cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=interpolation)


def stretch_to_output(frame: np.ndarray) -> np.ndarray:
    interpolation = (
        cv2.INTER_AREA
        if frame.shape[1] >= OUTPUT_WIDTH and frame.shape[0] >= OUTPUT_HEIGHT
        else cv2.INTER_LINEAR
    )
    return cv2.resize(
        frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=interpolation
    )


def apply_native_crop(
    frame: np.ndarray,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = int(np.clip(left, 0.0, 0.45) * width)
    y1 = int(np.clip(top, 0.0, 0.45) * height)
    x2 = int((1.0 - np.clip(right, 0.0, 0.45)) * width)
    y2 = int((1.0 - np.clip(bottom, 0.0, 0.45)) * height)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def shift_crop_margins(
    margins: tuple[float, float, float, float],
    delta_x: float,
    delta_y: float,
) -> tuple[float, float, float, float]:
    left, top, right, bottom = margins
    clamped_x = float(np.clip(delta_x, -left, right))
    clamped_y = float(np.clip(delta_y, -top, bottom))
    return (
        left + clamped_x,
        top + clamped_y,
        right - clamped_x,
        bottom - clamped_y,
    )


def interpolate_shape_corners(
    source: tuple[tuple[float, float], ...],
    target: tuple[tuple[float, float], ...],
    progress: float,
) -> tuple[tuple[float, float], ...]:
    amount = float(np.clip(progress, 0.0, 1.0))
    amount = amount * amount * (3.0 - 2.0 * amount)
    return tuple(
        (
            start_x + (end_x - start_x) * amount,
            start_y + (end_y - start_y) * amount,
        )
        for (start_x, start_y), (end_x, end_y) in zip(source, target)
    )


def transform_frame(
    frame: np.ndarray, yaw_radians: float, scale: float = 1.0
) -> np.ndarray:
    width, height = OUTPUT_WIDTH, OUTPUT_HEIGHT
    half_width, half_height = width / 2, height / 2
    tilt = np.deg2rad(TILT_DEGREES)
    cos_yaw, sin_yaw = np.cos(yaw_radians), np.sin(yaw_radians)
    cos_tilt, sin_tilt = np.cos(tilt), np.sin(tilt)
    rotation_y = np.array(
        [[cos_yaw, 0, sin_yaw], [0, 1, 0], [-sin_yaw, 0, cos_yaw]],
        dtype=np.float64,
    )
    rotation_x = np.array(
        [[1, 0, 0], [0, cos_tilt, -sin_tilt], [0, sin_tilt, cos_tilt]],
        dtype=np.float64,
    )
    corners = np.array(
        [
            [-half_width, -half_height, 0],
            [half_width, -half_height, 0],
            [half_width, half_height, 0],
            [-half_width, half_height, 0],
        ],
        dtype=np.float64,
    )
    rotated = corners @ (rotation_y @ rotation_x).T
    focal_length = width * 1.8
    perspective = focal_length / (focal_length - rotated[:, 2])
    destination = np.column_stack(
        (
            half_width + rotated[:, 0] * perspective,
            half_height + rotated[:, 1] * perspective,
        )
    ).astype(np.float32)
    center = np.array((half_width, half_height), dtype=np.float32)
    destination = center + (destination - center) * scale
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BACKGROUND,
    )


def render_transition(
    previous: np.ndarray,
    current: np.ndarray,
    name: str,
    progress: float,
) -> np.ndarray:
    progress = float(np.clip(progress, 0, 1))
    if progress >= 1:
        return current
    if name == "3D Spin":
        zoom_pulse = 1 + 0.16 * np.sin(np.pi * progress)
        return transform_frame(
            current, progress * 2 * np.pi * SPIN_TURNS, zoom_pulse
        )
    if progress <= 0:
        return previous.copy()
    height, width = current.shape[:2]
    if name == "Fade":
        return cv2.addWeighted(previous, 1 - progress, current, progress, 0)
    if name == "Push":
        offset = int(width * progress)
        result = current.copy()
        if offset < width:
            result[:, : width - offset] = previous[:, offset:]
        return result
    if name == "Zoom":
        scale = 1 - progress
        scaled_width = max(1, int(width * scale))
        scaled_height = max(1, int(height * scale))
        scaled = cv2.resize(previous, (scaled_width, scaled_height))
        result = current.copy()
        left = (width - scaled_width) // 2
        top = (height - scaled_height) // 2
        result[top : top + scaled_height, left : left + scaled_width] = scaled
        return result
    if name == "Glitch":
        result = previous.copy()
        band_height = 32
        for band, top in enumerate(range(0, height, band_height)):
            band_progress = ((band * 5 + 3) % 8) / 8
            if band_progress > progress:
                continue
            bottom = min(top + band_height, height)
            intensity = int((18 + band * 11) * (1 - progress))
            shift = intensity if band % 2 == 0 else -intensity
            shifted = np.roll(current[top:bottom], shift, axis=1)
            shifted[:, :, 0] = np.roll(shifted[:, :, 0], -intensity // 2, axis=1)
            shifted[:, :, 2] = np.roll(shifted[:, :, 2], intensity // 2, axis=1)
            result[top:bottom] = shifted
        return result
    raise ValueError(f"Unknown transition: {name}")


def apply_quality(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if size == (OUTPUT_WIDTH, OUTPUT_HEIGHT):
        return frame
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return cv2.resize(
        small, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_NEAREST
    )


def apply_filter(frame: np.ndarray, name: str, time_seconds: float = 0.0) -> np.ndarray:
    if name == "None":
        return frame
    if name == "Noir":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if name == "Sepia":
        return cv2.transform(frame, _SEPIA_MATRIX).clip(0, 255).astype(np.uint8)
    if name == "Negative":
        return cv2.bitwise_not(frame)
    if name == "Posterize":
        return np.minimum((frame // 64) * 64 + 32, 255).astype(np.uint8)
    if name == "VHS":
        result = frame.copy()
        result[:, :, 2] = np.roll(frame[:, :, 2], 3, axis=1)
        return np.clip(
            result.astype(np.float32) * _VHS_SCANLINES, 0, 255
        ).astype(np.uint8)
    if name == "Thermal":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    raise ValueError(f"Unknown filter: {name}")


def apply_corner_transform(
    frame: np.ndarray,
    corners: tuple[tuple[float, float], ...],
) -> np.ndarray:
    height, width = frame.shape[:2]
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    destination = np.array(
        [[point[0] * width, point[1] * height] for point in corners],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BACKGROUND,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(destination).astype(np.int32), 255)
    edge = np.clip(np.abs(cv2.Laplacian(mask, cv2.CV_32F, ksize=3)) * 0.18, 0, 20)
    return np.clip(warped.astype(np.float32) - edge[:, :, None], 0, 255).astype(np.uint8)


def apply_passive_motion(
    frame: np.ndarray,
    mode: str,
    time_seconds: float,
) -> np.ndarray:
    if mode == "Off":
        return frame

    height, width = frame.shape[:2]
    destination = passive_motion_destination(width, height, mode, time_seconds)
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BACKGROUND,
    )


def apply_bounded_ping_pong(
    frame: np.ndarray,
    corners: tuple[tuple[float, float], ...],
    x_position: float,
    y_position: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    points = np.asarray(corners, dtype=np.float64)
    left = float(points[:, 0].min())
    right = float(points[:, 0].max())
    top = float(points[:, 1].min())
    bottom = float(points[:, 1].max())
    tx_min = -left * width
    tx_max = (1.0 - right) * width
    ty_min = -top * height
    ty_max = (1.0 - bottom) * height
    tx = tx_min + np.clip(x_position, 0.0, 1.0) * (tx_max - tx_min)
    ty = ty_min + np.clip(y_position, 0.0, 1.0) * (ty_max - ty_min)
    tx = float(np.clip(np.rint(tx), np.ceil(tx_min), np.floor(tx_max)))
    ty = float(np.clip(np.rint(ty), np.ceil(ty_min), np.floor(ty_max)))
    matrix = np.array(
        [[1.0, 0.0, tx], [0.0, 1.0, ty]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BACKGROUND,
    )


def apply_positioned_ping_pong(
    frame: np.ndarray,
    x_position: float,
    y_position: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = 0.88
    tx = (np.clip(x_position, 0.0, 1.0) - 0.5) * width * (1.0 - scale)
    ty = (np.clip(y_position, 0.0, 1.0) - 0.5) * height * (1.0 - scale)
    plane_width = width * scale
    plane_height = height * scale
    center_x = width / 2 + tx
    center_y = height / 2 + ty
    destination = np.array(
        [
            [center_x - plane_width / 2, center_y - plane_height / 2],
            [center_x + plane_width / 2, center_y - plane_height / 2],
            [center_x + plane_width / 2, center_y + plane_height / 2],
            [center_x - plane_width / 2, center_y + plane_height / 2],
        ],
        dtype=np.float32,
    )
    source = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BACKGROUND,
    )


def passive_motion_destination(
    width: int,
    height: int,
    mode: str,
    time_seconds: float,
) -> np.ndarray:
    if mode == "Ping-Pong":
        x_phase = ping_pong_phase(time_seconds * 0.28)
        y_phase = ping_pong_phase(time_seconds * 0.18)
        scale = 0.88
        tx = x_phase * width * (1 - scale) / 2
        ty = y_phase * height * (1 - scale) / 2
    elif mode == "Orbit":
        angle = time_seconds * 0.55
        tx = np.sin(angle) * width * 0.035
        ty = np.sin(angle * 1.7) * height * 0.055
        scale = 0.94 + np.sin(angle * 0.5) * 0.025
    elif mode == "Drift":
        tx = np.sin(time_seconds * 0.34) * width * 0.065
        ty = np.sin(time_seconds * 0.23 + 1.4) * height * 0.05
        scale = 0.91 + np.sin(time_seconds * 0.28) * 0.035
    elif mode == "Hover":
        tx = np.sin(time_seconds * 0.82) * width * 0.018
        ty = np.sin(time_seconds * 1.31) * height * 0.026
        scale = 0.96 + np.sin(time_seconds * 1.05) * 0.018
    else:
        raise ValueError(f"Unknown passive movement: {mode}")

    center_x = width / 2 + tx
    center_y = height / 2 + ty
    plane_width = width * scale
    plane_height = height * scale
    if mode == "Orbit":
        tilt = np.sin(time_seconds * 0.55) * width * 0.035
        destination = np.array(
            [
                [center_x - plane_width / 2 + tilt, center_y - plane_height / 2],
                [center_x + plane_width / 2 + tilt, center_y - plane_height / 2],
                [center_x + plane_width / 2 - tilt, center_y + plane_height / 2],
                [center_x - plane_width / 2 - tilt, center_y + plane_height / 2],
            ],
            dtype=np.float32,
        )
    else:
        destination = np.array(
            [
                [center_x - plane_width / 2, center_y - plane_height / 2],
                [center_x + plane_width / 2, center_y - plane_height / 2],
                [center_x + plane_width / 2, center_y + plane_height / 2],
                [center_x - plane_width / 2, center_y + plane_height / 2],
            ],
            dtype=np.float32,
        )
    return destination


def ping_pong_phase(time_seconds: float, offset: float = 0.0) -> float:
    cycle = (time_seconds + offset) % 4.0
    if cycle <= 1.0:
        return cycle * 2.0 - 1.0
    if cycle <= 3.0:
        return 1.0 - (cycle - 1.0)
    return (cycle - 3.0) * 2.0 - 1.0
