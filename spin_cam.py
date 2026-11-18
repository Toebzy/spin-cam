from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
import pyvirtualcam
from cv2_enumerate_cameras import CameraInfo, enumerate_cameras
from PIL import Image, ImageDraw, ImageTk
from pynput import keyboard


OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_FPS = 30
SPIN_DURATION = 1.1
SPIN_TURNS = 1
TILT_DEGREES = 12
BACKGROUND = (30, 33, 40)
HOTKEYS = ("F6", "F7", "F8", "F9", "F10", "F11", "F12")
TRANSITIONS = (
    "3D Spin",
    "Push",
    "Wipe",
    "Checkerboard",
    "Pixel Dissolve",
    "Iris",
    "Zoom",
    "Fade",
)
FILTERS = ("None", "Noir", "Sepia", "Negative", "Posterize", "VHS", "Thermal")
QUALITY_PRESETS = {
    "Clean (1280x720)": (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    "Retro (640x360)": (640, 360),
    "Crunchy (320x180)": (320, 180),
    "Extra crunchy (160x90)": (160, 90),
    "Potato (80x45)": (80, 45),
    "Cursed (40x22)": (40, 22),
}
DEFAULT_QUALITY = "Clean (1280x720)"
_checker_rows, _checker_columns = np.indices((18, 32))
_checker_diagonal = (_checker_rows + _checker_columns) / (
    _checker_rows.max() + _checker_columns.max()
)
_CHECKERBOARD_RANKS = (
    ((_checker_rows + _checker_columns) % 2) * 0.45 + _checker_diagonal * 0.55
)
_DISSOLVE_RANKS = (
    np.random.default_rng(17).permutation(22 * 40).reshape(22, 40) / (22 * 40 - 1)
)
_VHS_SCANLINES = np.ones((OUTPUT_HEIGHT, 1, 1), dtype=np.float32)
_VHS_SCANLINES[::4] = 0.72


def create_app_icon() -> Image.Image:
    """Draw a small camera icon with a circular spin arrow."""
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, 62, 62), radius=14, fill="#7557e8")
    draw.polygon([(17, 22), (22, 15), (39, 15), (45, 22)], fill="#f7f8ff")
    draw.rounded_rectangle((8, 21, 56, 52), radius=6, fill="#f7f8ff")
    draw.ellipse((19, 26, 44, 50), fill="#212846", outline="#54d8ed", width=3)
    draw.ellipse((27, 34, 36, 43), fill="#54d8ed")
    draw.ellipse((42, 40, 62, 60), fill="#ffc857")
    draw.arc((46, 43, 58, 57), start=205, end=25, fill="#212846", width=2)
    draw.polygon([(56, 43), (61, 45), (57, 49)], fill="#212846")
    return image


def crop_to_output(frame: np.ndarray) -> np.ndarray:
    """Center-crop a camera frame to 16:9 and resize it for virtual-camera output."""
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


def transform_frame(
    frame: np.ndarray, yaw_radians: float, scale: float = 1.0
) -> np.ndarray:
    """Project the camera image as a slightly tilted 3D plane."""
    width, height = OUTPUT_WIDTH, OUTPUT_HEIGHT
    half_width, half_height = width / 2, height / 2
    tilt = np.deg2rad(TILT_DEGREES)
    cos_yaw, sin_yaw = np.cos(yaw_radians), np.sin(yaw_radians)
    cos_tilt, sin_tilt = np.cos(tilt), np.sin(tilt)

    rotation_y = np.array(
        [[cos_yaw, 0, sin_yaw], [0, 1, 0], [-sin_yaw, 0, cos_yaw]]
    )
    rotation_x = np.array(
        [[1, 0, 0], [0, cos_tilt, -sin_tilt], [0, sin_tilt, cos_tilt]]
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
    center = np.array([half_width, half_height], dtype=np.float32)
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
    """Blend a frozen frame into the live camera using a slide-style transition."""
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
        result[:, : width - offset] = previous[:, offset:]
        return result
    if name == "Wipe":
        cutoff = int(width * progress)
        result = previous.copy()
        result[:, :cutoff] = current[:, :cutoff]
        return result
    if name == "Checkerboard":
        mask = cv2.resize(
            (_CHECKERBOARD_RANKS < progress).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        return np.where(mask[:, :, None] != 0, current, previous)
    if name == "Pixel Dissolve":
        mask = cv2.resize(
            (_DISSOLVE_RANKS < progress).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        return np.where(mask[:, :, None] != 0, current, previous)
    if name == "Iris":
        mask = np.zeros((height, width), dtype=np.uint8)
        radius = int(np.hypot(width / 2, height / 2) * progress)
        cv2.circle(mask, (width // 2, height // 2), radius, 1, thickness=-1)
        return np.where(mask[:, :, None] != 0, current, previous)
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
    raise ValueError(f"Unknown transition: {name}")


def apply_quality(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Render a deliberately pixelated camera image at the chosen preview size."""
    if size == (OUTPUT_WIDTH, OUTPUT_HEIGHT):
        return frame
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return cv2.resize(
        small, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_NEAREST
    )


def apply_filter(frame: np.ndarray, name: str) -> np.ndarray:
    """Apply a selected live-video filter to a BGR camera frame."""
    if name == "None":
        return frame
    if name == "Noir":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if name == "Sepia":
        matrix = np.array(
            [
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189],
            ],
            dtype=np.float32,
        )
        return cv2.transform(frame, matrix).clip(0, 255).astype(np.uint8)
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


class SpinCamApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("")
        self.root.minsize(820, 650)
        self.root.geometry("820x650")
        self.app_icon = ImageTk.PhotoImage(create_app_icon())
        self.root.iconphoto(True, self.app_icon)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.spin_started: float | None = None
        self.active_transition = "3D Spin"
        self.transition_snapshot: np.ndarray | None = None
        self.selected_quality = QUALITY_PRESETS[DEFAULT_QUALITY]
        self.selected_filter = "None"
        self.camera_devices: dict[str, CameraInfo] = {}
        self.device_scan_running = False
        self.stop_event: threading.Event | None = None
        self.camera_thread: threading.Thread | None = None
        self.hotkey_listener: keyboard.GlobalHotKeys | None = None

        self.camera_device = tk.StringVar(value="Looking for cameras...")
        self.transition = tk.StringVar(value="3D Spin")
        self.quality = tk.StringVar(value=DEFAULT_QUALITY)
        self.video_filter = tk.StringVar(value="None")
        self.hotkey = tk.StringVar(value="F8")
        self.status = tk.StringVar(value="Camera stopped")

        self._build_ui()
        self._refresh_cameras()
        self._start_hotkey_listener()
        self.root.after(30, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=20)
        container.pack(fill=tk.BOTH, expand=True)
        device_controls = ttk.LabelFrame(container, text="Webcam", padding=10)
        device_controls.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(device_controls, text="Choose a connected camera:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8)
        )
        self.camera_selector = ttk.Combobox(
            device_controls,
            textvariable=self.camera_device,
            state="disabled",
            width=48,
        )
        self.camera_selector.grid(row=0, column=1, sticky=tk.EW, padx=(0, 8))
        self.refresh_button = ttk.Button(
            device_controls, text="Refresh", command=self._refresh_cameras
        )
        self.refresh_button.grid(row=0, column=2, padx=(0, 8))
        self.connect_button = ttk.Button(
            device_controls,
            text="Connect webcam",
            command=self._connect_camera,
            state=tk.DISABLED,
        )
        self.connect_button.grid(row=0, column=3, padx=(0, 8))
        self.disconnect_button = ttk.Button(
            device_controls,
            text="Disconnect",
            command=self._stop_camera,
            state=tk.DISABLED,
        )
        self.disconnect_button.grid(row=0, column=4)
        device_controls.columnconfigure(1, weight=1)

        self.preview = tk.Label(
            container,
            text="No camera connected",
            bg="#1e2128",
            fg="#f0f0f0",
            font=("Segoe UI", 12),
            width=80,
            height=18,
        )
        self.preview.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(container, padding=(0, 14, 0, 0))
        controls.pack(fill=tk.X)
        self.spin_button = ttk.Button(
            controls, text="Play transition", command=self._play_transition
        )
        self.spin_button.grid(row=1, column=4, padx=(0, 16), sticky=tk.W)

        ttk.Label(controls, text="Transition").grid(row=0, column=0, sticky=tk.W)
        transition_selector = ttk.Combobox(
            controls,
            textvariable=self.transition,
            values=TRANSITIONS,
            state="readonly",
            width=18,
        )
        transition_selector.grid(row=1, column=0, padx=(0, 12), sticky=tk.W)
        transition_selector.bind(
            "<<ComboboxSelected>>", self._change_transition
        )

        ttk.Label(controls, text="Video quality").grid(
            row=0, column=1, sticky=tk.W
        )
        quality_selector = ttk.Combobox(
            controls,
            textvariable=self.quality,
            values=tuple(QUALITY_PRESETS),
            state="readonly",
            width=23,
        )
        quality_selector.grid(row=1, column=1, padx=(0, 12), sticky=tk.W)
        quality_selector.bind("<<ComboboxSelected>>", self._change_quality)

        ttk.Label(controls, text="Filter").grid(row=0, column=2, sticky=tk.W)
        filter_selector = ttk.Combobox(
            controls,
            textvariable=self.video_filter,
            values=FILTERS,
            state="readonly",
            width=14,
        )
        filter_selector.grid(row=1, column=2, padx=(0, 12), sticky=tk.W)
        filter_selector.bind("<<ComboboxSelected>>", self._change_filter)

        ttk.Label(controls, text="Hotkey").grid(row=0, column=3, sticky=tk.W)
        hotkey_selector = ttk.Combobox(
            controls,
            textvariable=self.hotkey,
            values=HOTKEYS,
            state="readonly",
            width=6,
        )
        hotkey_selector.grid(row=1, column=3, padx=(0, 12), sticky=tk.W)
        hotkey_selector.bind("<<ComboboxSelected>>", self._change_hotkey)
        ttk.Label(container, textvariable=self.status).pack(anchor=tk.W, pady=(12, 0))

    def _refresh_cameras(self) -> None:
        if self.device_scan_running:
            return
        self.device_scan_running = True
        self.refresh_button.configure(state=tk.DISABLED)
        self.camera_selector.configure(state=tk.DISABLED)
        self.connect_button.configure(state=tk.DISABLED)
        self.camera_device.set("Looking for cameras...")
        self.status.set("Searching for connected cameras...")
        threading.Thread(target=self._enumerate_cameras, daemon=True).start()

    def _enumerate_cameras(self) -> None:
        try:
            cameras = enumerate_cameras(cv2.CAP_DSHOW)
            if not cameras:
                cameras = enumerate_cameras(cv2.CAP_MSMF)
        except (RuntimeError, OSError, cv2.error) as error:
            self.events.put(("devices_error", str(error)))
        else:
            self.events.put(("devices", cameras))

    def _connect_camera(self) -> None:
        camera = self.camera_devices.get(self.camera_device.get())
        if camera is None:
            messagebox.showerror(
                "No webcam selected", "Choose an available webcam, then connect."
            )
            return

        self.stop_event = threading.Event()
        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            args=(camera, self.stop_event),
            daemon=True,
        )
        self.camera_thread.start()
        self.connect_button.configure(state=tk.DISABLED)
        self.disconnect_button.configure(state=tk.NORMAL)
        self.refresh_button.configure(state=tk.DISABLED)
        self.camera_selector.configure(state=tk.DISABLED)
        self.status.set(f"Connecting to {camera.name}...")

    def _stop_camera(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        self.disconnect_button.configure(state=tk.DISABLED)
        self.status.set("Stopping camera...")

    def _play_transition(self) -> None:
        with self.state_lock:
            self.spin_started = time.monotonic()
            self.active_transition = self.transition.get()
            self.transition_snapshot = None

    def _change_transition(self, _event: tk.Event) -> None:
        with self.state_lock:
            self.active_transition = self.transition.get()

    def _change_quality(self, _event: tk.Event) -> None:
        with self.state_lock:
            self.selected_quality = QUALITY_PRESETS[self.quality.get()]

    def _change_filter(self, _event: tk.Event) -> None:
        with self.state_lock:
            self.selected_filter = self.video_filter.get()

    def _start_hotkey_listener(self) -> None:
        self._stop_hotkey_listener()
        key = self.hotkey.get().lower()
        self.hotkey_listener = keyboard.GlobalHotKeys(
            {f"<{key}>": lambda: self.events.put(("transition", ""))}
        )
        try:
            self.hotkey_listener.start()
        except OSError as error:
            self.hotkey_listener = None
            messagebox.showerror(
                "Hotkey unavailable",
                f"Could not register the global hotkey {self.hotkey.get()}:\n{error}",
            )

    def _stop_hotkey_listener(self) -> None:
        if self.hotkey_listener is not None:
            self.hotkey_listener.stop()
            self.hotkey_listener = None

    def _change_hotkey(self, _event: tk.Event) -> None:
        self._start_hotkey_listener()
        self.status.set(f"Spin key set to {self.hotkey.get()}.")

    def _camera_loop(
        self, camera: CameraInfo, stop_event: threading.Event
    ) -> None:
        capture: cv2.VideoCapture | None = None
        virtual_camera: pyvirtualcam.Camera | None = None
        try:
            capture = cv2.VideoCapture(camera.index, camera.backend)
            if not capture.isOpened():
                raise RuntimeError(
                    f"Could not open {camera.name}. Check that it is connected "
                    "and not already in use."
                )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, OUTPUT_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, OUTPUT_HEIGHT)
            try:
                virtual_camera = pyvirtualcam.Camera(
                    width=OUTPUT_WIDTH,
                    height=OUTPUT_HEIGHT,
                    fps=OUTPUT_FPS,
                )
            except (RuntimeError, OSError) as error:
                self.events.put(("virtual_error", str(error)))
            else:
                self.events.put(
                    (
                        "status",
                        f"Connected to {camera.name}; "
                        f"sending to {virtual_camera.device}.",
                    )
                )

            while not stop_event.is_set():
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("The camera stopped sending frames.")

                frame = crop_to_output(frame)
                now = time.monotonic()
                with self.state_lock:
                    spin_started = self.spin_started
                    active_transition = self.active_transition
                    snapshot = self.transition_snapshot
                    selected_filter = self.selected_filter

                frame = apply_filter(frame, selected_filter)

                if spin_started is not None:
                    progress = min((now - spin_started) / SPIN_DURATION, 1.0)
                    eased = progress * progress * (3 - 2 * progress)
                    if active_transition == "3D Spin":
                        frame = render_transition(
                            frame, frame, active_transition, eased
                        )
                    else:
                        if snapshot is None:
                            snapshot = frame.copy()
                            with self.state_lock:
                                self.transition_snapshot = snapshot
                        frame = render_transition(
                            snapshot, frame, active_transition, eased
                        )
                    if progress >= 1:
                        with self.state_lock:
                            if self.spin_started == spin_started:
                                self.spin_started = None
                                self.transition_snapshot = None

                with self.state_lock:
                    quality_size = self.selected_quality
                frame = apply_quality(frame, quality_size)

                if virtual_camera is not None:
                    virtual_camera.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    virtual_camera.sleep_until_next_frame()

                with self.frame_lock:
                    self.latest_frame = frame

        except (RuntimeError, OSError, cv2.error) as error:
            self.events.put(("camera_error", str(error)))
        finally:
            if virtual_camera is not None:
                virtual_camera.close()
            if capture is not None:
                capture.release()
            with self.frame_lock:
                self.latest_frame = None
            self.events.put(("stopped", ""))

    def _poll(self) -> None:
        while True:
            try:
                event, detail = self.events.get_nowait()
            except queue.Empty:
                break

            if event == "transition":
                self._play_transition()
            elif event == "status":
                self.status.set(str(detail))
            elif event == "devices":
                cameras = detail
                assert isinstance(cameras, list)
                self.camera_devices = {
                    f"{camera.name} (device {camera.index})": camera
                    for camera in cameras
                }
                choices = list(self.camera_devices)
                self.camera_selector.configure(
                    values=choices, state="readonly" if choices else "disabled"
                )
                if choices:
                    self.camera_device.set(choices[0])
                    self.connect_button.configure(state=tk.NORMAL)
                    self.status.set(
                        f"Found {len(choices)} camera(s). Select one and connect."
                    )
                else:
                    self.camera_device.set("No cameras found")
                    self.status.set(
                        "No cameras found. Check the connection and click Refresh."
                    )
                self.device_scan_running = False
                self.refresh_button.configure(state=tk.NORMAL)
            elif event == "devices_error":
                self.device_scan_running = False
                self.camera_devices.clear()
                self.camera_selector.configure(values=(), state="disabled")
                self.camera_device.set("Camera scan failed")
                self.refresh_button.configure(state=tk.NORMAL)
                self.status.set(f"Camera scan failed: {detail}")
            elif event == "camera_error":
                messagebox.showerror("Camera error", str(detail))
                self.status.set("Camera stopped.")
            elif event == "virtual_error":
                self.status.set("Virtual camera unavailable.")
                messagebox.showerror(
                    "Virtual camera error",
                    "Could not connect to OBS Virtual Camera. Start it in OBS, "
                    f"then disconnect and reconnect the webcam.\n\n{detail}",
                )
            elif event == "stopped":
                self.connect_button.configure(
                    state=tk.NORMAL if self.camera_devices else tk.DISABLED
                )
                self.disconnect_button.configure(state=tk.DISABLED)
                self.refresh_button.configure(state=tk.NORMAL)
                if self.camera_devices:
                    self.camera_selector.configure(state="readonly")
                if self.status.get() not in (
                    "Camera stopped.",
                    "Virtual camera unavailable.",
                ):
                    self.status.set("Camera stopped.")

        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((950, 535), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=photo, text="")
            self.preview.image = photo
        elif self.camera_thread is None or not self.camera_thread.is_alive():
            self.preview.configure(image="", text="No camera connected")
            self.preview.image = None

        self.root.after(30, self._poll)

    def _close(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        self._stop_hotkey_listener()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SpinCamApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
