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
from PIL import Image, ImageTk
from pynput import keyboard


class CameraConnectionError(RuntimeError):
    def __init__(self, title: str, message: str) -> None:
        super().__init__(message)
        self.title = title


def classify_camera_error(camera_name: str, error: BaseException) -> CameraConnectionError:
    text = str(error).lower()
    if isinstance(error, CameraConnectionError):
        return error
    if "busy" in text or "used" in text or "in use" in text:
        return CameraConnectionError(
            "Camera already in use",
            f"{camera_name} is already being used by another app.\n\n"
            "Close apps that use the camera, then try connecting again.",
        )
    if "not opened" in text or "could not open" in text:
        return CameraConnectionError(
            "Camera unavailable",
            f"Could not open {camera_name}.\n\n"
            "It may be disconnected or reserved by another app.",
        )
    return CameraConnectionError(
        "Camera stopped",
        f"{camera_name} stopped sending frames.\n\n"
        "Reconnect or refresh the camera and try again.",
    )

from core.config import (
    DEFAULT_QUALITY,
    FILTERS,
    HOTKEYS,
    OUTPUT_FPS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PASSIVE_MOVEMENTS,
    QUALITY_PRESETS,
    SHAPE_TRANSITION_DURATION,
    SPIN_DURATION,
    TRANSFORM_CORNERS,
    TRANSFORM_PRESETS,
    TRANSITIONS,
)
from core.video_processing import (
    apply_bounded_ping_pong,
    apply_positioned_ping_pong,
    PingPongMotionState,
    apply_corner_transform,
    apply_filter,
    apply_native_crop,
    apply_quality,
    create_app_icon,
    fit_to_output,
    apply_passive_motion,
    interpolate_shape_corners,
    render_transition,
    shift_crop_margins,
    stretch_to_output,
)


class SpinCamApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("")
        self.root.minsize(720, 560)
        self.root.geometry("720x560")
        self.app_icon = ImageTk.PhotoImage(create_app_icon())
        self.root.iconphoto(True, self.app_icon)

        self.events: queue.Queue[tuple[str, object, object | None]] = queue.Queue()
        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.latest_frame: np.ndarray | None = None
        self.latest_preview_frame: np.ndarray | None = None
        self.spin_started: float | None = None
        self.active_transition = "3D Spin"
        self.transition_snapshot: np.ndarray | None = None
        self.selected_quality = QUALITY_PRESETS[DEFAULT_QUALITY]
        self.selected_filter = "None"
        self.selected_transform = "Off"
        self.selected_passive_movement = "Off"
        self.passive_motion_started: float | None = None
        self.transform_corners = [
            list(point) for point in TRANSFORM_CORNERS["Off"]
        ]
        self.shape_transition_from: tuple[tuple[float, float], ...] | None = None
        self.shape_transition_started: float | None = None
        self.dragged_corner: int | None = None
        self.crop_enabled = False
        self.crop_left = 0.0
        self.crop_top = 0.0
        self.crop_right = 0.0
        self.crop_bottom = 0.0
        self.dragged_crop_edge: str | None = None
        self.dragged_crop_start: tuple[float, float] | None = None
        self.dragged_crop_margins: tuple[float, float, float, float] | None = None
        self.camera_devices: dict[str, CameraInfo] = {}
        self.device_scan_running = False
        self.stop_event: threading.Event | None = None
        self.camera_thread: threading.Thread | None = None
        self.hotkey_listener: keyboard.GlobalHotKeys | None = None

        self.camera_device = tk.StringVar(value="Looking for cameras...")
        self.transition = tk.StringVar(value="3D Spin")
        self.quality = tk.StringVar(value=DEFAULT_QUALITY)
        self.video_filter = tk.StringVar(value="None")
        self.transform = tk.StringVar(value="Off")
        self.passive_movement = tk.StringVar(value="Off")
        self.crop_enabled_var = tk.BooleanVar(value=False)
        self.hotkey = tk.StringVar(value="F8")
        self.status = tk.StringVar(value="Camera stopped")
        self.preview_image_size = (0, 0)

        self._build_ui()
        self._refresh_cameras()
        self._start_hotkey_listener()
        self.root.after(30, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=(20, 0, 20, 20))
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

        self.preview = tk.Canvas(
            container,
            bg="#000000",
            highlightthickness=0,
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

        frame_controls = ttk.Frame(container, padding=(0, 12, 0, 0))
        frame_controls.pack(fill=tk.X)

        shape_controls = ttk.LabelFrame(frame_controls, text="Shape", padding=8)
        shape_controls.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Label(shape_controls, text="Preset").pack(side=tk.LEFT)
        transform_selector = ttk.Combobox(
            shape_controls,
            textvariable=self.transform,
            values=TRANSFORM_PRESETS,
            state="readonly",
            width=16,
        )
        transform_selector.pack(side=tk.LEFT, padx=(6, 8))
        transform_selector.bind("<<ComboboxSelected>>", self._change_transform)
        ttk.Button(
            shape_controls, text="Reset", command=self._reset_transform
        ).pack(side=tk.LEFT)

        motion_controls = ttk.LabelFrame(
            frame_controls, text="Passive movement", padding=8
        )
        motion_controls.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Label(motion_controls, text="Motion").pack(side=tk.LEFT)
        movement_selector = ttk.Combobox(
            motion_controls,
            textvariable=self.passive_movement,
            values=PASSIVE_MOVEMENTS,
            state="readonly",
            width=12,
        )
        movement_selector.pack(side=tk.LEFT, padx=(6, 0))
        movement_selector.bind(
            "<<ComboboxSelected>>", self._change_passive_movement
        )

        crop_controls = ttk.LabelFrame(frame_controls, text="Crop", padding=8)
        crop_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Checkbutton(
            crop_controls,
            text="Enable crop",
            variable=self.crop_enabled_var,
            command=self._toggle_crop,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            crop_controls, text="Reset", command=self._reset_crop
        ).pack(side=tk.RIGHT)

        self.preview.bind("<ButtonPress-1>", self._start_edit_drag)
        self.preview.bind("<B1-Motion>", self._edit_drag)
        self.preview.bind("<ButtonRelease-1>", self._stop_edit_drag)
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
            self.events.put(
                ("devices_error", "Could not scan for cameras.", str(error))
            )
        else:
            self.events.put(("devices", cameras))

    def _connect_camera(self) -> None:
        camera = self.camera_devices.get(self.camera_device.get())
        if camera is None:
            messagebox.showerror(
                "No webcam selected", "Choose an available webcam, then connect."
            )
            return
        if self.camera_thread is not None and self.camera_thread.is_alive():
            return

        with self.state_lock:
            if self.selected_passive_movement != "Off":
                self.passive_motion_started = time.monotonic()

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

    def _toggle_crop(self) -> None:
        with self.state_lock:
            self.crop_enabled = self.crop_enabled_var.get()
        self.status.set(
            "Crop enabled. Drag the cyan edges directly."
            if self.crop_enabled
            else "Crop disabled."
        )

    def _change_passive_movement(self, _event: tk.Event) -> None:
        mode = self.passive_movement.get()
        with self.state_lock:
            self.selected_passive_movement = mode
            self.passive_motion_started = (
                time.monotonic() if mode != "Off" else None
            )
        self.status.set(
            f"Passive movement: {mode}."
            if mode != "Off"
            else "Passive movement off."
        )

    def _reset_crop(self) -> None:
        self.crop_enabled_var.set(False)
        with self.state_lock:
            self.crop_enabled = False
            self.crop_left = 0.0
            self.crop_top = 0.0
            self.crop_right = 0.0
            self.crop_bottom = 0.0
        self.status.set("Crop reset.")

    def _change_transform(self, _event: tk.Event) -> None:
        preset = self.transform.get()
        with self.state_lock:
            source = self._shape_corners_at_locked(time.monotonic())
            self.selected_transform = preset
            self.transform_corners = [
                list(point) for point in TRANSFORM_CORNERS[preset]
            ]
            self.shape_transition_from = source
            self.shape_transition_started = time.monotonic()
        self.status.set(
            f"{preset} shape active. Drag any of the four preview corners."
            if preset != "Off"
            else "Shape reset to full frame."
        )

    def _reset_transform(self) -> None:
        self.transform.set("Off")
        with self.state_lock:
            source = self._shape_corners_at_locked(time.monotonic())
            self.selected_transform = "Off"
            self.transform_corners = [
                list(point) for point in TRANSFORM_CORNERS["Off"]
            ]
            self.shape_transition_from = source
            self.shape_transition_started = time.monotonic()
        self.status.set("Shape reset to full frame.")

    def _shape_corners_at_locked(
        self, now: float
    ) -> tuple[tuple[float, float], ...]:
        target = tuple(tuple(point) for point in self.transform_corners)
        if (
            self.shape_transition_from is None
            or self.shape_transition_started is None
        ):
            return target
        progress = (now - self.shape_transition_started) / SHAPE_TRANSITION_DURATION
        return interpolate_shape_corners(
            self.shape_transition_from, target, progress
        )

    def _preview_point_to_normalized(self, x: int, y: int) -> tuple[float, float]:
        preview_width = self.preview.winfo_width()
        preview_height = self.preview.winfo_height()
        image_width, image_height = self.preview_image_size
        if preview_width <= 0 or preview_height <= 0 or not image_width:
            return (0.0, 0.0)
        offset_x = max(0, (preview_width - image_width) // 2)
        offset_y = max(0, (preview_height - image_height) // 2)
        return (
            float(np.clip((x - offset_x) / image_width, -0.20, 1.20)),
            float(np.clip((y - offset_y) / image_height, -0.20, 1.20)),
        )

    def _start_edit_drag(self, event: tk.Event) -> None:
        point = self._preview_point_to_normalized(event.x, event.y)
        if self.crop_enabled:
            distances = {
                "left": abs(point[0] - self.crop_left),
                "top": abs(point[1] - self.crop_top),
                "right": abs(point[0] - (1.0 - self.crop_right)),
                "bottom": abs(point[1] - (1.0 - self.crop_bottom)),
            }
            edge = min(distances, key=distances.get)
            if distances[edge] <= 0.12:
                self.dragged_crop_edge = edge
                self.dragged_corner = None
                return
            left = self.crop_left
            top = self.crop_top
            right = 1.0 - self.crop_right
            bottom = 1.0 - self.crop_bottom
            if left <= point[0] <= right and top <= point[1] <= bottom:
                self.dragged_crop_start = point
                self.dragged_crop_margins = (
                    left,
                    top,
                    self.crop_right,
                    self.crop_bottom,
                )
                self.dragged_corner = None
                return

        nearest = min(
            range(4),
            key=lambda index: (
                (point[0] - self.transform_corners[index][0]) ** 2
                + (point[1] - self.transform_corners[index][1]) ** 2
            ),
        )
        corner = self.transform_corners[nearest]
        distance = ((point[0] - corner[0]) ** 2 + (point[1] - corner[1]) ** 2) ** 0.5
        if distance <= 0.18:
            with self.state_lock:
                current = self._shape_corners_at_locked(time.monotonic())
                self.transform_corners = [list(point) for point in current]
                self.shape_transition_from = None
                self.shape_transition_started = None
            self.dragged_corner = nearest

    def _edit_drag(self, event: tk.Event) -> None:
        point = self._preview_point_to_normalized(event.x, event.y)
        if (
            self.dragged_crop_start is not None
            and self.dragged_crop_margins is not None
        ):
            left, top, right, bottom = shift_crop_margins(
                self.dragged_crop_margins,
                point[0] - self.dragged_crop_start[0],
                point[1] - self.dragged_crop_start[1],
            )
            with self.state_lock:
                self.crop_left = left
                self.crop_top = top
                self.crop_right = right
                self.crop_bottom = bottom
            return

        if self.dragged_crop_edge is not None:
            edge = self.dragged_crop_edge
            if edge == "left":
                value = float(np.clip(point[0], 0.0, 1.0 - self.crop_right - 0.10))
                with self.state_lock:
                    self.crop_left = value
            elif edge == "top":
                value = float(np.clip(point[1], 0.0, 1.0 - self.crop_bottom - 0.10))
                with self.state_lock:
                    self.crop_top = value
            elif edge == "right":
                value = float(np.clip(1.0 - point[0], 0.0, 1.0 - self.crop_left - 0.10))
                with self.state_lock:
                    self.crop_right = value
            else:
                value = float(np.clip(1.0 - point[1], 0.0, 1.0 - self.crop_top - 0.10))
                with self.state_lock:
                    self.crop_bottom = value
            return

        if self.dragged_corner is None:
            return
        with self.state_lock:
            self.transform_corners[self.dragged_corner] = list(point)
            self.selected_transform = "Custom"
        self.transform.set("Custom")

    def _stop_edit_drag(self, _event: tk.Event) -> None:
        self.dragged_corner = None
        self.dragged_crop_edge = None
        self.dragged_crop_start = None
        self.dragged_crop_margins = None

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
                raise CameraConnectionError(
                    "Camera unavailable",
                    f"{camera.name} could not be opened. It may be disconnected "
                    "or already in use by another app.",
                )
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, OUTPUT_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, OUTPUT_HEIGHT)
            capture.set(cv2.CAP_PROP_FPS, OUTPUT_FPS)
            for _ in range(5):
                success, frame = capture.read()
                if success and frame is not None and frame.size:
                    break
                time.sleep(0.1)
            else:
                raise CameraConnectionError(
                    "Camera already in use",
                    f"{camera.name} opened but returned no frames.\n\n"
                    "Close apps using the camera and try again.",
                )
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

            ping_pong_state = PingPongMotionState()
            ping_pong_started: float | None = None
            previous_frame_time = time.monotonic()
            while not stop_event.is_set():
                success, frame = capture.read()
                if not success or frame is None or not frame.size:
                    raise CameraConnectionError(
                        "Camera stopped",
                        f"{camera.name} stopped sending frames.\n\n"
                        "Reconnect or refresh the camera and try again.",
                    )
                now = time.monotonic()
                with self.state_lock:
                    spin_started = self.spin_started
                    active_transition = self.active_transition
                    snapshot = self.transition_snapshot
                    selected_filter = self.selected_filter
                    selected_transform = self.selected_transform
                    transform_corners = tuple(
                        point
                        for point in self._shape_corners_at_locked(now)
                    )
                    shape_transition_active = (
                        self.shape_transition_from is not None
                        and self.shape_transition_started is not None
                    )
                    if (
                        shape_transition_active
                        and now - self.shape_transition_started
                        >= SHAPE_TRANSITION_DURATION
                    ):
                        self.shape_transition_from = None
                        self.shape_transition_started = None
                    selected_passive_movement = self.selected_passive_movement
                    passive_motion_started = self.passive_motion_started
                    crop_enabled = self.crop_enabled
                    crop_margins = (
                        self.crop_left,
                        self.crop_top,
                        self.crop_right,
                        self.crop_bottom,
                    )
                if selected_passive_movement == "Ping-Pong":
                    if ping_pong_started != passive_motion_started:
                        ping_pong_state = PingPongMotionState()
                        ping_pong_started = passive_motion_started
                        previous_frame_time = now
                    else:
                        ping_pong_state.advance(now - previous_frame_time)
                        previous_frame_time = now
                preview_frame = frame.copy()
                if crop_enabled:
                    frame = apply_native_crop(frame, *crop_margins)
                frame = apply_filter(frame, selected_filter, now)
                frame = stretch_to_output(frame) if crop_enabled else fit_to_output(frame)

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

                shape_transform_active = (
                    selected_transform != "Off" or shape_transition_active
                )
                is_bounded_ping_pong = (
                    selected_passive_movement == "Ping-Pong"
                    and shape_transform_active
                )
                if (
                    selected_passive_movement != "Off"
                    and passive_motion_started is not None
                    and not is_bounded_ping_pong
                    and selected_passive_movement != "Ping-Pong"
                ):
                    frame = apply_passive_motion(
                        frame,
                        selected_passive_movement,
                        now - passive_motion_started,
                    )
                if shape_transform_active:
                    frame = apply_corner_transform(frame, transform_corners)
                if (
                    is_bounded_ping_pong
                    and passive_motion_started is not None
                ):
                    frame = apply_bounded_ping_pong(
                        frame,
                        transform_corners,
                        ping_pong_state.x,
                        ping_pong_state.y,
                    )
                elif (
                    selected_passive_movement == "Ping-Pong"
                    and passive_motion_started is not None
                ):
                    frame = apply_positioned_ping_pong(
                        frame,
                        ping_pong_state.x,
                        ping_pong_state.y,
                    )

                with self.state_lock:
                    quality_size = self.selected_quality
                frame = apply_quality(frame, quality_size)

                if virtual_camera is not None:
                    virtual_camera.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    virtual_camera.sleep_until_next_frame()

                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_preview_frame = preview_frame

        except (CameraConnectionError, RuntimeError, OSError, cv2.error) as error:
            classified = classify_camera_error(camera.name, error)
            self.events.put(
                ("camera_error", classified.title, classified.args[0])
            )
        finally:
            if virtual_camera is not None:
                virtual_camera.close()
            if capture is not None:
                capture.release()
            with self.frame_lock:
                self.latest_frame = None
                self.latest_preview_frame = None
            self.events.put(("stopped", ""))

    def _poll(self) -> None:
        while True:
            try:
                item = self.events.get_nowait()
                event = item[0]
                detail = item[1]
                error_detail = item[2] if len(item) > 2 else None
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
                    self.status.set(f"Found {len(choices)} camera(s). Connecting...")
                    self._connect_camera()
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
                self.status.set("Camera scan failed.")
                messagebox.showerror(
                    str(detail),
                    str(error_detail or "Could not scan for connected cameras."),
                )
            elif event == "camera_error":
                title = str(detail)
                messagebox.showerror(title, str(error_detail or "The camera stopped."))
                self.status.set(title + ".")
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
                    "Camera already in use.",
                    "Camera unavailable.",
                ):
                    self.status.set("Camera stopped.")

        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            preview_frame = (
                None
                if self.latest_preview_frame is None
                else self.latest_preview_frame.copy()
            )
        if frame is not None:
            if preview_frame is not None:
                rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            preview_width = max(1, self.preview.winfo_width())
            preview_height = max(1, self.preview.winfo_height())
            image.thumbnail(
                (preview_width, preview_height),
                Image.Resampling.LANCZOS,
            )
            offset_x = max(0, (preview_width - image.width) // 2)
            offset_y = max(0, (preview_height - image.height) // 2)
            self.preview_image_size = (image.width, image.height)
            self.preview.delete("all")
            photo = ImageTk.PhotoImage(image)
            self.preview.create_image(
                offset_x, offset_y, anchor=tk.NW, image=photo
            )
            self.preview.image = photo

            if self.crop_enabled:
                x1 = int(offset_x + self.crop_left * image.width)
                y1 = int(offset_y + self.crop_top * image.height)
                x2 = int(offset_x + (1.0 - self.crop_right) * image.width)
                y2 = int(offset_y + (1.0 - self.crop_bottom) * image.height)
                shade = "#000000"
                self.preview.create_rectangle(
                    offset_x, offset_y, offset_x + image.width, y1,
                    fill=shade, stipple="gray50", outline=""
                )
                self.preview.create_rectangle(
                    offset_x, y2, offset_x + image.width,
                    offset_y + image.height,
                    fill=shade, stipple="gray50", outline=""
                )
                self.preview.create_rectangle(
                    offset_x, y1, x1, y2,
                    fill=shade, stipple="gray50", outline=""
                )
                self.preview.create_rectangle(
                    x2, y1, offset_x + image.width, y2,
                    fill=shade, stipple="gray50", outline=""
                )
                self.preview.create_rectangle(
                    x1, y1, x2, y2,
                    outline="",
                )
                for handle_x, handle_y in (
                    (x1, y1),
                    (x2, y1),
                    (x2, y2),
                    (x1, y2),
                ):
                    self.preview.create_oval(
                        handle_x - 7, handle_y - 7,
                        handle_x + 7, handle_y + 7,
                        fill="#54d8ed", outline="#212846", width=2
                    )

            if self.selected_transform != "Off":
                polygon_points: list[int] = []
                for corner_x, corner_y in self.transform_corners:
                    handle_x = int(offset_x + corner_x * image.width)
                    handle_y = int(offset_y + corner_y * image.height)
                    polygon_points.extend((handle_x, handle_y))
                    self.preview.create_oval(
                        handle_x - 5, handle_y - 5,
                        handle_x + 5, handle_y + 5,
                        fill="#ffc857", outline="#000000", width=1
                    )
                self.preview.create_polygon(
                    polygon_points,
                    outline="#ffc857",
                    fill="",
                    width=2,
                )
        elif self.camera_thread is None or not self.camera_thread.is_alive():
            self.preview.delete("all")
            self.preview.create_text(
                self.preview.winfo_width() // 2,
                self.preview.winfo_height() // 2,
                text="No camera connected",
                fill="#f0f0f0",
                font=("Segoe UI", 12),
            )
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
