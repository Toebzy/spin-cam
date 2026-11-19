from __future__ import annotations

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 30
SPIN_DURATION = 1.1
SHAPE_TRANSITION_DURATION = 0.35
SPIN_TURNS = 1
TILT_DEGREES = 12
BACKGROUND = (0, 0, 0)

HOTKEYS = ("F6", "F7", "F8", "F9", "F10", "F11", "F12")
TRANSITIONS = ("3D Spin", "Push", "Zoom", "Fade", "Glitch")
FILTERS = (
    "None",
    "Noir",
    "Sepia",
    "Negative",
    "Posterize",
    "VHS",
    "Thermal",
)
TRANSFORM_PRESETS = (
    "Off",
    "Screen",
    "Tilted",
    "Squished",
    "Floor",
    "Floating",
    "Tiny",
)
PASSIVE_MOVEMENTS = ("Off", "Hover", "Ping-Pong", "Orbit", "Drift")
TRANSFORM_CORNERS = {
    "Off": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    "Screen": ((0.10, 0.08), (0.90, 0.08), (0.90, 0.92), (0.10, 0.92)),
    "Tilted": ((0.20, 0.12), (0.88, 0.20), (0.92, 0.82), (0.08, 0.88)),
    "Squished": ((0.28, 0.08), (0.72, 0.08), (0.72, 0.92), (0.28, 0.92)),
    "Floor": ((0.24, 0.48), (0.76, 0.48), (0.96, 0.80), (0.04, 0.80)),
    "Floating": ((0.20, 0.08), (0.80, 0.08), (0.92, 0.82), (0.08, 0.82)),
    "Tiny": ((0.40, 0.40), (0.60, 0.40), (0.60, 0.60), (0.40, 0.60)),
}
QUALITY_PRESETS = {
    "Clean (1920x1080)": (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    "HD (1280x720)": (1280, 720),
    "Retro (640x360)": (640, 360),
    "Crunchy (320x180)": (320, 180),
    "Extra crunchy (160x90)": (160, 90),
    "Potato (80x45)": (80, 45),
    "Cursed (40x22)": (40, 22),
}
DEFAULT_QUALITY = "Clean (1920x1080)"
