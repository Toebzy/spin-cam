import unittest

import numpy as np
from spin_cam import CameraConnectionError, classify_camera_error

from core.config import TRANSFORM_CORNERS, TRANSFORM_PRESETS
from core.video_processing import (
    apply_bounded_ping_pong,
    apply_corner_transform,
    apply_native_crop,
    apply_quality,
    apply_passive_motion,
    fit_to_output,
    interpolate_shape_corners,
    ping_pong_phase,
    passive_motion_destination,
    PingPongMotionState,
    shift_crop_margins,
    stretch_to_output,
)


class VideoProcessingTests(unittest.TestCase):
    def test_shape_corners_transition_smoothly(self) -> None:
        source = TRANSFORM_CORNERS["Off"]
        target = TRANSFORM_CORNERS["Tiny"]

        self.assertTrue(
            np.allclose(interpolate_shape_corners(source, target, 0.0), source)
        )
        midpoint = interpolate_shape_corners(source, target, 0.5)
        self.assertTrue(
            np.allclose(
                midpoint,
                (
                    (0.2, 0.2),
                    (0.8, 0.2),
                    (0.8, 0.8),
                    (0.2, 0.8),
                ),
            )
        )
        self.assertTrue(
            np.allclose(interpolate_shape_corners(source, target, 1.0), target)
        )

    def test_tiny_preset_is_centered_at_twenty_percent_scale(self) -> None:
        self.assertIn("Tiny", TRANSFORM_PRESETS)
        corners = np.asarray(TRANSFORM_CORNERS["Tiny"])
        center = corners.mean(axis=0)
        self.assertTrue(np.allclose(center, (0.5, 0.5)))
        self.assertTrue(np.allclose(corners[1] - corners[0], (0.2, 0.0)))
        self.assertTrue(np.allclose(corners[3] - corners[0], (0.0, 0.2)))

    def test_tiny_shape_bounces_inside_ping_pong_walls(self) -> None:
        frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
        tiny = apply_corner_transform(frame, TRANSFORM_CORNERS["Tiny"])
        moving_left = apply_bounded_ping_pong(
            tiny, TRANSFORM_CORNERS["Tiny"], 0.0, 0.0
        )
        moving_right = apply_bounded_ping_pong(
            tiny, TRANSFORM_CORNERS["Tiny"], 1.0, 1.0
        )
        left_y, left_x = np.nonzero(np.any(moving_left > 0, axis=2))
        right_y, right_x = np.nonzero(np.any(moving_right > 0, axis=2))

        self.assertLessEqual(left_x.min(), 1)
        self.assertLessEqual(left_y.min(), 1)
        self.assertGreaterEqual(right_x.max(), 1278)
        self.assertGreater(left_x.max() - left_x.min(), 200)
        self.assertGreater(right_x.max() - right_x.min(), 200)

        for position in np.linspace(0.0, 1.0, 80):
            moved = apply_bounded_ping_pong(
                tiny,
                TRANSFORM_CORNERS["Tiny"],
                float(position),
                float(1.0 - position),
            )
            y_points, x_points = np.nonzero(np.any(moved > 0, axis=2))
            self.assertGreaterEqual(x_points.min(), 0)
            self.assertLessEqual(x_points.max(), 1279)
            self.assertGreaterEqual(y_points.min(), 0)
            self.assertLessEqual(y_points.max(), 719)

    def test_ping_pong_reflects_only_the_axis_that_hits_a_wall(self) -> None:
        state = PingPongMotionState(
            x=0.99,
            y=0.40,
            velocity_x=0.28,
            velocity_y=0.18,
        )
        previous_x, previous_y = state.x, state.y
        state.advance(0.2)

        self.assertLess(state.velocity_x, 0.0)
        self.assertGreater(state.velocity_y, 0.0)
        self.assertLess(state.x, 1.0)
        self.assertLess(state.x, previous_x)
        self.assertGreater(state.y, previous_y)

    def test_ping_pong_large_frame_gap_folds_back_inside_wall(self) -> None:
        state = PingPongMotionState(x=0.5, y=0.5)
        previous = (state.x, state.y)
        state.advance(12.0)

        self.assertTrue(0.0 <= state.x <= 1.0)
        self.assertTrue(0.0 <= state.y <= 1.0)
        self.assertNotEqual((state.x, state.y), previous)

    def test_busy_camera_error_is_classified(self) -> None:
        error = classify_camera_error("Test Camera", RuntimeError("camera is already in use"))
        self.assertEqual(error.title, "Camera already in use")
        self.assertIn("Close apps", str(error))

    def test_custom_camera_error_is_preserved(self) -> None:
        error = CameraConnectionError("Camera unavailable", "Disconnected")
        classified = classify_camera_error("Test Camera", error)
        self.assertIs(classified, error)

    def test_fit_to_output_returns_output_size(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = fit_to_output(frame)
        self.assertEqual(result.shape, (1080, 1920, 3))

    def test_native_crop_reduces_frame(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = apply_native_crop(frame, 0.10, 0.20, 0.10, 0.20)
        self.assertEqual(result.shape, (432, 1024, 3))

    def test_stretch_to_output_uses_entire_crop(self) -> None:
        frame = np.arange(120, dtype=np.uint8).reshape(10, 4, 3)
        result = stretch_to_output(frame)
        self.assertEqual(result.shape, (1080, 1920, 3))
        self.assertTrue(np.array_equal(result[0, 0], frame[0, 0]))
        self.assertTrue(np.array_equal(result[-1, -1], frame[-1, -1]))

    def test_crop_region_moves_without_changing_size_or_leaving_frame(self) -> None:
        moved = shift_crop_margins((0.2, 0.1, 0.3, 0.4), 0.1, -0.2)
        self.assertTrue(np.allclose(moved, (0.3, 0.0, 0.2, 0.5)))
        self.assertAlmostEqual(moved[0] + moved[2], 0.5)
        self.assertAlmostEqual(moved[1] + moved[3], 0.5)

    def test_quality_cleans_to_output_size(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = apply_quality(frame, (640, 360))
        self.assertEqual(result.shape, (1080, 1920, 3))

    def test_passive_motion_off_is_identity(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = apply_passive_motion(frame, "Off", 0.0)
        self.assertIs(result, frame)

    def test_passive_motion_preserves_output_size(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for mode in ("Hover", "Ping-Pong", "Orbit", "Drift"):
            result = apply_passive_motion(frame, mode, 1.25)
            self.assertEqual(result.shape, frame.shape)

    def test_unknown_passive_motion_is_rejected(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Unknown passive movement"):
            apply_passive_motion(frame, "Unknown", 0.0)

    def test_ping_pong_stays_within_frame_and_reaches_each_wall(self) -> None:
        width, height = 1280, 720
        right = passive_motion_destination(width, height, "Ping-Pong", 1.0 / 0.28)
        left = passive_motion_destination(width, height, "Ping-Pong", 3.0 / 0.28)

        self.assertAlmostEqual(float(right[:, 0].max()), width, places=3)
        self.assertAlmostEqual(float(left[:, 0].min()), 0.0, places=3)

        bottom = passive_motion_destination(
            width, height, "Ping-Pong", 1.0 / 0.18
        )
        top = passive_motion_destination(width, height, "Ping-Pong", 0.0)
        self.assertAlmostEqual(float(bottom[:, 1].max()), height, places=3)
        self.assertAlmostEqual(float(top[:, 1].min()), 0.0, places=3)

        for time_seconds in np.linspace(-20, 20, 200):
            destination = passive_motion_destination(
                width, height, "Ping-Pong", float(time_seconds)
            )
            self.assertGreaterEqual(float(destination[:, 0].min()), -0.001)
            self.assertLessEqual(float(destination[:, 0].max()), width + 0.001)
            self.assertGreaterEqual(float(destination[:, 1].min()), -0.001)
            self.assertLessEqual(float(destination[:, 1].max()), height + 0.001)

    def test_ping_pong_phase_is_linear_between_reflections(self) -> None:
        samples = [ping_pong_phase(value) for value in (0.0, 0.5, 1.0, 1.5, 2.0)]
        self.assertTrue(np.allclose(samples, (-1.0, 0.0, 1.0, 0.5, 0.0)))
        self.assertTrue(np.allclose(np.diff(samples), (1.0, 1.0, -0.5, -0.5)))


if __name__ == "__main__":
    unittest.main()
