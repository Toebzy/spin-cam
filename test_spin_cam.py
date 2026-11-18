import unittest

import numpy as np

from spin_cam import (
    DEFAULT_QUALITY,
    FILTERS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    QUALITY_PRESETS,
    TRANSITIONS,
    apply_filter,
    apply_quality,
    crop_to_output,
    create_app_icon,
    render_transition,
    transform_frame,
)


class TransformFrameTests(unittest.TestCase):
    def test_camera_frame_is_center_cropped_to_output_size(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = crop_to_output(frame)

        self.assertEqual(result.shape, (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3))

    def test_output_dimensions_and_type_are_preserved(self) -> None:
        frame = np.full((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), 120, dtype=np.uint8)

        result = transform_frame(frame, np.pi / 2)

        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, frame.dtype)

    def test_full_rotation_returns_to_start_view(self) -> None:
        frame = np.random.default_rng(1).integers(
            0, 256, (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8
        )

        start = transform_frame(frame, 0)
        end = transform_frame(frame, 2 * np.pi)

        np.testing.assert_array_equal(start, end)

    def test_spin_moves_outward_midway_and_finishes_after_one_turn(self) -> None:
        frame = np.random.default_rng(3).integers(
            0, 256, (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8
        )

        midpoint = render_transition(frame, frame, "3D Spin", 0.5)
        finished = render_transition(frame, frame, "3D Spin", 1)

        self.assertFalse(np.array_equal(midpoint, transform_frame(frame, np.pi)))
        np.testing.assert_array_equal(finished, frame)

    def test_slide_transitions_reveal_live_frame_at_the_end(self) -> None:
        previous = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
        current = np.full_like(previous, 255)

        for transition in TRANSITIONS:
            with self.subTest(transition=transition):
                result = render_transition(previous, current, transition, 1)
                np.testing.assert_array_equal(result, current)

    def test_slide_transitions_render_intermediate_frames(self) -> None:
        previous = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
        current = np.random.default_rng(4).integers(
            0, 256, (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8
        )

        for transition in TRANSITIONS:
            with self.subTest(transition=transition):
                result = render_transition(previous, current, transition, 0.5)
                self.assertEqual(result.shape, current.shape)
                self.assertFalse(np.array_equal(result, previous))
                self.assertFalse(np.array_equal(result, current))

    def test_crunchy_quality_preserves_virtual_camera_dimensions(self) -> None:
        frame = np.random.default_rng(2).integers(
            0, 256, (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8
        )

        result = apply_quality(frame, (320, 180))

        self.assertEqual(result.shape, frame.shape)
        self.assertTrue(np.all(result[0, 0] == result[1, 1]))

    def test_quality_defaults_to_max_and_includes_lower_modes(self) -> None:
        self.assertEqual(
            QUALITY_PRESETS[DEFAULT_QUALITY], (OUTPUT_WIDTH, OUTPUT_HEIGHT)
        )
        self.assertEqual(QUALITY_PRESETS["Cursed (40x22)"], (40, 22))

    def test_app_icon_is_a_transparent_square_image(self) -> None:
        icon = create_app_icon()

        self.assertEqual(icon.size, (64, 64))
        self.assertEqual(icon.mode, "RGBA")
        self.assertIsNotNone(icon.getbbox())

    def test_filters_preserve_frame_size_and_change_the_image(self) -> None:
        frame = np.random.default_rng(5).integers(
            0, 256, (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8
        )

        for filter_name in FILTERS:
            with self.subTest(filter=filter_name):
                result = apply_filter(frame, filter_name)
                self.assertEqual(result.shape, frame.shape)
                self.assertEqual(result.dtype, frame.dtype)
                if filter_name == "None":
                    np.testing.assert_array_equal(result, frame)
                else:
                    self.assertFalse(np.array_equal(result, frame))


if __name__ == "__main__":
    unittest.main()
