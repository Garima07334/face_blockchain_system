"""Tests for Phase 10 app.py Streamlit UI module."""

from pathlib import Path
import unittest
import numpy as np
import cv2

import app


class TestAppModule(unittest.TestCase):
    """Test suite for app.py Streamlit UI helper functions."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test image paths."""
        cls.project_root = Path(__file__).parent.parent
        cls.sample_faces_dir = cls.project_root / "sample_data" / "test_faces"
        cls.single_face_path = cls.sample_faces_dir / "single_face.jpg"

    def test_get_available_sample_faces(self) -> None:
        """Test discovering real sample face images on disk."""
        samples = app.get_available_sample_faces()
        self.assertIsInstance(samples, dict)
        if self.single_face_path.is_file():
            self.assertGreater(len(samples), 0)
            found_single = any("single_face.jpg" in str(p) for p in samples.values())
            self.assertTrue(found_single)

    def test_draw_face_bounding_box_valid_image(self) -> None:
        """Test drawing bounding box on an existing face image."""
        if not self.single_face_path.is_file():
            self.skipTest("sample_data/test_faces/single_face.jpg not found.")

        box = {"x": 50, "y": 50, "w": 100, "h": 100}
        img_rgb = app.draw_face_bounding_box(self.single_face_path, box)

        self.assertIsNotNone(img_rgb)
        self.assertIsInstance(img_rgb, np.ndarray)
        self.assertEqual(len(img_rgb.shape), 3)  # Height, Width, RGB Channels
        self.assertEqual(img_rgb.shape[2], 3)

    def test_draw_face_bounding_box_non_existent_image(self) -> None:
        """Test drawing bounding box on a non-existent image path returns None."""
        missing_path = self.project_root / "non_existent_image_12345.jpg"
        img_rgb = app.draw_face_bounding_box(missing_path, {"x": 0, "y": 0, "w": 10, "h": 10})
        self.assertIsNone(img_rgb)

    def test_draw_face_bounding_box_no_box(self) -> None:
        """Test drawing bounding box when box is None or empty dict."""
        if not self.single_face_path.is_file():
            self.skipTest("sample_data/test_faces/single_face.jpg not found.")

        img_rgb = app.draw_face_bounding_box(self.single_face_path, None)
        self.assertIsNotNone(img_rgb)
        self.assertIsInstance(img_rgb, np.ndarray)


if __name__ == "__main__":
    unittest.main()
