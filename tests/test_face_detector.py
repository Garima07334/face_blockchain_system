"""Tests for face_detector module."""

import os
from pathlib import Path
import unittest
from core.face_detector import FaceDetector


class TestFaceDetector(unittest.TestCase):
    """Test suite for FaceDetector."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize shared detector instance and paths."""
        cls.detector = FaceDetector()
        cls.test_faces_dir = Path(__file__).parent.parent / "sample_data" / "test_faces"

    def test_detect_single_face(self) -> None:
        """Test detection on a valid image containing a detectable face."""
        image_path = self.test_faces_dir / "single_face.jpg"
        result = self.detector.detect_faces(image_path)

        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["face_count"], 1)
        self.assertEqual(len(result["faces"]), result["face_count"])
        self.assertIsNone(result["error"])

        # Validate bounding box structure
        first_face = result["faces"][0]
        self.assertIn("box", first_face)
        box = first_face["box"]
        self.assertIn("x", box)
        self.assertIn("y", box)
        self.assertIn("w", box)
        self.assertIn("h", box)
        self.assertGreater(box["w"], 0)
        self.assertGreater(box["h"], 0)

    def test_detect_no_face(self) -> None:
        """Test detection on an image containing no faces."""
        image_path = self.test_faces_dir / "no_face.jpg"
        result = self.detector.detect_faces(image_path)

        self.assertTrue(result["success"])
        self.assertEqual(result["face_count"], 0)
        self.assertEqual(len(result["faces"]), 0)
        self.assertIsNone(result["error"])

    def test_detect_multiple_faces(self) -> None:
        """Test detection on an image containing multiple detectable faces."""
        image_path = self.test_faces_dir / "multi_faces.jpg"
        result = self.detector.detect_faces(image_path)

        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["face_count"], 2)
        self.assertEqual(len(result["faces"]), result["face_count"])
        self.assertIsNone(result["error"])

    def test_missing_image_path(self) -> None:
        """Test graceful handling of a non-existent image path."""
        image_path = self.test_faces_dir / "non_existent_file_12345.jpg"
        result = self.detector.detect_faces(image_path)

        self.assertFalse(result["success"])
        self.assertEqual(result["face_count"], 0)
        self.assertEqual(len(result["faces"]), 0)
        self.assertIsNotNone(result["error"])
        self.assertIn("not found", result["error"].lower())

    def test_corrupt_image_file(self) -> None:
        """Test graceful handling of an unreadable/invalid image file."""
        corrupt_path = self.test_faces_dir / "corrupt.jpg"
        with open(corrupt_path, "w") as f:
            f.write("This is not an image file.")

        try:
            result = self.detector.detect_faces(corrupt_path)
            self.assertFalse(result["success"])
            self.assertEqual(result["face_count"], 0)
            self.assertIsNotNone(result["error"])
        finally:
            if corrupt_path.exists():
                os.remove(corrupt_path)


if __name__ == "__main__":
    unittest.main()

