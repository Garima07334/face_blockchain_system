"""Tests for face_encoder module using OpenCV SFace."""

import math
from pathlib import Path
import unittest
import numpy as np
from core.face_encoder import FaceEncoder


class TestFaceEncoder(unittest.TestCase):
    """Test suite for FaceEncoder using OpenCV SFace."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize shared SFace encoder instance and test image paths."""
        cls.encoder = FaceEncoder()
        cls.test_faces_dir = Path(__file__).parent.parent / "sample_data" / "test_faces"

    def test_encode_single_face(self) -> None:
        """Test encoding a valid image containing one face."""
        image_path = self.test_faces_dir / "single_face.jpg"
        result = self.encoder.encode_face(image_path)

        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["face_count"], 1)
        self.assertEqual(result["selected_face_index"], 0)
        self.assertIsNotNone(result["box"])
        self.assertIsNotNone(result["embedding"])
        self.assertEqual(result["embedding_dim"], 128)
        self.assertEqual(len(result["embedding"]), 128)
        self.assertIsNone(result["error"])

        # Check numeric types
        for val in result["embedding"]:
            self.assertIsInstance(val, float)
            self.assertFalse(math.isnan(val))

        # Check L2 unit normalization
        l2_norm = float(np.linalg.norm(result["embedding"]))
        self.assertAlmostEqual(l2_norm, 1.0, places=4)

    def test_encode_no_face(self) -> None:
        """Test encoding an image containing no faces."""
        image_path = self.test_faces_dir / "no_face.jpg"
        result = self.encoder.encode_face(image_path)

        self.assertFalse(result["success"])
        self.assertEqual(result["face_count"], 0)
        self.assertIsNone(result["embedding"])
        self.assertIsNotNone(result["error"])
        self.assertIn("no faces detected", result["error"].lower())

    def test_encode_missing_image(self) -> None:
        """Test encoding a non-existent image path."""
        image_path = self.test_faces_dir / "non_existent_file_999.jpg"
        result = self.encoder.encode_face(image_path)

        self.assertFalse(result["success"])
        self.assertEqual(result["face_count"], 0)
        self.assertIsNone(result["embedding"])
        self.assertIsNotNone(result["error"])

    def test_encode_multiple_faces_selection(self) -> None:
        """Test face selection when multiple faces are detected in an image."""
        image_path = self.test_faces_dir / "multi_faces.jpg"

        # Encode face index 0
        res0 = self.encoder.encode_face(image_path, face_index=0)
        self.assertTrue(res0["success"])
        self.assertEqual(res0["selected_face_index"], 0)
        self.assertEqual(len(res0["embedding"]), 128)

        # Encode face index 1
        res1 = self.encoder.encode_face(image_path, face_index=1)
        self.assertTrue(res1["success"])
        self.assertEqual(res1["selected_face_index"], 1)
        self.assertEqual(len(res1["embedding"]), 128)

        # The bounding boxes for face 0 and face 1 should differ
        self.assertNotEqual(res0["box"], res1["box"])

        # Out-of-range face index check
        res_invalid = self.encoder.encode_face(image_path, face_index=99)
        self.assertFalse(res_invalid["success"])
        self.assertIsNone(res_invalid["embedding"])
        self.assertIsNotNone(res_invalid["error"])
        self.assertIn("invalid face_index", res_invalid["error"].lower())

    def test_encode_deterministic(self) -> None:
        """Test that encoding the same image twice produces identical embeddings within floating-point tolerance."""
        image_path = self.test_faces_dir / "single_face.jpg"

        res_a = self.encoder.encode_face(image_path)
        res_b = self.encoder.encode_face(image_path)

        self.assertTrue(res_a["success"])
        self.assertTrue(res_b["success"])
        self.assertTrue(np.allclose(res_a["embedding"], res_b["embedding"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
