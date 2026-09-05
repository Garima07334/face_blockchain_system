"""Regression tests for face detection coordinates, face index sorting, and crop consistency."""

from pathlib import Path
import unittest
import numpy as np
import cv2

from core.face_detector import FaceDetector
from core.face_encoder import FaceEncoder
import app


class TestFaceDetectionCoordinates(unittest.TestCase):
    """Regression test suite for face bounding box coordinates and selection strategy."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test paths and detector/encoder instances."""
        cls.project_root = Path(__file__).parent.parent
        cls.test_faces_dir = cls.project_root / "sample_data" / "test_faces"
        cls.single_face_path = cls.test_faces_dir / "single_face.jpg"
        cls.detector = FaceDetector()
        cls.encoder = FaceEncoder(detector=cls.detector)

    def test_face_boxes_sorted_by_area_descending(self) -> None:
        """Test that detect_faces always returns face boxes sorted by area (w * h) descending."""
        if not self.single_face_path.is_file():
            self.skipTest("sample_data/test_faces/single_face.jpg not found.")

        detection = self.detector.detect_faces(self.single_face_path)
        self.assertTrue(detection["success"])

        faces = detection["faces"]
        if len(faces) > 1:
            areas = [f["box"]["w"] * f["box"]["h"] for f in faces]
            self.assertEqual(areas, sorted(areas, reverse=True), "Faces must be sorted by area descending.")

    def test_bounding_box_coordinate_boundaries(self) -> None:
        """Test bounding box coordinates are within valid image dimensions."""
        if not self.single_face_path.is_file():
            self.skipTest("sample_data/test_faces/single_face.jpg not found.")

        detection = self.detector.detect_faces(self.single_face_path)
        self.assertTrue(detection["success"])

        img_w = detection["image_size"]["width"]
        img_h = detection["image_size"]["height"]

        for face in detection["faces"]:
            box = face["box"]
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]

            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)
            self.assertLessEqual(x + w, img_w + 10)  # Allow slight tolerance
            self.assertLessEqual(y + h, img_h + 10)

    def test_encoder_crop_corresponds_to_detector_box(self) -> None:
        """Test that FaceEncoder uses the exact box coordinates returned by FaceDetector for face_index=0."""
        if not self.single_face_path.is_file():
            self.skipTest("sample_data/test_faces/single_face.jpg not found.")

        detection = self.detector.detect_faces(self.single_face_path)
        self.assertTrue(detection["success"])
        self.assertGreater(detection["face_count"], 0)

        expected_box = detection["faces"][0]["box"]

        encoding = self.encoder.encode_face(self.single_face_path, face_index=0)
        self.assertTrue(encoding["success"])
        self.assertEqual(encoding["box"], expected_box, "Encoder box must match detector sorted box at index 0.")

    def test_ui_crop_face_roi_utility(self) -> None:
        """Test app.crop_face_roi produces valid RGB image matching bounding box."""
        if not self.single_face_path.is_file():
            self.skipTest("sample_data/test_faces/single_face.jpg not found.")

        detection = self.detector.detect_faces(self.single_face_path)
        self.assertTrue(detection["success"])

        box = detection["faces"][0]["box"]
        crop_rgb = app.crop_face_roi(self.single_face_path, box)

        self.assertIsNotNone(crop_rgb)
        self.assertIsInstance(crop_rgb, np.ndarray)
        self.assertEqual(crop_rgb.shape[2], 3)  # RGB channels
        self.assertAlmostEqual(crop_rgb.shape[0], box["h"], delta=5)
        self.assertAlmostEqual(crop_rgb.shape[1], box["w"], delta=5)

    def test_draw_all_face_bounding_boxes(self) -> None:
        """Test app.draw_all_face_bounding_boxes renders multi-face image without error."""
        if not self.single_face_path.is_file():
            self.skipTest("sample_data/test_faces/single_face.jpg not found.")

        detection = self.detector.detect_faces(self.single_face_path)
        annotated = app.draw_all_face_bounding_boxes(self.single_face_path, detection["faces"], selected_index=0)

        self.assertIsNotNone(annotated)
        self.assertIsInstance(annotated, np.ndarray)


if __name__ == "__main__":
    unittest.main()
