"""Unit tests for Phase 10.5 / Phase 11 FaceMatcher module."""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import requests

from core.face_matcher import FaceMatcher


class TestFaceMatcher(unittest.TestCase):
    """Test suite for FaceMatcher face similarity verification."""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test embeddings and image paths."""
        cls.project_root = Path(__file__).parent.parent
        cls.test_faces_dir = cls.project_root / "sample_data" / "test_faces"
        cls.single_face_path = cls.test_faces_dir / "single_face.jpg"

        # Create synthetic 128D unit vectors for testing
        v1 = np.ones(128, dtype=np.float32)
        cls.unit_emb1 = (v1 / np.linalg.norm(v1)).tolist()

        # Vector with high similarity (> 0.363)
        v2 = np.ones(128, dtype=np.float32)
        v2[0] = 0.5
        cls.similar_emb = (v2 / np.linalg.norm(v2)).tolist()

        # Orthogonal vector (cosine similarity = 0.0 < 0.363)
        v3 = np.zeros(128, dtype=np.float32)
        v3[:64] = 1.0
        v3[64:] = -1.0
        # Create vector orthogonal to unit_emb1
        v_ortho = np.zeros(128, dtype=np.float32)
        v_ortho[:64] = 1.0
        v_ortho[64:] = -1.0
        cls.orthogonal_emb = (v_ortho / np.linalg.norm(v_ortho)).tolist()

    def test_identical_embeddings_match(self) -> None:
        """Test identical embeddings yield similarity 1.0 and match == True."""
        matcher = FaceMatcher(threshold=0.363)
        res = matcher.compare_embeddings(self.unit_emb1, [self.unit_emb1])

        self.assertTrue(res["match"])
        self.assertEqual(res["status"], FaceMatcher.STATUS_MATCH)
        self.assertAlmostEqual(res["similarity"], 1.0, places=3)
        self.assertEqual(res["candidate_face_count"], 1)
        self.assertEqual(res["best_candidate_face_index"], 0)
        self.assertIsNone(res["error"])

    def test_similar_embeddings_match(self) -> None:
        """Test embeddings with cosine similarity > 0.363 yield match == True."""
        matcher = FaceMatcher(threshold=0.363)
        res = matcher.compare_embeddings(self.unit_emb1, [self.similar_emb])

        self.assertTrue(res["match"])
        self.assertEqual(res["status"], FaceMatcher.STATUS_MATCH)
        self.assertGreaterEqual(res["similarity"], 0.363)
        self.assertIsNone(res["error"])

    def test_dissimilar_embeddings_no_match(self) -> None:
        """Test embeddings with cosine similarity < 0.363 yield match == False."""
        matcher = FaceMatcher(threshold=0.363)
        res = matcher.compare_embeddings(self.unit_emb1, [self.orthogonal_emb])

        self.assertFalse(res["match"])
        self.assertEqual(res["status"], FaceMatcher.STATUS_NO_MATCH)
        self.assertLess(res["similarity"], 0.363)
        self.assertIsNone(res["error"])

    def test_threshold_boundary_condition(self) -> None:
        """Test threshold boundary condition (similarity exactly at threshold)."""
        matcher = FaceMatcher(threshold=0.5)
        # Mock compute_similarity return value
        with patch.object(FaceMatcher, "compute_similarity", return_value=(0.5, 0.5)):
            res = matcher.compare_embeddings(self.unit_emb1, [self.similar_emb])
            self.assertTrue(res["match"])
            self.assertEqual(res["status"], FaceMatcher.STATUS_MATCH)

        with patch.object(FaceMatcher, "compute_similarity", return_value=(0.4999, 0.5001)):
            res = matcher.compare_embeddings(self.unit_emb1, [self.similar_emb])
            self.assertFalse(res["match"])
            self.assertEqual(res["status"], FaceMatcher.STATUS_NO_MATCH)

    def test_multiple_candidate_faces_selects_best_match(self) -> None:
        """Test multiple candidate embeddings selects the highest similarity face."""
        matcher = FaceMatcher(threshold=0.363)
        candidates = [self.orthogonal_emb, self.unit_emb1, self.similar_emb]
        res = matcher.compare_embeddings(self.unit_emb1, candidates)

        self.assertTrue(res["match"])
        self.assertEqual(res["candidate_face_count"], 3)
        self.assertEqual(res["best_candidate_face_index"], 1)
        self.assertAlmostEqual(res["best_similarity"], 1.0, places=3)

    def test_missing_or_empty_query_embedding_returns_verification_error(self) -> None:
        """Test missing query embedding returns VERIFICATION_ERROR (not NO_MATCH)."""
        matcher = FaceMatcher(threshold=0.363)
        res = matcher.compare_embeddings([], [self.unit_emb1])

        self.assertFalse(res["match"])
        self.assertEqual(res["status"], FaceMatcher.STATUS_VERIFICATION_ERROR)
        self.assertIsNotNone(res["error"])

    def test_empty_candidate_embeddings_returns_no_match(self) -> None:
        """Test empty candidate list returns NO_MATCH with candidate_face_count=0."""
        matcher = FaceMatcher(threshold=0.363)
        res = matcher.compare_embeddings(self.unit_emb1, [])

        self.assertFalse(res["match"])
        self.assertEqual(res["status"], FaceMatcher.STATUS_NO_MATCH)
        self.assertEqual(res["candidate_face_count"], 0)
        self.assertIsNone(res["best_candidate_face_index"])

    def test_invalid_url_scheme_returns_verification_error(self) -> None:
        """Test downloading non-HTTP URL returns VERIFICATION_ERROR."""
        matcher = FaceMatcher()
        res = matcher.download_candidate_image("ftp://example.com/image.jpg")

        self.assertFalse(res["success"])
        self.assertIsNotNone(res["error"])
        self.assertIn("invalid candidate image url scheme", res["error"].lower())

    @patch("requests.get")
    def test_http_404_download_failure_returns_verification_error(self, mock_get: MagicMock) -> None:
        """Test HTTP 404 response returns VERIFICATION_ERROR (distinct from NO_MATCH)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        matcher = FaceMatcher()
        res = matcher.verify_candidate_image(self.unit_emb1, "https://example.com/non_existent.jpg")

        self.assertFalse(res["match"])
        self.assertEqual(res["status"], FaceMatcher.STATUS_VERIFICATION_ERROR)
        self.assertIsNotNone(res["error"])
        self.assertIn("404", res["error"])

    @patch("requests.get")
    def test_http_timeout_returns_verification_error(self, mock_get: MagicMock) -> None:
        """Test HTTP timeout returns VERIFICATION_ERROR."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        matcher = FaceMatcher()
        res = matcher.verify_candidate_image(self.unit_emb1, "https://example.com/slow_image.jpg")

        self.assertFalse(res["match"])
        self.assertEqual(res["status"], FaceMatcher.STATUS_VERIFICATION_ERROR)
        self.assertIsNotNone(res["error"])
        self.assertIn("timed out", res["error"].lower())

    def test_invalid_image_bytes_returns_verification_error(self) -> None:
        """Test providing invalid image byte payload returns VERIFICATION_ERROR."""
        matcher = FaceMatcher()
        invalid_bytes = b"NOT_AN_IMAGE_PAYLOAD_12345"
        res = matcher.verify_candidate_image(self.unit_emb1, invalid_bytes)

        self.assertFalse(res["match"])
        self.assertEqual(res["status"], FaceMatcher.STATUS_VERIFICATION_ERROR)
        self.assertIsNotNone(res["error"])
        self.assertIn("failed to decode", res["error"].lower())

    def test_local_face_image_verification_match(self) -> None:
        """Test local face image verification against its own embedding yields MATCH."""
        if not self.single_face_path.is_file():
            self.skipTest("sample_data/test_faces/single_face.jpg not found.")

        matcher = FaceMatcher(threshold=0.363)

        # 1. Encode query face
        query_enc = matcher.encoder.encode_face(self.single_face_path)
        self.assertTrue(query_enc["success"])
        query_emb = query_enc["embedding"]

        # 2. Verify candidate image against query embedding
        res = matcher.verify_candidate_image(query_emb, self.single_face_path)

        self.assertTrue(res["match"])
        self.assertEqual(res["status"], FaceMatcher.STATUS_MATCH)
        self.assertGreaterEqual(res["candidate_face_count"], 1)
        self.assertGreaterEqual(res["similarity"], 0.363)
        self.assertIsNone(res["error"])


if __name__ == "__main__":
    unittest.main()
