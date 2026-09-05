"""Tests for core/tamper_detector.py Phase 8 Tamper Detection Layer."""

import unittest
from unittest.mock import MagicMock
from core.blockchain import BlockchainClient
from core.hasher import CandidateHasher, generate_candidate_hash
from core.tamper_detector import TamperDetector, verify_candidate_integrity


class TestTamperDetector(unittest.TestCase):
    """Test suite for Phase 8 TamperDetector."""

    def setUp(self) -> None:
        """Initialize test candidate and mock blockchain client."""
        self.sample_candidate = {
            "title": "Jane Doe Profile",
            "url": "https://example.com/jane_doe",
            "source": "example.com",
            "snippet": "Software developer profile",
            "thumbnail": "https://example.com/thumb.jpg",
            "result_type": "External visual-search candidate",
            "selection_rank": 1,
            "provider": "SerpApi (Google Lens)",
        }

        # Calculate exact Phase 5 hash
        self.hasher = CandidateHasher()
        self.original_hash = self.hasher.generate_candidate_hash(self.sample_candidate)["sha256"]

        # Mock BlockchainClient
        self.mock_client = MagicMock(spec=BlockchainClient)

    def test_1_original_candidate_verifies_successfully(self) -> None:
        """Test 1: Unchanged original registered candidate returns status VERIFIED."""
        self.mock_client.verify_hash.return_value = {
            "success": True,
            "hash": self.original_hash,
            "verified": True,
            "error": None,
        }
        self.mock_client.get_record.return_value = {
            "success": True,
            "hash": self.original_hash,
            "registered_by": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "timestamp": 1788518846,
            "exists": True,
            "error": None,
        }

        detector = TamperDetector(hasher=self.hasher, blockchain_client=self.mock_client)
        res = detector.verify_candidate_integrity(self.sample_candidate)

        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "VERIFIED")
        self.assertEqual(res["current_hash"], self.original_hash)
        self.assertTrue(res["blockchain_verified"])
        self.assertFalse(res["tampered"])
        self.assertIsNotNone(res["record"])
        self.assertEqual(res["record"]["data_hash"], self.original_hash)

    def test_2_changing_title_causes_tamper_detected(self) -> None:
        """Test 2: Modifying title field yields status TAMPER DETECTED."""
        tampered_candidate = dict(self.sample_candidate, title="Modified Title")
        tampered_hash = self.hasher.generate_candidate_hash(tampered_candidate)["sha256"]

        # Hash must differ from original
        self.assertNotEqual(self.original_hash, tampered_hash)

        # Mock returns verified=False for unregistered tampered hash
        self.mock_client.verify_hash.return_value = {
            "success": True,
            "hash": tampered_hash,
            "verified": False,
            "error": None,
        }

        detector = TamperDetector(hasher=self.hasher, blockchain_client=self.mock_client)
        res = detector.verify_candidate_integrity(tampered_candidate)

        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "TAMPER DETECTED")
        self.assertEqual(res["current_hash"], tampered_hash)
        self.assertFalse(res["blockchain_verified"])
        self.assertTrue(res["tampered"])
        self.assertIsNone(res["record"])

    def test_3_changing_snippet_causes_tamper_detected(self) -> None:
        """Test 3: Modifying snippet field yields status TAMPER DETECTED."""
        tampered_candidate = dict(self.sample_candidate, snippet="Modified snippet text")
        tampered_hash = self.hasher.generate_candidate_hash(tampered_candidate)["sha256"]

        self.assertNotEqual(self.original_hash, tampered_hash)

        self.mock_client.verify_hash.return_value = {
            "success": True,
            "hash": tampered_hash,
            "verified": False,
            "error": None,
        }

        res = verify_candidate_integrity(tampered_candidate, hasher=self.hasher, blockchain_client=self.mock_client)
        self.assertEqual(res["status"], "TAMPER DETECTED")
        self.assertTrue(res["tampered"])

    def test_4_changing_url_causes_tamper_detected(self) -> None:
        """Test 4: Modifying URL field yields status TAMPER DETECTED."""
        tampered_candidate = dict(self.sample_candidate, url="https://example.com/tampered_url")
        tampered_hash = self.hasher.generate_candidate_hash(tampered_candidate)["sha256"]

        self.assertNotEqual(self.original_hash, tampered_hash)

        self.mock_client.verify_hash.return_value = {
            "success": True,
            "hash": tampered_hash,
            "verified": False,
            "error": None,
        }

        res = verify_candidate_integrity(tampered_candidate, hasher=self.hasher, blockchain_client=self.mock_client)
        self.assertEqual(res["status"], "TAMPER DETECTED")
        self.assertTrue(res["tampered"])

    def test_5_changing_source_causes_tamper_detected(self) -> None:
        """Test 5: Modifying source field yields status TAMPER DETECTED."""
        tampered_candidate = dict(self.sample_candidate, source="tampered.org")
        tampered_hash = self.hasher.generate_candidate_hash(tampered_candidate)["sha256"]

        self.assertNotEqual(self.original_hash, tampered_hash)

        self.mock_client.verify_hash.return_value = {
            "success": True,
            "hash": tampered_hash,
            "verified": False,
            "error": None,
        }

        res = verify_candidate_integrity(tampered_candidate, hasher=self.hasher, blockchain_client=self.mock_client)
        self.assertEqual(res["status"], "TAMPER DETECTED")
        self.assertTrue(res["tampered"])

    def test_6_changing_thumbnail_causes_tamper_detected(self) -> None:
        """Test 6: Modifying thumbnail field yields status TAMPER DETECTED."""
        tampered_candidate = dict(self.sample_candidate, thumbnail="https://example.com/fake_thumb.jpg")
        tampered_hash = self.hasher.generate_candidate_hash(tampered_candidate)["sha256"]

        self.assertNotEqual(self.original_hash, tampered_hash)

        self.mock_client.verify_hash.return_value = {
            "success": True,
            "hash": tampered_hash,
            "verified": False,
            "error": None,
        }

        res = verify_candidate_integrity(tampered_candidate, hasher=self.hasher, blockchain_client=self.mock_client)
        self.assertEqual(res["status"], "TAMPER DETECTED")
        self.assertTrue(res["tampered"])

    def test_7_unchanged_candidate_remains_verified(self) -> None:
        """Test 7: Unchanged candidate metadata remains VERIFIED across multiple verification calls."""
        self.mock_client.verify_hash.return_value = {
            "success": True,
            "hash": self.original_hash,
            "verified": True,
            "error": None,
        }
        self.mock_client.get_record.return_value = {
            "success": True,
            "hash": self.original_hash,
            "registered_by": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "timestamp": 1788518846,
            "exists": True,
            "error": None,
        }

        detector = TamperDetector(hasher=self.hasher, blockchain_client=self.mock_client)
        res1 = detector.verify_candidate_integrity(self.sample_candidate)
        res2 = detector.verify_candidate_integrity(self.sample_candidate)

        self.assertEqual(res1["status"], "VERIFIED")
        self.assertEqual(res2["status"], "VERIFIED")
        self.assertEqual(res1["current_hash"], res2["current_hash"])

    def test_8_malformed_candidate_fails_safely(self) -> None:
        """Test 8: Non-dict or malformed candidate returns status VERIFICATION ERROR."""
        detector = TamperDetector(hasher=self.hasher, blockchain_client=self.mock_client)

        res_str = detector.verify_candidate_integrity("invalid_string_candidate")
        self.assertFalse(res_str["success"])
        self.assertEqual(res_str["status"], "VERIFICATION ERROR")
        self.assertFalse(res_str["tampered"])

        res_no_url = detector.verify_candidate_integrity({"title": "No URL Candidate"})
        self.assertFalse(res_no_url["success"])
        self.assertEqual(res_no_url["status"], "VERIFICATION ERROR")
        self.assertFalse(res_no_url["tampered"])

    def test_9_blockchain_unavailable_reported_as_verification_error(self) -> None:
        """Test 9: RPC node or blockchain unavailability returns VERIFICATION ERROR, NEVER TAMPER DETECTED."""
        self.mock_client.verify_hash.return_value = {
            "success": False,
            "hash": self.original_hash,
            "verified": False,
            "error": "Blockchain RPC node unavailable at http://127.0.0.1:8545.",
        }

        detector = TamperDetector(hasher=self.hasher, blockchain_client=self.mock_client)
        res = detector.verify_candidate_integrity(self.sample_candidate)

        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "VERIFICATION ERROR")
        self.assertFalse(res["blockchain_verified"])
        self.assertFalse(res["tampered"])  # Must NOT report tampered=True
        self.assertIsNotNone(res["error"])
        self.assertIn("RPC node unavailable", res["error"])

    def test_10_hash_matches_phase5_hasher(self) -> None:
        """Test 10: Hash calculation in TamperDetector strictly matches Phase 5 CandidateHasher."""
        expected_hash = generate_candidate_hash(self.sample_candidate)["sha256"]

        self.mock_client.verify_hash.return_value = {
            "success": True,
            "hash": expected_hash,
            "verified": True,
            "error": None,
        }

        detector = TamperDetector(hasher=self.hasher, blockchain_client=self.mock_client)
        res = detector.verify_candidate_integrity(self.sample_candidate)

        self.assertEqual(res["current_hash"], expected_hash)

    def test_11_explicit_proof_original_hash_differs_from_tampered_hash(self) -> None:
        """Test 11: Explicit proof that original hash != tampered hash."""
        tampered = dict(self.sample_candidate, title="Proof Modified Title")

        orig_hash = self.hasher.generate_candidate_hash(self.sample_candidate)["sha256"]
        tamp_hash = self.hasher.generate_candidate_hash(tampered)["sha256"]

        self.assertNotEqual(orig_hash, tamp_hash)
        self.assertEqual(len(orig_hash), 64)
        self.assertEqual(len(tamp_hash), 64)


if __name__ == "__main__":
    unittest.main()
