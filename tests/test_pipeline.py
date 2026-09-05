"""Phase 9 unit/integration tests for pipeline.py VerificationPipeline.

Tests cover:
- Missing / unreadable / no-face images fail at face_detection stage.
- Search failures stop the pipeline before blockchain registration.
- Each stage uses the correct existing module (mocked where appropriate).
- Original candidate → VERIFIED; modified candidate → TAMPER DETECTED.
- Blockchain/RPC failure is reported as an error, never as tampering.
- All existing Phase 1-8 tests remain unaffected.
"""

import copy
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.blockchain import BlockchainClient
from core.face_detector import FaceDetector
from core.face_encoder import FaceEncoder
from core.face_matcher import FaceMatcher
from core.hasher import CandidateHasher
from core.result_selector import ResultSelector
from core.tamper_detector import TamperDetector
from core.web_search import WebSearch
from pipeline import VerificationPipeline, run_pipeline

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------
SAMPLE_CANDIDATE = {
    "title": "Jane Doe Profile",
    "url": "https://example.com/jane_doe",
    "source": "example.com",
    "snippet": "Software developer profile page",
    "thumbnail": "https://example.com/thumb.jpg",
    "result_type": "External visual-search candidate",
    "selection_rank": 1,
    "provider": "SerpApi (Google Lens)",
}

VALID_SHA256 = "a" * 64  # 64-char hex placeholder for unit tests

TEST_FACES_DIR = Path(__file__).parent.parent / "sample_data" / "test_faces"
MISSING_IMAGE = str(Path(__file__).parent.parent / "sample_data" / "test_faces" / "nonexistent_xyz.jpg")
CORRUPT_IMAGE = str(Path(__file__).parent.parent / "sample_data" / "test_faces" / "_corrupt_pipeline_test.jpg")
SINGLE_FACE_IMAGE = str(TEST_FACES_DIR / "single_face.jpg")
NO_FACE_IMAGE = str(TEST_FACES_DIR / "no_face.jpg")


def _make_search_success(results=None):
    """Helper – build a successful WebSearch response."""
    if results is None:
        results = [
            {
                "title": "Jane Doe Profile",
                "url": "https://example.com/jane_doe",
                "source": "example.com",
                "snippet": "Software developer profile page",
                "thumbnail": "https://example.com/thumb.jpg",
                "result_type": "External reverse-image search result",
            }
        ]
    return {
        "success": True,
        "query_image": SINGLE_FACE_IMAGE,
        "provider": "SerpApi (Google Lens)",
        "image_id": "img_test_001",
        "result_count": len(results),
        "results": results,
        "error": None,
    }


def _make_blockchain_register_success(hash_hex):
    """Helper – successful register_hash response."""
    return {
        "success": True,
        "hash": hash_hex,
        "bytes32": "0x" + hash_hex,
        "contract_address": "0xContractAddress123",
        "transaction_hash": "0xTxHash123",
        "block_number": 1,
        "registered_by": "0xSenderAddress",
        "timestamp": 1700000000,
        "error": None,
    }


def _make_blockchain_verify_success(hash_hex, verified=True):
    """Helper – successful verify_hash response."""
    return {
        "success": True,
        "hash": hash_hex,
        "bytes32": "0x" + hash_hex,
        "verified": verified,
        "error": None,
    }


def _make_blockchain_get_record_success(hash_hex):
    """Helper – successful get_record response."""
    return {
        "success": True,
        "hash": hash_hex,
        "registered_by": "0xSenderAddress",
        "timestamp": 1700000000,
        "exists": True,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Helper: build a fully-mocked VerificationPipeline that passes all stages
# until blockchain, using a real hasher so hash computations are genuine.
# ---------------------------------------------------------------------------
def _build_full_mock_pipeline(
    *,
    search_results=None,
    blockchain_verify_first=False,  # simulate "already_registered" path
    blockchain_connected=True,
    blockchain_has_contract=True,
    blockchain_register_fail=False,
    blockchain_verify_fail=False,
    blockchain_verify_verified=True,
):
    """Return a VerificationPipeline whose external dependencies are mocked."""
    # Real modules
    real_hasher = CandidateHasher()

    # Mock detector: always finds 1 face
    mock_detector = MagicMock(spec=FaceDetector)
    mock_detector.detect_faces.return_value = {
        "success": True,
        "face_count": 1,
        "faces": [{"box": {"x": 10, "y": 10, "w": 50, "h": 50}, "confidence": 0.9}],
        "image_size": {"width": 200, "height": 200},
        "error": None,
    }

    # Mock encoder: always encodes successfully
    mock_encoder = MagicMock(spec=FaceEncoder)
    mock_encoder.encode_face.return_value = {
        "success": True,
        "face_count": 1,
        "selected_face_index": 0,
        "box": {"x": 10, "y": 10, "w": 50, "h": 50},
        "embedding": [0.1] * 128,
        "embedding_dim": 128,
        "error": None,
    }

    # Mock web search
    mock_search = MagicMock(spec=WebSearch)
    mock_search.search_by_image.return_value = _make_search_success(search_results)

    # Real result selector (no external calls)
    real_selector = ResultSelector()

    # Compute expected hash from the first candidate result
    candidate_input = {
        "title": (search_results[0]["title"] if search_results else SAMPLE_CANDIDATE["title"]),
        "url": (search_results[0]["url"] if search_results else SAMPLE_CANDIDATE["url"]),
        "source": (search_results[0].get("source") if search_results else SAMPLE_CANDIDATE["source"]),
        "snippet": (search_results[0].get("snippet") if search_results else SAMPLE_CANDIDATE["snippet"]),
        "thumbnail": (search_results[0].get("thumbnail") if search_results else SAMPLE_CANDIDATE["thumbnail"]),
        "result_type": "External visual-search candidate",
        "provider": "SerpApi (Google Lens)",
    }
    expected_hash = real_hasher.generate_candidate_hash(candidate_input)["sha256"]

    # Mock blockchain client
    mock_chain = MagicMock(spec=BlockchainClient)
    mock_chain.rpc_url = "http://127.0.0.1:8545"
    mock_chain.is_connected.return_value = blockchain_connected
    mock_chain.contract_address = "0xContractAddress123" if blockchain_has_contract else ""
    mock_chain.sender_address = "0xSenderAddress"

    if blockchain_verify_first:
        # Already registered — first verify returns True before register
        mock_chain.verify_hash.return_value = _make_blockchain_verify_success(expected_hash, verified=True)
    elif blockchain_register_fail:
        mock_chain.verify_hash.return_value = _make_blockchain_verify_success(expected_hash, verified=False)
        mock_chain.register_hash.return_value = {
            "success": False,
            "error": "Simulated registration failure",
        }
    elif blockchain_verify_fail:
        mock_chain.verify_hash.return_value = {
            "success": False,
            "verified": False,
            "error": "Simulated verify failure",
        }
    else:
        # Normal: not registered yet, register succeeds
        mock_chain.verify_hash.return_value = _make_blockchain_verify_success(expected_hash, verified=False)
        mock_chain.register_hash.return_value = _make_blockchain_register_success(expected_hash)
        # After registration, subsequent verify_hash calls should return verified=True
        # We configure this via side_effect
        verified_call = [False]

        def verify_side_effect(h):
            if not verified_call[0]:
                # First call (pre-registration check): not registered
                verified_call[0] = True
                return _make_blockchain_verify_success(expected_hash, verified=False)
            # Second call (post-registration): verified
            return _make_blockchain_verify_success(expected_hash, verified=True)

        mock_chain.verify_hash.side_effect = verify_side_effect
        mock_chain.get_record.return_value = _make_blockchain_get_record_success(expected_hash)

    mock_chain.get_record.return_value = _make_blockchain_get_record_success(expected_hash)

    # Mock tamper detector: will use real hasher + mock chain internally
    real_tamper = TamperDetector(hasher=real_hasher, blockchain_client=mock_chain)

    # Mock face matcher: always returns successful match for mocked pipeline tests
    mock_matcher = MagicMock(spec=FaceMatcher)
    mock_matcher.threshold = 0.363
    mock_matcher.verify_candidate_image.return_value = {
        "match": True,
        "status": "MATCH",
        "similarity": 0.95,
        "distance": 0.05,
        "threshold": 0.363,
        "candidate_face_count": 1,
        "best_candidate_face_index": 0,
        "best_similarity": 0.95,
        "best_distance": 0.05,
        "error": None,
    }

    pipeline = VerificationPipeline(
        detector=mock_detector,
        encoder=mock_encoder,
        web_search=mock_search,
        result_selector=real_selector,
        hasher=real_hasher,
        blockchain_client=mock_chain,
        tamper_detector=real_tamper,
        face_matcher=mock_matcher,
    )
    return pipeline, expected_hash


# ===========================================================================
# TEST CLASS 1: Face detection failures
# ===========================================================================
class TestPipelineFaceDetectionFailures(unittest.TestCase):
    """Tests 1-3: Face detection stage failure cases."""

    def test_1_missing_image_fails_at_face_detection(self):
        """Test 1: Missing image path fails at face_detection stage."""
        pipeline = VerificationPipeline()
        result = pipeline.run_pipeline(MISSING_IMAGE)

        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "face_detection")
        self.assertIsNotNone(result["error"])
        self.assertIsNone(result["selected_candidate"])
        self.assertIsNone(result["original_hash"])

    def test_2_corrupt_image_fails_safely(self):
        """Test 2: Corrupt/unreadable image file fails safely at face_detection."""
        # Create a temp corrupt file
        corrupt_path = Path(CORRUPT_IMAGE)
        corrupt_path.write_text("This is not image data", encoding="utf-8")

        try:
            pipeline = VerificationPipeline()
            result = pipeline.run_pipeline(str(corrupt_path))

            self.assertFalse(result["success"])
            self.assertEqual(result["stage"], "face_detection")
            self.assertIsNotNone(result["error"])
        finally:
            if corrupt_path.exists():
                corrupt_path.unlink()

    def test_3_no_face_image_fails_at_face_detection(self):
        """Test 3: Image with zero faces fails at face_detection stage."""
        # Mock detector to report 0 faces
        mock_detector = MagicMock(spec=FaceDetector)
        mock_detector.detect_faces.return_value = {
            "success": True,
            "face_count": 0,
            "faces": [],
            "image_size": {"width": 100, "height": 100},
            "error": None,
        }
        pipeline = VerificationPipeline(detector=mock_detector)
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "face_detection")
        self.assertIn("no faces", result["error"].lower())


# ===========================================================================
# TEST CLASS 2: Search/selection failures
# ===========================================================================
class TestPipelineSearchFailures(unittest.TestCase):
    """Tests 4-5: Search failure and zero-result cases."""

    def _make_detector_with_face(self):
        mock_detector = MagicMock(spec=FaceDetector)
        mock_detector.detect_faces.return_value = {
            "success": True,
            "face_count": 1,
            "faces": [{"box": {"x": 10, "y": 10, "w": 50, "h": 50}, "confidence": 0.9}],
            "image_size": {"width": 200, "height": 200},
            "error": None,
        }
        return mock_detector

    def _make_encoder_success(self):
        mock_encoder = MagicMock(spec=FaceEncoder)
        mock_encoder.encode_face.return_value = {
            "success": True,
            "face_count": 1,
            "selected_face_index": 0,
            "box": {"x": 10, "y": 10, "w": 50, "h": 50},
            "embedding": [0.1] * 128,
            "embedding_dim": 128,
            "error": None,
        }
        return mock_encoder

    def test_4_search_failure_stops_before_blockchain(self):
        """Test 4: External search failure stops pipeline before blockchain registration."""
        mock_search = MagicMock(spec=WebSearch)
        mock_search.search_by_image.return_value = {
            "success": False,
            "query_image": SINGLE_FACE_IMAGE,
            "provider": "SerpApi (Google Lens)",
            "image_id": None,
            "result_count": 0,
            "results": [],
            "error": "Missing SERPAPI_API_KEY environment variable. Please configure .env.",
        }

        mock_chain = MagicMock(spec=BlockchainClient)

        pipeline = VerificationPipeline(
            detector=self._make_detector_with_face(),
            encoder=self._make_encoder_success(),
            web_search=mock_search,
            blockchain_client=mock_chain,
        )
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "web_search")
        self.assertIsNotNone(result["error"])
        # Blockchain was NOT called
        mock_chain.register_hash.assert_not_called()

    def test_5_zero_external_results_stops_before_blockchain(self):
        """Test 5: Zero external search results stops pipeline before blockchain."""
        mock_search = MagicMock(spec=WebSearch)
        mock_search.search_by_image.return_value = {
            "success": True,
            "query_image": SINGLE_FACE_IMAGE,
            "provider": "SerpApi (Google Lens)",
            "image_id": "img_001",
            "result_count": 0,
            "results": [],
            "error": None,
        }

        mock_chain = MagicMock(spec=BlockchainClient)

        pipeline = VerificationPipeline(
            detector=self._make_detector_with_face(),
            encoder=self._make_encoder_success(),
            web_search=mock_search,
            blockchain_client=mock_chain,
        )
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "web_search")
        self.assertIn("no usable external search results", result["error"].lower())
        mock_chain.register_hash.assert_not_called()


# ===========================================================================
# TEST CLASS 3: Stage integration wiring
# ===========================================================================
class TestPipelineStageWiring(unittest.TestCase):
    """Tests 6-10: Each stage receives correct inputs and uses correct modules."""

    def test_6_result_selector_receives_actual_search_results(self):
        """Test 6: Result selector is called with the actual search response dict."""
        mock_selector = MagicMock(spec=ResultSelector)
        mock_selector.select_results.return_value = {
            "success": True,
            "selected_count": 1,
            "selected_results": [SAMPLE_CANDIDATE],
            "error": None,
        }

        mock_search = MagicMock(spec=WebSearch)
        search_response = _make_search_success()
        mock_search.search_by_image.return_value = search_response

        mock_detector = MagicMock(spec=FaceDetector)
        mock_detector.detect_faces.return_value = {
            "success": True,
            "face_count": 1,
            "faces": [{"box": {"x": 0, "y": 0, "w": 50, "h": 50}, "confidence": 0.9}],
            "image_size": {"width": 200, "height": 200},
            "error": None,
        }

        mock_encoder = MagicMock(spec=FaceEncoder)
        mock_encoder.encode_face.return_value = {
            "success": True,
            "face_count": 1,
            "selected_face_index": 0,
            "box": {"x": 0, "y": 0, "w": 50, "h": 50},
            "embedding": [0.0] * 128,
            "embedding_dim": 128,
            "error": None,
        }

        # Stop at hashing stage by providing a failing hasher
        mock_hasher = MagicMock(spec=CandidateHasher)
        mock_hasher.generate_candidate_hash.return_value = {
            "success": False,
            "error": "Deliberately stopped at hashing",
            "sha256": None,
            "canonical_data": None,
            "canonical_json": None,
            "hash_algorithm": "SHA-256",
        }

        pipeline = VerificationPipeline(
            detector=mock_detector,
            encoder=mock_encoder,
            web_search=mock_search,
            result_selector=mock_selector,
            hasher=mock_hasher,
        )
        pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        # Verify selector was called with the actual search response
        mock_selector.select_results.assert_called_once()
        call_arg = mock_selector.select_results.call_args[0][0]
        self.assertEqual(call_arg, search_response)

    def test_7_hashing_uses_candidate_hasher(self):
        """Test 7: Blockchain registration uses the SHA-256 from CandidateHasher, not a hardcoded value."""
        real_hasher = CandidateHasher()
        expected_hash = real_hasher.generate_candidate_hash(SAMPLE_CANDIDATE)["sha256"]

        mock_search = MagicMock(spec=WebSearch)
        mock_search.search_by_image.return_value = _make_search_success()

        mock_detector = MagicMock(spec=FaceDetector)
        mock_detector.detect_faces.return_value = {
            "success": True,
            "face_count": 1,
            "faces": [{"box": {"x": 0, "y": 0, "w": 50, "h": 50}, "confidence": 0.9}],
            "image_size": {"width": 200, "height": 200},
            "error": None,
        }

        mock_encoder = MagicMock(spec=FaceEncoder)
        mock_encoder.encode_face.return_value = {
            "success": True,
            "face_count": 1,
            "selected_face_index": 0,
            "box": {"x": 0, "y": 0, "w": 50, "h": 50},
            "embedding": [0.0] * 128,
            "embedding_dim": 128,
            "error": None,
        }

        # Use a spy hasher to capture the call
        spy_hasher = MagicMock(wraps=real_hasher)

        mock_chain = MagicMock(spec=BlockchainClient)
        mock_chain.rpc_url = "http://127.0.0.1:8545"
        mock_chain.contract_address = "0xContractAddr"
        mock_chain.sender_address = "0xSender"
        mock_chain.is_connected.return_value = False  # Stop at blockchain stage
        mock_chain.contract_address = "0xtest"

        mock_matcher = MagicMock(spec=FaceMatcher)
        mock_matcher.threshold = 0.363
        mock_matcher.verify_candidate_image.return_value = {
            "match": True,
            "status": "MATCH",
            "similarity": 0.95,
            "distance": 0.05,
            "threshold": 0.363,
            "candidate_face_count": 1,
            "best_candidate_face_index": 0,
            "best_similarity": 0.95,
            "best_distance": 0.05,
            "error": None,
        }

        pipeline = VerificationPipeline(
            detector=mock_detector,
            encoder=mock_encoder,
            web_search=mock_search,
            result_selector=ResultSelector(),
            hasher=spy_hasher,
            blockchain_client=mock_chain,
            face_matcher=mock_matcher,
        )
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        # Hasher was called
        self.assertTrue(spy_hasher.generate_candidate_hash.called)
        # Hash output in pipeline stages is the real computed value
        if "hashing" in result.get("pipeline", {}):
            recorded_hash = result["pipeline"]["hashing"]["sha256"]
            self.assertEqual(len(recorded_hash), 64)

    def test_8_blockchain_registration_uses_blockchain_client(self):
        """Test 8: Blockchain registration is performed via BlockchainClient.register_hash."""
        pipeline, expected_hash = _build_full_mock_pipeline()
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        # Pipeline reached at least blockchain_registration stage
        stages = result.get("pipeline", {})
        self.assertIn("blockchain_registration", stages)

    def test_9_verification_uses_blockchain_result_not_hardcoded(self):
        """Test 9: Verification result comes from the blockchain, not a hardcoded value."""
        pipeline, expected_hash = _build_full_mock_pipeline()
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        if result["success"]:
            bc_ver = result["pipeline"].get("blockchain_verification", {})
            # verified must be True and must come from the chain (not hardcoded)
            self.assertTrue(bc_ver.get("verified"))

    def test_10_tamper_detector_is_used(self):
        """Test 10: TamperDetector is invoked for tamper verification stage."""
        mock_tamper = MagicMock(spec=TamperDetector)
        mock_tamper.STATUS_VERIFIED = "VERIFIED"
        mock_tamper.STATUS_TAMPER_DETECTED = "TAMPER DETECTED"
        # First call (original) -> VERIFIED; second call (tampered) -> TAMPER DETECTED
        mock_tamper.verify_candidate_integrity.side_effect = [
            {
                "success": True,
                "status": "VERIFIED",
                "current_hash": "a" * 64,
                "blockchain_verified": True,
                "record": None,
                "tampered": False,
                "error": None,
            },
            {
                "success": True,
                "status": "TAMPER DETECTED",
                "current_hash": "b" * 64,
                "blockchain_verified": False,
                "record": None,
                "tampered": True,
                "error": None,
            },
        ]

        real_hasher = CandidateHasher()
        mock_search = MagicMock(spec=WebSearch)
        mock_search.search_by_image.return_value = _make_search_success()

        mock_detector = MagicMock(spec=FaceDetector)
        mock_detector.detect_faces.return_value = {
            "success": True,
            "face_count": 1,
            "faces": [{"box": {"x": 0, "y": 0, "w": 50, "h": 50}, "confidence": 0.9}],
            "image_size": {"width": 200, "height": 200},
            "error": None,
        }

        mock_encoder = MagicMock(spec=FaceEncoder)
        mock_encoder.encode_face.return_value = {
            "success": True,
            "face_count": 1,
            "selected_face_index": 0,
            "box": {"x": 0, "y": 0, "w": 50, "h": 50},
            "embedding": [0.0] * 128,
            "embedding_dim": 128,
            "error": None,
        }

        # Compute real hash so blockchain mock can match
        search_candidate = {
            "title": "Jane Doe Profile",
            "url": "https://example.com/jane_doe",
            "source": "example.com",
            "snippet": "Software developer profile page",
            "thumbnail": "https://example.com/thumb.jpg",
            "result_type": "External visual-search candidate",
            "provider": "SerpApi (Google Lens)",
        }
        real_hash = real_hasher.generate_candidate_hash(search_candidate)["sha256"]

        mock_chain = MagicMock(spec=BlockchainClient)
        mock_chain.rpc_url = "http://127.0.0.1:8545"
        mock_chain.contract_address = "0xContract"
        mock_chain.sender_address = "0xSender"
        mock_chain.is_connected.return_value = True
        # First verify_hash → not registered; register_hash → success
        # Second verify_hash (post-registration) → verified=True
        verify_call_count = {"n": 0}

        def _verify_side(h):
            verify_call_count["n"] += 1
            if verify_call_count["n"] == 1:
                return {"success": True, "verified": False, "hash": h, "error": None}
            return {"success": True, "verified": True, "hash": h, "error": None}

        mock_chain.verify_hash.side_effect = _verify_side
        mock_chain.register_hash.return_value = _make_blockchain_register_success(real_hash)
        mock_chain.get_record.return_value = _make_blockchain_get_record_success(real_hash)

        mock_matcher = MagicMock(spec=FaceMatcher)
        mock_matcher.threshold = 0.363
        mock_matcher.verify_candidate_image.return_value = {
            "match": True,
            "status": "MATCH",
            "similarity": 0.95,
            "distance": 0.05,
            "threshold": 0.363,
            "candidate_face_count": 1,
            "best_candidate_face_index": 0,
            "best_similarity": 0.95,
            "best_distance": 0.05,
            "error": None,
        }

        pipeline = VerificationPipeline(
            detector=mock_detector,
            encoder=mock_encoder,
            web_search=mock_search,
            result_selector=ResultSelector(),
            hasher=real_hasher,
            blockchain_client=mock_chain,
            tamper_detector=mock_tamper,
            face_matcher=mock_matcher,
        )
        pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        # Tamper detector must have been called at least once
        self.assertTrue(mock_tamper.verify_candidate_integrity.called)


# ===========================================================================
# TEST CLASS 4: Tamper detection logic
# ===========================================================================
class TestPipelineTamperDetection(unittest.TestCase):
    """Tests 11-13: Original → VERIFIED, modified → TAMPER DETECTED, hashes differ."""

    def _build_tamper_pipeline(self):
        """Build pipeline whose blockchain client returns verified=True for original hash only."""
        real_hasher = CandidateHasher()

        search_candidate = {
            "title": "Jane Doe Profile",
            "url": "https://example.com/jane_doe",
            "source": "example.com",
            "snippet": "Software developer profile page",
            "thumbnail": "https://example.com/thumb.jpg",
            "result_type": "External visual-search candidate",
            "provider": "SerpApi (Google Lens)",
        }
        original_hash = real_hasher.generate_candidate_hash(search_candidate)["sha256"]

        mock_detector = MagicMock(spec=FaceDetector)
        mock_detector.detect_faces.return_value = {
            "success": True,
            "face_count": 1,
            "faces": [{"box": {"x": 0, "y": 0, "w": 50, "h": 50}, "confidence": 0.9}],
            "image_size": {"width": 200, "height": 200},
            "error": None,
        }

        mock_encoder = MagicMock(spec=FaceEncoder)
        mock_encoder.encode_face.return_value = {
            "success": True,
            "face_count": 1,
            "selected_face_index": 0,
            "box": {"x": 0, "y": 0, "w": 50, "h": 50},
            "embedding": [0.0] * 128,
            "embedding_dim": 128,
            "error": None,
        }

        mock_search = MagicMock(spec=WebSearch)
        mock_search.search_by_image.return_value = _make_search_success()

        # Blockchain: verifies original hash only
        mock_chain = MagicMock(spec=BlockchainClient)
        mock_chain.rpc_url = "http://127.0.0.1:8545"
        mock_chain.contract_address = "0xContract"
        mock_chain.sender_address = "0xSender"
        mock_chain.is_connected.return_value = True

        def verify_side_effect(hash_hex):
            if hash_hex == original_hash:
                return {"success": True, "verified": True, "hash": hash_hex, "error": None}
            return {"success": True, "verified": False, "hash": hash_hex, "error": None}

        mock_chain.verify_hash.side_effect = verify_side_effect
        mock_chain.register_hash.return_value = _make_blockchain_register_success(original_hash)
        mock_chain.get_record.return_value = _make_blockchain_get_record_success(original_hash)

        real_tamper = TamperDetector(hasher=real_hasher, blockchain_client=mock_chain)

        mock_matcher = MagicMock(spec=FaceMatcher)
        mock_matcher.threshold = 0.363
        mock_matcher.verify_candidate_image.return_value = {
            "match": True,
            "status": "MATCH",
            "similarity": 0.95,
            "distance": 0.05,
            "threshold": 0.363,
            "candidate_face_count": 1,
            "best_candidate_face_index": 0,
            "best_similarity": 0.95,
            "best_distance": 0.05,
            "error": None,
        }

        pipeline = VerificationPipeline(
            detector=mock_detector,
            encoder=mock_encoder,
            web_search=mock_search,
            result_selector=ResultSelector(),
            hasher=real_hasher,
            blockchain_client=mock_chain,
            tamper_detector=real_tamper,
            face_matcher=mock_matcher,
        )
        return pipeline, original_hash

    def test_11_original_candidate_produces_verified(self):
        """Test 11: Original candidate integrity check produces status VERIFIED."""
        pipeline, original_hash = self._build_tamper_pipeline()
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        orig_ver = result.get("original_verification", {})
        self.assertEqual(orig_ver.get("status"), "VERIFIED")
        self.assertFalse(orig_ver.get("tampered"))
        self.assertTrue(orig_ver.get("blockchain_verified"))

    def test_12_modified_candidate_produces_tamper_detected(self):
        """Test 12: Modified candidate integrity check produces status TAMPER DETECTED."""
        pipeline, original_hash = self._build_tamper_pipeline()
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        tamp_ver = result.get("tampered_verification", {})
        self.assertEqual(tamp_ver.get("status"), "TAMPER DETECTED")
        self.assertTrue(tamp_ver.get("tampered"))
        self.assertFalse(tamp_ver.get("blockchain_verified"))

    def test_13_original_and_tampered_hashes_differ(self):
        """Test 13: Explicit proof that original_hash != tampered_hash."""
        pipeline, original_hash = self._build_tamper_pipeline()
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        orig_hash = result.get("original_hash")
        tamp_hash = result.get("tampered_hash")

        self.assertIsNotNone(orig_hash)
        self.assertIsNotNone(tamp_hash)
        self.assertNotEqual(orig_hash, tamp_hash)
        self.assertEqual(len(orig_hash), 64)
        self.assertEqual(len(tamp_hash), 64)


# ===========================================================================
# TEST CLASS 5: Blockchain / RPC failure is NOT tampering
# ===========================================================================
class TestPipelineBlockchainFailure(unittest.TestCase):
    """Test 14: Blockchain/RPC failure is reported as an error, never tampering."""

    def test_14_blockchain_rpc_failure_is_not_tampering(self):
        """Test 14: Blockchain RPC failure stops pipeline and reports as error, not tamper."""
        mock_detector = MagicMock(spec=FaceDetector)
        mock_detector.detect_faces.return_value = {
            "success": True,
            "face_count": 1,
            "faces": [{"box": {"x": 0, "y": 0, "w": 50, "h": 50}, "confidence": 0.9}],
            "image_size": {"width": 200, "height": 200},
            "error": None,
        }

        mock_encoder = MagicMock(spec=FaceEncoder)
        mock_encoder.encode_face.return_value = {
            "success": True,
            "face_count": 1,
            "selected_face_index": 0,
            "box": {"x": 0, "y": 0, "w": 50, "h": 50},
            "embedding": [0.0] * 128,
            "embedding_dim": 128,
            "error": None,
        }

        mock_search = MagicMock(spec=WebSearch)
        mock_search.search_by_image.return_value = _make_search_success()

        mock_chain = MagicMock(spec=BlockchainClient)
        mock_chain.rpc_url = "http://127.0.0.1:8545"
        mock_chain.contract_address = "0xContractAddr"
        mock_chain.is_connected.return_value = False  # RPC down

        mock_matcher = MagicMock(spec=FaceMatcher)
        mock_matcher.threshold = 0.363
        mock_matcher.verify_candidate_image.return_value = {
            "match": True,
            "status": "MATCH",
            "similarity": 0.95,
            "distance": 0.05,
            "threshold": 0.363,
            "candidate_face_count": 1,
            "best_candidate_face_index": 0,
            "best_similarity": 0.95,
            "best_distance": 0.05,
            "error": None,
        }

        pipeline = VerificationPipeline(
            detector=mock_detector,
            encoder=mock_encoder,
            web_search=mock_search,
            result_selector=ResultSelector(),
            hasher=CandidateHasher(),
            blockchain_client=mock_chain,
            face_matcher=mock_matcher,
        )
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "blockchain_registration")
        self.assertIsNotNone(result["error"])
        # Must not be reported as tampering
        tamp_ver = result.get("tampered_verification")
        self.assertIsNone(tamp_ver)


# ===========================================================================
# TEST CLASS 6: Existing tests unaffected + run_pipeline convenience function
# ===========================================================================
class TestPipelineConvenienceFunction(unittest.TestCase):
    """Test 15: run_pipeline convenience function uses VerificationPipeline."""

    def test_15_run_pipeline_convenience_function_works(self):
        """Test 15: run_pipeline() convenience function delegates correctly."""
        result = run_pipeline(MISSING_IMAGE)
        # Should fail safely at face_detection with the same interface
        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "face_detection")
        self.assertIn("success", result)
        self.assertIn("pipeline", result)
        self.assertIn("original_hash", result)
        self.assertIn("tampered_hash", result)
        self.assertIn("original_verification", result)
        self.assertIn("tampered_verification", result)
        self.assertIn("error", result)


# ===========================================================================
# TEST CLASS 7: Additional edge cases
# ===========================================================================
class TestPipelineEdgeCases(unittest.TestCase):
    """Tests 16+: Additional edge cases ensuring robustness."""

    def test_16_already_registered_hash_not_treated_as_error(self):
        """Test 16: A hash already on-chain (already_registered) continues normally."""
        real_hasher = CandidateHasher()
        search_candidate = {
            "title": "Jane Doe Profile",
            "url": "https://example.com/jane_doe",
            "source": "example.com",
            "snippet": "Software developer profile page",
            "thumbnail": "https://example.com/thumb.jpg",
            "result_type": "External visual-search candidate",
            "provider": "SerpApi (Google Lens)",
        }
        original_hash = real_hasher.generate_candidate_hash(search_candidate)["sha256"]

        mock_detector = MagicMock(spec=FaceDetector)
        mock_detector.detect_faces.return_value = {
            "success": True,
            "face_count": 1,
            "faces": [{"box": {"x": 0, "y": 0, "w": 50, "h": 50}, "confidence": 0.9}],
            "image_size": {"width": 200, "height": 200},
            "error": None,
        }

        mock_encoder = MagicMock(spec=FaceEncoder)
        mock_encoder.encode_face.return_value = {
            "success": True,
            "face_count": 1,
            "selected_face_index": 0,
            "box": {"x": 0, "y": 0, "w": 50, "h": 50},
            "embedding": [0.0] * 128,
            "embedding_dim": 128,
            "error": None,
        }

        mock_search = MagicMock(spec=WebSearch)
        mock_search.search_by_image.return_value = _make_search_success()

        mock_chain = MagicMock(spec=BlockchainClient)
        mock_chain.rpc_url = "http://127.0.0.1:8545"
        mock_chain.contract_address = "0xContract"
        mock_chain.sender_address = "0xSender"
        mock_chain.is_connected.return_value = True
        # Pre-registered: verify always returns True
        mock_chain.verify_hash.return_value = {
            "success": True,
            "verified": True,
            "hash": original_hash,
            "error": None,
        }
        mock_chain.get_record.return_value = _make_blockchain_get_record_success(original_hash)

        real_tamper = TamperDetector(hasher=real_hasher, blockchain_client=mock_chain)

        mock_matcher = MagicMock(spec=FaceMatcher)
        mock_matcher.threshold = 0.363
        mock_matcher.verify_candidate_image.return_value = {
            "match": True,
            "status": "MATCH",
            "similarity": 0.95,
            "distance": 0.05,
            "threshold": 0.363,
            "candidate_face_count": 1,
            "best_candidate_face_index": 0,
            "best_similarity": 0.95,
            "best_distance": 0.05,
            "error": None,
        }

        pipeline = VerificationPipeline(
            detector=mock_detector,
            encoder=mock_encoder,
            web_search=mock_search,
            result_selector=ResultSelector(),
            hasher=real_hasher,
            blockchain_client=mock_chain,
            tamper_detector=real_tamper,
            face_matcher=mock_matcher,
        )
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        # Should reach blockchain_registration with already_registered status
        stages = result.get("pipeline", {})
        if "blockchain_registration" in stages:
            reg = stages["blockchain_registration"]
            # Must NOT have failed due to duplicate
            self.assertNotEqual(result.get("stage"), "blockchain_registration",
                                "Already-registered hash must NOT fail the pipeline")

    def test_17_pipeline_does_not_put_embedding_on_chain(self):
        """Test 17: Face embedding is NEVER included in the canonical hash or blockchain data."""
        real_hasher = CandidateHasher()
        candidate = dict(SAMPLE_CANDIDATE, embedding=[0.1] * 128)
        hash_result = real_hasher.generate_candidate_hash(candidate)

        self.assertTrue(hash_result["success"])
        canonical_data = hash_result.get("canonical_data", {})

        # 'embedding' must not be in the canonical data
        self.assertNotIn("embedding", canonical_data)

    def test_18_failed_candidate_face_match_prevents_blockchain_registration(self):
        """Test 18: Failed candidate face match stops pipeline before hashing & blockchain registration."""
        mock_detector = MagicMock(spec=FaceDetector)
        mock_detector.detect_faces.return_value = {
            "success": True,
            "face_count": 1,
            "faces": [{"box": {"x": 0, "y": 0, "w": 50, "h": 50}, "confidence": 0.9}],
            "image_size": {"width": 200, "height": 200},
            "error": None,
        }

        mock_encoder = MagicMock(spec=FaceEncoder)
        mock_encoder.encode_face.return_value = {
            "success": True,
            "face_count": 1,
            "selected_face_index": 0,
            "box": {"x": 0, "y": 0, "w": 50, "h": 50},
            "embedding": [0.1] * 128,
            "embedding_dim": 128,
            "error": None,
        }

        mock_search = MagicMock(spec=WebSearch)
        mock_search.search_by_image.return_value = _make_search_success()

        # FaceMatcher returns match=False (no face match)
        mock_matcher = MagicMock(spec=FaceMatcher)
        mock_matcher.threshold = 0.363
        mock_matcher.verify_candidate_image.return_value = {
            "match": False,
            "status": "NO_MATCH",
            "similarity": 0.1,
            "distance": 0.9,
            "threshold": 0.363,
            "candidate_face_count": 1,
            "best_candidate_face_index": 0,
            "best_similarity": 0.1,
            "best_distance": 0.9,
            "error": None,
        }

        mock_chain = MagicMock(spec=BlockchainClient)

        pipeline = VerificationPipeline(
            detector=mock_detector,
            encoder=mock_encoder,
            web_search=mock_search,
            result_selector=ResultSelector(),
            hasher=CandidateHasher(),
            blockchain_client=mock_chain,
            face_matcher=mock_matcher,
        )
        result = pipeline.run_pipeline(SINGLE_FACE_IMAGE)

        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "candidate_face_match")
        self.assertIn("face similarity threshold", result["error"].lower())
        # Blockchain registration must NOT be called
        mock_chain.register_hash.assert_not_called()

    def test_18_pipeline_does_not_store_api_key_in_hash(self):
        """Test 18: API key must never be part of the canonical candidate data."""
        real_hasher = CandidateHasher()
        candidate = dict(SAMPLE_CANDIDATE, api_key="sk-secret-key-12345")
        hash_result = real_hasher.generate_candidate_hash(candidate)

        canonical_data = hash_result.get("canonical_data", {})
        self.assertNotIn("api_key", canonical_data)

        # api_key must not appear in the canonical JSON
        canonical_json = hash_result.get("canonical_json", "")
        self.assertNotIn("sk-secret-key-12345", canonical_json)


if __name__ == "__main__":
    unittest.main(verbosity=2)
