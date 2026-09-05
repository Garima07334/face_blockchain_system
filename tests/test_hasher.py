"""Tests for core/hasher.py Phase 5 canonical data construction and SHA-256 fingerprinting."""

import json
import unittest
from core.hasher import CandidateHasher, generate_candidate_hash


class TestCandidateHasher(unittest.TestCase):
    """Test suite for Phase 5 CandidateHasher."""

    def setUp(self) -> None:
        """Initialize CandidateHasher instance and sample candidate."""
        self.hasher = CandidateHasher()
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

    def test_1_valid_candidate_produces_sha256(self) -> None:
        """Test that a valid candidate produces a successful SHA-256 response."""
        res = self.hasher.generate_candidate_hash(self.sample_candidate)

        self.assertTrue(res["success"])
        self.assertEqual(res["hash_algorithm"], "SHA-256")
        self.assertIsNotNone(res["sha256"])
        self.assertIsNotNone(res["canonical_json"])
        self.assertIsNotNone(res["canonical_data"])
        self.assertIsNone(res["error"])

    def test_2_digest_is_64_lowercase_hex_chars(self) -> None:
        """Test that the SHA-256 digest is exactly 64 lowercase hexadecimal characters."""
        res = generate_candidate_hash(self.sample_candidate)
        sha256_hex = res["sha256"]

        self.assertIsNotNone(sha256_hex)
        self.assertEqual(len(sha256_hex), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in sha256_hex))

    def test_3_same_input_produces_same_hash(self) -> None:
        """Test that the same candidate metadata produces the exact same hash (reproducibility)."""
        res1 = self.hasher.generate_candidate_hash(self.sample_candidate)
        res2 = self.hasher.generate_candidate_hash(self.sample_candidate)

        self.assertEqual(res1["sha256"], res2["sha256"])
        self.assertEqual(res1["canonical_json"], res2["canonical_json"])

    def test_4_key_ordering_does_not_change_hash(self) -> None:
        """Test that field ordering in the input dictionary does not change the hash."""
        cand_ordered = {
            "provider": "SerpApi (Google Lens)",
            "result_type": "External visual-search candidate",
            "snippet": "Software developer profile",
            "source": "example.com",
            "thumbnail": "https://example.com/thumb.jpg",
            "title": "Jane Doe Profile",
            "url": "https://example.com/jane_doe",
        }
        cand_shuffled = {
            "url": "https://example.com/jane_doe",
            "title": "Jane Doe Profile",
            "thumbnail": "https://example.com/thumb.jpg",
            "snippet": "Software developer profile",
            "source": "example.com",
            "result_type": "External visual-search candidate",
            "provider": "SerpApi (Google Lens)",
        }

        hash1 = self.hasher.generate_candidate_hash(cand_ordered)["sha256"]
        hash2 = self.hasher.generate_candidate_hash(cand_shuffled)["sha256"]

        self.assertEqual(hash1, hash2)

    def test_5_missing_optional_fields_are_deterministic(self) -> None:
        """Test that missing optional fields default deterministically to empty string."""
        minimal_cand = {
            "url": "https://example.com/minimal",
        }

        res = self.hasher.generate_candidate_hash(minimal_cand)
        self.assertTrue(res["success"])
        cdata = res["canonical_data"]

        self.assertEqual(cdata["title"], "")
        self.assertEqual(cdata["source"], "")
        self.assertEqual(cdata["snippet"], "")
        self.assertEqual(cdata["thumbnail"], "")
        self.assertEqual(cdata["provider"], "SerpApi (Google Lens)")
        self.assertEqual(cdata["result_type"], "External visual-search candidate")

    def test_6_whitespace_normalization_is_deterministic(self) -> None:
        """Test that leading/trailing whitespace padding is cleaned deterministically."""
        padded_cand = {
            "title": "   Jane Doe Profile  ",
            "url": "  https://example.com/jane_doe   ",
            "source": " example.com ",
            "snippet": "Software developer profile  ",
            "thumbnail": " https://example.com/thumb.jpg ",
        }

        hash_clean = self.hasher.generate_candidate_hash(self.sample_candidate)["sha256"]
        hash_padded = self.hasher.generate_candidate_hash(padded_cand)["sha256"]

        self.assertEqual(hash_clean, hash_padded)

    def test_7_unicode_text_handled_correctly(self) -> None:
        """Test that full UTF-8 Unicode characters are handled correctly without errors."""
        unicode_cand = {
            "title": "山田太郎 - Profile 👤",
            "url": "https://example.jp/user/山田",
            "source": "例.jp",
            "snippet": "プログラマーのプロフィール résumé",
            "thumbnail": "https://example.jp/icon.png",
        }

        res = self.hasher.generate_candidate_hash(unicode_cand)
        self.assertTrue(res["success"])
        self.assertIsNotNone(res["sha256"])
        self.assertIn("山田太郎", res["canonical_json"])

    def test_8_url_included_in_canonical_data(self) -> None:
        """Test that URL is included in the canonical JSON and affects the hash."""
        cand_a = dict(self.sample_candidate, url="https://example.com/url_a")
        cand_b = dict(self.sample_candidate, url="https://example.com/url_b")

        res_a = self.hasher.generate_candidate_hash(cand_a)
        res_b = self.hasher.generate_candidate_hash(cand_b)

        self.assertIn("https://example.com/url_a", res_a["canonical_json"])
        self.assertNotEqual(res_a["sha256"], res_b["sha256"])

    def test_9_selection_rank_invariance(self) -> None:
        """Explicit proof: candidate_A (rank 1) and candidate_B (rank 2) produce IDENTICAL hashes."""
        cand_rank_1 = dict(self.sample_candidate, selection_rank=1)
        cand_rank_2 = dict(self.sample_candidate, selection_rank=2)
        cand_rank_99 = dict(self.sample_candidate, selection_rank=99)

        hash_1 = self.hasher.generate_candidate_hash(cand_rank_1)["sha256"]
        hash_2 = self.hasher.generate_candidate_hash(cand_rank_2)["sha256"]
        hash_99 = self.hasher.generate_candidate_hash(cand_rank_99)["sha256"]

        self.assertEqual(hash_1, hash_2)
        self.assertEqual(hash_1, hash_99)

    def test_10_provider_included(self) -> None:
        """Test that provider is included in canonical data and affects the hash."""
        cand_prov_a = dict(self.sample_candidate, provider="Provider A")
        cand_prov_b = dict(self.sample_candidate, provider="Provider B")

        res_a = self.hasher.generate_candidate_hash(cand_prov_a)
        res_b = self.hasher.generate_candidate_hash(cand_prov_b)

        self.assertEqual(res_a["canonical_data"]["provider"], "Provider A")
        self.assertNotEqual(res_a["sha256"], res_b["sha256"])

    def test_11_result_type_included(self) -> None:
        """Test that result_type is included in canonical data and affects the hash."""
        cand_type_a = dict(self.sample_candidate, result_type="Candidate Type A")
        cand_type_b = dict(self.sample_candidate, result_type="Candidate Type B")

        res_a = self.hasher.generate_candidate_hash(cand_type_a)
        res_b = self.hasher.generate_candidate_hash(cand_type_b)

        self.assertEqual(res_a["canonical_data"]["result_type"], "Candidate Type A")
        self.assertNotEqual(res_a["sha256"], res_b["sha256"])

    def test_12_tamper_test(self) -> None:
        """Explicit proof: modifying title MUST produce a DIFFERENT SHA-256 hash."""
        cand_original = dict(self.sample_candidate, title="Original Title")
        cand_modified = dict(self.sample_candidate, title="Modified Title")

        hash_orig = self.hasher.generate_candidate_hash(cand_original)["sha256"]
        hash_mod = self.hasher.generate_candidate_hash(cand_modified)["sha256"]

        self.assertNotEqual(hash_orig, hash_mod)

    def test_13_raw_image_data_excluded(self) -> None:
        """Test that raw image data fields in input dict are ignored and excluded."""
        cand_with_image = dict(
            self.sample_candidate,
            image_bytes=b"\x89PNG\r\n\x1a\n\x00\x00\x00",
            raw_image="base64_fake_data",
        )

        hash_clean = self.hasher.generate_candidate_hash(self.sample_candidate)["sha256"]
        res_image = self.hasher.generate_candidate_hash(cand_with_image)

        self.assertEqual(hash_clean, res_image["sha256"])
        self.assertNotIn("image_bytes", res_image["canonical_data"])
        self.assertNotIn("raw_image", res_image["canonical_data"])

    def test_14_face_embedding_excluded(self) -> None:
        """Test that face embedding vector fields in input dict are ignored and excluded."""
        cand_with_embedding = dict(
            self.sample_candidate,
            embedding=[0.123, -0.456, 0.789],
            face_vector=[0.0] * 512,
        )

        hash_clean = self.hasher.generate_candidate_hash(self.sample_candidate)["sha256"]
        res_emb = self.hasher.generate_candidate_hash(cand_with_embedding)

        self.assertEqual(hash_clean, res_emb["sha256"])
        self.assertNotIn("embedding", res_emb["canonical_data"])
        self.assertNotIn("face_vector", res_emb["canonical_data"])

    def test_15_api_key_excluded(self) -> None:
        """Test that API keys or credentials in input dict are ignored and excluded."""
        cand_with_key = dict(
            self.sample_candidate,
            api_key="secret_serpapi_key_12345",
            SERPAPI_API_KEY="secret_key",
        )

        hash_clean = self.hasher.generate_candidate_hash(self.sample_candidate)["sha256"]
        res_key = self.hasher.generate_candidate_hash(cand_with_key)

        self.assertEqual(hash_clean, res_key["sha256"])
        self.assertNotIn("api_key", res_key["canonical_json"])
        self.assertNotIn("secret_serpapi_key_12345", res_key["canonical_json"])

    def test_16_timestamp_excluded(self) -> None:
        """Test that transient timestamps and runtime state are ignored and excluded."""
        cand_t1 = dict(self.sample_candidate, timestamp="2026-09-04T10:00:00Z")
        cand_t2 = dict(self.sample_candidate, timestamp="2026-09-04T12:34:56Z")

        hash_t1 = self.hasher.generate_candidate_hash(cand_t1)["sha256"]
        hash_t2 = self.hasher.generate_candidate_hash(cand_t2)["sha256"]

        self.assertEqual(hash_t1, hash_t2)

    def test_17_invalid_input_handling(self) -> None:
        """Test that non-dict or invalid candidate inputs return success=False safely."""
        res_str = self.hasher.generate_candidate_hash("not_a_dict")
        self.assertFalse(res_str["success"])
        self.assertIsNone(res_str["sha256"])
        self.assertIsNotNone(res_str["error"])

        res_none = self.hasher.generate_candidate_hash(None)
        self.assertFalse(res_none["success"])

        res_no_url = self.hasher.generate_candidate_hash({"title": "No URL Candidate"})
        self.assertFalse(res_no_url["success"])
        self.assertIn("url", res_no_url["error"].lower())

    def test_18_canonical_json_is_reproducible(self) -> None:
        """Test that the canonical JSON output string is reproducible and exactly parsed."""
        res = self.hasher.generate_candidate_hash(self.sample_candidate)
        cjson = res["canonical_json"]

        # Parse back JSON and verify contents
        parsed = json.loads(cjson)
        self.assertEqual(parsed["url"], "https://example.com/jane_doe")
        self.assertEqual(parsed["title"], "Jane Doe Profile")
        self.assertEqual(parsed["provider"], "SerpApi (Google Lens)")

        # Verify exact string output format (no spaces around separators)
        expected_json = (
            '{"provider":"SerpApi (Google Lens)",'
            '"result_type":"External visual-search candidate",'
            '"snippet":"Software developer profile",'
            '"source":"example.com",'
            '"thumbnail":"https://example.com/thumb.jpg",'
            '"title":"Jane Doe Profile",'
            '"url":"https://example.com/jane_doe"}'
        )
        self.assertEqual(cjson, expected_json)


if __name__ == "__main__":
    unittest.main()
