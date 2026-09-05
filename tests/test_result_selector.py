"""Tests for core/result_selector.py Phase 4 result selection and normalization."""

import unittest
from core.result_selector import ResultSelector, canonicalize_url, select_results


class TestResultSelector(unittest.TestCase):
    """Test suite for Phase 4 ResultSelector."""

    def setUp(self) -> None:
        """Initialize ResultSelector instance."""
        self.selector = ResultSelector()

    def test_1_single_valid_result(self) -> None:
        """Test processing of a single valid Phase 3 result."""
        search_resp = {
            "success": True,
            "provider": "SerpApi (Google Lens)",
            "result_count": 1,
            "results": [
                {
                    "title": "Jane Doe Profile",
                    "url": "https://example.com/jane_doe",
                    "source": "example.com",
                    "snippet": "Software developer profile",
                    "thumbnail": "https://example.com/thumb.jpg",
                    "result_type": "External reverse-image search result",
                }
            ],
            "error": None,
        }

        output = self.selector.select_results(search_resp)

        self.assertTrue(output["success"])
        self.assertEqual(output["selected_count"], 1)
        self.assertEqual(len(output["selected_results"]), 1)
        self.assertIsNone(output["error"])

        item = output["selected_results"][0]
        self.assertEqual(item["title"], "Jane Doe Profile")
        self.assertEqual(item["url"], "https://example.com/jane_doe")
        self.assertEqual(item["source"], "example.com")
        self.assertEqual(item["snippet"], "Software developer profile")
        self.assertEqual(item["thumbnail"], "https://example.com/thumb.jpg")
        self.assertEqual(item["result_type"], "External visual-search candidate")
        self.assertEqual(item["selection_rank"], 1)
        self.assertEqual(item["provider"], "SerpApi (Google Lens)")

    def test_2_multiple_valid_results(self) -> None:
        """Test processing multiple valid candidate results."""
        search_resp = {
            "success": True,
            "provider": "SerpApi (Google Lens)",
            "result_count": 2,
            "results": [
                {
                    "title": "First Match",
                    "url": "https://a.com/page1",
                    "source": "a.com",
                    "snippet": "Snippet A",
                    "thumbnail": "https://a.com/thumb.jpg",
                },
                {
                    "title": "Second Match",
                    "url": "https://b.com/page2",
                    "source": "b.com",
                    "snippet": "Snippet B",
                    "thumbnail": "https://b.com/thumb.jpg",
                },
            ],
        }

        output = select_results(search_resp)
        self.assertTrue(output["success"])
        self.assertEqual(output["selected_count"], 2)
        self.assertEqual(output["selected_results"][0]["selection_rank"], 1)
        self.assertEqual(output["selected_results"][1]["selection_rank"], 2)

    def test_3_duplicate_urls(self) -> None:
        """Test deduplication of duplicate canonical URLs."""
        search_resp = {
            "success": True,
            "provider": "SerpApi (Google Lens)",
            "result_count": 3,
            "results": [
                {
                    "title": "Page 1 - Direct",
                    "url": "https://example.com/person?utm_source=google",
                    "source": "example.com",
                },
                {
                    "title": "Page 1 - Duplicate with trailing slash",
                    "url": "https://example.com/person/",
                    "source": "example.com",
                },
                {
                    "title": "Distinct Page 2",
                    "url": "https://example.com/other_person",
                    "source": "example.com",
                },
            ],
        }

        output = self.selector.select_results(search_resp)
        self.assertTrue(output["success"])
        self.assertEqual(output["selected_count"], 2)
        urls = [res["url"] for res in output["selected_results"]]
        self.assertIn("https://example.com/person", urls)
        self.assertIn("https://example.com/other_person", urls)

    def test_4_missing_title(self) -> None:
        """Test that missing titles are preserved as None and not invented/inferred."""
        search_resp = {
            "success": True,
            "provider": "SerpApi (Google Lens)",
            "results": [
                {
                    "url": "https://example.com/no_title",
                    "source": "example.com",
                    "snippet": "Snippet without title",
                }
            ],
        }

        output = self.selector.select_results(search_resp)
        self.assertTrue(output["success"])
        self.assertEqual(output["selected_count"], 1)
        res = output["selected_results"][0]
        self.assertIsNone(res["title"])

    def test_5_missing_url(self) -> None:
        """Test that entries with missing or null URLs are safely ignored."""
        search_resp = {
            "success": True,
            "provider": "SerpApi (Google Lens)",
            "results": [
                {"title": "No URL result 1"},
                {"title": "No URL result 2", "url": None},
                {"title": "No URL result 3", "url": "   "},
                {"title": "Valid result", "url": "https://example.com/valid"},
            ],
        }

        output = self.selector.select_results(search_resp)
        self.assertTrue(output["success"])
        self.assertEqual(output["selected_count"], 1)
        self.assertEqual(output["selected_results"][0]["url"], "https://example.com/valid")

    def test_6_invalid_url(self) -> None:
        """Test that entries with non-HTTP/HTTPS or malformed URLs are safely ignored."""
        search_resp = {
            "success": True,
            "provider": "SerpApi (Google Lens)",
            "results": [
                {"title": "FTP URL", "url": "ftp://files.example.com/photo.jpg"},
                {"title": "JS Scheme", "url": "javascript:alert(1)"},
                {"title": "Local file path", "url": "C:/images/face.jpg"},
                {"title": "Valid HTTPS", "url": "https://example.com/valid_photo"},
            ],
        }

        output = self.selector.select_results(search_resp)
        self.assertTrue(output["success"])
        self.assertEqual(output["selected_count"], 1)
        self.assertEqual(output["selected_results"][0]["url"], "https://example.com/valid_photo")

    def test_7_missing_optional_fields(self) -> None:
        """Test handling when optional fields (snippet, thumbnail, source) are missing."""
        search_resp = {
            "success": True,
            "provider": "SerpApi (Google Lens)",
            "results": [
                {
                    "url": "https://example.com/minimal",
                }
            ],
        }

        output = self.selector.select_results(search_resp)
        self.assertTrue(output["success"])
        self.assertEqual(output["selected_count"], 1)
        res = output["selected_results"][0]
        self.assertIsNone(res["title"])
        self.assertIsNone(res["source"])
        self.assertIsNone(res["snippet"])
        self.assertIsNone(res["thumbnail"])

    def test_8_zero_external_results(self) -> None:
        """Test response structure when Phase 3 returns zero search results."""
        search_resp = {
            "success": True,
            "provider": "SerpApi (Google Lens)",
            "result_count": 0,
            "results": [],
            "error": None,
        }

        output = self.selector.select_results(search_resp)
        self.assertTrue(output["success"])
        self.assertEqual(output["selected_count"], 0)
        self.assertEqual(output["selected_results"], [])
        self.assertIsNone(output["error"])

    def test_9_malformed_phase3_response(self) -> None:
        """Test handling of invalid/malformed Phase 3 search responses."""
        # Non-dict input
        res1 = self.selector.select_results("invalid_input_string")
        self.assertFalse(res1["success"])
        self.assertEqual(res1["selected_count"], 0)
        self.assertIsNotNone(res1["error"])

        # success: False input
        res2 = self.selector.select_results({
            "success": False,
            "error": "API Key Invalid",
            "results": [],
        })
        self.assertFalse(res2["success"])
        self.assertEqual(res2["selected_count"], 0)
        self.assertEqual(res2["error"], "API Key Invalid")

        # Invalid results field type
        res3 = self.selector.select_results({
            "success": True,
            "results": "not_a_list",
        })
        self.assertFalse(res3["success"])
        self.assertEqual(res3["selected_count"], 0)

    def test_10_ranking_is_deterministic(self) -> None:
        """Test that ranking logic is completely deterministic across repeated calls."""
        search_resp = {
            "success": True,
            "provider": "SerpApi (Google Lens)",
            "results": [
                {"title": "A", "url": "https://example.com/a", "snippet": "Snippet A"},
                {"title": "B", "url": "https://example.com/b", "source": "example.com", "snippet": "Snippet B", "thumbnail": "https://example.com/t.jpg"},
                {"title": "C", "url": "https://example.com/c"},
            ],
        }

        output1 = self.selector.select_results(search_resp)
        output2 = self.selector.select_results(search_resp)

        self.assertEqual(output1, output2)
        # Ensure B ranked #1 due to highest completeness score (title+source+snippet+thumbnail)
        self.assertEqual(output1["selected_results"][0]["url"], "https://example.com/b")
        self.assertEqual(output1["selected_results"][0]["selection_rank"], 1)

    def test_11_no_fake_results_generated(self) -> None:
        """Test that Phase 4 never invents synthetic/fake results when input is empty or invalid."""
        empty_resp = {
            "success": True,
            "provider": "SerpApi (Google Lens)",
            "results": [],
        }

        output = self.selector.select_results(empty_resp)
        self.assertEqual(output["selected_count"], 0)
        self.assertEqual(output["selected_results"], [])

    def test_12_identity_is_never_claimed(self) -> None:
        """Test that result_type is strictly candidate terminology and no identity claims exist."""
        search_resp = {
            "success": True,
            "provider": "SerpApi (Google Lens)",
            "results": [
                {
                    "title": "Person Profile",
                    "url": "https://example.com/profile",
                    "result_type": "External reverse-image search result",
                }
            ],
        }

        output = self.selector.select_results(search_resp)
        item = output["selected_results"][0]

        self.assertEqual(item["result_type"], "External visual-search candidate")
        self.assertNotIn("Confirmed identity", str(output))
        self.assertNotIn("identity_verified", item)

    def test_13_result_provenance_is_preserved(self) -> None:
        """Test that all original metadata fields and provenance are preserved accurately."""
        search_resp = {
            "success": True,
            "provider": "Custom SerpApi Client",
            "results": [
                {
                    "title": "Original Title",
                    "url": "https://domain.org/path",
                    "source": "domain.org",
                    "snippet": "Original snippet text",
                    "thumbnail": "https://domain.org/thumb.png",
                }
            ],
        }

        output = self.selector.select_results(search_resp)
        item = output["selected_results"][0]

        self.assertEqual(item["provider"], "Custom SerpApi Client")
        self.assertEqual(item["title"], "Original Title")
        self.assertEqual(item["url"], "https://domain.org/path")
        self.assertEqual(item["source"], "domain.org")
        self.assertEqual(item["snippet"], "Original snippet text")
        self.assertEqual(item["thumbnail"], "https://domain.org/thumb.png")


if __name__ == "__main__":
    unittest.main()
