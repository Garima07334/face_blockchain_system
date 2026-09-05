"""Tests for web_search module.

Verifies:
- Multipart image upload
- SerpApi image_id generation
- Google Lens search
- Public URL handling
- Image compression
- Rate-limit handling
- Request-layer behavior
"""

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from core.web_search import WebSearch


class TestWebSearch(unittest.TestCase):
    """Test suite for WebSearch reverse-image search client."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize test paths and test API key."""

        cls.test_faces_dir = (
            Path(__file__).parent.parent
            / "sample_data"
            / "test_faces"
        )

        cls.dummy_api_key = (
            "test_serpapi_key_98765"
        )

    # ============================================================
    # TEST 1: MISSING API KEY
    # ============================================================

    def test_missing_api_key(self) -> None:
        """Test search behavior when API key is missing."""

        client = WebSearch(api_key="")

        result = client.search_by_image(
            "https://example.com/test.jpg"
        )

        self.assertFalse(
            result["success"]
        )

        self.assertEqual(
            result["result_count"],
            0,
        )

        self.assertIsNotNone(
            result["error"]
        )

        self.assertIn(
            "missing serpapi_api_key",
            result["error"].lower(),
        )

    # ============================================================
    # TEST 2: MISSING LOCAL IMAGE
    # ============================================================

    def test_missing_local_image_file(self) -> None:
        """Test search behavior when local image does not exist."""

        client = WebSearch(
            api_key=self.dummy_api_key
        )

        missing_path = (
            self.test_faces_dir
            / "non_existent_file_9999.jpg"
        )

        result = client.search_by_image(
            missing_path
        )

        self.assertFalse(
            result["success"]
        )

        self.assertEqual(
            result["result_count"],
            0,
        )

        self.assertIsNotNone(
            result["error"]
        )

        self.assertIn(
            "not found",
            result["error"].lower(),
        )

    # ============================================================
    # TEST 3: LOCAL IMAGE -> UPLOAD -> IMAGE ID -> LENS
    # ============================================================

    @patch.object(
        WebSearch,
        "_search_google_lens",
    )
    @patch.object(
        WebSearch,
        "_upload_image_to_serpapi",
    )
    def test_local_image_upload_and_image_id_flow(
        self,
        mock_upload: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        """
        Test local image flow:

        Local image
            -> SerpApi upload
            -> image_id
            -> Google Lens search
        """

        mock_upload.return_value = (
            "img_id_abc123"
        )

        mock_search.return_value = {
            "visual_matches": [
                {
                    "title": "Visual Match 1",
                    "link": (
                        "https://example.com/match1"
                    ),
                    "source": "example.com",
                    "snippet": (
                        "Visual match description"
                    ),
                    "thumbnail": (
                        "https://example.com/"
                        "thumb1.jpg"
                    ),
                }
            ],
            "exact_matches": [
                {
                    "title": "Exact Match 1",
                    "link": (
                        "https://example.com/exact1"
                    ),
                    "source": "exact.com",
                    "snippet": (
                        "Exact match snippet"
                    ),
                    "thumbnail": (
                        "https://exact.com/"
                        "thumb_exact.jpg"
                    ),
                }
            ],
        }

        client = WebSearch(
            api_key=self.dummy_api_key
        )

        local_path = (
            self.test_faces_dir
            / "single_face.jpg"
        )

        result = client.search_by_image(
            local_path
        )

        # --------------------------------------------------------
        # Upload must happen exactly once
        # --------------------------------------------------------

        mock_upload.assert_called_once()

        args, kwargs = (
            mock_upload.call_args
        )

        # Support both:
        #
        #   _upload_image_to_serpapi(
        #       filename,
        #       image_bytes,
        #       mime_type
        #   )
        #
        # and:
        #
        #   _upload_image_to_serpapi(
        #       filename=...,
        #       image_bytes=...,
        #       mime_type=...
        #   )

        filename_arg = (
            kwargs.get("filename")
        )

        if filename_arg is None and len(args) > 0:
            filename_arg = args[0]

        image_bytes_arg = (
            kwargs.get("image_bytes")
        )

        if image_bytes_arg is None and len(args) > 1:
            image_bytes_arg = args[1]

        mime_arg = (
            kwargs.get("mime_type")
        )

        if mime_arg is None and len(args) > 2:
            mime_arg = args[2]

        # --------------------------------------------------------
        # Validate upload payload
        # --------------------------------------------------------

        self.assertEqual(
            filename_arg,
            "single_face.jpg",
        )

        self.assertIsNotNone(
            image_bytes_arg
        )

        self.assertGreater(
            len(image_bytes_arg),
            0,
        )

        # --------------------------------------------------------
        # Google Lens must receive image_id
        # --------------------------------------------------------

        mock_search.assert_called_once_with(
            image_url=None,
            image_id="img_id_abc123",
        )

        # --------------------------------------------------------
        # Validate final search response
        # --------------------------------------------------------

        self.assertTrue(
            result["success"]
        )

        self.assertEqual(
            result["image_id"],
            "img_id_abc123",
        )

        self.assertEqual(
            result["result_count"],
            2,
        )

        self.assertEqual(
            len(result["results"]),
            2,
        )

        self.assertIsNone(
            result["error"]
        )

    # ============================================================
    # TEST 4: REQUEST-LAYER VERIFICATION
    # ============================================================

    @patch("requests.post")
    @patch("requests.get")
    def test_end_to_end_request_layer_verification(
        self,
        mock_get: MagicMock,
        mock_post: MagicMock,
    ) -> None:
        """
        Test HTTP request parameters for:

        multipart POST upload
        +
        Google Lens GET search
        """

        # --------------------------------------------------------
        # Mock upload POST response
        # --------------------------------------------------------

        mock_upload_resp = MagicMock()

        mock_upload_resp.status_code = 200

        mock_upload_resp.json.return_value = {
            "image_id": (
                "test_gen_img_id_999"
            )
        }

        mock_post.return_value = (
            mock_upload_resp
        )

        # --------------------------------------------------------
        # Mock Google Lens GET response
        # --------------------------------------------------------

        mock_search_resp = MagicMock()

        mock_search_resp.status_code = 200

        mock_search_resp.json.return_value = {
            "visual_matches": [
                {
                    "title": "Match 1",
                    "link": (
                        "https://a.com/1"
                    ),
                    "source": "a.com",
                    "snippet": "Text 1",
                }
            ]
        }

        mock_get.return_value = (
            mock_search_resp
        )

        client = WebSearch(
            api_key=self.dummy_api_key
        )

        local_path = (
            self.test_faces_dir
            / "single_face.jpg"
        )

        result = client.search_by_image(
            local_path
        )

        # --------------------------------------------------------
        # Verify POST upload request
        # --------------------------------------------------------

        mock_post.assert_called_once()

        post_url = (
            mock_post.call_args[0][0]
        )

        self.assertEqual(
            post_url,
            "https://serpapi.com/image",
        )

        post_kwargs = (
            mock_post.call_args[1]
        )

        self.assertIn(
            "files",
            post_kwargs,
        )

        self.assertEqual(
            post_kwargs["data"]["api_key"],
            self.dummy_api_key,
        )

        # --------------------------------------------------------
        # Verify GET Google Lens request
        # --------------------------------------------------------

        mock_get.assert_called_once()

        get_url = (
            mock_get.call_args[0][0]
        )

        self.assertEqual(
            get_url,
            "https://serpapi.com/search.json",
        )

        get_params = (
            mock_get.call_args[1]["params"]
        )

        self.assertEqual(
            get_params["engine"],
            "google_lens",
        )

        self.assertEqual(
            get_params["image_id"],
            "test_gen_img_id_999",
        )

        # Local file path must NEVER be passed
        # as a public URL.
        self.assertNotIn(
            "url",
            get_params,
        )

        # --------------------------------------------------------
        # Final assertions
        # --------------------------------------------------------

        self.assertTrue(
            result["success"]
        )

        self.assertEqual(
            result["image_id"],
            "test_gen_img_id_999",
        )

        self.assertEqual(
            result["result_count"],
            1,
        )

    # ============================================================
    # TEST 5: PUBLIC URL BYPASSES UPLOAD
    # ============================================================

    @patch.object(
        WebSearch,
        "_search_google_lens",
    )
    def test_public_url_bypasses_upload(
        self,
        mock_search: MagicMock,
    ) -> None:
        """
        Public image URLs should bypass local upload
        and go directly to Google Lens.
        """

        mock_search.return_value = {
            "visual_matches": []
        }

        client = WebSearch(
            api_key=self.dummy_api_key
        )

        public_url = (
            "https://example.com/"
            "public_photo.jpg"
        )

        result = client.search_by_image(
            public_url
        )

        mock_search.assert_called_once_with(
            image_url=public_url,
            image_id=None,
        )

        self.assertTrue(
            result["success"]
        )

        self.assertIsNone(
            result["image_id"]
        )

    # ============================================================
    # TEST 6: OVERSIZED IMAGE COMPRESSION
    # ============================================================

    def test_oversized_image_compression(
        self,
    ) -> None:
        """
        Images exceeding 500 KB should be
        automatically compressed.
        """

        big_image_path = (
            self.test_faces_dir
            / "test_big_image.jpg"
        )

        big_img = np.random.randint(
            0,
            255,
            (
                1500,
                1500,
                3,
            ),
            dtype=np.uint8,
        )

        cv2.imwrite(
            str(big_image_path),
            big_img,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                100,
            ],
        )

        try:

            client = WebSearch(
                api_key=self.dummy_api_key
            )

            image_bytes, mime_type = (
                client._prepare_image_payload(
                    big_image_path
                )
            )

            self.assertLessEqual(
                len(image_bytes),
                500 * 1024,
            )

            self.assertEqual(
                mime_type,
                "image/jpeg",
            )

        finally:

            if big_image_path.exists():
                big_image_path.unlink()

    # ============================================================
    # TEST 7: RATE LIMIT 429
    # ============================================================

    @patch("requests.post")
    def test_upload_rate_limit_429(
        self,
        mock_post: MagicMock,
    ) -> None:
        """
        Test rate-limit handling during
        SerpApi image upload.
        """

        mock_resp = MagicMock()

        mock_resp.status_code = 429

        mock_post.return_value = (
            mock_resp
        )

        client = WebSearch(
            api_key=self.dummy_api_key
        )

        local_path = (
            self.test_faces_dir
            / "single_face.jpg"
        )

        result = client.search_by_image(
            local_path
        )

        self.assertFalse(
            result["success"]
        )

        self.assertEqual(
            result["result_count"],
            0,
        )

        self.assertIsNotNone(
            result["error"]
        )

        self.assertIn(
            "rate limit",
            result["error"].lower(),
        )


# ================================================================
# TEST RUNNER
# ================================================================

if __name__ == "__main__":
    unittest.main()