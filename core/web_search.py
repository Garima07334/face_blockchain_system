"""
Web Search module for external reverse-image searching using SerpApi Google Lens.

Flow:
    Local image
        ↓
    SerpApi Image Upload
        ↓
    image_id
        ↓
    Google Lens Search
        ↓
    External visual matches
        ↓
    Normalized candidate results

Important:
- `url` = webpage/post URL
- `image_url` = candidate image URL used for face verification
- `thumbnail` = preview image only
- Search results are NOT treated as identity matches automatically.
- FaceMatcher must verify the candidate face before blockchain notarization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import os
import cv2
import numpy as np
import requests

from dotenv import load_dotenv


load_dotenv()


class WebSearch:
    """External reverse-image search client using SerpApi Google Lens."""

    PROVIDER_NAME = "SerpApi (Google Lens)"

    SERPAPI_SEARCH_ENDPOINT = (
        "https://serpapi.com/search.json"
    )

    SERPAPI_UPLOAD_ENDPOINT = (
        "https://serpapi.com/image"
    )

    MAX_UPLOAD_SIZE_BYTES = 500 * 1024

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 20,
    ) -> None:

        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("SERPAPI_API_KEY")
        )

        self.timeout = timeout

    # ============================================================
    # PUBLIC SEARCH METHOD
    # ============================================================

    def search_by_image(
        self,
        image_input: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Perform genuine external reverse-image search.

        For local files:
            1. Upload image to SerpApi.
            2. Receive image_id.
            3. Search Google Lens using image_id.

        Returns normalized external search results.
        """

        image_str = str(
            image_input
        ).strip()

        # --------------------------------------------------------
        # Validate API key
        # --------------------------------------------------------

        if not self.api_key:

            return self._error_response(
                image_str,
                (
                    "Missing SERPAPI_API_KEY environment "
                    "variable. Please configure .env."
                ),
            )

        image_id: Optional[str] = None
        image_url: Optional[str] = None

        # ========================================================
        # INPUT HANDLING
        # ========================================================

        if self._is_http_url(image_str):

            # Public image URL.
            image_url = image_str

        else:

            # Local image path.
            path = Path(
                image_str
            )

            if not path.is_file():

                return self._error_response(
                    image_str,
                    (
                        "Local image file not found: "
                        f"{image_str}"
                    ),
                )

            # ----------------------------------------------------
            # Prepare image
            # ----------------------------------------------------

            try:

                image_bytes, mime_type = (
                    self._prepare_image_payload(
                        path
                    )
                )

            except Exception as err:

                return self._error_response(
                    image_str,
                    (
                        "Failed to process image file: "
                        f"{err}"
                    ),
                )

            if not image_bytes:

                return self._error_response(
                    image_str,
                    "Local image file is empty.",
                )

            # ----------------------------------------------------
            # Upload to SerpApi
            # ----------------------------------------------------

            try:

                image_id = (
                    self._upload_image_to_serpapi(
                        filename=path.name,
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                    )
                )

            except requests.exceptions.Timeout:

                return self._error_response(
                    image_str,
                    "Image upload request timed out.",
                )

            except requests.exceptions.RequestException as err:

                return self._error_response(
                    image_str,
                    (
                        "Failed to upload image to "
                        f"SerpApi: {err}"
                    ),
                )

            except Exception as err:

                return self._error_response(
                    image_str,
                    (
                        "Image upload error: "
                        f"{err}"
                    ),
                )

            if not image_id:

                return self._error_response(
                    image_str,
                    (
                        "SerpApi Image Upload API did "
                        "not return a valid image_id."
                    ),
                )

        # ========================================================
        # GOOGLE LENS SEARCH
        # ========================================================

        try:

            raw_response = (
                self._search_google_lens(
                    image_url=image_url,
                    image_id=image_id,
                )
            )

        except requests.exceptions.Timeout:

            return self._error_response(
                image_str,
                "Google Lens search request timed out.",
                image_id=image_id,
            )

        except requests.exceptions.RequestException as err:

            return self._error_response(
                image_str,
                (
                    "Google Lens search API request "
                    f"failed: {err}"
                ),
                image_id=image_id,
            )

        except Exception as err:

            return self._error_response(
                image_str,
                (
                    "Unexpected search error: "
                    f"{err}"
                ),
                image_id=image_id,
            )

        # ========================================================
        # PARSE RESULTS
        # ========================================================

        return self._parse_serpapi_response(
            query_image=image_str,
            image_id=image_id,
            response_data=raw_response,
        )

    # ============================================================
    # IMAGE PREPARATION
    # ============================================================

    def _prepare_image_payload(
        self,
        path: Path,
    ) -> Tuple[bytes, str]:
        """
        Read local image and compress if it exceeds
        SerpApi upload limit.
        """

        with open(
            path,
            "rb",
        ) as file:

            data = file.read()

        extension = (
            path.suffix.lower()
        )

        mime_type = (
            "image/png"
            if extension == ".png"
            else "image/jpeg"
        )

        if (
            len(data)
            <= self.MAX_UPLOAD_SIZE_BYTES
        ):

            return data, mime_type

        # --------------------------------------------------------
        # Decode
        # --------------------------------------------------------

        image_array = np.frombuffer(
            data,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise ValueError(
                "OpenCV could not decode image "
                "for compression."
            )

        # --------------------------------------------------------
        # Progressive resize/compression
        # --------------------------------------------------------

        quality = 85
        scale = 0.90

        encoded_bytes = data

        while (
            len(encoded_bytes)
            > self.MAX_UPLOAD_SIZE_BYTES
            and quality >= 20
        ):

            height, width = (
                image.shape[:2]
            )

            new_width = max(
                80,
                int(width * scale),
            )

            new_height = max(
                80,
                int(height * scale),
            )

            image = cv2.resize(
                image,
                (
                    new_width,
                    new_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

            success, buffer = cv2.imencode(
                ".jpg",
                image,
                [
                    int(
                        cv2.IMWRITE_JPEG_QUALITY
                    ),
                    quality,
                ],
            )

            if not success:

                raise ValueError(
                    "Failed to compress image."
                )

            encoded_bytes = (
                buffer.tobytes()
            )

            quality -= 15
            scale *= 0.80

        if (
            len(encoded_bytes)
            > self.MAX_UPLOAD_SIZE_BYTES
        ):

            raise ValueError(
                "Unable to compress image below "
                "the SerpApi upload size limit."
            )

        return (
            encoded_bytes,
            "image/jpeg",
        )

    # ============================================================
    # SERPAPI IMAGE UPLOAD
    # ============================================================

    def _upload_image_to_serpapi(
        self,
        filename: str,
        image_bytes: bytes,
        mime_type: str,
    ) -> str:
        """Upload image and receive SerpApi image_id."""

        files = {
            "image": (
                filename,
                image_bytes,
                mime_type,
            )
        }

        data = {
            "api_key": self.api_key,
        }

        response = requests.post(
            self.SERPAPI_UPLOAD_ENDPOINT,
            files=files,
            data=data,
            timeout=self.timeout,
        )

        if response.status_code == 429:

            raise requests.exceptions.RequestException(
                "Rate limit exceeded during image upload (HTTP 429)."
            )

        if response.status_code != 200:

            raise requests.exceptions.RequestException(
                (
                    "SerpApi upload failed with "
                    f"HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
            )

        try:

            response_json = response.json()

        except ValueError as err:

            raise ValueError(
                "Malformed JSON returned by "
                "SerpApi Image Upload API."
            ) from err

        if not isinstance(
            response_json,
            dict,
        ):

            raise ValueError(
                "SerpApi Image Upload API returned "
                "an invalid response."
            )

        if "error" in response_json:

            raise RuntimeError(
                (
                    "SerpApi Image Upload Error: "
                    f"{response_json['error']}"
                )
            )

        image_id = (
            response_json.get(
                "image_id"
            )
            or response_json.get(
                "id"
            )
        )

        if not image_id:

            raise ValueError(
                (
                    "SerpApi upload response does "
                    "not contain image_id."
                )
            )

        return str(
            image_id
        )

    # ============================================================
    # GOOGLE LENS REQUEST
    # ============================================================

    def _search_google_lens(
        self,
        image_url: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Google Lens search through SerpApi."""

        params: Dict[str, Any] = {
            "engine": "google_lens",
            "api_key": self.api_key,
        }

        if image_id:

            params["image_id"] = image_id

        elif image_url:

            params["url"] = image_url

        else:

            raise ValueError(
                (
                    "Either image_id or image_url "
                    "must be supplied."
                )
            )

        response = requests.get(
            self.SERPAPI_SEARCH_ENDPOINT,
            params=params,
            timeout=self.timeout,
        )

        if response.status_code == 429:

            raise requests.exceptions.RequestException(
                "Google Lens rate limit exceeded (HTTP 429)."
            )

        if response.status_code != 200:

            raise requests.exceptions.RequestException(
                (
                    "Google Lens search returned "
                    f"HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
            )

        try:

            result = response.json()

        except ValueError as err:

            raise ValueError(
                "Malformed JSON response from "
                "Google Lens."
            ) from err

        if not isinstance(
            result,
            dict,
        ):

            raise ValueError(
                "Google Lens returned a non-dictionary response."
            )

        return result

    # ============================================================
    # RESULT PARSING
    # ============================================================

    def _parse_serpapi_response(
        self,
        query_image: str,
        image_id: Optional[str],
        response_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert SerpApi response into normalized candidate results.

        Priority:
            1. visual_matches
            2. exact_matches
            3. organic_results
            4. reverse_image_results
        """

        if not isinstance(
            response_data,
            dict,
        ):

            return self._error_response(
                query_image,
                (
                    "Malformed Google Lens response "
                    "(expected dictionary)."
                ),
                image_id=image_id,
            )

        if "error" in response_data:

            return self._error_response(
                query_image,
                (
                    "SerpApi Error: "
                    f"{response_data['error']}"
                ),
                image_id=image_id,
            )

        parsed_results: List[
            Dict[str, Any]
        ] = []

        # ========================================================
        # 1. VISUAL MATCHES
        # ========================================================

        visual_matches = (
            response_data.get(
                "visual_matches",
                [],
            )
        )

        if isinstance(
            visual_matches,
            list,
        ):

            for rank, item in enumerate(
                visual_matches
            ):

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                result = self._normalize_result(
                    item=item,
                    result_type="visual_match",
                    provider_rank=rank,
                )

                if result:

                    parsed_results.append(
                        result
                    )

        # ========================================================
        # 2. EXACT MATCHES
        # ========================================================

        exact_matches = (
            response_data.get(
                "exact_matches",
                [],
            )
        )

        if isinstance(
            exact_matches,
            list,
        ):

            for rank, item in enumerate(
                exact_matches
            ):

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                result = self._normalize_result(
                    item=item,
                    result_type="exact_match",
                    provider_rank=rank,
                )

                if result:

                    parsed_results.append(
                        result
                    )

        # ========================================================
        # 3. ORGANIC RESULTS
        # ========================================================

        organic_results = (
            response_data.get(
                "organic_results",
                [],
            )
        )

        if isinstance(
            organic_results,
            list,
        ):

            for rank, item in enumerate(
                organic_results
            ):

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                result = self._normalize_result(
                    item=item,
                    result_type="organic_result",
                    provider_rank=rank,
                )

                if result:

                    parsed_results.append(
                        result
                    )

        # ========================================================
        # 4. FALLBACK
        # ========================================================

        if not parsed_results:

            reverse_results = (
                response_data.get(
                    "reverse_image_results",
                    [],
                )
            )

            if isinstance(
                reverse_results,
                list,
            ):

                for rank, item in enumerate(
                    reverse_results
                ):

                    if not isinstance(
                        item,
                        dict,
                    ):
                        continue

                    result = self._normalize_result(
                        item=item,
                        result_type="reverse_image_result",
                        provider_rank=rank,
                    )

                    if result:

                        parsed_results.append(
                            result
                        )

        return {
            "success": True,
            "query_image": query_image,
            "provider": self.PROVIDER_NAME,
            "image_id": image_id,
            "result_count": len(
                parsed_results
            ),
            "results": parsed_results,
            "error": None,
        }

    # ============================================================
    # RESULT NORMALIZATION
    # ============================================================

    def _normalize_result(
        self,
        item: Dict[str, Any],
        result_type: str,
        provider_rank: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Normalize a single SerpApi result.

        IMPORTANT:

        url:
            The actual webpage/post URL.

        image_url:
            The candidate image URL used by FaceMatcher.

        thumbnail:
            Preview image URL.

        We deliberately keep these fields separate.
        """

        # --------------------------------------------------------
        # Webpage URL
        # --------------------------------------------------------

        url = self._first_url(
            item,
            [
                "link",
                "url",
                "source_url",
                "page_url",
            ],
        )

        if not url:

            return None

        # --------------------------------------------------------
        # Title
        # --------------------------------------------------------

        title = self._first_string(
            item,
            [
                "title",
                "name",
            ],
            default="Untitled Result",
        )

        # --------------------------------------------------------
        # Source/domain
        # --------------------------------------------------------

        source = self._first_string(
            item,
            [
                "source",
                "domain",
                "displayed_link",
            ],
            default="Unknown Source",
        )

        # --------------------------------------------------------
        # Snippet
        # --------------------------------------------------------

        snippet = self._first_string(
            item,
            [
                "snippet",
                "description",
                "text",
            ],
            default="",
        )

        # --------------------------------------------------------
        # ACTUAL IMAGE URL
        # --------------------------------------------------------

        image_url = self._first_url(
            item,
            [
                "image",
                "image_url",
                "original",
                "original_image",
                "image_link",
                "image_src",
            ],
        )

        # --------------------------------------------------------
        # Thumbnail
        # --------------------------------------------------------

        thumbnail = self._first_url(
            item,
            [
                "thumbnail",
                "thumbnail_url",
            ],
        )

        # --------------------------------------------------------
        # Some SerpApi Lens results store image information
        # inside nested structures.
        # --------------------------------------------------------

        if not image_url:

            image_url = self._extract_nested_image_url(
                item
            )

        if not thumbnail:

            thumbnail = self._extract_nested_thumbnail(
                item
            )

        return {
            "title": title,
            "url": url,
            "source": source,
            "snippet": snippet,

            # CRITICAL FIELD FOR FACEMATCHER
            "image_url": image_url,

            # Preview only
            "thumbnail": thumbnail,

            "result_type": (
                "External reverse-image search result"
            ),

            "search_result_type": result_type,

            # Preserve provider order.
            "provider_rank": provider_rank,

            "provider": self.PROVIDER_NAME,
        }

    # ============================================================
    # NESTED IMAGE EXTRACTION
    # ============================================================

    def _extract_nested_image_url(
        self,
        item: Dict[str, Any],
    ) -> Optional[str]:
        """Try common nested SerpApi image structures."""

        possible_objects = [
            item.get("image"),
            item.get("images"),
            item.get("visual_match"),
        ]

        for obj in possible_objects:

            if isinstance(
                obj,
                dict,
            ):

                result = self._first_url(
                    obj,
                    [
                        "url",
                        "link",
                        "image_url",
                        "original",
                        "src",
                    ],
                )

                if result:

                    return result

            elif isinstance(
                obj,
                list,
            ):

                for nested in obj:

                    if not isinstance(
                        nested,
                        dict,
                    ):
                        continue

                    result = self._first_url(
                        nested,
                        [
                            "url",
                            "link",
                            "image_url",
                            "original",
                            "src",
                        ],
                    )

                    if result:

                        return result

        return None

    def _extract_nested_thumbnail(
        self,
        item: Dict[str, Any],
    ) -> Optional[str]:
        """Try common nested thumbnail structures."""

        possible_objects = [
            item.get("thumbnail"),
            item.get("images"),
        ]

        for obj in possible_objects:

            if isinstance(
                obj,
                dict,
            ):

                result = self._first_url(
                    obj,
                    [
                        "url",
                        "link",
                        "src",
                    ],
                )

                if result:

                    return result

            elif isinstance(
                obj,
                list,
            ):

                for nested in obj:

                    if not isinstance(
                        nested,
                        dict,
                    ):
                        continue

                    result = self._first_url(
                        nested,
                        [
                            "url",
                            "link",
                            "src",
                        ],
                    )

                    if result:

                        return result

        return None

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _is_http_url(
        value: str,
    ) -> bool:

        return (
            value.startswith(
                "http://"
            )
            or value.startswith(
                "https://"
            )
        )

    @staticmethod
    def _first_string(
        item: Dict[str, Any],
        keys: List[str],
        default: str = "",
    ) -> str:
        """Return first non-empty string value."""

        for key in keys:

            value = item.get(
                key
            )

            if value is None:
                continue

            if isinstance(
                value,
                str,
            ):

                cleaned = value.strip()

                if cleaned:

                    return cleaned

            elif isinstance(
                value,
                (int, float),
            ):

                return str(
                    value
                )

        return default

    @classmethod
    def _first_url(
        cls,
        item: Dict[str, Any],
        keys: List[str],
    ) -> Optional[str]:
        """Return first valid HTTP/HTTPS URL."""

        for key in keys:

            value = item.get(
                key
            )

            if not isinstance(
                value,
                str,
            ):
                continue

            value = value.strip()

            if not value:
                continue

            if cls._is_http_url(
                value
            ):

                return value

        return None

    # ============================================================
    # ERROR RESPONSE
    # ============================================================

    def _error_response(
        self,
        query_image: str,
        error: str,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        return {
            "success": False,
            "query_image": query_image,
            "provider": self.PROVIDER_NAME,
            "image_id": image_id,
            "result_count": 0,
            "results": [],
            "error": error,
        }