"""
Phase 4: External Search Result Normalizer.

Responsibilities:
- Validate search results.
- Canonicalize webpage/post URLs.
- Deduplicate results.
- Preserve actual candidate image URLs.
- Preserve thumbnails.
- Preserve provider ranking.
- Never interpret metadata completeness as face similarity.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlparse,
    urlunparse,
)

import logging


logger = logging.getLogger(__name__)


TRACKING_PARAMS: Set[str] = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "fbclid",
    "gclid",
}


def canonicalize_url(
    url_str: str,
) -> Optional[str]:
    """Normalize a valid HTTP/HTTPS webpage URL."""

    if not isinstance(url_str, str):
        return None

    cleaned = url_str.strip()

    if not cleaned:
        return None

    try:
        parsed = urlparse(cleaned)
    except Exception:
        return None

    scheme = parsed.scheme.lower()

    if scheme not in {
        "http",
        "https",
    }:
        return None

    netloc = parsed.netloc.lower()

    if not netloc:
        return None

    # Remove standard ports.
    if (
        scheme == "http"
        and netloc.endswith(":80")
    ):
        netloc = netloc[:-3]

    elif (
        scheme == "https"
        and netloc.endswith(":443")
    ):
        netloc = netloc[:-4]

    path = parsed.path

    if (
        len(path) > 1
        and path.endswith("/")
    ):
        path = path[:-1]

    query_pairs = parse_qsl(
        parsed.query,
        keep_blank_values=True,
    )

    filtered_query = [
        (key, value)
        for key, value in query_pairs
        if key.lower()
        not in TRACKING_PARAMS
    ]

    new_query = urlencode(
        filtered_query
    )

    return urlunparse(
        (
            scheme,
            netloc,
            path,
            parsed.params,
            new_query,
            "",
        )
    )


class ResultSelector:
    """Normalize external reverse-image search results."""

    DEFAULT_RESULT_TYPE = (
        "External visual-search candidate"
    )

    def select_results(
        self,
        search_response: Any,
        max_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Normalize and deduplicate external search results.

        IMPORTANT:
        max_results=None returns ALL valid unique candidates.

        The pipeline should perform biometric verification after
        normalization rather than treating metadata ranking as
        face-match ranking.
        """

        # ================================================================
        # VALIDATE RESPONSE
        # ================================================================

        if not isinstance(
            search_response,
            dict,
        ):
            return {
                "success": False,
                "selected_count": 0,
                "selected_results": [],
                "error": (
                    "Invalid search response format: "
                    "expected a dictionary."
                ),
            }

        if not search_response.get(
            "success",
            False,
        ):
            return {
                "success": False,
                "selected_count": 0,
                "selected_results": [],
                "error": str(
                    search_response.get(
                        "error"
                    )
                    or "External web search failed."
                ),
            }

        raw_results = search_response.get(
            "results"
        )

        if not isinstance(
            raw_results,
            list,
        ):
            return {
                "success": False,
                "selected_count": 0,
                "selected_results": [],
                "error": (
                    "Invalid search response: "
                    "'results' must be a list."
                ),
            }

        provider = str(
            search_response.get(
                "provider",
                "SerpApi (Google Lens)",
            )
        )

        if not raw_results:
            return {
                "success": True,
                "selected_count": 0,
                "selected_results": [],
                "error": None,
            }

        # ================================================================
        # PROCESS
        # ================================================================

        seen_urls: Set[str] = set()

        candidates: List[
            Tuple[float, int, Dict[str, Any]]
        ] = []

        for original_index, raw_item in enumerate(
            raw_results
        ):

            if not isinstance(
                raw_item,
                dict,
            ):
                continue

            # ------------------------------------------------------------
            # WEBPAGE / POST URL
            # ------------------------------------------------------------

            raw_url = raw_item.get(
                "url"
            )

            if not isinstance(
                raw_url,
                str,
            ):
                continue

            canonical_url = canonicalize_url(
                raw_url
            )

            if not canonical_url:
                continue

            if canonical_url in seen_urls:
                continue

            seen_urls.add(
                canonical_url
            )

            # ------------------------------------------------------------
            # METADATA
            # ------------------------------------------------------------

            title = self._clean_optional_string(
                raw_item.get("title")
            )

            source = self._clean_optional_string(
                raw_item.get("source")
            )

            snippet = self._clean_optional_string(
                raw_item.get("snippet")
            )

            # ------------------------------------------------------------
            # IMAGE URL
            # ------------------------------------------------------------

            image_url = self._clean_optional_url(
                raw_item.get("image_url")
            )

            thumbnail = self._clean_optional_url(
                raw_item.get("thumbnail")
            )

            search_result_type = (
                self._clean_optional_string(
                    raw_item.get(
                        "search_result_type"
                    )
                )
            )

            provider_rank = raw_item.get(
                "provider_rank"
            )

            if not isinstance(
                provider_rank,
                int,
            ):
                provider_rank = None

            # ------------------------------------------------------------
            # COMPLETENESS
            # ------------------------------------------------------------

            completeness = (
                self._calculate_completeness_score(
                    title=title,
                    source=source,
                    snippet=snippet,
                    thumbnail=thumbnail,
                )
            )

            candidate = {
                "title": title,

                # Webpage / social post.
                "url": canonical_url,

                "source": source,

                "snippet": snippet,

                # Actual image for biometric verification.
                "image_url": image_url,

                # Preview fallback.
                "thumbnail": thumbnail,

                "result_type": (
                    self.DEFAULT_RESULT_TYPE
                ),

                "search_result_type": (
                    search_result_type
                ),

                "provider_rank": provider_rank,

                "provider": provider,
            }

            candidates.append(
                (
                    completeness,
                    original_index,
                    candidate,
                )
            )

        # ================================================================
        # NO VALID RESULTS
        # ================================================================

        if not candidates:
            return {
                "success": True,
                "selected_count": 0,
                "selected_results": [],
                "error": None,
            }

        # ================================================================
        # METADATA SORT
        # ================================================================

        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        # ================================================================
        # OPTIONAL LIMIT
        # ================================================================

        if (
            max_results is not None
            and max_results > 0
        ):
            candidates = candidates[
                :max_results
            ]

        # ================================================================
        # BUILD OUTPUT
        # ================================================================

        selected_results = []

        for rank, (
            completeness,
            original_index,
            candidate,
        ) in enumerate(
            candidates,
            start=1,
        ):

            candidate[
                "selection_rank"
            ] = rank

            candidate[
                "metadata_completeness_score"
            ] = completeness

            candidate[
                "original_search_rank"
            ] = original_index + 1

            selected_results.append(
                candidate
            )

        return {
            "success": True,
            "selected_count": len(
                selected_results
            ),
            "selected_results": selected_results,
            "error": None,
        }

    # ====================================================================
    # HELPERS
    # ====================================================================

    @staticmethod
    def _clean_optional_string(
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        if not isinstance(
            value,
            str,
        ):
            value = str(value)

        value = value.strip()

        return value or None

    @staticmethod
    def _clean_optional_url(
        value: Any,
    ) -> Optional[str]:

        if not isinstance(
            value,
            str,
        ):
            return None

        value = value.strip()

        if (
            value.startswith("http://")
            or value.startswith("https://")
        ):
            return value

        return None

    @staticmethod
    def _calculate_completeness_score(
        title: Optional[str],
        source: Optional[str],
        snippet: Optional[str],
        thumbnail: Optional[str],
    ) -> float:

        score = 10.0

        if title:
            score += 5.0

        if source:
            score += 3.0

        if snippet:
            score += 2.0

        if thumbnail:
            score += 2.0

        return score


def select_results(
    search_response: Any,
    max_results: Optional[int] = None,
) -> Dict[str, Any]:

    selector = ResultSelector()

    return selector.select_results(
        search_response,
        max_results=max_results,
    )