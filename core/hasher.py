"""Phase 5: Canonical Data Construction & SHA-256 Fingerprinting.

Converts selected Phase 4 external visual-search candidates into a deterministic
canonical JSON payload and generates a cryptographic SHA-256 fingerprint for
downstream blockchain registration.
"""

import hashlib
import json
from typing import Any, Dict, Optional


class CandidateHasher:
    """Hasher for generating deterministic SHA-256 fingerprints from candidate metadata."""

    HASH_ALGORITHM = "SHA-256"
    DEFAULT_PROVIDER = "SerpApi (Google Lens)"
    DEFAULT_RESULT_TYPE = "External visual-search candidate"

    CANONICAL_FIELDS = (
        "provider",
        "result_type",
        "snippet",
        "source",
        "thumbnail",
        "title",
        "url",
    )

    def generate_candidate_hash(self, candidate: Any) -> Dict[str, Any]:
        """Construct deterministic canonical representation and compute SHA-256 fingerprint.

        Args:
            candidate: Selected Phase 4 candidate result dictionary.

        Returns:
            Structured dictionary:
            {
                "success": bool,
                "hash_algorithm": "SHA-256",
                "canonical_data": Dict[str, str] | None,
                "canonical_json": str | None,
                "sha256": str | None,
                "error": str | None
            }
        """
        # 1. Validate input
        if not isinstance(candidate, dict):
            return {
                "success": False,
                "hash_algorithm": self.HASH_ALGORITHM,
                "canonical_data": None,
                "canonical_json": None,
                "sha256": None,
                "error": "Invalid candidate payload: expected a dictionary.",
            }

        url = candidate.get("url")
        if not url or not isinstance(url, str) or not url.strip():
            return {
                "success": False,
                "hash_algorithm": self.HASH_ALGORITHM,
                "canonical_data": None,
                "canonical_json": None,
                "sha256": None,
                "error": "Missing or invalid 'url' in candidate dictionary.",
            }

        # 2. Extract and normalize canonical fields
        provider = self._clean_string(candidate.get("provider")) or self.DEFAULT_PROVIDER
        result_type = self._clean_string(candidate.get("result_type")) or self.DEFAULT_RESULT_TYPE
        snippet = self._clean_string(candidate.get("snippet"))
        source = self._clean_string(candidate.get("source"))
        thumbnail = self._clean_string(candidate.get("thumbnail"))
        title = self._clean_string(candidate.get("title"))
        clean_url = url.strip()

        # Explicitly construct canonical data dictionary
        # Transient fields (selection_rank, score, timestamp, api_key, face embeddings/images) are NOT included.
        canonical_data = {
            "provider": provider,
            "result_type": result_type,
            "snippet": snippet,
            "source": source,
            "thumbnail": thumbnail,
            "title": title,
            "url": clean_url,
        }

        # 3. Deterministic canonical JSON serialization
        try:
            canonical_json = json.dumps(
                canonical_data,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except Exception as err:
            return {
                "success": False,
                "hash_algorithm": self.HASH_ALGORITHM,
                "canonical_data": None,
                "canonical_json": None,
                "sha256": None,
                "error": f"Failed to serialize canonical data: {str(err)}",
            }

        # 4. Generate SHA-256 hex digest
        sha256_hex = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest().lower()

        return {
            "success": True,
            "hash_algorithm": self.HASH_ALGORITHM,
            "canonical_data": canonical_data,
            "canonical_json": canonical_json,
            "sha256": sha256_hex,
            "error": None,
        }

    @staticmethod
    def _clean_string(val: Any) -> str:
        """Normalize value to a clean string, mapping None/empty values to ""."""
        if val is None:
            return ""
        if not isinstance(val, str):
            val = str(val)
        return val.strip()


def generate_candidate_hash(candidate: Any) -> Dict[str, Any]:
    """Convenience function for CandidateHasher.generate_candidate_hash."""
    hasher = CandidateHasher()
    return hasher.generate_candidate_hash(candidate)


# Backwards compatibility helper if needed
class Hasher(CandidateHasher):
    """Alias for CandidateHasher."""

    pass
