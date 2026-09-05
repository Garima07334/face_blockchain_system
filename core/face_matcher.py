"""
Phase 10.5 / Phase 11: Candidate Face Matcher.

Verifies whether an external candidate image contains a face that
matches the query face embedding using OpenCV SFace cosine similarity.

Important:
- External web search is candidate discovery only.
- A candidate is accepted only after SFace verification.
- Blockchain registration must never happen without a valid face match.
- No face image or biometric embedding is stored on-chain.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import tempfile
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from core.face_detector import FaceDetector
from core.face_encoder import FaceEncoder


load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SFACE_COSINE_THRESHOLD = 0.363


class FaceMatcher:
    """Verify external candidate faces using OpenCV SFace."""

    STATUS_MATCH = "MATCH"
    STATUS_NO_MATCH = "NO_MATCH"
    STATUS_VERIFICATION_ERROR = "VERIFICATION_ERROR"

    def __init__(
        self,
        threshold: Optional[float] = None,
        detector: Optional[FaceDetector] = None,
        encoder: Optional[FaceEncoder] = None,
        timeout: int = 10,
        max_download_size_bytes: int = 5 * 1024 * 1024,
    ) -> None:

        # ================================================================
        # THRESHOLD
        # ================================================================

        if threshold is not None:
            self.threshold = float(threshold)

        else:
            env_value = os.getenv("FACE_MATCH_THRESHOLD")

            self.threshold = (
                float(env_value)
                if env_value
                else DEFAULT_SFACE_COSINE_THRESHOLD
            )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "Face match threshold must be between 0.0 and 1.0."
            )

        # ================================================================
        # MODELS
        # ================================================================

        self.detector = detector or FaceDetector()

        # More tolerant detector for small thumbnails.
        self.small_detector = FaceDetector(
            min_size=(15, 15)
        )

        self.encoder = encoder or FaceEncoder(
            detector=self.detector
        )

        self.small_encoder = FaceEncoder(
            detector=self.small_detector
        )

        self.timeout = timeout

        self.max_download_size_bytes = (
            max_download_size_bytes
        )

    # ====================================================================
    # COSINE SIMILARITY
    # ====================================================================

    @staticmethod
    def compute_similarity(
        embedding1: List[float],
        embedding2: List[float],
    ) -> Tuple[float, float]:
        """
        Calculate cosine similarity and cosine distance.

        Returns:
            similarity, distance
        """

        if not embedding1 or not embedding2:
            return 0.0, 1.0

        try:
            v1 = np.asarray(
                embedding1,
                dtype=np.float32,
            ).flatten()

            v2 = np.asarray(
                embedding2,
                dtype=np.float32,
            ).flatten()

        except Exception:
            return 0.0, 1.0

        if v1.size == 0 or v2.size == 0:
            return 0.0, 1.0

        if v1.size != v2.size:
            return 0.0, 1.0

        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0, 1.0

        similarity = float(
            np.dot(v1, v2) /
            (norm1 * norm2)
        )

        similarity = max(
            -1.0,
            min(1.0, similarity),
        )

        distance = max(
            0.0,
            1.0 - similarity,
        )

        return similarity, distance

    # ====================================================================
    # EMBEDDING COMPARISON
    # ====================================================================

    def compare_embeddings(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        candidate_boxes: Optional[
            List[Dict[str, int]]
        ] = None,
    ) -> Dict[str, Any]:
        """Compare query face against every candidate face."""

        if (
            not isinstance(query_embedding, list)
            or not query_embedding
        ):
            return self._build_match_error(
                "Query embedding is missing, empty, or invalid."
            )

        if not candidate_embeddings:
            return {
                "match": False,
                "status": self.STATUS_NO_MATCH,
                "similarity": None,
                "distance": None,
                "threshold": self.threshold,
                "candidate_face_count": 0,
                "encoded_candidate_face_count": 0,
                "best_candidate_face_index": None,
                "best_similarity": None,
                "best_distance": None,
                "candidate_faces": [],
                "all_face_similarities": [],
                "error": None,
            }

        candidate_faces_detail = []
        all_similarities = []

        best_similarity = -2.0
        best_distance = 2.0
        best_index = None

        for index, candidate_embedding in enumerate(
            candidate_embeddings
        ):

            if (
                not isinstance(
                    candidate_embedding,
                    list,
                )
                or not candidate_embedding
            ):
                continue

            similarity, distance = self.compute_similarity(
                query_embedding,
                candidate_embedding,
            )

            similarity_rounded = round(
                similarity,
                4,
            )

            distance_rounded = round(
                distance,
                4,
            )

            all_similarities.append(
                similarity_rounded
            )

            box = {}

            if (
                candidate_boxes
                and index < len(candidate_boxes)
            ):
                box = candidate_boxes[index] or {}

            candidate_faces_detail.append(
                {
                    "face_index": index,
                    "box": box,
                    "similarity": similarity_rounded,
                    "distance": distance_rounded,
                }
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_distance = distance
                best_index = index

        # Nothing could be evaluated.
        if best_index is None:
            return self._build_match_error(
                "Failed to evaluate candidate embeddings."
            )

        # ----------------------------------------------------------------
        # HARD MATCH CONDITION
        # ----------------------------------------------------------------

        matched = (
            best_index is not None
            and len(candidate_embeddings) > 0
            and best_similarity >= self.threshold
        )

        return {
            "match": bool(matched),

            "status": (
                self.STATUS_MATCH
                if matched
                else self.STATUS_NO_MATCH
            ),

            "similarity": round(
                best_similarity,
                4,
            ),

            "distance": round(
                best_distance,
                4,
            ),

            "threshold": self.threshold,

            "candidate_face_count": len(
                candidate_embeddings
            ),

            "encoded_candidate_face_count": len(
                candidate_embeddings
            ),

            "best_candidate_face_index": best_index,

            "best_similarity": round(
                best_similarity,
                4,
            ),

            "best_distance": round(
                best_distance,
                4,
            ),

            "candidate_faces": candidate_faces_detail,

            "all_face_similarities": all_similarities,

            "error": None,
        }

    # ====================================================================
    # DOWNLOAD IMAGE
    # ====================================================================

    def download_candidate_image(
        self,
        url: str,
    ) -> Dict[str, Any]:
        """Download and validate candidate image."""

        url_str = str(url).strip()

        try:
            parsed = urlparse(url_str)
        except Exception:
            parsed = None

        if (
            parsed is None
            or parsed.scheme.lower()
            not in {"http", "https"}
            or not parsed.netloc
        ):
            return {
                "success": False,
                "temp_path": None,
                "image_bytes": None,
                "image_size": {
                    "width": 0,
                    "height": 0,
                },
                "error": (
                    "Invalid candidate image URL scheme. "
                    "Only HTTP/HTTPS URLs are allowed."
                ),
            }

        try:

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/151.0 Safari/537.36"
                )
            }

            response = requests.get(
                url_str,
                headers=headers,
                stream=True,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "temp_path": None,
                    "image_bytes": None,
                    "image_size": {
                        "width": 0,
                        "height": 0,
                    },
                    "error": (
                        "Candidate image request failed "
                        f"with HTTP {response.status_code}."
                    ),
                }

            content_length = response.headers.get(
                "Content-Length"
            )

            if content_length:

                try:
                    if (
                        int(content_length)
                        > self.max_download_size_bytes
                    ):
                        return {
                            "success": False,
                            "temp_path": None,
                            "image_bytes": None,
                            "image_size": {
                                "width": 0,
                                "height": 0,
                            },
                            "error": (
                                "Candidate image exceeds "
                                "maximum allowed size."
                            ),
                        }

                except ValueError:
                    pass

            downloaded = bytearray()

            for chunk in response.iter_content(
                chunk_size=8192
            ):

                if not chunk:
                    continue

                downloaded.extend(chunk)

                if (
                    len(downloaded)
                    > self.max_download_size_bytes
                ):
                    return {
                        "success": False,
                        "temp_path": None,
                        "image_bytes": None,
                        "image_size": {
                            "width": 0,
                            "height": 0,
                        },
                        "error": (
                            "Downloaded candidate image "
                            "exceeded maximum allowed size."
                        ),
                    }

            image_bytes = bytes(downloaded)

            if not image_bytes:
                return {
                    "success": False,
                    "temp_path": None,
                    "image_bytes": None,
                    "image_size": {
                        "width": 0,
                        "height": 0,
                    },
                    "error": (
                        "Downloaded candidate image is empty."
                    ),
                }

            image_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR,
            )

            if image is None or image.size == 0:
                return {
                    "success": False,
                    "temp_path": None,
                    "image_bytes": None,
                    "image_size": {
                        "width": 0,
                        "height": 0,
                    },
                    "error": (
                        "Downloaded data is not a "
                        "valid decodable image."
                    ),
                }

            height, width = image.shape[:2]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg",
            ) as temp_file:

                temp_file.write(image_bytes)

                temp_path = Path(
                    temp_file.name
                )

            return {
                "success": True,
                "temp_path": temp_path,
                "image_bytes": image_bytes,
                "image_size": {
                    "width": int(width),
                    "height": int(height),
                },
                "error": None,
            }

        except requests.exceptions.Timeout:

            return {
                "success": False,
                "temp_path": None,
                "image_bytes": None,
                "image_size": {
                    "width": 0,
                    "height": 0,
                },
                "error": (
                    "Candidate image download timed out."
                ),
            }

        except requests.exceptions.RequestException as err:

            return {
                "success": False,
                "temp_path": None,
                "image_bytes": None,
                "image_size": {
                    "width": 0,
                    "height": 0,
                },
                "error": (
                    "Candidate image network error: "
                    f"{err}"
                ),
            }

        except Exception as err:

            return {
                "success": False,
                "temp_path": None,
                "image_bytes": None,
                "image_size": {
                    "width": 0,
                    "height": 0,
                },
                "error": (
                    "Unexpected candidate image "
                    f"download error: {err}"
                ),
            }

    # ====================================================================
    # VERIFY CANDIDATE IMAGE
    # ====================================================================

    def verify_candidate_image(
        self,
        query_embedding: List[float],
        image_source: Union[
            str,
            Path,
            bytes,
        ],
        candidate_url: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect and compare all faces in candidate image."""

        temporary_path = None
        target_path = None

        image_size = {
            "width": 0,
            "height": 0,
        }

        # ================================================================
        # BYTES
        # ================================================================

        if isinstance(
            image_source,
            bytes,
        ):

            image_array = np.frombuffer(
                image_source,
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR,
            )

            if image is None or image.size == 0:
                return self._build_match_error(
                    "Failed to decode provided candidate image bytes.",
                    candidate_url,
                    thumbnail_url,
                )

            image_size = {
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            }

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg",
            ) as temp_file:

                temp_file.write(image_source)

                target_path = Path(
                    temp_file.name
                )

                temporary_path = target_path

        # ================================================================
        # PATH
        # ================================================================

        elif isinstance(
            image_source,
            Path,
        ):

            if not image_source.is_file():
                return self._build_match_error(
                    (
                        "Local candidate image file "
                        f"not found: '{image_source}'"
                    ),
                    candidate_url,
                    thumbnail_url,
                )

            target_path = image_source

            image = cv2.imread(
                str(target_path)
            )

            if image is None:
                return self._build_match_error(
                    "OpenCV could not read local candidate image.",
                    candidate_url,
                    thumbnail_url,
                )

            image_size = {
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            }

        # ================================================================
        # STRING
        # ================================================================

        elif isinstance(
            image_source,
            str,
        ):

            source = image_source.strip()

            parsed = urlparse(source)

            if (
                parsed.scheme.lower()
                in {"http", "https"}
                and parsed.netloc
            ):

                download = self.download_candidate_image(
                    source
                )

                if not download["success"]:
                    return self._build_match_error(
                        download["error"],
                        candidate_url or source,
                        thumbnail_url,
                    )

                target_path = download["temp_path"]
                temporary_path = target_path
                image_size = download["image_size"]

            elif parsed.scheme:

                return self._build_match_error(
                    (
                        "Invalid candidate image URL scheme. "
                        "Only HTTP/HTTPS URLs are allowed."
                    ),
                    candidate_url or source,
                    thumbnail_url,
                )

            else:

                local_path = Path(source)

                if not local_path.is_file():
                    return self._build_match_error(
                        (
                            "Local candidate image "
                            f"file not found: '{source}'"
                        ),
                        candidate_url,
                        thumbnail_url,
                    )

                target_path = local_path

                image = cv2.imread(
                    str(target_path)
                )

                if image is None:
                    return self._build_match_error(
                        "OpenCV could not read local candidate image.",
                        candidate_url,
                        thumbnail_url,
                    )

                image_size = {
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }

        else:

            return self._build_match_error(
                "Unsupported candidate image source type.",
                candidate_url,
                thumbnail_url,
            )

        # ================================================================
        # PROCESS
        # ================================================================

        try:

            if target_path is None:
                return self._build_match_error(
                    "Candidate image path could not be resolved.",
                    candidate_url,
                    thumbnail_url,
                    image_size,
                )

            # Small thumbnails need more tolerant detection.
            is_small_image = (
                image_size["width"] > 0
                and (
                    image_size["width"] < 250
                    or image_size["height"] < 250
                )
            )

            active_detector = (
                self.small_detector
                if is_small_image
                else self.detector
            )

            active_encoder = (
                self.small_encoder
                if is_small_image
                else self.encoder
            )

            # ============================================================
            # FACE DETECTION
            # ============================================================

            detection = active_detector.detect_faces(
                target_path
            )

            if not detection.get(
                "success",
                False,
            ):
                return self._build_match_error(
                    (
                        "Candidate face detection failed: "
                        f"{detection.get('error')}"
                    ),
                    candidate_url,
                    thumbnail_url,
                    image_size,
                )

            face_count = int(
                detection.get(
                    "face_count",
                    0,
                )
            )

            detected_faces = detection.get(
                "faces",
                [],
            )

            # ============================================================
            # NO FACE
            # ============================================================

            if face_count == 0:
                return {
                    "match": False,
                    "status": self.STATUS_NO_MATCH,
                    "similarity": None,
                    "distance": None,
                    "threshold": self.threshold,
                    "candidate_url": (
                        candidate_url
                        or str(image_source)
                    ),
                    "thumbnail_url": thumbnail_url,
                    "candidate_image_size": image_size,
                    "candidate_face_count": 0,
                    "encoded_candidate_face_count": 0,
                    "best_candidate_face_index": None,
                    "best_similarity": None,
                    "best_distance": None,
                    "candidate_faces": [],
                    "all_face_similarities": [],
                    "error": None,
                }

            # ============================================================
            # ENCODE FACES
            # ============================================================

            candidate_embeddings = []
            candidate_boxes = []

            for face_index in range(face_count):

                encoding = active_encoder.encode_face(
                    target_path,
                    face_index=face_index,
                )

                if not encoding.get(
                    "success",
                    False,
                ):
                    continue

                embedding = encoding.get(
                    "embedding"
                )

                if not embedding:
                    continue

                candidate_embeddings.append(
                    embedding
                )

                encoded_box = encoding.get(
                    "box"
                )

                if encoded_box:
                    candidate_boxes.append(
                        encoded_box
                    )

                elif face_index < len(
                    detected_faces
                ):
                    candidate_boxes.append(
                        detected_faces[
                            face_index
                        ].get(
                            "box",
                            {},
                        )
                    )

                else:
                    candidate_boxes.append({})

            # ============================================================
            # NO ENCODINGS
            # ============================================================

            if not candidate_embeddings:
                return self._build_match_error(
                    (
                        "Failed to extract SFace "
                        "feature embeddings from candidate face(s)."
                    ),
                    candidate_url,
                    thumbnail_url,
                    image_size,
                )

            # ============================================================
            # COMPARE
            # ============================================================

            result = self.compare_embeddings(
                query_embedding,
                candidate_embeddings,
                candidate_boxes=candidate_boxes,
            )

            result["candidate_url"] = (
                candidate_url
                or str(image_source)
            )

            result["thumbnail_url"] = thumbnail_url

            result["candidate_image_size"] = image_size

            # IMPORTANT:
            # candidate_face_count means detected faces,
            # while encoded_candidate_face_count means successfully
            # encoded faces.
            result["candidate_face_count"] = face_count

            result["encoded_candidate_face_count"] = len(
                candidate_embeddings
            )

            return result

        finally:

            if (
                temporary_path
                and temporary_path.is_file()
            ):

                try:
                    temporary_path.unlink()
                except Exception:
                    pass

    # ====================================================================
    # ERROR BUILDER
    # ====================================================================

    def _build_match_error(
        self,
        err_msg: str,
        candidate_url: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
        image_size: Optional[
            Dict[str, int]
        ] = None,
    ) -> Dict[str, Any]:

        return {
            "match": False,
            "status": self.STATUS_VERIFICATION_ERROR,
            "similarity": None,
            "distance": None,
            "threshold": self.threshold,
            "candidate_url": candidate_url,
            "thumbnail_url": thumbnail_url,
            "candidate_image_size": (
                image_size
                or {
                    "width": 0,
                    "height": 0,
                }
            ),
            "candidate_face_count": 0,
            "encoded_candidate_face_count": 0,
            "best_candidate_face_index": None,
            "best_similarity": None,
            "best_distance": None,
            "candidate_faces": [],
            "all_face_similarities": [],
            "error": str(err_msg),
        }