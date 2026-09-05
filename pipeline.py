"""
End-to-End Face Identification & Blockchain Verification Pipeline.

Pipeline:

Face Image
    ↓
Face Detection
    ↓
SFace Face Encoding
    ↓
Google Lens Reverse Image Search
    ↓
External Candidate Normalization
    ↓
Candidate Face Detection + SFace Comparison
    ↓
Best Biometric Candidate
    ↓
MATCH / NO_MATCH / VERIFICATION_ERROR
    ↓
[ONLY IF MATCH]
    ↓
Canonical Metadata
    ↓
SHA-256
    ↓
Blockchain Registration
    ↓
On-Chain Verification
    ↓
Tamper Detection

Security rule:
Blockchain registration is NEVER performed unless a valid
candidate face match has been confirmed.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging

from core.face_detector import FaceDetector
from core.face_encoder import FaceEncoder
from core.web_search import WebSearch
from core.result_selector import ResultSelector
from core.face_matcher import FaceMatcher
from core.hasher import CandidateHasher
from core.blockchain import BlockchainClient
from core.tamper_detector import TamperDetector


logger = logging.getLogger(__name__)


class VerificationPipeline:

    def __init__(self, max_candidates: Optional[int] = None) -> None:
        """
        Initialize all pipeline components.

        max_candidates:
            None = evaluate every normalized candidate.
            Positive integer = evaluate only that many candidates.
        """

        self.detector = FaceDetector()

        self.encoder = FaceEncoder(
            detector=self.detector
        )

        self.web_search = WebSearch()
        self.result_selector = ResultSelector()
        self.face_matcher = FaceMatcher()

        # Your Phase 5 class
        self.hasher = CandidateHasher()

        self.blockchain = BlockchainClient()

        self.tamper_detector = TamperDetector(
            blockchain_client=self.blockchain
        )

        self.max_candidates = max_candidates

    # ------------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        image_path: Path,
        face_index: int = 0,
    ) -> Dict[str, Any]:

        pipeline_result = {
            "success": False,
            "message": None,
            "pipeline": {},
            "original_hash": None,
            "tampered_hash": None,
            "original_verification": None,
            "tampered_verification": None,
        }

        # ==============================================================
        # PHASE 1 — FACE DETECTION
        # ==============================================================

        image_path = Path(image_path)

        if not image_path.is_file():
            pipeline_result["message"] = (
                f"Input image not found: {image_path}"
            )
            return pipeline_result

        detection = self.detector.detect_faces(image_path)

        pipeline_result["pipeline"]["face_detection"] = detection

        if not detection.get("success", False):
            pipeline_result["message"] = (
                "Face detection failed."
            )
            return pipeline_result

        detected_faces = detection.get("faces", [])

        face_count = int(
            detection.get("face_count", len(detected_faces))
        )

        if face_count == 0:
            pipeline_result["message"] = (
                "No face detected in input image."
            )
            return pipeline_result

        if face_index < 0 or face_index >= face_count:
            pipeline_result["message"] = (
                f"Invalid face index {face_index}. "
                f"Detected {face_count} face(s)."
            )
            return pipeline_result

        pipeline_result["pipeline"]["face_detection"][
            "selected_face_index"
        ] = face_index

        # ==============================================================
        # PHASE 2 — SFace FACE ENCODING
        # ==============================================================

        encoding = self.encoder.encode_face(
            image_path,
            face_index=face_index,
        )

        pipeline_result["pipeline"]["face_encoding"] = encoding

        if not encoding.get("success", False):
            pipeline_result["message"] = (
                "Query face encoding failed."
            )
            return pipeline_result

        query_embedding = encoding.get("embedding")

        if not isinstance(query_embedding, list) or not query_embedding:
            pipeline_result["message"] = (
                "Query SFace embedding is empty."
            )
            return pipeline_result

        pipeline_result["pipeline"]["face_encoding"][
            "embedding_dim"
        ] = len(query_embedding)

        # ==============================================================
        # PHASE 3 — EXTERNAL REVERSE IMAGE SEARCH
        # ==============================================================

        search_response = self.web_search.search_by_image(
            image_path
        )

        pipeline_result["pipeline"]["web_search"] = search_response

        if not search_response.get("success", False):
            pipeline_result["message"] = (
                "External reverse-image search failed."
            )
            return pipeline_result

        raw_results = search_response.get(
            "results",
            []
        )

        raw_result_count = len(raw_results)

        pipeline_result["pipeline"]["web_search"][
            "result_count"
        ] = raw_result_count

        if raw_result_count == 0:
            pipeline_result["message"] = (
                "External search returned no candidates."
            )
            return pipeline_result

        # ==============================================================
        # PHASE 4 — NORMALIZE ALL SEARCH RESULTS
        # ==============================================================

        # IMPORTANT:
        # Do NOT truncate before biometric verification.
        #
        # Google Lens might return:
        #
        # 69 candidates
        #       ↓
        # genuine match could be #47
        #
        # Therefore normalize all candidates first.

        selection = self.result_selector.select_results(
            search_response,
            max_results=None,
        )

        pipeline_result["pipeline"]["result_selection"] = selection

        if not selection.get("success", False):
            pipeline_result["message"] = (
                "Search result normalization failed."
            )
            return pipeline_result

        all_candidates = selection.get(
            "selected_results",
            []
        )

        if not all_candidates:
            pipeline_result["message"] = (
                "No valid external candidates remained "
                "after URL normalization."
            )
            return pipeline_result

        # Optional performance limit.
        if (
            self.max_candidates is not None
            and self.max_candidates > 0
        ):
            candidates_to_evaluate = (
                all_candidates[:self.max_candidates]
            )
        else:
            candidates_to_evaluate = all_candidates

        pipeline_result["pipeline"]["result_selection"][
            "total_normalized_candidates"
        ] = len(all_candidates)

        pipeline_result["pipeline"]["result_selection"][
            "evaluation_candidate_count"
        ] = len(candidates_to_evaluate)

        # ==============================================================
        # PHASE 5 — BIOMETRIC CANDIDATE VERIFICATION
        # ==============================================================

        candidate_evaluations = []

        technical_errors = []

        usable_matches = []

        for evaluation_index, candidate in enumerate(
            candidates_to_evaluate,
            start=1,
        ):

            candidate_page_url = candidate.get(
                "url"
            )

            image_url = candidate.get(
                "image_url"
            )

            thumbnail_url = candidate.get(
                "thumbnail"
            )

            # ----------------------------------------------------------
            # Candidate has no image to analyze
            # ----------------------------------------------------------

            if not image_url and not thumbnail_url:

                verification = {
                    "match": False,
                    "status": FaceMatcher.STATUS_VERIFICATION_ERROR,
                    "similarity": None,
                    "distance": None,
                    "threshold": self.face_matcher.threshold,
                    "candidate_face_count": 0,
                    "encoded_candidate_face_count": 0,
                    "best_candidate_face_index": None,
                    "error": (
                        "Candidate has no usable image URL "
                        "or thumbnail."
                    ),
                }

                evaluation = {
                    "evaluation_index": evaluation_index,
                    "candidate": candidate,
                    "verification": verification,
                    "image_source": None,
                }

                candidate_evaluations.append(
                    evaluation
                )

                technical_errors.append(
                    evaluation
                )

                continue

            verification = None
            image_source_used = None

            # ----------------------------------------------------------
            # Try original candidate image
            # ----------------------------------------------------------

            if image_url:

                verification = (
                    self.face_matcher.verify_candidate_image(
                        query_embedding=query_embedding,
                        image_source=image_url,
                        candidate_url=candidate_page_url,
                        thumbnail_url=thumbnail_url,
                    )
                )

                image_source_used = "image_url"

            # ----------------------------------------------------------
            # Fallback to thumbnail
            # ----------------------------------------------------------

            should_try_thumbnail = (
                bool(thumbnail_url)
                and thumbnail_url != image_url
                and (
                    verification is None
                    or verification.get("status")
                    in {
                        FaceMatcher.STATUS_VERIFICATION_ERROR,
                        FaceMatcher.STATUS_NO_MATCH,
                    }
                )
            )

            if should_try_thumbnail:

                thumbnail_result = (
                    self.face_matcher.verify_candidate_image(
                        query_embedding=query_embedding,
                        image_source=thumbnail_url,
                        candidate_url=candidate_page_url,
                        thumbnail_url=thumbnail_url,
                    )
                )

                current_similarity = (
                    verification.get("similarity")
                    if verification
                    else None
                )

                thumbnail_similarity = (
                    thumbnail_result.get(
                        "similarity"
                    )
                )

                # Thumbnail produced a genuine match
                if thumbnail_result.get("match"):

                    verification = thumbnail_result

                    image_source_used = "thumbnail"

                # Original failed technically
                elif (
                    verification is None
                    or verification.get("status")
                    == FaceMatcher.STATUS_VERIFICATION_ERROR
                ):

                    verification = thumbnail_result

                    image_source_used = "thumbnail"

                # Thumbnail has a better similarity
                elif (
                    thumbnail_similarity is not None
                    and (
                        current_similarity is None
                        or thumbnail_similarity
                        > current_similarity
                    )
                ):

                    verification = thumbnail_result

                    image_source_used = "thumbnail"

            # ----------------------------------------------------------
            # Store evaluation
            # ----------------------------------------------------------

            evaluation = {
                "evaluation_index": evaluation_index,
                "candidate": candidate,
                "verification": verification,
                "image_source": image_source_used,
            }

            candidate_evaluations.append(
                evaluation
            )

            # ----------------------------------------------------------
            # Technical error
            # ----------------------------------------------------------

            if (
                verification.get("status")
                == FaceMatcher.STATUS_VERIFICATION_ERROR
            ):

                technical_errors.append(
                    evaluation
                )

                continue

            # ----------------------------------------------------------
            # Usable biometric result
            # ----------------------------------------------------------

            similarity = verification.get(
                "similarity"
            )

            if similarity is not None:

                usable_matches.append(
                    evaluation
                )

        # Save every candidate evaluation
        pipeline_result["pipeline"][
            "candidate_evaluations"
        ] = candidate_evaluations

        # ==============================================================
        # NO USABLE CANDIDATE
        # ==============================================================

        if not usable_matches:

            pipeline_result["pipeline"][
                "candidate_face_match"
            ] = {

                "match": False,

                "status":
                    FaceMatcher.STATUS_VERIFICATION_ERROR,

                "similarity": None,

                "distance": None,

                "threshold":
                    self.face_matcher.threshold,

                "candidate_face_count": 0,

                "encoded_candidate_face_count": 0,

                "best_candidate_face_index": None,

                "evaluated_candidates_count":
                    len(candidate_evaluations),

                "technical_error_count":
                    len(technical_errors),

                "selected_candidate": None,

                "error":
                    "No candidate produced a usable "
                    "biometric similarity score.",
            }

            pipeline_result["message"] = (
                "Candidate face verification could not "
                "produce a usable result."
            )

            return pipeline_result

        # ==============================================================
        # SELECT BEST BIOMETRIC CANDIDATE
        # ==============================================================

        # IMPORTANT:
        #
        # We select based on SFace similarity.
        #
        # NOT:
        # metadata score
        # Google rank
        # title
        # source
        #
        # This fixes the earlier bug where a visually unrelated
        # candidate could be selected simply because its metadata
        # was more complete.

        best_evaluation = max(
            usable_matches,
            key=lambda item:
                item["verification"].get(
                    "similarity",
                    -1.0,
                ),
        )

        best_candidate = (
            best_evaluation["candidate"]
        )

        best_verification = (
            best_evaluation["verification"]
        )

        best_similarity = (
            best_verification.get(
                "similarity"
            )
        )

        threshold = (
            best_verification.get(
                "threshold",
                self.face_matcher.threshold,
            )
        )

        # ==============================================================
        # HARD FACE-MATCH SECURITY GATE
        # ==============================================================

        face_match_confirmed = (

            best_verification.get(
                "match"
            ) is True

            and best_verification.get(
                "status"
            ) == FaceMatcher.STATUS_MATCH

            and best_verification.get(
                "candidate_face_count",
                0,
            ) > 0

            and best_verification.get(
                "best_candidate_face_index"
            ) is not None

            and best_similarity is not None

            and threshold is not None

            and best_similarity >= threshold
        )

        candidate_face_match = {

            **best_verification,

            # Recalculate this independently.
            "match":
                bool(face_match_confirmed),

            "status":
                (
                    FaceMatcher.STATUS_MATCH
                    if face_match_confirmed
                    else FaceMatcher.STATUS_NO_MATCH
                ),

            "selected_candidate":
                best_candidate,

            "selected_candidate_evaluation_index":
                best_evaluation[
                    "evaluation_index"
                ],

            "selected_candidate_image_source":
                best_evaluation[
                    "image_source"
                ],

            "evaluated_candidates_count":
                len(candidate_evaluations),

            "usable_candidate_count":
                len(usable_matches),

            "technical_error_count":
                len(technical_errors),

            "best_similarity":
                best_similarity,

            "best_distance":
                best_verification.get(
                    "distance"
                ),
        }

        pipeline_result["pipeline"][
            "candidate_face_match"
        ] = candidate_face_match

        # ==============================================================
        # BLOCKCHAIN GATE
        # ==============================================================

        if not face_match_confirmed:

            pipeline_result["message"] = (
                "NO SAME-FACE MATCH CONFIRMED. "
                "Blockchain notarization was blocked."
            )

            pipeline_result["pipeline"][
                "blockchain_gate"
            ] = {

                "allowed": False,

                "reason":
                    (
                        "Best SFace similarity "
                        "did not meet the configured "
                        "face-match threshold."
                    ),
            }

            return pipeline_result

        # Face match passed.
        pipeline_result["pipeline"][
            "blockchain_gate"
        ] = {

            "allowed": True,

            "reason":
                "Valid SFace face match confirmed.",
        }

        # ==============================================================
        # PHASE 6 — SHA-256 HASHING
        # ==============================================================

        # IMPORTANT:
        # Hash EXACTLY the candidate that passed the face match.
        #
        # This guarantees:
        #
        # face match candidate
        #       ==
        # hashed candidate
        #       ==
        # blockchain candidate

        hash_result = (
            self.hasher.generate_candidate_hash(
                best_candidate
            )
        )

        pipeline_result["pipeline"][
            "hashing"
        ] = hash_result

        if not hash_result.get(
            "success",
            False,
        ):

            pipeline_result["message"] = (
                "SHA-256 hashing failed."
            )

            return pipeline_result

        original_hash = hash_result.get(
            "sha256"
        )

        if not original_hash:

            pipeline_result["message"] = (
                "SHA-256 hash was not generated."
            )

            return pipeline_result

        pipeline_result["original_hash"] = (
            original_hash
        )

        # ==============================================================
        # PHASE 7 — BLOCKCHAIN CONNECTION
        # ==============================================================

        if not self.blockchain.is_connected():

            pipeline_result["message"] = (
                "Blockchain RPC is unavailable. "
                "Metadata was NOT notarized."
            )

            pipeline_result["pipeline"][
                "blockchain_gate"
            ]["rpc_available"] = False

            return pipeline_result

        pipeline_result["pipeline"][
            "blockchain_gate"
        ]["rpc_available"] = True

        # ==============================================================
        # PHASE 8 — BLOCKCHAIN REGISTRATION
        # ==============================================================

        registration = (
            self.blockchain.register_hash(
                original_hash
            )
        )

        pipeline_result["pipeline"][
            "blockchain_registration"
        ] = registration

        if not registration.get(
            "success",
            False,
        ):

            pipeline_result["message"] = (
                "Blockchain registration failed."
            )

            return pipeline_result

        # ==============================================================
        # PHASE 9 — ON-CHAIN VERIFICATION
        # ==============================================================

        blockchain_verification = (
            self.blockchain.verify_hash(
                original_hash
            )
        )

        pipeline_result["pipeline"][
            "blockchain_verification"
        ] = blockchain_verification

        if not blockchain_verification.get(
            "verified",
            False,
        ):

            pipeline_result["message"] = (
                "Hash was registered but could not "
                "be verified on-chain."
            )

            return pipeline_result

        # ==============================================================
        # PHASE 10 — TAMPER DETECTION
        # ==============================================================

        tampered_candidate = dict(
            best_candidate
        )

        tampered_title = (
            tampered_candidate.get(
                "title"
            )
            or ""
        )

        tampered_candidate["title"] = (
            tampered_title
            + " [TAMPERED]"
        )

        tampered_hash_result = (
            self.hasher.generate_candidate_hash(
                tampered_candidate
            )
        )

        pipeline_result["pipeline"][
            "tamper_test"
        ] = {

            "success":
                tampered_hash_result.get(
                    "success",
                    False,
                ),
        }

        if tampered_hash_result.get(
            "success",
            False,
        ):

            tampered_hash = (
                tampered_hash_result.get(
                    "sha256"
                )
            )

            pipeline_result[
                "tampered_hash"
            ] = tampered_hash

            tampered_verification = (
                self.tamper_detector.verify_hash(
                    tampered_hash
                )
            )

            pipeline_result[
                "tampered_verification"
            ] = tampered_verification

        else:

            pipeline_result[
                "tampered_verification"
            ] = {

                "status":
                    "VERIFICATION ERROR",

                "verified":
                    False,

                "error":
                    "Failed to generate tampered hash.",
            }

        # ==============================================================
        # VERIFY ORIGINAL AGAIN
        # ==============================================================

        original_verification = (
            self.tamper_detector.verify_hash(
                original_hash
            )
        )

        pipeline_result[
            "original_verification"
        ] = original_verification

        # ==============================================================
        # FINAL SUCCESS CONDITIONS
        # ==============================================================

        original_is_verified = (

            original_verification.get(
                "status"
            ) == "VERIFIED"

            or original_verification.get(
                "verified"
            ) is True
        )

        tamper_is_detected = (

            pipeline_result[
                "tampered_verification"
            ].get(
                "status"
            )
            == "TAMPER DETECTED"
        )

        if (
            original_is_verified
            and tamper_is_detected
        ):

            pipeline_result["success"] = True

            pipeline_result["message"] = (
                "Face match confirmed, metadata notarized, "
                "on-chain verification passed, and "
                "tamper detection passed."
            )

        else:

            pipeline_result["success"] = False

            pipeline_result["message"] = (
                "Face match was confirmed, but the complete "
                "blockchain/tamper verification workflow "
                "did not fully pass."
            )

        return pipeline_result


# ----------------------------------------------------------------------
# CONVENIENCE FUNCTION
# ----------------------------------------------------------------------

def run_verification_pipeline(
    image_path: Path,
    face_index: int = 0,
    max_candidates: Optional[int] = None,
) -> Dict[str, Any]:

    pipeline = VerificationPipeline(
        max_candidates=max_candidates
    )

    return pipeline.run_pipeline(
        image_path=image_path,
        face_index=face_index,
    )