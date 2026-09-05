"""Phase 11 Diagnostic Debugger for Candidate Face Match Verification.

Inspects:
1. Query face detection, bounding box coordinates, and crop ROI passed to SFace.
2. Candidate image download, URL vs Thumbnail URL, byte size, image dimensions.
3. Candidate face detection, every candidate face bounding box and crop ROI.
4. Cosine similarity score for EVERY candidate face.
5. Verification of SFace encoding preprocessing consistency.
"""

from pathlib import Path
import sys
import os
import cv2
import numpy as np
import requests

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from core.face_detector import FaceDetector
from core.face_encoder import FaceEncoder
from core.face_matcher import FaceMatcher
from core.web_search import WebSearch
from pipeline import VerificationPipeline


def debug_image_pair(query_image_path: Path, candidate_image_path_or_url: str) -> None:
    """Debug similarity between a query face image and a candidate image."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC FACE MATCH DEBUGGER")
    print("=" * 60)

    detector = FaceDetector()
    encoder = FaceEncoder(detector=detector)
    matcher = FaceMatcher(detector=detector, encoder=encoder)

    # 1. Query Image Analysis
    print(f"\n[QUERY IMAGE] {query_image_path}")
    q_det = detector.detect_faces(query_image_path)
    print(f"  Success         : {q_det.get('success')}")
    print(f"  Image Size      : {q_det.get('image_size')}")
    print(f"  Face Count      : {q_det.get('face_count')}")

    q_faces = q_det.get("faces", [])
    for idx, f in enumerate(q_faces):
        box = f["box"]
        area = box["w"] * box["h"]
        print(f"  Query Face #{idx}: box={box}, Area={area} px, Conf={f.get('confidence')}")

    if not q_faces:
        print("  ERROR: No query face detected.")
        return

    # Use primary/largest face (index 0)
    q_box = q_faces[0]["box"]
    print(f"  Selected Query Face Box (#0): {q_box}")

    q_enc = encoder.encode_face(query_image_path, face_index=0)
    if not q_enc.get("success"):
        print(f"  ERROR: Query face encoding failed: {q_enc.get('error')}")
        return

    q_emb = q_enc["embedding"]
    print(f"  Query Embedding Dim: {len(q_emb)}")
    print(f"  Query Embedding Preview (first 5): {[round(v, 4) for v in q_emb[:5]]}")

    # Test Self-Similarity
    self_sim, self_dist = matcher.compute_similarity(q_emb, q_emb)
    print(f"  Query Self-Similarity (Identity Check): {self_sim:.6f}")

    # 2. Candidate Image Retrieval
    print(f"\n[CANDIDATE IMAGE] Input: {candidate_image_path_or_url}")
    cand_path: Path
    temp_clean: bool = False

    if str(candidate_image_path_or_url).startswith("http://") or str(candidate_image_path_or_url).startswith("https://"):
        dl_res = matcher.download_candidate_image(candidate_image_path_or_url)
        print(f"  Download Success : {dl_res.get('success')}")
        if not dl_res.get("success"):
            print(f"  Download Error   : {dl_res.get('error')}")
            return
        cand_path = dl_res["temp_path"]
        temp_clean = True
        print(f"  Downloaded Bytes : {len(dl_res.get('image_bytes', b''))} bytes")
    else:
        cand_path = Path(candidate_image_path_or_url)
        if not cand_path.is_file():
            print(f"  ERROR: Candidate local file not found: {cand_path}")
            return

    # Decode candidate image for dimensions
    cand_img = cv2.imread(str(cand_path))
    if cand_img is None:
        print("  ERROR: Failed to decode candidate image.")
        return

    c_h, c_w = cand_img.shape[:2]
    print(f"  Candidate Image Dimensions: {c_w} x {c_h}")

    # 3. Candidate Face Detection
    c_det = detector.detect_faces(cand_path)
    print(f"  Candidate Face Count : {c_det.get('face_count')}")
    c_faces = c_det.get("faces", [])

    for idx, f in enumerate(c_faces):
        box = f["box"]
        area = box["w"] * box["h"]
        print(f"  Candidate Face #{idx}: box={box}, Area={area} px, Conf={f.get('confidence')}")

    if not c_faces:
        print("  WARNING: No face detected in candidate image.")
        if temp_clean and cand_path.is_file():
            cand_path.unlink()
        return

    # 4. Compare Query Embedding against EVERY Candidate Face
    print("\n[FACE SIMILARITY COMPARISON PER CANDIDATE FACE]")
    best_sim = -2.0
    best_idx = -1

    for idx in range(len(c_faces)):
        c_enc = encoder.encode_face(cand_path, face_index=idx)
        if c_enc.get("success") and c_enc.get("embedding"):
            c_emb = c_enc["embedding"]
            sim, dist = matcher.compute_similarity(q_emb, c_emb)
            print(f"  Candidate Face #{idx}: Cosine Similarity = {sim:.4f}, Cosine Distance = {dist:.4f}")
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        else:
            print(f"  Candidate Face #{idx}: Encoding failed - {c_enc.get('error')}")

    print("\n[FINAL MATCH DECISION]")
    print(f"  Best Candidate Face Index : {best_idx}")
    print(f"  Best Similarity Score     : {best_sim:.4f}")
    print(f"  Configured Threshold      : {matcher.threshold}")
    print(f"  Match Decision            : {'MATCH CONFIRMED' if best_sim >= matcher.threshold else 'NO SAME-FACE MATCH'}")

    if temp_clean and cand_path.is_file():
        try:
            cand_path.unlink()
        except Exception:
            pass
    print("=" * 60 + "\n")


if __name__ == "__main__":
    query_file = _project_root / "sample_data" / "test_faces" / "single_face.jpg"
    if query_file.is_file():
        # Debug self-match
        debug_image_pair(query_file, str(query_file))
