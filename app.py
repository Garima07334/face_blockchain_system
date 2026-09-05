"""
Phase 10 / Phase 11: Streamlit Dashboard.

Displays:
- Face detection
- SFace embedding
- Google Lens external discovery
- Candidate face verification
- SHA-256 metadata fingerprint
- Blockchain notarization
- Tamper detection

Security rule:
Blockchain stages are displayed only when a valid SFace
candidate match has been confirmed.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import streamlit as st
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Project Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

load_dotenv(
    PROJECT_ROOT / ".env"
)


from pipeline import VerificationPipeline
from core.blockchain import BlockchainClient
from core.face_detector import FaceDetector


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Face Verification & Blockchain",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>

.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

.hash-code {
    font-family: 'Courier New', monospace;
    background-color: #1F2937;
    color: #6EE7B7;
    padding: 8px 12px;
    border-radius: 6px;
    word-break: break-all;
    display: inline-block;
}

.disclaimer-box {
    background-color: rgba(59, 130, 246, 0.1);
    border-left: 4px solid #3B82F6;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 16px;
}

</style>
"""

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def draw_all_face_bounding_boxes(
    image_path: Path,
    faces: List[Dict[str, Any]],
    selected_index: int = 0,
) -> Optional[Any]:

    if not image_path.is_file():
        return None

    img = cv2.imread(
        str(image_path)
    )

    if img is None:
        return None

    for idx, face in enumerate(
        faces
    ):

        box = face.get(
            "box",
            {}
        )

        if not all(
            key in box
            for key in (
                "x",
                "y",
                "w",
                "h",
            )
        ):
            continue

        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]

        if idx == selected_index:
            color = (
                0,
                255,
                0,
            )
            thickness = 3
            label = (
                f"Selected Face #{idx}"
            )
        else:
            color = (
                0,
                215,
                255,
            )
            thickness = 2
            label = (
                f"Face #{idx}"
            )

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            color,
            thickness,
        )

        cv2.putText(
            img,
            label,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB,
    )


def crop_face_roi(
    image_path: Path,
    box: Optional[Dict[str, int]],
) -> Optional[Any]:

    if (
        not image_path.is_file()
        or not box
    ):
        return None

    img = cv2.imread(
        str(image_path)
    )

    if img is None:
        return None

    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    img_h, img_w = img.shape[:2]

    x1 = max(
        0,
        x,
    )

    y1 = max(
        0,
        y,
    )

    x2 = min(
        img_w,
        x + w,
    )

    y2 = min(
        img_h,
        y + h,
    )

    crop = img[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    return cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB,
    )


def get_available_sample_faces() -> Dict[str, Path]:

    sample_dir = (
        PROJECT_ROOT
        / "sample_data"
        / "test_faces"
    )

    samples = {}

    if sample_dir.is_dir():

        for file in sample_dir.glob(
            "*.jpg"
        ):
            samples[
                f"Sample: {file.name}"
            ] = file

        for file in sample_dir.glob(
            "*.png"
        ):
            samples[
                f"Sample: {file.name}"
            ] = file

    fallback = (
        PROJECT_ROOT
        / "test_synth.jpg"
    )

    if fallback.is_file():

        samples[
            f"Sample: {fallback.name}"
        ] = fallback

    return samples


def is_real_face_match(
    face_match: Dict[str, Any],
) -> bool:
    """
    Final defensive match gate.

    A candidate can NEVER be considered matched unless:
    - status == MATCH
    - match == True
    - at least one face was detected
    - a best face index exists
    - similarity exists
    - similarity >= threshold
    """

    if not isinstance(
        face_match,
        dict,
    ):
        return False

    similarity = face_match.get(
        "similarity"
    )

    threshold = face_match.get(
        "threshold"
    )

    face_count = face_match.get(
        "candidate_face_count",
        0,
    )

    best_index = face_match.get(
        "best_candidate_face_index"
    )

    return bool(
        face_match.get("match") is True
        and face_match.get("status") == "MATCH"
        and face_count > 0
        and best_index is not None
        and similarity is not None
        and threshold is not None
        and similarity >= threshold
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    st.title(
        "🛡️ Face Verification & Blockchain Notarization"
    )

    st.caption(
        "External Discovery → SFace Verification → SHA-256 → Blockchain"
    )

    # ======================================================================
    # SIDEBAR
    # ======================================================================

    st.sidebar.header(
        "⚙️ System Status"
    )

    blockchain_client = BlockchainClient()

    if blockchain_client.is_connected():

        st.sidebar.success(
            "🟢 Blockchain RPC Connected"
        )

    else:

        st.sidebar.error(
            "🔴 Blockchain RPC Disconnected"
        )

        st.sidebar.caption(
            "Run `npx hardhat node`."
        )

    contract_address = os.getenv(
        "CONTRACT_ADDRESS",
        "",
    )

    if contract_address:

        st.sidebar.info(
            f"📜 Contract: "
            f"`{contract_address[:10]}...`"
        )

    else:

        st.sidebar.warning(
            "📜 Contract address missing."
        )

    serp_key = os.getenv(
        "SERPAPI_API_KEY",
        "",
    )

    if serp_key:

        st.sidebar.success(
            "🔑 SerpApi Key Configured"
        )

    else:

        st.sidebar.warning(
            "🔑 SerpApi Key Missing"
        )

    threshold = os.getenv(
        "FACE_MATCH_THRESHOLD",
        "0.363",
    )

    st.sidebar.info(
        f"🎯 SFace Threshold: `{threshold}`"
    )

    # ======================================================================
    # TABS
    # ======================================================================

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🚀 Live Verification",
            "🔍 Blockchain Audit",
            "🛠️ Contract Setup",
            "🏗️ Architecture & Logs",
        ]
    )

    # ======================================================================
    # TAB 1
    # ======================================================================

    with tab1:

        st.subheader(
            "Step 1: Input Face Image"
        )

        input_mode = st.radio(
            "Input Source",
            [
                "Live Webcam Snapshot",
                "Upload Image",
                "Select Real Sample Image",
            ],
            horizontal=True,
        )

        selected_image_path = None
        temp_file = None

        # ---------------------------------------------------------------
        # LIVE WEBCAM
        # ---------------------------------------------------------------

        if input_mode == "Live Webcam Snapshot":
            cam_pic = st.camera_input("Capture a face snapshot")
            if cam_pic:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as file:
                    file.write(cam_pic.getvalue())
                    selected_image_path = Path(file.name)
                    temp_file = selected_image_path

        # ---------------------------------------------------------------
        # UPLOAD
        # ---------------------------------------------------------------

        elif input_mode == "Upload Image":

            uploaded_file = st.file_uploader(
                "Upload a face photo",
                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp",
                ],
            )

            if uploaded_file:

                suffix = (
                    Path(
                        uploaded_file.name
                    ).suffix
                    or ".jpg"
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as file:

                    file.write(
                        uploaded_file.getvalue()
                    )

                    selected_image_path = Path(
                        file.name
                    )

                    temp_file = (
                        selected_image_path
                    )

        # ---------------------------------------------------------------
        # SAMPLE
        # ---------------------------------------------------------------

        else:

            samples = (
                get_available_sample_faces()
            )

            if samples:

                selected = st.selectbox(
                    "Choose sample",
                    list(samples.keys()),
                )

                selected_image_path = (
                    samples[selected]
                )

            else:

                st.warning(
                    "No sample images found."
                )

        # =================================================================
        # IMAGE AVAILABLE
        # =================================================================

        if (
            selected_image_path
            and selected_image_path.is_file()
        ):

            detector = FaceDetector()

            preview_detection = (
                detector.detect_faces(
                    selected_image_path
                )
            )

            detected_faces = (
                preview_detection.get(
                    "faces",
                    []
                )
            )

            face_count = (
                preview_detection.get(
                    "face_count",
                    0,
                )
            )

            # -------------------------------------------------------------
            # FACE SELECTION
            # -------------------------------------------------------------

            face_options = {}

            for idx, face in enumerate(
                detected_faces
            ):

                box = face["box"]

                face_options[
                    f"Face #{idx}: "
                    f"{box['w']}x{box['h']} "
                    f"at ({box['x']},{box['y']})"
                ] = idx

            selected_label = st.selectbox(
                "Select Query Face",
                (
                    list(
                        face_options.keys()
                    )
                    if face_options
                    else ["Face #0"]
                ),
            )

            selected_face_index = (
                face_options.get(
                    selected_label,
                    0,
                )
            )

            col1, col2, col3 = st.columns(
                [2, 1, 2]
            )

            # -------------------------------------------------------------
            # SOURCE
            # -------------------------------------------------------------

            with col1:

                annotated = (
                    draw_all_face_bounding_boxes(
                        selected_image_path,
                        detected_faces,
                        selected_face_index,
                    )
                )

                if annotated is not None:

                    st.image(
                        annotated,
                        caption=(
                            f"{face_count} Face(s) Detected"
                        ),
                        use_container_width=True,
                    )

            # -------------------------------------------------------------
            # CROP
            # -------------------------------------------------------------

            with col2:

                st.markdown(
                    "### Selected Face"
                )

                if (
                    face_count > 0
                    and selected_face_index
                    < len(detected_faces)
                ):

                    box = detected_faces[
                        selected_face_index
                    ]["box"]

                    crop = crop_face_roi(
                        selected_image_path,
                        box,
                    )

                    if crop is not None:

                        st.image(
                            crop,
                            width=160,
                        )

                    st.caption(
                        f"x={box['x']} "
                        f"y={box['y']} "
                        f"w={box['w']} "
                        f"h={box['h']}"
                    )

            # -------------------------------------------------------------
            # RUN
            # -------------------------------------------------------------

            with col3:

                st.markdown(
                    "### Pipeline Execution"
                )

                st.write(
                    "The system will perform:"
                )

                st.write(
                    "Face Detection → "
                    "SFace → Google Lens → "
                    "Candidate SFace Verification → "
                    "SHA-256 → Blockchain"
                )

                run_button = st.button(
                    "🚀 Run Verification Pipeline",
                    type="primary",
                    use_container_width=True,
                )

            # =============================================================
            # EXECUTE
            # =============================================================

            if run_button:

                with st.status(
                    "Executing pipeline...",
                    expanded=True,
                ) as status:

                    st.write(
                        "🔍 Detecting and encoding query face..."
                    )

                    pipeline = (
                        VerificationPipeline()
                    )

                    result = (
                        pipeline.run_pipeline(
                            selected_image_path,
                            face_index=selected_face_index,
                        )
                    )

                    st.session_state[
                        "last_pipeline_result"
                    ] = result

                    st.session_state[
                        "last_image_path"
                    ] = str(
                        selected_image_path
                    )

                    fm = (
                        result
                        .get("pipeline", {})
                        .get(
                            "candidate_face_match",
                            {},
                        )
                    )

                    if (
                        result.get("success")
                        and is_real_face_match(fm)
                    ):

                        status.update(
                            label=(
                                "✅ FACE MATCH CONFIRMED "
                                "— METADATA NOTARIZED"
                            ),
                            state="complete",
                            expanded=False,
                        )

                    elif fm.get(
                        "status"
                    ) == "NO_MATCH":

                        status.update(
                            label=(
                                "⚠️ NO SAME-FACE MATCH CONFIRMED"
                            ),
                            state="error",
                            expanded=True,
                        )

                    else:

                        status.update(
                            label=(
                                "❌ PIPELINE STOPPED"
                            ),
                            state="error",
                            expanded=True,
                        )

    # ======================================================================
    # RESULTS
    # ======================================================================

    if (
        "last_pipeline_result"
        in st.session_state
    ):

        result = st.session_state[
            "last_pipeline_result"
        ]

        stages = result.get(
            "pipeline",
            {},
        )

        face_match = stages.get(
            "candidate_face_match",
            {},
        )

        actual_match = is_real_face_match(
            face_match
        )

        st.divider()

        st.subheader(
            "Pipeline Results"
        )

        # ================================================================
        # OVERALL RESULT
        # ================================================================

        if (
            result.get("success")
            and actual_match
        ):

            st.success(
                "🎉 **FACE MATCH CONFIRMED — "
                "METADATA NOTARIZED ON BLOCKCHAIN**"
            )

        elif face_match.get(
            "status"
        ) == "NO_MATCH":

            st.error(
                "⚠️ **NO SAME-FACE MATCH CONFIRMED**"
            )

        else:

            st.error(
                "❌ **PIPELINE VERIFICATION FAILED**"
            )

        # ================================================================
        # DISCLAIMER
        # ================================================================

        st.markdown(
            """
            <div class="disclaimer-box">

            ℹ️ <b>Verification Notice:</b>
            SFace provides biometric visual similarity between
            detected face regions. It is not legal identity certification.
            Google Lens provides external candidate discovery.
            Blockchain verification proves integrity of the registered
            metadata fingerprint; it does not prove identity.

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ================================================================
        # STAGE 1
        # ================================================================

        with st.expander(
            "📌 1. Face Detection & SFace Encoding",
            expanded=True,
        ):

            fd = stages.get(
                "face_detection",
                {},
            )

            fe = stages.get(
                "face_encoding",
                {},
            )

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Faces Detected",
                    fd.get(
                        "face_count",
                        0,
                    ),
                )

                st.metric(
                    "Selected Face",
                    fd.get(
                        "selected_face_index"
                    ),
                )

            with c2:

                st.metric(
                    "Embedding Dimension",
                    fe.get(
                        "embedding_dim",
                        128,
                    ),
                )

                st.write(
                    "Model: OpenCV SFace"
                )

        # ================================================================
        # STAGE 2
        # ================================================================

        with st.expander(
            "🌐 2. External Search",
            expanded=True,
        ):

            ws = stages.get(
                "web_search",
                {},
            )

            rs = stages.get(
                "result_selection",
                {},
            )

            if ws.get("success"):

                st.write(
                    f"**Provider:** "
                    f"`{ws.get('provider')}`"
                )

                st.write(
                    f"**Raw Results:** "
                    f"`{ws.get('result_count')}`"
                )

            if rs.get("success"):

                st.write(
                    f"**Normalized Candidates:** "
                    f"`{rs.get('selected_count')}`"
                )

                st.caption(
                    "These are discovery candidates. "
                    "They are NOT automatically considered the same person."
                )

        # ================================================================
        # STAGE 3 — ACTUAL SFACE RESULT
        # ================================================================

        with st.expander(
            "🎯 3. Candidate Face Verification",
            expanded=True,
        ):

            if face_match:

                if actual_match:

                    st.success(
                        "✅ FACE MATCH CONFIRMED"
                    )

                elif (
                    face_match.get(
                        "status"
                    )
                    == "NO_MATCH"
                ):

                    st.error(
                        "❌ NO SAME-FACE MATCH CONFIRMED"
                    )

                else:

                    st.warning(
                        "⚠️ FACE VERIFICATION ERROR"
                    )

                m1, m2 = st.columns(2)

                with m1:

                    st.write(
                        "**Candidate Face Count:**",
                        face_match.get(
                            "candidate_face_count",
                            0,
                        ),
                    )

                    st.write(
                        "**Best Candidate Face:**",
                        face_match.get(
                            "best_candidate_face_index"
                        ),
                    )

                    st.write(
                        "**Evaluated Candidates:**",
                        face_match.get(
                            "evaluated_candidates_count",
                            0,
                        ),
                    )

                with m2:

                    st.write(
                        "**Cosine Similarity:**",
                        face_match.get(
                            "similarity"
                        )
                        if face_match.get(
                            "similarity"
                        )
                        is not None
                        else "N/A",
                    )

                    st.write(
                        "**Cosine Distance:**",
                        face_match.get(
                            "distance"
                        )
                        if face_match.get(
                            "distance"
                        )
                        is not None
                        else "N/A",
                    )

                    st.write(
                        "**Threshold:**",
                        face_match.get(
                            "threshold"
                        ),
                    )

                if face_match.get(
                    "error"
                ):

                    st.warning(
                        face_match["error"]
                    )

                # --------------------------------------------------------
                # ACTUAL MATCHED CANDIDATE
                # --------------------------------------------------------

                matched_candidate = (
                    face_match.get(
                        "selected_candidate"
                    )
                    or {}
                )

                if matched_candidate:

                    st.markdown(
                        "### 🔎 Candidate Selected By SFace"
                    )

                    candidate_col1, candidate_col2 = (
                        st.columns(
                            [1, 3]
                        )
                    )

                    with candidate_col1:

                        image_url = (
                            matched_candidate.get(
                                "image_url"
                            )
                            or matched_candidate.get(
                                "thumbnail"
                            )
                        )

                        if image_url:

                            st.image(
                                image_url,
                                caption=(
                                    "External Candidate Image"
                                ),
                                width=180,
                            )

                    with candidate_col2:

                        st.markdown(
                            f"**Title:** "
                            f"{matched_candidate.get('title') or 'Untitled'}"
                        )

                        st.markdown(
                            f"**Source:** "
                            f"`{matched_candidate.get('source') or 'Unknown'}`"
                        )

                        candidate_url = (
                            matched_candidate.get(
                                "url"
                            )
                        )

                        if candidate_url:

                            st.markdown(
                                f"**Post URL:** "
                                f"{candidate_url}"
                            )

                        st.markdown(
                            f"**SFace Similarity:** "
                            f"`{face_match.get('similarity')}`"
                        )

        # ================================================================
        # BLOCKCHAIN ONLY AFTER MATCH
        # ================================================================

        if actual_match:

            # ------------------------------------------------------------
            # HASH
            # ------------------------------------------------------------

            with st.expander(
                "🔑 4. SHA-256 Metadata Fingerprint",
                expanded=True,
            ):

                hashing = stages.get(
                    "hashing",
                    {},
                )

                if hashing.get(
                    "success"
                ):

                    st.write(
                        "**Algorithm:** `SHA-256`"
                    )

                    st.markdown(
                        f"""
                        <span class="hash-code">
                        {hashing.get("sha256")}
                        </span>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.write(
                        "**Canonical JSON:**"
                    )

                    st.code(
                        hashing.get(
                            "canonical_json",
                            "",
                        ),
                        language="json",
                    )

            # ------------------------------------------------------------
            # BLOCKCHAIN
            # ------------------------------------------------------------

            with st.expander(
                "⛓️ 5. Blockchain Notarization",
                expanded=True,
            ):

                registration = stages.get(
                    "blockchain_registration",
                    {},
                )

                verification = stages.get(
                    "blockchain_verification",
                    {},
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown(
                        "### Registration"
                    )

                    if registration.get(
                        "success"
                    ):

                        st.write(
                            "**Status:**",
                            registration.get(
                                "registration_status"
                            ),
                        )

                        st.write(
                            "**Contract:**",
                            registration.get(
                                "contract_address"
                            ),
                        )

                        st.write(
                            "**Transaction:**",
                            registration.get(
                                "transaction_hash"
                            ),
                        )

                        st.write(
                            "**Block:**",
                            registration.get(
                                "block_number"
                            ),
                        )

                with c2:

                    st.markdown(
                        "### Verification"
                    )

                    if verification.get(
                        "verified"
                    ):

                        st.success(
                            "✅ METADATA VERIFIED ON-CHAIN"
                        )

                    else:

                        st.error(
                            "❌ NOT VERIFIED"
                        )

            # ------------------------------------------------------------
            # TAMPER
            # ------------------------------------------------------------

            with st.expander(
                "🛡️ 6. Anti-Tamper Proof",
                expanded=True,
            ):

                original_hash = result.get(
                    "original_hash"
                )

                tampered_hash = result.get(
                    "tampered_hash"
                )

                # --------------------------------------------------------
                # FIX:
                # result may contain None instead of {}
                # when blockchain verification did not complete.
                # Using "or {}" safely converts None to an empty dict.
                # --------------------------------------------------------

                original_verification = (
                    result.get(
                        "original_verification"
                    ) or {}
                )

                tampered_verification = (
                    result.get(
                        "tampered_verification"
                    ) or {}
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown(
                        "### Original"
                    )

                    st.code(
                        original_hash
                        or "N/A"
                    )

                    st.success(
                        original_verification.get(
                            "status",
                            "UNKNOWN",
                        )
                    )

                with c2:

                    st.markdown(
                        "### Tampered"
                    )

                    st.code(
                        tampered_hash
                        or "N/A"
                    )

                    if (
                        tampered_verification.get(
                            "status"
                        )
                        == "TAMPER DETECTED"
                    ):

                        st.error(
                            "TAMPER DETECTED"
                        )

                    else:

                        st.warning(
                            tampered_verification.get(
                                "status",
                                "UNKNOWN",
                            )
                        )

                if (
                    original_hash
                    and tampered_hash
                    and original_hash
                    != tampered_hash
                ):

                    st.success(
                        "🔒 SHA-256 fingerprints differ — "
                        "metadata modification is detectable."
                    )

        else:

            st.info(
                "🔒 **Blockchain Gate:** "
                "No valid SFace face match was confirmed. "
                "Hashing and blockchain registration were blocked."
            )

    # ======================================================================
    # TAB 2
    # ======================================================================

    with tab2:

        st.subheader(
            "🔍 Direct On-Chain Hash Audit"
        )

        query_hash = st.text_input(
            "SHA-256 Hash",
            max_chars=64,
        )

        if st.button(
            "🔎 Query Blockchain"
        ):

            client = BlockchainClient()

            if not client.is_connected():

                st.error(
                    "Blockchain RPC unavailable."
                )

            else:

                verification = (
                    client.verify_hash(
                        query_hash.strip()
                    )
                )

                record = (
                    client.get_record(
                        query_hash.strip()
                    )
                )

                if verification.get(
                    "verified"
                ):

                    st.success(
                        "✅ METADATA VERIFIED ON-CHAIN"
                    )

                else:

                    st.error(
                        "❌ HASH NOT FOUND"
                    )

                if record.get(
                    "exists"
                ):

                    st.write(
                        "**Registered By:**",
                        record.get(
                            "registered_by"
                        ),
                    )

                    st.write(
                        "**Timestamp:**",
                        record.get(
                            "timestamp"
                        ),
                    )

    # ======================================================================
    # TAB 3
    # ======================================================================

    with tab3:

        st.subheader(
            "🛠️ Contract Setup"
        )

        client = BlockchainClient()

        st.write(
            "**RPC:**",
            client.rpc_url,
        )

        st.write(
            "**Contract:**",
            client.contract_address
            or "Not configured",
        )

        st.write(
            "**Account:**",
            client.sender_address
            or "Not configured",
        )

        st.info(
            "Deploy a fresh contract only when restarting "
            "a local Hardhat blockchain."
        )

        confirmation = st.checkbox(
            "I understand this deploys a new contract."
        )

        if st.button(
            "🚀 Deploy Contract",
            disabled=not confirmation,
        ):

            deployment = (
                client.deploy_contract()
            )

            if deployment.get(
                "success"
            ):

                st.success(
                    "✅ Contract deployed."
                )

                st.code(
                    deployment.get(
                        "contract_address"
                    )
                )

                st.write(
                    "Block:",
                    deployment.get(
                        "block_number"
                    ),
                )

            else:

                st.error(
                    deployment.get(
                        "error"
                    )
                )

    # ======================================================================
    # TAB 4
    # ======================================================================

    with tab4:

        st.subheader(
            "🏗️ System Architecture"
        )

        st.code(
            """
Face Image
    ↓
Face Detection
    ↓
SFace 128D Embedding
    ↓
Google Lens Reverse Image Search
    ↓
External Candidate Discovery
    ↓
Candidate Image Download
    ↓
Candidate Face Detection
    ↓
SFace Comparison
    ↓
Best Candidate Similarity
    ↓
Threshold Gate
    ↓
        ┌───────────────┐
        │               │
     NO MATCH        MATCH
        │               │
        ▼               ▼
      STOP           SHA-256
                        ↓
                    Blockchain
                        ↓
                  On-Chain Verify
                        ↓
                   Tamper Test
            """
        )

        st.subheader(
            "Raw Pipeline Response"
        )

        if (
            "last_pipeline_result"
            in st.session_state
        ):

            st.json(
                st.session_state[
                    "last_pipeline_result"
                ]
            )

        else:

            st.info(
                "No pipeline execution yet."
            )


if __name__ == "__main__":
    main()