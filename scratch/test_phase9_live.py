"""Phase 9 Live End-to-End Pipeline Demonstration.

Executes the REAL pipeline using:
- Real local face image
- Real Phase 1 FaceDetector (OpenCV Haar)
- Real Phase 2 SFace encoder
- Real Phase 3 SerpApi Google Lens search (blocked if no API key)
- Real Phase 4 ResultSelector
- Real Phase 5 CandidateHasher (SHA-256)
- Real Phase 7 Hardhat blockchain node (http://127.0.0.1:8545)
- Real deployed FaceVerification contract
- Real Phase 8 TamperDetector

NO fake data. NO mocked blockchain. NO hardcoded results.

Usage:
    python scratch/test_phase9_live.py

Requires:
    - Hardhat node running: npx hardhat node (in project root)
    - .env with BLOCKCHAIN_RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS
    - SERPAPI_API_KEY set in .env for live search (optional; reported if missing)
"""

import copy
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env")

from pipeline import VerificationPipeline


# ---------------------------------------------------------------------------
# Colour/formatting helpers (simple ANSI, safe on most terminals)
# ---------------------------------------------------------------------------
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def _ok(msg: str) -> str:
    return f"{GREEN}PASS{RESET}  {msg}"


def _fail(msg: str) -> str:
    return f"{RED}FAIL{RESET}  {msg}"


def _warn(msg: str) -> str:
    return f"{YELLOW}BLOCKED{RESET}  {msg}"


def _section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{title}{RESET}")
    print("-" * 60)


def _check_env() -> dict:
    """Return environment configuration status (without printing secrets)."""
    return {
        "rpc_url": os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545"),
        "has_private_key": bool(os.getenv("PRIVATE_KEY")),
        "has_contract_address": bool(os.getenv("CONTRACT_ADDRESS")),
        "has_serpapi_key": bool(os.getenv("SERPAPI_API_KEY")),
        "contract_address": os.getenv("CONTRACT_ADDRESS", ""),
    }


def main() -> None:
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}         PHASE 9 — LIVE END-TO-END PIPELINE{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    # -------------------------------------------------------------------
    # Environment check (no secrets printed)
    # -------------------------------------------------------------------
    _section("ENVIRONMENT CHECK")
    env = _check_env()
    print(f"  RPC URL         : {env['rpc_url']}")
    print(f"  Private Key     : {'[SET]' if env['has_private_key'] else '[MISSING]'}")
    print(f"  Contract Address: {'[SET] ' + env['contract_address'] if env['has_contract_address'] else '[MISSING]'}")
    print(f"  SerpApi Key     : {'[SET]' if env['has_serpapi_key'] else '[MISSING — live search will be BLOCKED]'}")

    if not env["has_serpapi_key"]:
        print(f"\n  {YELLOW}WARNING: SERPAPI_API_KEY is not configured.{RESET}")
        print(f"  {YELLOW}         Live external search will be BLOCKED.{RESET}")
        print(f"  {YELLOW}         Set SERPAPI_API_KEY in .env to enable.{RESET}")
        SERPAPI_BLOCKED = True
    else:
        SERPAPI_BLOCKED = False

    # -------------------------------------------------------------------
    # Locate real face image
    # -------------------------------------------------------------------
    _section("SELECTING TEST IMAGE")
    image_path = _project_root / "sample_data" / "test_faces" / "single_face.jpg"
    fallback_path = _project_root / "test_synth.jpg"

    if image_path.is_file():
        print(f"  Using: {image_path}")
    elif fallback_path.is_file():
        image_path = fallback_path
        print(f"  Using fallback: {image_path}")
    else:
        print(f"  {RED}ERROR: No test face image found. Aborting.{RESET}")
        sys.exit(1)

    # -------------------------------------------------------------------
    # Build real pipeline (no mocks)
    # -------------------------------------------------------------------
    pipeline = VerificationPipeline()

    # -------------------------------------------------------------------
    # Execute pipeline
    # -------------------------------------------------------------------
    _section("EXECUTING PIPELINE")
    print("  Running — this may take several seconds for external search and blockchain...\n")

    try:
        result = pipeline.run_pipeline(str(image_path))
    except Exception as exc:
        print(f"  {RED}FATAL: Unhandled exception during pipeline: {exc}{RESET}")
        sys.exit(1)

    stages = result.get("pipeline", {})

    # -------------------------------------------------------------------
    # STEP 1: Face Detection
    # -------------------------------------------------------------------
    _section("[1] Face Detection")
    fd = stages.get("face_detection")
    if fd and fd.get("success"):
        fc = fd["face_count"]
        print(f"  {_ok(f'Faces detected: {fc}')}")
        print(f"  Selected face index : {fd['selected_face_index']}")
        box = fd.get("box", {})
        print(f"  Bounding box        : x={box.get('x')}, y={box.get('y')}, w={box.get('w')}, h={box.get('h')}")
    else:
        err = result.get("error") if result.get("stage") == "face_detection" else "Stage not reached"
        print(f"  {_fail(f'FAILED — {err}')}")
        _print_final_summary(result, SERPAPI_BLOCKED, env)
        return

    # -------------------------------------------------------------------
    # STEP 2: Face Encoding
    # -------------------------------------------------------------------
    _section("[2] Face Encoding  (SFace / Phase 2)")
    fe = stages.get("face_encoding")
    if fe and fe.get("success"):
        edim = fe["embedding_dim"]
        print(f"  {_ok(f'Embedding dimension: {edim}')}")
        preview = fe.get("embedding_preview", [])
        if preview:
            print(f"  Embedding preview   : [{', '.join(f'{v:.4f}' for v in preview[:5])}, ...]")
    else:
        err = result.get("error") if result.get("stage") == "face_encoding" else "Stage not reached"
        print(f"  {_fail(f'FAILED — {err}')}")
        _print_final_summary(result, SERPAPI_BLOCKED, env)
        return

    # -------------------------------------------------------------------
    # STEP 3: External Reverse-Image Search
    # -------------------------------------------------------------------
    _section("[3] External Reverse-Image Search  (SerpApi Google Lens)")
    ws = stages.get("web_search")
    if ws and ws.get("success"):
        wrc = ws["result_count"]
        print(f"  {_ok(f'Results received: {wrc}')}")
        print(f"  Provider  : {ws.get('provider', 'SerpApi (Google Lens)')}")
        print(f"  Image ID  : {ws.get('image_id', 'N/A')}")
        search_status = "PASS"
    else:
        err = result.get("error") if result.get("stage") == "web_search" else "Stage not reached"
        if SERPAPI_BLOCKED or (err and "serpapi_api_key" in str(err).lower()):
            print(f"  {_warn(f'SerpApi key not configured. External search BLOCKED.')}")
            search_status = "BLOCKED"
        else:
            print(f"  {_fail(f'FAILED — {err}')}")
            search_status = "FAILED"
        _print_final_summary(result, SERPAPI_BLOCKED, env, search_status=search_status)
        return

    # -------------------------------------------------------------------
    # STEP 4: Result Selection
    # -------------------------------------------------------------------
    _section("[4] Result Selection  (Phase 4)")
    rs = stages.get("result_selection")
    if rs and rs.get("success"):
        cand = rs.get("selected_candidate", {})
        rsc = rs["selected_count"]
        print(f"  {_ok(f'Candidate selected: {rsc}')}")
        print(f"  Title    : {cand.get('title') or '(none)'}")
        print(f"  URL      : {cand.get('url')}")
        print(f"  Source   : {cand.get('source') or '(none)'}")
        print(f"  Snippet  : {str(cand.get('snippet') or '(none)')[:80]}")
    else:
        err = result.get("error") if result.get("stage") == "result_selection" else "Stage not reached"
        print(f"  {_fail(f'FAILED — {err}')}")
        _print_final_summary(result, SERPAPI_BLOCKED, env)
        return

    # -------------------------------------------------------------------
    # STEP 5: SHA-256 Hashing
    # -------------------------------------------------------------------
    _section("[5] Canonical SHA-256  (Phase 5)")
    hsh = stages.get("hashing")
    if hsh and hsh.get("success"):
        original_hash = hsh.get("sha256")
        halg = hsh.get("hash_algorithm", "SHA-256")
        print(f"  {_ok(f'Algorithm : {halg}')}")
        print(f"  Hash      : {original_hash}")
        canonical_json = hsh.get("canonical_json", "")
        print(f"  Canonical : {canonical_json[:100]}{'...' if len(canonical_json) > 100 else ''}")
    else:
        err = result.get("error") if result.get("stage") == "hashing" else "Stage not reached"
        print(f"  {_fail(f'FAILED — {err}')}")
        _print_final_summary(result, SERPAPI_BLOCKED, env)
        return

    # -------------------------------------------------------------------
    # STEP 6: Blockchain Registration
    # -------------------------------------------------------------------
    _section("[6] Blockchain Registration  (Hardhat + FaceVerification.sol)")
    reg = stages.get("blockchain_registration", {})
    if reg and reg.get("success"):
        status_label = reg.get("registration_status", "registered")
        tx_hash = reg.get("transaction_hash") or "(already registered — no new TX)"
        block_num = reg.get("block_number") or "(N/A)"
        contract_addr = reg.get("contract_address") or env["contract_address"]
        registered_by = reg.get("registered_by") or "(unknown)"

        print(f"  {_ok('Status    : ' + str(status_label))}")
        print(f"  Contract  : {contract_addr}")
        print(f"  TX        : {tx_hash}")
        print(f"  Block     : {block_num}")
        print(f"  Sender    : {registered_by}")
    else:
        err = result.get("error") if result.get("stage") == "blockchain_registration" else "Stage not reached"
        print(f"  {_fail(f'FAILED — {err}')}")
        _print_final_summary(result, SERPAPI_BLOCKED, env)
        return

    # -------------------------------------------------------------------
    # STEP 7: Blockchain Verification
    # -------------------------------------------------------------------
    _section("[7] Blockchain Verification  (Phase 7)")
    bver = stages.get("blockchain_verification", {})
    if bver and bver.get("verified"):
        print(f"  {_ok('Status: VERIFIED')}")
        raw = bver.get("raw_record", {})
        print(f"  Registered by: {raw.get('registered_by') or '(unknown)'}")
        print(f"  Timestamp    : {raw.get('timestamp') or '(unknown)'}")
    else:
        err = result.get("error") if result.get("stage") == "blockchain_verification" else "Stage not reached"
        print(f"  {_fail(f'FAILED — {err}')}")
        _print_final_summary(result, SERPAPI_BLOCKED, env)
        return

    # -------------------------------------------------------------------
    # STEP 8: Tamper Detection
    # -------------------------------------------------------------------
    _section("[8] Tamper Detection  (Phase 8)")
    td = stages.get("tamper_detection", {})

    orig_ver = result.get("original_verification", {})
    tamp_ver = result.get("tampered_verification", {})
    orig_hash = result.get("original_hash")
    tamp_hash = result.get("tampered_hash")

    orig_status = orig_ver.get("status", "UNKNOWN")
    tamp_status = tamp_ver.get("status", "UNKNOWN")

    if orig_status == "VERIFIED":
        print(f"  {_ok(f'Original  -> {orig_status}')}")
    else:
        print(f"  {_fail(f'Original  -> {orig_status} (expected VERIFIED)')}")

    if tamp_status == "TAMPER DETECTED":
        print(f"  {_ok(f'Tampered  -> {tamp_status}')}")
    else:
        print(f"  {_fail(f'Tampered  -> {tamp_status} (expected TAMPER DETECTED)')}")

    hashes_different = (orig_hash != tamp_hash) if (orig_hash and tamp_hash) else False
    if hashes_different:
        print(f"  {_ok('Hashes are DIFFERENT (tamper proof confirmed)')}")
    else:
        print(f"  {_fail('Hashes are SAME or missing (tamper proof failed)')}")

    print(f"\n  Original hash : {orig_hash}")
    print(f"  Tampered hash : {tamp_hash}")

    # -------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------
    _print_final_summary(result, SERPAPI_BLOCKED, env,
                         search_status="PASS",
                         orig_status=orig_status,
                         tamp_status=tamp_status,
                         orig_hash=orig_hash,
                         tamp_hash=tamp_hash,
                         hashes_different=hashes_different,
                         reg=reg,
                         contract_addr=reg.get("contract_address") or env["contract_address"],
                         tx_hash=reg.get("transaction_hash"),
                         block_num=reg.get("block_number"))


def _print_final_summary(
    result,
    serpapi_blocked,
    env,
    *,
    search_status="BLOCKED",
    orig_status="N/A",
    tamp_status="N/A",
    orig_hash=None,
    tamp_hash=None,
    hashes_different=None,
    reg=None,
    contract_addr=None,
    tx_hash=None,
    block_num=None,
):
    stages = result.get("pipeline", {})
    overall_success = result.get("success", False)

    print(f"\n{'=' * 60}")
    print(f"{BOLD}PHASE 9 STATUS{RESET}")
    print(f"{'=' * 60}")

    def _s(stage_key):
        s = stages.get(stage_key, {})
        if s and s.get("success"):
            return f"{GREEN}PASS{RESET}"
        elif stage_key == result.get("stage"):
            return f"{RED}FAILED{RESET}"
        else:
            return f"{YELLOW}NOT REACHED{RESET}"

    print(f"  Face detection          : {_s('face_detection')}")
    print(f"  Face encoding           : {_s('face_encoding')}")

    if serpapi_blocked:
        print(f"  External search         : {YELLOW}BLOCKED (no SERPAPI_API_KEY){RESET}")
    else:
        print(f"  External search         : {_s('web_search')}")

    print(f"  Result selection        : {_s('result_selection')}")
    print(f"  SHA-256 hashing         : {_s('hashing')}")
    print(f"  Blockchain registration : {_s('blockchain_registration')}")
    print(f"  Blockchain verification : {_s('blockchain_verification')}")
    print(f"  Tamper detection        : {_s('tamper_detection')}")

    print()
    hsh = stages.get("hashing", {})
    print(f"  Original hash           : {orig_hash or hsh.get('sha256') or 'N/A'}")
    print(f"  Tampered hash           : {tamp_hash or 'N/A'}")

    if hashes_different is None:
        print(f"  Hashes different        : N/A")
    else:
        print(f"  Hashes different        : {'YES' if hashes_different else 'NO'}")

    print(f"  Original verification   : {orig_status}")
    print(f"  Tampered verification   : {tamp_status}")

    print()
    print(f"  Contract address        : {contract_addr or env.get('contract_address') or 'N/A'}")
    print(f"  Registration TX         : {tx_hash or 'N/A'}")
    print(f"  Registration block      : {block_num or 'N/A'}")
    print(f"  SerpApi live search     : {search_status}")

    print()
    if overall_success:
        print(f"  {BOLD}{GREEN}END-TO-END PIPELINE: PASS [OK]{RESET}")
    elif serpapi_blocked and result.get("stage") == "web_search":
        print(f"  {BOLD}{YELLOW}END-TO-END PIPELINE: BLOCKED (SERPAPI_API_KEY missing){RESET}")
    else:
        failed_stage = result.get("stage", "unknown")
        print(f"  {BOLD}{RED}END-TO-END PIPELINE: FAILED (at stage: {failed_stage}){RESET}")
        if result.get("error"):
            print(f"  Error: {result['error']}")

    print()
    print(f"  Phase 10/UI implemented : NO")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
