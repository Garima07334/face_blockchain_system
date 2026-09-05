"""Phase 8: Tamper Detection Layer.

Detects tampering in external visual-search candidate metadata by recalculating
its canonical SHA-256 fingerprint using Phase 5 CandidateHasher and querying
the Phase 7 BlockchainClient smart contract record on the local Ethereum network.
"""

from typing import Any, Dict, Optional
import logging

from core.blockchain import BlockchainClient
from core.hasher import CandidateHasher

logger = logging.getLogger(__name__)


class TamperDetector:
    """Tamper detector verifying candidate metadata integrity against the blockchain."""

    STATUS_VERIFIED = "VERIFIED"
    STATUS_TAMPER_DETECTED = "TAMPER DETECTED"
    STATUS_VERIFICATION_ERROR = "VERIFICATION ERROR"

    def __init__(
        self,
        hasher: Optional[CandidateHasher] = None,
        blockchain_client: Optional[BlockchainClient] = None,
    ) -> None:
        """Initialize TamperDetector with optional hasher and blockchain_client.

        Args:
            hasher: CandidateHasher instance. Defaults to new CandidateHasher.
            blockchain_client: BlockchainClient instance. Defaults to new BlockchainClient.
        """
        self.hasher = hasher or CandidateHasher()
        self.blockchain_client = blockchain_client or BlockchainClient()

    def verify_candidate_integrity(self, candidate: Any) -> Dict[str, Any]:
        """Verify candidate metadata integrity by hashing and querying the blockchain.

        States distinguished:
        1. VERIFIED: Hash calculation succeeds, blockchain query succeeds, verifyHash returns True.
        2. TAMPER DETECTED: Hash calculation succeeds, blockchain query succeeds, verifyHash returns False.
        3. VERIFICATION ERROR: Hash/validation fails, RPC is unavailable, contract is missing, or network call fails.
           (API/network failure is NEVER treated as tampering).

        Args:
            candidate: Candidate dictionary from Phase 4/5.

        Returns:
            Structured dictionary:
            {
                "success": bool,
                "status": "VERIFIED" | "TAMPER DETECTED" | "VERIFICATION ERROR",
                "current_hash": str | None,
                "blockchain_verified": bool,
                "record": dict | None,
                "tampered": bool,
                "error": str | None
            }
        """
        # 1. Recalculate canonical SHA-256 fingerprint using Phase 5 CandidateHasher
        hasher_res = self.hasher.generate_candidate_hash(candidate)
        if not hasher_res.get("success", False):
            err_msg = hasher_res.get("error") or "Failed to generate SHA-256 hash from candidate metadata."
            return {
                "success": False,
                "status": self.STATUS_VERIFICATION_ERROR,
                "current_hash": None,
                "blockchain_verified": False,
                "record": None,
                "tampered": False,
                "error": f"Hashing error: {err_msg}",
            }

        current_hash = hasher_res.get("sha256")
        if not current_hash:
            return {
                "success": False,
                "status": self.STATUS_VERIFICATION_ERROR,
                "current_hash": None,
                "blockchain_verified": False,
                "record": None,
                "tampered": False,
                "error": "Generated candidate hash is empty.",
            }

        # 2. Query the blockchain via Phase 7 BlockchainClient
        verify_res = self.blockchain_client.verify_hash(current_hash)
        if not verify_res.get("success", False):
            # RPC unavailable, contract missing, or network error must be reported as VERIFICATION ERROR, NEVER TAMPER DETECTED
            err_msg = verify_res.get("error") or "Blockchain verification query failed."
            return {
                "success": False,
                "status": self.STATUS_VERIFICATION_ERROR,
                "current_hash": current_hash,
                "blockchain_verified": False,
                "record": None,
                "tampered": False,
                "error": f"Blockchain query error: {err_msg}",
            }

        is_verified = verify_res.get("verified", False)

        # 3. Handle status based on on-chain verification
        if is_verified:
            # Query record details
            record_res = self.blockchain_client.get_record(current_hash)
            record_data = None
            if record_res.get("success", False) and record_res.get("exists", False):
                record_data = {
                    "data_hash": record_res.get("hash"),
                    "registered_by": record_res.get("registered_by"),
                    "timestamp": record_res.get("timestamp"),
                    "exists": True,
                }

            return {
                "success": True,
                "status": self.STATUS_VERIFIED,
                "current_hash": current_hash,
                "blockchain_verified": True,
                "record": record_data,
                "tampered": False,
                "error": None,
            }
        else:
            # Hash does not exist on the blockchain -> TAMPER DETECTED
            return {
                "success": True,
                "status": self.STATUS_TAMPER_DETECTED,
                "current_hash": current_hash,
                "blockchain_verified": False,
                "record": None,
                "tampered": True,
                "error": None,
            }


def verify_candidate_integrity(
    candidate: Any,
    hasher: Optional[CandidateHasher] = None,
    blockchain_client: Optional[BlockchainClient] = None,
) -> Dict[str, Any]:
    """Convenience function for TamperDetector.verify_candidate_integrity."""
    detector = TamperDetector(hasher=hasher, blockchain_client=blockchain_client)
    return detector.verify_candidate_integrity(candidate)
