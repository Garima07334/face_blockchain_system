"""Phase 7: Web3.py & Local Blockchain Integration module.

Provides Web3.py client interface for interacting with the FaceVerification Solidity smart contract
on a local Ethereum-compatible development blockchain.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import os
from dotenv import load_dotenv
from web3 import Web3

# Load environment variables from .env
load_dotenv()

# Standard ABI matching contracts/FaceVerification.sol
DEFAULT_FACE_VERIFICATION_ABI = [
    {
        "inputs": [{"internalType": "bytes32", "name": "dataHash", "type": "bytes32"}],
        "name": "registerHash",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "dataHash", "type": "bytes32"}],
        "name": "verifyHash",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "dataHash", "type": "bytes32"}],
        "name": "getRecord",
        "outputs": [
            {"internalType": "bytes32", "name": "", "type": "bytes32"},
            {"internalType": "address", "name": "", "type": "address"},
            {"internalType": "uint256", "name": "", "type": "uint256"},
            {"internalType": "bool", "name": "", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "bytes32", "name": "dataHash", "type": "bytes32"},
            {"indexed": True, "internalType": "address", "name": "registeredBy", "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
        ],
        "name": "HashRegistered",
        "type": "event",
    },
]


def sha256_hex_to_bytes32(hash_hex: str) -> bytes:
    """Validate SHA-256 hexadecimal digest string and convert to exact 32-byte bytes32 format.

    Args:
        hash_hex: 64-character SHA-256 hexadecimal string (optional '0x' prefix accepted).

    Returns:
        32-byte binary payload (bytes32).

    Raises:
        ValueError: If hash_hex is invalid, non-hex, incorrect length, empty, or zero hash.
    """
    if not hash_hex or not isinstance(hash_hex, str):
        raise ValueError("Hash string must be a non-empty string.")

    cleaned = hash_hex.strip()
    if cleaned.startswith("0x") or cleaned.startswith("0X"):
        cleaned = cleaned[2:]

    if not cleaned:
        raise ValueError("Hash string cannot be empty.")

    if len(cleaned) != 64:
        raise ValueError(
            f"Invalid SHA-256 hex length: expected exactly 64 hexadecimal characters, got {len(cleaned)}."
        )

    if not all(c in "0123456789abcdefABCDEF" for c in cleaned):
        raise ValueError("Invalid SHA-256 hex string: contains non-hexadecimal characters.")

    byte_val = bytes.fromhex(cleaned)
    if byte_val == b"\x00" * 32:
        raise ValueError("Zero hash (0x00...00) is rejected.")

    return byte_val


class BlockchainClient:
    """Web3.py client interface for local Ethereum smart contract interactions."""

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        contract_address: Optional[str] = None,
        abi_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize Web3.py BlockchainClient.

        Args:
            rpc_url: Local RPC endpoint URL. Defaults to BLOCKCHAIN_RPC_URL env var.
            private_key: Local development private key. Defaults to PRIVATE_KEY env var.
            contract_address: Deployed contract address. Defaults to CONTRACT_ADDRESS env var.
            abi_path: Path to local artifact JSON containing ABI & bytecode. Defaults to contracts/FaceVerification.json.
        """
        if rpc_url is not None:
            self.rpc_url = rpc_url.strip()
        else:
            self.rpc_url = (os.getenv("BLOCKCHAIN_RPC_URL") or "http://127.0.0.1:8545").strip()

        if private_key is not None:
            self.private_key = private_key.strip()
        else:
            self.private_key = (os.getenv("PRIVATE_KEY") or "").strip()

        if contract_address is not None:
            self.contract_address = contract_address.strip()
        else:
            self.contract_address = (os.getenv("CONTRACT_ADDRESS") or "").strip()

        # Load ABI and bytecode from local artifact if available
        self.abi = DEFAULT_FACE_VERIFICATION_ABI
        self.bytecode = None

        artifact_path = Path(abi_path) if abi_path else Path(__file__).parent.parent / "contracts" / "FaceVerification.json"
        if artifact_path.is_file():
            try:
                with open(artifact_path, "r", encoding="utf-8") as f:
                    artifact_data = json.load(f)
                    if isinstance(artifact_data, dict):
                        if "abi" in artifact_data:
                            self.abi = artifact_data["abi"]
                        if "bytecode" in artifact_data:
                            self.bytecode = artifact_data["bytecode"]
            except Exception:
                pass

        # Initialize Web3 provider
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        # Derive account address from private key if configured
        self.account = None
        self.sender_address = None
        if self.private_key:
            try:
                self.account = self.w3.eth.account.from_key(self.private_key)
                self.sender_address = self.account.address
            except Exception:
                pass

        # Initialize Web3 contract instance if contract address configured
        self.contract = None
        if self.contract_address and Web3.is_address(self.contract_address):
            try:
                checksum_addr = Web3.to_checksum_address(self.contract_address)
                self.contract = self.w3.eth.contract(address=checksum_addr, abi=self.abi)
            except Exception:
                pass

    def _sanitize_error(self, err: Exception) -> str:
        """Sanitize error messages to guarantee no private key string is ever leaked."""
        err_str = str(err)
        if self.private_key and self.private_key in err_str:
            err_str = err_str.replace(self.private_key, "[REDACTED_PRIVATE_KEY]")
        return err_str

    def is_connected(self) -> bool:
        """Check if connected to local Ethereum RPC node."""
        try:
            return self.w3.is_connected()
        except Exception:
            return False

    def deploy_contract(self) -> Dict[str, Any]:
        """Deploy compiled FaceVerification contract to the local Ethereum development node.

        Returns:
            Structured dictionary with deployment receipt details or explicit error reason.
        """
        if not self.is_connected():
            return {
                "success": False,
                "contract_address": None,
                "transaction_hash": None,
                "error": f"Blockchain RPC node unavailable at {self.rpc_url}.",
            }

        if not self.private_key or not self.account:
            return {
                "success": False,
                "contract_address": None,
                "transaction_hash": None,
                "error": "Missing or invalid PRIVATE_KEY environment variable.",
            }

        if not self.bytecode:
            return {
                "success": False,
                "contract_address": None,
                "transaction_hash": None,
                "error": "Compiled contract bytecode missing from artifact JSON.",
            }

        try:
            balance = self.w3.eth.get_balance(self.sender_address)
            if balance == 0:
                return {
                    "success": False,
                    "contract_address": None,
                    "transaction_hash": None,
                    "error": f"Account {self.sender_address} has zero local ETH balance.",
                }

            contract_factory = self.w3.eth.contract(abi=self.abi, bytecode=self.bytecode)
            nonce = self.w3.eth.get_transaction_count(self.sender_address)
            gas_price = self.w3.eth.gas_price

            tx = contract_factory.constructor().build_transaction(
                {
                    "from": self.sender_address,
                    "nonce": nonce,
                    "gasPrice": gas_price,
                }
            )

            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            raw_tx = getattr(signed_tx, "raw_transaction", getattr(signed_tx, "rawTransaction", None))
            tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            contract_address = receipt.contractAddress
            tx_hash_hex = receipt.transactionHash.hex() if hasattr(receipt.transactionHash, "hex") else str(receipt.transactionHash)
            if not tx_hash_hex.startswith("0x"):
                tx_hash_hex = "0x" + tx_hash_hex

            # Update client instance attributes
            self.contract_address = contract_address
            self.contract = self.w3.eth.contract(address=contract_address, abi=self.abi)

            return {
                "success": True,
                "contract_address": contract_address,
                "transaction_hash": tx_hash_hex,
                "block_number": receipt.blockNumber,
                "deployed_by": self.sender_address,
                "error": None,
            }

        except Exception as err:
            return {
                "success": False,
                "contract_address": None,
                "transaction_hash": None,
                "error": f"Contract deployment failed: {self._sanitize_error(err)}",
            }

    def register_hash(self, hash_hex: str) -> Dict[str, Any]:
        """Register a Phase 5 SHA-256 metadata fingerprint on the local blockchain.

        Args:
            hash_hex: 64-character hexadecimal SHA-256 digest string.

        Returns:
            Structured response dictionary.
        """
        # 1. Validate hash conversion
        try:
            bytes32_val = sha256_hex_to_bytes32(hash_hex)
            clean_hex = bytes32_val.hex().lower()
        except ValueError as err:
            return {
                "success": False,
                "hash": str(hash_hex),
                "bytes32": None,
                "contract_address": self.contract_address or None,
                "transaction_hash": None,
                "block_number": None,
                "registered_by": self.sender_address,
                "timestamp": None,
                "error": str(err),
            }

        # 2. Check prerequisites (configuration before network I/O)
        if not self.private_key or not self.account:
            return {
                "success": False,
                "hash": clean_hex,
                "bytes32": "0x" + clean_hex,
                "contract_address": self.contract_address or None,
                "transaction_hash": None,
                "block_number": None,
                "registered_by": None,
                "timestamp": None,
                "error": "Missing or invalid PRIVATE_KEY environment variable.",
            }

        if not self.contract_address or not self.contract:
            return {
                "success": False,
                "hash": clean_hex,
                "bytes32": "0x" + clean_hex,
                "contract_address": self.contract_address or None,
                "transaction_hash": None,
                "block_number": None,
                "registered_by": self.sender_address,
                "timestamp": None,
                "error": "Missing or invalid CONTRACT_ADDRESS environment variable.",
            }

        if not self.is_connected():
            return {
                "success": False,
                "hash": clean_hex,
                "bytes32": "0x" + clean_hex,
                "contract_address": self.contract_address or None,
                "transaction_hash": None,
                "block_number": None,
                "registered_by": self.sender_address,
                "timestamp": None,
                "error": f"Blockchain RPC node unavailable at {self.rpc_url}.",
            }

        # 3. Build, sign, and broadcast transaction
        try:
            balance = self.w3.eth.get_balance(self.sender_address)
            if balance == 0:
                return {
                    "success": False,
                    "hash": clean_hex,
                    "bytes32": "0x" + clean_hex,
                    "contract_address": self.contract_address,
                    "transaction_hash": None,
                    "block_number": None,
                    "registered_by": self.sender_address,
                    "timestamp": None,
                    "error": f"Account {self.sender_address} has zero local ETH balance.",
                }

            nonce = self.w3.eth.get_transaction_count(self.sender_address)
            gas_price = self.w3.eth.gas_price

            tx = self.contract.functions.registerHash(bytes32_val).build_transaction(
                {
                    "from": self.sender_address,
                    "nonce": nonce,
                    "gasPrice": gas_price,
                }
            )

            signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
            raw_tx = getattr(signed_tx, "raw_transaction", getattr(signed_tx, "rawTransaction", None))
            tx_hash = self.w3.eth.send_raw_transaction(raw_tx)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            block = self.w3.eth.get_block(receipt.blockNumber)
            block_timestamp = block.get("timestamp", 0)

            tx_hash_hex = receipt.transactionHash.hex() if hasattr(receipt.transactionHash, "hex") else str(receipt.transactionHash)
            if not tx_hash_hex.startswith("0x"):
                tx_hash_hex = "0x" + tx_hash_hex

            return {
                "success": True,
                "hash": clean_hex,
                "bytes32": "0x" + clean_hex,
                "contract_address": self.contract_address,
                "transaction_hash": tx_hash_hex,
                "block_number": receipt.blockNumber,
                "registered_by": self.sender_address,
                "timestamp": block_timestamp,
                "error": None,
            }

        except Exception as err:
            return {
                "success": False,
                "hash": clean_hex,
                "bytes32": "0x" + clean_hex,
                "contract_address": self.contract_address,
                "transaction_hash": None,
                "block_number": None,
                "registered_by": self.sender_address,
                "timestamp": None,
                "error": f"Transaction execution failed: {self._sanitize_error(err)}",
            }

    def verify_hash(self, hash_hex: str) -> Dict[str, Any]:
        """Verify whether a Phase 5 SHA-256 metadata fingerprint exists on the blockchain.

        Args:
            hash_hex: 64-character hexadecimal SHA-256 digest string.

        Returns:
            Structured response dictionary.
        """
        try:
            bytes32_val = sha256_hex_to_bytes32(hash_hex)
            clean_hex = bytes32_val.hex().lower()
        except ValueError as err:
            return {
                "success": False,
                "hash": str(hash_hex),
                "bytes32": None,
                "verified": False,
                "error": str(err),
            }

        if not self.contract_address or not self.contract:
            return {
                "success": False,
                "hash": clean_hex,
                "bytes32": "0x" + clean_hex,
                "verified": False,
                "error": "Missing or invalid CONTRACT_ADDRESS environment variable.",
            }

        if not self.is_connected():
            return {
                "success": False,
                "hash": clean_hex,
                "bytes32": "0x" + clean_hex,
                "verified": False,
                "error": f"Blockchain RPC node unavailable at {self.rpc_url}.",
            }

        try:
            is_verified = self.contract.functions.verifyHash(bytes32_val).call()
            return {
                "success": True,
                "hash": clean_hex,
                "bytes32": "0x" + clean_hex,
                "verified": bool(is_verified),
                "error": None,
            }
        except Exception as err:
            return {
                "success": False,
                "hash": clean_hex,
                "bytes32": "0x" + clean_hex,
                "verified": False,
                "error": f"Contract call failed: {self._sanitize_error(err)}",
            }

    def get_record(self, hash_hex: str) -> Dict[str, Any]:
        """Retrieve recorded on-chain metadata for a SHA-256 fingerprint.

        Args:
            hash_hex: 64-character hexadecimal SHA-256 digest string.

        Returns:
            Structured response dictionary.
        """
        try:
            bytes32_val = sha256_hex_to_bytes32(hash_hex)
            clean_hex = bytes32_val.hex().lower()
        except ValueError as err:
            return {
                "success": False,
                "hash": str(hash_hex),
                "registered_by": None,
                "timestamp": None,
                "exists": False,
                "error": str(err),
            }

        if not self.contract_address or not self.contract:
            return {
                "success": False,
                "hash": clean_hex,
                "registered_by": None,
                "timestamp": None,
                "exists": False,
                "error": "Missing or invalid CONTRACT_ADDRESS environment variable.",
            }

        if not self.is_connected():
            return {
                "success": False,
                "hash": clean_hex,
                "registered_by": None,
                "timestamp": None,
                "exists": False,
                "error": f"Blockchain RPC node unavailable at {self.rpc_url}.",
            }

        try:
            data_hash, registered_by, timestamp, exists = self.contract.functions.getRecord(bytes32_val).call()
            return {
                "success": True,
                "hash": clean_hex,
                "registered_by": registered_by if exists else None,
                "timestamp": timestamp if exists else 0,
                "exists": bool(exists),
                "error": None,
            }
        except Exception as err:
            return {
                "success": False,
                "hash": clean_hex,
                "registered_by": None,
                "timestamp": None,
                "exists": False,
                "error": f"Contract call failed: {self._sanitize_error(err)}",
            }
