"""Tests for contracts/FaceVerification.sol Phase 6 Smart Contract and core/blockchain.py Phase 7 Web3.py client."""

from pathlib import Path
import re
import unittest
from core.blockchain import BlockchainClient, sha256_hex_to_bytes32


class TestFaceVerificationContractStatic(unittest.TestCase):
    """Static source code analysis test suite for FaceVerification Solidity smart contract."""

    @classmethod
    def setUpClass(cls) -> None:
        """Locate and read contracts/FaceVerification.sol."""
        cls.contract_path = Path(__file__).parent.parent / "contracts" / "FaceVerification.sol"
        cls.assertTrue(cls.contract_path.is_file(), f"Contract file missing at {cls.contract_path}")
        with open(cls.contract_path, "r", encoding="utf-8") as f:
            cls.sol_code = f.read()

    def test_1_contract_file_exists(self) -> None:
        """Test 1: Verify contracts/FaceVerification.sol exists."""
        self.assertTrue(self.contract_path.exists())

    def test_2_contract_name_is_face_verification(self) -> None:
        """Test 2: Verify contract name is FaceVerification."""
        self.assertRegex(self.sol_code, r"\bcontract\s+FaceVerification\b")

    def test_3_solidity_pragma_version(self) -> None:
        """Test 3: Verify pragma solidity uses ^0.8.x."""
        match = re.search(r"pragma\s+solidity\s+\^0\.8\.\d+;", self.sol_code)
        self.assertIsNotNone(match, "Solidity pragma must be ^0.8.x")

    def test_4_data_hash_uses_bytes32(self) -> None:
        """Test 4: Verify dataHash uses bytes32 data type."""
        self.assertIn("bytes32 dataHash", self.sol_code)
        self.assertIn("bytes32 => Record", self.sol_code)

    def test_5_register_hash_function_exists(self) -> None:
        """Test 5: Verify registerHash function exists."""
        self.assertRegex(self.sol_code, r"function\s+registerHash\s*\(\s*bytes32\s+dataHash\s*\)")

    def test_6_verify_hash_function_exists(self) -> None:
        """Test 6: Verify verifyHash function exists."""
        self.assertRegex(self.sol_code, r"function\s+verifyHash\s*\(\s*bytes32\s+dataHash\s*\)")

    def test_7_get_record_function_exists(self) -> None:
        """Test 7: Verify getRecord function exists."""
        self.assertRegex(self.sol_code, r"function\s+getRecord\s*\(\s*bytes32\s+dataHash\s*\)")

    def test_8_duplicate_registration_protection(self) -> None:
        """Test 8: Verify duplicate hash registration protection exists."""
        self.assertIn("!records[dataHash].exists", self.sol_code)
        self.assertIn("Hash already registered", self.sol_code)

    def test_9_zero_hash_rejection(self) -> None:
        """Test 9: Verify zero hash rejection exists."""
        self.assertIn("dataHash != bytes32(0)", self.sol_code)
        self.assertIn("Zero hash rejected", self.sol_code)

    def test_10_hash_registered_event_exists(self) -> None:
        """Test 10: Verify HashRegistered event exists."""
        self.assertRegex(self.sol_code, r"event\s+HashRegistered\b")

    def test_11_event_contains_bytes32_indexed_data_hash(self) -> None:
        """Test 11: Verify HashRegistered event contains 'bytes32 indexed dataHash'."""
        self.assertIn("bytes32 indexed dataHash", self.sol_code)

    def test_12_event_contains_address_indexed_registered_by(self) -> None:
        """Test 12: Verify HashRegistered event contains 'address indexed registeredBy'."""
        self.assertIn("address indexed registeredBy", self.sol_code)

    def test_13_event_contains_uint256_timestamp(self) -> None:
        """Test 13: Verify HashRegistered event contains 'uint256 timestamp'."""
        self.assertIn("uint256 timestamp", self.sol_code)

    def test_14_block_timestamp_stored(self) -> None:
        """Test 14: Verify block.timestamp is stored in record."""
        self.assertIn("timestamp: block.timestamp", self.sol_code)

    def test_15_msg_sender_stored(self) -> None:
        """Test 15: Verify msg.sender is stored in record."""
        self.assertIn("registeredBy: msg.sender", self.sol_code)

    def test_16_exists_flag_stored(self) -> None:
        """Test 16: Verify exists boolean flag is stored in record."""
        self.assertIn("exists: true", self.sol_code)
        self.assertIn("bool exists;", self.sol_code)

    def test_17_no_raw_face_image_storage(self) -> None:
        """Test 17: Verify contract contains no raw face image storage fields."""
        code_without_comments = re.sub(r"/\*.*?\*/|//.*", "", self.sol_code, flags=re.DOTALL)
        self.assertNotIn("image", code_without_comments.lower())
        self.assertNotIn("jpg", code_without_comments.lower())
        self.assertNotIn("png", code_without_comments.lower())

    def test_18_no_embedding_vector_storage(self) -> None:
        """Test 18: Verify contract contains no face embedding vector storage fields."""
        code_without_comments = re.sub(r"/\*.*?\*/|//.*", "", self.sol_code, flags=re.DOTALL)
        self.assertNotIn("embedding", code_without_comments.lower())
        self.assertNotIn("vector", code_without_comments.lower())
        self.assertNotIn("float", code_without_comments.lower())

    def test_19_no_api_private_secret_key_storage(self) -> None:
        """Test 19: Verify contract contains no API key, private key, or secret storage."""
        code_without_comments = re.sub(r"/\*.*?\*/|//.*", "", self.sol_code, flags=re.DOTALL)
        self.assertNotIn("api_key", code_without_comments.lower())
        self.assertNotIn("privatekey", code_without_comments.lower())
        self.assertNotIn("secret", code_without_comments.lower())

    def test_20_no_identity_claims_or_assertions(self) -> None:
        """Test 20: Verify contract contains no identity claims or personal identity assertions."""
        code_without_comments = re.sub(r"/\*.*?\*/|//.*", "", self.sol_code, flags=re.DOTALL)
        self.assertNotIn("person_id", code_without_comments.lower())
        self.assertNotIn("ssn", code_without_comments.lower())
        self.assertNotIn("passport", code_without_comments.lower())


class TestBlockchainClientUnit(unittest.TestCase):
    """Unit test suite for Phase 7 core/blockchain.py Web3.py client."""

    def setUp(self) -> None:
        """Initialize test parameters."""
        self.valid_sha256_hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        self.dummy_private_key = "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

    def test_sha256_hex_to_bytes32_valid(self) -> None:
        """Test conversion of a valid 64-char SHA-256 hex string to bytes32."""
        res_bytes = sha256_hex_to_bytes32(self.valid_sha256_hex)
        self.assertEqual(len(res_bytes), 32)
        self.assertEqual(res_bytes.hex(), self.valid_sha256_hex)

    def test_sha256_hex_to_bytes32_prefix_0x(self) -> None:
        """Test conversion of a valid '0x'-prefixed SHA-256 hex string."""
        prefixed_hex = "0x" + self.valid_sha256_hex
        res_bytes = sha256_hex_to_bytes32(prefixed_hex)
        self.assertEqual(len(res_bytes), 32)
        self.assertEqual(res_bytes.hex(), self.valid_sha256_hex)

    def test_sha256_hex_to_bytes32_invalid_hex(self) -> None:
        """Test rejection of invalid non-hexadecimal characters."""
        invalid_hex = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85Z"
        with self.assertRaises(ValueError) as ctx:
            sha256_hex_to_bytes32(invalid_hex)
        self.assertIn("non-hexadecimal", str(ctx.exception).lower())

    def test_sha256_hex_to_bytes32_invalid_length(self) -> None:
        """Test rejection of hex strings with length != 64."""
        short_hex = "e3b0c44298fc1c14"
        with self.assertRaises(ValueError) as ctx:
            sha256_hex_to_bytes32(short_hex)
        self.assertIn("64", str(ctx.exception))

    def test_sha256_hex_to_bytes32_empty(self) -> None:
        """Test rejection of empty or None hash strings."""
        with self.assertRaises(ValueError):
            sha256_hex_to_bytes32("")
        with self.assertRaises(ValueError):
            sha256_hex_to_bytes32(None)  # type: ignore

    def test_sha256_hex_to_bytes32_zero_hash(self) -> None:
        """Test rejection of zero hash (0x00...00)."""
        zero_hex = "0" * 64
        with self.assertRaises(ValueError) as ctx:
            sha256_hex_to_bytes32(zero_hex)
        self.assertIn("zero hash", str(ctx.exception).lower())

    def test_missing_rpc_configuration(self) -> None:
        """Test that missing/unreachable RPC handles requests gracefully with error dictionary."""
        client = BlockchainClient(rpc_url="http://127.0.0.1:999999")  # Unreachable port
        self.assertFalse(client.is_connected())

        res_reg = client.register_hash(self.valid_sha256_hex)
        self.assertFalse(res_reg["success"])
        self.assertIsNotNone(res_reg["error"])

        res_ver = client.verify_hash(self.valid_sha256_hex)
        self.assertFalse(res_ver["success"])
        self.assertFalse(res_ver["verified"])

        res_rec = client.get_record(self.valid_sha256_hex)
        self.assertFalse(res_rec["success"])
        self.assertFalse(res_rec["exists"])

    def test_missing_private_key(self) -> None:
        """Test that missing private key returns explicit error without crashing."""
        client = BlockchainClient(private_key="", contract_address="0x1111111111111111111111111111111111111111")
        res = client.register_hash(self.valid_sha256_hex)
        self.assertFalse(res["success"])
        self.assertIn("private_key", res["error"].lower())

    def test_missing_contract_address(self) -> None:
        """Test that missing contract address returns explicit error without crashing."""
        client = BlockchainClient(private_key=self.dummy_private_key, contract_address="")
        res_reg = client.register_hash(self.valid_sha256_hex)
        self.assertFalse(res_reg["success"])
        self.assertIn("contract_address", res_reg["error"].lower())

        res_ver = client.verify_hash(self.valid_sha256_hex)
        self.assertFalse(res_ver["success"])

    def test_private_key_security(self) -> None:
        """Test that private key is NEVER leaked in error messages or logs."""
        client = BlockchainClient(private_key=self.dummy_private_key)
        # Induce an error and check sanitized string output
        err = Exception(f"Failed transaction with key {self.dummy_private_key}")
        sanitized = client._sanitize_error(err)

        self.assertNotIn(self.dummy_private_key, sanitized)
        self.assertIn("[REDACTED_PRIVATE_KEY]", sanitized)

    def test_deploy_contract_compiler_missing(self) -> None:
        """Test that deploy_contract returns informative error when bytecode is missing."""
        client = BlockchainClient()
        client.bytecode = None
        res = client.deploy_contract()

        self.assertFalse(res["success"])
        self.assertIsNotNone(res["error"])
        self.assertIn("bytecode", res["error"].lower())


if __name__ == "__main__":
    unittest.main()
