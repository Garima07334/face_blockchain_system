// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FaceVerification
 * @dev Smart Contract for registering and verifying SHA-256 metadata integrity fingerprints.
 * 
 * PURPOSE & INTENT:
 * - This contract registers and verifies 32-byte (bytes32) SHA-256 metadata integrity fingerprints
 *   produced by canonicalizing external visual-search candidate results.
 * - PRIVACY & SECURITY BOUNDS:
 *   - No raw face images are stored or processed on-chain.
 *   - No face embeddings or biometric vectors are stored on-chain.
 *   - No API keys, private keys, passwords, or secrets are stored.
 *   - No personal identity data or identity claims are encoded into this contract.
 * - DISCLAIMER:
 *   - The registered SHA-256 fingerprint represents external metadata integrity only.
 *   - The blockchain record is NOT proof of human identity.
 */
contract FaceVerification {

    /// @dev Structure representing an on-chain metadata integrity record.
    struct Record {
        bytes32 dataHash;
        address registeredBy;
        uint256 timestamp;
        bool exists;
    }

    /// @dev Internal mapping from 32-byte SHA-256 digest to Record.
    mapping(bytes32 => Record) private records;

    /// @dev Event emitted whenever a new SHA-256 metadata fingerprint is registered.
    event HashRegistered(
        bytes32 indexed dataHash,
        address indexed registeredBy,
        uint256 timestamp
    );

    /**
     * @dev Register a 32-byte SHA-256 metadata fingerprint on-chain.
     * @param dataHash 32-byte SHA-256 digest of the canonical external result metadata.
     */
    function registerHash(bytes32 dataHash) public {
        require(
            dataHash != bytes32(0),
            "FaceVerification: Zero hash rejected"
        );

        require(
            !records[dataHash].exists,
            "FaceVerification: Hash already registered"
        );

        records[dataHash] = Record({
            dataHash: dataHash,
            registeredBy: msg.sender,
            timestamp: block.timestamp,
            exists: true
        });

        emit HashRegistered(
            dataHash,
            msg.sender,
            block.timestamp
        );
    }

    /**
     * @dev Verify whether a given SHA-256 metadata fingerprint has been registered on-chain.
     * @param dataHash 32-byte SHA-256 digest to verify.
     * @return True if the hash exists on-chain, false otherwise.
     */
    function verifyHash(bytes32 dataHash) public view returns (bool) {
        return records[dataHash].exists;
    }

    /**
     * @dev Retrieve the stored record metadata for a registered SHA-256 fingerprint.
     * @param dataHash 32-byte SHA-256 digest to query.
     * @return dataHash The 32-byte SHA-256 digest.
     * @return registeredBy Address that registered the hash.
     * @return timestamp Block timestamp when registered.
     * @return exists Boolean flag indicating if record exists on-chain.
     */
    function getRecord(bytes32 dataHash)
        public
        view
        returns (
            bytes32,
            address,
            uint256,
            bool
        )
    {
        Record memory record = records[dataHash];
        return (
            record.dataHash,
            record.registeredBy,
            record.timestamp,
            record.exists
        );
    }
}
