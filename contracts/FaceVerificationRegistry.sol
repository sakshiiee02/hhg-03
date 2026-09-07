// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FaceVerificationRegistry
 * @notice Stores cryptographic attestation fingerprints for verified public web and social posts.
 * @dev Designed for HH Goa 2026 — Task 3.
 */
contract FaceVerificationRegistry {
    struct Record {
        bytes32 contentHash;  // SHA-256 hash of raw image media bytes
        bytes32 recordHash;   // SHA-256 hash of canonical RFC-8785 JSON metadata record
        uint256 timestamp;    // Block timestamp of attestation
        address submitter;    // Submitting wallet address
    }

    // Mapping from recordHash to on-chain Record
    mapping(bytes32 => Record) public records;

    // Event emitted upon successful attestation
    event Attested(
        bytes32 indexed recordHash,
        bytes32 indexed contentHash,
        address indexed submitter,
        uint256 timestamp
    );

    /**
     * @notice Records an immutable cryptographic fingerprint on-chain.
     * @param contentHash SHA-256 fingerprint of the verified media bytes.
     * @param recordHash SHA-256 fingerprint of the canonical post record.
     */
    function attest(bytes32 contentHash, bytes32 recordHash) external {
        require(records[recordHash].timestamp == 0, "Record already attested");
        records[recordHash] = Record(
            contentHash,
            recordHash,
            block.timestamp,
            msg.sender
        );
        emit Attested(recordHash, contentHash, msg.sender, block.timestamp);
    }

    /**
     * @notice Retrieves an attested record by its canonical record hash.
     * @param recordHash SHA-256 fingerprint of the canonical record.
     * @return The stored Record struct.
     */
    function getRecord(bytes32 recordHash) external view returns (Record memory) {
        require(records[recordHash].timestamp != 0, "Record not found");
        return records[recordHash];
    }
}
