"""
Unit tests for cryptographic hashing and blockchain attestation layer.
Verifies RFC-8785 canonical JSON determinism, SHA-256 digests, bytes32 conversion,
and on-chain record retrieval & tamper detection.
"""

import pytest
from src.blockchain.contract import MockBlockchainClient
from src.blockchain.hash import (
    build_canonical_record,
    canonicalize_json,
    hash_bytes,
    hash_canonical_record,
    to_bytes32,
    to_hex32,
)


def test_hash_bytes():
    data = b"hello world"
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert hash_bytes(data) == expected


def test_canonicalize_json_order_invariance():
    """Verify that different key insertion orders produce identical canonical JSON strings."""
    dict_a = {
        "title": "NVIDIA Keynote",
        "canonical_url": "https://instagram.com/p/123",
        "source_domain": "instagram.com",
        "discovered_at": "2026-09-06T12:00:00Z",
        "image_sha256": "abc123def456",
        "text_excerpt": "Jensen Huang on stage",
    }
    dict_b = {
        "text_excerpt": "Jensen Huang on stage",
        "image_sha256": "abc123def456",
        "discovered_at": "2026-09-06T12:00:00Z",
        "source_domain": "instagram.com",
        "canonical_url": "https://instagram.com/p/123",
        "title": "NVIDIA Keynote",
    }

    canon_a = canonicalize_json(dict_a)
    canon_b = canonicalize_json(dict_b)

    assert canon_a == canon_b
    assert " " not in canon_a.split(":")[1][:2]  # No whitespace after separator
    assert hash_canonical_record(dict_a) == hash_canonical_record(dict_b)


def test_to_bytes32_and_to_hex32():
    hex_str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    b32 = to_bytes32(hex_str)
    assert len(b32) == 32
    assert to_hex32(b32) == "0x" + hex_str

    # Accepts 0x-prefixed hex string
    b32_prefixed = to_bytes32("0x" + hex_str)
    assert b32 == b32_prefixed


def test_to_bytes32_invalid_length():
    with pytest.raises(ValueError):
        to_bytes32("deadbeef")


def test_mock_blockchain_attestation_and_retrieval():
    client = MockBlockchainClient()
    content_hash = "0xe3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    record_hash = "0x8a7f92b490c21345d8b7612f08a8e8412c1b2f901234abcd5678ef0123456789"

    receipt = client.attest(content_hash, record_hash)
    assert receipt.tx_hash.startswith("0x")
    assert receipt.block_number > 0
    assert receipt.content_hash.lower() == content_hash.lower()
    assert receipt.record_hash.lower() == record_hash.lower()
    assert receipt.is_simulated is True

    record = client.get_record(record_hash)
    assert record["contentHash"].lower() == content_hash.lower()
    assert record["recordHash"].lower() == record_hash.lower()
    assert record["timestamp"] > 0


def test_mock_blockchain_tamper_detection():
    """Verify that tampering with a record hash causes on-chain retrieval failure."""
    client = MockBlockchainClient()
    content_hash = "0xe3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    record_hash = "0x8a7f92b490c21345d8b7612f08a8e8412c1b2f901234abcd5678ef0123456789"

    client.attest(content_hash, record_hash)

    # Invert last byte of record_hash
    tampered_hash = record_hash[:-2] + "ff"
    with pytest.raises(KeyError):
        client.get_record(tampered_hash)
