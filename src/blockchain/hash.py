"""
Deterministic cryptographic hashing module for HH Goa 2026 Task 3.
Provides RFC-8785 compliant canonical JSON serialization, SHA-256 image & record hashing,
and conversions to/from Solidity bytes32 types.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union


def hash_bytes(data: bytes) -> str:
    """Computes standard SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_file(file_path: Union[str, Path]) -> str:
    """Reads a file in binary mode and computes its SHA-256 hex digest."""
    path = Path(file_path)
    with path.open("rb") as f:
        return hash_bytes(f.read())


def build_canonical_record(
    canonical_url: str,
    source_domain: str,
    title: str,
    image_sha256: str,
    discovered_at: str,
    text_excerpt: str = "",
    identified_name: Optional[str] = None,
    social_profiles: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Constructs a standardized, immutable metadata dictionary for discovered post evidence.
    Fields are deterministic and non-volatile to ensure reproducible re-verification.
    """
    record = {
        "canonical_url": canonical_url.strip(),
        "discovered_at": discovered_at.strip(),
        "image_sha256": image_sha256.lower().replace("0x", ""),
        "source_domain": source_domain.strip().lower(),
        "text_excerpt": text_excerpt.strip(),
        "title": title.strip(),
    }
    if identified_name:
        record["identified_name"] = identified_name.strip()
    if social_profiles:
        record["social_profiles"] = {k.strip(): v.strip() for k, v in sorted(social_profiles.items())}
    return record


def canonicalize_json(record: Dict[str, Any]) -> str:
    """
    Serializes a dictionary to a deterministic JSON string following RFC-8785 rules:
    - Keys sorted lexicographically
    - Compact separators without extraneous whitespace (',', ':')
    - UTF-8 compatible encoding
    """
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_canonical_record(record: Dict[str, Any]) -> str:
    """Computes SHA-256 hex digest of the deterministic canonical JSON representation."""
    canonical_str = canonicalize_json(record)
    return hash_bytes(canonical_str.encode("utf-8"))


def to_bytes32(hex_str: Union[str, bytes]) -> bytes:
    """
    Converts a 64-character hex string (with or without '0x' prefix) to 32 raw bytes.
    Ensures compatibility with Solidity bytes32 parameters.
    """
    if isinstance(hex_str, bytes):
        if len(hex_str) == 32:
            return hex_str
        hex_str = hex_str.decode("utf-8")
    
    clean_hex = hex_str.lower().strip()
    if clean_hex.startswith("0x"):
        clean_hex = clean_hex[2:]
    
    if len(clean_hex) != 64:
        raise ValueError(f"Expected 64 hex characters (32 bytes), got length {len(clean_hex)}: '{hex_str}'")
    
    return bytes.fromhex(clean_hex)


def to_hex32(data: Union[bytes, str]) -> str:
    """Formats 32 bytes or hex string as a normalized '0x'-prefixed 64-char lowercase hex string."""
    if isinstance(data, bytes):
        return "0x" + data.hex().lower()
    clean_hex = data.lower().strip()
    if not clean_hex.startswith("0x"):
        clean_hex = "0x" + clean_hex
    return clean_hex
