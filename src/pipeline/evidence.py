"""
Run evidence export and audit logging ledger.
Persists immutable forensic artifacts under runs/run_<timestamp>_<id>/ and evidence/.
Includes standalone cryptographic re-verification against Ethereum Sepolia / simulated EVM registry.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from src.blockchain.contract import BaseBlockchainClient, get_blockchain_client
from src.blockchain.hash import (
    hash_bytes,
    hash_canonical_record,
    to_hex32,
)

logger = logging.getLogger(__name__)


class EvidenceLedger:
    """Manages local persistence of run audit logs, candidate evidence, and canonical records."""

    def __init__(self, run_id: Optional[str] = None):
        if not run_id:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{ts}"
        self.run_id = run_id

        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.run_dir = self.root_dir / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = self.run_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

    def save_input_crop(self, crop_bgr: np.ndarray) -> Path:
        dest = self.run_dir / "input_face.jpg"
        cv2.imwrite(str(dest), crop_bgr)
        return dest

    def save_matched_media(self, media_bytes: bytes) -> Path:
        dest = self.run_dir / "matched_media.jpg"
        dest.write_bytes(media_bytes)
        return dest

    def save_canonical_record(self, record: Dict[str, Any]) -> Path:
        dest = self.run_dir / "canonical_record.json"
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        return dest

    def save_evidence_report(self, report_data: Dict[str, Any]) -> Path:
        dest = self.run_dir / "evidence.json"
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        return dest

    def save_evidence_bundle(
        self,
        matched_image_bytes: bytes,
        canonical_record: Dict[str, Any],
        attestation_receipt: Any,
    ) -> Path:
        """
        Saves a self-contained forensic verification bundle containing raw matched media
        and cryptographic attestation metadata.
        """
        # 1. Raw image bytes
        bin_path = self.evidence_dir / "matched_image.bin"
        bin_path.write_bytes(matched_image_bytes)

        # Also save matched_media.jpg for human viewing
        media_jpg = self.evidence_dir / "matched_image.jpg"
        media_jpg.write_bytes(matched_image_bytes)

        # 2. Compute canonical fingerprints
        meta_sha = hash_canonical_record(canonical_record)
        img_sha = hash_bytes(matched_image_bytes)

        # 3. Structure metadata bundle
        metadata_payload = {
            "evidence_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "record": canonical_record,
            "fingerprints": {
                "metadata_sha256": meta_sha,
                "image_sha256": img_sha,
            },
            "blockchain": {
                "network": getattr(attestation_receipt, "network", "Ethereum Sepolia"),
                "transaction": getattr(attestation_receipt, "tx_hash", ""),
                "block_number": getattr(attestation_receipt, "block_number", 0),
                "submitter": getattr(attestation_receipt, "submitter", ""),
                "is_simulated": getattr(attestation_receipt, "is_simulated", False),
                "explorer_url": getattr(attestation_receipt, "explorer_url", None),
            },
        }

        meta_path = self.evidence_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"Forensic evidence bundle saved: {self.evidence_dir}")
        return self.evidence_dir


def verify_evidence_bundle(
    bundle_target: Union[str, Path],
    blockchain_client: Optional[BaseBlockchainClient] = None,
) -> Dict[str, Any]:
    """
    Independently inspects and verifies a saved forensic evidence bundle against Ethereum.
    Recomputes SHA-256 digests of saved media and canonical JSON metadata,
    and cross-examines the on-chain registry for cryptographic integrity.
    
    Returns a structured verification report with exit_code:
      - 0: Verified (authentic & untampered)
      - 6: Cryptographic tampering detected
    """
    root_dir = Path(__file__).resolve().parent.parent.parent
    target_path = Path(bundle_target)

    # 1. Resolve path to evidence directory and metadata.json
    if not target_path.is_absolute():
        # Check if it's a run_id like "run_20260907_..."
        if (root_dir / "runs" / str(bundle_target)).exists():
            target_path = root_dir / "runs" / str(bundle_target)
        elif (root_dir / str(bundle_target)).exists():
            target_path = root_dir / str(bundle_target)

    if target_path.is_file() and target_path.name == "metadata.json":
        meta_file = target_path
        bundle_dir = target_path.parent
    elif target_path.is_dir():
        if (target_path / "evidence" / "metadata.json").is_file():
            bundle_dir = target_path / "evidence"
            meta_file = bundle_dir / "metadata.json"
        elif (target_path / "metadata.json").is_file():
            bundle_dir = target_path
            meta_file = bundle_dir / "metadata.json"
        else:
            raise FileNotFoundError(f"metadata.json not found in directory: {target_path}")
    else:
        raise FileNotFoundError(f"Invalid evidence bundle path: {bundle_target}")

    # 2. Load metadata.json
    payload = json.loads(meta_file.read_text(encoding="utf-8"))
    record = payload.get("record", {})
    fingerprints = payload.get("fingerprints", {})
    blockchain_info = payload.get("blockchain", {})
    evidence_id = payload.get("evidence_id", bundle_dir.parent.name if bundle_dir.name == "evidence" else bundle_dir.name)

    # 3. Locate image media
    image_file = None
    for cand_name in ("matched_image.bin", "matched_image.jpg", "matched_media.jpg"):
        candidate_loc = bundle_dir / cand_name
        if candidate_loc.is_file():
            image_file = candidate_loc
            break
        parent_loc = bundle_dir.parent / cand_name
        if parent_loc.is_file():
            image_file = parent_loc
            break

    image_bytes = image_file.read_bytes() if image_file and image_file.exists() else b""

    # 4. Recompute independent hashes
    recomputed_meta_sha = hash_canonical_record(record)
    recomputed_img_sha = hash_bytes(image_bytes) if image_bytes else ""

    saved_meta_sha = str(fingerprints.get("metadata_sha256", "")).lower().replace("0x", "")
    saved_img_sha = str(fingerprints.get("image_sha256", "")).lower().replace("0x", "")

    local_meta_consistent = (recomputed_meta_sha.lower() == saved_meta_sha)
    local_img_consistent = (recomputed_img_sha.lower() == saved_img_sha) if saved_img_sha else bool(image_bytes)

    # 5. Query Blockchain
    if blockchain_client is None:
        from src.config import get_config
        cfg = get_config()
        blockchain_client = get_blockchain_client(
            rpc_url=cfg.sepolia_rpc_url,
            private_key=cfg.sepolia_private_key,
            contract_address=cfg.sepolia_contract_address or None,
            force_simulated=cfg.force_simulated_blockchain,
        )

    on_chain_record = None
    on_chain_error = None
    try:
        on_chain_record = blockchain_client.get_record(recomputed_meta_sha)
    except Exception as e:
        on_chain_error = str(e)
        logger.warning(f"Could not retrieve on-chain record for {recomputed_meta_sha}: {e}")

    on_chain_cnt = ""
    on_chain_rec = ""
    if on_chain_record:
        on_chain_cnt = str(on_chain_record.get("contentHash", "")).lower().replace("0x", "")
        on_chain_rec = str(on_chain_record.get("recordHash", "")).lower().replace("0x", "")

    # Compare hashes
    meta_match = bool(local_meta_consistent and on_chain_rec and recomputed_meta_sha.lower() == on_chain_rec)
    img_match = bool(local_img_consistent and on_chain_cnt and recomputed_img_sha.lower() == on_chain_cnt)

    # If simulated mode and not in live chain records, check local consistency
    is_simulated = blockchain_info.get("is_simulated", False)
    if not on_chain_record and is_simulated and local_meta_consistent and local_img_consistent:
        meta_match = True
        img_match = True

    verified = meta_match and img_match
    tampered = not verified

    return {
        "success": True,
        "verified": verified,
        "tampered": tampered,
        "exit_code": 0 if verified else 6,
        "evidence_id": evidence_id,
        "bundle_path": str(bundle_dir),
        "metadata_hash_match": meta_match,
        "image_hash_match": img_match,
        "local_metadata_consistent": local_meta_consistent,
        "local_image_consistent": local_img_consistent,
        "stored_metadata_hash": saved_meta_sha,
        "recomputed_metadata_hash": recomputed_meta_sha,
        "stored_image_hash": saved_img_sha,
        "recomputed_image_hash": recomputed_img_sha,
        "on_chain_record_hash": on_chain_rec or (saved_meta_sha if is_simulated else "unrecorded"),
        "on_chain_content_hash": on_chain_cnt or (saved_img_sha if is_simulated else "unrecorded"),
        "transaction": blockchain_info.get("transaction"),
        "block_number": blockchain_info.get("block_number"),
        "network": blockchain_info.get("network", "Ethereum Sepolia"),
        "submitter": blockchain_info.get("submitter"),
        "explorer_url": blockchain_info.get("explorer_url"),
        "on_chain_error": on_chain_error,
        "record": record,
    }
