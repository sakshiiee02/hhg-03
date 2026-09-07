"""
Run evidence export and audit logging ledger.
Persists immutable artifacts for every execution under runs/run_<timestamp>_<id>/.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


class EvidenceLedger:
    """Manages local persistence of run audit logs, candidate evidence, and canonical records."""

    def __init__(self, run_id: Optional[str] = None):
        if not run_id:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{ts}"
        self.run_id = run_id

        root_dir = Path(__file__).resolve().parent.parent.parent
        self.run_dir = root_dir / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

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
