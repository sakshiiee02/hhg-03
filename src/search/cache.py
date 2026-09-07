"""
Persistent disk-backed search cache for reverse image discovery.
Caches visual candidate discoveries keyed by query image SHA-256 digest,
saving external SerpApi search credits and eliminating network latency on repeat runs.
"""

from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Union

from src.search.base import CandidateResult

logger = logging.getLogger(__name__)


class PersistentSearchCache:
    """
    Disk-backed cache for visual candidate search results.
    Preserves discovered web candidate lists across server restarts and test runs.
    """

    CACHE_VERSION = 1

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            root_dir = Path(__file__).resolve().parent.parent.parent
            cache_dir = root_dir / "runs" / "cache" / "searches"
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @staticmethod
    def compute_image_digest(image_input: Union[bytes, Path, str]) -> str:
        """Computes SHA-256 hex digest for an image path or raw bytes."""
        if isinstance(image_input, (bytes, bytearray)):
            return hashlib.sha256(image_input).hexdigest()
        
        path = Path(image_input)
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _path_for(self, digest: str) -> Path:
        return self.root / f"{digest}.json"

    def get(self, image_input: Union[bytes, Path, str]) -> Optional[List[CandidateResult]]:
        """Retrieves cached candidates for the given image if available."""
        try:
            digest = self.compute_image_digest(image_input)
            path = self._path_for(digest)
            with self._lock:
                if not path.is_file():
                    self.misses += 1
                    return None
                
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("version") != self.CACHE_VERSION:
                    return None
                
                raw_candidates = payload.get("candidates", [])
                candidates = []
                for c in raw_candidates:
                    candidates.append(CandidateResult(
                        rank=int(c.get("rank", 1)),
                        page_url=str(c.get("page_url", "")),
                        image_url=c.get("image_url"),
                        source_domain=str(c.get("source_domain", "")),
                        title=c.get("title"),
                        provider=str(c.get("provider", "cached")),
                    ))
                self.hits += 1
                logger.info(f"Search Cache HIT for image {digest[:12]} ({len(candidates)} candidates)")
                return candidates
        except Exception as e:
            logger.debug(f"Search cache read error: {e}")
            self.misses += 1
            return None

    def put(self, image_input: Union[bytes, Path, str], candidates: List[CandidateResult]) -> None:
        """Stores candidate discovery results in the persistent disk cache."""
        try:
            digest = self.compute_image_digest(image_input)
            path = self._path_for(digest)
            payload = {
                "version": self.CACHE_VERSION,
                "image_sha256": digest,
                "saved_at": time.time(),
                "candidates": [asdict(c) for c in candidates],
            }
            tmp = path.with_suffix(".tmp")
            with self._lock:
                tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
                tmp.replace(path)
                self.writes += 1
            logger.info(f"Saved {len(candidates)} candidate(s) to Search Cache: {digest[:12]}")
        except Exception as e:
            logger.warning(f"Failed to write to search cache: {e}")

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "writes": self.writes,
            }
