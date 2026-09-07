"""
Bounded async candidate image downloader with multi-tiered fallback.
Handles direct page media extraction with graceful fallback to search engine CDN previews.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
import cv2
import numpy as np

from src.blockchain.hash import hash_bytes
from src.search.base import CandidateResult
from src.web.extract import extract_page_metadata

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class DownloadedCandidate:
    """Represents successfully retrieved candidate media and metadata ready for biometric scoring."""
    candidate_id: str
    rank: int
    page_url: str
    canonical_url: str
    image_url: str
    source_domain: str
    title: str
    text_excerpt: str
    image_bgr: np.ndarray
    image_bytes: bytes
    image_sha256: str
    tier_used: str  # 'tier1_page_dom' or 'tier2_indexed_cdn'


class CandidateDownloader:
    """Bounded async candidate downloader implementing multi-tiered fallback."""

    def __init__(self, concurrency_limit: int = 8, timeout_seconds: float = 6.0):
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def _fetch_bytes(self, session: aiohttp.ClientSession, url: str) -> Optional[bytes]:
        try:
            headers = {"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"}
            async with session.get(url, headers=headers, timeout=self.timeout) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    if len(data) > 500:  # Minimum viable image size
                        return data
        except Exception as e:
            logger.debug(f"Failed to fetch bytes from {url}: {e}")
        return None

    async def _download_single(
        self,
        session: aiohttp.ClientSession,
        candidate: CandidateResult,
    ) -> Optional[DownloadedCandidate]:
        async with self.semaphore:
            cand_id = f"cand_{candidate.rank}_{candidate.source_domain}"
            image_bytes = None
            resolved_img_url = candidate.image_url or ""
            canonical_url = candidate.page_url
            title = candidate.title or candidate.source_domain
            excerpt = ""
            tier_used = "tier2_indexed_cdn"

            # -------------------------------------------------------------
            # Tier 1: Attempt direct page extraction
            # -------------------------------------------------------------
            try:
                headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
                async with session.get(candidate.page_url, headers=headers, timeout=self.timeout) as page_resp:
                    if page_resp.status == 200:
                        html_text = await page_resp.text(errors="ignore")
                        meta = extract_page_metadata(html_text, candidate.page_url)
                        if meta.title:
                            title = meta.title
                        if meta.canonical_url:
                            canonical_url = meta.canonical_url
                        if meta.description:
                            excerpt = meta.description

                        # If page provided an og:image, attempt to download it
                        if meta.image_url:
                            img_data = await self._fetch_bytes(session, meta.image_url)
                            if img_data:
                                image_bytes = img_data
                                resolved_img_url = meta.image_url
                                tier_used = "tier1_page_dom"
            except Exception as e:
                logger.debug(f"Tier 1 page visit failed for {candidate.page_url} ({e}). Falling back to Tier 2.")

            # -------------------------------------------------------------
            # Tier 2: Fallback to direct search engine CDN image preview
            # -------------------------------------------------------------
            if not image_bytes and candidate.image_url:
                img_data = await self._fetch_bytes(session, candidate.image_url)
                if img_data:
                    image_bytes = img_data
                    resolved_img_url = candidate.image_url
                    tier_used = "tier2_indexed_cdn"

            if not image_bytes:
                return None

            # Decode image into OpenCV BGR numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None

            sha256 = hash_bytes(image_bytes)

            return DownloadedCandidate(
                candidate_id=cand_id,
                rank=candidate.rank,
                page_url=candidate.page_url,
                canonical_url=canonical_url,
                image_url=resolved_img_url,
                source_domain=candidate.source_domain,
                title=title,
                text_excerpt=excerpt,
                image_bgr=img_bgr,
                image_bytes=image_bytes,
                image_sha256=sha256,
                tier_used=tier_used,
            )

    async def download_candidates(
        self,
        candidates: List[CandidateResult],
    ) -> List[DownloadedCandidate]:
        """Concurrently downloads candidate imagery with bounded async concurrency."""
        conn = aiohttp.TCPConnector(ssl=False, limit=20)
        async with aiohttp.ClientSession(connector=conn) as session:
            tasks = [self._download_single(session, c) for c in candidates]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return [r for r in results if r is not None]
