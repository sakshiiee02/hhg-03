"""
Search Router coordinating primary automated discovery with multi-engine fallback.
Prioritizes Yandex Images (Playwright Stealth) -> Google Lens -> SerpApi Safety Net.
"""

import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image

from src.search.base import CandidateResult
from src.search.google_lens import GoogleLensSearchEngine
from src.search.serpapi import SerpApiSearchEngine
from src.search.yandex import YandexSearchEngine

logger = logging.getLogger(__name__)


def _get_search_query_path(image_path: Path) -> Path:
    """Downsample huge raw/camera photos (>2.5MB) for web search upload while preserving original for biometrics."""
    try:
        if image_path.exists() and image_path.stat().st_size > 2.5 * 1024 * 1024:
            query_path = image_path.parent / f"_query_{image_path.stem[:8]}.jpg"
            with Image.open(image_path) as img:
                img.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                img.convert("RGB").save(query_path, "JPEG", quality=85)
            logger.info(f"Optimized large query image for search upload: {image_path.name} -> {query_path.name}")
            return query_path
    except Exception as e:
        logger.warning(f"Query image optimization bypassed: {e}")
    return image_path


class SearchRouter:
    """Coordinates reverse-image search discovery across multiple providers."""

    def __init__(
        self,
        primary_engine: str = "auto",
        serpapi_api_key: Optional[str] = None,
    ):
        self.primary_engine = primary_engine.lower()
        self.serpapi_key = serpapi_api_key
        self.yandex = YandexSearchEngine()
        self.google_lens = GoogleLensSearchEngine()
        self.serpapi = SerpApiSearchEngine(serpapi_api_key) if serpapi_api_key else None

    async def discover_candidates(
        self,
        image_path: Path,
        max_candidates: int = 35,
    ) -> List[CandidateResult]:
        """
        Conducts genuine dynamic reverse search with automatic fallback routing.
        """
        candidates: List[CandidateResult] = []
        active_provider = ""
        query_image = _get_search_query_path(image_path)

        # A. Explicit Engine: SerpApi
        if self.primary_engine == "serpapi":
            if not self.serpapi:
                logger.warning(
                    "SerpApi is selected as discovery engine, but SERPAPI_API_KEY is not configured in .env. "
                    "To use SerpApi, sign up for a free key at https://serpapi.com/ and set SERPAPI_API_KEY in .env."
                )
                return []
            logger.info("Executing SerpApi Google Lens discovery...")
            try:
                candidates = await self.serpapi.search(query_image, max_results=max_candidates)
                if candidates:
                    logger.info(f"SerpApi discovered {len(candidates)} candidates.")
            except Exception as e:
                logger.warning(f"SerpApi query failed: {e}")
            return candidates

        # B. Explicit Engine: Google Lens (Playwright)
        if self.primary_engine == "google_lens":
            logger.info("Executing Google Lens discovery via Playwright...")
            try:
                candidates = await self.google_lens.search(query_image, max_results=max_candidates)
                if not candidates:
                    logger.warning(
                        "Google Lens returned 0 results. Google actively blocks automated headless image uploads (CAPTCHA/sorry page). "
                        "Use the Auto Router (Yandex) or configure SERPAPI_API_KEY for 100% reliable Google Lens discovery."
                    )
            except Exception as e:
                logger.warning(f"Google Lens discovery failed: {e}")
            return candidates

        # C. Explicit Engine: Yandex Only
        if self.primary_engine == "yandex":
            logger.info("Executing Yandex Images discovery via Playwright...")
            try:
                candidates = await self.yandex.search(query_image, max_results=max_candidates)
            except Exception as e:
                logger.warning(f"Yandex search failed: {e}")
            return candidates

        # D. Auto Router: Multi-engine fallback (Yandex -> Google Lens -> SerpApi)
        logger.info("Initiating dynamic reverse search on Yandex Images (Primary)...")
        try:
            candidates = await self.yandex.search(query_image, max_results=max_candidates)
            if candidates:
                active_provider = "Yandex Images"
                logger.info(f"Yandex successfully discovered {len(candidates)} candidates.")
        except Exception as e:
            logger.warning(f"Yandex search failed ({e}). Proceeding to fallback...")

        # Fallback 1: Google Lens
        if len(candidates) < 5:
            logger.info("Attempting Google Lens discovery fallback...")
            try:
                lens_candidates = await self.google_lens.search(query_image, max_results=max_candidates)
                if lens_candidates:
                    active_provider = "Google Lens" if not candidates else f"{active_provider} + Google Lens"
                    candidates.extend(lens_candidates)
                    logger.info(f"Google Lens discovered {len(lens_candidates)} candidates.")
            except Exception as e:
                logger.warning(f"Google Lens fallback failed ({e}).")

        # Fallback 2: SerpApi Safety Net
        if len(candidates) < 5 and self.serpapi:
            logger.info("Activating SerpApi safety net fallback...")
            try:
                serp_candidates = await self.serpapi.search(query_image, max_results=max_candidates)
                if serp_candidates:
                    active_provider = "SerpApi" if not candidates else f"{active_provider} + SerpApi"
                    candidates.extend(serp_candidates)
                    logger.info(f"SerpApi safety net discovered {len(serp_candidates)} candidates.")
            except Exception as e:
                logger.warning(f"SerpApi fallback failed ({e}).")

        # Deduplicate candidates across engines by normalized page_url
        seen_urls = set()
        deduped: List[CandidateResult] = []
        rank = 1

        for c in candidates:
            if c.page_url not in seen_urls:
                seen_urls.add(c.page_url)
                deduped.append(CandidateResult(
                    rank=rank,
                    page_url=c.page_url,
                    image_url=c.image_url,
                    source_domain=c.source_domain,
                    title=c.title,
                    provider=c.provider,
                ))
                rank += 1

        return deduped[:max_candidates]
