"""
Search Router coordinating multi-engine visual discovery.
Runs Yandex Images (Playwright Stealth) and SerpApi Multi-Engine
(Google Lens, Bing Visual Search, Google Reverse Image) in parallel,
merging and deduplicating results via round-robin interleaving.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image

from src.search.base import CandidateResult, normalize_url
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
    """
    Coordinates reverse-image search discovery across multiple visual engines.
    Supports parallel execution across Yandex Images (Playwright Stealth) and
    SerpApi multi-engine (Google Lens, Bing Visual Search, Google Reverse Image).
    """

    def __init__(
        self,
        primary_engine: str = "auto",
        serpapi_api_key: Optional[str] = None,
    ):
        self.primary_engine = primary_engine.lower()
        self.serpapi_key = serpapi_api_key
        self.yandex = YandexSearchEngine()
        self.serpapi = SerpApiSearchEngine(serpapi_api_key) if serpapi_api_key else None

    @staticmethod
    def _interleave_and_deduplicate(
        result_lists: List[List[CandidateResult]],
        max_candidates: int,
    ) -> List[CandidateResult]:
        """
        Interleaves multiple engine result sets in a round-robin sequence to ensure
        balanced representation across providers, deduplicating by normalized page URL.
        """
        merged: List[CandidateResult] = []
        seen_urls = set()
        if not result_lists:
            return []

        max_len = max((len(l) for l in result_lists), default=0)
        rank = 1

        for i in range(max_len):
            for l in result_lists:
                if i < len(l):
                    c = l[i]
                    norm_url = normalize_url(c.page_url)
                    if norm_url not in seen_urls:
                        seen_urls.add(norm_url)
                        merged.append(CandidateResult(
                            rank=rank,
                            page_url=norm_url,
                            image_url=c.image_url,
                            source_domain=c.source_domain,
                            title=c.title,
                            provider=c.provider,
                        ))
                        rank += 1
                        if len(merged) >= max_candidates:
                            return merged

        return merged

    async def discover_candidates(
        self,
        image_path: Path,
        max_candidates: int = 35,
    ) -> List[CandidateResult]:
        """
        Conducts reverse search across configured visual search engines.
        In 'auto' mode, queries Yandex and SerpApi concurrently, merging results round-robin.
        """
        query_image = _get_search_query_path(image_path)

        # A. Explicit Engine: SerpApi Multi-Engine
        if self.primary_engine == "serpapi":
            if not self.serpapi:
                logger.warning(
                    "SerpApi is selected as discovery engine, but SERPAPI_API_KEY is not configured in .env. "
                    "To use SerpApi, set SERPAPI_API_KEY in .env or switch to Auto / Yandex."
                )
                return []
            logger.info("Executing SerpApi Multi-Engine discovery (Google Lens + Bing + Google Reverse)...")
            try:
                candidates = await self.serpapi.search(query_image, max_results=max_candidates)
                logger.info(f"SerpApi discovered {len(candidates)} candidates.")
                return candidates
            except Exception as e:
                logger.warning(f"SerpApi query failed: {e}")
                return []

        # B. Explicit Engine: Google Lens (via SerpApi with fallback)
        if self.primary_engine in ("google_lens", "lens"):
            if not self.serpapi:
                logger.warning(
                    "Google Lens now runs reliably via SerpApi, but SERPAPI_API_KEY is not set. "
                    "Falling back to Yandex Images."
                )
                return await self.yandex.search(query_image, max_results=max_candidates)
            try:
                return await self.serpapi.search(
                    query_image,
                    max_results=max_candidates,
                    engines=["google_lens"],
                )
            except Exception as e:
                logger.warning(f"SerpApi Google Lens query failed: {e}")
                return []

        # C. Explicit Engine: Yandex Only
        if self.primary_engine == "yandex":
            logger.info("Executing Yandex Images discovery via Playwright...")
            try:
                candidates = await self.yandex.search(query_image, max_results=max_candidates)
                logger.info(f"Yandex discovered {len(candidates)} candidates.")
                return candidates
            except Exception as e:
                logger.warning(f"Yandex search failed: {e}")
                return []

        # D. Auto / Parallel Multi-Engine Discovery
        # Runs Yandex and SerpApi concurrently when SerpApi is configured.
        logger.info("Initiating parallel visual discovery across available search engines...")
        engines_to_run = [
            ("Yandex Images", self.yandex.search(query_image, max_results=max_candidates))
        ]

        if self.serpapi:
            engines_to_run.append(
                ("SerpApi Multi-Engine", self.serpapi.search(query_image, max_results=max_candidates))
            )
        else:
            logger.info("SERPAPI_API_KEY not configured; running visual search on Yandex Images.")

        tasks = [task for _, task in engines_to_run]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        engine_candidate_lists: List[List[CandidateResult]] = []
        for (name, _), result in zip(engines_to_run, raw_results):
            if isinstance(result, Exception):
                logger.warning(f"Search provider '{name}' error: {result}")
            elif isinstance(result, list) and result:
                logger.info(f"Search provider '{name}' discovered {len(result)} candidate(s).")
                engine_candidate_lists.append(result)

        deduped = self._interleave_and_deduplicate(engine_candidate_lists, max_candidates=max_candidates)
        logger.info(f"Multi-engine discovery complete. Total unique candidates: {len(deduped)}")
        return deduped
