"""
Search Router coordinating multi-engine visual discovery.
Runs Yandex Images (Playwright Stealth) and SerpApi Multi-Engine
(Google Lens, Bing Visual Search, Google Reverse Image) in parallel,
merging and deduplicating results via quality-weighted round-robin interleaving.
Includes persistent search disk caching to minimize latency and save API credits.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from PIL import Image

from src.search.base import (
    CandidateResult,
    canonical_page_key,
    extract_domain,
    normalize_url,
    resolve_google_goto,
)
from src.search.cache import PersistentSearchCache
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


def _candidate_quality(c: CandidateResult) -> tuple:
    """Computes a tuple quality score to prioritize superior candidate entries."""
    has_image = 1 if bool(c.image_url) else 0
    not_redirect = 0 if "google." in c.source_domain or "goto" in c.page_url else 1
    has_title = 1 if bool(c.title) else 0
    return (has_image, not_redirect, has_title)


class SearchRouter:
    """
    Coordinates reverse-image search discovery across multiple visual engines.
    Supports parallel execution across Yandex Images (Playwright Stealth) and
    SerpApi multi-engine (Google Lens, Bing Visual Search, Google Reverse Image).
    Features quality-scored candidate deduplication and persistent disk caching.
    """

    def __init__(
        self,
        primary_engine: str = "auto",
        serpapi_api_key: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ):
        self.primary_engine = primary_engine.lower()
        self.serpapi_key = serpapi_api_key
        self.yandex = YandexSearchEngine()
        self.serpapi = SerpApiSearchEngine(serpapi_api_key) if serpapi_api_key else None

        if use_cache is None:
            import os
            is_testing = "PYTEST_CURRENT_TEST" in os.environ
            use_cache = not is_testing and os.environ.get("DISABLE_SEARCH_CACHE", "0") not in ("1", "true", "True")

        self.use_cache = use_cache
        self.cache = PersistentSearchCache() if use_cache else None

    @staticmethod
    def _interleave_and_deduplicate(
        result_lists: List[List[CandidateResult]],
        max_candidates: int,
    ) -> List[CandidateResult]:
        """
        Interleaves multiple engine result sets in a round-robin sequence to ensure
        balanced representation across providers, deduplicating with quality scoring
        and canonical social URL normalization.
        """
        merged: List[CandidateResult] = []
        canonical_indices = {}  # canon_key -> index in merged
        image_url_indices = {}  # clean_image_url -> index in merged

        if not result_lists:
            return []

        max_len = max((len(l) for l in result_lists), default=0)

        for i in range(max_len):
            for l in result_lists:
                if i < len(l):
                    c = l[i]
                    resolved_url = resolve_google_goto(c.page_url)
                    norm_url = normalize_url(resolved_url)
                    canon_key = canonical_page_key(norm_url)
                    img_clean = (c.image_url or "").strip().split("?")[0].lower()

                    existing_idx = None
                    if canon_key and canon_key in canonical_indices:
                        existing_idx = canonical_indices[canon_key]
                    elif img_clean and img_clean in image_url_indices:
                        existing_idx = image_url_indices[img_clean]

                    new_cand = CandidateResult(
                        rank=0,  # re-indexed below
                        page_url=norm_url,
                        image_url=c.image_url,
                        source_domain=extract_domain(norm_url),
                        title=c.title,
                        provider=c.provider,
                    )

                    if existing_idx is None:
                        # Brand new candidate
                        new_idx = len(merged)
                        merged.append(new_cand)
                        if canon_key:
                            canonical_indices[canon_key] = new_idx
                        if img_clean:
                            image_url_indices[img_clean] = new_idx
                    else:
                        # Duplicate found: upgrade if new candidate has higher quality
                        prev = merged[existing_idx]
                        if _candidate_quality(new_cand) > _candidate_quality(prev):
                            merged[existing_idx] = new_cand

                    if len(merged) >= max_candidates:
                        break

        # Re-assign clean consecutive ranks 1..N
        final_list: List[CandidateResult] = []
        for idx, item in enumerate(merged[:max_candidates], 1):
            final_list.append(CandidateResult(
                rank=idx,
                page_url=item.page_url,
                image_url=item.image_url,
                source_domain=item.source_domain,
                title=item.title,
                provider=item.provider,
            ))

        return final_list

    async def discover_candidates(
        self,
        image_path: Path,
        max_candidates: int = 35,
    ) -> List[CandidateResult]:
        """
        Conducts reverse search across configured visual search engines.
        In 'auto' mode, queries Yandex and SerpApi concurrently, merging results round-robin.
        Checks and updates persistent search cache when active.
        """
        query_image = _get_search_query_path(image_path)

        # Check persistent search cache
        if self.use_cache and self.cache:
            cached = self.cache.get(query_image)
            if cached is not None and len(cached) > 0:
                logger.info(f"Loaded {len(cached)} candidate(s) directly from persistent search cache.")
                return cached[:max_candidates]

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
                if self.use_cache and self.cache and candidates:
                    self.cache.put(query_image, candidates)
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
                candidates = await self.yandex.search(query_image, max_results=max_candidates)
            else:
                try:
                    candidates = await self.serpapi.search(
                        query_image,
                        max_results=max_candidates,
                        engines=["google_lens"],
                    )
                except Exception as e:
                    logger.warning(f"SerpApi Google Lens query failed: {e}")
                    candidates = []
            if self.use_cache and self.cache and candidates:
                self.cache.put(query_image, candidates)
            return candidates

        # C. Explicit Engine: Yandex Only
        if self.primary_engine == "yandex":
            logger.info("Executing Yandex Images discovery via Playwright...")
            try:
                candidates = await self.yandex.search(query_image, max_results=max_candidates)
                logger.info(f"Yandex discovered {len(candidates)} candidates.")
                if self.use_cache and self.cache and candidates:
                    self.cache.put(query_image, candidates)
                return candidates
            except Exception as e:
                logger.warning(f"Yandex search failed: {e}")
                return []

        # D. Auto / Parallel Multi-Engine Discovery
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

        if self.use_cache and self.cache and deduped:
            self.cache.put(query_image, deduped)

        return deduped
