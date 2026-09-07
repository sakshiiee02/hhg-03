"""
Unit tests for Parallel Multi-Engine Search, Round-Robin Interleaving, and Focused Face Cropping.
"""

import asyncio
from pathlib import Path
import numpy as np
import pytest

from src.face.detector import FaceDetector
from src.search.base import CandidateResult
from src.search.router import SearchRouter
from src.search.serpapi import SerpApiSearchEngine



def test_round_robin_interleaving_and_deduplication():
    """Verifies round-robin interleaving ensures balanced representation across multiple engines."""
    yandex_results = [
        CandidateResult(rank=1, page_url="https://example.com/page1", image_url="https://img.com/1.jpg", source_domain="example.com", title="Page 1", provider="Yandex"),
        CandidateResult(rank=2, page_url="https://duplicate.com/shared", image_url="https://img.com/shared1.jpg", source_domain="duplicate.com", title="Shared Page", provider="Yandex"),
        CandidateResult(rank=3, page_url="https://example.com/page3", image_url="https://img.com/3.jpg", source_domain="example.com", title="Page 3", provider="Yandex"),
    ]

    serpapi_results = [
        CandidateResult(rank=1, page_url="https://duplicate.com/shared?utm_source=serp", image_url="https://img.com/shared2.jpg", source_domain="duplicate.com", title="Shared Page Lens", provider="SerpApi (Google Lens)"),
        CandidateResult(rank=2, page_url="https://bing.com/match", image_url="https://img.com/bing.jpg", source_domain="bing.com", title="Bing Match", provider="SerpApi (Bing Visual)"),
        CandidateResult(rank=3, page_url="https://lens.com/match", image_url="https://img.com/lens.jpg", source_domain="lens.com", title="Lens Match", provider="SerpApi (Google Lens)"),
    ]

    merged = SearchRouter._interleave_and_deduplicate([yandex_results, serpapi_results], max_candidates=10)

    # Candidate 1: Yandex rank 1 (page1)
    # Candidate 2: SerpApi rank 1 (shared - normalized to https://duplicate.com/shared)
    # Candidate 3: Yandex rank 2 (shared - duplicate, should be SKIPPED)
    # Candidate 4: SerpApi rank 2 (bing.com/match)
    # Candidate 5: Yandex rank 3 (page3)
    # Candidate 6: SerpApi rank 3 (lens.com/match)
    urls = [c.page_url for c in merged]
    assert urls == [
        "https://example.com/page1",
        "https://duplicate.com/shared",
        "https://bing.com/match",
        "https://example.com/page3",
        "https://lens.com/match",
    ]
    # Check ranks are sequential 1..5
    assert [c.rank for c in merged] == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_search_router_auto_fallback_without_serpapi(monkeypatch, tmp_path):
    """Verifies that when SERPAPI_API_KEY is not set, Auto router falls back cleanly to Yandex."""
    router = SearchRouter(primary_engine="auto", serpapi_api_key="")
    assert router.serpapi is None

    test_img = tmp_path / "test.jpg"
    test_img.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xFF\xDB\x00C\x00")

    mock_candidates = [
        CandidateResult(rank=1, page_url="https://yandex.com/match1", image_url="https://img.com/y1.jpg", source_domain="yandex.com", title="Yandex Match", provider="Yandex Images"),
    ]

    async def mock_yandex_search(*args, **kwargs):
        return mock_candidates

    monkeypatch.setattr(router.yandex, "search", mock_yandex_search)

    results = await router.discover_candidates(test_img)
    assert len(results) == 1
    assert results[0].page_url == "https://yandex.com/match1"
    assert results[0].provider == "Yandex Images"


@pytest.mark.asyncio
async def test_search_router_parallel_with_serpapi(monkeypatch, tmp_path):
    """Verifies parallel execution across Yandex and SerpApi with round-robin interleaving."""
    router = SearchRouter(primary_engine="auto", serpapi_api_key="mock_key")
    assert router.serpapi is not None

    test_img = tmp_path / "test.jpg"
    test_img.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xFF\xDB\x00C\x00")

    yandex_candidates = [
        CandidateResult(rank=1, page_url="https://example.com/yandex1", image_url="", source_domain="example.com", title="Y1", provider="Yandex Images"),
    ]
    serpapi_candidates = [
        CandidateResult(rank=1, page_url="https://example.com/serp1", image_url="", source_domain="example.com", title="S1", provider="SerpApi (Google Lens)"),
    ]

    async def mock_yandex(*args, **kwargs):
        await asyncio.sleep(0.01)
        return yandex_candidates

    async def mock_serpapi(*args, **kwargs):
        await asyncio.sleep(0.01)
        return serpapi_candidates

    monkeypatch.setattr(router.yandex, "search", mock_yandex)
    monkeypatch.setattr(router.serpapi, "search", mock_serpapi)

    results = await router.discover_candidates(test_img)
    assert len(results) == 2
    assert results[0].page_url == "https://example.com/yandex1"
    assert results[1].page_url == "https://example.com/serp1"
