"""Reverse-image search discovery package."""

from src.search.base import BaseSearchEngine, CandidateResult, extract_domain, normalize_url
from src.search.google_lens import GoogleLensSearchEngine
from src.search.router import SearchRouter
from src.search.serpapi import SerpApiSearchEngine
from src.search.yandex import YandexSearchEngine

__all__ = [
    "BaseSearchEngine",
    "CandidateResult",
    "extract_domain",
    "normalize_url",
    "GoogleLensSearchEngine",
    "SearchRouter",
    "SerpApiSearchEngine",
    "YandexSearchEngine",
]
