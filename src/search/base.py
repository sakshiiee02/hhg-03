"""
Base search engine interface, data contracts, and URL normalization utilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


@dataclass(frozen=True)
class CandidateResult:
    """Represents a single discovered web candidate returned by a reverse search engine."""
    rank: int
    page_url: str
    image_url: Optional[str]
    source_domain: str
    title: Optional[str]
    provider: str


def extract_domain(url: str) -> str:
    """Extracts a normalized domain name from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().split(":")[0]
        if domain.startswith("www."):
            domain = domain[4:]
        return domain or "unknown"
    except Exception:
        return "unknown"


def normalize_url(url: str) -> str:
    """
    Cleans tracking query parameters (utm_*, fbclid, igsh, etc.) from a URL.
    Ensures consistent candidate deduplication.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        tracking_prefixes = ("utm_", "fbclid", "igsh", "ref", "gclid")
        filtered_query = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if not any(k.lower().startswith(p) for p in tracking_prefixes)
        ]
        clean_query = urlencode(filtered_query)
        cleaned = parsed._replace(query=clean_query, fragment="")
        return urlunparse(cleaned)
    except Exception:
        return url


class BaseSearchEngine(ABC):
    """Abstract base class for reverse-image search discovery providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the discovery provider."""
        raise NotImplementedError

    @abstractmethod
    async def search(self, image_path: Path, max_results: int = 30) -> List[CandidateResult]:
        """Executes a reverse-image search and returns a list of candidate results."""
        raise NotImplementedError
