"""
Base search engine interface, data contracts, and URL normalization utilities.
Includes Google Lens redirect unwrapping, social URL canonicalization, and SSRF validation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import ipaddress
import logging
from pathlib import Path
import re
import socket
from typing import List, Optional
from urllib.parse import parse_qs, parse_qsl, unquote, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)


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


def resolve_google_goto(url: str) -> str:
    """
    Unwraps Google Lens redirect links (e.g. google.com/goto?url=... or google.com/url?q=...).
    Extracts the direct publisher destination URL.
    """
    if not url or not isinstance(url, str):
        return ""
    
    clean = url.strip()
    try:
        parsed = urlparse(clean)
        netloc = parsed.netloc.lower()
        if "google." in netloc and (parsed.path.startswith("/goto") or parsed.path.startswith("/url")):
            query = parse_qs(parsed.query)
            for key in ("url", "q", "target", "dest", "destination"):
                vals = query.get(key)
                if vals:
                    target = unquote(vals[0]).strip()
                    if target.startswith(("http://", "https://")) and "google." not in urlparse(target).netloc.lower():
                        return target
    except Exception:
        pass
    return clean


def canonical_page_key(url: str) -> str:
    """
    Builds a stable deduplication key across search engines and social platforms.
    Strips tracking query parameters (?igsh, ?s=20, ?trk=, utm_*) and redundant path slashes.
    """
    if not url:
        return ""

    unwrapped = resolve_google_goto(url)
    try:
        parsed = urlparse(unwrapped)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        
        # Strip redundant slashes and trailing slashes
        path = re.sub(r"/{2,}", "/", parsed.path).rstrip("/").lower()
        if not path:
            path = "/"

        # Specific social networks: path is the canonical identity, queries are pure tracking
        social_hosts = {
            "instagram.com", "x.com", "twitter.com", "linkedin.com",
            "threads.net", "facebook.com", "tiktok.com", "reddit.com",
            "github.com", "youtube.com",
        }
        if host in social_hosts:
            return f"{host}{path}"

        # Generic web: strip common tracking query params
        tracking_keys = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "igsh", "gclid", "ref", "ref_src", "trk", "feature", "context", "si", "s", "t"
        }
        filtered_query = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k.lower() not in tracking_keys and not any(k.lower().startswith(p) for p in ("utm_", "fbc_"))
        ]
        clean_query = f"?{urlencode(filtered_query)}" if filtered_query else ""
        return f"{host}{path}{clean_query}"
    except Exception:
        return unwrapped.strip().lower()


def normalize_url(url: str) -> str:
    """
    Cleans tracking query parameters from a URL and unwraps redirects.
    Ensures consistent candidate deduplication.
    """
    if not url:
        return ""
    unwrapped = resolve_google_goto(url)
    try:
        parsed = urlparse(unwrapped)
        tracking_prefixes = ("utm_", "fbclid", "igsh", "ref", "gclid", "trk", "si")
        filtered_query = [
            (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if not any(k.lower().startswith(p) for p in tracking_prefixes)
        ]
        clean_query = urlencode(filtered_query)
        cleaned = parsed._replace(query=clean_query, fragment="")
        return urlunparse(cleaned)
    except Exception:
        return unwrapped


def is_safe_public_url(url: str) -> bool:
    """
    Server-Side Request Forgery (SSRF) defense.
    Ensures candidate URLs only point to valid public internet resources and cannot
    target loopback (127.0.0.1), private subnets (10.0.0.0/8, 192.168.0.0/16, etc.),
    link-local addresses (169.254.0.0/16), or cloud instance metadata (169.254.169.254).
    """
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() not in ("http", "https"):
            return False
        if not parsed.hostname:
            return False

        hostname = parsed.hostname.strip()

        # Check direct IP
        try:
            ip = ipaddress.ip_address(hostname)
            return bool(ip.is_global and not ip.is_private and not ip.is_loopback and not ip.is_link_local and not ip.is_multicast and not ip.is_reserved)
        except ValueError:
            pass

        # Resolve hostname via DNS
        addr_infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        if not addr_infos:
            return False

        for family, _, _, _, sockaddr in addr_infos:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if not ip.is_global or ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                logger.warning(f"SSRF protection blocked private/loopback IP address: {ip_str} for host: {hostname}")
                return False

        return True
    except Exception as e:
        logger.debug(f"URL safety check failed for '{url}': {e}")
        return False


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
