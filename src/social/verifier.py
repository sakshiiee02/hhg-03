"""
Verification and metadata extraction engine for candidate social profiles.
Ensures that only social links matching the identified subject's name are verified and displayed.
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

from src.social.models import SocialPlatform, SocialProfile

logger = logging.getLogger(__name__)

# Known news, publisher, and media brand handles to categorically reject
PUBLISHER_MEDIA_BRANDS: Set[str] = {
    "techcrunch", "forbes", "theverge", "wired", "bloomberg", "reuters",
    "cnbc", "nytimes", "guardian", "wsj", "cnn", "bbc", "mashable",
    "businessinsider", "cnet", "gizmodo", "engadget", "venturebeat",
    "fortune", "economist", "financialtimes", "latimes", "washingtonpost",
    "npr", "axios", "semaphor", "theinformation", "fastcompany", "reddit",
}

# Generic non-personal account slugs to reject
GENERIC_HANDLE_SLUGS: Set[str] = {
    "support", "jobs", "developer", "engineering", "careers",
    "help", "press", "media", "investors", "advertising", "privacy",
}

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class SocialProfileVerifier:
    """
    Multi-tier social profile verification engine.
    Extracts HTML metadata (<title>, OpenGraph, description) and checks name tokens,
    with handle-slug fallback and in-memory LRU caching.
    """

    def __init__(self, timeout_seconds: float = 2.5, max_concurrency: int = 4, cache_ttl_seconds: float = 3600.0):
        self.timeout_seconds = timeout_seconds
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.cache_ttl_seconds = cache_ttl_seconds
        # In-memory LRU cache: (url, canonical_name_lower) -> (is_valid, extracted_title, reason, timestamp)
        self._cache: Dict[Tuple[str, str], Tuple[bool, Optional[str], str, float]] = {}

    def _normalize_name_tokens(self, canonical_name: str) -> Dict[str, any]:
        """Deconstructs candidate name into tokens, initials, and concatenated forms."""
        raw = canonical_name.strip()
        words = [w.lower() for w in re.findall(r"[a-zA-Z]+", raw) if len(w) >= 2]
        first_name = words[0] if words else ""
        last_name = words[-1] if len(words) >= 2 else ""

        return {
            "raw": raw,
            "words": words,
            "first_name": first_name,
            "last_name": last_name,
            "full_clean": " ".join(words),
            "full_concat": "".join(words),
            "first_init_last": f"{first_name[0]}{last_name}" if first_name and last_name else "",
        }

    def _clean_handle(self, handle: str) -> str:
        """Strips formatting like '@', 'in/', or URLs to get clean alphanumeric handle."""
        h = handle.strip().lstrip("@")
        if h.startswith("in/"):
            h = h[3:]
        h = re.sub(r"[^a-zA-Z0-9_\.\-]", "", h).lower()
        return h

    def verify_handle_slug(self, handle: str, name_info: Dict[str, any]) -> Tuple[bool, str]:
        """
        Tier 3: Checks if the profile handle or URL slug matches the person's name tokens.
        """
        clean_h = self._clean_handle(handle)
        if not clean_h:
            return False, "empty_handle"

        # 1. Immediate rejection of publisher/media accounts and generic services
        if clean_h in PUBLISHER_MEDIA_BRANDS or clean_h in GENERIC_HANDLE_SLUGS:
            return False, f"rejected_publisher_handle:{clean_h}"

        first = name_info["first_name"]
        last = name_info["last_name"]
        full_concat = name_info["full_concat"]
        first_init_last = name_info["first_init_last"]

        # 2. Exact match on concatenated full name (e.g. samaltman, jensenhuang)
        alpha_h = re.sub(r"[^a-z0-9]", "", clean_h)
        if full_concat and (alpha_h == full_concat or full_concat in alpha_h):
            return True, "handle_exact_fullname"

        # 3. Match on first initial + last name (e.g. saltman, jhuang)
        if first_init_last and (alpha_h == first_init_last or first_init_last in alpha_h):
            return True, "handle_initial_lastname"

        # 4. Handle contains both first and last names (e.g. sam_altman, sam.altman)
        if first and last and (first in clean_h and last in clean_h):
            return True, "handle_contains_first_and_last"

        # 5. Handle matches or starts with first name (e.g. 'sama' for Sam Altman, 'jensen')
        # Allowed per user requirement: "option 1 but also allow first name only"
        if first and len(first) >= 3:
            if clean_h == first or clean_h.startswith(first):
                return True, f"handle_firstname_match:{clean_h}"

        return False, "handle_mismatch"

    def match_metadata_text(self, text: str, name_info: Dict[str, any]) -> Tuple[bool, str]:
        """
        Checks if page metadata text (HTML title, OG title/description) matches the person's name.
        """
        if not text:
            return False, "no_metadata_text"

        text_lower = text.lower()
        full_clean = name_info["full_clean"]
        first = name_info["first_name"]
        last = name_info["last_name"]

        # 1. Contiguous full name match (e.g. "Sam Altman")
        if full_clean and full_clean in text_lower:
            return True, "metadata_full_name_match"

        # 2. Both first and last name present in metadata text
        if first and last and (first in text_lower and last in text_lower):
            return True, "metadata_both_names_match"

        # Check for publisher names in title (if not already matched by full/both names)
        for pub in PUBLISHER_MEDIA_BRANDS:
            if pub in text_lower and not any(w in pub for w in name_info["words"]):
                if text_lower.startswith(pub) or f"@{pub}" in text_lower:
                    return False, f"publisher_metadata:{pub}"

        # 3. First name prominent match (e.g. "Sam (@sama) / X")
        if first and len(first) >= 3:
            # Check word boundary for first name
            if re.search(rf"\b{re.escape(first)}\b", text_lower):
                return True, "metadata_first_name_match"

        return False, "metadata_name_mismatch"

    async def fetch_page_metadata(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetches the first 16KB of a candidate social profile page and extracts
        the <title> and OpenGraph / description meta tags.
        """
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        try:
            async with self.semaphore:
                async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                    async with session.get(url, allow_redirects=True) as resp:
                        if resp.status != 200:
                            logger.debug(f"Social URL fetch HTTP {resp.status} for {url}")
                            return None, None

                        # Read only first 16KB to capture <head>
                        chunk = await resp.content.read(16384)
                        html_text = chunk.decode("utf-8", errors="ignore")

                        soup = BeautifulSoup(html_text, "html.parser")
                        title = None
                        if soup.title and soup.title.string:
                            title = soup.title.string.strip()

                        # Check OpenGraph and description
                        og_title = None
                        meta_og = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
                        if meta_og and meta_og.get("content"):
                            og_title = meta_og["content"].strip()

                        og_desc = None
                        meta_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
                        if meta_desc and meta_desc.get("content"):
                            og_desc = meta_desc["content"].strip()

                        combined_meta = " | ".join(filter(None, [title, og_title, og_desc]))
                        return title or og_title, combined_meta

        except Exception as e:
            logger.debug(f"HTTP metadata fetch failed for {url}: {e}")
            return None, None

    async def verify_profile(self, profile: SocialProfile, canonical_name: str) -> Tuple[bool, Optional[str], str]:
        """
        Verifies a single social profile against the subject's canonical name.
        Returns: (is_verified, extracted_title, reason)
        """
        # Tier 1: Wikidata claims are authoritative
        if profile.source == "wikidata":
            return True, f"{profile.handle} (Wikidata Claim)", "authoritative_wikidata_claim"

        name_info = self._normalize_name_tokens(canonical_name)
        if not name_info["first_name"]:
            return False, None, "missing_canonical_name"

        # Check in-memory LRU cache
        cache_key = (profile.url, canonical_name.lower().strip())
        now = time.time()
        if cache_key in self._cache:
            is_valid, title, reason, ts = self._cache[cache_key]
            if now - ts < self.cache_ttl_seconds:
                return is_valid, title, f"cached_{reason}"

        # Quick reject for known publisher handles before making HTTP requests
        clean_h = self._clean_handle(profile.handle)
        if clean_h in PUBLISHER_MEDIA_BRANDS or clean_h in GENERIC_HANDLE_SLUGS:
            self._cache[cache_key] = (False, None, "rejected_publisher_handle", now)
            return False, None, "rejected_publisher_handle"

        # Tier 2: Fetch HTML page metadata (<title>, OpenGraph)
        title, meta_text = await self.fetch_page_metadata(profile.url)

        if meta_text:
            matched, reason = self.match_metadata_text(meta_text, name_info)
            if matched:
                self._cache[cache_key] = (True, title, reason, now)
                return True, title, reason

        # Tier 3: Handle / URL slug fallback (if HTTP failed, was login-walled, or 403)
        matched_slug, slug_reason = self.verify_handle_slug(profile.handle, name_info)
        if matched_slug:
            self._cache[cache_key] = (True, title or profile.handle, slug_reason, now)
            return True, title or profile.handle, slug_reason

        self._cache[cache_key] = (False, title, "name_mismatch", now)
        return False, title, "name_mismatch"

    async def verify_profiles(
        self,
        profiles: List[SocialProfile],
        canonical_name: str,
    ) -> List[SocialProfile]:
        """
        Concurrently verifies all candidate profiles against the person's name.
        Only returns profiles that successfully match the person's name.
        """
        if not profiles:
            return []

        tasks = [self.verify_profile(p, canonical_name) for p in profiles]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        verified_profiles: List[SocialProfile] = []
        for prof, res in zip(profiles, results):
            if isinstance(res, Exception):
                logger.warning(f"Verification error for {prof.url}: {res}")
                continue

            is_verified, title, reason = res
            if is_verified:
                prof.verified = True
                prof.extracted_title = title
                prof.verification_reason = reason
                verified_profiles.append(prof)
                logger.info(f"Verified social link: {prof.platform.value} -> {prof.url} ({reason})")
            else:
                logger.info(f"Rejected inaccurate social link: {prof.platform.value} -> {prof.url} ({reason})")

        return verified_profiles
