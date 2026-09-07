"""
Regex and HTML DOM parsers for extracting social media profile links and handles.
"""

import logging
import re
from typing import List, Optional, Set
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup

from src.social.models import SocialPlatform, SocialProfile

logger = logging.getLogger(__name__)

# Patterns matching social profile URLs and capturing the user handle
PLATFORM_PATTERNS = [
    (
        SocialPlatform.X_TWITTER,
        re.compile(
            r"https?://(?:www\.)?(?:twitter|x)\.com/(?!home|explore|search|intent|login|share|hashtag|i/|privacy|tos|about)(@?[a-zA-Z0-9_]{1,50})(?:[/?#]|$)",
            re.IGNORECASE,
        ),
    ),
    (
        SocialPlatform.LINKEDIN,
        re.compile(
            r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub)/([a-zA-Z0-9\-_%]+)(?:[/?#]|$)",
            re.IGNORECASE,
        ),
    ),
    (
        SocialPlatform.INSTAGRAM,
        re.compile(
            r"https?://(?:www\.)?instagram\.com/(?!p/|reel/|stories/|explore/|direct/|accounts/|developer/)(@?[a-zA-Z0-9_\.]{1,50})(?:[/?#]|$)",
            re.IGNORECASE,
        ),
    ),
    (
        SocialPlatform.GITHUB,
        re.compile(
            r"https?://(?:www\.)?github\.com/(?!about|pricing|features|topics|orgs|pulls|issues|explore|marketplace|trending|settings)(@?[a-zA-Z0-9\-]{1,50})(?:[/?#]|$)",
            re.IGNORECASE,
        ),
    ),
    (
        SocialPlatform.WIKIPEDIA,
        re.compile(
            r"https?://([a-z]{2,3}\.)?wikipedia\.org/wiki/(?!Special:|File:|Wikipedia:|Help:|Portal:|Template:|Talk:)([a-zA-Z0-9_\-%]+)(?:[/?#]|$)",
            re.IGNORECASE,
        ),
    ),
    (
        SocialPlatform.YOUTUBE,
        re.compile(
            r"https?://(?:www\.)?youtube\.com/((?:@|c/|channel/|user/)[a-zA-Z0-9_\.\-]+)(?:[/?#]|$)",
            re.IGNORECASE,
        ),
    ),
    (
        SocialPlatform.FACEBOOK,
        re.compile(
            r"https?://(?:www\.)?facebook\.com/(?!sharer|share|login|policies|terms|help|pages/|groups/)([a-zA-Z0-9_\.]{3,50})(?:[/?#]|$)",
            re.IGNORECASE,
        ),
    ),
]


def extract_social_from_url(url: str, source: str = "url") -> Optional[SocialProfile]:
    """
    Examines a single URL to determine if it is a direct profile link on a supported platform.
    Returns SocialProfile if matched, else None.
    """
    if not url or not url.startswith("http"):
        return None

    clean_url = url.split("?")[0].rstrip("/")

    for platform, pattern in PLATFORM_PATTERNS:
        match = pattern.search(clean_url)
        if match:
            raw_handle = match.group(1) if platform != SocialPlatform.WIKIPEDIA else match.group(2)
            raw_handle = unquote(raw_handle).lstrip("@")
            
            # Canonicalize normalized profile URL
            if platform == SocialPlatform.X_TWITTER:
                norm_url = f"https://x.com/{raw_handle}"
                handle_fmt = f"@{raw_handle}"
            elif platform == SocialPlatform.LINKEDIN:
                norm_url = f"https://linkedin.com/in/{raw_handle}"
                handle_fmt = f"in/{raw_handle}"
            elif platform == SocialPlatform.INSTAGRAM:
                norm_url = f"https://instagram.com/{raw_handle}"
                handle_fmt = f"@{raw_handle}"
            elif platform == SocialPlatform.GITHUB:
                norm_url = f"https://github.com/{raw_handle}"
                handle_fmt = raw_handle
            elif platform == SocialPlatform.WIKIPEDIA:
                norm_url = f"https://en.wikipedia.org/wiki/{raw_handle}"
                handle_fmt = raw_handle.replace("_", " ")
            elif platform == SocialPlatform.YOUTUBE:
                norm_url = f"https://youtube.com/{raw_handle}"
                handle_fmt = raw_handle
            elif platform == SocialPlatform.FACEBOOK:
                norm_url = f"https://facebook.com/{raw_handle}"
                handle_fmt = raw_handle
            else:
                norm_url = clean_url
                handle_fmt = raw_handle

            return SocialProfile(
                platform=platform,
                handle=handle_fmt,
                url=norm_url,
                source=source,
                verified=True,
            )

    return None


def extract_socials_from_html(html_text: str, base_url: str = "") -> List[SocialProfile]:
    """
    Parses HTML content and extracts all outbound social profile links found in <a> tags.
    Deduplicates across the document.
    """
    if not html_text:
        return []

    soup = BeautifulSoup(html_text, "html.parser")
    found_profiles: List[SocialProfile] = []
    seen_urls: Set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()
        profile = extract_social_from_url(href, source="candidate_dom")
        if profile and profile.url not in seen_urls:
            seen_urls.add(profile.url)
            found_profiles.append(profile)

    return found_profiles
