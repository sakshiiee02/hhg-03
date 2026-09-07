"""
Entity name extraction and Wikidata Knowledge Graph social media resolution.
"""

import asyncio
import collections
import logging
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import aiohttp

from src.social.models import PersonSocialIdentity, SocialPlatform, SocialProfile

logger = logging.getLogger(__name__)

# Noise tokens commonly found in web page titles, slugs, and search snippets
TITLE_NOISE_TERMS = [
    "wikipedia", "the free encyclopedia", "linkedin", "forbes", "bloomberg",
    "instagram", "twitter", "x.com", "facebook", "youtube", "crunchbase",
    "reuters", "cnbc", "techcrunch", "the verge", "wired", "biography",
    "net worth", "photos", "images", "news", "articles", "profile",
    "home", "about", "interviews", "quotes", "latest news", "official site",
    "who is", "ceo", "founder", "president", "director", "executive",
]

SLUG_NOISE_WORDS = {
    "image", "images", "photo", "photos", "portrait", "portraits", "author",
    "autori", "thumb", "thumbnail", "thumbnails", "small", "medium", "large",
    "upload", "uploads", "media", "cache", "avatar", "cropped", "crop", "hires",
    "headshot", "cutout", "closeup", "pic", "picture", "pictures", "wallpaper",
    "full", "official", "banner", "cover", "article", "news", "content", "wp",
    "static", "assets", "gallery", "default", "profile", "icon", "vector",
    "illustration", "png", "jpg", "jpeg", "webp", "gif", "svg", "scale", "hd",
    "hq", "res", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "siggraph", "interview", "keynote", "summit", "conference", "meeting",
    "speech", "talk", "stage", "press",
}

TITLE_TRAILING_NOISE = {
    "cropped", "crop", "hires", "headshot", "portrait", "face", "photo", "image",
    "pic", "picture", "square", "circle", "vector", "official", "ceo", "founder",
    "president", "keynote", "speaker", "interview", "nvidia", "openai", "microsoft",
    "meta", "google", "apple", "tesla", "amazon", "news", "law", "wallpaper",
    "wikipedia", "commons", "wikimedia", "file", "jpg", "jpeg", "png", "webp",
}

CLEAN_DELIMITERS_REGEX = re.compile(r"[\-|–|—|:|•|\|]")


def clean_title_candidate(title: str) -> str:
    """Removes platform and site prefixes/suffixes and extracts person name from page titles."""
    if not title:
        return ""

    # 1. Strip parenthetical/bracketed qualifiers (e.g. "(cropped)", "[photo]", "(born 1963)")
    clean_str = re.sub(r"\(.*?\)", " ", title)
    clean_str = re.sub(r"\[.*?\]", " ", clean_str)

    # 2. Clean noisy prefixes (e.g. "Who is ", "Meet ", "Photo of ", "File:")
    clean_str = re.sub(
        r"^(file|image|photo|picture|who is|meet|photo of|portrait of|inside)\s*[:\-]?\s*",
        "",
        clean_str,
        flags=re.IGNORECASE,
    ).strip()

    # 3. Split on common delimiters
    parts = CLEAN_DELIMITERS_REGEX.split(clean_str)

    # 4. Look across parts for consecutive capitalized name tokens
    name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")

    for part in parts:
        cand = part.strip()
        lower_cand = cand.lower()
        if any(noise in lower_cand for noise in ("wikipedia", "linkedin", "twitter", "instagram", "facebook", "youtube")):
            continue

        # Look for 2 to 4 capitalized words (e.g. "Jensen Huang", "Sam Altman")
        matches = name_pattern.findall(cand)
        for m in matches:
            words = m.split()
            # Trim trailing noise words (e.g. "Jensen Huang Cropped" -> "Jensen Huang")
            if len(words) > 2 and words[-1].lower() in TITLE_TRAILING_NOISE:
                words = words[:-1]
            if 2 <= len(words) <= 3 and not any(w.lower() in ("net worth", "free encyclopedia", "breaking news", "united states") for w in words):
                return " ".join(words)

        words = [w for w in cand.split() if w.isalpha()]
        if len(words) > 2 and words[-1].lower() in TITLE_TRAILING_NOISE:
            words = words[:-1]
        if 2 <= len(words) <= 3 and all(w[0].isupper() for w in words):
            return " ".join(words)

    return parts[0].strip() if parts else title.strip()


def extract_entity_name(
    titles: List[str],
    candidate_urls: Optional[List[str]] = None,
    text_excerpts: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Extracts the most probable person/entity name strictly from a collection of candidate
    web page titles, candidate URL slugs, and text excerpts.
    Uses pattern cleaning and frequency voting.
    """
    cleaned_names: List[str] = []

    # 1. Extract names from URL slugs (e.g. /sam-altman.jpg, /jensen-huang-cropped-1.jpg)
    if candidate_urls:
        for url in candidate_urls:
            slugs = re.findall(r'([a-z]{3,20}(?:[-_][a-z0-9]{1,20})+)', url.lower())
            for s in slugs:
                parts = re.split(r'[-_]', s)
                clean_parts = [p for p in parts if p.lower() not in SLUG_NOISE_WORDS and p.isalpha()]
                if 2 <= len(clean_parts) <= 3:
                    cand_name = " ".join(p.capitalize() for p in clean_parts)
                    cleaned_names.append(cand_name)

    # 2. Extract names from candidate titles
    for t in (titles or []):
        cleaned = clean_title_candidate(t)
        words = [w for w in cleaned.split() if w.isalpha()]
        if len(words) > 2 and words[-1].lower() in TITLE_TRAILING_NOISE:
            words = words[:-1]
        if 2 <= len(words) <= 3 and all(w[0].isupper() for w in words):
            cleaned_names.append(" ".join(words))

    # 3. Extract names from text excerpts
    if text_excerpts:
        name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
        for exc in text_excerpts:
            if exc:
                for m in name_pattern.findall(exc):
                    words = m.split()
                    if len(words) > 2 and words[-1].lower() in TITLE_TRAILING_NOISE:
                        words = words[:-1]
                    if 2 <= len(words) <= 3 and not any(w.lower() in ("face verification", "breaking news", "united states", "executive director") for w in words):
                        cleaned_names.append(" ".join(words))

    if not cleaned_names:
        if titles:
            fallback_words = [w for w in titles[0].split() if w.isalpha()]
            if len(fallback_words) >= 2:
                return f"{fallback_words[0]} {fallback_words[1]}".title()
        return None

    # Count occurrences
    counts = collections.Counter(cleaned_names)
    most_common_name, _ = counts.most_common(1)[0]
    return most_common_name


class WikidataSocialResolver:
    """
    Asynchronously queries Wikipedia and Wikidata APIs to retrieve verified official social accounts.
    Requires no API keys, has zero rate-limits for moderate queries, and returns authoritative IDs.
    """

    USER_AGENT = "FaceVerificationBot/1.0 (https://github.com/sakshiiee02/hhg-03; contact: sakshisingh020504@gmail.com)"

    def __init__(self, timeout_seconds: float = 6.0):
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def resolve(self, person_name: str) -> Optional[PersonSocialIdentity]:
        """
        Looks up the person on Wikipedia/Wikidata and returns their structured social profiles.
        Supports fallback query tokenization for multi-word queries (e.g. 'Jensen Huang Cropped' -> 'Jensen Huang').
        """
        if not person_name or len(person_name.strip()) < 3:
            return None

        clean_name = person_name.strip()
        headers = {"User-Agent": self.USER_AGENT}

        # Multi-stage query strategy: try full name, then first 2 tokens
        query_attempts = [clean_name]
        words = clean_name.split()
        if len(words) > 2:
            query_attempts.append(" ".join(words[:2]))

        try:
            async with aiohttp.ClientSession(headers=headers, timeout=self.timeout) as session:
                wiki_title = None

                for query in query_attempts:
                    search_url = (
                        "https://en.wikipedia.org/w/api.php?action=query&list=search"
                        f"&srsearch={quote_plus(query)}&utf8=&format=json"
                    )
                    async with session.get(search_url) as resp:
                        if resp.status != 200:
                            continue
                        data = await resp.json()
                        search_results = data.get("query", {}).get("search", [])
                        if not search_results:
                            continue

                        # Check top 3 search hits for shared significant name tokens
                        name_tokens = set(query.lower().split())
                        for candidate_hit in search_results[:3]:
                            hit_title = candidate_hit["title"]
                            title_tokens = set(hit_title.lower().split())
                            if name_tokens.intersection(title_tokens):
                                wiki_title = hit_title
                                break

                    if wiki_title:
                        break

                if not wiki_title:
                    return None

                # 2. Retrieve Wikidata entity ID (Q-number) and page summary
                page_info_url = (
                    "https://en.wikipedia.org/w/api.php?action=query&prop=pageprops"
                    f"&titles={quote_plus(wiki_title)}&format=json"
                )
                wikidata_id = None
                bio_summary = None

                async with session.get(page_info_url) as resp:
                    if resp.status == 200:
                        p_data = await resp.json()
                        pages = p_data.get("query", {}).get("pages", {})
                        for pid, pinfo in pages.items():
                            props = pinfo.get("pageprops", {})
                            wikidata_id = props.get("wikibase_item")
                            bio_summary = props.get("wikibase-shortdesc")
                            break

                profiles: List[SocialProfile] = [
                    SocialProfile(
                        platform=SocialPlatform.WIKIPEDIA,
                        handle=wiki_title,
                        url=f"https://en.wikipedia.org/wiki/{quote_plus(wiki_title.replace(' ', '_'))}",
                        source="wikidata",
                        verified=True,
                    )
                ]

                # 3. Query Wikidata claims for authoritative social media handles
                if wikidata_id:
                    wd_url = (
                        "https://www.wikidata.org/w/api.php?action=wbgetentities"
                        f"&ids={wikidata_id}&props=claims&format=json"
                    )
                    async with session.get(wd_url) as resp:
                        if resp.status == 200:
                            wd_data = await resp.json()
                            claims = wd_data.get("entities", {}).get(wikidata_id, {}).get("claims", {})

                            # P2002: Twitter/X username
                            if "P2002" in claims:
                                handle = claims["P2002"][0]["mainsnak"]["datavalue"]["value"]
                                profiles.append(SocialProfile(
                                    platform=SocialPlatform.X_TWITTER,
                                    handle=f"@{handle}",
                                    url=f"https://x.com/{handle}",
                                    source="wikidata",
                                    verified=True,
                                ))

                            # P6634: LinkedIn personal profile ID
                            if "P6634" in claims:
                                handle = claims["P6634"][0]["mainsnak"]["datavalue"]["value"]
                                profiles.append(SocialProfile(
                                    platform=SocialPlatform.LINKEDIN,
                                    handle=f"in/{handle}",
                                    url=f"https://linkedin.com/in/{handle}",
                                    source="wikidata",
                                    verified=True,
                                ))

                            # P2003: Instagram username
                            if "P2003" in claims:
                                handle = claims["P2003"][0]["mainsnak"]["datavalue"]["value"]
                                profiles.append(SocialProfile(
                                    platform=SocialPlatform.INSTAGRAM,
                                    handle=f"@{handle}",
                                    url=f"https://instagram.com/{handle}",
                                    source="wikidata",
                                    verified=True,
                                ))

                            # P2037: GitHub username
                            if "P2037" in claims:
                                handle = claims["P2037"][0]["mainsnak"]["datavalue"]["value"]
                                profiles.append(SocialProfile(
                                    platform=SocialPlatform.GITHUB,
                                    handle=handle,
                                    url=f"https://github.com/{handle}",
                                    source="wikidata",
                                    verified=True,
                                ))

                            # P2397: YouTube channel ID
                            if "P2397" in claims:
                                handle = claims["P2397"][0]["mainsnak"]["datavalue"]["value"]
                                profiles.append(SocialProfile(
                                    platform=SocialPlatform.YOUTUBE,
                                    handle=handle,
                                    url=f"https://youtube.com/channel/{handle}",
                                    source="wikidata",
                                    verified=True,
                                ))

                            # P856: Official website
                            if "P856" in claims:
                                site_url = claims["P856"][0]["mainsnak"]["datavalue"]["value"]
                                profiles.append(SocialProfile(
                                    platform=SocialPlatform.WEBSITE,
                                    handle="Official Website",
                                    url=site_url,
                                    source="wikidata",
                                    verified=True,
                                ))

                return PersonSocialIdentity(
                    canonical_name=wiki_title,
                    confidence=0.95,
                    profiles=profiles,
                    bio_summary=bio_summary,
                )

        except Exception as e:
            logger.warning(f"Wikidata lookup failed for '{person_name}': {e}")
            return None
