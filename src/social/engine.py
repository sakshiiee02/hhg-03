"""
High-level Social Discovery Engine orchestrating multi-tiered social media resolution.
"""

import logging
from typing import Dict, List, Optional, Set

from src.social.extractor import extract_social_from_url, extract_socials_from_html
from src.social.models import PersonSocialIdentity, SocialPlatform, SocialProfile
from src.social.resolver import WikidataSocialResolver, extract_entity_name
from src.social.verifier import SocialProfileVerifier

logger = logging.getLogger(__name__)


class SocialDiscoveryEngine:
    """
    Coordinates entity name deduction, Wikidata Knowledge Graph queries,
    direct HTML DOM link extraction, and strict name-matching verification
    to discover a person's online social profiles.
    """

    def __init__(self, timeout_seconds: float = 6.0, verify_timeout_seconds: float = 2.5):
        self.wikidata_resolver = WikidataSocialResolver(timeout_seconds=timeout_seconds)
        self.verifier = SocialProfileVerifier(timeout_seconds=verify_timeout_seconds)

    async def discover_socials(
        self,
        candidate_titles: List[str],
        candidate_urls: List[str],
        candidate_htmls: Optional[List[str]] = None,
        candidate_excerpts: Optional[List[str]] = None,
    ) -> Optional[PersonSocialIdentity]:
        """
        Main entry point for resolving a person's social profiles strictly from reverse search candidates.
        Ensures that only profiles matching the person's name are verified and returned.
        """
        # 1. Deduce most probable person entity name strictly from candidate web results
        resolved_name = extract_entity_name(
            titles=candidate_titles,
            candidate_urls=candidate_urls,
            text_excerpts=candidate_excerpts,
        )

        profiles_by_platform: Dict[SocialPlatform, SocialProfile] = {}
        bio_summary: Optional[str] = None
        confidence = 0.50

        # 2. Tier 1: Authoritative Wikidata resolution if name is known
        if resolved_name:
            logger.info(f"Initiating Wikidata social resolution for deduced name: '{resolved_name}'")
            wiki_identity = await self.wikidata_resolver.resolve(resolved_name)
            if wiki_identity:
                resolved_name = wiki_identity.canonical_name
                bio_summary = wiki_identity.bio_summary
                confidence = wiki_identity.confidence
                for prof in wiki_identity.profiles:
                    profiles_by_platform[prof.platform] = prof

        # 3. Tier 2 & 3: Extract and strictly verify candidate links from URLs and HTML pages
        raw_candidates: List[SocialProfile] = []
        for url in candidate_urls:
            p = extract_social_from_url(url, source="candidate_url")
            if p and p.platform not in profiles_by_platform:
                raw_candidates.append(p)

        if candidate_htmls:
            for html in candidate_htmls:
                dom_profiles = extract_socials_from_html(html)
                for p in dom_profiles:
                    if p.platform not in profiles_by_platform and p.url not in [c.url for c in raw_candidates]:
                        raw_candidates.append(p)

        # Only verify and include candidate profiles if we have a resolved person name
        if resolved_name and raw_candidates:
            logger.info(f"Verifying {len(raw_candidates)} candidate social profile(s) against name: '{resolved_name}'")
            verified_candidates = await self.verifier.verify_profiles(raw_candidates, canonical_name=resolved_name)
            for p in verified_candidates:
                if p.platform not in profiles_by_platform:
                    profiles_by_platform[p.platform] = p
        elif not resolved_name and raw_candidates:
            logger.info("Omitting candidate social links because person entity name could not be deduced.")

        if not profiles_by_platform and not resolved_name:
            logger.info("No social profiles or entity names discovered.")
            return None

        final_profiles = list(profiles_by_platform.values())

        # Sort profiles deterministically: X, LinkedIn, Instagram, Threads, Bluesky, GitHub, Wikipedia, YouTube, Facebook, Website
        order = [
            SocialPlatform.X_TWITTER,
            SocialPlatform.LINKEDIN,
            SocialPlatform.INSTAGRAM,
            SocialPlatform.THREADS,
            SocialPlatform.BLUESKY,
            SocialPlatform.GITHUB,
            SocialPlatform.WIKIPEDIA,
            SocialPlatform.YOUTUBE,
            SocialPlatform.FACEBOOK,
            SocialPlatform.WEBSITE,
        ]
        final_profiles.sort(key=lambda p: order.index(p.platform) if p.platform in order else 99)

        return PersonSocialIdentity(
            canonical_name=resolved_name or "Verified Subject",
            confidence=confidence if final_profiles else 0.40,
            profiles=final_profiles,
            bio_summary=bio_summary,
        )
