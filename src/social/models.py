"""
Data models and contracts for social media profile resolution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SocialPlatform(str, Enum):
    """Supported social and online identity platforms."""
    X_TWITTER = "x_twitter"
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    GITHUB = "github"
    WIKIPEDIA = "wikipedia"
    YOUTUBE = "youtube"
    FACEBOOK = "facebook"
    THREADS = "threads"
    BLUESKY = "bluesky"
    WEBSITE = "website"


PLATFORM_DISPLAY_NAMES = {
    SocialPlatform.X_TWITTER: "X / Twitter",
    SocialPlatform.LINKEDIN: "LinkedIn",
    SocialPlatform.INSTAGRAM: "Instagram",
    SocialPlatform.GITHUB: "GitHub",
    SocialPlatform.WIKIPEDIA: "Wikipedia",
    SocialPlatform.YOUTUBE: "YouTube",
    SocialPlatform.FACEBOOK: "Facebook",
    SocialPlatform.THREADS: "Threads",
    SocialPlatform.BLUESKY: "Bluesky",
    SocialPlatform.WEBSITE: "Official Website",
}


@dataclass
class SocialProfile:
    """Represents a discovered social media or web profile."""
    platform: SocialPlatform
    handle: str
    url: str
    source: str = "wikidata"  # "wikidata", "candidate_dom", "search"
    verified: bool = True
    extracted_title: Optional[str] = None
    verification_reason: Optional[str] = None

    @property
    def display_name(self) -> str:
        return PLATFORM_DISPLAY_NAMES.get(self.platform, self.platform.value.upper())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform.value,
            "platform_label": self.display_name,
            "handle": self.handle,
            "url": self.url,
            "source": self.source,
            "verified": self.verified,
            "extracted_title": self.extracted_title,
            "verification_reason": self.verification_reason,
        }


@dataclass
class PersonSocialIdentity:
    """Identified entity name and aggregated social media accounts for a detected subject."""
    canonical_name: str
    confidence: float
    profiles: List[SocialProfile] = field(default_factory=list)
    bio_summary: Optional[str] = None

    def get_profile(self, platform: SocialPlatform) -> Optional[SocialProfile]:
        for p in self.profiles:
            if p.platform == platform:
                return p
        return None

    def to_dict(self) -> Dict:
        return {
            "canonical_name": self.canonical_name,
            "confidence": round(self.confidence, 4),
            "bio_summary": self.bio_summary,
            "profiles": [p.to_dict() for p in self.profiles],
            "platforms_found": [p.platform.value for p in self.profiles],
        }
