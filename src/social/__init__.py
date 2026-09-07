"""
Social Media Profile Discovery & Identity Resolution subsystem.
"""

from src.social.engine import SocialDiscoveryEngine
from src.social.extractor import extract_social_from_url, extract_socials_from_html
from src.social.models import PersonSocialIdentity, SocialPlatform, SocialProfile
from src.social.resolver import WikidataSocialResolver, extract_entity_name

__all__ = [
    "SocialPlatform",
    "SocialProfile",
    "PersonSocialIdentity",
    "SocialDiscoveryEngine",
    "WikidataSocialResolver",
    "extract_social_from_url",
    "extract_socials_from_html",
    "extract_entity_name",
]
