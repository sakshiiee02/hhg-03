"""
Unit tests for social profile extraction, entity name resolution, and social discovery engine.
"""

import pytest
from src.social.extractor import extract_social_from_url, extract_socials_from_html
from src.social.models import PersonSocialIdentity, SocialPlatform, SocialProfile
from src.social.resolver import clean_title_candidate, extract_entity_name


def test_extract_social_from_url():
    # X / Twitter
    prof_x = extract_social_from_url("https://x.com/sama?s=20")
    assert prof_x is not None
    assert prof_x.platform == SocialPlatform.X_TWITTER
    assert prof_x.handle == "@sama"
    assert prof_x.url == "https://x.com/sama"

    prof_tw = extract_social_from_url("https://twitter.com/JensenHuang/")
    assert prof_tw is not None
    assert prof_tw.platform == SocialPlatform.X_TWITTER
    assert prof_tw.handle == "@JensenHuang"

    # Negative matches for system endpoints
    assert extract_social_from_url("https://x.com/home") is None
    assert extract_social_from_url("https://twitter.com/explore") is None

    # LinkedIn
    prof_li = extract_social_from_url("https://www.linkedin.com/in/jenhsunhuang/")
    assert prof_li is not None
    assert prof_li.platform == SocialPlatform.LINKEDIN
    assert prof_li.handle == "in/jenhsunhuang"
    assert prof_li.url == "https://linkedin.com/in/jenhsunhuang"

    # Instagram
    prof_ig = extract_social_from_url("https://instagram.com/sama")
    assert prof_ig is not None
    assert prof_ig.platform == SocialPlatform.INSTAGRAM
    assert prof_ig.handle == "@sama"

    assert extract_social_from_url("https://instagram.com/p/C3abc123/") is None

    # GitHub
    prof_gh = extract_social_from_url("https://github.com/torvalds")
    assert prof_gh is not None
    assert prof_gh.platform == SocialPlatform.GITHUB
    assert prof_gh.handle == "torvalds"
    assert prof_gh.url == "https://github.com/torvalds"

    assert extract_social_from_url("https://github.com/pricing") is None

    # Wikipedia
    prof_wiki = extract_social_from_url("https://en.wikipedia.org/wiki/Jensen_Huang")
    assert prof_wiki is not None
    assert prof_wiki.platform == SocialPlatform.WIKIPEDIA
    assert prof_wiki.handle == "Jensen Huang"


def test_extract_socials_from_html():
    sample_html = """
    <html>
      <body>
        <h1>Founder Bio</h1>
        <p>Follow on social:</p>
        <a href="https://twitter.com/sama">Twitter</a>
        <a href="https://www.linkedin.com/in/samaltman">LinkedIn Profile</a>
        <a href="https://github.com/sama">GitHub</a>
        <a href="https://example.com/about">About Page</a>
      </body>
    </html>
    """
    profiles = extract_socials_from_html(sample_html)
    assert len(profiles) == 3
    platforms = {p.platform for p in profiles}
    assert SocialPlatform.X_TWITTER in platforms
    assert SocialPlatform.LINKEDIN in platforms
    assert SocialPlatform.GITHUB in platforms


def test_clean_title_candidate():
    assert clean_title_candidate("Jensen Huang - Wikipedia") == "Jensen Huang"
    assert clean_title_candidate("Sam Altman | LinkedIn") == "Sam Altman"
    assert clean_title_candidate("Who is Jensen Huang? Biography and Net Worth - Forbes") == "Jensen Huang"


def test_extract_entity_name():
    titles = [
        "Jensen Huang - Wikipedia",
        "Jensen Huang, President and CEO - NVIDIA",
        "Who is Jensen Huang? - Forbes",
        "NVIDIA Keynote 2024 - YouTube",
    ]
    name = extract_entity_name(titles)
    assert name == "Jensen Huang"


def test_extract_entity_name_with_noise_slugs():
    """Verifies that image/event noise in URL slugs (like 'cropped', 'siggraph') is cleanly filtered."""
    urls = [
        "https://profitonline.cz/wp-content/uploads/2026/05/jensen-huang-cropped-1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/c/c4/Jensen_Huang_SIGGRAPH_2023.jpg",
    ]
    titles = ["jensen-huang-cropped-1.jpg"]
    name = extract_entity_name(titles=titles, candidate_urls=urls)
    assert name == "Jensen Huang"


@pytest.mark.asyncio
async def test_social_discovery_engine_live():
    from src.social.engine import SocialDiscoveryEngine

    engine = SocialDiscoveryEngine()
    titles = ["Jensen Huang - Wikipedia", "Jensen Huang Keynote - Forbes"]
    urls = ["https://en.wikipedia.org/wiki/Jensen_Huang"]
    
    identity = await engine.discover_socials(candidate_titles=titles, candidate_urls=urls)
    assert identity is not None
    assert "Jensen Huang" in identity.canonical_name
    assert len(identity.profiles) >= 2
    
    platforms = [p.platform for p in identity.profiles]
    assert SocialPlatform.WIKIPEDIA in platforms
    # Should find X or LinkedIn from Wikidata
    assert any(p in (SocialPlatform.X_TWITTER, SocialPlatform.LINKEDIN) for p in platforms)

