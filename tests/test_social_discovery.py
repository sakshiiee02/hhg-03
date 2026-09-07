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


def test_threads_and_bluesky_extraction():
    prof_th = extract_social_from_url("https://threads.net/@sama")
    assert prof_th is not None
    assert prof_th.platform == SocialPlatform.THREADS
    assert prof_th.handle == "@sama"
    assert prof_th.url == "https://threads.net/@sama"

    prof_bs = extract_social_from_url("https://bsky.app/profile/sama.bsky.social")
    assert prof_bs is not None
    assert prof_bs.platform == SocialPlatform.BLUESKY
    assert prof_bs.handle == "@sama.bsky.social"
    assert prof_bs.url == "https://bsky.app/profile/sama.bsky.social"


@pytest.mark.asyncio
async def test_verifier_publisher_rejection():
    from src.social.verifier import SocialProfileVerifier

    verifier = SocialProfileVerifier()
    canonical_name = "Sam Altman"

    # Publisher profiles should be rejected immediately
    for pub_url, handle in [
        ("https://x.com/TechCrunch", "@TechCrunch"),
        ("https://x.com/Forbes", "@Forbes"),
        ("https://x.com/TheVerge", "@TheVerge"),
        ("https://youtube.com/@Bloomberg", "@Bloomberg"),
    ]:
        prof = SocialProfile(platform=SocialPlatform.X_TWITTER, handle=handle, url=pub_url, source="candidate_dom")
        is_verified, _, reason = await verifier.verify_profile(prof, canonical_name)
        assert not is_verified, f"Expected {pub_url} to be rejected, but passed with reason: {reason}"
        assert "publisher" in reason.lower()


@pytest.mark.asyncio
async def test_verifier_handle_matching():
    from src.social.verifier import SocialProfileVerifier

    verifier = SocialProfileVerifier()

    # Sam Altman matching handles
    prof_sama = SocialProfile(platform=SocialPlatform.X_TWITTER, handle="@sama", url="https://x.com/sama", source="candidate_dom")
    is_verified, _, reason = await verifier.verify_profile(prof_sama, "Sam Altman")
    assert is_verified
    assert any(k in reason for k in ("firstname", "handle", "metadata"))

    prof_li = SocialProfile(platform=SocialPlatform.LINKEDIN, handle="in/samaltman", url="https://linkedin.com/in/samaltman", source="candidate_dom")
    is_verified, _, reason = await verifier.verify_profile(prof_li, "Sam Altman")
    assert is_verified
    assert any(k in reason for k in ("handle", "metadata", "firstname"))

    # Direct Tier 3 slug verification when HTTP is not used or fails
    name_info = verifier._normalize_name_tokens("Sam Altman")
    assert verifier.verify_handle_slug("samaltman", name_info)[0] is True
    assert verifier.verify_handle_slug("in/samaltman", name_info)[0] is True
    assert verifier.verify_handle_slug("@sama", name_info)[0] is True
    assert verifier.verify_handle_slug("saltman", name_info)[0] is True
    assert verifier.verify_handle_slug("@TechCrunch", name_info)[0] is False
    assert verifier.verify_handle_slug("@random_user_99", name_info)[0] is False

    # Jensen Huang matching handles
    prof_jh = SocialProfile(platform=SocialPlatform.LINKEDIN, handle="in/jensenhuang", url="https://linkedin.com/in/jensenhuang", source="candidate_dom")
    is_verified, _, _ = await verifier.verify_profile(prof_jh, "Jensen Huang")
    assert is_verified

    # Unrelated person handle should be rejected
    prof_unrelated = SocialProfile(platform=SocialPlatform.X_TWITTER, handle="@random_user_99", url="https://x.com/random_user_99", source="candidate_dom")
    is_verified, _, reason = await verifier.verify_profile(prof_unrelated, "Sam Altman")
    assert not is_verified


@pytest.mark.asyncio
async def test_verifier_metadata_text_matching():
    from src.social.verifier import SocialProfileVerifier

    verifier = SocialProfileVerifier()
    name_info = verifier._normalize_name_tokens("Sam Altman")

    # Positive matches
    assert verifier.match_metadata_text("Sam Altman (@sama) / X", name_info)[0] is True
    assert verifier.match_metadata_text("Sam Altman - Co-Founder & CEO - OpenAI | LinkedIn", name_info)[0] is True
    assert verifier.match_metadata_text("Official profile of Sam Altman. Thoughts on AI and tech.", name_info)[0] is True

    # Negative matches
    assert verifier.match_metadata_text("TechCrunch (@TechCrunch) / X", name_info)[0] is False
    assert verifier.match_metadata_text("John Doe - Software Engineer | LinkedIn", name_info)[0] is False


@pytest.mark.asyncio
async def test_verifier_end_to_end_filtering():
    from src.social.verifier import SocialProfileVerifier

    verifier = SocialProfileVerifier()
    candidates = [
        SocialProfile(platform=SocialPlatform.X_TWITTER, handle="@sama", url="https://x.com/sama", source="candidate_dom"),
        SocialProfile(platform=SocialPlatform.X_TWITTER, handle="@TechCrunch", url="https://x.com/TechCrunch", source="candidate_dom"),
        SocialProfile(platform=SocialPlatform.X_TWITTER, handle="@unrelated_guy", url="https://x.com/unrelated_guy", source="candidate_dom"),
        SocialProfile(platform=SocialPlatform.LINKEDIN, handle="in/samaltman", url="https://linkedin.com/in/samaltman", source="candidate_dom"),
    ]

    verified = await verifier.verify_profiles(candidates, canonical_name="Sam Altman")
    urls = [p.url for p in verified]

    assert "https://x.com/sama" in urls
    assert "https://linkedin.com/in/samaltman" in urls
    assert "https://x.com/TechCrunch" not in urls
    assert "https://x.com/unrelated_guy" not in urls


