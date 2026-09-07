"""
Unit tests for candidate URL normalization, domain extraction, and OpenGraph HTML parsing.
"""

from src.search.base import extract_domain, normalize_url
from src.web.extract import extract_page_metadata


def test_extract_domain():
    assert extract_domain("https://www.instagram.com/p/DB123XYZ/") == "instagram.com"
    assert extract_domain("https://twitter.com/nvidia/status/98765") == "twitter.com"
    assert extract_domain("http://sub.domain.co.uk:8080/path?q=1") == "sub.domain.co.uk"


def test_normalize_url():
    url_with_tracking = "https://instagram.com/p/123/?utm_source=ig_web_copy_link&utm_medium=social&igsh=XYZ123"
    cleaned = normalize_url(url_with_tracking)
    assert "utm_" not in cleaned
    assert "igsh" not in cleaned
    assert cleaned == "https://instagram.com/p/123/"


def test_extract_page_metadata():
    sample_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tech Summit 2026 Keynote</title>
        <meta property="og:title" content="Jensen Huang Fireside Chat" />
        <meta property="og:image" content="https://cdn.example.com/photos/jensen_stage.jpg" />
        <meta property="og:description" content="Discussion on generative AI architectures." />
        <link rel="canonical" href="https://example.com/articles/jensen-keynote" />
    </head>
    <body><h1>Event Overview</h1></body>
    </html>
    """
    meta = extract_page_metadata(sample_html, base_url="https://example.com/current")
    assert meta.title == "Jensen Huang Fireside Chat"
    assert meta.image_url == "https://cdn.example.com/photos/jensen_stage.jpg"
    assert meta.canonical_url == "https://example.com/articles/jensen-keynote"
    assert "generative AI" in meta.description
