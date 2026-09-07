"""
HTML metadata extraction for OpenGraph tags, page titles, and canonical links.
"""

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin
from bs4 import BeautifulSoup


@dataclass
class PageMetadata:
    """Extracted metadata attributes from an HTML document."""
    title: Optional[str]
    image_url: Optional[str]
    canonical_url: Optional[str]
    description: Optional[str]


def extract_page_metadata(html_content: str, base_url: str) -> PageMetadata:
    """
    Parses HTML to extract OpenGraph image, title, canonical URL, and description.
    Resolves relative URLs against base_url.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Image URL (og:image, twitter:image, link rel=image_src, first img)
    image_url = None
    og_img = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "og:image"})
    tw_img = soup.find("meta", property="twitter:image") or soup.find("meta", attrs={"name": "twitter:image"})
    rel_img = soup.find("link", rel="image_src")

    raw_img = (
        (og_img.get("content") if og_img else None)
        or (tw_img.get("content") if tw_img else None)
        or (rel_img.get("href") if rel_img else None)
    )
    if raw_img:
        image_url = urljoin(base_url, raw_img.strip())

    # 2. Page Title
    og_title = soup.find("meta", property="og:title")
    tw_title = soup.find("meta", attrs={"name": "twitter:title"})
    title_tag = soup.find("title")

    raw_title = (
        (og_title.get("content") if og_title else None)
        or (tw_title.get("content") if tw_title else None)
        or (title_tag.string if title_tag else None)
    )
    title = raw_title.strip() if raw_title else None

    # 3. Canonical URL
    canonical = None
    canon_tag = soup.find("link", rel="canonical")
    if canon_tag and canon_tag.get("href"):
        canonical = urljoin(base_url, canon_tag["href"].strip())

    # 4. Description / Excerpt
    og_desc = soup.find("meta", property="og:description")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    raw_desc = (
        (og_desc.get("content") if og_desc else None)
        or (meta_desc.get("content") if meta_desc else None)
    )
    desc = raw_desc.strip() if raw_desc else None

    return PageMetadata(
        title=title,
        image_url=image_url,
        canonical_url=canonical or base_url,
        description=desc,
    )
