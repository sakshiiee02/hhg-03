"""
SerpApi Google Lens engine fallback discovery provider.
Activated when SERPAPI_API_KEY is configured in the environment.
"""

import logging
from pathlib import Path
from typing import List, Optional
import requests

from src.search.base import BaseSearchEngine, CandidateResult, extract_domain, normalize_url

logger = logging.getLogger(__name__)


class SerpApiSearchEngine(BaseSearchEngine):
    """Fallback search provider using SerpApi Google Lens engine."""

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    @property
    def provider_name(self) -> str:
        return "SerpApi (Google Lens API)"

    async def search(self, image_path: Path, max_results: int = 30) -> List[CandidateResult]:
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

        results: List[CandidateResult] = []
        url = "https://serpapi.com/search"

        try:
            with open(path, "rb") as img_file:
                # Upload directly or pass URL
                files = {"image": img_file}
                params = {
                    "engine": "google_lens",
                    "api_key": self.api_key,
                    "hl": "en",
                }
                logger.info("Executing SerpApi Google Lens query...")
                resp = requests.post(url, params=params, files=files, timeout=20)

            if resp.status_code != 200:
                logger.warning(f"SerpApi returned HTTP {resp.status_code}: {resp.text[:200]}")
                return []

            data = resp.json()
            visual_matches = data.get("visual_matches", [])

            rank = 1
            seen_urls = set()

            for item in visual_matches:
                if len(results) >= max_results:
                    break
                link = item.get("link")
                if not link:
                    continue
                norm_link = normalize_url(link)
                if norm_link in seen_urls:
                    continue
                seen_urls.add(norm_link)

                results.append(CandidateResult(
                    rank=rank,
                    page_url=norm_link,
                    image_url=item.get("thumbnail"),
                    source_domain=extract_domain(norm_link),
                    title=item.get("title"),
                    provider="SerpApi (Google Lens)",
                ))
                rank += 1

        except Exception as e:
            logger.error(f"SerpApi discovery error: {e}")

        return results
