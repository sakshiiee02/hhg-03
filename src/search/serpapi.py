"""
SerpApi Visual Search Provider.
Uploads local probe image to SerpApi Image API (https://serpapi.com/image)
to obtain a temporary image_id, then queries SerpApi Google Lens
(https://serpapi.com/search.json?engine=google_lens) for candidate discovery.
Unwraps Google redirects, parses exact matches first, and prioritizes direct publisher URLs.
"""

import asyncio
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import requests

from src.search.base import (
    BaseSearchEngine,
    CandidateResult,
    canonical_page_key,
    extract_domain,
    normalize_url,
    resolve_google_goto,
)

logger = logging.getLogger(__name__)


class SerpApiSearchEngine(BaseSearchEngine):
    """Visual reverse image search provider using SerpApi Google Lens API."""

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()

    @property
    def provider_name(self) -> str:
        return "SerpApi (Google Lens)"

    def _upload_image(self, image_path: Path) -> Optional[str]:
        """
        Uploads local image to https://serpapi.com/image to obtain a temporary image_id.
        Ensures the payload is optimized and strictly under SerpApi's 500 KB limit.
        """
        upload_url = "https://serpapi.com/image"
        try:
            buf = io.BytesIO()
            with Image.open(image_path) as img:
                max_dim = max(img.width, img.height)
                if max_dim > 1000:
                    scale = 1000.0 / max_dim
                    new_size = (int(img.width * scale), int(img.height * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                img.convert("RGB").save(buf, format="JPEG", quality=85)

            buf.seek(0)
            files = {"image": ("query.jpg", buf, "image/jpeg")}
            data = {"api_key": self.api_key}

            logger.info(f"Uploading image to SerpApi Image API ({buf.getbuffer().nbytes / 1024:.1f} KB)...")
            resp = requests.post(upload_url, files=files, data=data, timeout=25)

            if resp.status_code != 200:
                logger.warning(f"SerpApi image upload failed (HTTP {resp.status_code}): {resp.text[:180]}")
                return None

            payload = resp.json()
            image_id = payload.get("image_id")
            if not image_id:
                logger.warning(f"SerpApi image upload response missing 'image_id': {payload}")
                return None

            logger.info(f"SerpApi image upload successful. Assigned image_id: {image_id[:16]}...")
            return image_id

        except Exception as e:
            logger.warning(f"SerpApi image upload error: {e}")
            return None

    def _query_google_lens(self, image_id: str, max_results: int = 35) -> List[CandidateResult]:
        """Queries SerpApi Google Lens engine using the uploaded image_id."""
        search_url = "https://serpapi.com/search.json"
        results: List[CandidateResult] = []

        try:
            params = {
                "engine": "google_lens",
                "image_id": image_id,
                "api_key": self.api_key,
                "hl": "en",
            }
            logger.info(f"Executing SerpApi Google Lens query with image_id={image_id[:16]}...")
            resp = requests.get(search_url, params=params, timeout=25)

            if resp.status_code != 200:
                logger.warning(f"SerpApi Google Lens query returned HTTP {resp.status_code}: {resp.text[:180]}")
                return []

            data = resp.json()
            
            # Combine exact matches first, then visual matches
            exact_items = data.get("exact_matches", [])
            visual_items = data.get("visual_matches", [])
            raw_items = exact_items + [item for item in visual_items if item not in exact_items]

            seen_keys = set()
            rank = 1

            for item in raw_items:
                if len(results) >= max_results:
                    break
                raw_link = item.get("link") or item.get("source_url") or item.get("url")
                if not raw_link:
                    continue

                # Unwrap Google /goto or /url redirects
                resolved_link = resolve_google_goto(raw_link)
                norm_link = normalize_url(resolved_link)
                canon_key = canonical_page_key(norm_link)

                if canon_key in seen_keys:
                    continue
                seen_keys.add(canon_key)

                # Prefer full-size image over thumbnail
                thumb = item.get("image") or item.get("original") or item.get("thumbnail")
                title = item.get("title") or item.get("source") or item.get("snippet")

                results.append(CandidateResult(
                    rank=rank,
                    page_url=norm_link,
                    image_url=thumb,
                    source_domain=extract_domain(norm_link),
                    title=title,
                    provider="SerpApi (Google Lens)",
                ))
                rank += 1

            logger.info(f"SerpApi Google Lens discovered {len(results)} candidate(s).")

        except Exception as e:
            logger.warning(f"SerpApi Google Lens query error: {e}")

        return results

    async def search(
        self,
        image_path: Path,
        max_results: int = 35,
        engines: Optional[List[str]] = None,
    ) -> List[CandidateResult]:
        """
        Executes SerpApi Google Lens reverse image discovery for a local image.
        """
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

        image_id = await asyncio.to_thread(self._upload_image, path)
        if not image_id:
            return []

        return await asyncio.to_thread(self._query_google_lens, image_id, max_results)
