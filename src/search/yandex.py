"""
Reverse-image search provider using Yandex Images via Playwright with stealth evasions.
"""

import asyncio
import logging
from pathlib import Path
from typing import List
from urllib.parse import unquote

from playwright.async_api import async_playwright
from playwright_stealth import Stealth

from src.search.base import BaseSearchEngine, CandidateResult, extract_domain, normalize_url

logger = logging.getLogger(__name__)


class YandexSearchEngine(BaseSearchEngine):
    """Playwright-automated reverse image discovery using Yandex Images."""

    @property
    def provider_name(self) -> str:
        return "Yandex Images (Playwright Stealth)"

    async def search(self, image_path: Path, max_results: int = 30) -> List[CandidateResult]:
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

        results: List[CandidateResult] = []
        stealth = Stealth()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await stealth.apply_stealth_async(page)

            try:
                logger.info("Navigating to https://yandex.com/images/ ...")
                await page.goto("https://yandex.com/images/", wait_until="domcontentloaded", timeout=20000)
                await asyncio.sleep(2.0)

                file_input = await page.query_selector("input.CbirCore-FileInput, input[type='file']")
                if not file_input:
                    cam_btn = await page.query_selector(".cbir-icon, button.input__cbir-button, [aria-label*='image']")
                    if cam_btn:
                        await cam_btn.click()
                        await asyncio.sleep(1.5)
                        file_input = await page.query_selector("input.CbirCore-FileInput, input[type='file']")

                if not file_input:
                    logger.warning("Could not locate file input on Yandex Images.")
                    await browser.close()
                    return []

                logger.info(f"Uploading image: {path.name}")
                await file_input.set_input_files(str(path))
                await file_input.dispatch_event("change")

                # Poll for URL navigation (as verified in test_upload_trigger.py)
                logger.info("Awaiting visual results navigation...")
                for sec in range(15):
                    await asyncio.sleep(1.0)
                    if "search" in page.url or "cbir" in page.url:
                        logger.info(f"Navigated to results at second {sec+1}: {page.url[:80]}")
                        break

                await asyncio.sleep(3.0)
                logger.info(f"Results page URL: {page.url}")

                # Scroll down to trigger lazy loading of candidate visual matches
                await page.evaluate("window.scrollBy(0, 1200)")
                await asyncio.sleep(2.0)

                # Query all anchor links on the results page
                links = await page.query_selector_all("a[href]")
                logger.info(f"Yandex results page contains {len(links)} total links.")

                rank = 1
                seen_urls = set()

                for link_el in links:
                    if len(results) >= max_results:
                        break

                    href = await link_el.get_attribute("href")
                    if not href:
                        continue

                    # Unquote Yandex redirects
                    if "rdr=" in href or "url=" in href:
                        for part in href.split("&"):
                            if part.startswith("url=") or part.startswith("rdr="):
                                try:
                                    href = unquote(part.split("=", 1)[1])
                                except Exception:
                                    pass
                                break

                    if not href.startswith("http://") and not href.startswith("https://"):
                        continue

                    domain = extract_domain(href)
                    if "yandex" in domain or "passport" in domain or "w3.org" in domain:
                        continue

                    norm_url = normalize_url(href)
                    if norm_url in seen_urls:
                        continue
                    seen_urls.add(norm_url)

                    # Extract inner text or image preview
                    title_text = await link_el.inner_text()
                    title = title_text.strip() if title_text else None

                    img_el = await link_el.query_selector("img[src]")
                    img_src = None
                    if img_el:
                        src = await img_el.get_attribute("src")
                        if src and src.startswith("//"):
                            img_src = "https:" + src
                        elif src and src.startswith("http"):
                            img_src = src

                    # Check if the href itself is a direct image file
                    is_image_file = any(
                        norm_url.lower().endswith(ext)
                        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                    )
                    image_url = norm_url if is_image_file else img_src

                    results.append(CandidateResult(
                        rank=rank,
                        page_url=norm_url,
                        image_url=image_url,
                        source_domain=domain,
                        title=title or domain,
                        provider="Yandex Images",
                    ))
                    rank += 1

                logger.info(f"Extracted {len(results)} valid external candidates from Yandex.")

            except Exception as e:
                logger.error(f"Yandex discovery encountered error: {e}")
            finally:
                await browser.close()

        return results
