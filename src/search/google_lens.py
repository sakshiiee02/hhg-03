"""
Reverse-image search provider using Google Lens via Playwright with stealth evasions.
"""

import asyncio
import logging
from pathlib import Path
from typing import List

from playwright.async_api import async_playwright
from playwright_stealth import Stealth

from src.search.base import BaseSearchEngine, CandidateResult, extract_domain, normalize_url

logger = logging.getLogger(__name__)


class GoogleLensSearchEngine(BaseSearchEngine):
    """Playwright-automated reverse image discovery using Google Lens."""

    @property
    def provider_name(self) -> str:
        return "Google Lens (Playwright Stealth)"

    async def search(self, image_path: Path, max_results: int = 30) -> List[CandidateResult]:
        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")

        results: List[CandidateResult] = []
        stealth = Stealth()

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                    "--window-size=1920,1080",
                ],
            )
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            )
            page = await context.new_page()
            await stealth.apply_stealth_async(page)

            try:
                logger.info("Navigating to Google Images...")
                await page.goto("https://images.google.com/", wait_until="domcontentloaded", timeout=20000)
                await asyncio.sleep(1.0)

                # Locate the lens/camera button
                cam_btn = await page.query_selector("div[aria-label*='Search by image'], div.nDcEnd, [jsname='R55cEe']")
                if cam_btn:
                    await cam_btn.click()
                    await asyncio.sleep(1.0)

                file_input = await page.query_selector("input[type='file']")
                if not file_input:
                    logger.warning("Could not find file input on Google Lens.")
                    await browser.close()
                    return []

                logger.info(f"Uploading image to Google Lens: {path.name}")
                await file_input.set_input_files(str(path))

                # Check if Google redirected to anti-bot CAPTCHA / sorry page
                if "google.com/sorry" in page.url or "sorry/index" in page.url:
                    logger.warning(
                        "Google Lens anti-bot challenge triggered (google.com/sorry). "
                        "Google blocks automated headless image uploads from public IP addresses. "
                        "To use Google Lens reliably, set SERPAPI_API_KEY in .env or use Yandex/Auto router."
                    )
                    await browser.close()
                    return []

                # Wait for Lens visual matches panel
                try:
                    await page.wait_for_url("**/lens.google.com/**", timeout=15000)
                except Exception:
                    await asyncio.sleep(4.0)

                if "google.com/sorry" in page.url or "sorry/index" in page.url:
                    logger.warning("Google Lens anti-bot CAPTCHA triggered on upload.")
                    await browser.close()
                    return []

                await page.wait_for_load_state("domcontentloaded")
                await asyncio.sleep(2.5)

                # Query visual matches links
                links = await page.query_selector_all("a[href*='http'][data-lens-type], a.LBdpif, [jsname='fKznid'] a")
                logger.info(f"Google Lens returned {len(links)} potential match links.")

                rank = 1
                seen_urls = set()

                for link in links:
                    if len(results) >= max_results:
                        break

                    href = await link.get_attribute("href")
                    if not href or not href.startswith("http") or "google." in extract_domain(href):
                        continue

                    norm_url = normalize_url(href)
                    if norm_url in seen_urls:
                        continue
                    seen_urls.add(norm_url)

                    # Extract thumbnail and title if available
                    img_el = await link.query_selector("img[src*='http']")
                    img_url = await img_el.get_attribute("src") if img_el else None
                    title_text = await link.inner_text()
                    title = title_text.strip() if title_text else None

                    results.append(CandidateResult(
                        rank=rank,
                        page_url=norm_url,
                        image_url=img_url,
                        source_domain=extract_domain(norm_url),
                        title=title,
                        provider="Google Lens",
                    ))
                    rank += 1

            except Exception as e:
                logger.error(f"Google Lens discovery encountered error: {e}")
            finally:
                await browser.close()

        return results
