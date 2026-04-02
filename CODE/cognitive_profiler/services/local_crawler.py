import asyncio
import logging
import tempfile
from pathlib import Path
from playwright.async_api import BrowserContext, Page
from cognitive_profiler.data_contracts import DownloadedArtifact

class LocalCrawler:
    def __init__(self, browser_context: BrowserContext):
        self.context = browser_context
        self.logger = logging.getLogger(self.__class__.__name__)

    async def crawl(self, start_url: str, lpa_name: str) -> list[DownloadedArtifact]:
        """
        Explores the start_url to find the Search Entry Point.
        Returns artifacts for the TriageAgent.
        """
        self.logger.info(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        page = await self.context.new_page()
        artifacts = []
        
        try:
            # 1. Capture Homepage
            try:
                await page.goto(start_url, timeout=30000)
                await page.wait_for_load_state("domcontentloaded")
            except Exception as e:
                self.logger.error(f"[REDACTED_BY_SCRIPT]")
                return []

            print(f"[REDACTED_BY_SCRIPT]")
            artifacts.append(await self._capture(page, lpa_name, "homepage"))
            print(f"[REDACTED_BY_SCRIPT]")

            # 2. Extract Candidate Links (Heuristic: Look for "Search" or "Planning")
            print(f"[REDACTED_BY_SCRIPT]")
            links = await page.locator("a[href]").all()
            candidate_urls = set()
            from urllib.parse import urljoin
            
            for link in links:
                try:
                    # Comprehensive Text Extraction (Visible + Accessibility Attributes)
                    visible_text = (await link.inner_text()).lower()
                    aria_label = (await link.get_attribute("aria-label") or "").lower()
                    title_attr = (await link.get_attribute("title") or "").lower()
                    combined_text = f"[REDACTED_BY_SCRIPT]"

                    href = await link.get_attribute("href")
                    
                    if not href or any(x in href.lower() for x in ["javascript:", "mailto:", "tel:", "#"]):
                        continue
                        
                    # Filter for high-probability keywords
                    # "planning" is the critical missing keyword from previous versions.
                    keywords = ["planning", "search", "find", "view", "application", "public access", "register", "building control"]
                    
                    if any(kw in combined_text for kw in keywords):
                        # Robust URL Normalization
                        full_url = urljoin(start_url, href)
                        
                        # Exclude obvious noise (files, social media)
                        if any(ext in full_url.lower() for ext in ['.pdf', '.doc', '.zip', 'facebook.com', 'twitter.com', 'linkedin.com']):
                            continue
                            
                        candidate_urls.add(full_url)
                except Exception:
                    continue

            # 3. Visit Top Candidates (Limit 30 to cast a wider net)
            # PRIORITY SORT: URLs containing "planning" are visited first.
            sorted_candidates = sorted(
                list(candidate_urls), 
                key=lambda u: 0 if "planning" in u.lower() else 1
            )

            self.logger.info(f"[REDACTED_BY_SCRIPT]")
            print(f"[REDACTED_BY_SCRIPT]")
            
            for i, url in enumerate(sorted_candidates[:30]):
                print(f"[REDACTED_BY_SCRIPT]")
                try:
                    await page.goto(url, timeout=15000)
                    await page.wait_for_load_state("domcontentloaded")
                    print(f"[REDACTED_BY_SCRIPT]")
                    artifacts.append(await self._capture(page, lpa_name, f"candidate_{i}"))
                    print(f"[REDACTED_BY_SCRIPT]")
                except Exception:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Crawl failed: {e}")
        finally:
            await page.close()
            
        return artifacts

    async def _capture(self, page: Page, lpa_name: str, tag: str) -> DownloadedArtifact:
        """[REDACTED_BY_SCRIPT]"""
        # Use system temp dir + amaryllis
        temp_dir = Path(tempfile.gettempdir()) / "amaryllis_cache" / lpa_name
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        filename_base = f"[REDACTED_BY_SCRIPT]"
        html_path = temp_dir / f"[REDACTED_BY_SCRIPT]"
        img_path = temp_dir / f"{filename_base}.png"
        
        print(f"[REDACTED_BY_SCRIPT]")
        
        await page.screenshot(path=str(img_path), full_page=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(await page.content())
            
        return DownloadedArtifact(
            url=page.url,
            html_path=str(html_path),
            screenshot_path=str(img_path)
        )