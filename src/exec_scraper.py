import asyncio
import csv
import json
import logging
import os
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from playwright.async_api import async_playwright, Page, BrowserContext

# --- PATH INJECTION (CRITICAL FOR LOCAL EXECUTION) ---
# Allows src/exec_scraper.py to see cognitive_profiler/ as a module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now we can import agents if needed (for Step 3 placeholder)
# from cognitive_profiler.agents.strategist_agent import StrategistAgent 

# --- CONFIGURATION (ADJUST AS NEEDED) ---
# Assuming running from amaryllis/src or similar relative path
PROJECT_ROOT = Path(parent_dir).parent
CSV_PATH = PROJECT_ROOT / "[REDACTED_BY_SCRIPT]"
LPA_PLANS_DIR = PROJECT_ROOT / "lpa_plans"
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='[REDACTED_BY_SCRIPT]',
    handlers=[
        logging.FileHandler("[REDACTED_BY_SCRIPT]"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AmaryllisExecutor")

class PlanExecutor:
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None

    async def start(self):
        """[REDACTED_BY_SCRIPT]"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        # Create a context that accepts downloads
        self.context = await self.browser.new_context(accept_downloads=True)
        logger.info("Browser session started.")

    async def stop(self):
        """Closes the browser session."""
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()
        logger.info("Browser session closed.")

    async def _annihilate_cookies(self, page: Page):
        """[REDACTED_BY_SCRIPT]"""
        try:
            # 1. Try clicking common "Accept" buttons
            accept_selectors = [
                "button:has-text('Accept all')", 
                "button:has-text('Accept AllCookies')",
                "button:has-text('Accept Necessary')",
                "a:has-text('Accept')",
                "#ccc-notify-accept",
                "#accept-cookies"
            ]
            for sel in accept_selectors:
                if await page.is_visible(sel):
                    logger.debug(f"[REDACTED_BY_SCRIPT]")
                    await page.click(sel, force=True)
                    await page.wait_for_timeout(500)
            
            # 2. Force hide known annoying overlays and fix scroll
            hide_script = """
            () => {
                const elements = document.querySelectorAll('[REDACTED_BY_SCRIPT]');
                elements.forEach(el => el.style.display = 'none');
                document.body.style.overflow = 'auto';
            }
            """
            await page.evaluate(hide_script)
        except Exception as e:
            logger.debug(f"[REDACTED_BY_SCRIPT]")

    async def execute_plan(self, lpa_name: str, app_ref: str, plan: Dict[str, Any]):
        """
        Executes a scraping plan for a specific application reference.
        """
        page = await self.context.new_page()
        
        # Prepare download directory: downloads/{LPA}/{AppRef}/
        # Sanitize folder names
        safe_lpa = re.sub(r'[<>:"/\\|?*]', '_', lpa_name).strip()
        safe_ref = re.sub(r'[<>:"/\\|?*]', '_', app_ref).strip()
        app_download_dir = DOWNLOADS_DIR / safe_lpa / safe_ref
        app_download_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[REDACTED_BY_SCRIPT]")

        try:
            # 1. Execute Navigation Instructions
            instructions = plan.get("instructions", [])
            for step in instructions:
                await self.execute_instruction(page, step, app_ref)

            # 2. Document Extraction & Downloading
            # Check if we are on a results page or details page. 
            # The instructions normally lead to the details/documents page.
            # We use document_page_processing configuration.
            doc_config = plan.get("[REDACTED_BY_SCRIPT]", {})
            if doc_config:
                await self.process_documents(page, doc_config, app_download_dir, app_ref)
            else:
                logger.warning(f"[REDACTED_BY_SCRIPT]")

        except Exception as e:
            import traceback
            logger.error(f"[REDACTED_BY_SCRIPT]")
            # Optional: Screenshot on failure
            try:
                await page.screenshot(path=str(app_download_dir / "error_screenshot.png"))
            except:
                pass
        finally:
            try:
                await page.close()
            except:
                pass

    async def execute_instruction(self, page: Page, instruction: Dict[str, Any], app_ref: str):
        """[REDACTED_BY_SCRIPT]"""
        cmd = instruction.get("command")
        params = instruction.get("params", {})
        desc = instruction.get("description", "No description")
        
        logger.debug(f"[REDACTED_BY_SCRIPT]")

        if cmd == "GOTO_URL":
            response = await page.goto(params["url"], timeout=60000)
            if response and response.status >= 400:
                raise Exception(f"[REDACTED_BY_SCRIPT]")
            
            # Content check for typical "dead end" pages
            content = (await page.content()).lower()
            title = (await page.title()).lower()
            if "page not found" in title or "404" in title or "session has expired" in content:
                raise Exception("[REDACTED_BY_SCRIPT]'Page not found' or 'stale session'.")
                
            await self._annihilate_cookies(page)
        
        elif cmd == "CLICK_ELEMENT":
            await self._annihilate_cookies(page)
            selector = params["selector"]
            # Wait for element to be visible/enabled before clicking
            try:
                await page.wait_for_selector(selector, state="visible", timeout=10000)
                await page.click(selector)
            except Exception as e:
                logger.warning(f"[REDACTED_BY_SCRIPT]")
                try:
                    await page.click(selector, force=True, timeout=5000)
                except Exception as force_e:
                    # Could lead to 404 or page shifts
                    logger.warning(f"[REDACTED_BY_SCRIPT]")
                    raise

        elif cmd == "FILL_INPUT":
            selector = params["selector"]
            value = params["value"].replace("{{REFERENCE_ID}}", app_ref)
            try:
                await page.wait_for_selector(selector, state="visible", timeout=10000)
                await page.fill(selector, value)
            except Exception as e:
                logger.warning(f"[REDACTED_BY_SCRIPT]")
                await page.fill(selector, value, timeout=5000)

        elif cmd == "PRESS_KEY":
            selector = params.get("selector", "body")
            key = params["key"]
            await page.press(selector, key)

        elif cmd == "WAIT_FOR_SELECTOR":
            selector = params["selector"]
            timeout_val = params.get("timeout", 30000)
            try:
                await page.wait_for_selector(selector, timeout=timeout_val)
            except Exception as e:
                logger.warning(f"[REDACTED_BY_SCRIPT]")

        elif cmd == "TRY_CLICK_ELEMENT":
            await self._annihilate_cookies(page)
            selector = params["selector"]
            timeout_val = params.get("timeout", 5000)
            try:
                await page.wait_for_selector(selector, state="visible", timeout=timeout_val)
                await page.click(selector, timeout=timeout_val)
                logger.info(f"[REDACTED_BY_SCRIPT]")
            except Exception as e:
                logger.info(f"[REDACTED_BY_SCRIPT]")

    async def process_documents(self, page: Page, config: Dict[str, Any], download_dir: Path, app_ref: str):
        """[REDACTED_BY_SCRIPT]"""
        # Determine strategy: Batch or Iterative
        batch_config = config.get("batch_processing")
        
        if batch_config:
            logger.info("[REDACTED_BY_SCRIPT]")
            await self._download_batch(page, batch_config, download_dir, app_ref)
        else:
            logger.info("[REDACTED_BY_SCRIPT]")
            await self._download_iterative(page, config, download_dir, app_ref)

    async def _download_batch(self, page: Page, batch_config: Dict[str, Any], download_dir: Path, app_ref: str):
        # 1. Select All Checkbox
        select_all_selector = batch_config.get("select_all_selector")
        if select_all_selector:
            if await page.is_visible(select_all_selector):
                await page.check(select_all_selector)
            else:
                logger.warning("[REDACTED_BY_SCRIPT]")
        
        # 2. Click Download Button
        download_btn_selector = batch_config.get("[REDACTED_BY_SCRIPT]")
        if download_btn_selector:
            async with page.expect_download() as download_info:
                await page.click(download_btn_selector)
            
            download = await download_info.value
            # Rename zip/file
            ext = os.path.splitext(download.suggested_filename)[1]
            filename = f"[REDACTED_BY_SCRIPT]"
            # Remove invalid chars from filename
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
            
            await download.save_as(download_dir / filename)
            logger.info(f"[REDACTED_BY_SCRIPT]")

    async def _download_iterative(self, page: Page, config: Dict[str, Any], download_dir: Path, app_ref: str):
        # Extract blueprints
        container_sel = config.get("[REDACTED_BY_SCRIPT]")
        link_bp = config.get("[REDACTED_BY_SCRIPT]", {})
        type_bp = config.get("[REDACTED_BY_SCRIPT]", {})
        date_bp = config.get("[REDACTED_BY_SCRIPT]", {}) # Optional
        
        if not container_sel:
            logger.warning("[REDACTED_BY_SCRIPT]")
            return

        while True:
            # Locate all document rows/items
            await page.wait_for_selector(container_sel, timeout=10000)
            rows = await page.query_selector_all(container_sel)
            logger.info(f"[REDACTED_BY_SCRIPT]")

            for i, row in enumerate(rows):
                try:
                    # Extract Metadata for filename
                    doc_type = "Document"
                    if type_bp and type_bp.get("selector"):
                        # Relative selector from row
                        type_el = await row.query_selector(type_bp["selector"])
                        if type_el:
                            doc_type = (await type_el.text_content()).strip()

                    doc_date = ""
                    if date_bp and date_bp.get("selector"):
                         date_el = await row.query_selector(date_bp["selector"])
                         if date_el:
                             doc_date = (await date_el.text_content()).strip().replace("/", "-")
                    
                    # Extract Link Element to click
                    link_sel = link_bp.get("selector")
                    if not link_sel: continue
                    
                    link_el = await row.query_selector(link_sel)
                    if not link_el: continue

                    # Construct Filename
                    # Format: {AppRef}_{Date}_{Type}_{Index}.ext
                    # Note: We don't know extension until download starts, usually PDF
                    safe_type = re.sub(r'[<>:"/\\|?*]', '_', doc_type)[:30] # Truncate if too long
                    safe_date = re.sub(r'[<>:"/\\|?*]', '_', doc_date)
                    prefix = f"{app_ref}_{safe_date}_{safe_type}_{i}"
                    
                    extraction_method = link_bp.get("extraction_method", "CLICK")
                    
                    if extraction_method == "ATTRIBUTE":
                        attr_name = link_bp.get("attribute_name", "href")
                        file_url = await link_el.get_attribute(attr_name)
                        
                        if file_url:
                            # Convert to absolute URL
                            from urllib.parse import urljoin
                            full_url = urljoin(page.url, file_url)
                            logger.info(f"[REDACTED_BY_SCRIPT]")
                            
                            # Fetch bypassing expect_download (handles inline PDF viewing)
                            response = await page.request.get(full_url, timeout=30000)
                            if response.status == 200:
                                ext = ".pdf"
                                cd = response.headers.get("content-disposition", "")
                                if cd and "filename=" in cd:
                                    match = re.search(r'filename="?([^"]+)"?', cd)
                                    if match:
                                        _, parsed_ext = os.path.splitext(match.group(1))
                                        if parsed_ext: ext = parsed_ext
                                elif "." in file_url.split("/")[-1]:
                                    parsed_ext = os.path.splitext(file_url.split("/")[-1])[1]
                                    if parsed_ext and len(parsed_ext) <= 5:
                                        ext = parsed_ext
                                        
                                final_name = f"{prefix}{ext}"
                                body = await response.body()
                                with open(download_dir / final_name, "wb") as f:
                                    f.write(body)
                                logger.info(f"[REDACTED_BY_SCRIPT]")
                            else:
                                logger.warning(f"[REDACTED_BY_SCRIPT]")
                        else:
                            logger.warning(f"Empty attribute '{attr_name}'[REDACTED_BY_SCRIPT]")
                    else:
                        # Initiate Download (Standard Click fallback)
                        async with page.expect_download() as download_info:
                            # Some sites open in new tab, some download directly. 
                            # This expects a 'download' event. 
                            # If the link opens a PDF in viewer, we might need a different approach (checking URL extension)
                            # but start with standard click->download.
                            await link_el.click()
                        
                        download = await download_info.value
                        ext = os.path.splitext(download.suggested_filename)[1]
                        if not ext: ext = ".pdf" # Default fallback
                        
                        final_name = f"{prefix}{ext}"
                        await download.save_as(download_dir / final_name)
                        logger.info(f"[REDACTED_BY_SCRIPT]")
                    
                except Exception as e:
                    logger.warning(f"[REDACTED_BY_SCRIPT]")

            # Pagination Check
            pagination = config.get("[REDACTED_BY_SCRIPT]")
            next_btn = None
            if pagination and pagination.get("selector"):
                next_btn_sel = pagination["selector"]
                # specific check for 'Next' text or class
                next_btn = await page.query_selector(next_btn_sel)

            if next_btn and await next_btn.is_visible():
                logger.info("[REDACTED_BY_SCRIPT]")
                await next_btn.click()
                await page.wait_for_load_state("networkidle")
            else:
                break # No more pages

async def run_scraper():
    # 1. Initialize Executor
    executor = PlanExecutor(headless=False) # Headless=False to see it working
    await executor.start()

    # 2. Read CSV
    if not CSV_PATH.exists():
        logger.error(f"[REDACTED_BY_SCRIPT]")
        return

    logger.info(f"[REDACTED_BY_SCRIPT]")
    
    apps_to_process = []
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = row.get("[REDACTED_BY_SCRIPT]", "").strip()
            lpa = row.get("Planning Authority", "").strip()
            
            # Filter Logic: Must have Ref ID and LPA
            if ref and lpa: 
                apps_to_process.append((lpa, ref))

    logger.info(f"[REDACTED_BY_SCRIPT]")

    # 3. Execution Loop
    for lpa_name, app_ref in apps_to_process:
        # Check if Plan Exists
        # Naive check: match exact filename first
        plan_path = LPA_PLANS_DIR / f"{lpa_name}.json"
        
        # If not exact match, try normalization (simple case)
        if not plan_path.exists():
             # Try finding folder with same name
             possible_folder = LPA_PLANS_DIR / lpa_name
             if possible_folder.is_dir():
                 plan_path = possible_folder / f"{lpa_name}.json"

        if plan_path.exists():
            try:
                with open(plan_path, 'r', encoding='utf-8') as f:
                    plan_data = json.load(f)
                
                await executor.execute_plan(lpa_name, app_ref, plan_data)
            except Exception as e:
                logger.error(f"[REDACTED_BY_SCRIPT]")
                if "[REDACTED_BY_SCRIPT]" in str(e):
                    logger.info("[REDACTED_BY_SCRIPT]")
                    try:
                        await executor.stop()
                    except:
                        pass
                    await executor.start()
        else:
            # TODO: Call StrategistAgent if plan missing
            logger.warning(f"[REDACTED_BY_SCRIPT]")
    
    await executor.stop()

if __name__ == "__main__":
    asyncio.run(run_scraper())
