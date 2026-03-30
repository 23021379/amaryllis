import logging
from typing import final, Any, Literal
from urllib.parse import urlparse
from pathlib import Path
from playwright._impl._errors import Error as PlaywrightImplError
from playwright.async_api import BrowserContext, Page, TimeoutError as PlaywrightTimeoutError
import re
import tempfile
import copy
from pydantic import ValidationError as PydanticValidationError

from ..data_contracts import DraftPlan, ValidationReport, ProbeResult, ExecutionPlan, ResultsPageProcessing, DetailPageProcessing, Instruction, ExtractionBlueprint
from ..exceptions import ValidationError


import asyncio

@final
class ValidationAgent:
    """
    Executes a DraftPlan in a real headless browser to verify its
    executability and correctness.
    """
    def __init__(self, browser_context: BrowserContext, default_timeout_ms: int = 10000):
        self.browser_context = browser_context
        self.default_timeout_ms = default_timeout_ms
        self.logger = logging.getLogger(self.__class__.__name__)

    async def _resolve_locator(self, page: Page, params: dict, state: str = 'visible', timeout_ms: int = None):
        """[REDACTED_BY_SCRIPT]"""
        selector = params.get('selector')
        frame_name = params.get('frame_name')
        if timeout_ms is None: timeout_ms = self.default_timeout_ms
        if not selector: raise ValidationError("Instruction requires 'selector'")

        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        import asyncio

        if frame_name:
            target = page.frame_locator(f"iframe[name='{frame_name}'], frame[name='{frame_name}']").locator(selector).first
            await target.wait_for(state=state, timeout=timeout_ms)
            return target

        async def try_loc(loc, t):
            try:
                await loc.wait_for(state=state, timeout=t)
                return loc
            except Exception:
                return None

        # 1. Fast main page
        target = page.locator(selector).first
        res = await try_loc(target, 2000)
        if res: return res

        # 2. Check frames
        tasks = []
        for child_frame in page.frames:
            if child_frame != page.main_frame:
                tasks.append(try_loc(child_frame.locator(selector).first, 3000))
        if tasks:
            results = await asyncio.gather(*tasks)
            for r in results:
                if r: return r

        # 3. Final wait to cleanly throw
        await target.wait_for(state=state, timeout=max(1000, timeout_ms - 2000))
        return target

    async def _take_clean_screenshot(self, page: Page, path: str | Path):
        """[REDACTED_BY_SCRIPT]"""
        cleanup_js = """
        () => {
            // 1. Target Cookiebot and general cookie banners
            const cookieSelectors = [
                '#CybotCookiebotDialog', '#ccc-module', '.cookie-banner', '#cookie-banner', 
                '.cc-window', '[aria-label*="cookie" i]', '[id*="cookie" i][class*="banner" i]',
                '.CivicaCookieBanner', '.cb-banner'
            ];
            
            document.querySelectorAll(cookieSelectors.join(', ')).forEach(el => {
                try { el.remove(); } catch(e) {}
            });

            // 2. Expand scrollable areas so full_page screenshot captures everything inside them
            document.querySelectorAll('*').forEach(el => {
                try {
                    const style = window.getComputedStyle(el);
                    if (style.overflow === 'auto' || style.overflow === 'scroll' || style.overflowY === 'auto' || style.overflowY === 'scroll') {
                        if(el.tagName.toLowerCase() !== 'body' && el.tagName.toLowerCase() !== 'html') {
                            el.style.overflow = 'visible';
                            el.style.overflowY = 'visible';
                            el.style.maxHeight = 'none';
                            el.style.height = 'auto';
                        }
                    }
                } catch(e) {}
            });
        }
        """
        try:
            await page.evaluate(cleanup_js)
            await asyncio.sleep(0.2) # Yield to event loop to allow render reflow
        except Exception:
            pass # Ignore if JS fails (e.g. page closed)
        
        await page.screenshot(path=str(path), full_page=True)

    async def _debug_snapshot(self, page: Page, prefix: str, save_dir: str | None = None) -> None:
        """[REDACTED_BY_SCRIPT]"""
        try:
            if save_dir:
                target_dir = Path(save_dir)
            else:
                target_dir = Path(tempfile.gettempdir()) / "amaryllis_debug"
            
            target_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(asyncio.get_event_loop().time() * 1000)
            
            # Save Screenshot
            img_path = target_dir / f"[REDACTED_BY_SCRIPT]"
            await self._take_clean_screenshot(page, img_path)
            
            print(f"[REDACTED_BY_SCRIPT]") # Indented for readability in loop
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]")


    async def execute_probe(self, probe_plan: DraftPlan, temp_dir: str) -> ProbeResult:
        """[REDACTED_BY_SCRIPT]"""
        self.logger.info(f"[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        page = await self.browser_context.new_page()
        
        # --- SUBSTITUTION LOGIC (Deep & Global) ---
        # We recursively swap {{REFERENCE_ID}} in ALL parameters (selectors, values, etc.)
        live_instructions = copy.deepcopy(probe_plan.instructions)
        real_ref_id = probe_plan.target_address
        
        # Determine the fallback origin URL if hallucinatory variables are inserted
        safe_url = "about:blank"
        if probe_plan.source_triage_result and probe_plan.source_triage_result.candidate_urls:
            # Prefer search page if available, else homepage
            safe_url = probe_plan.source_triage_result.candidate_urls.get("search_page", probe_plan.source_triage_result.candidate_urls.get("homepage", "about:blank"))

        for i, instr in enumerate(live_instructions):
            params = instr.get('params', {})
            for key, val in params.items():
                if isinstance(val, str):
                    if "{{REFERENCE_ID}}" in val:
                        # Perform replacement
                        val = val.replace("{{REFERENCE_ID}}", real_ref_id)
                        self.logger.info(f"[REDACTED_BY_SCRIPT]'{key}' -> '{val}'")
                    if "$search_page_url" in val:
                        # LLM Hallucinated variable substitution
                        val = val.replace("$search_page_url", safe_url)
                        self.logger.info(f"[REDACTED_BY_SCRIPT]'{key}' -> '{val}'")
                    params[key] = val

        try:
            for i, instruction in enumerate(live_instructions):
                is_final_assert = (
                    i == len(live_instructions) - 1 and
                    instruction.get("command") == "WAIT_FOR_SELECTOR"
                )
                # Update 'page' reference if instruction (click) changed tabs
                print(f"[REDACTED_BY_SCRIPT]'command'[REDACTED_BY_SCRIPT]'params')}")
                page = await self._execute_instruction(instruction, page, is_final_assert=is_final_assert)
                # Snapshot with index for sequence tracking
                await self._debug_snapshot(page, f"[REDACTED_BY_SCRIPT]", save_dir=temp_dir)
            
            # Probe was successful, now capture the state
            html = await page.content()
            final_url = page.url
            _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="probe_capture_")
            await self._take_clean_screenshot(page, screenshot_path)
            self.logger.info(f"[REDACTED_BY_SCRIPT]")
            
            return ProbeResult(
                is_successful=True,
                probe_plan=probe_plan.instructions,
                final_url=final_url,
                captured_html=html,
                captured_screenshot_path=screenshot_path
            )
        except Exception as e:
            self.logger.error(f"[REDACTED_BY_SCRIPT]")
            
            # --- BLACK BOX RECORDING ---
            # Attempt to capture the state at the moment of crash
            fail_html = None
            fail_img = None
            try:
                fail_html = await page.content()
                _, fail_img = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="crash_dump_")
                await self._take_clean_screenshot(page, fail_img)
                self.logger.info(f"[REDACTED_BY_SCRIPT]")
            except Exception as capture_error:
                self.logger.error(f"[REDACTED_BY_SCRIPT]")

            return ProbeResult(
                is_successful=False, 
                probe_plan=probe_plan.instructions, 
                failure_reason=str(e),
                captured_html=fail_html,
                captured_screenshot_path=fail_img
            )
        finally:
            await page.close()


    async def run(self, draft_plan: DraftPlan, temp_dir: str) -> ValidationReport:
        """Performs a 'dry run'[REDACTED_BY_SCRIPT]"""
        self.logger.info("[REDACTED_BY_SCRIPT]")
        print(f"[REDACTED_BY_SCRIPT]")
        page = await self.browser_context.new_page()
        initial_url = page.url 
        extracted_links: list[str] = []

        # --- SUBSTITUTION LOGIC (Deep & Global) ---
        test_instructions = copy.deepcopy(draft_plan.instructions)
        real_ref_id = draft_plan.target_address
        
        # Determine the fallback origin URL if hallucinatory variables are inserted
        safe_url = "about:blank"
        if draft_plan.source_triage_result and draft_plan.source_triage_result.candidate_urls:
            # Prefer search page if available, else homepage
            safe_url = draft_plan.source_triage_result.candidate_urls.get("search_page", draft_plan.source_triage_result.candidate_urls.get("homepage", "about:blank"))

        for i, instr in enumerate(test_instructions):
            params = instr.get('params', {})
            for key, val in params.items():
                if isinstance(val, str):
                    if "{{REFERENCE_ID}}" in val:
                        val = val.replace("{{REFERENCE_ID}}", real_ref_id)
                    if "$search_page_url" in val:
                        val = val.replace("$search_page_url", safe_url)
                        self.logger.info(f"[REDACTED_BY_SCRIPT]'{key}' -> '{val}'")
                    params[key] = val

        try:
            # --- Stage 1: Execute Navigation Plan ---
            for i, instruction in enumerate(test_instructions):
                self.logger.info(f"[REDACTED_BY_SCRIPT]")
                print(f"[REDACTED_BY_SCRIPT]'command'[REDACTED_BY_SCRIPT]'params')}")
                try:
                    page = await self._execute_instruction(instruction, page)
                    await self._debug_snapshot(page, f"step_{i+1}_success", save_dir=temp_dir)
                except PlaywrightTimeoutError as e:
                    self.logger.warning(f"[REDACTED_BY_SCRIPT]")
                    failure_html = await page.content()
                    _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="failure_")
                    await self._take_clean_screenshot(page, screenshot_path)
                    self.logger.info(f"[REDACTED_BY_SCRIPT]")

                    # --- Mandate C-2: THE DIAGNOSTIC SCALPEL ---
                    is_final_heracles_assert = (
                        i == len(draft_plan.instructions) - 1 and
                        instruction.get("command") == "WAIT_FOR_SELECTOR"
                    )

                    if is_final_heracles_assert:
                        failure_type = "[REDACTED_BY_SCRIPT]"
                        analysis_notes = (
                            "Analysis: The plan's final assertion (WAIT_FOR_SELECTOR) failed. "
                            "[REDACTED_BY_SCRIPT]"
                            "(e.g., 'No results found'[REDACTED_BY_SCRIPT]"
                        )
                    else:
                        # Fallback to existing Sentinel diagnostics
                        failure_type, analysis_notes = self._diagnose_timeout_failure(failure_html)
                    
                    return ValidationReport(
                        is_valid=False, original_plan=draft_plan, failure_index=i,
                        failure_type=failure_type,
                        failure_reason=f"[REDACTED_BY_SCRIPT]",
                        failure_context_html=failure_html,
                        failure_screenshot_path=screenshot_path,
                        failure_stage="NAVIGATION",
                        failure_analysis_notes=analysis_notes
                    )
                except Exception as e:
                    self.logger.warning(f"[REDACTED_BY_SCRIPT]")
                    error_str = str(e).lower()
                    failure_type: Literal["TIMEOUT", "VALIDATION_ERROR", "AMBIGUOUS_SELECTOR", "UNKNOWN"]

                    if "[REDACTED_BY_SCRIPT]" in error_str:
                        failure_type = "AMBIGUOUS_SELECTOR"
                    else:
                        failure_type = "VALIDATION_ERROR"
                    
                    failure_html = await page.content()
                    _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="failure_")
                    await self._take_clean_screenshot(page, screenshot_path)
                    self.logger.info(f"[REDACTED_BY_SCRIPT]")

                    return ValidationReport(
                        is_valid=False, original_plan=draft_plan, failure_index=i, failure_type=failure_type,
                        failure_reason=f"[REDACTED_BY_SCRIPT]",
                        failure_context_html=failure_html,
                        failure_screenshot_path=screenshot_path,
                        failure_stage="NAVIGATION"
                    )
            
            # --- MANDATE P-1: "PROOF-OF-SEARCH" VALIDATION ---
            final_url = page.url
            if not self._validate_search_intent(initial_url, final_url, draft_plan.instructions):
                self.logger.error("[REDACTED_BY_SCRIPT]")
                failure_html = await page.content()
                _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="mission_failure_")
                await self._take_clean_screenshot(page, screenshot_path)
                return ValidationReport(
                    is_valid=False,
                    original_plan=draft_plan,
                    failure_index=len(draft_plan.instructions),
                    failure_type="MISSION_FAILURE",
                    failure_reason="[REDACTED_BY_SCRIPT]",
                    failure_context_html=failure_html,
                    failure_screenshot_path=screenshot_path,
                    failure_stage="NAVIGATION" # This is a navigation-level failure
                )

            # --- Stage 2: Validate Results Page Blueprints ---
            self.logger.info("[REDACTED_BY_SCRIPT]")
            results_processing = draft_plan.results_page_processing
            
            if results_processing:
                try:
                    # First, ensure the container exists
                    container_selector = results_processing.get('[REDACTED_BY_SCRIPT]')
                    container_locator = page.locator(container_selector)
                    await container_locator.first.wait_for(state='visible', timeout=self.default_timeout_ms)
                    
                    # Then, validate the link blueprint within the scope of the first container
                    link_blueprint = results_processing.get('listing_link_blueprint')
                    await self._validate_blueprint(link_blueprint, page, scope=container_locator.first, blueprint_name="listing_link_blueprint")
                    
                    # If valid, extract the link for the next stage
                    link_attr = link_blueprint.get('attribute_name') or 'href'
                    first_link_href = await container_locator.first.locator(link_blueprint['selector']).first.get_attribute(link_attr)
                    
                    from urllib.parse import urljoin
                    extracted_links.append(urljoin(page.url, first_link_href))

                    # --- PROJECT ATLAS/MERIDIAN VALIDATION (MANDATORY INSERTION) ---
                    [REDACTED_BY_SCRIPT] = results_processing.get('[REDACTED_BY_SCRIPT]')
                    if pagination_blueprint:
                        self.logger.info("[REDACTED_BY_SCRIPT]")
                        pagination_selector = pagination_blueprint.get('next_page_selector')
                        await page.locator(pagination_selector).first.wait_for(state='visible', timeout=self.default_timeout_ms)
                        self.logger.info("[REDACTED_BY_SCRIPT]")
                    # --- END INSERTION ---

                except Exception as e:
                    self.logger.warning(f"[REDACTED_BY_SCRIPT]")
                    failure_html = await page.content()
                    _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="failure_")
                    await self._take_clean_screenshot(page, screenshot_path)
                    self.logger.info(f"[REDACTED_BY_SCRIPT]")
                    return ValidationReport(
                        is_valid=False, original_plan=draft_plan, failure_index=len(draft_plan.instructions),
                        failure_type="VALIDATION_ERROR",
                        # UPDATED: More specific error reason
                        failure_reason=f"[REDACTED_BY_SCRIPT]",
                        failure_context_html=failure_html,
                        failure_screenshot_path=screenshot_path,
                        failure_stage="RESULTS_PAGE"
                    )

            ## --- Stage 3: Validate Detail Page Blueprints (Project ANVIL Refactor) ---
            self.logger.info("[REDACTED_BY_SCRIPT]")
            detail_blueprints = draft_plan.detail_page_processing or {}

            # --- PROJECT SENTRY DIRECTIVE: Validate only what is present. ---
            if detail_blueprints:
                self.logger.info("[REDACTED_BY_SCRIPT]")
                if not extracted_links:
                     raise ValidationError("[REDACTED_BY_SCRIPT]")
                
                detail_page_url = extracted_links[0]
                try:
                    await page.goto(detail_page_url, timeout=self.default_timeout_ms)
                    
                    failed_blueprint_keys: list[str] = []
                    
                    for name, blueprint in detail_blueprints.items():
                        if not blueprint: continue
                        try:
                            self.logger.info(f"[REDACTED_BY_SCRIPT]")
                            await self._validate_blueprint(blueprint, page, blueprint_name=name)
                        except Exception as e:
                            # AMPUTATE, DON'T RESUSCITATE
                            self.logger.warning(f"[REDACTED_BY_SCRIPT]'{name}'[REDACTED_BY_SCRIPT]")
                            failed_blueprint_keys.append(name)
                    
                    # --- Post-Iteration Review ---
                    if not failed_blueprint_keys:
                        # All blueprints passed. Success. The final return is handled outside this block.
                        self.logger.info("[REDACTED_BY_SCRIPT]")
                    elif len(failed_blueprint_keys) == len(detail_blueprints):
                        # CATASTROPHIC FAILURE: All blueprints failed. Trigger self-healing.
                        self.logger.error("[REDACTED_BY_SCRIPT]")
                        failure_html = await page.content()
                        _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="failure_")
                        await self._take_clean_screenshot(page, screenshot_path)
                        return ValidationReport(
                            is_valid=False, original_plan=draft_plan, failure_index=len(draft_plan.instructions) + 1,
                            failure_type="VALIDATION_ERROR",
                            failure_reason=f"[REDACTED_BY_SCRIPT]",
                            failure_context_html=failure_html, failure_screenshot_path=screenshot_path, failure_stage="DETAIL_PAGE"
                        )
                    else:
                        # PARTIAL SUCCESS: Harden the plan, then perform final integrity check.
                        self.logger.info(f"[REDACTED_BY_SCRIPT]")
                        hardened_plan = copy.deepcopy(draft_plan)
                        for key in failed_blueprint_keys:
                            hardened_plan.detail_page_processing.pop(key, None)
                        
                        # --- PROJECT GENESIS: SIMPLIFIED QUARANTINE GATE ---
                        try:
                            self.logger.info("[REDACTED_BY_SCRIPT]")
                            for name, blueprint in hardened_plan.detail_page_processing.items():
                                if blueprint: # Ensure we don't validate nulls
                                    ExtractionBlueprint.model_validate(blueprint)
                            self.logger.info("[REDACTED_BY_SCRIPT]")
                            # --- PROJECT PHOENIX DIRECTIVE: Capture success state for PARTIAL success ---
                            final_url = page.url
                            final_html = await page.content()
                            _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="success_capture_")
                            await self._take_clean_screenshot(page, screenshot_path)
                            self.logger.info(f"[REDACTED_BY_SCRIPT]")

                            return ValidationReport(
                                is_valid=True,
                                original_plan=hardened_plan,
                                final_url=final_url,
                                final_html=final_html,
                                final_screenshot_path=screenshot_path
                            )
                        except PydanticValidationError as e:
                            self.logger.error(f"[REDACTED_BY_SCRIPT]")
                            return ValidationReport(
                                is_valid=False, original_plan=draft_plan,
                                failure_index=len(draft_plan.instructions) + 1,
                                failure_type="VALIDATION_ERROR",
                                failure_reason=f"[REDACTED_BY_SCRIPT]",
                                failure_context_html=await page.content(),
                                failure_screenshot_path=None,
                                failure_stage="DETAIL_PAGE"
                            )
                except Exception as e:
                    # This block catches failures in page navigation or UNEXPECTED code errors.
                    self.logger.error(f"[REDACTED_BY_SCRIPT]", exc_info=True)
                    failure_html = await page.content()
                    _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="failure_")
                    await self._take_clean_screenshot(page, screenshot_path)
                    
                    if isinstance(e, (NameError, TypeError)):
                        raise ValidationError(f"[REDACTED_BY_SCRIPT]") from e

                    return ValidationReport(
                        is_valid=False, original_plan=draft_plan, failure_index=len(draft_plan.instructions) + 1,
                        failure_type="VALIDATION_ERROR",
                        failure_reason=f"[REDACTED_BY_SCRIPT]",
                        failure_context_html=failure_html, failure_screenshot_path=screenshot_path, failure_stage="DETAIL_PAGE"
                    )

            self.logger.info("[REDACTED_BY_SCRIPT]")
            # --- PROJECT PHOENIX DIRECTIVE: Capture success state ---
            final_url = page.url
            final_html = await page.content()
            _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="success_capture_")
            await self._take_clean_screenshot(page, screenshot_path)
            self.logger.info(f"[REDACTED_BY_SCRIPT]")

            return ValidationReport(
                is_valid=True,
                original_plan=draft_plan,
                final_url=final_url,
                final_html=final_html,
                final_screenshot_path=screenshot_path
            )
        finally:
            await page.close()

    async def _detect_and_handle_challenge(self, page: Page) -> bool:
        """
        Detects Cloudflare/Turnstile challenges and attempts to pass them.
        Returns True if a challenge was detected and a mitigation attempt was made.
        """
        try:
            # 1. Quick Content Check
            content = await page.content()
            content_lower = content.lower()
            
            CHALLENGE_KEYWORDS = [
                "[REDACTED_BY_SCRIPT]", "cloudflare", "challenge-stage", 
                "turnstile", "security check", "please wait...", "[REDACTED_BY_SCRIPT]"
            ]
            
            # If no keywords found, it's not a challenge page
            if not any(k in content_lower for k in CHALLENGE_KEYWORDS):
                return False

            self.logger.warning("[REDACTED_BY_SCRIPT]")
            
            # 2. Locate the Challenge Checkbox (often in an iframe or shadow root)
            # Strategy A: Iterate all frames looking for the Cloudflare/Turnstile specific checkbox
            challenge_handled = False
            
            # We explicitly check for iframes that look like turnstile or cloudflare
            frame_candidates = [f for f in page.frames if "cloudflare" in f.url or "turnstile" in f.url or "challenge" in f.url]
            
            # If no obvious frames, check all frames just in case
            frames_to_check = frame_candidates if frame_candidates else page.frames
            
            for frame in frames_to_check:
                try:
                    # Common selectors for the checkbox input or label
                    checkbox_selectors = [
                        "input[type='checkbox']", 
                        ".ctp-checkbox-label",
                        "[REDACTED_BY_SCRIPT]",
                        "label:has-text('Verify you are human')",
                         # Turnstile often uses shadow DOM, verifying implicit handling
                    ]
                    
                    for sel in checkbox_selectors:
                         locator = frame.locator(sel).first
                         if await locator.isVisible():
                             self.logger.info(f"[REDACTED_BY_SCRIPT]'{frame.name or frame.url}'. Clicking '{sel}'...")
                             await locator.click(force=True, timeout=5000)
                             challenge_handled = True
                             break
                    
                    if challenge_handled: break
                except Exception:
                    continue

            # Strategy B: If frame logic failed, try main page text click (fallback)
            if not challenge_handled:
                verify_text = page.get_by_text("[REDACTED_BY_SCRIPT]")
                if await verify_text.is_visible():
                    self.logger.info("Clicking 'Verify you are human' text on main page.")
                    await verify_text.click(force=True)
                    challenge_handled = True

            if challenge_handled:
                # Wait a bit for the challenge to process and redirect
                self.logger.info("[REDACTED_BY_SCRIPT]")
                await page.wait_for_timeout(5000) 
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"[REDACTED_BY_SCRIPT]")
            return False

    def _diagnose_timeout_failure(self, html_content: str) -> tuple[str, str | None]:
        """[REDACTED_BY_SCRIPT]"""
        OBSTRUCTION_KEYWORDS = [
            'cookie', 'consent', 'privacy', 'accept', 'agree', 'continue',
            'gdpr', 'ccpa', 'banner', 'dialog', '[REDACTED_BY_SCRIPT]', 'cloudflare', 'security check'
        ]
        
        html_lower = html_content.lower()
        found_keywords = [kw for kw in OBSTRUCTION_KEYWORDS if kw in html_lower]

        if found_keywords:
            # Check specifically for Security Challenge keywords first
            if any(k in found_keywords for k in ['[REDACTED_BY_SCRIPT]', 'cloudflare', 'security check']):
                 notes = (
                     "[REDACTED_BY_SCRIPT]"
                     "[REDACTED_BY_SCRIPT]"
                     "[REDACTED_BY_SCRIPT]"
                 )
                 self.logger.warning("[REDACTED_BY_SCRIPT]")
                 return "SECURITY_CHALLENGE", notes

            # --- Mandate S-2: Enhanced Diagnostic Hint ---
            notes = (
                "[REDACTED_BY_SCRIPT]"
                "[REDACTED_BY_SCRIPT]"
                "[REDACTED_BY_SCRIPT]"
                "[REDACTED_BY_SCRIPT]'Enter' is more appropriate."
            )
            self.logger.warning("[REDACTED_BY_SCRIPT]")
            return "[REDACTED_BY_SCRIPT]", notes
        
        # --- Mandate S-3: The "Wrong Neighborhood" Detector ---
        ZERO_RESULT_KEYWORDS = ["no results found", "no application found", "search returned no results", "zero results"]
        if any(z in html_lower for z in ZERO_RESULT_KEYWORDS):
             notes = (
                 "[REDACTED_BY_SCRIPT]'No Results'. "
                 "[REDACTED_BY_SCRIPT]"
                 "[REDACTED_BY_SCRIPT]'Legacy', 'Old', or 'Existing' systems (e.g., 'Visit our Aylesbury system')."
             )
             self.logger.warning("[REDACTED_BY_SCRIPT]")
             return "[REDACTED_BY_SCRIPT]", notes

        else:
            self.logger.info("[REDACTED_BY_SCRIPT]")
            return "TIMEOUT", None

            
    def _validate_search_intent(self, initial_url: str, final_url: str, instructions: list[dict]) -> bool:
        """
        V2 Heuristic: Checks for mission success based on the plan's contract.
        - If the plan ends with WAIT_FOR_SELECTOR, success is assumed as the command already passed.
        - If the plan ends with WAIT_FOR_NAVIGATION, it falls back to URL comparison.
        """
        if not instructions:
            return False

        last_instruction = instructions[-1]
        
        # New Heracles Logic: Trust the AI's "assert" command.
        if last_instruction.get("command") == "WAIT_FOR_SELECTOR":
            self.logger.info("[REDACTED_BY_SCRIPT]")
            return True

        # Fallback Pathfinder Logic for traditional websites.
        self.logger.info("[REDACTED_BY_SCRIPT]")
        if final_url == initial_url or final_url.strip('/') == initial_url.strip('/'):
            self.logger.warning(f"[REDACTED_BY_SCRIPT]'{initial_url}'.")
            return False

        search_patterns = [r'\/search', r'\?q=', r'\?s=', r'\?query=', r'\?keyword=', r'\?location=']
        if any(re.search(pattern, final_url, re.IGNORECASE) for pattern in search_patterns):
            self.logger.info(f"[REDACTED_BY_SCRIPT]'{final_url}'[REDACTED_BY_SCRIPT]")
            return True

        self.logger.warning(f"[REDACTED_BY_SCRIPT]'{final_url}'[REDACTED_BY_SCRIPT]")
        return False

    def _apply_post_processing(self, value: str, steps: list[dict]) -> str | None:
        """[REDACTED_BY_SCRIPT]"""
        current_value = value
        for step in steps:
            if step['method'] == 'REGEX_EXTRACT':
                pattern = step['params']['pattern']
                match = re.search(pattern, current_value)
                current_value = match.group(1) if match else None
                if current_value is None:
                    return None # Stop processing if regex fails
            # Add other methods here in the future
        return current_value

    async def execute_instructions(self, instructions: list[dict[str, Any]], page: Page, debug_dir: str | None = None) -> Page:
        """
        Public API: Executes a sequence of instructions on a given page.
        Returns the active Page (which may change during execution).
        """
        current_page = page
        for i, instruction in enumerate(instructions):
            is_final_assert = (
                i == len(instructions) - 1 and
                instruction.get("command") == "WAIT_FOR_SELECTOR"
            )
            current_page = await self._execute_instruction(instruction, current_page, is_final_assert=is_final_assert)
            await self._debug_snapshot(current_page, f"exec_step_{i+1:02d}", save_dir=debug_dir)
        return current_page

    async def _execute_instruction(self, instruction: dict[str, Any], page: Page, is_final_assert: bool = False) -> Page:
        """
        Dispatches an instruction.
        Returns: The active Page object (which may have changed if a new tab was opened).
        """
        command = instruction["command"]
        params = instruction["params"]
        
        current_page = page

        match command:
            case "FILL_INPUT":
                await self._execute_fill_input(params, current_page)
            case "CLICK_ELEMENT":
                # Special handling for clicks that might open new windows
                current_page = await self._execute_click_element(params, current_page)
            case "TRY_CLICK_ELEMENT":
                current_page = await self._execute_try_click_element(params, current_page)
            case "WAIT_FOR_NAVIGATION":
                await self._execute_wait_for_navigation(params, current_page)
            case "WAIT_FOR_SELECTOR":
                await self._execute_wait_for_selector(params, current_page, is_final_assert=is_final_assert)
            case "GOTO_URL":
                await self._execute_goto_url(params, current_page)
            case "PRESS_KEY":
                await self._execute_press_key(params, current_page)
            case "CHECK_OPTION":
                await self._execute_check_option(params, current_page)
            case _:
                raise ValidationError(f"[REDACTED_BY_SCRIPT]'{command}'")
        
        return current_page

    async def _execute_goto_url(self, params: dict[str, Any], page: Page):
        """[REDACTED_BY_SCRIPT]"""
        url = params.get('url')
        if not url:
            raise ValidationError("Instruction 'GOTO_URL'[REDACTED_BY_SCRIPT]'url'")

        # --- PROTOCOL: HALLUCINATION FIREWALL ---
        if any(x in url.lower() for x in ["example.com", "example.gov", "council.url", "demo.com"]):
            raise ValidationError(f"[REDACTED_BY_SCRIPT]'{url}'[REDACTED_BY_SCRIPT]")

        # --- PROTOCOL: STAY PUT (The NI Fix) ---
        # Normalize URLs to ignore protocol/www/trailing slashes for comparison
        def normalize_url(u: str) -> str:
            u = u.lower().strip().rstrip('/')
            u = u.replace("https://", "").replace("http://", "").replace("www.", "")
            return u

        if normalize_url(page.url) == normalize_url(url):
            self.logger.info(f"[REDACTED_BY_SCRIPT]'{url}'[REDACTED_BY_SCRIPT]")
            return

        try:
            response = await page.goto(url, timeout=self.default_timeout_ms)
            await page.wait_for_load_state("domcontentloaded", timeout=self.default_timeout_ms)

            # --- FAIL-FAST: HTTP STATUS CHECK ---
            if response and response.status >= 400:
                 raise ValidationError(f"[REDACTED_BY_SCRIPT]")

            # --- FAIL-FAST: CONTENT CHECK (404 INDICATORS) ---
            # Some portals return 200 OK but show a "Page Not Found" component.
            title = await page.title()
            
            if "page not found" in title.lower() or "404" in title or "error has occurred" in title.lower():
                 raise ValidationError(f"[REDACTED_BY_SCRIPT]'{title}'")

        except Exception as e:
            raise ValidationError(f"[REDACTED_BY_SCRIPT]")

    async def _execute_fill_input(self, params: dict[str, Any], page: Page):
        """Fills a text input."""
        selector = params.get('selector')
        value = params.get('value')
        if not selector or value is None:
            raise ValidationError("Instruction 'FILL_INPUT' requires 'selector' and 'value'.")
        
        try:
            target = page.locator(selector).first
            await target.wait_for(state='visible', timeout=self.default_timeout_ms)
            await target.fill(value)
        except Exception as e:
             raise ValidationError(f"Failed to fill input '{selector}'. Error: {e}")

    async def _execute_press_key(self, params: dict[str, Any], page: Page):
        """[REDACTED_BY_SCRIPT]"""
        selector = params.get('selector')
        key = params.get('key')
        if not selector or not key:
             raise ValidationError("Instruction 'PRESS_KEY' requires 'selector' and 'key'.")
        
        try:
            target = page.locator(selector).first
            await target.wait_for(state='visible', timeout=self.default_timeout_ms)
            await target.press(key)
        except Exception as e:
             raise ValidationError(f"[REDACTED_BY_SCRIPT]'{key}' on '{selector}'. Error: {e}")

    async def _execute_check_option(self, params: dict[str, Any], page: Page):
        """[REDACTED_BY_SCRIPT]"""
        selector = params.get('selector')
        if not selector:
             raise ValidationError("Instruction 'CHECK_OPTION' requires 'selector'.")
        
        try:
            target = page.locator(selector).first
            await target.wait_for(state='visible', timeout=self.default_timeout_ms)
            await target.check()
        except Exception as e:
             raise ValidationError(f"[REDACTED_BY_SCRIPT]'{selector}'. Error: {e}")

    async def _execute_click_element(self, params: dict[str, Any], page: Page) -> Page:
        """
        Finds a clickable element and executes a click.
        Detects if a new tab/window was opened and returns the new Page if so.
        """
        selector = params.get('selector', '')
        if not selector and not params.get('frame_name'):
            raise ValidationError("Instruction 'CLICK_ELEMENT'[REDACTED_BY_SCRIPT]'selector'")

        # Capture context state before click to detect new pages
        context = page.context
        pages_before = context.pages

        try:
            target = await self._resolve_locator(page, params)
            await target.click(timeout=self.default_timeout_ms)

        except (PlaywrightTimeoutError, PlaywrightImplError, Exception) as e:
            # Catch generic Exception to handle potential subtype mismatches or 'Error: Timeout' strings
            error_msg = str(e).lower()
            is_timeout = "timeout" in error_msg or isinstance(e, (PlaywrightTimeoutError, PlaywrightImplError))
            
            if not is_timeout:
                raise e # Re-raise unrelated errors (e.g. navigation errors)

            # --- PROJECT AEGIS: SMART TIMEOUTS ---
            selector_lower = selector.lower()
            # Expanded keyword list for Soft Failure (added 'ccc', 'notify', 'continue', etc.)
            is_consent = any(x in selector_lower for x in [
                'accept', 'agree', 'cookie', 'consent', 'close', 'dismiss', 'ccc', 'notify', 
                'continue', 'verify', 'confirm', 'necessary', 'start', 'acknowledge'
            ])
            
            if is_consent:
                # If it's just a consent banner, we soft fail immediately if not found quickly.
                self.logger.warning(f"[REDACTED_BY_SCRIPT]'{selector}'[REDACTED_BY_SCRIPT]")
                return page

            # DIAGNOSTIC: Check if it's hidden
            target_locator = page.locator(selector) if selector else None
            
            if target_locator and await target_locator.count() > 0 and not await target_locator.first.is_visible():
                # Attempt to scroll into view before failing
                await target_locator.first.scroll_into_view_if_needed()
                if not await target_locator.first.is_visible():
                     raise ValidationError(
                        f"Target element '{selector}'[REDACTED_BY_SCRIPT]"
                        "[REDACTED_BY_SCRIPT]"
                     )

            self.logger.warning(f"[REDACTED_BY_SCRIPT]'{selector}'[REDACTED_BY_SCRIPT]")
            
            # Force Scroll & Click Strategy
            try:
                target_raw = target_locator.first if target_locator else target
                # Increased scroll timeout for robust navigation (2s -> 5s)
                try:
                    await target_raw.scroll_into_view_if_needed(timeout=5000)
                except Exception:
                    self.logger.warning(f"[REDACTED_BY_SCRIPT]'{selector}'[REDACTED_BY_SCRIPT]")
                
                await target_raw.click(force=True, timeout=5000)
            except Exception as e:
                self.logger.warning(f"Force click failed for '{selector}'[REDACTED_BY_SCRIPT]")
                try:
                    await target_raw.evaluate("[REDACTED_BY_SCRIPT]")
                except Exception as js_e:
                    self.logger.warning(f"[REDACTED_BY_SCRIPT]'{selector}'[REDACTED_BY_SCRIPT]")
                    try:
                        await target_raw.scroll_into_view_if_needed()
                        await page.wait_for_timeout(500)
                        box = await target_raw.bounding_box()
                        if box:
                            x = box['x'] + (box['width'] / 2)
                            y = box['y'] + (box['height'] / 2)
                            await page.mouse.click(x, y)
                        else:
                            raise ValidationError("[REDACTED_BY_SCRIPT]")
                    except Exception as mouse_e:
                        raise ValidationError(f"[REDACTED_BY_SCRIPT]'{selector}'[REDACTED_BY_SCRIPT]")

        # --- NEW WINDOW DETECTION ---
        # Wait briefly for any new page logic to trigger (Playwright events are async)
        # We assume if a new page appears within 2 seconds, it was due to this click.
        await page.wait_for_timeout(2000)
        if len(context.pages) > len(pages_before):
            new_page = context.pages[-1]
            try:
                await new_page.wait_for_load_state("domcontentloaded")
                # Add networkidle wait for React/Angular portals (Antrim fix)
                try:
                    await new_page.wait_for_load_state("networkidle", timeout=5000)
                except:
                    pass # Don't fail if network is chatty, just proceed
                self.logger.info(f"[REDACTED_BY_SCRIPT]")
                await new_page.bring_to_front()
                page_to_return = new_page
            except Exception as e:
                self.logger.warning(f"[REDACTED_BY_SCRIPT]")
                page_to_return = page
        else:
            # No new page detected, sticking with current page
            page_to_return = page

        # Soft 404 Check After Click
        try:
            title = (await page_to_return.title()).lower()
            content = (await page_to_return.content()).lower()
            if "page not found" in title or "404" in title or "session has expired" in content:
                raise ValidationError("[REDACTED_BY_SCRIPT]'Page not found' or 'stale session'.")
        except Exception as e:
            if isinstance(e, ValidationError):
                raise e
            pass # Ignore errors getting title/content

        return page_to_return
            
    
    async def _execute_try_click_element(self, params: dict[str, Any], page: Page) -> Page:
        """
        Attempts to click a transitional element with a lower timeout.
        If it fails (e.g. because of a direct redirect), it logs it and returns gracefully.
        """
        selector = params.get('selector', '')
        if not selector and not params.get('frame_name'):
            raise ValidationError("Instruction 'TRY_CLICK_ELEMENT'[REDACTED_BY_SCRIPT]'selector'")

        timeout_val = params.get("timeout", min(5000, self.default_timeout_ms))
        context = page.context
        pages_before = context.pages

        try:
            target = await self._resolve_locator(page, params, timeout_ms=timeout_val)
            await target.click(timeout=timeout_val)
            self.logger.info(f"[REDACTED_BY_SCRIPT]'selector')}")

            # Allow slight time for new tab to open if it's going to
            await asyncio.sleep(1)
            if len(context.pages) > len(pages_before):
                new_page = context.pages[-1]
                await new_page.bring_to_front()
                return new_page
        except Exception as e:
            self.logger.info(f"[REDACTED_BY_SCRIPT]")
            
        return page

    async def _execute_wait_for_navigation(self, params: dict[str, Any], page: Page):
        """[REDACTED_BY_SCRIPT]"""
        timeout = params.get('timeout_ms', self.default_timeout_ms)
        
        try:
            # UPGRADED: From 'domcontentloaded' to the much more robust 'networkidle'.
            # This waits for the network to be quiet, indicating async calls are likely complete.
            await page.wait_for_load_state('networkidle', timeout=timeout)
        except PlaywrightTimeoutError:
             # --- CLOUDFLARE SHIELD ---
             # If navigation times out, it could be a challenge screen.
             # We check for it, and if handled, we wait again.
             if await self._detect_and_handle_challenge(page):
                 self.logger.info("[REDACTED_BY_SCRIPT]")
                 await page.wait_for_load_state('networkidle', timeout=timeout)
             else:
                 raise # Re-raise if no challenge found/handled

    async def _execute_wait_for_selector(self, params: dict[str, Any], page: Page, is_final_assert: bool = False):
        """[REDACTED_BY_SCRIPT]"""
        selector = params.get('selector')
        if not selector:
            raise ValidationError("Instruction 'WAIT_FOR_SELECTOR'[REDACTED_BY_SCRIPT]'selector'")
        
        state = params.get('state', 'visible')
        timeout = params.get('timeout', params.get('timeout_ms', self.default_timeout_ms))

        try:
            await page.wait_for_load_state("domcontentloaded", timeout=timeout/2)
        except Exception:
            pass

        try:
            if state == 'visible' and is_final_assert:
                # Branched logic for dynamic final assert
                branched_selector = f"[REDACTED_BY_SCRIPT]'No results found'), :has-text('0 results found'[REDACTED_BY_SCRIPT]'No application found')"
                await page.wait_for_selector(branched_selector, state=state, timeout=timeout)
                
                # Check what actually matched: if it was a failure container, raise a Timeout-like exception so diagnostics catch it
                inner_html = (await page.content()).lower()
                zero_kws = ["no results found", "no application found", "search returned no results", "zero results", "0 results found"]
                
                # Use wait_for_selector to check if the main success selector matched instead of the "No Results"
                try:
                    await page.wait_for_selector(selector, state=state, timeout=1000)
                except PlaywrightTimeoutError:
                    if any(kw in inner_html for kw in zero_kws):
                         raise PlaywrightTimeoutError("Found 'No Results'[REDACTED_BY_SCRIPT]")
                    else:
                         raise PlaywrightTimeoutError("[REDACTED_BY_SCRIPT]'No results' container wasn't explicitly found either.")
            else:
                await page.wait_for_selector(selector, state=state, timeout=timeout)

        except PlaywrightTimeoutError as e:
            # --- PROJECT AEGIS: SOFT FAIL ON CONSENT WAITS ---
            # If the AI adds an explicit wait for a consent banner that isn't there,
            # we treat it as a non-fatal "Soft Failure" and proceed.
            selector_lower = selector.lower()
            is_consent = any(x in selector_lower for x in ['accept', 'agree', 'cookie', 'consent', 'close', 'dismiss', 'gdpr'])
            
            if is_consent:
                self.logger.warning(f"[REDACTED_BY_SCRIPT]'{selector}'[REDACTED_BY_SCRIPT]")
                return
            
            # If it's not a consent element, it's a real failure. Re-raise.
            # We sanitize the error message to prevent Unicode encoding crashes on Windows consoles
            safe_msg = str(e).encode('ascii', 'replace').decode('ascii')
            raise PlaywrightTimeoutError(safe_msg) from e
    
    async def _validate_blueprint(self, blueprint: dict[str, Any], page: Page, scope: Page | Any = None, blueprint_name: str = "") -> None:
        """
        Validates a single extraction blueprint, NOW INCLUDING POST-PROCESSING.
        Raises a ValidationError on failure.
        """
        # --- PROJECT SYNAPSE DIRECTIVE: FAIL-FAST ON STRUCTURAL CORRUPTION ---
        if not isinstance(blueprint, dict):
            raise ValidationError(
                f"[REDACTED_BY_SCRIPT]'{blueprint_name}'[REDACTED_BY_SCRIPT]"
                f"It is of type '{type(blueprint).__name__}'[REDACTED_BY_SCRIPT]"
            )
        # --- END DIRECTIVE ---

        scope = scope or page
        selector = blueprint.get('selector')
        method = blueprint.get('extraction_method')
        attr_name = blueprint.get('attribute_name')
        
        if not selector or not method:
            raise ValidationError(f"Blueprint is missing 'selector' or 'extraction_method': {blueprint}")

        # Handle self-reference gracefully
        is_self = selector.strip() in [".", ":scope", ""]
        
        if is_self:
            # If scope is Page (which has no inner_text), treat "." as "body"
            if isinstance(scope, Page):
                locator = scope.locator("body")
            else:
                # If scope is Locator, treat "." as the locator itself
                locator = scope
        else:
            locator = scope.locator(selector)
        
        try:
            await locator.first.wait_for(state='visible', timeout=self.default_timeout_ms)
            
            raw_value = None
            if method in ["TEXT", "MULTIPLE_TEXT"]:
                raw_value = await locator.first.inner_text(timeout=self.default_timeout_ms / 2)
            elif method in ["ATTRIBUTE", "MULTIPLE_ATTRIBUTE"]:
                raw_value = await locator.first.get_attribute(attr_name, timeout=self.default_timeout_ms / 2)

            if raw_value is None:
                if method == "TEXT":
                    raise ValidationError(f"[REDACTED_BY_SCRIPT]'{selector}'[REDACTED_BY_SCRIPT]")
                else:
                    raise ValidationError(f"[REDACTED_BY_SCRIPT]'{selector}'[REDACTED_BY_SCRIPT]'{attr_name}'.")

            post_processing_steps = blueprint.get('post_processing')
            if post_processing_steps:
                final_value = self._apply_post_processing(raw_value, post_processing_steps)
                if final_value is None:
                    raise ValidationError(f"[REDACTED_BY_SCRIPT]'{selector}'. Raw value was '{raw_value}'.")

                if "url" in blueprint_name.lower():
                    parsed_url = urlparse(final_value)
                    if not (parsed_url.scheme in ['http', 'https'] and parsed_url.netloc):
                        raise ValidationError(f"[REDACTED_BY_SCRIPT]'{final_value}' from '{blueprint_name}'[REDACTED_BY_SCRIPT]")
        
        except PlaywrightTimeoutError as e:
            raise ValidationError(f"[REDACTED_BY_SCRIPT]'{selector}'[REDACTED_BY_SCRIPT]") from e