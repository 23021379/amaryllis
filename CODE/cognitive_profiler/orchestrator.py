import logging
import sys
import copy
from pathlib import Path
from .agents.triage_agent import TriageAgent
from .agents.strategist_agent import StrategistAgent
from .agents.validation_agent import ValidationAgent
from .agents.synthesis_agent import SynthesisAgent
from .data_contracts import (
    PipelineResult, 
    ValidationReport, 
    DraftPlan, 
    ProbeResult, 
    TriageResult, 
    ExecutionPlan, 
    Instruction,
    ResultsPageProcessing, 
    DetailPageProcessing, 
    ExtractionBlueprint
)
from .exceptions import (
    PipelineError, 
    MaxRetriesExceededError, 
    StrategyError, 
    ValidationError, 
    TriageError
)
from urllib.parse import urlparse, urljoin
from playwright._impl._errors import Error as PlaywrightImplError
import asyncio
from bs4 import BeautifulSoup
from .data_contracts import DownloadedArtifact
from jsonschema import validate, ValidationError as JsonSchemaValidationError
from .command_schema import COMMAND_SCHEMA


class CognitiveProfiler:
    """
    Orchestrates the multi-stage AI pipeline to generate a website execution plan.
    """
    def __init__(
        self,
        triage_agent: TriageAgent,
        strategist_agent: StrategistAgent,
        validation_agent: ValidationAgent,
        synthesis_agent: SynthesisAgent,
        max_correction_attempts: int,
        validation_retries: int = 2,
    ):
        """
        Initializes the orchestrator by injecting agent dependencies.
        ...
        """
        self.triage_agent = triage_agent
        self.strategist_agent = strategist_agent
        self.validation_agent = validation_agent
        self.synthesis_agent = synthesis_agent
        self.max_correction_attempts = max_correction_attempts
        self.validation_retries = validation_retries
        self.logger = logging.getLogger(self.__class__.__name__)

    async def run_pipeline(self, artifact_manifest: list[DownloadedArtifact], target_candidates: list[str], target_description: str, temp_data_path: str) -> PipelineResult:
        """[REDACTED_BY_SCRIPT]"""
        domain = "unknown_domain"
        
        try:
            # --- PHASE 1: DISCOVERY ---
            self.logger.info("[REDACTED_BY_SCRIPT]")
            
            # Step A: Passive Triage
            search_page_url = None
            try:
                triage_result = await self.triage_agent.run(artifact_manifest)
                domain = triage_result.domain
                
                # Check if we found the Golden Ticket (SEARCH_PAGE)
                search_page_url = self._find_best_candidate(triage_result)
            except TriageError:
                self.logger.warning("[REDACTED_BY_SCRIPT]")

            # Step B: Active Fallback (Global Search)
            if not search_page_url:
                self.logger.warning("[REDACTED_BY_SCRIPT]")
                
                # 1. Get Homepage HTML
                homepage_artifact = next((a for a in artifact_manifest if "homepage" in a.html_path), artifact_manifest[0])
                homepage_html = Path(homepage_artifact.html_path).read_text(encoding='utf-8')

                # 2. Generate Plan
                # PASS URL TO AGENT to ensure GOTO_URL is included
                search_plan = await self.strategist_agent.generate_global_search_plan(homepage_artifact.url, homepage_html)
                
                # 3. Execute Plan (Blindly for now, assuming it works)
                self.logger.info("[REDACTED_BY_SCRIPT]")
                search_result = await self.validation_agent.execute_probe(search_plan, temp_data_path)
                
                if not search_result.is_successful:
                    raise StrategyError(f"[REDACTED_BY_SCRIPT]")

                # 4. Analyze Search Results (Harvest Mode)
                self.logger.info("[REDACTED_BY_SCRIPT]")
                fallback_artifacts = []
                
                # Add the result page itself (just in case)
                main_artifact = DownloadedArtifact(
                    url=search_result.final_url,
                    html_path="[REDACTED_BY_SCRIPT]", 
                    screenshot_path=search_result.captured_screenshot_path
                )
                main_path = Path(temp_data_path) / "[REDACTED_BY_SCRIPT]"
                main_path.write_text(search_result.captured_html, encoding='utf-8')
                main_artifact.html_path = str(main_path)
                fallback_artifacts.append(main_artifact)

                # Parse and Extract Links
                soup = BeautifulSoup(search_result.captured_html, 'html.parser')
                found_links = []
                seen_urls = set()
                base_url = search_result.final_url

                for a in soup.find_all('a', href=True):
                    href = a['href']
                    text = a.get_text(" ", strip=True).lower()
                    
                    # Filtering Heuristics
                    if any(x in href.lower() for x in ['javascript:', 'mailto:', 'tel:', '#']): continue
                    
                    # Scoring Heuristics
                    score = 0
                    if 'planning' in text: score += 10
                    if 'search' in text: score += 5
                    if 'application' in text: score += 5
                    if 'view' in text: score += 5
                    if 'find' in text: score += 5
                    
                    if score >= 10: # Threshold for relevance
                        full_url = urljoin(base_url, href)
                        if full_url not in seen_urls:
                            found_links.append((score, full_url))
                            seen_urls.add(full_url)

                # Sort by score and pick Top 3
                found_links.sort(key=lambda x: x[0], reverse=True)
                top_links = found_links[:3]
                
                self.logger.info(f"[REDACTED_BY_SCRIPT]")

                # Fetch Content for Top Links
                for i, (_, url) in enumerate(top_links):
                    try:
                        page = await self.validation_agent.browser_context.new_page()
                        await page.goto(url, timeout=15000) # Short timeout
                        await page.wait_for_load_state("domcontentloaded")
                        content = await page.content()
                        
                        # Save artifact
                        art_path = Path(temp_data_path) / f"[REDACTED_BY_SCRIPT]"
                        art_path.write_text(content, encoding='utf-8')
                        
                        fallback_artifacts.append(DownloadedArtifact(
                            url=url,
                            html_path=str(art_path),
                            screenshot_path=None # Optimization: Skip screenshot for triage
                        ))
                    except Exception as e:
                        self.logger.warning(f"[REDACTED_BY_SCRIPT]")
                    finally:
                        await page.close()

                # 5. Re-Run Triage on the BATCH
                if not fallback_artifacts:
                     raise PipelineError("[REDACTED_BY_SCRIPT]")

                triage_result = await self.triage_agent.run(fallback_artifacts)
                search_page_url = self._find_best_candidate(triage_result)
                
                if not search_page_url:
                    raise PipelineError("[REDACTED_BY_SCRIPT]")

            self.logger.info(f"[REDACTED_BY_SCRIPT]")

            # --- PHASE 2: TARGETED SEARCH EXECUTION ---
            self.logger.info("[REDACTED_BY_SCRIPT]")

            # Step A: Reality Capture (Fetch the Search Page HTML)
            # We use a temporary page via validation_agent to grab the source
            temp_page = await self.validation_agent.browser_context.new_page()
            try:
                await temp_page.goto(search_page_url, timeout=30000)
                await temp_page.wait_for_load_state("domcontentloaded")
                search_page_html = await temp_page.content()
            finally:
                await temp_page.close()

            # --- PHASE 2 LOOP: NAVIGATE & SEARCH ---
            navigation_iterations = 0
            max_nav_iterations = 5 
            current_url = search_page_url
            current_html = search_page_html
            
            ref_search_plan = None
            search_result = None
            
            # STATE MEMORY: Track visual history
            breadcrumb_trail = []
            
            # Initial seed: The Search Page screenshot from Phase 1 (if available)
            # (We skip this for now to keep it clean, relying on Phase 2 snapshots)

            while navigation_iterations < max_nav_iterations:
                self.logger.info(f"[REDACTED_BY_SCRIPT]")
                
                # 1. Generate the Plan (With Context)
                # Note: '[REDACTED_BY_SCRIPT]' is empty here because we are starting a *fresh* plan. (removed undefined var)
                previous_failures_context = []
                # If we fail inside the inner loop, the correction agent handles those specific failure contexts.
                # Need a representative ID for the plan generation (use the first one)
                # The actual ID used during execution will be swapped in the loop below.
                representative_target = target_candidates[0]
                
                ref_search_plan = await self.strategist_agent.generate_planning_search_plan(
                    search_page_url=current_url,
                    search_page_html=current_html,
                    target_ref_id=representative_target,
                    breadcrumbs=breadcrumb_trail,

                    previous_failures=previous_failures_context # Pass context if needed, but separate from candidate loop failures
                )

                # --- FIX: INJECT GOTO_URL FOR CLEAN SLATE EXECUTION ---
                # The Strategist assumes "Law of Inertia" (we are already here), but ValidationAgent.execute_probe
                # creates a NEW page/context. We must explicitly tell it to go to 'current_url' if it doesn't already.
                if not ref_search_plan.instructions or ref_search_plan.instructions[0].get("command") != "GOTO_URL":
                    self.logger.info(f"[REDACTED_BY_SCRIPT]'{current_url}'[REDACTED_BY_SCRIPT]")
                    goto_instr = Instruction(
                        command="GOTO_URL",
                        params={"url": current_url},
                        description="[REDACTED_BY_SCRIPT]"
                    )
                    ref_search_plan.instructions.insert(0, goto_instr.model_dump())

                # 2. Execute with Self-Healing (Logic applies to BOTH navigation-only and final-search)
                # The only difference is the success exit condition.
                
                # 2. Execute with Self-Healing (Logic applies to BOTH navigation-only and final-search)
                # The only difference is the success exit condition.
                
                if ref_search_plan.is_navigation_only:
                    # NAVIGATION MODE (Standard Execution)
                    plan_result = await self.validation_agent.execute_probe(ref_search_plan, temp_data_path)
                    # ... [Standard navigation error handling would go here, omitting for brevity matching original structure] ...
                    
                    # For navigation, we just execute once. If it fails, the outer loop/agent handles it.
                    # (In original code, there was a while loop for corrections. I will allow validation_agent to handle basic executions)
                    # Wait, strictly sticking to the original structure which had a correction loop:
                    
                    correction_attempts = 0
                    while not plan_result.is_successful and correction_attempts < self.max_correction_attempts:
                         correction_attempts += 1
                         self.logger.warning(f"[REDACTED_BY_SCRIPT]")
                         
                         _, analysis_notes = self.strategist_agent._diagnose_timeout_failure(plan_result.captured_html)
                         failure_report = ValidationReport(
                             is_valid=False, original_plan=ref_search_plan, failure_reason=plan_result.failure_reason,
                             failure_context_html=plan_result.captured_html, failure_screenshot_path=plan_result.captured_screenshot_path,
                             failure_stage="NAVIGATION", failure_analysis_notes=analysis_notes
                         )
                         corrected = await self.strategist_agent.correct_navigation_plan(failure_report)
                         ref_search_plan.instructions = corrected
                         plan_result = await self.validation_agent.execute_probe(ref_search_plan, temp_data_path)

                    if not plan_result.is_successful:
                        raise StrategyError(f"[REDACTED_BY_SCRIPT]")

                    self.logger.info("[REDACTED_BY_SCRIPT]")
                    if plan_result.captured_screenshot_path:
                        breadcrumb_trail.append(plan_result.captured_screenshot_path)
                    current_url = plan_result.final_url
                    current_html = plan_result.captured_html
                    navigation_iterations += 1
                    continue

                else:
                    # SEARCH EXECUTION MODE (Multi-ID Retry Loop)
                    self.logger.info(f"[REDACTED_BY_SCRIPT]")
                    
                    search_success = False
                    final_successful_id = None
                    
                    for cand_idx, candidate_id in enumerate(target_candidates):
                        self.logger.info(f"[REDACTED_BY_SCRIPT]'{candidate_id}' ---")
                        
                        # A. Prepare Plan (Substitute ID)
                        current_search_plan = copy.deepcopy(ref_search_plan)
                        for instr in current_search_plan.instructions:
                             if instr.get("command") == "FILL_INPUT":
                                 instr["params"]["value"] = candidate_id
                        
                        # B. Clean Slate (Reload Search Page if > 0 attempts)
                        # We only need to physically reload if we aren't on the first attempt,
                        # OR if the plan doesn't start with GOTO.
                        # However, relying on the Agent's GOTO is safest.
                        # But typically, we should ensure we are at 'current_url' before executing.
                        # ValidationAgent creates a new context/page usually? No, it uses the passed buffer or new page?
                        # execute_probe creates a new page. So we are safe.
                        
                        # C. Execute
                        plan_result = await self.validation_agent.execute_probe(current_search_plan, temp_data_path)
                        
                        # D. Correction Loop (For TIMEOUTS/ERRORS, not "No Results")
                        correction_attempts = 0
                        while not plan_result.is_successful and correction_attempts < self.max_correction_attempts:
                             # Check if it's a "No Results" logical failure vs a Technical Failure
                             # execute_probe returns is_successful=False for timeouts/crashes.
                             # If it navigated but found "No Results" (based on explicit check), it might return Success=True but we need to check content?
                             # WAIT. execute_probe checks the final WAIT_FOR_SELECTOR.
                             # If the final selector (which includes #searchresults) is NOT found, it effectively fails.
                             # So "No Results" usually manifests as a TIMEOUT waiting for results.
                             
                             correction_attempts += 1
                             self.logger.warning(f"[REDACTED_BY_SCRIPT]'{candidate_id}'[REDACTED_BY_SCRIPT]")
                             
                             _, analysis_notes = self.strategist_agent._diagnose_timeout_failure(plan_result.captured_html)
                             failure_report = ValidationReport(
                                 is_valid=False, original_plan=current_search_plan, failure_reason=plan_result.failure_reason,
                                 failure_context_html=plan_result.captured_html, failure_screenshot_path=plan_result.captured_screenshot_path,
                                 failure_stage="SEARCH_EXECUTION", failure_analysis_notes=analysis_notes
                             )
                             # Use correct_navigation_plan (it's generic enough)
                             corrected = await self.strategist_agent.correct_navigation_plan(failure_report)
                             current_search_plan.instructions = corrected
                             
                             # Re-substitute (just in case correction wiped it, though strict rules prevent it)
                             for instr in current_search_plan.instructions:
                                 if instr.get("command") == "FILL_INPUT":
                                     instr["params"]["value"] = candidate_id
                                     
                             plan_result = await self.validation_agent.execute_probe(current_search_plan, temp_data_path)

                        if plan_result.is_successful:
                            self.logger.info(f"[REDACTED_BY_SCRIPT]'{candidate_id}'.")
                            search_result = plan_result
                            final_successful_id = candidate_id
                            search_success = True
                            break # Break the candidate loop
                        else:
                            self.logger.warning(f"Candidate '{candidate_id}'[REDACTED_BY_SCRIPT]")
                            
                    if search_success:
                        search_result.final_url = plan_result.final_url # Ensure consistency
                        # Also we must update 'target_address' to the one that worked, for downstream usage (Re-enactment)
                        target_address = final_successful_id 
                        break # Break the Navigation Loop (We are done)
                    else:
                        raise StrategyError(f"[REDACTED_BY_SCRIPT]")

            if navigation_iterations >= max_nav_iterations:
                raise StrategyError(f"[REDACTED_BY_SCRIPT]")

            self.logger.info("[REDACTED_BY_SCRIPT]")

            # --- PHASE 3: RESULTS EXTRACTION ---
            self.logger.info(f"[REDACTED_BY_SCRIPT]")

            # 1. Check for Direct Redirect (Skip Extraction)
            # Idox URL pattern for details: '[REDACTED_BY_SCRIPT]'
            # Northgate URL pattern for details: '[REDACTED_BY_SCRIPT]' (No, that's results)
            # Heuristic: Check URL OR content for "Application Summary".
            url_lower = search_result.final_url.lower()
            html_lower = search_result.captured_html.lower()
            
            # Explicitly exclude known list-view URL patterns (Idox/Northgate specific)
            is_list_view_url = (
                "searchresults" in url_lower or 
                "results" in url_lower or 
                "pagedresult" in url_lower
            )

            # Detail view indicators
            url_has_details = "details" in url_lower or "summary" in url_lower
            html_has_header = "application summary" in html_lower or "[REDACTED_BY_SCRIPT]" in html_lower

            is_direct_details = (url_has_details or html_has_header) and not is_list_view_url
            
            self.logger.info(
                f"[REDACTED_BY_SCRIPT]'{search_result.final_url}' | "
                f"[REDACTED_BY_SCRIPT]"
                f"[REDACTED_BY_SCRIPT]"
                f"[REDACTED_BY_SCRIPT]"
            )
            target_detail_url = None
            results_blueprints = {} 

            if is_direct_details:
                self.logger.info("[REDACTED_BY_SCRIPT]")
                target_detail_url = search_result.final_url
                # Dummy schema
                results_blueprints = {
                    "[REDACTED_BY_SCRIPT]": "body",
                    "listing_link_blueprint": {"selector": "body", "extraction_method": "ATTRIBUTE", "attribute_name": "id"},
                    "[REDACTED_BY_SCRIPT]": {"selector": "body", "extraction_method": "TEXT"}
                }
            else:
                # 2. Perform Extraction
                results_blueprints, wait_instruction = await self.strategist_agent.generate_adaptive_components(search_result)
                self.logger.info(f"[REDACTED_BY_SCRIPT]'listing_container_selector']}")

                # 3. Verify & Extract Link (With Self-Healing)
                extraction_attempts = 0
                max_extraction_retries = 2
                
                while extraction_attempts <= max_extraction_retries:
                    container_sel = results_blueprints['[REDACTED_BY_SCRIPT]']
                    link_bp = results_blueprints['listing_link_blueprint']
                    ref_bp = results_blueprints['[REDACTED_BY_SCRIPT]']
                    
                    verify_page = await self.validation_agent.browser_context.new_page()
                    try:
                        await verify_page.set_content(search_result.captured_html)
                        
                        if container_sel:
                            count = await verify_page.locator(container_sel).count()
                        else:
                            self.logger.warning("[REDACTED_BY_SCRIPT]")
                            count = 0
                            
                        self.logger.info(f"[REDACTED_BY_SCRIPT]'{container_sel}'.")
                        
                        if count > 0:
                            for i in range(count):
                                row = verify_page.locator(container_sel).nth(i)
                                
                                async def extract_field(bp):
                                    if not bp: return ""
                                    sel = bp.get('selector')
                                    if not sel: return ""

                                    # SANITIZATION: Auto-correct common AI selector errors
                                    if ":contains(" in sel:
                                        self.logger.warning(f"[REDACTED_BY_SCRIPT]'{sel}' -> replaced ':contains' with ':has-text'")
                                        sel = sel.replace(":contains(", ":has-text(")

                                    if sel.strip() in [".", ":scope", ""]:
                                        target = row
                                    else:
                                        target = row.locator(sel)
                                    
                                    count = await target.count()
                                    if count == 0: return ""

                                    # CRITICAL FIX: Strict Mode compliance.
                                    # If the selector is loose and matches multiple items (e.g. 'p'), 
                                    # we must explicitly grab the first one to avoid a crash.
                                    if count > 1:
                                        target = target.first
                                    
                                    method = bp.get('extraction_method')
                                    if method == 'TEXT':
                                        return await target.inner_text()
                                    elif method == 'ATTRIBUTE':
                                        return await target.get_attribute(bp.get('attribute_name', 'href')) or ""
                                    return ""

                                ref_text = await extract_field(ref_bp)
                                link_url = await extract_field(link_bp)

                                self.logger.info(f"Row {i}: Ref '{ref_text}' -> Link '{link_url}'")

                                # FALLBACK LOGIC: If AI blueprint missed the ref, check the whole row text
                                row_text = await row.inner_text()
                                is_match = target_address.replace("/","").lower() in ref_text.replace("/","").lower()
                                
                                if not is_match and target_address.replace("/","").lower() in row_text.replace("/","").lower():
                                    self.logger.info(f"Match fallback! '{target_address}'[REDACTED_BY_SCRIPT]'s full text. Overriding bad blueprint.")
                                    is_match = True

                                # Check for match
                                if is_match:
                                     if link_url:
                                         target_detail_url = link_url
                                     else:
                                         # MATCH FOUND, BUT NO LINK.
                                         # This implies the result *is* the details view (e.g. a Modal or Popup).
                                         self.logger.info(f"[REDACTED_BY_SCRIPT]")
                                         target_detail_url = "IMPLICIT_DETAILS_VIEW"
                                     break
                            
                            if not target_detail_url and count == 1:
                                # CRITICAL CHECK: Verify this isn't just a Search Page masquerading as a result.
                                # If we found 1 item, no link, and the page has search inputs, it's a failed search.
                                has_search_inputs = await verify_page.locator("input[type='text'], input[type='search'], input[name*='search'], input[name*='ref']").count() > 0
                                has_submit = await verify_page.locator("button[type='submit'], input[type='submit'], button:has-text('Search')").count() > 0
                                
                                if has_search_inputs and has_submit:
                                    self.logger.warning("[REDACTED_BY_SCRIPT]")
                                else:
                                    self.logger.warning("[REDACTED_BY_SCRIPT]")
                                    target_detail_url = link_url if link_url else "IMPLICIT_DETAILS_VIEW"
                                
                        if target_detail_url:
                            break # Success
                            
                        # If we are here, either count was 0 OR we found rows but no match.
                        self.logger.warning(f"[REDACTED_BY_SCRIPT]")
                        
                        if extraction_attempts >= max_extraction_retries:
                            break

                        # Trigger Self-Healing
                        self.logger.info("[REDACTED_BY_SCRIPT]")
                        failure_report = ValidationReport(
                            is_valid=False,
                            original_plan=ref_search_plan, # Context only
                            failure_reason=f"[REDACTED_BY_SCRIPT]'{container_sel}'[REDACTED_BY_SCRIPT]'{target_address}'[REDACTED_BY_SCRIPT]",
                            failure_context_html=search_result.captured_html,
                            failure_screenshot_path=search_result.captured_screenshot_path,
                            failure_stage="RESULTS_PAGE"
                        )
                        
                        # Ask Strategist to fix the blueprints based on the screenshot/HTML
                        new_blueprints = await self.strategist_agent.correct_results_page_blueprint(failure_report)
                        results_blueprints = new_blueprints # Update for next loop
                        
                    except Exception as e:
                        self.logger.error(f"[REDACTED_BY_SCRIPT]")
                        if extraction_attempts >= max_extraction_retries:
                            raise e
                    finally:
                        await verify_page.close()
                        
                    extraction_attempts += 1

            if not target_detail_url:
                 # Final Diagnostic Dump
                 snippet = search_result.captured_html[:1000].replace("\n", " ")
                 self.logger.error(f"[REDACTED_BY_SCRIPT]")
                 raise PipelineError("[REDACTED_BY_SCRIPT]")
            
            # --- SPA & IMPLICIT VIEW DETECTION ---
            is_spa_interaction = False
            is_implicit_view = False
            
            if target_detail_url == "IMPLICIT_DETAILS_VIEW":
                is_spa_interaction = True
                is_implicit_view = True
                self.logger.info("[REDACTED_BY_SCRIPT]")
            
            elif not target_detail_url.startswith("http") and len(target_detail_url) < 50:
                 if target_address in target_detail_url or "/" in target_detail_url:
                     is_spa_interaction = True
                     self.logger.info(f"[REDACTED_BY_SCRIPT]'{target_detail_url}'[REDACTED_BY_SCRIPT]")

            # Only normalize if it's a standard URL
            if not is_spa_interaction:
                if not target_detail_url.startswith("http"):
                     # FIX: Use the final URL from the search result to handle subdomain redirects (Split-Brain Fix)
                     # urljoin correctly handles both root-relative ('/foo') and path-relative ('foo') links against the base.
                     target_detail_url = urljoin(search_result.final_url, target_detail_url)
                self.logger.info(f"[REDACTED_BY_SCRIPT]")
            else:
                self.logger.info(f"[REDACTED_BY_SCRIPT]")

            # --- PHASE 4: TAB NAVIGATION & DOCUMENT HARVESTING ---
            self.logger.info("[REDACTED_BY_SCRIPT]")
            
            details_page = await self.validation_agent.browser_context.new_page()
            document_blueprints = None

            try:
                if is_spa_interaction:
                    # RE-ENACTMENT: We must reach the state again.
                    self.logger.info("[REDACTED_BY_SCRIPT]")
                    
                    # 1. Goto Search
                    # IMPORTANT: Use the URL from the plan if possible, but we fallback to search_page_url
                    await details_page.goto(search_page_url)
                    
                    # 2. Re-run Phase 2 (Search) via Validation Agent
                    # This ensures New Tabs, Soft Fails, and Smart Waits are handled identical to Phase 2.
                    
                    # We must substitute the placeholders first
                    reenact_instructions = copy.deepcopy(ref_search_plan.instructions)
                    for instr in reenact_instructions:
                        params = instr.get('params', {})
                        for key, val in params.items():
                            if isinstance(val, str) and "{{REFERENCE_ID}}" in val:
                                params[key] = val.replace("{{REFERENCE_ID}}", target_address)
                    
                    # EXECUTE via Validation Agent
                    # Returns the active page (which might be a new tab!)
                    details_page = await self.validation_agent.execute_instructions(reenact_instructions, details_page, debug_dir=temp_data_path)
                    self.logger.info(f"[REDACTED_BY_SCRIPT]")

                    # 3. Click Result (ONLY if NOT implicit)
                    if not is_implicit_view:
                        raw_link_selector = results_blueprints['listing_link_blueprint']['selector']
                        container_selector = results_blueprints['[REDACTED_BY_SCRIPT]']
                        
                        # SANITIZATION: Handle '.' selector and Chaining
                        if raw_link_selector.strip() in [".", ":scope", ""]:
                            effective_link_selector = container_selector
                        else:
                            effective_link_selector = f"[REDACTED_BY_SCRIPT]"

                        self.logger.info(f"[REDACTED_BY_SCRIPT]'{effective_link_selector}' to open details.")
                        
                        # Use Validation Agent logic for this click too to handle potential new tabs
                        details_page = await self.validation_agent._execute_instruction({
                            "command": "CLICK_ELEMENT", 
                            "params": {"selector": effective_link_selector}
                        }, details_page)
                        
                        await details_page.wait_for_load_state("networkidle")
                        await self.validation_agent._debug_snapshot(details_page, "spa_result_click_success", save_dir=temp_data_path)
                    else:
                        self.logger.info("[REDACTED_BY_SCRIPT]")
                    
                else:
                    await details_page.goto(target_detail_url, timeout=30000)
                    await details_page.wait_for_load_state("domcontentloaded")
                
                details_html = await details_page.content()

                # CAPTURE SCREENSHOT FOR CONTEXT
                import tempfile
                _, nav_img_path = tempfile.mkstemp(suffix=".png")
                await details_page.screenshot(path=nav_img_path, full_page=True)
                from PIL import Image
                nav_image = Image.open(nav_img_path)
                
                # --- PHASE 4.1: Tab Discovery ---
                tab_prompt = self.strategist_agent.detail_prompt_template.render(html_content=details_html)
                # PASS IMAGE TO GEMINI
                tab_response = await self.strategist_agent.gemini_service.generate_content(tab_prompt, images=nav_image)
                tab_data = self.strategist_agent._parse_llm_json_response(tab_response)
                
                docs_tab_selector = tab_data.get("tabs", {}).get("[REDACTED_BY_SCRIPT]")
                interstitial_selector = tab_data.get("tabs", {}).get("[REDACTED_BY_SCRIPT]")
                
                # --- PHASE 5: DOCUMENT HARVESTING ---
                if docs_tab_selector:
                    # SANITIZATION: Auto-correct common AI selector errors
                    if ":contains(" in docs_tab_selector:
                        self.logger.warning(f"[REDACTED_BY_SCRIPT]'{docs_tab_selector}' -> replaced ':contains' with ':has-text'")
                        docs_tab_selector = docs_tab_selector.replace(":contains(", ":has-text(")

                    self.logger.info(f"[REDACTED_BY_SCRIPT]")
                    await details_page.click(docs_tab_selector)
                    
                    # Robust Wait: Wait for table or list to update
                    # We wait for network idle to ensure AJAX table loads
                    await details_page.wait_for_load_state("networkidle")
                    await asyncio.sleep(2) # Grace period for render

                    # --- INTERSTITIAL HANDLING (The "[REDACTED_BY_SCRIPT]" Trap) ---
                    if interstitial_selector:
                        self.logger.info(f"[REDACTED_BY_SCRIPT]")
                        try:
                            if ":contains(" in interstitial_selector:
                                interstitial_selector = interstitial_selector.replace(":contains(", ":has-text(")
                            
                            await details_page.wait_for_selector(interstitial_selector, timeout=5000)
                            await details_page.click(interstitial_selector)
                            await details_page.wait_for_load_state("networkidle")
                            await asyncio.sleep(2) 
                        except Exception as e:
                            self.logger.warning(f"[REDACTED_BY_SCRIPT]")
                    
                    await self.validation_agent._debug_snapshot(details_page, "[REDACTED_BY_SCRIPT]", save_dir=temp_data_path)

                    doc_html = await details_page.content()
                    
                    # Capture Screenshot for multimodal analysis
                    import tempfile
                    _, img_path = tempfile.mkstemp(suffix=".png")
                    await details_page.screenshot(path=img_path)
                    from PIL import Image
                    doc_image = Image.open(img_path)

                    self.logger.info("[REDACTED_BY_SCRIPT]")
                    document_blueprints_raw = await self.strategist_agent.generate_document_blueprints(doc_html, doc_image)
                    
                    from .data_contracts import DocumentPageProcessing
                    document_blueprints = DocumentPageProcessing.model_validate(document_blueprints_raw)
                    self.logger.info("[REDACTED_BY_SCRIPT]")
                    
                else:
                    self.logger.warning("[REDACTED_BY_SCRIPT]")

            finally:
                await details_page.close()

            # --- FINAL SYNTHESIS ---
            final_instructions = list(ref_search_plan.instructions)
            
            # Step 1: Click the Result (Link) - ONLY IF NOT IMPLICIT
            if not is_implicit_view:
                raw_link_selector = results_blueprints['listing_link_blueprint']['selector']
                container_selector = results_blueprints['[REDACTED_BY_SCRIPT]']
                
                # RE-USE SANITIZATION LOGIC
                if raw_link_selector.strip() in [".", ":scope", ""]:
                    effective_link_selector = container_selector
                else:
                    effective_link_selector = f"[REDACTED_BY_SCRIPT]"
                
                final_instructions.append(Instruction(
                    command="WAIT_FOR_SELECTOR",
                    params={"selector": effective_link_selector},
                    description="[REDACTED_BY_SCRIPT]"
                ))
                final_instructions.append(Instruction(
                    command="CLICK_ELEMENT",
                    params={"selector": effective_link_selector},
                    description="[REDACTED_BY_SCRIPT]"
                ))
            else:
                 # If implicit, we just verify the details container is visible.
                 # We use the container found in Phase 3 (which was the modal itself)
                 container_sel = results_blueprints['[REDACTED_BY_SCRIPT]']
                 final_instructions.append(Instruction(
                    command="WAIT_FOR_SELECTOR",
                    params={"selector": container_sel},
                    description="[REDACTED_BY_SCRIPT]"
                ))

            
            # Step 2: Navigate to Documents (if applicable)
            if docs_tab_selector:
                final_instructions.append(Instruction(
                    command="WAIT_FOR_SELECTOR",
                    params={"selector": docs_tab_selector},
                    description="[REDACTED_BY_SCRIPT]"
                ))
                final_instructions.append(Instruction(
                    command="CLICK_ELEMENT",
                    params={"selector": docs_tab_selector},
                    description="Open Documents tab."
                ))
                
                # Inject Interstitial Step if present
                if interstitial_selector:
                    final_instructions.append(Instruction(
                        command="WAIT_FOR_SELECTOR",
                        params={"selector": interstitial_selector},
                        description="Wait for 'View Documents' link/button."
                    ))
                    final_instructions.append(Instruction(
                        command="CLICK_ELEMENT",
                        params={"selector": interstitial_selector},
                        description="Click 'View Documents' link."
                    ))

                final_instructions.append(Instruction(
                    command="WAIT_FOR_SELECTOR",
                    params={"selector": document_blueprints.document_container_selector if document_blueprints else "table"},
                    description="[REDACTED_BY_SCRIPT]"
                ))

            # Data Contract Sanitization: Ensure blueprints satisfy strict Pydantic models.
            # In Implicit/SPA mode, the AI might return None for specific blueprints.
            if results_blueprints.get("listing_link_blueprint") is None:
                self.logger.info("[REDACTED_BY_SCRIPT]")
                results_blueprints["listing_link_blueprint"] = {
                    "selector": "body",
                    "extraction_method": "ATTRIBUTE",
                    "attribute_name": "id"
                }

            if results_blueprints.get("[REDACTED_BY_SCRIPT]") is None:
                self.logger.info("[REDACTED_BY_SCRIPT]")
                results_blueprints["[REDACTED_BY_SCRIPT]"] = {
                    "selector": "body",
                    "extraction_method": "TEXT"
                }

            self.logger.info(f"[REDACTED_BY_SCRIPT]")

            return PipelineResult(
                status="SUCCESS", 
                domain=domain, 
                plan_path=target_detail_url,
                execution_plan=ExecutionPlan(
                    domain=domain,
                    instructions=final_instructions, 
                    results_page_processing=ResultsPageProcessing.model_validate(results_blueprints),
                    detail_page_processing=DetailPageProcessing(root={}),
                    document_page_processing=document_blueprints # THE NEW HARVEST
                )
            )

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            return PipelineResult(status="FAILURE", domain=domain, error_message=str(e))
            

    def _find_best_candidate(self, triage_result: TriageResult) -> str | None:
        """[REDACTED_BY_SCRIPT]"""
        # Priority 1: Explicit SEARCH_PAGE
        for url, category in triage_result.full_classification.items():
            if category == "SEARCH_PAGE":
                return url
        
        # Priority 2: DISCLAIMER_PAGE (Valid entry point)
        for url, category in triage_result.full_classification.items():
            if category == "DISCLAIMER_PAGE":
                return url

        # Priority 3: LANDING_PAGE (Valid entry point)
        for url, category in triage_result.full_classification.items():
            if category == "LANDING_PAGE":
                return url

        # Priority 3: Trust the Triage Agent's sort order
        # The TriageAgent populates candidate_urls based on _SELECTION_PRIORITY.
        # If we have candidates, the first one is the best available bet.
        if triage_result.candidate_urls:
             best_url = list(triage_result.candidate_urls.keys())[0]
             self.logger.info(f"[REDACTED_BY_SCRIPT]")
             return best_url
             
        return None

    def _combine_probe_and_adaptive(
        self,
        probe_plan: 'DraftPlan',
        adaptive_results_components: dict,
        adaptive_final_instruction: list[dict],
        detail_page_processing: dict
    ) -> 'DraftPlan':
        """[REDACTED_BY_SCRIPT]"""
        combined_instructions = probe_plan.instructions + adaptive_final_instruction
        
        return DraftPlan(
            instructions=combined_instructions,
            source_triage_result=probe_plan.source_triage_result,
            target_address=probe_plan.target_address,
            target_description=probe_plan.target_description,
            results_page_processing=adaptive_results_components,
            detail_page_processing=detail_page_processing
        )

    def _handle_failure(self, domain: str, error: Exception):
        """
        Standardized procedure for handling a pipeline failure.
        """
        self.logger.info(f"[REDACTED_BY_SCRIPT]")
        pass