import asyncio
import logging
import json
import csv
import sys
import os
import tempfile
from pathlib import Path
from playwright.async_api import async_playwright

# --- PATH INJECTION (CRITICAL FOR LOCAL EXECUTION) ---
# Allows src/main.py to see cognitive_profiler/ as a module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from cognitive_profiler.settings import settings
from cognitive_profiler.orchestrator import CognitiveProfiler
from cognitive_profiler.services.api_key_manager import SimpleApiKeyManager
from cognitive_profiler.services.gemini_service import GeminiService
from cognitive_profiler.agents.triage_agent import TriageAgent
from cognitive_profiler.agents.strategist_agent import StrategistAgent
from cognitive_profiler.agents.validation_agent import ValidationAgent
from cognitive_profiler.agents.synthesis_agent import SynthesisAgent
from cognitive_profiler.services.local_crawler import LocalCrawler

# --- CONFIGURATION (YOUR PATHS) ---
LPA_REGISTRY_PATH = r"[REDACTED_BY_SCRIPT]"
CSV_PATH = r"[REDACTED_BY_SCRIPT]"
OUTPUT_DIR = r"[REDACTED_BY_SCRIPT]"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[REDACTED_BY_SCRIPT]',
        handlers=[
            logging.FileHandler("amaryllis_run.log"),
            logging.StreamHandler()
        ]
    )

from datetime import datetime

def get_test_candidates(lpa_name: str, csv_data: list, limit: int = 15) -> list[str]:
    """
    Finds a list of valid reference IDs for the given LPA from the CSV.
    Prioritizes recent applications (2015+).
    """
    # Normalize lpa_name
    search_term = lpa_name.lower().replace("council", "").strip()
    
    candidates = []
    
    for row in csv_data:
        # 1. Filter by LPA Name
        csv_authority = row.get('Planning Authority', '').lower()
        if search_term not in csv_authority:
            continue

        # 2. Filter out withdrawn/abandoned/expired
        dev_status = row.get('[REDACTED_BY_SCRIPT]', '').lower()
        withdrawn_date = row.get('[REDACTED_BY_SCRIPT]', '').strip()
        expired_date = row.get('[REDACTED_BY_SCRIPT]', '').strip()
        
        if "withdrawn" in dev_status or "abandoned" in dev_status or "expired" in dev_status or withdrawn_date != "" or expired_date != "":
            continue

        # 3. Filter by Date (>= 2015)
        # Try '[REDACTED_BY_SCRIPT]', fallback to 'Record Last Updated' if needed, 
        # but user specifically asked for submission date.
        submitted_date_str = row.get('[REDACTED_BY_SCRIPT]', '').strip()
        
        # If no submission date, we might skip or check another field. 
        # Sticking to strict requirement: "[REDACTED_BY_SCRIPT]"
        if not submitted_date_str:
            continue
            
        try:
            # Parse dd/mm/yyyy
            submitted_date = datetime.strptime(submitted_date_str, "%d/%m/%Y")
            if submitted_date.year < 2015:
                continue
        except ValueError:
            # If date parse fails, safer to skip than risk a bad ID? Or include?
            # User requirement is strict on "before 2015 ignored".
            continue

        # 4. Valid ID check
        ref = row.get('[REDACTED_BY_SCRIPT]')
        if ref and len(str(ref)) > 3:
            candidates.append(str(ref))
            if len(candidates) >= limit:
                break
                
    return candidates

async def main():
    setup_logging()
    logger = logging.getLogger("AmaryllisLocal")

    # 1. Load Configuration Data
    if not os.path.exists(LPA_REGISTRY_PATH):
        logger.error(f"[REDACTED_BY_SCRIPT]")
        return

    with open(LPA_REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    
    csv_data = []
    if os.path.exists(CSV_PATH):
        try:
            # Primary attempt: Windows-1252 (Standard for Excel on Windows)
            with open(CSV_PATH, 'r', encoding='cp1252') as f:
                reader = csv.DictReader(f)
                csv_data = list(reader)
        except UnicodeDecodeError:
            logger.warning("[REDACTED_BY_SCRIPT]")
            # Fallback: Latin-1 (Aggressive fallback that rarely raises errors)
            with open(CSV_PATH, 'r', encoding='latin1') as f:
                reader = csv.DictReader(f)
                csv_data = list(reader)
    else:
        logger.error(f"[REDACTED_BY_SCRIPT]")
        return

    # Ensure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 2. Initialize Services
    api_key_manager = SimpleApiKeyManager(keys=settings.gemini_api_keys)
    gemini_service = GeminiService(
        api_key_manager=api_key_manager, 
        models=settings.gemini_models
    )

    async with async_playwright() as p:
        # Launch Headed for debugging (change to headless=True later)
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent="[REDACTED_BY_SCRIPT]"
        )
        
        crawler = LocalCrawler(context)
        
        # Initialize Agents
        triage_agent = TriageAgent(gemini_service=gemini_service)
        strategist_agent = StrategistAgent(gemini_service=gemini_service)
        # Project Atlas: Increasing global timeout to 30s to accommodate slow portals (Antrim, etc.)
        validation_agent = ValidationAgent(browser_context=context, default_timeout_ms=30000)
        synthesis_agent = SynthesisAgent() # Local instantiation
        
        profiler = CognitiveProfiler(
            triage_agent=triage_agent,
            strategist_agent=strategist_agent,
            validation_agent=validation_agent,
            synthesis_agent=synthesis_agent, 
            max_correction_attempts=2,
            validation_retries=1
        )

        # 3. Iterate through LPAs
        for lpa_entry in registry:
            lpa_name = lpa_entry.get('lpa_name_clean')
            portal_url = lpa_entry.get('portal_url')
            
            if not lpa_name or not portal_url:
                continue

            # Check if we already have a plan
            plan_file = Path(OUTPUT_DIR) / f"{lpa_name}.json"
            if plan_file.exists():
                logger.info(f"[REDACTED_BY_SCRIPT]")
                continue

            logger.info(f"[REDACTED_BY_SCRIPT]")

            # A. Find Test Candidates
            candidates = get_test_candidates(lpa_name, csv_data)
            if not candidates:
                logger.warning(f"[REDACTED_BY_SCRIPT]")
                continue
            
            logger.info(f"[REDACTED_BY_SCRIPT]")

            try:
                # B. Crawl for Context (The Spider) - ONCE per LPA
                # The search page URL is the same regardless of the ID we verify.
                artifacts = await crawler.crawl(portal_url, lpa_name)
                if not artifacts:
                    logger.error(f"[REDACTED_BY_SCRIPT]")
                    continue

                # C. Run The Cognitive Profiler (Orchestrator handles retries)
                # Define a predictable debug path: output_dir/lpa_name/debug_snapshots
                debug_path = Path(OUTPUT_DIR) / lpa_name / "debug_snapshots"
                debug_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"[REDACTED_BY_SCRIPT]")

                result = await profiler.run_pipeline(
                    artifact_manifest=artifacts,
                    target_candidates=candidates, 
                    target_description="[REDACTED_BY_SCRIPT]",
                    temp_data_path=str(debug_path)
                )

                if result.status == "SUCCESS" and result.execution_plan:
                    # Save the Plan
                    with open(plan_file, 'w', encoding='utf-8') as f:
                        json.dump(result.execution_plan.model_dump(mode='json'), f, indent=2)
                    logger.info(f"[REDACTED_BY_SCRIPT]")
                else:
                    logger.error(f"[REDACTED_BY_SCRIPT]")

            except Exception as e:
                logger.critical(f"[REDACTED_BY_SCRIPT]", exc_info=True)
            
            # Brief pause to be polite
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())