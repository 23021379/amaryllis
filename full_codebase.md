c:\Users\brand\Desktop\renewables\amaryllis\codebase_to_markdown.py
```
import os

def create_markdown_from_codebase(root_dir, output_file):
    """
    Traverses the directory tree starting from root_dir, identifies all Python files,
    and consolidates their content into a single Markdown file.
    Each file section is prefixed with its full path and wrapped in a code block.
    """
    with open(output_file, 'w', encoding='utf-8') as md_file:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.py'):
                    full_path = os.path.join(dirpath, filename)
                    
                    # Optional: Skip the output file itself if it happens to be a .py file (unlikely for .md)
                    # or skip this script itself if desired. Currently including everything as requested.
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8') as source_file:
                            content = source_file.read()
                            
                        # Write the full path
                        md_file.write(f"{full_path}\n")
                        # Write the code block start
                        md_file.write("```\n")
                        # Write the file content
                        md_file.write(content)
                        # Ensure there's a newline before closing the block
                        if content and not content.endswith('\n'):
                            md_file.write('\n')
                        # Write the code block end
                        md_file.write("```\n\n")
                        
                        print(f"Processed: {full_path}")
                        
                    except Exception as e:
                        print(f"Failed to process {full_path}: {e}")

if __name__ == "__main__":
    # define the root directory as the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Output file name
    output_md_path = os.path.join(current_dir, 'full_codebase.md')
    
    print(f"Scanning directory: {current_dir}")
    print(f"Writing to: {output_md_path}")
    
    create_markdown_from_codebase(current_dir, output_md_path)
    
    print("Done!")
```

c:\Users\brand\Desktop\renewables\amaryllis\extract_scripts_to_md.py
```

import os

def extract_python_scripts(root_dir, output_file):
    # Get the absolute path of the current script and output file to avoid including them
    current_script = os.path.abspath(__file__)
    output_file_abs = os.path.abspath(output_file)

    with open(output_file, 'w', encoding='utf-8') as md_file:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = os.path.join(dirpath, filename)
                    abs_path = os.path.abspath(file_path)

                    # Skip this script and the output file (though output file usually isn't .py)
                    if abs_path == current_script:
                        continue
                    
                    # Calculate relative path for display
                    relative_path = os.path.relpath(file_path, root_dir)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as source_file:
                            content = source_file.read()
                            
                        # Write to markdown
                        md_file.write(f"{relative_path}\n")
                        md_file.write("```\n")
                        md_file.write(content)
                        if not content.endswith('\n'):
                            md_file.write('\n')
                        md_file.write("```\n\n")
                        
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    extract_python_scripts('.', 'codebase_summary.md')
    print("Extraction complete. Saved to codebase_summary.md")
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\command_schema.py
```
COMMAND_SCHEMA = {
    "type": "array",
    "description": "A sequence of instructions for the web automation robot.",
    "items": {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "GOTO_URL"},
                    "params": {
                        "type": "object",
                        "properties": {"url": {"type": "string", "format": "uri"}},
                        "required": ["url"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "FILL_INPUT"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "value": {"type": "string"}
                        },
                        "required": ["selector", "value"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "CLICK_ELEMENT"},
                    "params": {
                        "type": "object",
                        "properties": {"selector": {"type": "string"}},
                        "required": ["selector"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "CHECK_OPTION"},
                    "params": {
                        "type": "object",
                        "properties": {"selector": {"type": "string"}},
                        "required": ["selector"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "PRESS_KEY"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "key": {"type": "string"}
                        },
                        "required": ["selector", "key"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "WAIT_FOR_SELECTOR"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "state": {
                                "type": "string", 
                                "enum": ["visible", "hidden", "attached", "detached"],
                                "description": "The state to wait for. Defaults to 'visible'."
                            }
                        },
                        "required": ["selector"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "WAIT_FOR_NAVIGATION"},
                    "params": {"type": "object"},
                    "description": {"type": "string"}   
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "SELECT_OPTION"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "value": {"type": "string"}
                        },
                        "required": ["selector", "value"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            }
        ]
    }
}
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\data_contracts.py
```
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pydantic import BaseModel, Field, RootModel

# --- Ingestion Contracts ---
@dataclass
class DownloadedArtifact:
    """A unified record linking a URL to its downloaded files."""
    url: str
    html_path: str
    screenshot_path: str | None = None

# --- Orchestrator Contracts ---
@dataclass
class PipelineResult:
    """Standardized output from the CognitiveProfiler."""
    status: Literal["SUCCESS", "FAILURE", "CRASH"]
    domain: str
    plan_path: str | None = None
    error_message: str | None = None
    execution_plan: 'ExecutionPlan' | None = None

@dataclass
class ProbeResult:
    """
    The ground-truth context captured after successfully executing a Probe Plan.
    This is the bridge between the ValidationAgent and the adaptive StrategistAgent.
    """
    is_successful: bool
    probe_plan: list[dict[str, Any]] # The plan that was executed
    final_url: str | None = None
    captured_html: str | None = None
    captured_screenshot_path: str | None = None
    failure_reason: str | None = None # Reason if is_successful is False

# --- Triage Agent Contracts ---
@dataclass
class HtmlPageMetadata:
    url: str
    path: str
    title: str | None
    description: str | None
    screenshot_path: str | None = None
    structural_summary: dict[str, int] = field(default_factory=dict)

@dataclass
class TriageResult:
    """Standardized output from the TriageAgent."""
    domain: str
    candidate_urls: dict[str, str]
    full_classification: dict[str, str]
    source_metadata: list[HtmlPageMetadata]

# --- Strategist Agent Contracts ---

@dataclass
class DraftPlan:
    """The internal representation of the plan being built, ready for validation."""
    instructions: list[dict[str, Any]]
    source_triage_result: TriageResult
    target_address: str
    target_description: str
    results_page_processing: dict[str, Any] | None = None
    # MODIFIED: This now holds a dynamically discovered dictionary of blueprints.
    detail_page_processing: dict[str, Any] | None = None
    # --- PROJECT ATLAS DIRECTIVE: MULTI-STEP NAVIGATION ---
    is_navigation_only: bool = False


# --- Validation Agent Contracts ---
@dataclass
class ValidationReport:
    """Standardized output from the ValidationAgent."""
    is_valid: bool
    original_plan: DraftPlan
    failure_index: int | None = None
    failure_reason: str | None = None
    failure_context_html: str | None = None
    failure_screenshot_path: str | None = None
    failure_type: Literal[
        "TIMEOUT",
        "VALIDATION_ERROR",
        "AMBIGUOUS_SELECTOR",
        "MISSION_FAILURE",
        "PROBABLE_OBSTRUCTION",
        "SEARCH_OUTCOME_MISMATCH",
        "UNKNOWN"
    ] | None = None
    failure_analysis_notes: str | None = None
    # --- PROJECT SCALPEL DIRECTIVE: ADD EXPLICIT FAILURE STAGE ---
    failure_stage: Literal["NAVIGATION", "RESULTS_PAGE", "DETAIL_PAGE"] | None = None
    # --- PROJECT PHOENIX DIRECTIVE: ADD SUCCESS CONTEXT ---
    final_url: str | None = None
    final_html: str | None = None
    final_screenshot_path: str | None = None

# --- Synthesis Agent & Final Output Contracts (Pydantic) ---
class Instruction(BaseModel):
    """A single, machine-readable command in the Execution Plan."""
    command: str
    params: dict[str, Any]
    description: str | None = Field(None, description="An optional human-readable description of the step's purpose.")

from pydantic import model_validator

# --- New Extraction Blueprint Contracts ---

class PostProcessingStep(BaseModel):
    """A single, explicit data transformation step."""
    method: Literal["REGEX_EXTRACT"] = Field(
        description="The transformation method to apply."
    )
    params: dict[str, Any] = Field(
        description="Parameters for the method, e.g., {'pattern': '...'}"
    )

    @model_validator(mode='after')
    def check_params(self) -> 'PostProcessingStep':
        if self.method == "REGEX_EXTRACT":
            if "pattern" not in self.params or not isinstance(self.params["pattern"], str):
                raise ValueError("REGEX_EXTRACT method requires a 'pattern' string parameter.")
        return self

class ExtractionBlueprint(BaseModel):
    """
    An unambiguous blueprint for extracting and transforming data from a webpage.
    """
    selector: str = Field(description="The CSS selector to locate the target element(s).")
    extraction_method: Literal["TEXT", "ATTRIBUTE", "MULTIPLE_TEXT", "MULTIPLE_ATTRIBUTE"] = Field(
        description="The method to use for extracting data from the located element(s)."
    )
    attribute_name: str | None = Field(
        None, 
        description="Specifies the attribute name (e.g., 'href', 'src', 'data-price') when extraction_method is 'ATTRIBUTE' or 'MULTIPLE_ATTRIBUTE'."
    )
    post_processing: list[PostProcessingStep] | None = Field(
        None,
        description="An optional list of post-processing steps to apply to the extracted data."
    )

    @model_validator(mode='after')
    def check_attribute_name_logic(self) -> 'ExtractionBlueprint':
        if self.extraction_method in ["ATTRIBUTE", "MULTIPLE_ATTRIBUTE"]:
            if not self.attribute_name:
                raise ValueError("'attribute_name' must be provided for ATTRIBUTE extraction methods.")
        elif self.attribute_name:
            raise ValueError("'attribute_name' must be null for non-ATTRIBUTE extraction methods.")
        return self

# --- Modified Processing Contracts ---

class PaginationBlueprint(BaseModel):
    """A blueprint for navigating through paginated search results."""
    next_page_selector: str = Field(description="A robust CSS selector for the element that navigates to the next page of results (e.g., a 'Next' button or 'Page 2' link).")

class ResultsPageProcessing(BaseModel):
    """Selectors for extracting data from a page listing multiple properties."""
    listing_container_selector: str = Field(description="CSS selector for the repeating element that contains each individual property listing.")
    
    listing_link_blueprint: ExtractionBlueprint = Field(
        description="The blueprint for extracting the property detail page URL from within the listing container."
    )
    listing_address_blueprint: ExtractionBlueprint = Field(
        description="The blueprint for extracting the visible property address text from within the listing container."
    )
    # --- PROJECT ATLAS DIRECTIVE: ADD PAGINATION SUPPORT ---
    pagination_blueprint: PaginationBlueprint | None = Field(
        None,
        description="An optional blueprint describing how to navigate to the next page of results. If null, the results are assumed to be on a single page."
    )

# REFACTOR: Use RootModel directly for a cleaner, more idiomatic Pydantic v2 implementation.
class DetailPageProcessing(RootModel[dict[str, ExtractionBlueprint | None]]):
    def __iter__(self):
        return iter(self.root)
    def __getitem__(self, item):
        return self.root[item]
    def items(self):
        return self.root.items()

# --- PROJECT DEMETER: DOCUMENT HARVESTING CONTRACTS ---
class DocumentPageProcessing(BaseModel):
    """Blueprints for extracting document links and metadata from the documents tab."""
    document_container_selector: str = Field(description="CSS selector for the row/card containing a single document.")
    document_link_blueprint: ExtractionBlueprint = Field(description="Blueprint to extract the download/view URL.")
    document_date_blueprint: ExtractionBlueprint | None = Field(None, description="Blueprint to extract the document date.")
    document_type_blueprint: ExtractionBlueprint | None = Field(None, description="Blueprint to extract the document type (e.g., 'Decision Notice').")
    pagination_blueprint: PaginationBlueprint | None = Field(None, description="Navigation logic for multi-page document lists.")

class ExecutionPlan(BaseModel):
    """The final, production-ready, four-phase plan for the UPLC Actor."""
    domain: str
    plan_version: str = "3.0" # Major version bump for Document Harvesting
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    instructions: list[Instruction]
    results_page_processing: ResultsPageProcessing
    detail_page_processing: DetailPageProcessing
    document_page_processing: DocumentPageProcessing | None = None
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\exceptions.py
```
class PipelineError(Exception):
    """Base exception for all predictable pipeline failures."""
    pass

class TriageError(PipelineError):
    """Failure during the Triage stage."""
    pass

class StrategyError(PipelineError):
    """Failure during the Strategy stage."""
    pass

class ValidationError(PipelineError):
    """Failure during the Validation stage that is not correctable."""
    pass

class MaxRetriesExceededError(ValidationError):
    """Thrown when the self-healing loop fails too many times."""
    pass

class SynthesisError(PipelineError):
    """A critical failure during the final Synthesis and serialization stage."""
    pass

class PreflightCheckError(PipelineError):
    """A critical, non-recoverable error in the upstream data contract."""
    pass
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\orchestrator.py
```
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

    async def run_pipeline(self, artifact_manifest: list[DownloadedArtifact], target_address: str, target_description: str, temp_data_path: str) -> PipelineResult:
        """Executes the iterative discovery and search pipeline."""
        domain = "unknown_domain"
        
        try:
            # --- PHASE 1: DISCOVERY ---
            self.logger.info("--- PHASE 1: DISCOVERY ---")
            
            # Step A: Passive Triage
            search_page_url = None
            try:
                triage_result = await self.triage_agent.run(artifact_manifest)
                domain = triage_result.domain
                
                # Check if we found the Golden Ticket (SEARCH_PAGE)
                search_page_url = self._find_best_candidate(triage_result)
            except TriageError:
                self.logger.warning("Passive triage yielded no candidates.")

            # Step B: Active Fallback (Global Search)
            if not search_page_url:
                self.logger.warning("No Search Page found via crawling. Initiating Active Global Search Fallback.")
                
                # 1. Get Homepage HTML
                homepage_artifact = next((a for a in artifact_manifest if "homepage" in a.html_path), artifact_manifest[0])
                homepage_html = Path(homepage_artifact.html_path).read_text(encoding='utf-8')

                # 2. Generate Plan
                # PASS URL TO AGENT to ensure GOTO_URL is included
                search_plan = await self.strategist_agent.generate_global_search_plan(homepage_artifact.url, homepage_html)
                
                # 3. Execute Plan (Blindly for now, assuming it works)
                self.logger.info("Executing Global Search Plan...")
                search_result = await self.validation_agent.execute_probe(search_plan, temp_data_path)
                
                if not search_result.is_successful:
                    raise StrategyError(f"Global Search Fallback failed: {search_result.failure_reason}")

                # 4. Analyze Search Results (Harvest Mode)
                self.logger.info("Analyzing Global Search Results... Harvesting high-value links.")
                fallback_artifacts = []
                
                # Add the result page itself (just in case)
                main_artifact = DownloadedArtifact(
                    url=search_result.final_url,
                    html_path="memory://search_results", 
                    screenshot_path=search_result.captured_screenshot_path
                )
                main_path = Path(temp_data_path) / "fallback_search_results.html"
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
                
                self.logger.info(f"Harvested {len(top_links)} candidate links from search results: {[u for s, u in top_links]}")

                # Fetch Content for Top Links
                for i, (_, url) in enumerate(top_links):
                    try:
                        page = await self.validation_agent.browser_context.new_page()
                        await page.goto(url, timeout=15000) # Short timeout
                        await page.wait_for_load_state("domcontentloaded")
                        content = await page.content()
                        
                        # Save artifact
                        art_path = Path(temp_data_path) / f"fallback_harvest_{i}.html"
                        art_path.write_text(content, encoding='utf-8')
                        
                        fallback_artifacts.append(DownloadedArtifact(
                            url=url,
                            html_path=str(art_path),
                            screenshot_path=None # Optimization: Skip screenshot for triage
                        ))
                    except Exception as e:
                        self.logger.warning(f"Failed to harvest fallback link {url}: {e}")
                    finally:
                        await page.close()

                # 5. Re-Run Triage on the BATCH
                if not fallback_artifacts:
                     raise PipelineError("Global Search yielded no artifacts to analyze.")

                triage_result = await self.triage_agent.run(fallback_artifacts)
                search_page_url = self._find_best_candidate(triage_result)
                
                if not search_page_url:
                    raise PipelineError("Even Global Search failed to find the Planning Search Page. Critical Failure.")

            self.logger.info(f"DISCOVERY SUCCESS. Target Search Page: {search_page_url}")

            # --- PHASE 2: TARGETED SEARCH EXECUTION ---
            self.logger.info("--- PHASE 2: REFERENCE ID SEARCH ---")

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
            max_nav_iterations = 5 # Prevent infinite loops
            current_url = search_page_url
            current_html = search_page_html
            
            ref_search_plan = None
            search_result = None

            # --- PHASE 2 LOOP: UNIVERSAL NAVIGATION & CORRECTION ---
            # We treat every step (intermediate or final) as a "mission" that can fail and be corrected.
            navigation_iterations = 0
            max_nav_iterations = 5 
            current_url = search_page_url
            current_html = search_page_html
            
            ref_search_plan = None
            search_result = None

            while navigation_iterations < max_nav_iterations:
                self.logger.info(f"Phase 2 Iteration {navigation_iterations + 1}: Generating Plan for {current_url}")
                
                # 1. Generate the Plan
                ref_search_plan = await self.strategist_agent.generate_planning_search_plan(
                    search_page_url=current_url,
                    search_page_html=current_html,
                    target_ref_id=target_address
                )

                # 2. Execute with Self-Healing (Logic applies to BOTH navigation-only and final-search)
                # The only difference is the success exit condition.
                
                plan_result = await self.validation_agent.execute_probe(ref_search_plan, temp_data_path)
                
                correction_attempts = 0
                while not plan_result.is_successful and correction_attempts < self.max_correction_attempts:
                     correction_attempts += 1
                     self.logger.warning(f"Plan Execution Failed (Attempt {correction_attempts}/{self.max_correction_attempts}). Initiating Self-Healing...")
                     
                     # Diagnosis
                     analysis_notes = None
                     if plan_result.captured_html:
                          _, analysis_notes = self.strategist_agent._diagnose_timeout_failure(plan_result.captured_html)

                     failure_report = ValidationReport(
                         is_valid=False,
                         original_plan=ref_search_plan,
                         failure_reason=plan_result.failure_reason,
                         failure_context_html=plan_result.captured_html,
                         failure_screenshot_path=plan_result.captured_screenshot_path,
                         failure_stage="NAVIGATION",
                         failure_analysis_notes=analysis_notes
                     )
                     
                     # Correction
                     corrected_instructions = await self.strategist_agent.correct_navigation_plan(failure_report)
                     ref_search_plan.instructions = corrected_instructions
                     
                     # Re-apply placeholder substitution logic (only if it's the final search)
                     if not ref_search_plan.is_navigation_only:
                         for instr in ref_search_plan.instructions:
                             if instr.get("command") == "FILL_INPUT":
                                 instr["params"]["value"] = "{{REFERENCE_ID}}"

                     # Retry
                     plan_result = await self.validation_agent.execute_probe(ref_search_plan, temp_data_path)

                if not plan_result.is_successful:
                    # If it still fails after corrections, this specific iteration is dead.
                    raise StrategyError(f"Plan failed after {correction_attempts} corrections. Final Error: {plan_result.failure_reason}")

                # 3. Handle Success
                if ref_search_plan.is_navigation_only:
                    self.logger.info("Intermediate navigation step successful. Advancing state.")
                    current_url = plan_result.final_url
                    current_html = plan_result.captured_html
                    navigation_iterations += 1
                    continue # Loop back to generate the next step
                else:
                    self.logger.info("Final Search Plan executed successfully.")
                    search_result = plan_result # Promote to final result
                    break # Success! Exit the loop.

            if navigation_iterations >= max_nav_iterations:
                raise StrategyError(f"Navigation limit reached ({max_nav_iterations}). Could not reach search form.")

            self.logger.info("REFERENCE SEARCH SUCCESS. Analyzing Results Page...")

            # --- PHASE 3: RESULTS EXTRACTION ---
            self.logger.info(f"Current URL: {search_result.final_url}")

            # 1. Check for Direct Redirect (Skip Extraction)
            # Idox URL pattern for details: 'applicationDetails.do'
            # Northgate URL pattern for details: 'generic/public/planning/web/search/results' (No, that's results)
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
            html_has_header = "application summary" in html_lower or "planning – application summary" in html_lower

            is_direct_details = (url_has_details or html_has_header) and not is_list_view_url
            
            self.logger.info(
                f"Direct Redirect Analysis: URL='{search_result.final_url}' | "
                f"Is List View={is_list_view_url} | "
                f"Has Detail Keywords={url_has_details or html_has_header} -> "
                f"Is Direct Details={is_direct_details}"
            )
            target_detail_url = None
            results_blueprints = {} 

            if is_direct_details:
                self.logger.info("Direct Redirect detected. Skipping list extraction.")
                target_detail_url = search_result.final_url
                # Dummy schema
                results_blueprints = {
                    "listing_container_selector": "body",
                    "listing_link_blueprint": {"selector": "body", "extraction_method": "ATTRIBUTE", "attribute_name": "id"},
                    "listing_address_blueprint": {"selector": "body", "extraction_method": "TEXT"}
                }
            else:
                # 2. Perform Extraction
                results_blueprints, wait_instruction = await self.strategist_agent.generate_adaptive_components(search_result)
                self.logger.info(f"Generated extraction blueprints. Container: {results_blueprints['listing_container_selector']}")

                # 3. Verify & Extract Link (With Self-Healing)
                extraction_attempts = 0
                max_extraction_retries = 2
                
                while extraction_attempts <= max_extraction_retries:
                    container_sel = results_blueprints['listing_container_selector']
                    link_bp = results_blueprints['listing_link_blueprint']
                    ref_bp = results_blueprints['listing_address_blueprint']
                    
                    verify_page = await self.validation_agent.browser_context.new_page()
                    try:
                        await verify_page.set_content(search_result.captured_html)
                        count = await verify_page.locator(container_sel).count()
                        self.logger.info(f"Attempt {extraction_attempts+1}: Found {count} result rows with selector '{container_sel}'.")
                        
                        if count > 0:
                            for i in range(count):
                                row = verify_page.locator(container_sel).nth(i)
                                
                                async def extract_field(bp):
                                    if not bp: return ""
                                    sel = bp.get('selector')
                                    if not sel: return ""

                                    # SANITIZATION: Auto-correct common AI selector errors
                                    if ":contains(" in sel:
                                        self.logger.warning(f"Sanitizing invalid selector '{sel}' -> replaced ':contains' with ':has-text'")
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
                                
                                # Check for match
                                if target_address.replace("/","").lower() in ref_text.replace("/","").lower():
                                     if link_url:
                                         target_detail_url = link_url
                                     else:
                                         # MATCH FOUND, BUT NO LINK.
                                         # This implies the result *is* the details view (e.g. a Modal or Popup).
                                         self.logger.info(f"Ref match found in Row {i}, but link is empty. Assuming Implicit Details View.")
                                         target_detail_url = "IMPLICIT_DETAILS_VIEW"
                                     break
                            
                            if not target_detail_url and count == 1:
                                self.logger.warning("Single result found with no exact text match. Assuming match.")
                                target_detail_url = link_url if link_url else "IMPLICIT_DETAILS_VIEW"
                                
                        if target_detail_url:
                            break # Success
                            
                        # If we are here, either count was 0 OR we found rows but no match.
                        self.logger.warning(f"Extraction Attempt {extraction_attempts+1} failed. Rows: {count}. Target found: {bool(target_detail_url)}.")
                        
                        if extraction_attempts >= max_extraction_retries:
                            break

                        # Trigger Self-Healing
                        self.logger.info("Triggering Extraction Self-Healing...")
                        failure_report = ValidationReport(
                            is_valid=False,
                            original_plan=ref_search_plan, # Context only
                            failure_reason=f"Extraction failed. Selector '{container_sel}' found {count} items. Target '{target_address}' not found in results.",
                            failure_context_html=search_result.captured_html,
                            failure_screenshot_path=search_result.captured_screenshot_path,
                            failure_stage="RESULTS_PAGE"
                        )
                        
                        # Ask Strategist to fix the blueprints based on the screenshot/HTML
                        new_blueprints = await self.strategist_agent.correct_results_page_blueprint(failure_report)
                        results_blueprints = new_blueprints # Update for next loop
                        
                    except Exception as e:
                        self.logger.error(f"Extraction error: {e}")
                        if extraction_attempts >= max_extraction_retries:
                            raise e
                    finally:
                        await verify_page.close()
                        
                    extraction_attempts += 1

            if not target_detail_url:
                 # Final Diagnostic Dump
                 snippet = search_result.captured_html[:1000].replace("\n", " ")
                 self.logger.error(f"HTML Dump (Head): {snippet}")
                 raise PipelineError("Could not identify a link to the application details in the search results after retries.")
            
            # --- SPA & IMPLICIT VIEW DETECTION ---
            is_spa_interaction = False
            is_implicit_view = False
            
            if target_detail_url == "IMPLICIT_DETAILS_VIEW":
                is_spa_interaction = True
                is_implicit_view = True
                self.logger.info("DETECTION: Implicit Details View (Modal/Popup). No navigation required after search.")
            
            elif not target_detail_url.startswith("http") and len(target_detail_url) < 50:
                 if target_address in target_detail_url or "/" in target_detail_url:
                     is_spa_interaction = True
                     self.logger.info(f"SPA DETECTION: Extracted link '{target_detail_url}' is likely a JS trigger.")

            # Only normalize if it's a standard URL
            if not is_spa_interaction:
                if not target_detail_url.startswith("http"):
                     base_domain = f"{urlparse(search_page_url).scheme}://{urlparse(search_page_url).netloc}"
                     target_detail_url = base_domain + target_detail_url if target_detail_url.startswith("/") else f"{base_domain}/{target_detail_url}"
                self.logger.info(f"Target Detail URL identified: {target_detail_url}")
            else:
                self.logger.info(f"SPA/Implicit Mode confirmed. Placeholder: {target_detail_url}")

            # --- PHASE 4: TAB NAVIGATION & DOCUMENT HARVESTING ---
            self.logger.info("--- PHASE 4: DOCUMENT ACCESS ---")
            
            details_page = await self.validation_agent.browser_context.new_page()
            document_blueprints = None

            try:
                if is_spa_interaction:
                    # RE-ENACTMENT: We must reach the state again.
                    self.logger.info("SPA Mode: Re-enacting search to restore state...")
                    
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
                    self.logger.info(f"Re-enactment complete. Active page URL: {details_page.url}")

                    # 3. Click Result (ONLY if NOT implicit)
                    if not is_implicit_view:
                        result_selector = results_blueprints['listing_link_blueprint']['selector']
                        self.logger.info("SPA Mode: Clicking result link to open details.")
                        
                        # Use Validation Agent logic for this click too to handle potential new tabs
                        details_page = await self.validation_agent._execute_instruction({
                            "command": "CLICK_ELEMENT", 
                            "params": {"selector": result_selector}
                        }, details_page)
                        
                        await details_page.wait_for_load_state("networkidle")
                        await self.validation_agent._debug_snapshot(details_page, "spa_result_click_success", save_dir=temp_data_path)
                    else:
                        self.logger.info("Implicit Mode: Skipping result click. Details should be already visible (Modal/Popup).")
                    
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
                tab_response = await self.strategist_agent.gemini_service.generate_content(tab_prompt, image=nav_image)
                tab_data = self.strategist_agent._parse_llm_json_response(tab_response)
                
                docs_tab_selector = tab_data.get("tabs", {}).get("documents_tab_selector")
                
                # --- PHASE 5: DOCUMENT HARVESTING ---
                if docs_tab_selector:
                    self.logger.info(f"Found Documents Tab: {docs_tab_selector}. Navigating...")
                    await details_page.click(docs_tab_selector)
                    
                    # Robust Wait: Wait for table or list to update
                    # We wait for network idle to ensure AJAX table loads
                    await details_page.wait_for_load_state("networkidle")
                    await asyncio.sleep(2) # Grace period for render
                    
                    await self.validation_agent._debug_snapshot(details_page, "document_tab_click_success", save_dir=temp_data_path)

                    doc_html = await details_page.content()
                    
                    # Capture Screenshot for multimodal analysis
                    import tempfile
                    _, img_path = tempfile.mkstemp(suffix=".png")
                    await details_page.screenshot(path=img_path)
                    from PIL import Image
                    doc_image = Image.open(img_path)

                    self.logger.info("Generating Document Blueprints...")
                    document_blueprints_raw = await self.strategist_agent.generate_document_blueprints(doc_html, doc_image)
                    
                    from .data_contracts import DocumentPageProcessing
                    document_blueprints = DocumentPageProcessing.model_validate(document_blueprints_raw)
                    self.logger.info("Document Blueprints successfully validated.")
                    
                else:
                    self.logger.warning("No Documents Tab found. Skipping Phase 5.")

            finally:
                await details_page.close()

            # --- FINAL SYNTHESIS ---
            final_instructions = list(ref_search_plan.instructions)
            
            # Step 1: Click the Result (Link) - ONLY IF NOT IMPLICIT
            if not is_implicit_view:
                result_link_selector = results_blueprints['listing_link_blueprint']['selector']
                
                final_instructions.append(Instruction(
                    command="WAIT_FOR_SELECTOR",
                    params={"selector": result_link_selector},
                    description="Wait for search results."
                ))
                final_instructions.append(Instruction(
                    command="CLICK_ELEMENT",
                    params={"selector": result_link_selector},
                    description="Click the search result to view details."
                ))
            else:
                 # If implicit, we just verify the details container is visible.
                 # We use the container found in Phase 3 (which was the modal itself)
                 container_sel = results_blueprints['listing_container_selector']
                 final_instructions.append(Instruction(
                    command="WAIT_FOR_SELECTOR",
                    params={"selector": container_sel},
                    description="Wait for Application Details modal/view to appear."
                ))

            
            # Step 2: Navigate to Documents (if applicable)
            if docs_tab_selector:
                final_instructions.append(Instruction(
                    command="WAIT_FOR_SELECTOR",
                    params={"selector": docs_tab_selector},
                    description="Wait for detail page."
                ))
                final_instructions.append(Instruction(
                    command="CLICK_ELEMENT",
                    params={"selector": docs_tab_selector},
                    description="Open Documents tab."
                ))
                final_instructions.append(Instruction(
                    command="WAIT_FOR_SELECTOR",
                    params={"selector": document_blueprints.document_container_selector if document_blueprints else "table"},
                    description="Wait for documents table."
                ))

            # Data Contract Sanitization: Ensure blueprints satisfy strict Pydantic models.
            # In Implicit/SPA mode, the AI might return None for specific blueprints.
            if results_blueprints.get("listing_link_blueprint") is None:
                self.logger.info("Sanitizing results_blueprints: Injecting dummy link blueprint for Pydantic compliance.")
                results_blueprints["listing_link_blueprint"] = {
                    "selector": "body",
                    "extraction_method": "ATTRIBUTE",
                    "attribute_name": "id"
                }

            if results_blueprints.get("listing_address_blueprint") is None:
                self.logger.info("Sanitizing results_blueprints: Injecting dummy address blueprint for Pydantic compliance.")
                results_blueprints["listing_address_blueprint"] = {
                    "selector": "body",
                    "extraction_method": "TEXT"
                }

            self.logger.info(f"Finalizing Plan with {len(final_instructions)} steps.")

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
        """Helper to extract the best SEARCH_PAGE url."""
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
             self.logger.info(f"No explicit SEARCH_PAGE found. Promoting top candidate: {best_url}")
             return best_url
             
        return None

    def _combine_probe_and_adaptive(
        self,
        probe_plan: 'DraftPlan',
        adaptive_results_components: dict,
        adaptive_final_instruction: list[dict],
        detail_page_processing: dict
    ) -> 'DraftPlan':
        """Merges the results of the probe and adaptive stages into a single, complete DraftPlan."""
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
        self.logger.info(f"[{domain}] A failure occurred. Upstream data may require manual review.")
        pass
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\settings.py
```
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


ALL_API_KEYS = [
    "AIzaSyB4_Uj8I5g5ve0bAxg3u-p1EFRR4NHM8lg",#1
    "AIzaSyCynGCEe6FgGPGc-V59t5d16FYW_OXTXOM",#2
    "AIzaSyB-qS9-3_Gv7Em1HK_0DzivW0KalQVWVpY",#3
    "AIzaSyCT6U2NITZLc9C3QTHtmsoF61-npaRZHgM",#4
    "AIzaSyCVKKQQXAxLFhfNZQ1ou7hVceL-j79UbWQ",#5
    "AIzaSyCN0l94G2uU7gYit1CqWmqAGvA0gZFjbn4",#6
    "AIzaSyCK2vOVdpJdJrMwvZOFqfHvdZVClWMPatk",#7
    "AIzaSyDv0TB8uSWauL6KC7FxHCpZ9OmVe6TeaFo",#8
    "AIzaSyD_s_8-hzlpLdcvF-qutF1NmDXdT3dO4jQ",#9
    "AIzaSyBhFJ03fucN51Qt-atcvj4h5NyD9Yww02A",#10
    "AIzaSyDJKjLj5-kxq4hptnwHAEAHheV0quJtofE",#11
    "AIzaSyCV65RciECDCb782HtzjzORMCri6nNspTM",#12
    "AIzaSyCEhAuTtJUafMF4GL1oO60vUo6_Qg8udlw",#13
    "AIzaSyBeVj_ez0FnIHWBp1fw2dKpuqRA1FFLDLU",#14
    "AIzaSyAP2XC4jlyWw63HaidD0uHcrjCE5GaNOYM",#15
    "AIzaSyDfziLuFsVtzDGv4DLLVOHxg21Ph_Vhe2Y",#16
    "AIzaSyABDCR8SdJLJvDdXVmyD1oyBNFN4qcYFaM",#17
    "AIzaSyCRWrUekD224t_5ojs8mtY4Gf_c268xFXM",#18
    "AIzaSyDMWs4cQYg0gXWMT5ckb9MJMhv-bQBOmqY",#19
    "AIzaSyAV9H0QeIPhSrEq_1L-gsXvga8ulFeI8ZA",#20
    "AIzaSyCRLHACrFTh-EbUTKM79AiXzigkth9ngHY",#21
    "AIzaSyBb3iLlWa7FSR-QACyuXPyQS18p8lQ057k",#22
    "AIzaSyCMX2yeJluwMtJmMtiWcOgzIRraTxwcjW0",#23
    "AIzaSyAtvCglgtp8vn05qTDx_wzoSVPRkJ4C5I8",#24
    "AIzaSyBF5KF6_fJcMX2USFNtz_vpuwskKpNnnCM"#25
]
class Settings(BaseSettings):
    """
    Centralized, environment-aware application configuration.
    """
    gemini_api_keys: list[str] = Field(default=ALL_API_KEYS)
    log_level: str = Field('INFO', alias='LOG_LEVEL')
    validation_timeout_ms: int = Field(30000, alias='VALIDATION_TIMEOUT_MS')
    validation_retries: int = Field(2, alias='VALIDATION_RETRIES') # Default to 2 retries (3 total attempts)
    max_correction_attempts: int = Field(3, alias='MAX_CORRECTION_ATTEMPTS') # Default to 3 correction attempts
    
    # Fallback chain: 2.5 Flash (Fastest/New) -> 1.5 Flash (Reliable) -> 1.5 Pro (Heavy Duty)
    gemini_models: list[str] = Field(
        default=["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite"],
        alias='GEMINI_MODELS'
    )

    @field_validator('gemini_api_keys', mode='before')
    @classmethod
    def _parse_comma_separated_str(cls, v: str) -> list[str]:
        if isinstance(v, str):
            keys = [key.strip() for key in v.split(',') if key.strip()]
            if not keys:
                raise ValueError("GEMINI_API_KEYS environment variable is empty or contains only commas.")
            return keys
        return v

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = True

# This line acts as a fail-fast gate. If config is invalid, the app will
# crash on import, which is the desired behavior.
settings = Settings()
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\agents\strategist_agent.py
```
import json
import logging
import re
from pathlib import Path
from typing import final, Any, List
from urllib.parse import urlparse

import jinja2
from PIL import Image
from jsonschema import validate, ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from ..data_contracts import TriageResult, DraftPlan, ValidationReport, ProbeResult, DetailPageProcessing
from ..exceptions import StrategyError
from ..command_schema import COMMAND_SCHEMA
from ..services.gemini_service import GeminiService

# --- JUDGEMENT DIRECTIVE: These prompts are the final, stable versions. ---

_PROBE_PLAN_PROMPT_TEMPLATE = """
You are an expert web automation strategist. Your mission is to create a "Universal Path" to find a Planning Application on a Council website given its Reference ID.

### THE LAW (ABSOLUTE & NON-NEGOTIABLE) ###
You must adhere to every rule in this section. Your entire response will be programmatically validated against these rules.

#### Rule 1: The Law of Non-Repetition (NEW AND CRITICAL)
1.1. You have been shown the "PREVIOUS FAILED PLAN".
1.2. You **MUST NOT** reuse the failed selectors or submission strategies from that plan.
1.3. If the previous plan failed while trying to `CLICK_ELEMENT`, you **MUST** now formulate a plan that uses a different strategy, such as `PRESS_KEY` with the "Enter" key on the search input. Do not try to find another button to click.

#### Rule 2: The Law of Pre-emption (The Modal Defense)
2.1. **Universal Applicability:** This rule applies after **EVERY** `GOTO_URL` or `WAIT_FOR_NAVIGATION` command.
2.2. **Visual Check:** Look at the screenshot. Is there a "Cookie Notice", "Terms of Use", or "Welcome" modal blocking the main content?
2.3. **The Mandate:** You **MUST** dismiss these overlays before attempting to `FILL_INPUT`.
2.4. **Execution:** Insert a `CLICK_ELEMENT` command targeting the "Accept", "Agree", "Continue", or "Close" button.
2.5. **Robust Selectors:**
    - Standard: `button:has-text("Accept")`, `button:has-text("Agree")`, `button:has-text("Continue")`
    - NI Planning Portal Specific: `button.btn-primary:has-text("Accept")`, `button:has-text("Next")`, `button:has-text("Start")`, `button:has-text("Necessary")`
    - Cookiebot (Critical): `#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll`, `#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowVariant`, `a#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll`
2.6. **CRITICAL:** Do NOT use `WAIT_FOR_SELECTOR` for these banners. Just `CLICK_ELEMENT`.

#### Rule 3: The Law of Disclosure (Accordion Logic)
3.1. Look at the Search Form. Is the input field currently visible?
3.2. If the input is hidden inside an accordion, collapsed section, or tab (indicated by text like "Search by Reference", "Ref Number", or icons like `+`, `v`, or arrows), you **MUST** click that header first.
3.3. **Sequence:** `CLICK_ELEMENT` (Header) -> `WAIT_FOR_SELECTOR` (Input visible) -> `FILL_INPUT`.
3.4. **Amber Valley/Northgate Specific:** Look for "Ref number" or "Search options" text that acts as a toggle.

#### Rule 4: Mission Directives (Formerly Rule 2)
Each command MUST have a `params` object with these exact keys. There are no exceptions.
- `GOTO_URL`: `params` MUST contain `url`.
- `FILL_INPUT`: `params` MUST contain `selector` and `value`. 
    - **CRITICAL MANDATE:** The `value` for the primary search input (e.g., location, address, postcode) **MUST ALWAYS AND ONLY BE** the literal string `{{TARGET_ADDRESS}}`.
    - **This is a template placeholder for the next robot, not a value to be replaced by you.** Your final JSON output must contain this exact placeholder string.
    - **A CRITICAL FAILURE IS to use the example address '{{target_address}}' in your output.** You MUST use the placeholder `{{TARGET_ADDRESS}}`.
- `CLICK_ELEMENT`: `params` MUST contain `selector`.
- `CHECK_OPTION`: `params` MUST contain `selector`.
- `PRESS_KEY`: `params` MUST contain `selector` and `key`.
- `WAIT_FOR_SELECTOR`: `params` MUST contain `selector`. Optional: `state` ("visible", "hidden", "attached", "detached"). Use `state: "detached"` to wait for cookie banners to disappear.
- `WAIT_FOR_NAVIGATION`: `params` MUST be an empty object (`{}`).

#### Rule 5: Command Jurisdiction (The Pantheon Accords) (Formerly Rule 3)
- **`CHECK_OPTION`** is **EXCLUSIVELY** for `<input type="radio">` and `<input type="checkbox">`.
- **`CLICK_ELEMENT`** is for all other clickable elements.
- **LAW OF EXCLUSIVITY**: You **MUST NOT** use `CLICK_ELEMENT` on a radio button or checkbox.

#### Rule 6: Strategic Efficiency (Formerly Rule 4)
- Action commands have built-in waits. **DO NOT** precede them with a redundant `WAIT_FOR_SELECTOR`.

#### Rule 7: The Law of Literal Selectors (NEW)
- All CSS selectors MUST be syntactically valid.
- You **MUST NOT** use regular expressions inside selectors (e.g., `:has-text(/Accept/i)` is ILLEGAL).
- Use standard CSS and Playwright `:has-text("text")` pseudo-classes ONLY.

#### Rule 8: The Law of Implicit Submission (NEW)
- Web forms can be submitted in two ways. You must choose the correct one based on the visual evidence.
- **Explicit Submission:** If a clear, visible "Search", "Apply", or "Submit" button is present, your plan should be `FILL_INPUT` followed by `CLICK_ELEMENT` on that button.
- **Implicit Submission:** If no such button is visible or obvious, your plan **MUST** be `FILL_INPUT` followed by `PRESS_KEY` on the *same input field*, with the `key` parameter set to "Enter". Do not invent a button selector if you cannot see one.
- **Icon Buttons:** If the search button is an icon (e.g., magnifying glass) with no text, verify if it has an `aria-label` or `title` containing "Search". Do NOT rely on `:has-text("Search")` for these. If unsure, prefer Implicit Submission.

#### Rule 9: The Law of Transit (The Intermediary Page Clause)
- **Visual Scan:** Look at the screenshots. Are you actually on a Search Form, or a "Landing Page" describing the service?
- **The Mandate:** If you do not see a text input field for the Reference Number, but you DO see a link or button saying "Search for an application", "View planning applications", or "Public Access", you **MUST** click that link first.
- **Forbidden Act:** Do NOT hallucinate an input field (e.g., `#searchCriteria_application_number`) if the page is just a CMS article with a link to the real system.
- **Sequence:** `CLICK_ELEMENT` (The "Go to Search" link) -> `WAIT_FOR_NAVIGATION` -> `FILL_INPUT` (The actual input on the next page).

#### Rule 10: The Law of Stable Selectors
- You **MUST NOT** use IDs that look generated or dynamic (e.g., `id="c_3919..."`, `id="guid-..."`, `id="ember..."`).
- **Detection:** If an ID contains a long string of numbers or random characters, it is likely dynamic.
- **Alternative:** Use stable attributes like `name`, `placeholder`, `aria-label`, or combine `form` classes with tags (e.g., `form.search-form input`).

#### Rule 11: The Law of Negative Constraints (NEW)
11.1. You **MUST NOT** interact with "Site Search", "General Search", or "Website Search" inputs.
11.2. **Golden Rule:** If the input's placeholder or label says "Search this site", "Search website", or just "Search...", IT IS WRONG.
11.3. **DOM Location Constraint:** You **MUST NOT** select an input located inside a `<header>`, `#top`, or `.site-header` container. You **MUST** select the input inside the `<main>`, `#content`, or `article` area.
11.4. **Visual Constraint:** If the input is in the top-right corner of the page, it is almost certainly the Site Search. IGNORE IT.
11.5. You are looking for "Planning Search", "Application Search", or inputs specifically labelled "Reference Number", "Application Number", "Keyword".

#### Rule 12: The Law of Wrong Neighborhoods (The Legacy System Clause)
12.1. **Symptom:** The failure report indicates "SEARCH_OUTCOME_MISMATCH" or "No Results Found".
12.2. **Analysis:** Read the text *around* the warning box. Does it say "For applications before 2025..." or "Visit our existing system"?
12.3. **The Fix:** Your corrected plan **MUST** click the link associated with that warning (e.g., "existing Aylesbury Vale system", "view legacy applications").
12.4. **Application:** Do NOT retry the search term in the same box. You are in the wrong database. Move to the correct one.

### OUTPUT FORMAT (MANDATORY) ###
You **MUST** respond with a single JSON object with THREE keys: `search_term`, `instructions`, and `is_navigation_only`.

#### Rule 13: The Law of Incremental Navigation (NEW)
13.1. **Situation:** You are on a "Landing Page", "Disclaimer", or "Menu" and you can see a link to the search (e.g., "Search Planning Applications"), but NOT the input field itself.
13.2. **The Mandate:** You cannot search yet. You must navigate closer.
13.3. **Output:** Set `"is_navigation_only": true` in your JSON response. The `instructions` should strictly navigate to the next page.
13.4. **Situation:** You see the "Reference Number" or "Keyword" input field.
13.5. **Output:** Set `"is_navigation_only": false`. This is the final search plan.

### EXAMPLE OF A PERFECT MODERN RESPONSE ###
```json
{
  "search_term": "Newcastle Upon Tyne",
  "is_navigation_only": false,
  "instructions": [
    {
      "command": "GOTO_URL",
      "params": { "url": "https://publicaccess.westberks.gov.uk/" },
      "description": "Navigate to the publicaccess.westberks.gov.uk page."
    },
    {
      "command": "FILL_INPUT",
      "params": { "selector": "#ps--searchbox", "value": "{{TARGET_ADDRESS}}" },
      "description": "Fill the search input with the target address."
    },
    {
      "command": "CLICK_ELEMENT",
      "params": { "selector": "button.btn--norm:has-text('Apply')" },
      "description": "Submit the search by clicking the apply button."
    },
    {
      "command": "WAIT_FOR_SELECTOR",
      "params": { "selector": "div.mw-prop-box" },
      "description": "Wait for the property listing containers to update on the page, confirming search success."
    }
  ]
}


1. **Pre-emption:** Look for "Terms and Conditions" or "Copyright" pages. You MUST click "Accept" or "Agree" if they exist.
2. **Input:** The `value` for the search input (Reference Number/Application ID) **MUST ALWAYS AND ONLY BE** the literal string `{{REFERENCE_ID}}`.
3. **Outcome (If is_navigation_only=false):** The final instruction MUST be `WAIT_FOR_SELECTOR` to prove the search worked.
4. **The Omnibus Selector:** The `selector` for your final `WAIT_FOR_SELECTOR` command **MUST BE EXACTLY** this string (copy-paste it): `#searchresults, #simpleSearchResults, .search-results, h1:has-text("Summary"), h1:has-text("Details"), #tab_summary, #tab_details, .results-container, div[class*="Result"], [data-testid*="result"], div[role="list"]`

### OUTPUT FORMAT ###
Respond with a JSON object: `{"search_term": "Ref Search", "is_navigation_only": false, "instructions": [...]}`.
```

--- CONTEXTUAL HTML ---
{{ homepage_html }}
{% if searchpage_html %}
--- SEARCH PAGE HTML ---
{{ searchpage_html }}
{% endif %}
"""

_NAVIGATION_CORRECTION_PROMPT = """
You are an expert web automation strategist. The navigation plan you previously created has FAILED. Your task is to analyze the failure and create a new, corrected `instructions` array that follows THE LAW.

### PREVIOUS FAILED PLAN (DO NOT REPEAT) ###
```json
{{ report.original_plan.instructions | tojson(indent=2) }}
```

### FAILURE ANALYSIS ###
- **Error**: `{{ report.failure_reason }}`
{% if report.failure_analysis_notes %}
- **Architect's Analysis**: `{{ report.failure_analysis_notes }}`
{% endif %}

### VISUAL ANALYSIS (GROUND TRUTH) ###
A screenshot of the webpage at the exact moment of failure is attached. This image is the absolute ground truth.

### THE LAW (ABSOLUTE & NON-NEGOTIABLE) ###
(The law is identical to the generation phase. All rules apply.)

#### Rule 1: The Law of Non-Repetition (NEW AND CRITICAL)
1.1. You have been shown the "PREVIOUS FAILED PLAN".
1.2. You **MUST NOT** reuse the failed selectors or submission strategies from that plan.
1.3. If the previous plan failed while trying to `CLICK_ELEMENT`, you **MUST** now formulate a plan that uses a different strategy, such as `PRESS_KEY` with the "Enter" key on the search input. Do not try to find another button to click.

#### Rule 2: The Law of Pre-emption (The Modal Defense)
2.1. **Universal Applicability:** This rule applies after **EVERY** `GOTO_URL` or `WAIT_FOR_NAVIGATION` command.
2.2. **Cross-Domain Reset:** If the plan redirects to a new domain (e.g. Council Site -> National Portal), previous cookie clicks are irrelevant. You MUST check for a new banner.
2.3. **Visual Check:** Look at the "Failure Context" screenshot. Is there a modal (like "Cookie notice") blocking the input?
2.4. **The Mandate:** Insert a `CLICK_ELEMENT` command targeting the "Accept", "Agree", or "Continue" button BEFORE retrying the failed input.
2.5. **Robust Selectors:**
    - Standard: `button:has-text("Accept")`, `button:has-text("Agree")`, `button:has-text("Continue")`
    - NI Planning Portal Specific: `button.btn-primary:has-text("Accept")`, `button:has-text("Next")`, `button:has-text("Start")`, `button:has-text("Necessary")`
    - Cookiebot: `#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll`

#### Rule 3: The Law of Disclosure (Accordion/Toggle Logic)
3.1. **Visual Scan:** Look at the search form area. Is the input field for the Reference ID visible?
3.2. **The "Plain Text" Trap:** Many councils (e.g., Amber Valley, Northgate systems) hide inputs behind text labels that don't look like buttons.
    - Look for text like: "Ref number", "Search by reference", "Search Options".
    - Look for icons: `v`, `+`, `▼`, or arrows next to text.
3.3. **The Mandate:** If the input is not clearly visible, you **MUST** click the text label/header first.
3.4. **Execution Sequence:** 
    1. `CLICK_ELEMENT` (The Header/Text Label) e.g., `div:has-text("Ref number")`, `a:has-text("Search options")`.
    2. `WAIT_FOR_SELECTOR` (The Input Field) - Verify it appeared.
    3. `FILL_INPUT` (The Input Field).

#### Rule 4: Mission Directives (Formerly Rule 2)
4.1. The plan MUST focus on the "FOR SALE" user journey.
4.2. The final instruction **MUST** be `WAIT_FOR_SELECTOR` to prove the search worked.
4.3. **Hybrid Outcome Strategy:** The search might yield a list of results OR redirect directly to a single Application Details page. Your final selector MUST account for both possibilities.
4.4. **The Omnibus Selector:** The `selector` for your final `WAIT_FOR_SELECTOR` command **MUST BE EXACTLY** this string (copy-paste it): `#searchresults, #simpleSearchResults, .search-results, h1:has-text("Summary"), h1:has-text("Details"), #tab_summary, #tab_details, .results-container, div[class*="Result"], [data-testid*="result"], div[role="list"]`


#### Rule 5: Command Parameter Manifest (Formerly Rule 3)
- `GOTO_URL`: `params` MUST contain `url`.
- `FILL_INPUT`: `params` MUST contain `selector` and `value`. The value MUST be "{{REFERENCE_ID}}".
- `CLICK_ELEMENT`: `params` MUST contain `selector`.
- `CHECK_OPTION`: `params` MUST contain `selector`.
- `PRESS_KEY`: `params` MUST contain `selector` and `key`.
- `WAIT_FOR_SELECTOR`: `params` MUST contain `selector`. Optional: `state` ("visible", "hidden", "attached", "detached"). Use `state: "detached"` to wait for obstructions to clear.
- `WAIT_FOR_NAVIGATION`: `params` MUST be an empty object (`{}`).

#### Rule 6: Command Jurisdiction (The Pantheon Accords) (Formerly Rule 4)
- **`CHECK_OPTION`** is **EXCLUSIVELY** for `<input type="radio">` and `<input type="checkbox">`.
- **`CLICK_ELEMENT`** is for all other clickable elements.
- **LAW OF EXCLUSIVITY**: You **MUST NOT** use `CLICK_ELEMENT` on a radio button or checkbox.

#### Rule 7: The Law of Literal Selectors (Formerly Rule 5)
7.1. All CSS selectors MUST be syntactically valid.
7.2. You **MUST NOT** use regular expressions inside selectors (e.g., `:has-text(/Accept/i)` is ILLEGAL).
7.3. Use standard CSS and Playwright `:has-text("text")` pseudo-classes ONLY.

#### Rule 8: The Law of Implicit Submission (Formerly Rule 6)
8.1. Web forms can be submitted in two ways. You must choose the correct one based on the visual evidence.
8.2. **Explicit Submission:** If a clear, visible "Search", "Apply", or "Submit" button is present, your plan should be `FILL_INPUT` followed by `CLICK_ELEMENT` on that button.
8.3. **Implicit Submission:** If no such button is visible or obvious, your plan **MUST** be `FILL_INPUT` followed by `PRESS_KEY` on the *same input field*, with the `key` parameter set to "Enter". Do not invent a button selector if you cannot see one.
8.4. **Icon Buttons:** If the search button is an icon (e.g., magnifying glass) with no text, verify if it has an `aria-label` or `title` containing "Search". Do NOT rely on `:has-text("Search")` for these. If unsure, prefer Implicit Submission.

#### Rule 9: The Law of Transit (The Intermediary Page Clause)
9.1. **Visual Scan:** Look at the screenshot. Are you actually on a Search Form, or a "Landing Page"?
9.2. **The Mandate:** If the previous error was a TIMEOUT looking for an input, and you see a link like "Search for an application" or "Public Access", you **MUST** click that link.
9.3. **Forbidden Act:** Do NOT retry the same input selector if it failed to find anything. The page is likely an intermediary. Navigate through it.

#### Rule 10: The Law of Stable Selectors
- You **MUST NOT** use IDs that look generated or dynamic (e.g., `id="c_3919..."`, `id="guid-..."`, `id="ember..."`).
- **Detection:** If an ID contains a long string of numbers or random characters, it is likely dynamic.
- **Alternative:** Use stable attributes like `name`, `placeholder`, `aria-label`, or combine `form` classes with tags (e.g., `form.search-form input`).

#### Rule 11: The Law of the "Other" Search (The Dual-Portal Trap)
11.1. **Symptom:** The failure report indicates "SEARCH_OUTCOME_MISMATCH" (No Results), BUT the page text contains phrases like "Search Planning Applications", "View Planning Applications", or "Public Access".
11.2. **Analysis:** The council has TWO systems. The main site has a "Site Search" (which you tried) and a hidden "Planning Search".
11.3. **The Fix:** Your corrected plan **MUST** click the link/button that explicitly says "Planning Applications", "View Planning Applications", or "Public Access".
11.4. **Forbidden Act:** Do NOT retry the search term in the main search box. You are in the wrong database.

#### Rule 12: The Law of Wrong Neighborhoods (The Legacy System Clause)
12.1. **Symptom:** The failure report indicates "SEARCH_OUTCOME_MISMATCH" or "No Results Found".
12.2. **Analysis:** Read the text *around* the warning box. Does it say "For applications before 2025..." or "Visit our existing system"?
12.3. **The Fix:** Your corrected plan **MUST** click the link associated with that warning (e.g., "existing Aylesbury Vale system", "view legacy applications").
12.4. **Application:** Do NOT retry the search term in the same box. You are in the wrong database. Move to the correct one.

### YOUR TASK ###
Analyze the failure, the screenshot, the Architect's analysis, and the HTML to create a corrected plan. If your previous attempt to CLICK a button failed, STRONGLY CONSIDER switching to an implicit submission strategy using PRESS_KEY with "Enter".
Respond ONLY with a single JSON object containing one key: `instructions`.

--- HTML AT POINT OF FAILURE ---
{{ report.failure_context_html }}
"""


_ADAPTIVE_RESULTS_PROMPT_TEMPLATE = """
You are an expert web scraping analyst for UK Planning Portals (Idox, Northgate, Agile). 
Your task is to analyze the **Search Results** page to create blueprints for extracting the list of Planning Applications.

### VISUAL CUES
Look for a list or table containing:
1.  **Reference Numbers** (e.g., "22/01234/FUL", "121281")
2.  **Addresses**
3.  **Descriptions** (e.g., "Erection of extension...")

### MISSION
Analyze the provided HTML and Screenshot to produce:
1.  A `results_page_selectors` blueprint to extract data from the results list.
2.  A `final_wait_instruction` to confirm the results have loaded.

### DATA MAPPING (CRITICAL)
- `listing_container_selector`: The CSS selector for the repeating item. 
    - **IDOX Hint:** Often `#searchresults > li` or `.searchresult`.
    - **Northgate Hint:** Often `table.data > tr`.
- `listing_link_blueprint`: The selector for the `<a>` tag linking to the details.
- `listing_address_blueprint`: The selector for the text containing the **Reference Number**.
- **SYNTAX RULE:** You **MUST NOT** use `:contains()`. It is not valid CSS. Use `:has-text("...")` instead.

### OMNIBUS CLAUSE
If you cannot find a list of results, but the page looks like a **Single Application Summary** (e.g., it has tabs like "Details", "Comments"), you MUST return a valid selector for the "Summary" tab or the main heading as the `listing_container_selector` so the pipeline knows we are done.

### OUTPUT SCHEMA
Respond with a single JSON object.

Example:
```json
{
  "final_wait_instruction": {
    "command": "WAIT_FOR_SELECTOR",
    "params": { "selector": "#searchresults" },
    "description": "Wait for results."
  },
  "results_page_selectors": {
    "listing_container_selector": "li.searchresult",
    "listing_link_blueprint": {
      "selector": "a",
      "extraction_method": "ATTRIBUTE",
      "attribute_name": "href"
    },
    "listing_address_blueprint": {
      "selector": "p.spec",
      "extraction_method": "TEXT"
    },
    "pagination_blueprint": null
  }
}
```

--- LIVE RESULTS PAGE HTML ---
{{ captured_html }}
"""

_DETAIL_PAGE_PROMPT_TEMPLATE = """
You are an expert web scraping analyst for UK Planning Portals. Your task is to analyze the **Application Details** page.

### MISSION ###
1.  **Tab/Accordion Navigation:** Identify the selectors for the "Documents", "Comments", and "Constraints" sections.
    - **CRITICAL FOR ACCORDIONS:** If the data is inside a collapsible section (common in Idox/Northgate), you **MUST** select the **Visible Trigger** (Header, Link, or Button) that expands the section.
    - **NEVER** select the hidden content container itself (e.g., `div.collapse`, `div.panel-body`).
    - **CORRECT:** `a[href='#collapse-documents']`, `h3.panel-title`, `li.active > a`.
    - **INCORRECT:** `#collapse-documents` (This is the hidden div).

2.  **Key Data:** Create extraction blueprints for:
    - `application_status`: The status text (e.g., "Decided", "Pending").
    - `decision`: The decision value (e.g., "Approved", "Refused").
    - `proposal`: The description of works.
    - `address`: The site address.

### OUTPUT SCHEMA ###
Respond with a JSON object. Keys MUST be snake_case.

Example:
```json
{
  "tabs": {
    "documents_tab_selector": "a[href='#documents']",
    "comments_tab_selector": "#tab_consultations",
    "constraints_tab_selector": "#tab_constraints"
  },

  "data_blueprints": {
    "application_status": { "selector": "span.status", "extraction_method": "TEXT" },
    "decision": { "selector": "#decision_text", "extraction_method": "TEXT" },
    "proposal": { "selector": ".description", "extraction_method": "TEXT" }
  }
}
```

--- HTML OF THE APPLICATION PAGE ---
{{ html_content }}
"""


_RESULTS_CORRECTION_PROMPT = """
You are an expert web scraping analyst. The `results_page_processing` blueprint you created previously FAILED VALIDATION. It could not correctly identify elements on the results page.

### MISSION ###
- Analyze the provided screenshot and HTML of the live property search results page.
- Create a new, corrected `results_page_processing` JSON object.

### BLUEPRINT SCHEMA ###
```json
{
  "listing_container_selector": "div.property-card-container",
  "listing_link_blueprint": {
    "selector": "a.property-link",
    "extraction_method": "ATTRIBUTE",
    "attribute_name": "href"
  },
  "listing_address_blueprint": {
    "selector": "h2.property-address",
    "extraction_method": "TEXT"
  }
}
```

### YOUR TASK ###
Respond ONLY with the corrected `results_page_processing` JSON object.

--- HTML OF RESULTS PAGE (AT POINT OF FAILURE) ---
{{ report.failure_context_html }}
"""

_DETAIL_CORRECTION_PROMPT = """
You are an expert web data extraction analyst. The `detail_page_processing` blueprints you created FAILED VALIDATION on a live property detail page.

### MISSION ###
- Analyze the provided screenshot and HTML of the property detail page.
- DISCOVER all available, high-value data points (e.g., Price, Address, Bedrooms, etc.).
- Generate a new, complete, and robust `detail_page_processing` JSON object.
- The keys of the object MUST be snake_cased. The values MUST be valid `ExtractionBlueprint` objects.

### BLUEPRINT SCHEMA ###
```json
{
  "selector": "css-selector-for-the-element",
  "extraction_method": "TEXT",
  "attribute_name": null
}
```

### YOUR TASK & OUTPUT FORMAT ###
Respond ONLY with a single JSON object that is the dictionary of the blueprints themselves. DO NOT wrap it in a parent key.

**CORRECT Response Format:**
```json
{
  "address": { "...blueprint..." },
  "price": { "...blueprint..." }
}
```

**INCORRECT Response Format (DO NOT DO THIS):**
```json
{
  "detail_page_processing": {
    "address": { "...blueprint..." },
    "price": { "...blueprint..." }
  }
}
```

--- HTML OF DETAIL PAGE (AT POINT OF FAILURE) ---
{{ report.failure_context_html }}
"""
@final
class StrategistAgent:
    """Builds a draft execution plan via an empirical, two-stage process."""

    def __init__(self, gemini_service: GeminiService):
        self.gemini_service = gemini_service
        self.jinja_env = jinja2.Environment(autoescape=True)
        self.jinja_env.filters['tojson'] = lambda data, indent=2: json.dumps(data, indent=indent)
        
        # Templates
        self.probe_prompt_template = self.jinja_env.from_string(_PROBE_PLAN_PROMPT_TEMPLATE)
        self.adaptive_prompt_template = self.jinja_env.from_string(_ADAPTIVE_RESULTS_PROMPT_TEMPLATE)
        self.detail_prompt_template = self.jinja_env.from_string(_DETAIL_PAGE_PROMPT_TEMPLATE)
        self.navigation_correction_template = self.jinja_env.from_string(_NAVIGATION_CORRECTION_PROMPT)
        self.results_correction_template = self.jinja_env.from_string(_RESULTS_CORRECTION_PROMPT)
        self.detail_correction_template = self.jinja_env.from_string(_DETAIL_CORRECTION_PROMPT)
        
        # --- DOCUMENT HARVESTING PROMPT ---
        self._document_scraping_template = self.jinja_env.from_string("""
        You are an expert Data Engineer specializing in tabular data extraction. 
        You are looking at the **Documents Tab** of a Planning Application.

        ### MISSION
        Create a precise JSON blueprint to extract every document in the list.

        ### VISUAL ANALYSIS
        1. **Container:** Identify the repeating element for each document (e.g., `tr`, `div.document-row`).
        2. **Link:** Find the specific `<a>` tag that opens/downloads the file.
        3. **Metadata:** Find the Date and Document Type (e.g., "Decision Notice", "Site Plan").
        4. **Pagination:** Look for a "Next", ">", or "2" button if the list is long. A 'next' button is preferred to a number, as it is more universal.

        ### OUTPUT SCHEMA (STRICT JSON)
        ```json
        {
          "document_container_selector": "table#docs > tbody > tr",
          "document_link_blueprint": { "selector": "a.download", "extraction_method": "ATTRIBUTE", "attribute_name": "href" },
          "document_date_blueprint": { "selector": "td:nth-child(4)", "extraction_method": "TEXT" },
          "document_type_blueprint": { "selector": "td:nth-child(2)", "extraction_method": "TEXT" },
          "pagination_blueprint": { "next_page_selector": "a.next-page" } 
        }
        ```
        *Note: Set `pagination_blueprint` to null if no pagination controls are visible.*

        --- DOCUMENT LIST HTML ---
        {{ html_content }}
        """)

        # New Template for Global Search Fallback
        self._global_search_template = self.jinja_env.from_string("""
        You are a Web Navigation Specialist. You are currently on the Homepage of a Local Council.

        ### THE MISSION
        1. Locate the main site search input (often at the top right, or behind a magnifying glass icon).
        2. Create a plan to type "planning application search" into this box.
        3. Submit the search (Click Search or Press Enter).

        ### THE LAW (ABSOLUTE)
        1. **Cookie Pre-emption:** If there is a cookie banner, dismiss it first.
        2. **Input Value:** The value for `FILL_INPUT` MUST be exactly "planning application search".
        3. **Outcome:** The final command MUST be `WAIT_FOR_NAVIGATION` (as search usually reloads the page).
        4. **Selector Syntax:** You **MUST NOT** use `:contains()`. It is illegal. Use `:has-text("...")` instead.
        5. **The Law of Stable Selectors:** You **MUST NOT** use IDs that look generated or dynamic (e.g., `id="c_3919..."`, `id="guid-..."`, `id="ember..."`). Instead, use stable attributes like `name`, `placeholder`, `aria-label`, or combine `form` classes with tags (e.g., `form.search-form input`).
        6. **The Law of Implicit Submission:** If the search button is an icon (e.g., magnifying glass) or ambiguous, your plan **MUST** use `PRESS_KEY` with "Enter" on the input field instead of `CLICK_ELEMENT`. This is much more robust and avoids layout obstruction issues.

        ### COMMAND SCHEMA (STRICT)
        Each instruction must match this format EXACTLY:
        - `GOTO_URL`: `{"command": "GOTO_URL", "params": {"url": "..."}}`
        - `FILL_INPUT`: `{"command": "FILL_INPUT", "params": {"selector": "...", "value": "..."}}`
        - `CLICK_ELEMENT`: `{"command": "CLICK_ELEMENT", "params": {"selector": "..."}}`
        - `PRESS_KEY`: `{"command": "PRESS_KEY", "params": {"selector": "...", "key": "Enter"}}`
        - `WAIT_FOR_SELECTOR`: `{"command": "WAIT_FOR_SELECTOR", "params": {"selector": "...", "state": "visible"}}`
        - `WAIT_FOR_NAVIGATION`: `{"command": "WAIT_FOR_NAVIGATION", "params": {}}`

        ### OUTPUT SCHEMA
        Respond with a JSON object: `{"instructions": [...]}`.

        --- HOMEPAGE HTML ---
        {{ homepage_html }}
        """)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.command_schema_json = json.dumps(COMMAND_SCHEMA, indent=2)

    async def generate_global_search_plan(self, homepage_url: str, homepage_html: str) -> DraftPlan:
        """Generates a plan to use the site's internal search engine."""
        self.logger.info(f"Generating Global Site Search fallback plan for {homepage_url}")
        
        prompt = self._global_search_template.render(homepage_html=homepage_html)
        response_text = await self.gemini_service.generate_content(prompt)
        plan_data = self._parse_llm_json_response(response_text)
        
        if 'instructions' not in plan_data:
             raise StrategyError("AI response for global search is missing 'instructions'.")

        instructions = plan_data['instructions']

        # SYSTEM OVERRIDE: Enforce GOTO_URL as Step 0.
        # The ValidationAgent spawns a fresh page (about:blank). We cannot trust the AI to remember this.
        if not instructions or instructions[0].get("command") != "GOTO_URL":
            self.logger.warning("AI forgot GOTO_URL in global search plan. Injecting system override.")
            instructions.insert(0, {
                "command": "GOTO_URL",
                "params": {"url": homepage_url},
                "description": "Navigate to the homepage (System Injected Override)."
            })
        
        # Verify the URL matches if it WAS provided (edge case check)
        elif instructions[0].get("command") == "GOTO_URL":
             # Force alignment to the trusted artifact URL
             instructions[0]["params"]["url"] = homepage_url

        self._validate_instructions_against_schema(instructions, context="global search generation")

        # Return a DraftPlan (some fields are dummy placeholders as we are in discovery mode)
        return DraftPlan(
            instructions=instructions,
            source_triage_result=None, # Not applicable
            target_address="planning application search",
            target_description="Finding the planning search page"
        )

    async def generate_planning_search_plan(self, search_page_url: str, search_page_html: str, target_ref_id: str) -> DraftPlan:
        """
        Phase 2: Generates the plan to input the specific Reference ID into the identified Search Page.
        """
        self.logger.info(f"Generating Planning Search Plan for {search_page_url}")

        # Reuse the robust probe template, but with specific context
        prompt = self.probe_prompt_template.render(
            domain=urlparse(search_page_url).netloc,
            target_address=target_ref_id, # This fills the prompt's context context
            homepage_html="", # Not needed, we are already deep
            searchpage_html=search_page_html # The critical context
        )

        response_text = await self.gemini_service.generate_content(prompt)
        plan_data = self._parse_llm_json_response(response_text)

        if 'instructions' not in plan_data:
             raise StrategyError("AI response for planning search is missing 'instructions'.")

        instructions = plan_data['instructions']
        is_navigation_only = plan_data.get('is_navigation_only', False)
        
        # ENFORCE PLACEHOLDER LOGIC (Only if not purely navigational)
        # We ensure the AI uses {{REFERENCE_ID}} so the Validator can swap it
        if not is_navigation_only:
            for instr in instructions:
                if instr.get("command") == "FILL_INPUT":
                    instr["params"]["value"] = "{{REFERENCE_ID}}"

        self._validate_instructions_against_schema(instructions, context="planning search generation")

        return DraftPlan(
            instructions=instructions,
            source_triage_result=None, 
            target_address=target_ref_id,
            target_description=f"Search for Planning Ref {target_ref_id}",
            is_navigation_only=is_navigation_only
        )

    def _diagnose_timeout_failure(self, html_content: str) -> tuple[str, str | None]:
        """Inspects HTML content to determine the likely cause of a timeout."""
        # REFINED KEYWORDS: Removed 'privacy', 'cookie', 'agree' to prevent false positives from footers.
        OBSTRUCTION_KEYWORDS = [
            'cookiebot', 'onetrust', 'termly', 'civic-cookie', 'consent-banner', 
            'modal-dialog', 'overlay', 'gdpr-banner', 'cookie-notice'
        ]
        
        html_lower = html_content.lower()
        found_keywords = [kw for kw in OBSTRUCTION_KEYWORDS if kw in html_lower]

        if found_keywords:
            notes = (
                "Analysis: Timeout occurred. The page HTML contains explicit technical indicators "
                f"({', '.join(sorted(list(set(found_keywords))))}) of a Cookie/Consent Management Platform. "
                "The strategist should look for and click an acceptance button (e.g., 'Accept All', 'I Agree')."
            )
            self.logger.warning("Timeout Diagnostics: Probable obstruction detected (High Confidence).")
            return "PROBABLE_OBSTRUCTION", notes
        else:
            self.logger.info("Timeout Diagnostics: No obvious signs of obstruction found.")
            return "TIMEOUT", None

            
    def _validate_instructions_against_schema(self, instructions: list[dict[str, Any]], context: str):
        """
        Validates a list of instructions against the canonical COMMAND_SCHEMA.
        Raises StrategyError on failure.
        
        Args:
            instructions: The list of instruction dictionaries to validate.
            context: A string describing the operation (e.g., "probe plan generation") for clear error logging.
        """
        try:
            validate(instance=instructions, schema=COMMAND_SCHEMA)
            self.logger.info(f"Plan for '{context}' passed canonical schema validation.")
        except JsonSchemaValidationError as e:
            self.logger.error(f"CRITICAL: Generated plan for '{context}' failed schema validation. Error: {e.message}")
            # This is a non-recoverable failure of the AI's core contract.
            raise StrategyError(f"Generated plan for '{context}' is structurally invalid: {e.message}")


    async def generate_probe_plan(self, triage_result: TriageResult, target_address: str, target_description: str) -> DraftPlan:
        self.logger.info("Generating probe plan.")
        cat_to_url = {v: k for k, v in triage_result.full_classification.items()}
        homepage_url = cat_to_url.get("HOMEPAGE")
        searchpage_url = cat_to_url.get("SEARCH_PAGE")

        if not homepage_url:
            raise StrategyError("No 'HOMEPAGE' found in TriageResult to begin probe.")

        homepage_html = await self._get_contextual_html(homepage_url, triage_result)
        searchpage_html = await self._get_contextual_html(searchpage_url, triage_result) if searchpage_url else None

        prompt = self.probe_prompt_template.render(
            domain=triage_result.domain,
            target_address=target_address,
            homepage_html=homepage_html,
            searchpage_html=searchpage_html
        )
        response_text = await self.gemini_service.generate_content(prompt)
        plan_data = self._parse_llm_json_response(response_text)

        if 'instructions' not in plan_data:
             raise StrategyError("AI response for probe plan is missing required key: 'instructions'.")

        instructions = plan_data['instructions']

        # --- PROJECT GENESIS DIRECTIVE: The Programmatic Judge ---
        # We can no longer trust the AI. We must programmatically enforce the placeholder.
        # This Judge ensures that the plan is reusable and not overfitted to the test data.
        placeholder_found_in_plan = False
        for instruction in instructions:
            if instruction.get("command") == "FILL_INPUT":
                value = instruction.get("params", {}).get("value", "")
                
                # Heuristic: if the AI fills an input with the test reference ID, auto-correct it.
                if value and value.strip() == target_address.strip(): # target_address holds the ref_id in this context
                     self.logger.warning(f"JUDGE: Correcting hardcoded reference ID ('{value}') in FILL_INPUT instruction with placeholder.")
                     instruction["params"]["value"] = "{{REFERENCE_ID}}"

                if instruction["params"].get("value") == "{{REFERENCE_ID}}":
                    placeholder_found_in_plan = True

        # Final check: A valid probe plan MUST have at least one placeholder.
        if not placeholder_found_in_plan:
            raise StrategyError("JUDGE: The generated plan is fatally flawed. It contains no {{REFERENCE_ID}} placeholder and is not reusable.")
        
        self._validate_instructions_against_schema(instructions, context="probe plan generation")

        return DraftPlan(
            instructions=instructions,
            source_triage_result=triage_result,
            target_address=target_address,
            target_description=target_description
        )

    async def correct_navigation_plan(self, report: ValidationReport) -> list[dict]:
        self.logger.info("Correcting navigation plan with context reinforcement.")
        data = await self._run_base_correction(report, self.navigation_correction_template)
        
        if 'instructions' not in data:
            raise StrategyError("AI response for navigation correction is missing 'instructions' key.")
        
        # JUDGEMENT PROTOCOL: Verify before returning.
        instructions = data['instructions']
        self._validate_instructions_against_schema(instructions, context="navigation plan correction")
        return instructions

    # --- Methods below this line are considered stable ---

    async def _run_base_correction(self, report: ValidationReport, template: jinja2.Template) -> dict:
        image = None
        if report.failure_screenshot_path:
            try:
                image = Image.open(report.failure_screenshot_path)
            except Exception as e:
                self.logger.warning(f"Could not load failure screenshot for correction: {e}")
        
        prompt = template.render(report=report)
        response_text = await self.gemini_service.generate_content(prompt, image=image)
        return self._parse_llm_json_response(response_text)

    async def generate_adaptive_components(self, probe_result: ProbeResult) -> tuple[dict, list[dict]]:
        self.logger.info("Generating adaptive components from probe result.")
        if not probe_result.captured_screenshot_path or not probe_result.captured_html:
            raise StrategyError("Probe result is missing screenshot or HTML for adaptive generation.")

        image = Image.open(probe_result.captured_screenshot_path)
        
        prompt = self.adaptive_prompt_template.render(
            captured_html=probe_result.captured_html
        )

        response_text = await self.gemini_service.generate_content(prompt, image=image)
        adaptive_data = self._parse_llm_json_response(response_text)

        if 'final_wait_instruction' not in adaptive_data or 'results_page_selectors' not in adaptive_data:
            raise StrategyError("AI response for adaptive components is missing required keys.")
        
        final_instruction = [adaptive_data['final_wait_instruction']]
        results_processing = adaptive_data['results_page_selectors']

        # CRITICAL CHECK: Ensure we actually got a dict, not None
        if not results_processing:
             raise StrategyError("AI returned 'null' for results_page_selectors. It failed to identify the results table.")
        
        return results_processing, final_instruction
    
    async def get_detail_page_selectors(self, triage_result: TriageResult) -> dict[str, Any]:
        self.logger.info("Attempting to determine detail page selectors.")
        cat_to_url = {v: k for k, v in triage_result.full_classification.items()}
        
        detail_page_url = cat_to_url.get("DETAIL_PAGE")
        if not detail_page_url:
            self.logger.warning("No 'DETAIL_PAGE' URL was found in TriageResult. Detail selectors will be empty.")
            return {}

        html_content = await self._get_contextual_html(detail_page_url, triage_result)
        prompt = self.detail_prompt_template.render(html_content=html_content)
        
        response_text = await self.gemini_service.generate_content(prompt)
        
        try:
            data = self._parse_llm_json_response(response_text)
            validated_blueprints = DetailPageProcessing.model_validate(data)
            self.logger.info(f"Successfully discovered and validated {len(data)} detail page blueprints.")
            return validated_blueprints.model_dump()
        except Exception as e:
            raise StrategyError(f"Failed to parse or validate detail page selector JSON from AI: {e}") from e

    async def generate_document_blueprints(self, html_content: str, image: Image.Image | None) -> dict[str, Any]:
        """Generates blueprints for extracting documents from the Documents tab."""
        self.logger.info("Generating Document Scraping Blueprints...")
        prompt = self._document_scraping_template.render(html_content=html_content)
        
        response_text = await self.gemini_service.generate_content(prompt, image=image)
        data = self._parse_llm_json_response(response_text)
        
        # We return the raw dict here; validation happens in the orchestrator/data contract layer
        return data


    async def _get_contextual_html(self, url: str | None, triage_result: TriageResult) -> str:
        if not url: return ""
        url_to_path_map = {meta.url: meta.path for meta in triage_result.source_metadata}
        file_path_str = url_to_path_map.get(url)
        if not file_path_str:
            self.logger.warning(f"Could not find a local file path for URL: {url}")
            return ""
        
        self.logger.info(f"Loading HTML for {url} from local file: {file_path_str}")
        try:
            return Path(file_path_str).read_text(encoding='utf-8', errors='ignore')
        except FileNotFoundError:
            self.logger.error(f"Local file for {url} not found at path: {file_path_str}")
            return ""

    def _parse_llm_json_response(self, response_text: str) -> dict[str, Any]:
        self.logger.debug(f"Parsing LLM JSON response: {response_text}")
        try:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
            json_text = match.group(1).strip() if match else response_text.strip()
            data = json.loads(json_text)
            if not isinstance(data, dict):
                raise StrategyError(f"LLM response was valid JSON but not a dictionary: {data}")
            return data
        except (json.JSONDecodeError, AttributeError) as e:
            raise StrategyError(f"Failed to parse JSON response from AI: {response_text}") from e

    async def correct_results_page_blueprint(self, report: ValidationReport) -> dict:
        self.logger.info("Correcting results page blueprint.")
        return await self._run_base_correction(report, self.results_correction_template)

    async def correct_detail_page_blueprints(self, report: ValidationReport) -> dict:
        self.logger.info("Correcting detail page blueprints.")
        return await self._run_base_correction(report, self.detail_correction_template)
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\agents\synthesis_agent.py
```
import logging
from typing import final, Any
from urllib.parse import urlparse

from pydantic import ValidationError as PydanticValidationError

from ..data_contracts import DraftPlan, ExecutionPlan
from ..exceptions import SynthesisError

@final
class SynthesisAgent:
    """
    Takes a validated DraftPlan and synthesizes it into a final,
    production-ready ExecutionPlan.
    """
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    async def run(self, draft_plan: DraftPlan) -> ExecutionPlan:
        """
        Transforms a DraftPlan into a final ExecutionPlan object.
        Returns the validated ExecutionPlan object.
        """
        domain = self._get_domain(draft_plan)
        self.logger.info(f"Starting synthesis of final execution plan for domain '{domain}'.")
        try:
            execution_plan_data = self._transform_to_final_schema(draft_plan, domain)
            validated_plan = ExecutionPlan.model_validate(execution_plan_data)
            
            # Local Mode: We do not push to dataset here. 
            # The orchestrator/main loop handles file saving.
            
            self.logger.info(f"Synthesis complete for domain '{domain}'.")
            return validated_plan

        except PydanticValidationError as e:
            raise SynthesisError(f"CRITICAL: Final data failed schema validation for domain {domain}. Error: {e}")
        except Exception as e:
            raise SynthesisError(f"An unexpected error occurred during synthesis for domain {domain}: {e}")

    def _get_domain(self, draft_plan: DraftPlan) -> str:
        """Robustly extracts the domain from the source triage result."""
        domain = draft_plan.source_triage_result.domain
        if not domain:
            raise SynthesisError("Cannot determine domain: 'domain' field is missing from the source TriageResult.")
        return domain

    def _transform_to_final_schema(self, draft_plan: DraftPlan, domain: str) -> dict[str, Any]:
        """
        Pure data mapping function from the internal DraftPlan to a dictionary
        matching the ExecutionPlan schema.
        """
        # PROJECT GENESIS DIRECTIVE: This is the designated point for final
        # structural transformation. The raw dict from the DraftPlan is wrapped
        # to match the Pydantic model required by ExecutionPlan.
        detail_processing_for_synthesis = draft_plan.detail_page_processing
        
        # Note: 'document_page_processing' is currently populated in the Orchestrator,
        # not the DraftPlan, because Phase 5 is an Orchestrator-level loop.
        # This agent simply passes through what it has.
        # However, for consistency, if DraftPlan is updated later, we map it here.
        # For now, DraftPlan doesn't carry document_page_processing, so we return None.
        # The Orchestrator overrides this in the final construction anyway.

        return {
            "domain": domain,
            "instructions": draft_plan.instructions,
            "results_page_processing": draft_plan.results_page_processing,
            "detail_page_processing": detail_processing_for_synthesis,
            "document_page_processing": None
        }
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\agents\triage_agent.py
```
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import final, Coroutine, Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from ..data_contracts import TriageResult, HtmlPageMetadata
from ..exceptions import TriageError
from ..services.gemini_service import GeminiService
from ..data_contracts import DownloadedArtifact

# This prompt is the agent's core IP. It MUST NOT be modified.
_TRIAGE_PROMPT_TEMPLATE = """
You are an expert web intelligence analyst for UK Local Planning Authorities. Your task is to classify URLs to find the entry point for searching Planning Applications.

Valid Categories:
- "HOMEPAGE": The main landing page of the planning section.
- "SEARCH_PAGE": A page containing a form to search for applications by **Reference Number** (Key indicators: "Simple Search", "Application Search", inputs for "Reference").
- "DISCLAIMER_PAGE": A page asking to "Agree" or "Accept" terms before searching (Copyright/GDPR).
- "LANDING_PAGE": A page describing the planning service with a link to the search system (e.g. "View Planning Applications", "Search for an application").
- "ADVANCED_SEARCH": A complex search form (Second priority).
- "IRRELEVANT": Building control, licencing, weekly lists, contact pages.

Analyze the metadata. Your goal is to find the **SEARCH_PAGE** or the **DISCLAIMER_PAGE** blocking it.

Respond ONLY with a single JSON object mapping URLs to categories.

Example Response:
{
  "https://.../online-applications/": "HOMEPAGE",
  "https://.../simpleSearch.do": "SEARCH_PAGE"
}

--- METADATA BATCH ---
{metadata_json}
"""

_VALID_CATEGORIES = frozenset([
    "HOMEPAGE", "SEARCH_PAGE", "DISCLAIMER_PAGE",
    "ADVANCED_SEARCH", "LANDING_PAGE", "IRRELEVANT"
])

_SELECTION_PRIORITY = [
    "SEARCH_PAGE", "DISCLAIMER_PAGE", "LANDING_PAGE", "HOMEPAGE"
]

@final
class TriageAgent:
    """
    Scans a directory of raw web data, classifies pages using an LLM,
    and selects high-value candidates for the next pipeline stage.
    """
    def __init__(
        self,
        gemini_service: GeminiService,
        max_candidates: int = 10,
        min_required_candidates: int = 1,
    ):
        self.gemini_service = gemini_service
        self.max_candidates = max_candidates
        self.min_required_candidates = min_required_candidates
        self.logger = logging.getLogger(self.__class__.__name__)

    from urllib.parse import urlparse

    async def run(self, artifact_manifest: list[DownloadedArtifact]) -> TriageResult:
        """Executes the full triage process."""
        self.logger.info(f"Running triage on manifest with {len(artifact_manifest)} artifacts.")
        if not artifact_manifest:
            raise TriageError("No artifacts found in the provided manifest.")

        metadata_objects = await self._extract_metadata_batch(artifact_manifest)
        if not metadata_objects:
            raise TriageError("Could not extract any metadata from the found HTML files.")

        prompt = self._build_gemini_prompt(metadata_objects)
        response_text = await self.gemini_service.generate_content(prompt)
        classifications = self._parse_and_validate_response(response_text)
        
        selected_urls = self._select_candidate_urls(classifications)
        self._validate_candidate_set(selected_urls, classifications)

        url_to_path_map = {meta.url: meta.path for meta in metadata_objects}
        candidate_paths = {url: url_to_path_map[url] for url in selected_urls if url in url_to_path_map}
        
        # Mandate 4: Determine domain once and pass it forward.
        domain = urlparse(metadata_objects[0].url).netloc

        self.logger.info(f"Triage successful for domain '{domain}'. Selected {len(candidate_paths)} candidates.")
        return TriageResult(
            domain=domain,
            candidate_urls=candidate_paths,
            full_classification=classifications,
            source_metadata=metadata_objects
        )

    async def _parse_single_html(self, artifact: DownloadedArtifact) -> HtmlPageMetadata | None:
        """Parses a single artifact's HTML for its metadata."""
        try:
            html_path = Path(artifact.html_path)
            content = await asyncio.to_thread(html_path.read_text, encoding='utf-8', errors='ignore')
            soup = BeautifulSoup(content, 'html.parser')

            # The URL from the manifest is the source of truth
            url = artifact.url
            
            title_tag = soup.find('title')
            title = title_tag.string.strip() if title_tag else None
            
            desc_tag = soup.find('meta', {'name': 'description'})
            description = desc_tag['content'].strip() if desc_tag and 'content' in desc_tag.attrs else None

            tags_to_count = ['h1', 'form', 'input', 'article', 'section']
            structural_summary = {tag: len(soup.find_all(tag)) for tag in tags_to_count}

            return HtmlPageMetadata(
                url=url, 
                path=artifact.html_path, 
                title=title, 
                description=description, 
                screenshot_path=artifact.screenshot_path, 
                structural_summary=structural_summary
            )
        except Exception as e:
            self.logger.warning(f"Could not parse metadata from {artifact.html_path}: {e}")
            return None

    async def _extract_metadata_batch(self, artifact_manifest: list[DownloadedArtifact]) -> list[HtmlPageMetadata]:
        """Asynchronously parses artifacts from the manifest to extract metadata."""
        self.logger.info(f"Extracting metadata from {len(artifact_manifest)} artifacts concurrently.")
        tasks: list[Coroutine[Any, Any, HtmlPageMetadata | None]] = [self._parse_single_html(artifact) for artifact in artifact_manifest]
        results = await asyncio.gather(*tasks)
        
        valid_metadata = [res for res in results if res]
        self.logger.info(f"Successfully extracted metadata for {len(valid_metadata)} artifacts.")
        return valid_metadata

    def _build_gemini_prompt(self, metadata_objects: list[HtmlPageMetadata]) -> str:
        """Formats the metadata into the final prompt for the LLM."""
        metadata_for_json = [
            {
                "url": meta.url, 
                "title": meta.title, 
                "description": meta.description,
                "structural_summary": meta.structural_summary
            }
            for meta in metadata_objects
        ]
        metadata_json_string = json.dumps(metadata_for_json, indent=2)
        # Use .replace() for a single, unambiguous substitution to avoid conflicts with JSON braces.
        return _TRIAGE_PROMPT_TEMPLATE.replace('{metadata_json}', metadata_json_string)

    def _parse_and_validate_response(self, response_text: str) -> dict[str, str]:
        """Parses, cleans, and validates the structure and content of the LLM's response."""
        self.logger.info("Parsing and validating Gemini response.")
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
        json_text = match.group(1) if match else response_text

        try:
            parsed_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise TriageError("Failed to parse non-JSON response from AI.") from e

        if not isinstance(parsed_data, dict):
            raise TriageError("AI response is valid JSON but has an invalid structure (expected a dictionary).")

        for category in parsed_data.values():
            if category not in _VALID_CATEGORIES:
                raise TriageError(f"AI response contained an invalid category: '{category}'.")
        
        self.logger.info(f"Successfully parsed and validated {len(parsed_data)} classifications.")
        return parsed_data

    def _select_candidate_urls(self, classifications: dict[str, str]) -> list[str]:
        """Selects the best URLs based on the priority list."""
        selected = []
        seen_urls = set()
        
        for category in _SELECTION_PRIORITY:
            for url, cat in classifications.items():
                if cat == category and url not in seen_urls:
                    selected.append(url)
                    seen_urls.add(url)
                    if len(selected) >= self.max_candidates:
                        self.logger.info(f"Reached max candidates ({self.max_candidates}). Finalizing selection.")
                        return selected
        return selected

    def _validate_candidate_set(self, candidates: list[str], classifications: dict[str, str]):
        """Enforces the Fail-Fast Protocol by checking for essential page types."""
        self.logger.info(f"Validating candidate set of size {len(candidates)}.")
        
        if len(candidates) < self.min_required_candidates:
            self.logger.error(f"Candidate validation failed. Classifications: {classifications}")
            raise TriageError(f"Triage failed: only found {len(candidates)} valid candidates (Threshold: {self.min_required_candidates}).")
        
        candidate_categories = {classifications[url] for url in candidates if url in classifications}
        
        # LOGIC UPDATE: We accept success if we found the Target (Search/Disclaimer) OR the Homepage.
        # We do not strictly require the Homepage if we already have the Search Page.
        valid_entry_points = {"SEARCH_PAGE", "DISCLAIMER_PAGE", "LANDING_PAGE", "HOMEPAGE"}
        
        if not valid_entry_points.intersection(candidate_categories):
             raise TriageError(f"Triage failed: No valid entry point ({valid_entry_points}) found in candidates: {candidate_categories}")
        
        self.logger.info("Candidate set passed all validation checks.")
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\agents\validation_agent.py
```
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

    async def _debug_snapshot(self, page: Page, prefix: str, save_dir: str | None = None) -> None:
        """Helper to create a debug snapshot immediately."""
        try:
            if save_dir:
                target_dir = Path(save_dir)
            else:
                target_dir = Path(tempfile.gettempdir()) / "amaryllis_debug"
            
            target_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(asyncio.get_event_loop().time() * 1000)
            
            # Save Screenshot
            img_path = target_dir / f"{prefix}_{timestamp}.png"
            await page.screenshot(path=str(img_path), full_page=True)
            
            print(f"    [SNAPSHOT] {img_path}") # Indented for readability in loop
        except Exception as e:
            print(f"    [SNAPSHOT FAILED] {e}")


    async def execute_probe(self, probe_plan: DraftPlan, temp_dir: str) -> ProbeResult:
        """Executes a minimal plan and captures the final multimodal state."""
        self.logger.info(f"Executing probe plan with {len(probe_plan.instructions)} instructions.")
        print(f"\n[DEBUG] --- START PROBE EXECUTION ({len(probe_plan.instructions)} steps) ---")
        page = await self.browser_context.new_page()
        
        # --- SUBSTITUTION LOGIC (Deep & Global) ---
        # We recursively swap {{REFERENCE_ID}} in ALL parameters (selectors, values, etc.)
        live_instructions = copy.deepcopy(probe_plan.instructions)
        real_ref_id = probe_plan.target_address

        for i, instr in enumerate(live_instructions):
            params = instr.get('params', {})
            for key, val in params.items():
                if isinstance(val, str) and "{{REFERENCE_ID}}" in val:
                    # Perform replacement
                    new_val = val.replace("{{REFERENCE_ID}}", real_ref_id)
                    params[key] = new_val
                    self.logger.info(f"Instr {i}: Substituted placeholder in '{key}' -> '{new_val}'")

        try:
            for i, instruction in enumerate(live_instructions):
                # Update 'page' reference if instruction (click) changed tabs
                print(f"[DEBUG] PROBE Step {i+1}: {instruction['command']} params={instruction.get('params')}")
                page = await self._execute_instruction(instruction, page)
                # Snapshot with index for sequence tracking
                await self._debug_snapshot(page, f"probe_step_{i+1:02d}", save_dir=temp_dir)
            
            # Probe was successful, now capture the state
            html = await page.content()
            final_url = page.url
            _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="probe_capture_")
            await page.screenshot(path=screenshot_path, full_page=True)
            self.logger.info(f"Probe successful. Captured live state to {screenshot_path}")
            
            return ProbeResult(
                is_successful=True,
                probe_plan=probe_plan.instructions,
                final_url=final_url,
                captured_html=html,
                captured_screenshot_path=screenshot_path
            )
        except Exception as e:
            self.logger.error(f"Probe execution failed. Reason: {e}")
            
            # --- BLACK BOX RECORDING ---
            # Attempt to capture the state at the moment of crash
            fail_html = None
            fail_img = None
            try:
                fail_html = await page.content()
                _, fail_img = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="crash_dump_")
                await page.screenshot(path=fail_img, full_page=True)
                self.logger.info(f"Crash dump saved to {fail_img}")
            except Exception as capture_error:
                self.logger.error(f"Could not capture crash dump: {capture_error}")

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
        """Performs a 'dry run' of the given DraftPlan."""
        self.logger.info("Starting validation dry run.")
        print(f"\n[DEBUG] --- START VALIDATION DRY RUN ---")
        page = await self.browser_context.new_page()
        initial_url = page.url 
        extracted_links: list[str] = []

        # --- SUBSTITUTION LOGIC (Deep & Global) ---
        test_instructions = copy.deepcopy(draft_plan.instructions)
        real_ref_id = draft_plan.target_address

        for i, instr in enumerate(test_instructions):
            params = instr.get('params', {})
            for key, val in params.items():
                if isinstance(val, str) and "{{REFERENCE_ID}}" in val:
                    new_val = val.replace("{{REFERENCE_ID}}", real_ref_id)
                    params[key] = new_val

        try:
            # --- Stage 1: Execute Navigation Plan ---
            for i, instruction in enumerate(test_instructions):
                self.logger.info(f"Executing instruction {i+1}/{len(test_instructions)}: {instruction}")
                print(f"[DEBUG] Step {i+1}: {instruction['command']} params={instruction.get('params')}")
                try:
                    page = await self._execute_instruction(instruction, page)
                    await self._debug_snapshot(page, f"step_{i+1}_success", save_dir=temp_dir)
                except PlaywrightTimeoutError as e:
                    self.logger.warning(f"Instruction {i} failed with a Timeout. Running diagnostics.")
                    failure_html = await page.content()
                    _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="failure_")
                    await page.screenshot(path=screenshot_path, full_page=True)
                    self.logger.info(f"Captured failure state screenshot to {screenshot_path}")

                    # --- Mandate C-2: THE DIAGNOSTIC SCALPEL ---
                    is_final_heracles_assert = (
                        i == len(draft_plan.instructions) - 1 and
                        instruction.get("command") == "WAIT_FOR_SELECTOR"
                    )

                    if is_final_heracles_assert:
                        failure_type = "SEARCH_OUTCOME_MISMATCH"
                        analysis_notes = (
                            "Analysis: The plan's final assertion (WAIT_FOR_SELECTOR) failed. "
                            "This indicates the search action executed but did not produce the expected outcome "
                            "(e.g., 'No results found'). The strategist should analyze the screenshot for such messages."
                        )
                    else:
                        # Fallback to existing Sentinel diagnostics
                        failure_type, analysis_notes = self._diagnose_timeout_failure(failure_html)
                    
                    return ValidationReport(
                        is_valid=False, original_plan=draft_plan, failure_index=i,
                        failure_type=failure_type,
                        failure_reason=f"Instruction failed: {instruction}. Error: {type(e).__name__} - {e}",
                        failure_context_html=failure_html,
                        failure_screenshot_path=screenshot_path,
                        failure_stage="NAVIGATION",
                        failure_analysis_notes=analysis_notes
                    )
                except Exception as e:
                    self.logger.warning(f"Instruction {i} failed. Reason: {type(e).__name__} - {e}")
                    error_str = str(e).lower()
                    failure_type: Literal["TIMEOUT", "VALIDATION_ERROR", "AMBIGUOUS_SELECTOR", "UNKNOWN"]

                    if "strict mode violation" in error_str:
                        failure_type = "AMBIGUOUS_SELECTOR"
                    else:
                        failure_type = "VALIDATION_ERROR"
                    
                    failure_html = await page.content()
                    _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="failure_")
                    await page.screenshot(path=screenshot_path, full_page=True)
                    self.logger.info(f"Captured failure state screenshot to {screenshot_path}")

                    return ValidationReport(
                        is_valid=False, original_plan=draft_plan, failure_index=i, failure_type=failure_type,
                        failure_reason=f"Instruction failed: {instruction}. Error: {type(e).__name__} - {e}",
                        failure_context_html=failure_html,
                        failure_screenshot_path=screenshot_path,
                        failure_stage="NAVIGATION"
                    )
            
            # --- MANDATE P-1: "PROOF-OF-SEARCH" VALIDATION ---
            final_url = page.url
            if not self._validate_search_intent(initial_url, final_url, draft_plan.instructions):
                self.logger.error("MISSION FAILURE: Navigation plan executed but did not result in a valid search state.")
                failure_html = await page.content()
                _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="mission_failure_")
                await page.screenshot(path=screenshot_path, full_page=True)
                return ValidationReport(
                    is_valid=False,
                    original_plan=draft_plan,
                    failure_index=len(draft_plan.instructions),
                    failure_type="MISSION_FAILURE",
                    failure_reason="The executed plan did not navigate to a recognizable search results page (URL did not change as expected).",
                    failure_context_html=failure_html,
                    failure_screenshot_path=screenshot_path,
                    failure_stage="NAVIGATION" # This is a navigation-level failure
                )

            # --- Stage 2: Validate Results Page Blueprints ---
            self.logger.info("Checking for results page blueprints...")
            results_processing = draft_plan.results_page_processing
            
            if results_processing:
                try:
                    # First, ensure the container exists
                    container_selector = results_processing.get('listing_container_selector')
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
                    pagination_blueprint = results_processing.get('pagination_blueprint')
                    if pagination_blueprint:
                        self.logger.info("Validating pagination blueprint...")
                        pagination_selector = pagination_blueprint.get('next_page_selector')
                        await page.locator(pagination_selector).first.wait_for(state='visible', timeout=self.default_timeout_ms)
                        self.logger.info("Pagination blueprint validation successful.")
                    # --- END INSERTION ---

                except Exception as e:
                    self.logger.warning(f"Results page blueprint validation failed. Reason: {type(e).__name__} - {e}")
                    failure_html = await page.content()
                    _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="failure_")
                    await page.screenshot(path=screenshot_path, full_page=True)
                    self.logger.info(f"Captured failure state screenshot to {screenshot_path}")
                    return ValidationReport(
                        is_valid=False, original_plan=draft_plan, failure_index=len(draft_plan.instructions),
                        failure_type="VALIDATION_ERROR",
                        # UPDATED: More specific error reason
                        failure_reason=f"Results page validation failed. A blueprint (for listings or pagination) did not execute correctly. Error: {e}",
                        failure_context_html=failure_html,
                        failure_screenshot_path=screenshot_path,
                        failure_stage="RESULTS_PAGE"
                    )

            ## --- Stage 3: Validate Detail Page Blueprints (Project ANVIL Refactor) ---
            self.logger.info("Checking for detail page blueprints...")
            detail_blueprints = draft_plan.detail_page_processing or {}

            # --- PROJECT SENTRY DIRECTIVE: Validate only what is present. ---
            if detail_blueprints:
                self.logger.info("Validating detail page blueprints on first extracted link...")
                if not extracted_links:
                     raise ValidationError("Cannot validate detail page: No link was extracted from the results page.")
                
                detail_page_url = extracted_links[0]
                try:
                    await page.goto(detail_page_url, timeout=self.default_timeout_ms)
                    
                    failed_blueprint_keys: list[str] = []
                    
                    for name, blueprint in detail_blueprints.items():
                        if not blueprint: continue
                        try:
                            self.logger.info(f"Validating detail blueprint: {name}")
                            await self._validate_blueprint(blueprint, page, blueprint_name=name)
                        except Exception as e:
                            # AMPUTATE, DON'T RESUSCITATE
                            self.logger.warning(f"Amputating failing detail blueprint '{name}'. Reason: {type(e).__name__} - {e}")
                            failed_blueprint_keys.append(name)
                    
                    # --- Post-Iteration Review ---
                    if not failed_blueprint_keys:
                        # All blueprints passed. Success. The final return is handled outside this block.
                        self.logger.info("All detail page blueprints validated successfully.")
                    elif len(failed_blueprint_keys) == len(detail_blueprints):
                        # CATASTROPHIC FAILURE: All blueprints failed. Trigger self-healing.
                        self.logger.error("All detail page blueprints failed. Triggering a correction cycle.")
                        failure_html = await page.content()
                        _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="failure_")
                        await page.screenshot(path=screenshot_path, full_page=True)
                        return ValidationReport(
                            is_valid=False, original_plan=draft_plan, failure_index=len(draft_plan.instructions) + 1,
                            failure_type="VALIDATION_ERROR",
                            failure_reason=f"Catastrophic failure: All {len(detail_blueprints)} detail page blueprints failed validation.",
                            failure_context_html=failure_html, failure_screenshot_path=screenshot_path, failure_stage="DETAIL_PAGE"
                        )
                    else:
                        # PARTIAL SUCCESS: Harden the plan, then perform final integrity check.
                        self.logger.info(f"Hardening plan by removing {len(failed_blueprint_keys)} failed blueprints: {failed_blueprint_keys}")
                        hardened_plan = copy.deepcopy(draft_plan)
                        for key in failed_blueprint_keys:
                            hardened_plan.detail_page_processing.pop(key, None)
                        
                        # --- PROJECT GENESIS: SIMPLIFIED QUARANTINE GATE ---
                        try:
                            self.logger.info("Performing final structural integrity check on hardened plan components.")
                            for name, blueprint in hardened_plan.detail_page_processing.items():
                                if blueprint: # Ensure we don't validate nulls
                                    ExtractionBlueprint.model_validate(blueprint)
                            self.logger.info("Structural integrity check of detail blueprints PASSED.")
                            # --- PROJECT PHOENIX DIRECTIVE: Capture success state for PARTIAL success ---
                            final_url = page.url
                            final_html = await page.content()
                            _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="success_capture_")
                            await page.screenshot(path=screenshot_path, full_page=True)
                            self.logger.info(f"Captured successful final state to {screenshot_path}")

                            return ValidationReport(
                                is_valid=True,
                                original_plan=hardened_plan,
                                final_url=final_url,
                                final_html=final_html,
                                final_screenshot_path=screenshot_path
                            )
                        except PydanticValidationError as e:
                            self.logger.error(f"CRITICAL: A hardened blueprint failed final structural validation. Error: {e}")
                            return ValidationReport(
                                is_valid=False, original_plan=draft_plan,
                                failure_index=len(draft_plan.instructions) + 1,
                                failure_type="VALIDATION_ERROR",
                                failure_reason=f"Structural Integrity Failure: A component of the plan is malformed. Error: {e}",
                                failure_context_html=await page.content(),
                                failure_screenshot_path=None,
                                failure_stage="DETAIL_PAGE"
                            )
                except Exception as e:
                    # This block catches failures in page navigation or UNEXPECTED code errors.
                    self.logger.error(f"Detail page validation stage failed non-recoverably. Reason: {type(e).__name__} - {e}", exc_info=True)
                    failure_html = await page.content()
                    _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="failure_")
                    await page.screenshot(path=screenshot_path, full_page=True)
                    
                    if isinstance(e, (NameError, TypeError)):
                        raise ValidationError(f"A fatal, non-recoverable implementation error occurred in the validation agent: {e}") from e

                    return ValidationReport(
                        is_valid=False, original_plan=draft_plan, failure_index=len(draft_plan.instructions) + 1,
                        failure_type="VALIDATION_ERROR",
                        failure_reason=f"Detail page navigation or setup failed on URL {detail_page_url}. Error: {e}",
                        failure_context_html=failure_html, failure_screenshot_path=screenshot_path, failure_stage="DETAIL_PAGE"
                    )

            self.logger.info("Validation dry run completed successfully.")
            # --- PROJECT PHOENIX DIRECTIVE: Capture success state ---
            final_url = page.url
            final_html = await page.content()
            _, screenshot_path = tempfile.mkstemp(suffix=".png", dir=temp_dir, prefix="success_capture_")
            await page.screenshot(path=screenshot_path, full_page=True)
            self.logger.info(f"Captured successful final state to {screenshot_path}")

            return ValidationReport(
                is_valid=True,
                original_plan=draft_plan,
                final_url=final_url,
                final_html=final_html,
                final_screenshot_path=screenshot_path
            )
        finally:
            await page.close()

    def _diagnose_timeout_failure(self, html_content: str) -> tuple[str, str | None]:
        """Inspects HTML content to determine the likely cause of a timeout."""
        OBSTRUCTION_KEYWORDS = [
            'cookie', 'consent', 'privacy', 'accept', 'agree', 'continue',
            'gdpr', 'ccpa', 'banner', 'dialog'
        ]
        
        html_lower = html_content.lower()
        found_keywords = [kw for kw in OBSTRUCTION_KEYWORDS if kw in html_lower]

        if found_keywords:
            # --- Mandate S-2: Enhanced Diagnostic Hint ---
            notes = (
                "Analysis: Timeout occurred. This may be due to one of two reasons: "
                "(A) A cookie banner or other modal is obstructing interaction. Check the screenshot for overlays. "
                "(B) The targeted element is incorrect. If this was a button click, consider if an implicit "
                "submission using PRESS_KEY 'Enter' is more appropriate."
            )
            self.logger.warning("Timeout Diagnostics: Probable obstruction or incorrect submission strategy detected.")
            return "PROBABLE_OBSTRUCTION", notes
        
        # --- Mandate S-3: The "Wrong Neighborhood" Detector ---
        ZERO_RESULT_KEYWORDS = ["no results found", "no application found", "search returned no results", "zero results"]
        if any(z in html_lower for z in ZERO_RESULT_KEYWORDS):
             notes = (
                 "Analysis: The search executed, but returned 'No Results'. "
                 "The robot is likely on the wrong portal/database for this specific reference ID. "
                 "Check the page text for links to 'Legacy', 'Old', or 'Existing' systems (e.g., 'Visit our Aylesbury system')."
             )
             self.logger.warning("Timeout Diagnostics: Search returned zero results. Probable Portal Mismatch.")
             return "SEARCH_OUTCOME_MISMATCH", notes

        else:
            self.logger.info("Timeout Diagnostics: No obvious signs of obstruction found.")
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
            self.logger.info("Search intent validation PASSED based on successful execution of final WAIT_FOR_SELECTOR command.")
            return True

        # Fallback Pathfinder Logic for traditional websites.
        self.logger.info("Final command is not WAIT_FOR_SELECTOR. Falling back to URL-based validation.")
        if final_url == initial_url or final_url.strip('/') == initial_url.strip('/'):
            self.logger.warning(f"Search intent validation failed: URL did not change from initial state '{initial_url}'.")
            return False

        search_patterns = [r'\/search', r'\?q=', r'\?s=', r'\?query=', r'\?keyword=', r'\?location=']
        if any(re.search(pattern, final_url, re.IGNORECASE) for pattern in search_patterns):
            self.logger.info(f"Search intent validation PASSED. Final URL '{final_url}' contains a search pattern.")
            return True

        self.logger.warning(f"Search intent validation failed: Final URL '{final_url}' does not contain a recognizable search pattern.")
        return False

    def _apply_post_processing(self, value: str, steps: list[dict]) -> str | None:
        """Applies a sequence of post-processing steps to a single string value."""
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
            current_page = await self._execute_instruction(instruction, current_page)
            await self._debug_snapshot(current_page, f"exec_step_{i+1:02d}", save_dir=debug_dir)
        return current_page

    async def _execute_instruction(self, instruction: dict[str, Any], page: Page) -> Page:
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
            case "WAIT_FOR_NAVIGATION":
                await self._execute_wait_for_navigation(params, current_page)
            case "WAIT_FOR_SELECTOR":
                await self._execute_wait_for_selector(params, current_page)
            case "GOTO_URL":
                await self._execute_goto_url(params, current_page)
            case "PRESS_KEY":
                await self._execute_press_key(params, current_page)
            case "CHECK_OPTION":
                await self._execute_check_option(params, current_page)
            case _:
                raise ValidationError(f"Unknown command encountered in plan: '{command}'")
        
        return current_page

    async def _execute_goto_url(self, params: dict[str, Any], page: Page):
        """Navigates to a specific URL."""
        url = params.get('url')
        if not url:
            raise ValidationError("Instruction 'GOTO_URL' is missing required parameter: 'url'")
        
        try:
            await page.goto(url, timeout=self.default_timeout_ms)
            await page.wait_for_load_state("domcontentloaded", timeout=self.default_timeout_ms)
        except Exception as e:
            raise ValidationError(f"Failed to navigate to {url}. Error: {e}")

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
        """Presses a specific key on a focused element or selector."""
        selector = params.get('selector')
        key = params.get('key')
        if not selector or not key:
             raise ValidationError("Instruction 'PRESS_KEY' requires 'selector' and 'key'.")
        
        try:
            target = page.locator(selector).first
            await target.wait_for(state='visible', timeout=self.default_timeout_ms)
            await target.press(key)
        except Exception as e:
             raise ValidationError(f"Failed to press key '{key}' on '{selector}'. Error: {e}")

    async def _execute_check_option(self, params: dict[str, Any], page: Page):
        """Checks a radio button or checkbox."""
        selector = params.get('selector')
        if not selector:
             raise ValidationError("Instruction 'CHECK_OPTION' requires 'selector'.")
        
        try:
            target = page.locator(selector).first
            await target.wait_for(state='visible', timeout=self.default_timeout_ms)
            await target.check()
        except Exception as e:
             raise ValidationError(f"Failed to check option '{selector}'. Error: {e}")

    async def _execute_click_element(self, params: dict[str, Any], page: Page) -> Page:
        """
        Finds a clickable element and executes a click.
        Detects if a new tab/window was opened and returns the new Page if so.
        """
        selector = params.get('selector')
        if not selector:
            raise ValidationError("Instruction 'CLICK_ELEMENT' is missing required parameter: 'selector'")
        
        locator = page.locator(selector)
        visible_locator = locator.filter(visible=True)
        
        # Capture context state before click to detect new pages
        context = page.context
        
        try:
            target = visible_locator.first
            await target.wait_for(state='visible', timeout=self.default_timeout_ms)
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
                self.logger.warning(f"Soft Failure: Consent element '{selector}' not found in time. Proceeding.")
                return page

            # DIAGNOSTIC: Check if it's hidden
            if await locator.count() > 0 and not await locator.first.is_visible():
                # Attempt to scroll into view before failing
                await locator.first.scroll_into_view_if_needed()
                if not await locator.first.is_visible():
                     raise ValidationError(
                        f"Target element '{selector}' exists in the DOM but is HIDDEN/OBSCURED. "
                        "Accordion/Tab logic or scrolling required."
                     )
            
            self.logger.warning(f"Click failed on visible filter for '{selector}'. Reverting to raw locator with scroll+force.")
            
            # Force Scroll & Click Strategy
            try:
                target_raw = locator.first
                # Increased scroll timeout for robust navigation (2s -> 5s)
                try:
                    await target_raw.scroll_into_view_if_needed(timeout=5000)
                except Exception:
                    self.logger.warning(f"Scroll into view failed for '{selector}'. Attempting blind force click.")
                
                await target_raw.click(force=True, timeout=5000)
            except Exception as e:
                self.logger.warning(f"Force click failed for '{selector}'. Attempting JS dispatch click. Error: {e}")
                try:
                    await target_raw.evaluate("element => element.click()")
                except Exception as js_e:
                    raise ValidationError(f"All click methods (standard, force, JS) failed for '{selector}'. Final Error: {js_e}")

        # --- NEW WINDOW DETECTION ---
        # Wait briefly for any new page logic to trigger (Playwright events are async)
        # We assume if a new page appears within 2 seconds, it was due to this click.
        try:
            # Wait for a new page event or timeout
            async with context.expect_page(timeout=2000) as page_info:
                new_page = await page_info.value
                await new_page.wait_for_load_state("domcontentloaded")
                # Add networkidle wait for React/Angular portals (Antrim fix)
                try:
                    await new_page.wait_for_load_state("networkidle", timeout=5000)
                except:
                    pass # Don't fail if network is chatty, just proceed
                self.logger.info(f"New tab detected! Switching context to new page: {new_page.url}")
                return new_page
        except Exception:
            # No new page detected, sticking with current page
            pass
            
        return page
            
    
    async def _execute_wait_for_navigation(self, params: dict[str, Any], page: Page):
        """Waits for the page to transition and for network activity to cease."""
        timeout = params.get('timeout_ms', self.default_timeout_ms)
        
        # UPGRADED: From 'domcontentloaded' to the much more robust 'networkidle'.
        # This waits for the network to be quiet, indicating async calls are likely complete.
        await page.wait_for_load_state('networkidle', timeout=timeout)

    async def _execute_wait_for_selector(self, params: dict[str, Any], page: Page):
        """Waits for a specific selector to appear in a given state."""
        selector = params.get('selector')
        if not selector:
            raise ValidationError("Instruction 'WAIT_FOR_SELECTOR' is missing required parameter: 'selector'")
        
        state = params.get('state', 'visible')
        timeout = params.get('timeout_ms', self.default_timeout_ms)
        
        try:
            await page.locator(selector).first.wait_for(state=state, timeout=timeout)
        except PlaywrightTimeoutError as e:
            # --- PROJECT AEGIS: SOFT FAIL ON CONSENT WAITS ---
            # If the AI adds an explicit wait for a consent banner that isn't there,
            # we treat it as a non-fatal "Soft Failure" and proceed.
            selector_lower = selector.lower()
            is_consent = any(x in selector_lower for x in ['accept', 'agree', 'cookie', 'consent', 'close', 'dismiss', 'gdpr'])
            
            if is_consent:
                self.logger.warning(f"Soft Failure: Wait for consent element '{selector}' timed out. Assuming already handled. Proceeding.")
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
                f"CRITICAL PLAN CORRUPTION: The blueprint for '{blueprint_name}' is not a valid dictionary object. "
                f"It is of type '{type(blueprint).__name__}'. The self-healing mechanism is likely using an obsolete prompt."
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
                    raise ValidationError(f"Blueprint validation failed: Selector '{selector}' found an element but it has no inner text.")
                else:
                    raise ValidationError(f"Blueprint validation failed: Selector '{selector}' found an element but it lacks the attribute '{attr_name}'.")

            post_processing_steps = blueprint.get('post_processing')
            if post_processing_steps:
                final_value = self._apply_post_processing(raw_value, post_processing_steps)
                if final_value is None:
                    raise ValidationError(f"Post-processing failed for selector '{selector}'. Raw value was '{raw_value}'.")

                if "url" in blueprint_name.lower():
                    parsed_url = urlparse(final_value)
                    if not (parsed_url.scheme in ['http', 'https'] and parsed_url.netloc):
                        raise ValidationError(f"Post-processed value '{final_value}' from '{blueprint_name}' is not a valid absolute URL.")
        
        except PlaywrightTimeoutError as e:
            raise ValidationError(f"Blueprint validation for selector '{selector}' failed with timeout.") from e
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\services\api_key_manager.py
```
import logging

class SimpleApiKeyManager:
    """A simple, non-thread-safe API key manager for a single asyncio worker."""
    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("API key list cannot be empty.")
        self.keys = keys
        self.current_key_index = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"API Key Manager initialized with {len(self.keys)} keys.")

    def get_key(self) -> str:
        return self.keys[self.current_key_index]

    def rotate_key(self) -> str:
        """Rotates to the next key in the list, returns the new key."""
        self.current_key_index = (self.current_key_index + 1) % len(self.keys)
        self.logger.warning(f"Rotated to next API key (index {self.current_key_index}).")
        return self.get_key()
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\services\gemini_service.py
```
import asyncio
import logging
from google import genai
from google.genai import types
from PIL import Image
from .api_key_manager import SimpleApiKeyManager
from ..exceptions import PipelineError

class GeminiService:
    """A centralized service for interacting with the Gemini API (v1 SDK)."""
    def __init__(self, api_key_manager: SimpleApiKeyManager, models: list[str]):
        self.api_key_manager = api_key_manager
        self.models = models
        self.current_model_index = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not self.models:
            raise ValueError("Model list cannot be empty.")
        self.logger.info(f"Initialized GeminiService with models: {self.models}")

    def _get_current_model(self) -> str:
        return self.models[self.current_model_index]

    def _rotate_model(self):
        """Rotates to the next model in the priority list."""
        old = self._get_current_model()
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        new = self._get_current_model()
        self.logger.warning(f"Rotating Model due to overload: {old} -> {new}")

    async def generate_content(self, prompt: str, image: Image.Image | None = None, max_retries: int = 3) -> str:
        """
        Generates content with strategies for both Quota (429) and Overload (503).
        """
        contents = [prompt]
        if image:
            contents.append(image)
            self.logger.info("Image provided for multimodal generation.")

        total_keys = len(self.api_key_manager.keys)
        quota_retries = 0   # Counter for Key Rotation (429)
        generic_retries = 0 # Counter for Standard Failures (Timeout, etc)
        
        # We loop until success or exhaustion
        while True:
            current_model = self._get_current_model()
            api_key = self.api_key_manager.get_key()
            
            try:
                # self.logger.info(f"Invoking Gemini API (Model: {current_model})...")
                client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
                
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=current_model,
                    contents=contents
                )
                return response.text

            except Exception as e:
                error_str = str(e).lower()
                
                # --- STRATEGY 1: QUOTA / RATE LIMIT (429) -> ROTATE KEY ---
                if any(x in error_str for x in ['exhausted', 'quota', 'limit', '429']):
                    quota_retries += 1
                    if quota_retries >= total_keys:
                        raise PipelineError(f"Gemini API Quota exhausted on all {total_keys} keys.") from e
                    
                    self.logger.warning(f"Quota Hit (429). Rotating Key...")
                    self.api_key_manager.rotate_key()
                    continue

                # --- STRATEGY 2: SERVICE OVERLOAD (503) -> ROTATE MODEL ---
                elif '503' in error_str or 'overloaded' in error_str:
                    self.logger.warning(f"Model Overloaded (503). Switching Model...")
                    self._rotate_model()
                    # We treat a model switch as a 'free' retry regarding generic counts, 
                    # but we pause briefly to let the system settle.
                    await asyncio.sleep(1)
                    continue

                # --- STRATEGY 3: GENERIC FAILURE -> EXPONENTIAL BACKOFF ---
                else:
                    generic_retries += 1
                    self.logger.error(f"Generic API Error ({generic_retries}/{max_retries}): {e}")
                    
                    if generic_retries >= max_retries:
                        raise PipelineError(f"Gemini API failed after {max_retries} generic retries. Last error: {e}") from e
                    
                    await asyncio.sleep(2 ** generic_retries)
```

c:\Users\brand\Desktop\renewables\amaryllis\cognitive_profiler\services\local_crawler.py
```
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
        self.logger.info(f"Crawling {start_url} for {lpa_name}")
        print(f"\n[DEBUG] Crawling started for {lpa_name} at {start_url}")
        page = await self.context.new_page()
        artifacts = []
        
        try:
            # 1. Capture Homepage
            try:
                await page.goto(start_url, timeout=30000)
                await page.wait_for_load_state("domcontentloaded")
            except Exception as e:
                self.logger.error(f"Failed to load homepage {start_url}: {e}")
                return []

            print(f"[DEBUG] Capturing homepage...")
            artifacts.append(await self._capture(page, lpa_name, "homepage"))
            print(f"[DEBUG] Homepage captured.")

            # 2. Extract Candidate Links (Heuristic: Look for "Search" or "Planning")
            print(f"[DEBUG] Extracting links from homepage...")
            links = await page.locator("a[href]").all()
            candidate_urls = set()
            from urllib.parse import urljoin
            
            for link in links:
                try:
                    # Comprehensive Text Extraction (Visible + Accessibility Attributes)
                    visible_text = (await link.inner_text()).lower()
                    aria_label = (await link.get_attribute("aria-label") or "").lower()
                    title_attr = (await link.get_attribute("title") or "").lower()
                    combined_text = f"{visible_text} {aria_label} {title_attr}"

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

            self.logger.info(f"Found {len(sorted_candidates)} candidate links. Visiting top 30.")
            print(f"[DEBUG] Found {len(sorted_candidates)} candidate links. Top 5: {sorted_candidates[:5]}")
            
            for i, url in enumerate(sorted_candidates[:30]):
                print(f"[DEBUG] Visiting candidate {i+1}: {url}")
                try:
                    await page.goto(url, timeout=15000)
                    await page.wait_for_load_state("domcontentloaded")
                    print(f"[DEBUG] Capturing candidate {i+1}...")
                    artifacts.append(await self._capture(page, lpa_name, f"candidate_{i}"))
                    print(f"[DEBUG] Candidate {i+1} captured.")
                except Exception:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Crawl failed: {e}")
        finally:
            await page.close()
            
        return artifacts

    async def _capture(self, page: Page, lpa_name: str, tag: str) -> DownloadedArtifact:
        """Saves current page state to a temp directory."""
        # Use system temp dir + amaryllis
        temp_dir = Path(tempfile.gettempdir()) / "amaryllis_cache" / lpa_name
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        filename_base = f"{tag}_{int(asyncio.get_event_loop().time())}"
        html_path = temp_dir / f"{filename_base}.html"
        img_path = temp_dir / f"{filename_base}.png"
        
        print(f"[DEBUG] Saving screenshot to: {img_path}")
        
        await page.screenshot(path=str(img_path), full_page=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(await page.content())
            
        return DownloadedArtifact(
            url=page.url,
            html_path=str(html_path),
            screenshot_path=str(img_path)
        )
```

c:\Users\brand\Desktop\renewables\amaryllis\src\check_models.py
```

import sys
import os
import asyncio

# --- PATH INJECTION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from cognitive_profiler.settings import settings
from google import genai
from google.genai import types

def list_models():
    if not settings.gemini_api_keys:
        print("No API keys found.")
        return

    api_key = settings.gemini_api_keys[0]
    print(f"Using API Key: {api_key[:5]}...{api_key[-5:]}")

    try:
        # Try v1beta first as it is more likely to have newer models
        print("\n--- Checking models (v1beta) ---")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
        for m in client.models.list():
            if hasattr(m, 'supported_actions') and "generateContent" in m.supported_actions:
               print(f"{m.name} - {m.display_name}")
            elif not hasattr(m, 'supported_actions'):
                 print(f"{m.name} (No supported_actions info)")

    except Exception as e:
        print(f"Error checking v1beta: {e}")

def test_specific_models(models_to_test: list[str]):
    """
    Tests a specific list of model names to see if they are available for generation,
    even if they don't appear in the standard list() output.
    """
    if not settings.gemini_api_keys:
        print("No API keys found.")
        return

    api_key = settings.gemini_api_keys[0]
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
    
    print(f"\n--- Testing Specific Models (v1beta) ---")
    for model_name in models_to_test:
        # Normalize name (remove 'models/' prefix if user added it, or ensure it's handled)
        # The 'model' argument in generate_content usually handles just the ID or 'models/ID'
        
        print(f"Testing '{model_name}':".ljust(40), end="", flush=True)
        try:
             # Attempt a minimal generation
             response = client.models.generate_content(
                model=model_name,
                contents="Hello",
                config=types.GenerateContentConfig(max_output_tokens=5)
             )
             print(f"✅ AVAILABLE")
        except Exception as e:
            err_str = str(e).lower()
            if "cw: 404" in err_str or "not found" in err_str:
                 print(f"❌ NOT FOUND / INVALID")
            elif "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                 print(f"⚠️  AVAILABLE (But Rate Limited)")
            elif "400" in err_str: # Bad request often means invalid model name for some APIs
                 print(f"❌ INVALID REQUEST (Possible 404)")
            else:
                 print(f"❓ ERROR: {e}")

if __name__ == "__main__":
    # 1. OPTIONAL: List all standard models
    list_models() 
    
    # 2. Test specific models (Edit this list to check for hidden/unlisted models)
    custom_models = [
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-2.5-pro-1p-freebie"
    ]
    test_specific_models(custom_models)
```

c:\Users\brand\Desktop\renewables\amaryllis\src\main.py
```
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
LPA_REGISTRY_PATH = r"C:\Users\brand\Desktop\renewables\config\lpa_registry.json"
CSV_PATH = r"C:\Users\brand\Desktop\renewables\all_solar_applications_and_dates.csv"
OUTPUT_DIR = r"C:\Users\brand\Desktop\renewables\lpa_plans"

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("amaryllis_run.log"),
            logging.StreamHandler()
        ]
    )

def get_test_reference(lpa_name: str, csv_data: list) -> str | None:
    """
    Finds a valid reference ID for the given LPA from the CSV.
    We prioritize 'Appeals' or 'Refusals' as they often have more documents,
    but for now, any valid reference will do to map the path.
    """
    # Normalize lpa_name for comparison (e.g., "Aberdeen City" -> "aberdeen")
    search_term = lpa_name.lower().replace("council", "").strip()
    
    for row in csv_data:
        # Filter out withdrawn or abandoned applications to ensure a valid reference is used
        dev_status = row.get('Development Status (short)', '').lower()
        withdrawn_date = row.get('Planning Application Withdrawn', '').strip()
        expired_date = row.get('Planning Permission Expired', '').strip()

        
        if "withdrawn" in dev_status.lower() or "abandoned" in dev_status.lower() or "expired" in dev_status.lower() or withdrawn_date != "" or expired_date != "":
            continue

        csv_authority = row.get('Planning Authority', '').lower()
        if search_term in csv_authority:
            ref = row.get('Planning Application Reference')
            if ref and len(ref) > 3: # Basic validation
                return ref
    return None

async def main():
    setup_logging()
    logger = logging.getLogger("AmaryllisLocal")

    # 1. Load Configuration Data
    if not os.path.exists(LPA_REGISTRY_PATH):
        logger.error(f"Registry not found at {LPA_REGISTRY_PATH}")
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
            logger.warning("CSV failed to open with cp1252, trying latin1 fallback.")
            # Fallback: Latin-1 (Aggressive fallback that rarely raises errors)
            with open(CSV_PATH, 'r', encoding='latin1') as f:
                reader = csv.DictReader(f)
                csv_data = list(reader)
    else:
        logger.error(f"CSV not found at {CSV_PATH}")
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
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
                logger.info(f"Skipping {lpa_name} - Plan already exists.")
                continue

            logger.info(f"--- Starting Analysis for: {lpa_name} ---")

            # A. Find a Test Reference ID
            test_ref = get_test_reference(lpa_name, csv_data)
            if not test_ref:
                logger.warning(f"Skipping {lpa_name} - No matching reference ID found in CSV.")
                continue
            
            logger.info(f"Test Subject: {test_ref} | URL: {portal_url}")

            try:
                # B. Crawl for Context (The Spider)
                artifacts = await crawler.crawl(portal_url, lpa_name)
                if not artifacts:
                    logger.error(f"Crawler failed to retrieve pages for {lpa_name}")
                    continue

                # C. Run The Cognitive Profiler
                # Define a predictable debug path: output_dir/lpa_name/debug_snapshots
                debug_path = Path(OUTPUT_DIR) / lpa_name / "debug_snapshots"
                debug_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Debug snapshots will be saved to: {debug_path}")

                # NOTE: We pass the reference ID as 'target_address'
                result = await profiler.run_pipeline(
                    artifact_manifest=artifacts,
                    target_address=test_ref, 
                    target_description="Planning Application Summary and Documents",
                    temp_data_path=str(debug_path)
                )

                if result.status == "SUCCESS" and result.execution_plan:
                    # Save the Plan
                    with open(plan_file, 'w', encoding='utf-8') as f:
                        # We dump the raw model dict using mode='json' to serialize datetimes
                        json.dump(result.execution_plan.model_dump(mode='json'), f, indent=2)
                    logger.info(f"SUCCESS: Plan generated and saved to {plan_file}")
                else:
                    logger.error(f"FAILURE for {lpa_name}: {result.error_message}")

            except Exception as e:
                logger.critical(f"CRITICAL ERROR processing {lpa_name}: {e}", exc_info=True)

            # Brief pause to be polite
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
```

c:\Users\brand\Desktop\renewables\amaryllis\tests\verify_strategist_fix.py
```
import asyncio
import logging
from unittest.mock import MagicMock
from cognitive_profiler.agents.strategist_agent import StrategistAgent
from cognitive_profiler.services.gemini_service import GeminiService

# Mock Logger
logging.basicConfig(level=logging.INFO)

async def test_prompt_hardening():
    print("Testing StrategistAgent Prompt Hardening...")
    
    mock_gemini = MagicMock(spec=GeminiService)
    agent = StrategistAgent(gemini_service=mock_gemini)
    
    # Render the probe template
    prompt = agent.probe_prompt_template.render(
        domain="example.com",
        target_address="123",
        homepage_html="<body></body>",
        searchpage_html="<body></body>"
    )
    
    print("Checking for Rule 11...")
    if "Rule 11: The Law of Negative Constraints" in prompt:
        print("SUCCESS: Rule 11 is present.")
    else:
        print("FAILURE: Rule 11 is missing.")
        exit(1)

    print("Checking for 'Site Search' prohibition...")
    if "interact with \"Site Search\"" in prompt:
        print("SUCCESS: Site Search prohibition found.")
    else:
        print("FAILURE: Site Search prohibition missing.")
        exit(1)

if __name__ == "__main__":
    asyncio.run(test_prompt_hardening())
```

c:\Users\brand\Desktop\renewables\amaryllis\tests\verify_triage_fix.py
```
import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock
from cognitive_profiler.agents.triage_agent import TriageAgent, DownloadedArtifact 
from cognitive_profiler.services.gemini_service import GeminiService
from cognitive_profiler.data_contracts import HtmlPageMetadata

# Mock Logger
logging.basicConfig(level=logging.INFO)

async def test_landing_page_logic():
    print("Testing TriageAgent LANDING_PAGE logic...")
    
    # Mock Gemini Service
    mock_gemini = MagicMock(spec=GeminiService)
    # Return a JSON where one URL is a LANDING_PAGE
    mock_gemini.generate_content = AsyncMock(return_value='''
    ```json
    {
        "https://www.belfastcity.gov.uk/planning": "LANDING_PAGE",
        "https://www.belfastcity.gov.uk/contact": "IRRELEVANT"
    }
    ```
    ''')

    agent = TriageAgent(gemini_service=mock_gemini, min_required_candidates=1)
    
    # Dummy Artifacts
    artifacts = [
        DownloadedArtifact(url="https://www.belfastcity.gov.uk/planning", html_path="dummy.html", screenshot_path=None),
        DownloadedArtifact(url="https://www.belfastcity.gov.uk/contact", html_path="dummy.html", screenshot_path=None)
    ]
    
    # Directly monkeypatch the instance method
    agent._extract_metadata_batch = AsyncMock(return_value=[
        HtmlPageMetadata(url="https://www.belfastcity.gov.uk/planning", path="d", title="Plan", description="Desc", screenshot_path=None, structural_summary={}),
        HtmlPageMetadata(url="https://www.belfastcity.gov.uk/contact", path="d", title="Cont", description="Desc", screenshot_path=None, structural_summary={})
    ])

    result = await agent.run(artifacts)
    
    print(f"Result candidates: {result.candidate_urls}")
    
    assert "https://www.belfastcity.gov.uk/planning" in result.candidate_urls
    assert result.full_classification["https://www.belfastcity.gov.uk/planning"] == "LANDING_PAGE"
    print("SUCCESS: LANDING_PAGE was accepted and selected.")

if __name__ == "__main__":
    asyncio.run(test_landing_page_logic())
```

