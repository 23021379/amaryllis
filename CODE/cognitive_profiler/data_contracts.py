from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pydantic import BaseModel, Field, RootModel

# --- Ingestion Contracts ---
@dataclass
class DownloadedArtifact:
    """[REDACTED_BY_SCRIPT]"""
    url: str
    html_path: str
    screenshot_path: str | None = None

# --- Orchestrator Contracts ---
@dataclass
class PipelineResult:
    """[REDACTED_BY_SCRIPT]"""
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
    """[REDACTED_BY_SCRIPT]"""
    domain: str
    candidate_urls: dict[str, str]
    full_classification: dict[str, str]
    source_metadata: list[HtmlPageMetadata]

# --- Strategist Agent Contracts ---

@dataclass
class DraftPlan:
    """[REDACTED_BY_SCRIPT]"""
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
    """[REDACTED_BY_SCRIPT]"""
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
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "SECURITY_CHALLENGE",
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
    """[REDACTED_BY_SCRIPT]"""
    command: str
    params: dict[str, Any]
    description: str | None = Field(None, description="[REDACTED_BY_SCRIPT]'s purpose.")
    frame_name: str | None = Field(None, description="[REDACTED_BY_SCRIPT]")

from pydantic import model_validator

# --- New Extraction Blueprint Contracts ---

class PostProcessingStep(BaseModel):
    """[REDACTED_BY_SCRIPT]"""
    method: Literal["REGEX_EXTRACT"] = Field(
        description="[REDACTED_BY_SCRIPT]"
    )
    params: dict[str, Any] = Field(
        description="[REDACTED_BY_SCRIPT]'pattern': '...'}"
    )

    @model_validator(mode='after')
    def check_params(self) -> 'PostProcessingStep':
        if self.method == "REGEX_EXTRACT":
            if "pattern" not in self.params or not isinstance(self.params["pattern"], str):
                raise ValueError("[REDACTED_BY_SCRIPT]'pattern' string parameter.")
        return self

class ExtractionBlueprint(BaseModel):
    """
    An unambiguous blueprint for extracting and transforming data from a webpage.
    """
    selector: str = Field(description="[REDACTED_BY_SCRIPT]")
    extraction_method: Literal["TEXT", "ATTRIBUTE", "MULTIPLE_TEXT", "MULTIPLE_ATTRIBUTE"] = Field(
        description="[REDACTED_BY_SCRIPT]"
    )
    attribute_name: str | None = Field(
        None, 
        description="[REDACTED_BY_SCRIPT]'href', 'src', 'data-price'[REDACTED_BY_SCRIPT]'ATTRIBUTE' or 'MULTIPLE_ATTRIBUTE'."
    )
    post_processing: list[PostProcessingStep] | None = Field(
        None,
        description="[REDACTED_BY_SCRIPT]"
    )

    @model_validator(mode='after')
    def check_attribute_name_logic(self) -> 'ExtractionBlueprint':
        if self.extraction_method in ["ATTRIBUTE", "MULTIPLE_ATTRIBUTE"]:
            if not self.attribute_name:
                raise ValueError("'attribute_name'[REDACTED_BY_SCRIPT]")
        elif self.attribute_name:
            raise ValueError("'attribute_name'[REDACTED_BY_SCRIPT]")
        return self

# --- Modified Processing Contracts ---

class PaginationBlueprint(BaseModel):
    """[REDACTED_BY_SCRIPT]"""
    next_page_selector: str = Field(description="[REDACTED_BY_SCRIPT]'Next' button or 'Page 2' link).")

class BatchProcessingBlueprint(BaseModel):
    """[REDACTED_BY_SCRIPT]"""
    checkbox_selector: str = Field(description="[REDACTED_BY_SCRIPT]")
    select_all_selector: str | None = Field(None, description="Selector for the 'Select All' header checkbox.")
    batch_action_button_selector: str = Field(description="Selector for the 'Download Selected' or 'Archive' button.")

class ResultsPageProcessing(BaseModel):
    """[REDACTED_BY_SCRIPT]"""
    listing_container_selector: str = Field(description="[REDACTED_BY_SCRIPT]")
    
    listing_link_blueprint: ExtractionBlueprint = Field(
        description="[REDACTED_BY_SCRIPT]"
    )
    listing_address_blueprint: ExtractionBlueprint = Field(
        description="[REDACTED_BY_SCRIPT]"
    )
    # --- PROJECT ATLAS DIRECTIVE: ADD PAGINATION SUPPORT ---
    pagination_blueprint: PaginationBlueprint | None = Field(
        None,
        description="[REDACTED_BY_SCRIPT]"
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
    """[REDACTED_BY_SCRIPT]"""
    document_tab_selector: str | None = Field(None, description="CSS selector for the 'Documents'[REDACTED_BY_SCRIPT]")
    document_container_selector: str = Field(description="[REDACTED_BY_SCRIPT]")
    document_link_blueprint: ExtractionBlueprint = Field(description="[REDACTED_BY_SCRIPT]")
    document_date_blueprint: ExtractionBlueprint | None = Field(None, description="[REDACTED_BY_SCRIPT]")
    document_type_blueprint: ExtractionBlueprint | None = Field(None, description="[REDACTED_BY_SCRIPT]'Decision Notice').")
    pagination_blueprint: PaginationBlueprint | None = Field(None, description="[REDACTED_BY_SCRIPT]")
    batch_processing: BatchProcessingBlueprint | None = Field(None, description="[REDACTED_BY_SCRIPT]'Select All' -> 'Download').")

class ExecutionPlan(BaseModel):
    """[REDACTED_BY_SCRIPT]"""
    domain: str
    plan_version: str = "3.0" # Major version bump for Document Harvesting
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    instructions: list[Instruction]
    results_page_processing: ResultsPageProcessing
    detail_page_processing: DetailPageProcessing
    document_page_processing: DocumentPageProcessing | None = None