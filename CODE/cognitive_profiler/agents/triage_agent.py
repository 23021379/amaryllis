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
- "LANDING_PAGE": A page describing the planning service with a link to the search system (e.g. "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]").
- "ADVANCED_SEARCH": A complex search form (Second priority).
- "IRRELEVANT": Building control, licencing, weekly lists, contact pages.

Analyze the metadata. Your goal is to find the **SEARCH_PAGE** or the **DISCLAIMER_PAGE** blocking it.

Respond ONLY with a single JSON object mapping URLs to categories.

Example Response:
{
  "[REDACTED_BY_SCRIPT]": "HOMEPAGE",
  "[REDACTED_BY_SCRIPT]": "SEARCH_PAGE"
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
        """[REDACTED_BY_SCRIPT]"""
        self.logger.info(f"[REDACTED_BY_SCRIPT]")
        if not artifact_manifest:
            raise TriageError("[REDACTED_BY_SCRIPT]")

        metadata_objects = await self._extract_metadata_batch(artifact_manifest)
        if not metadata_objects:
            raise TriageError("[REDACTED_BY_SCRIPT]")

        prompt = self._build_gemini_prompt(metadata_objects)
        response_text = await self.gemini_service.generate_content(prompt)
        classifications = self._parse_and_validate_response(response_text)
        
        selected_urls = self._select_candidate_urls(classifications)
        self._validate_candidate_set(selected_urls, classifications)

        url_to_path_map = {meta.url: meta.path for meta in metadata_objects}
        candidate_paths = {url: url_to_path_map[url] for url in selected_urls if url in url_to_path_map}
        
        # Mandate 4: Determine domain once and pass it forward.
        domain = urlparse(metadata_objects[0].url).netloc

        self.logger.info(f"[REDACTED_BY_SCRIPT]'{domain}'[REDACTED_BY_SCRIPT]")
        return TriageResult(
            domain=domain,
            candidate_urls=candidate_paths,
            full_classification=classifications,
            source_metadata=metadata_objects
        )

    async def _parse_single_html(self, artifact: DownloadedArtifact) -> HtmlPageMetadata | None:
        """[REDACTED_BY_SCRIPT]'s HTML for its metadata."""
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
            self.logger.warning(f"[REDACTED_BY_SCRIPT]")
            return None

    async def _extract_metadata_batch(self, artifact_manifest: list[DownloadedArtifact]) -> list[HtmlPageMetadata]:
        """[REDACTED_BY_SCRIPT]"""
        self.logger.info(f"[REDACTED_BY_SCRIPT]")
        tasks: list[Coroutine[Any, Any, HtmlPageMetadata | None]] = [self._parse_single_html(artifact) for artifact in artifact_manifest]
        results = await asyncio.gather(*tasks)
        
        valid_metadata = [res for res in results if res]
        self.logger.info(f"[REDACTED_BY_SCRIPT]")
        return valid_metadata

    def _build_gemini_prompt(self, metadata_objects: list[HtmlPageMetadata]) -> str:
        """[REDACTED_BY_SCRIPT]"""
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
        """[REDACTED_BY_SCRIPT]'s response."""
        self.logger.info("[REDACTED_BY_SCRIPT]")
        match = re.search(r"[REDACTED_BY_SCRIPT]", response_text, re.DOTALL)
        json_text = match.group(1) if match else response_text

        try:
            parsed_data = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise TriageError("[REDACTED_BY_SCRIPT]") from e

        if not isinstance(parsed_data, dict):
            raise TriageError("[REDACTED_BY_SCRIPT]")

        for category in parsed_data.values():
            if category not in _VALID_CATEGORIES:
                raise TriageError(f"[REDACTED_BY_SCRIPT]'{category}'.")
        
        self.logger.info(f"[REDACTED_BY_SCRIPT]")
        return parsed_data

    def _select_candidate_urls(self, classifications: dict[str, str]) -> list[str]:
        """[REDACTED_BY_SCRIPT]"""
        selected = []
        seen_urls = set()
        
        for category in _SELECTION_PRIORITY:
            for url, cat in classifications.items():
                if cat == category and url not in seen_urls:
                    selected.append(url)
                    seen_urls.add(url)
                    if len(selected) >= self.max_candidates:
                        self.logger.info(f"[REDACTED_BY_SCRIPT]")
                        return selected
        return selected

    def _validate_candidate_set(self, candidates: list[str], classifications: dict[str, str]):
        """[REDACTED_BY_SCRIPT]"""
        self.logger.info(f"[REDACTED_BY_SCRIPT]")
        
        if len(candidates) < self.min_required_candidates:
            self.logger.error(f"[REDACTED_BY_SCRIPT]")
            raise TriageError(f"[REDACTED_BY_SCRIPT]")
        
        candidate_categories = {classifications[url] for url in candidates if url in classifications}
        
        # LOGIC UPDATE: We accept success if we found the Target (Search/Disclaimer) OR the Homepage.
        # We do not strictly require the Homepage if we already have the Search Page.
        valid_entry_points = {"SEARCH_PAGE", "DISCLAIMER_PAGE", "LANDING_PAGE", "HOMEPAGE"}
        
        if not valid_entry_points.intersection(candidate_categories):
             raise TriageError(f"[REDACTED_BY_SCRIPT]")
        
        self.logger.info("[REDACTED_BY_SCRIPT]")