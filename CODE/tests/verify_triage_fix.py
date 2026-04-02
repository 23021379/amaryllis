import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock
from cognitive_profiler.agents.triage_agent import TriageAgent, DownloadedArtifact 
from cognitive_profiler.services.gemini_service import GeminiService
from cognitive_profiler.data_contracts import HtmlPageMetadata

# Mock Logger
logging.basicConfig(level=logging.INFO)

async def test_landing_page_logic():
    print("[REDACTED_BY_SCRIPT]")
    
    # Mock Gemini Service
    mock_gemini = MagicMock(spec=GeminiService)
    # Return a JSON where one URL is a LANDING_PAGE
    mock_gemini.generate_content = AsyncMock(return_value='''
    ```json
    {
        "[REDACTED_BY_SCRIPT]": "LANDING_PAGE",
        "[REDACTED_BY_SCRIPT]": "IRRELEVANT"
    }
    ```
    ''')

    agent = TriageAgent(gemini_service=mock_gemini, min_required_candidates=1)
    
    # Dummy Artifacts
    artifacts = [
        DownloadedArtifact(url="[REDACTED_BY_SCRIPT]", html_path="dummy.html", screenshot_path=None),
        DownloadedArtifact(url="[REDACTED_BY_SCRIPT]", html_path="dummy.html", screenshot_path=None)
    ]
    
    # Directly monkeypatch the instance method
    agent._extract_metadata_batch = AsyncMock(return_value=[
        HtmlPageMetadata(url="[REDACTED_BY_SCRIPT]", path="d", title="Plan", description="Desc", screenshot_path=None, structural_summary={}),
        HtmlPageMetadata(url="[REDACTED_BY_SCRIPT]", path="d", title="Cont", description="Desc", screenshot_path=None, structural_summary={})
    ])

    result = await agent.run(artifacts)
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    assert "[REDACTED_BY_SCRIPT]" in result.candidate_urls
    assert result.full_classification["[REDACTED_BY_SCRIPT]"] == "LANDING_PAGE"
    print("[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    asyncio.run(test_landing_page_logic())
