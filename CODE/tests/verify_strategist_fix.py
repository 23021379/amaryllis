import asyncio
import logging
from unittest.mock import MagicMock
from cognitive_profiler.agents.strategist_agent import StrategistAgent
from cognitive_profiler.services.gemini_service import GeminiService

# Mock Logger
logging.basicConfig(level=logging.INFO)

async def test_prompt_hardening():
    print("[REDACTED_BY_SCRIPT]")
    
    mock_gemini = MagicMock(spec=GeminiService)
    agent = StrategistAgent(gemini_service=mock_gemini)
    
    # Render the probe template
    prompt = agent.probe_prompt_template.render(
        domain="example.com",
        target_address="123",
        homepage_html="<body></body>",
        searchpage_html="<body></body>"
    )
    
    print("[REDACTED_BY_SCRIPT]")
    if "[REDACTED_BY_SCRIPT]" in prompt:
        print("[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")
        exit(1)

    print("Checking for 'Site Search' prohibition...")
    if "interact with \"Site Search\"" in prompt:
        print("[REDACTED_BY_SCRIPT]")
    else:
        print("[REDACTED_BY_SCRIPT]")
        exit(1)

if __name__ == "__main__":
    asyncio.run(test_prompt_hardening())
