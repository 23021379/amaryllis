
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
    print(f"[REDACTED_BY_SCRIPT]")

    try:
        # Try v1beta first as it is more likely to have newer models
        print("[REDACTED_BY_SCRIPT]")
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
        for m in client.models.list():
            if hasattr(m, 'supported_actions') and "generateContent" in m.supported_actions:
               print(f"[REDACTED_BY_SCRIPT]")
            elif not hasattr(m, 'supported_actions'):
                 print(f"[REDACTED_BY_SCRIPT]")

    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]")

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
    
    print(f"[REDACTED_BY_SCRIPT]")
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
                 print(f"[REDACTED_BY_SCRIPT]")
            elif "400" in err_str: # Bad request often means invalid model name for some APIs
                 print(f"[REDACTED_BY_SCRIPT]")
            else:
                 print(f"❓ ERROR: {e}")

if __name__ == "__main__":
    # 1. OPTIONAL: List all standard models
    list_models() 
    
    # 2. Test specific models (Edit this list to check for hidden/unlisted models)
    custom_models = [
        "gemini-2.5-flash",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]"
    ]
    test_specific_models(custom_models)
