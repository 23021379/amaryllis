import asyncio
import logging
from google import genai
from google.genai import types
from PIL import Image
from .api_key_manager import SimpleApiKeyManager
from ..exceptions import PipelineError

class GeminiService:
    """[REDACTED_BY_SCRIPT]"""
    def __init__(self, api_key_manager: SimpleApiKeyManager, models: list[str]):
        self.api_key_manager = api_key_manager
        self.models = models
        self.current_model_index = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not self.models:
            raise ValueError("[REDACTED_BY_SCRIPT]")
        self.logger.info(f"[REDACTED_BY_SCRIPT]")

    def _get_current_model(self) -> str:
        return self.models[self.current_model_index]

    def _rotate_model(self):
        """[REDACTED_BY_SCRIPT]"""
        old = self._get_current_model()
        self.current_model_index = (self.current_model_index + 1) % len(self.models)
        new = self._get_current_model()
        self.logger.warning(f"[REDACTED_BY_SCRIPT]")

    async def generate_content(self, prompt: str, images: list[Image.Image] | Image.Image | None = None, max_retries: int = 3) -> str:
        """
        Generates content with strategies for both Quota (429) and Overload (503).
        Supports single image or list of images for context (Breadcrumbs/Failures).
        """
        contents = [prompt]
        
        if images:
            if isinstance(images, list):
                contents.extend(images)
                self.logger.info(f"[REDACTED_BY_SCRIPT]")
            else:
                contents.append(images)
                self.logger.info("[REDACTED_BY_SCRIPT]")

        total_keys = len(self.api_key_manager.keys)
        quota_retries = 0   # Counter for Key Rotation (429)
        generic_retries = 0 # Counter for Standard Failures (Timeout, etc)
        
        # We loop until success or exhaustion
        while True:
            current_model = self._get_current_model()
            api_key = self.api_key_manager.get_key()
            
            try:
                # self.logger.info(f"[REDACTED_BY_SCRIPT]")
                client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
                
                generation_task = asyncio.to_thread(
                    client.models.generate_content,
                    model=current_model,
                    contents=contents
                )
                
                # Check 2: Add timeout to prevent hanging
                response = await asyncio.wait_for(generation_task, timeout=120.0)
                
                return response.text

            except Exception as e:
                error_str = str(e).lower()
                
                # --- STRATEGY 1: QUOTA / RATE LIMIT (429) -> ROTATE KEY ---
                if any(x in error_str for x in ['exhausted', 'quota', 'limit', '429']):
                    quota_retries += 1
                    if quota_retries >= total_keys:
                        raise PipelineError(f"[REDACTED_BY_SCRIPT]") from e
                    
                    self.logger.warning(f"[REDACTED_BY_SCRIPT]")
                    self.api_key_manager.rotate_key()
                    continue

                # --- STRATEGY 2: SERVICE OVERLOAD (503) -> ROTATE MODEL ---
                elif '503' in error_str or 'overloaded' in error_str:
                    self.logger.warning(f"[REDACTED_BY_SCRIPT]")
                    self._rotate_model()
                    # We treat a model switch as a 'free' retry regarding generic counts, 
                    # but we pause briefly to let the system settle.
                    await asyncio.sleep(1)
                    continue

                # --- STRATEGY 3: GENERIC FAILURE -> EXPONENTIAL BACKOFF ---
                else:
                    generic_retries += 1
                    self.logger.error(f"[REDACTED_BY_SCRIPT]")
                    
                    if generic_retries >= max_retries:
                        raise PipelineError(f"[REDACTED_BY_SCRIPT]") from e
                    
                    await asyncio.sleep(2 ** generic_retries)