import re

with open('[REDACTED_BY_SCRIPT]', 'r', encoding='utf-8') as f:
    c = f.read()

# I will find the end of `__init__` and insert the new `_resolve_locator` method.
init_end_str = "[REDACTED_BY_SCRIPT]"
if init_end_str in c:
    new_method = """self.logger = logging.getLogger(self.__class__.__name__)

    async def _resolve_locator(self, container, selector: str, frame_name: str | None = None) -> Locator:
        \"\"\"
        Universal locator resolver that natively pierces iframes and respects frame_name if provided.
        Returns a robust Playwright locator.
        \"\"\"
        # 1. Explicit Frame Addressing
        if frame_name:
            # Playwright page.frame_locator
            return container.frame_locator(f"iframe[name='{frame_name}'], frame[name='{frame_name}']").locator(selector).first
            
        # 2. Universal Frame Piercing (AcolNet/Legacy)
        # Check main page quickly
        try:
            target = container.locator(selector).first
            return target
            # wait, returning target doesn't 'throw' if it's not visible, but wait_for does.
        except Exception:
            pass
            
        return container.locator(selector).first"""
    
    # Wait, the universal frame piercing needs to actually resolve.
    # It's better to just search page.frames if container is a Page object.
    # Let's write a better _resolve_locator

