import sys, re

def patch_validation_agent():
    path = "[REDACTED_BY_SCRIPT]"
    with open(path, "r", encoding="utf-8") as f:
        c = f.read()

    # 1. Inject _resolve_locator after __init__
    init_end = "[REDACTED_BY_SCRIPT]"
    resolver_code = """self.logger = logging.getLogger(self.__class__.__name__)

    async def _resolve_locator(self, page: Page, params: dict, state: str = 'visible', timeout_ms: int = None):
        \"\"\"[REDACTED_BY_SCRIPT]"\"\"
        selector = params.get('selector')
        frame_name = params.get('frame_name')
        if timeout_ms is None: timeout_ms = self.default_timeout_ms
        if not selector: raise ValidationError("Instruction requires 'selector'")

        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        import asyncio

        if frame_name:
            target = page.frame_locator(f"iframe[name='{frame_name}'], frame[name='{frame_name}']").locator(selector).first
            await target.wait_for(state=state, timeout=timeout_ms)
            return target

        async def try_loc(loc, t):
            try:
                await loc.wait_for(state=state, timeout=t)
                return loc
            except Exception:
                return None

        # 1. Fast main page
        target = page.locator(selector).first
        res = await try_loc(target, 2000)
        if res: return res

        # 2. Check frames
        tasks = []
        for child_frame in page.frames:
            if child_frame != page.main_frame:
                tasks.append(try_loc(child_frame.locator(selector).first, 3000))
        if tasks:
            results = await asyncio.gather(*tasks)
            for r in results:
                if r: return r

        # 3. Final wait to cleanly throw
        await target.wait_for(state=state, timeout=max(1000, timeout_ms - 2000))
        return target"""
    
    if "[REDACTED_BY_SCRIPT]" not in c:
        c = c.replace(init_end, resolver_code)

    # 2. Update _execute_fill_input
    fill_old = """    async def _execute_fill_input(self, params: dict[str, Any], page: Page):
        \"\"\"[REDACTED_BY_SCRIPT]"\"\"
        selector = params.get('selector')
        value = params.get('value')
        if not selector or value is None:
            raise ValidationError("Instruction 'FILL_INPUT' requires 'selector' and 'value'.")

        try:
            target = page.locator(selector).first
            await target.wait_for(state='visible', timeout=self.default_timeout_ms)
            await target.fill(value)
        except Exception as e:
             raise ValidationError(f"Failed to fill input '{selector}'. Error: {e}")"""
    
    fill_new = """    async def _execute_fill_input(self, params: dict[str, Any], page: Page):
        \"\"\"[REDACTED_BY_SCRIPT]"\"\"
        value = params.get('value')
        if value is None:
            raise ValidationError("Instruction 'FILL_INPUT' requires 'value'.")

        try:
            target = await self._resolve_locator(page, params)
            await target.fill(value)
        except Exception as e:
             raise ValidationError(f"Failed to fill input '{params.get('selector')}'. Error: {e}")"""
    
    c = c.replace(fill_old, fill_new)

    # Write changes
    with open(path, "w", encoding="utf-8") as f:
        f.write(c)
        print("[REDACTED_BY_SCRIPT]")

patch_validation_agent()
