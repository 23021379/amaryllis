import sys, re

def update_val_agent():
    path = "[REDACTED_BY_SCRIPT]"
    with open(path, "r", encoding="utf-8") as f:
        c = f.read()

    click_old = """    async def _execute_click_element(self, params: dict[str, Any], page: Page) -> Page:
        \"\"\"
        Finds a clickable element and executes a click.
        Detects if a new tab/window was opened and returns the new Page if so.
        \"\"\"
        selector = params.get('selector')
        if not selector:
            raise ValidationError("Instruction 'CLICK_ELEMENT'[REDACTED_BY_SCRIPT]'selector'")

        locator = page.locator(selector)
        visible_locator = locator.filter(visible=True)

        # Capture context state before click to detect new pages
        context = page.context

        try:
            target = visible_locator.first
            await target.wait_for(state='visible', timeout=self.default_timeout_ms)
            await target.click(timeout=self.default_timeout_ms)"""
    
    click_new = """    async def _execute_click_element(self, params: dict[str, Any], page: Page) -> Page:
        \"\"\"
        Finds a clickable element and executes a click.
        Detects if a new tab/window was opened and returns the new Page if so.
        \"\"\"
        selector = params.get('selector')
        if not selector:
            raise ValidationError("Instruction 'CLICK_ELEMENT'[REDACTED_BY_SCRIPT]'selector'")

        context = page.context

        try:
            target = await self._resolve_locator(page, params)
            await target.click(timeout=self.default_timeout_ms)"""

    c = c.replace(click_old, click_new)

    try_click_old = """    async def _execute_try_click_element(self, params: dict[str, Any], page: Page) -> Page:
        \"\"\"
        Attempts to click a transitional element (like a search result link in Idox).
        If it times out or fails (e.g. because we were already redirected directly to the details page),
        it cleanly catches the error and proceeds.
        \"\"\"
        selector = params.get('selector')
        timeout = params.get('timeout', 3000)

        if not selector:
            raise ValidationError("Instruction 'TRY_CLICK_ELEMENT'[REDACTED_BY_SCRIPT]'selector'")

        context = page.context

        try:
            target = page.locator(selector).first
            await target.wait_for(state='visible', timeout=timeout)
            await target.click(timeout=timeout)"""

    try_click_new = """    async def _execute_try_click_element(self, params: dict[str, Any], page: Page) -> Page:
        \"\"\"
        Attempts to click a transitional element (like a search result link in Idox).
        If it times out or fails (e.g. because we were already redirected directly to the details page),
        it cleanly catches the error and proceeds.
        \"\"\"
        selector = params.get('selector')
        timeout = params.get('timeout', 3000)

        if not selector:
            raise ValidationError("Instruction 'TRY_CLICK_ELEMENT'[REDACTED_BY_SCRIPT]'selector'")

        context = page.context

        try:
            target = await self._resolve_locator(page, params, timeout_ms=timeout)
            await target.click(timeout=timeout)"""

    c = c.replace(try_click_old, try_click_new)
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(c)

update_val_agent()
