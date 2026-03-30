import sys

def update_file():
    with open('[REDACTED_BY_SCRIPT]', 'r', encoding='utf-8') as f:
        c = f.read()

    r16_old = """#### Rule 16: The Law of Legacy Frames (The Northgate Ghost)
16.1. **Diagnosis:** If the page has a static left-hand navigation menu and a central "Planning Search" form, it is likely a legacy Northgate/M3 system using `<frameset>`.
16.2. **The Architecture:** The visible form is often inside a child frame (e.g., `name="content"` or `name="main"`).
16.3. **The Mandate:** Your selectors MUST target the input field directly (e.g., `#txtApplicationNumber`) and be robust.
16.4. **The Trap:** Do not assume the page has loaded just because the navigation bar is visible. The central frame might be slow.
16.5. **Action:** If interaction fails, consider that the element might be encapsulated."""

    r16_new = """#### Rule 16: The Law of Legacy Frames (The AcolNet/Northgate Ghost)
16.1. **Diagnosis:** If the page has a static left-hand navigation menu or seems old (like 'AcolNetCGI'), it is likely a legacy system using `<frameset>`.
16.2. **The Architecture:** The visible form is often inside a child frame (e.g., `name="content"`, `name="main"`, or `name="AcolNetCGI"`).
16.3. **The Mandate:** The executor engine now automatically pierces frames. Your job is ONLY to provide the robust CSS selector targeting the input field directly (e.g., `#txtApplicationNumber`, `input[name='casefullref']`).
16.4. **The Trap:** Do not assume the page has loaded just because the navigation bar is visible. 
16.5. **Action:** Feel free to provide the `frame_name` parameter to the Instruction object if you are certain of the frame's name attribute."""

    c = c.replace(r16_old, r16_new)

    r18_old = """#### Rule 18: The Law of Post-Search Boundaries (CRITICAL STOP CONDITION)
18.1. Your task is ENTIRELY finished the exact moment the search is submitted and the page begins loading the results.
18.2. You **MUST NOT** include instructions to click on search results in a list.
18.3. You **MUST NOT** include instructions to click on a 'Documents' tab.
18.4. You **MUST NOT** try to download files.
18.5. If the last command is submitting the search form (via `CLICK_ELEMENT` or `PRESS_KEY`), the ONLY valid instruction to follow it is a final `WAIT_FOR_SELECTOR`. DO NOTHING ELSE."""

    r18_new = """#### Rule 18: The Law of Post-Search Boundaries (CRITICAL STOP CONDITION)
18.1. Your task is ENTIRELY finished the exact moment the search is submitted and the page begins loading the results.
18.2. You **MUST NOT** include instructions to click on search results in a list.
18.3. You **MUST NOT** include instructions to click on a 'Documents' tab. If the portal requires clicking a documents tab later, the orchestrator handles that natively via `document_tab_selector`.
18.4. You **MUST NOT** try to download files.
18.5. If the last command is submitting the search form (via `CLICK_ELEMENT` or `PRESS_KEY`), the ONLY valid instruction to follow it is a final `WAIT_FOR_SELECTOR`. DO NOTHING ELSE."""

    c = c.replace(r18_old, r18_new)

    with open('[REDACTED_BY_SCRIPT]', 'w', encoding='utf-8') as f:
        f.write(c)

update_file()
