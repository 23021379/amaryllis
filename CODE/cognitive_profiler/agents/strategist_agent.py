import json
import logging
import re
from pathlib import Path
from typing import final, Any, List
from urllib.parse import urlparse

import jinja2
from PIL import Image
from jsonschema import validate, ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from ..data_contracts import TriageResult, DraftPlan, ValidationReport, ProbeResult, DetailPageProcessing
from ..exceptions import StrategyError
from ..command_schema import COMMAND_SCHEMA
from ..services.gemini_service import GeminiService

# --- JUDGEMENT DIRECTIVE: These prompts are the final, stable versions. ---

_PROBE_PLAN_PROMPT_TEMPLATE = """
You are an expert web automation strategist. Your mission is to create a "Universal Path" to find a Planning Application on a Council website given its Reference ID.

### VISUAL CONTEXT & HISTORY ###
I have attached a gallery of images to this request:
1. **BREADCRUMBS:** Screenshots of the pages we successfully navigated to get here. Use these to ground yourself in the user journey.
2. **FAILURE CONTEXT (If applicable):** Screenshots of previous failed attempts *on this specific page*. Look at them. Why did they fail? Was there an error message?
3. **CURRENT VIEW:** The last image is the current page state.

### THE LAW (ABSOLUTE & NON-NEGOTIABLE) ###
You must adhere to every rule in this section. Your entire response will be programmatically validated against these rules.

#### Rule 1: The Law of Non-Repetition (NEW AND CRITICAL)
1.1. Review the "FAILURE CONTEXT" images and the "PREVIOUS ERROR LOGS" below.
1.2. You **MUST NOT** reuse the failed selectors or submission strategies from those attempts.
1.3. If a previous plan failed on a CLICK, switch to PRESS_KEY "Enter".

#### Rule 2: The Law of Pre-emption (The Modal Defense)
2.1. **Universal Applicability:** This rule applies after **EVERY** `GOTO_URL` or `WAIT_FOR_NAVIGATION` command.
2.2. **Visual Check:** Look at the screenshot. Is there a "Cookie Notice", "Terms of Use", **"Copyright Notice" (Idox)**, or "Welcome" modal blocking the main content?
2.3. **The Mandate:** You **MUST** dismiss these overlays before attempting to `FILL_INPUT`.
2.4. **Execution:** Insert a `CLICK_ELEMENT` command targeting the "Accept", "Agree", "Continue", or "Close" button.
2.5. **Robust Selectors:**
    - Standard: `button:has-text("Accept")`, `button:has-text("Agree")`, `button:has-text("Continue")`
    - **Idox Disclaimer Gate:** `a:has-text("Accept")`, `a:has-text("Agree")`, `input[value="Accept"]` (Often found at the bottom of a text wall).
    - **NI Planning Portal Specific:** The button is blue and labeled **"Continue"**. Use `button.btn-primary:has-text("Continue")` or just `button:has-text("Continue")`. (Do NOT click "Necessary" - that is a checkbox).
    - Cookiebot (Critical): `#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll`, `#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowVariant`, `a#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll`
2.6. **CRITICAL:** Do NOT use `WAIT_FOR_SELECTOR` for these banners. Just `CLICK_ELEMENT`.

#### Rule 3: The Law of Disclosure (Accordion Logic)
3.1. Look at the Search Form. Is the input field currently visible?
3.2. If the input is hidden inside an accordion, collapsed section, or tab (indicated by text like "Search by Reference", "Ref Number", or icons like `+`, `v`, or arrows), you **MUST** click that header first.
3.3. **Sequence:** `CLICK_ELEMENT` (Header) -> `WAIT_FOR_SELECTOR` (Input visible) -> `FILL_INPUT`.
3.4. **Amber Valley/Northgate Specific:** Look for "Ref number" or "Search options" text that acts as a toggle.

#### Rule 4: Mission Directives (Formerly Rule 2)
Each command MUST have a `params` object with these exact keys. There are no exceptions.
- `GOTO_URL`: `params` MUST contain `url`.
- `FILL_INPUT`: `params` MUST contain `selector` and `value`. 
    - **CRITICAL MANDATE:** The `value` for the primary search input (e.g., location, address, postcode, reference number) **MUST ALWAYS AND ONLY BE** the literal string `{{REFERENCE_ID}}`.
    - **This is a template placeholder for the next robot, not a value to be replaced by you.** Your final JSON output must contain this exact placeholder string.
    - **A CRITICAL FAILURE IS to use the example address '{{target_address}}' or '{{TARGET_ADDRESS}}' in your output.** You MUST use the placeholder `{{REFERENCE_ID}}`.
- `CLICK_ELEMENT`: `params` MUST contain `selector`.
- `TRY_CLICK_ELEMENT`: `params` MUST contain `selector`. Optional: `timeout` (in ms, e.g., 3000). Use this instead of `CLICK_ELEMENT` for transitional elements (like search result links) that might be bypassed by direct redirects (e.g., Idox portals skipping the results page).
- `CHECK_OPTION`: `params` MUST contain `selector`.
- `PRESS_KEY`: `params` MUST contain `selector` and `key`.
- `WAIT_FOR_SELECTOR`: `params` MUST contain `selector`. Optional: `state` ("visible", "hidden", "attached", "detached"), `timeout` (in ms, e.g., 3000). Use `state: "detached"` to wait for cookie banners to disappear. For transitional elements that might not appear, use a short timeout like 3000.
- `WAIT_FOR_NAVIGATION`: `params` MUST be an empty object (`{}`).

#### Rule 5: Command Jurisdiction (The Pantheon Accords) (Formerly Rule 3)
- **`CHECK_OPTION`** is **EXCLUSIVELY** for `<input type="radio">` and `<input type="checkbox">`.
- **`CLICK_ELEMENT`** is for all other clickable elements that are guaranteed to be present.
- **`TRY_CLICK_ELEMENT`** is for transitional clickable elements that might be skipped by redirection.
- **LAW OF EXCLUSIVITY**: You **MUST NOT** use `CLICK_ELEMENT` or `TRY_CLICK_ELEMENT` on a radio button or checkbox.

#### Rule 6: Strategic Efficiency (Formerly Rule 4)
- Action commands have built-in waits. **DO NOT** precede them with a redundant `WAIT_FOR_SELECTOR`.

#### Rule 7: The Law of Literal Selectors (NEW)
- All CSS selectors MUST be syntactically valid.
- You **MUST NOT** use regular expressions inside selectors (e.g., `:has-text(/Accept/i)` is ILLEGAL).
- Use standard CSS and Playwright `:has-text("text")` pseudo-classes ONLY.

#### Rule 8: The Law of Implicit Submission (NEW)
- Web forms can be submitted in two ways. You must choose the correct one based on the visual evidence.
- **Explicit Submission:** If a clear, visible "Search", "Apply", or "Submit" button is present, your plan should be `FILL_INPUT` followed by `CLICK_ELEMENT` on that button.
- **Implicit Submission:** If no such button is visible or obvious, your plan **MUST** be `FILL_INPUT` followed by `PRESS_KEY` on the *same input field*, with the `key` parameter set to "Enter". Do not invent a button selector if you cannot see one.
- **Icon Buttons:** If the search button is an icon (e.g., magnifying glass) with no text, verify if it has an `aria-label` or `title` containing "Search". Do NOT rely on `:has-text("Search")` for these. If unsure, prefer Implicit Submission.

#### Rule 9: The Law of Transit (The Intermediary Page Clause)
- **Visual Scan:** Look at the screenshots. Are you actually on a Search Form, or a "Landing Page" describing the service?
- **The Mandate:** If you do not see a text input field for the Reference Number, but you DO see a link or button saying "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", or "Public Access", you **MUST** click that link first.
- **Forbidden Act:** Do NOT hallucinate an input field (e.g., `#searchCriteria_application_number`) if the page is just a CMS article with a link to the real system.
- **Sequence:** `CLICK_ELEMENT` (The "Go to Search" link) -> `WAIT_FOR_NAVIGATION` -> `FILL_INPUT` (The actual input on the next page).

#### Rule 10: The Law of Stable Selectors (The Anti-Hallucination Protocol)
- You **MUST NOT** use IDs that look generated or dynamic (e.g., `id="c_3919..."`, `id="guid-..."`, `id="ember..."`, `id="view123..."`).
- **Idox Trap:** Stop heavily relying on generic Idox IDs like `table#simpleDetailsTable` or `#searchresults`. The portals frequently modernize. Instead, favor more resilient locator strategies.
- **Alternative:** Prefer broader semantic attributes like `.get_by_text()`, `has-text`, input `name` attributes, or generic tags (e.g., `button:has-text('Search')` rather than `#searchButton`).
- **Civica Warning:** IDs starting with `view` (e.g., `#view136`) are strictly forbidden. They change on every reload.

#### Rule 11: The Law of Negative Constraints (The "Site Search" Trap)
11.1. You **MUST NOT** interact with "Site Search", "General Search", or "Website Search" inputs.
11.2. **The CMS Distinction:** If you are on a Council CMS landing page (e.g., WordPress/Jadu/Drupal) and see a prominent "Site Search" bar AND a link saying "[REDACTED_BY_SCRIPT]", "Search Applications", or "Public Access", you **MUST** click the link.
    - **Why:** The header search indexes news articles, NOT the planning database. Using it causes the "News Article Trap."
11.3. **DOM Location Constraint:** You **MUST NOT** select an input located inside a `<header>`, `#top`, or `.site-header` container. You **MUST** select the input inside the `<main>`, `#content`, or `article` area.
11.4. **Visual Constraint:** If the input is in the top-right corner of the page, it is almost certainly the Site Search. IGNORE IT.
11.5. You are looking for "Planning Search", "Application Search", or inputs specifically labelled "Reference Number", "Application Number", "Keyword".

#### Rule 12: The Law of Wrong Neighborhoods (The Legacy System Clause)
12.1. **Symptom:** The failure report indicates "[REDACTED_BY_SCRIPT]" or "No Results Found".
12.2. **Analysis:** Read the text *around* the warning box. Does it say "[REDACTED_BY_SCRIPT]" or "[REDACTED_BY_SCRIPT]"?
12.3. **The Fix:** Your corrected plan **MUST** click the link associated with that warning (e.g., "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]").
12.4. **Application:** Do NOT retry the search term in the same box. You are in the wrong database. Move to the correct one.

#### Rule 13: The Law of Realism (Anti-Hallucination) (NEW)
13.1. **Forbidden:** You **MUST NOT** use placeholder URLs like `example.com`, `mycouncil.gov.uk`, or `council.url`. You **MUST NOT** use template variables prefixed with `$` such as `$search_page_url`.
13.2. **Source of Truth:** Only use actual, fully resolved absolute URLs derived from the **Live Context** (links you see in the HTML). The ONLY permitted template variable in the entire JSON is `{{REFERENCE_ID}}`.
13.3. **The Law of Inertia:** If the `search_page_url` provided in the prompt matches the page you need to be on, **DO NOT** issue a `GOTO_URL` command. Start immediately with `FILL_INPUT`.

#### Rule 14: The Law of Incremental Navigation (NEW)
14.1. **Situation:** You are on a "Landing Page", "Disclaimer", or "Menu" and you can see a link to the search (e.g., "[REDACTED_BY_SCRIPT]"), but NOT the input field itself.
14.2. **The Mandate:** You cannot search yet. You must navigate closer.
14.3. **Output:** Set `"is_navigation_only": true` in your JSON response. The `instructions` should strictly navigate to the next page.
14.4. **Situation:** You see the "Reference Number" or "Keyword" input field.
14.5. **Output:** Set `"is_navigation_only": false`. This is the final search plan.

#### Rule 15: The Law of Jurisdiction (The Portal Trap) (CRITICAL FIX)
15.1. **The Trap:** On `planningsystemni.gov.uk`, the header contains a "Planning Portal" logo. Clicking this is FATAL (redirects to `nidirect.gov.uk` or `planningportal.co.uk`).
15.2. **The Mandate:** You **MUST NOT** interact with the Header, Logo, or Footer. You are already at the destination.
15.3. **NI Exception (Terminal Node):** `planningsystemni.gov.uk` IS the correct final destination.
    - **Status:** You have arrived. STOP navigating.
    - **Action:** You **MUST** target the central search input immediately.
    - **Selectors:** `#simpleSearchString`, `input[name='search']`.
    - **Forbidden:** Do NOT click `.navbar-brand`, `img[alt='Planning Portal']`, or any element in `<header>`.

#### Rule 16: The Law of Legacy Frames (The AcolNet/Northgate Ghost)
16.1. **Diagnosis:** If the page has a static left-hand navigation menu or seems old (like 'AcolNetCGI'), it is likely a legacy system using `<frameset>`.
16.2. **The Architecture:** The visible form is often inside a child frame (e.g., `name="content"`, `name="main"`, or `name="AcolNetCGI"`).
16.3. **The Mandate:** The executor engine now automatically pierces frames. Your job is ONLY to provide the robust CSS selector targeting the input field directly (e.g., `#txtApplicationNumber`, `input[name='casefullref']`).
16.4. **The Trap:** Do not assume the page has loaded just because the navigation bar is visible. 
16.5. **Action:** Feel free to provide the `frame_name` parameter to the Instruction object if you are certain of the frame's name attribute.

#### Rule 17: The Law of Vertical Forms (The Civica Solver) (NEW)
17.1. **Situation:** The form has text labels visually *above* the input fields (common in Civica/Ashfield systems).
17.2. **The Trap:** Using direct sibling combinators (e.g., `label + input`, `label + div input`) often fails because the DOM has hidden wrapper `<div>`s you cannot guess.
17.3. **The Silver Bullet:** Use the **Container Strategy** with `:has()`.
    - **Pattern:** `div:has(label:has-text("Reference")) input`
    - **Logic:** "[REDACTED_BY_SCRIPT]'Reference'[REDACTED_BY_SCRIPT]"
    - **Why:** This works regardless of how many `div`s are nested between the label and the input.
17.4. **Priority:**
    1. `input[aria-label*="Reference"]` (If accessible)
    2. `div:has(label:has-text("Reference")) input` (The Vertical Form Solver)
    3. `input[name*="Reference"]` (If stable)


#### Rule 18: The Law of Post-Search Boundaries (CRITICAL STOP CONDITION)
18.1. Your task is ENTIRELY finished the exact moment the search is submitted and the page begins loading the results.
18.2. You **MUST NOT** include instructions to click on search results in a list.
18.3. You **MUST NOT** include instructions to click on a 'Documents' tab. If the portal requires clicking a documents tab later, the orchestrator handles that natively via `document_tab_selector`.
18.4. You **MUST NOT** try to download files.
18.5. If the last command is submitting the search form (via `CLICK_ELEMENT` or `PRESS_KEY`), the ONLY valid instruction to follow it is a final `WAIT_FOR_SELECTOR`. DO NOTHING ELSE.

### OUTPUT FORMAT (MANDATORY) ###
You **MUST** respond with a single JSON object with THREE keys: `search_term`, `instructions`, and `is_navigation_only`.

### EXAMPLE OF A PERFECT MODERN RESPONSE ###
```json
{
  "search_term": "Newcastle Upon Tyne",
  "is_navigation_only": false,
  "instructions": [
    {
      "command": "GOTO_URL",
      "params": { "url": "[REDACTED_BY_SCRIPT]" },
      "description": "[REDACTED_BY_SCRIPT]"
    },
    {
      "command": "FILL_INPUT",
      "params": { "selector": "#ps--searchbox", "value": "{{REFERENCE_ID}}" },
      "description": "[REDACTED_BY_SCRIPT]"
    },
    {
      "command": "CLICK_ELEMENT",
      "params": { "selector": "[REDACTED_BY_SCRIPT]'Apply')" },
      "description": "[REDACTED_BY_SCRIPT]"
    },
    {
      "command": "WAIT_FOR_SELECTOR",
      "params": { "selector": "[REDACTED_BY_SCRIPT]'Summary')" },
      "description": "[REDACTED_BY_SCRIPT]"
    }
  ]
}


1. **Pre-emption:** Look for "[REDACTED_BY_SCRIPT]" or "Copyright" pages. You MUST click "Accept" or "Agree" if they exist.
2. **Input:** The `value` for the search input (Reference Number/Application ID) **MUST ALWAYS AND ONLY BE** the literal string `{{REFERENCE_ID}}`.
3. **Outcome (If is_navigation_only=false):** The final instruction MUST be `WAIT_FOR_SELECTOR` to prove the search worked.
4. **The Outcome Validation Selector:** Do NOT use a hardcoded list of IDs. Some portals return a "results list", while others instantly forward you to the "Application Details" modal/page. You MUST construct a generic CSS selector that accounts for BOTH possibilities (using comma separation) based *solely* on the HTML provided above to capture whatever state appears after submission.

### OUTPUT FORMAT ###
Respond with a JSON object: `{"search_term": "Ref Search", "is_navigation_only": false, "instructions": [...]}`.
```

--- CONTEXTUAL HTML ---
{{ homepage_html }}
{% if searchpage_html %}
--- SEARCH PAGE HTML ---
{{ searchpage_html }}
{% endif %}
"""

_NAVIGATION_CORRECTION_PROMPT = """
You are an expert web automation strategist. The navigation plan you previously created has FAILED. Your task is to analyze the failure and create a new, corrected `instructions` array that follows THE LAW.

### PREVIOUS FAILED PLAN (DO NOT REPEAT) ###
```json
{{ report.original_plan.instructions | tojson(indent=2) }}
```

### FAILURE ANALYSIS ###
- **Error**: `{{ report.failure_reason }}`
{% if report.failure_analysis_notes %}
- **Architect's Analysis**: `{{ report.failure_analysis_notes }}`
{% endif %}

### VISUAL ANALYSIS (GROUND TRUTH) ###
A screenshot of the webpage at the exact moment of failure is attached. This image is the absolute ground truth.

### THE LAW (ABSOLUTE & NON-NEGOTIABLE) ###
(The law is identical to the generation phase. All rules apply.)

#### Rule 1: The Law of Non-Repetition (NEW AND CRITICAL)
1.1. You have been shown the "[REDACTED_BY_SCRIPT]".
1.2. You **MUST NOT** reuse the failed selectors or submission strategies from that plan.
1.3. If the previous plan failed while trying to `CLICK_ELEMENT`, you **MUST** now formulate a plan that uses a different strategy, such as `PRESS_KEY` with the "Enter" key on the search input. Do not try to find another button to click.

#### Rule 2: The Law of Pre-emption (The Modal Defense)
2.1. **Universal Applicability:** This rule applies after **EVERY** `GOTO_URL` or `WAIT_FOR_NAVIGATION` command.
2.2. **Cross-Domain Reset:** If the plan redirects to a new domain, you MUST check for a new banner.
2.3. **Visual Check:** Look at the "Failure Context" screenshot. Is there a "Cookie Notice", "Terms of Use", **"Copyright Notice" (Idox)**, or "Welcome" modal?
2.4. **The Mandate:** Insert a `CLICK_ELEMENT` command targeting the "Accept", "Agree", "Continue" button BEFORE retrying the failed input.
2.5. **Robust Selectors:**
    - Standard: `button:has-text("Accept")`, `button:has-text("Agree")`, `button:has-text("Continue")`
    - **Idox Disclaimer Gate:** `a:has-text("Accept")`, `a:has-text("Agree")`, `input[value="Accept"]`.
    - **NI Planning Portal Specific:** `button.btn-primary:has-text("Continue")`, `button:has-text("Continue")`.
    - Cookiebot: `#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll`

#### Rule 3: The Law of Disclosure (Accordion/Toggle Logic)
3.1. **Visual Scan:** Look at the search form area. Is the input field for the Reference ID visible?
3.2. **The "Plain Text" Trap:** Many councils (e.g., Amber Valley, Northgate systems) hide inputs behind text labels that don't look like buttons.
    - Look for text like: "Ref number", "Search by reference", "Search Options".
    - Look for icons: `v`, `+`, `▼`, or arrows next to text.
3.3. **The Mandate:** If the input is not clearly visible, you **MUST** click the text label/header first.
3.4. **Execution Sequence:** 
    1. `CLICK_ELEMENT` (The Header/Text Label) e.g., `div:has-text("Ref number")`, `a:has-text("Search options")`.
    2. `WAIT_FOR_SELECTOR` (The Input Field) - Verify it appeared.
    3. `FILL_INPUT` (The Input Field).

#### Rule 4: Mission Directives (Formerly Rule 2)
4.1. The plan MUST focus on the "FOR SALE" user journey.
4.2. The final instruction **MUST** be `WAIT_FOR_SELECTOR` to prove the search worked.
4.3. **Hybrid Outcome Strategy:** The search might yield a list of results OR redirect directly to a single Application Details page. Your final selector MUST account for both possibilities.
4.4. **The Outcome Validation Selector:** Do NOT use a hardcoded list of IDs. You MUST construct a generic CSS selector that accounts for BOTH possibilities (using comma separation) based *solely* on the HTML provided above to capture whatever state appears after submission.


#### Rule 5: Command Parameter Manifest (Formerly Rule 3)
- `GOTO_URL`: `params` MUST contain `url`.
- `FILL_INPUT`: `params` MUST contain `selector` and `value`. The value MUST be "{{REFERENCE_ID}}".
- `CLICK_ELEMENT`: `params` MUST contain `selector`.
- `TRY_CLICK_ELEMENT`: `params` MUST contain `selector`. Optional: `timeout` (in ms, e.g., 3000). Use this instead of `CLICK_ELEMENT` for transitional elements (like search result links) that might be bypassed by direct redirects (e.g., Idox portals skipping the results page).
- `CHECK_OPTION`: `params` MUST contain `selector`.
- `PRESS_KEY`: `params` MUST contain `selector` and `key`.
- `WAIT_FOR_SELECTOR`: `params` MUST contain `selector`. Optional: `state` ("visible", "hidden", "attached", "detached"), `timeout` (in ms, e.g., 3000). Use `state: "detached"` to wait for obstructions to clear. For transitional elements that might not appear, use a short timeout like 3000.
- `WAIT_FOR_NAVIGATION`: `params` MUST be an empty object (`{}`).

#### Rule 6: Command Jurisdiction (The Pantheon Accords) (Formerly Rule 4)
- **`CHECK_OPTION`** is **EXCLUSIVELY** for `<input type="radio">` and `<input type="checkbox">`.
- **`CLICK_ELEMENT`** is for all other clickable elements that are guaranteed to be present.
- **`TRY_CLICK_ELEMENT`** is for transitional clickable elements that might be skipped by redirection.
- **LAW OF EXCLUSIVITY**: You **MUST NOT** use `CLICK_ELEMENT` or `TRY_CLICK_ELEMENT` on a radio button or checkbox.

#### Rule 7: The Law of Literal Selectors (Formerly Rule 5)
7.1. All CSS selectors MUST be syntactically valid.
7.2. You **MUST NOT** use regular expressions inside selectors (e.g., `:has-text(/Accept/i)` is ILLEGAL).
7.3. Use standard CSS and Playwright `:has-text("text")` pseudo-classes ONLY.

#### Rule 8: The Law of Implicit Submission (Formerly Rule 6)
8.1. Web forms can be submitted in two ways. You must choose the correct one based on the visual evidence.
8.2. **Explicit Submission:** If a clear, visible "Search", "Apply", or "Submit" button is present, your plan should be `FILL_INPUT` followed by `CLICK_ELEMENT` on that button.
8.3. **Implicit Submission:** If no such button is visible or obvious, your plan **MUST** be `FILL_INPUT` followed by `PRESS_KEY` on the *same input field*, with the `key` parameter set to "Enter". Do not invent a button selector if you cannot see one.
8.4. **Icon Buttons:** If the search button is an icon (e.g., magnifying glass) with no text, verify if it has an `aria-label` or `title` containing "Search". Do NOT rely on `:has-text("Search")` for these. If unsure, prefer Implicit Submission.

#### Rule 9: The Law of Transit (The Intermediary Page Clause)
9.1. **Visual Scan:** Look at the screenshot. Are you actually on a Search Form, or a "Landing Page"?
9.2. **The Mandate:** If the previous error was a TIMEOUT looking for an input, and you see a link like "[REDACTED_BY_SCRIPT]" or "Public Access", you **MUST** click that link.
9.3. **Forbidden Act:** Do NOT retry the same input selector if it failed to find anything. The page is likely an intermediary. Navigate through it.

#### Rule 10: The Law of Stable Selectors (The Anti-Hallucination Protocol)
- You **MUST NOT** use IDs that look generated or dynamic (e.g., `id="c_3919..."`, `id="guid-..."`, `id="ember..."`, `id="view123..."`).
- **Idox Trap:** Stop heavily relying on generic Idox IDs like `table#simpleDetailsTable` or `#searchresults`. Favor `.get_by_text()`, `.get_by_role()`, or broader structural attributes when rewriting failed plans.
- **Civica Warning:** IDs starting with `view` (e.g., `#view136`) are strictly forbidden. They change on every reload.

#### Rule 11: The Law of the "Other" Search (The Dual-Portal Trap)
11.1. **Symptom:** The failure report indicates "[REDACTED_BY_SCRIPT]" (No Results), BUT the page text contains phrases like "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]", or "Public Access".
11.2. **Analysis:** The council has TWO systems. The main site has a "Site Search" (which you tried) and a hidden "Planning Search".
11.3. **The Fix:** Your corrected plan **MUST** click the link/button that explicitly says "Planning Applications", "[REDACTED_BY_SCRIPT]", or "Public Access".
11.4. **Forbidden Act:** Do NOT retry the search term in the main search box. You are in the wrong database.

#### Rule 12: The Law of Wrong Neighborhoods (The Legacy System Clause)
12.1. **Symptom:** The failure report indicates "[REDACTED_BY_SCRIPT]" or "No Results Found".
12.2. **Analysis:** Read the text *around* the warning box. Does it say "[REDACTED_BY_SCRIPT]" or "[REDACTED_BY_SCRIPT]"?
12.3. **The Fix:** Your corrected plan **MUST** click the link associated with that warning (e.g., "[REDACTED_BY_SCRIPT]", "[REDACTED_BY_SCRIPT]").
12.4. **Application:** Do NOT retry the search term in the same box. You are in the wrong database. Move to the correct one.

#### Rule 13: The Law of Realism (Anti-Hallucination) (NEW)
13.1. **Forbidden:** You **MUST NOT** use placeholder URLs like `example.com`, `mycouncil.gov.uk`, or `council.url`. You **MUST NOT** use template variables prefixed with `$` such as `$search_page_url`.
13.2. **Source of Truth:** Only use actual, fully resolved absolute URLs derived from the **Live Context** (links you see in the HTML). The ONLY permitted template variable in the entire JSON is `{{REFERENCE_ID}}`.
13.3. **The Law of Inertia:** If the `search_page_url` provided in the prompt matches the page you need to be on, **DO NOT** issue a `GOTO_URL` command. Start immediately with `FILL_INPUT`.

#### Rule 14: The Law of Incremental Navigation (NEW)
14.1. **Situation:** You are on a "Landing Page", "Disclaimer", or "Menu" and you can see a link to the search (e.g., "[REDACTED_BY_SCRIPT]"), but NOT the input field itself.
14.2. **The Mandate:** You cannot search yet. You must navigate closer.
14.3. **Output:** Set `"is_navigation_only": true` in your JSON response. The `instructions` should strictly navigate to the next page.
14.4. **Situation:** You see the "Reference Number" or "Keyword" input field.
14.5. **Output:** Set `"is_navigation_only": false`. This is the final search plan.

#### Rule 15: The Law of Jurisdiction (The Portal Trap) (CRITICAL FIX)
15.1. **The Trap:** On `planningsystemni.gov.uk`, the header contains a "Planning Portal" logo. Clicking this is FATAL (redirects to `nidirect.gov.uk` or `planningportal.co.uk`).
15.2. **The Mandate:** You **MUST NOT** interact with the Header, Logo, or Footer. You are already at the destination.
15.3. **NI Exception (Terminal Node):** `planningsystemni.gov.uk` IS the correct final destination.
    - **Status:** You have arrived. STOP navigating.
    - **Action:** You **MUST** target the central search input immediately.
    - **Selectors:** `#simpleSearchString`, `input[name='search']`.
    - **Forbidden:** Do NOT click `.navbar-brand`, `img[alt='Planning Portal']`, or any element in `<header>`.

#### Rule 16: The Law of Legacy Frames (The AcolNet/Northgate Ghost)
16.1. **Diagnosis:** If the page has a static left-hand navigation menu or seems old (like 'AcolNetCGI'), it is likely a legacy system using `<frameset>`.
16.2. **The Architecture:** The visible form is often inside a child frame (e.g., `name="content"`, `name="main"`, or `name="AcolNetCGI"`).
16.3. **The Mandate:** The executor engine now automatically pierces frames. Your job is ONLY to provide the robust CSS selector targeting the input field directly (e.g., `#txtApplicationNumber`, `input[name='casefullref']`).
16.4. **The Trap:** Do not assume the page has loaded just because the navigation bar is visible. 
16.5. **Action:** Feel free to provide the `frame_name` parameter to the Instruction object if you are certain of the frame's name attribute.

#### Rule 17: The Law of Vertical Forms (The Civica Solver) (NEW)
17.1. **Situation:** The form has text labels visually *above* the input fields (common in Civica/Ashfield systems).
17.2. **The Trap:** Using direct sibling combinators (e.g., `label + input`, `label + div input`) often fails because the DOM has hidden wrapper `<div>`s you cannot guess.
17.3. **The Silver Bullet:** Use the **Container Strategy** with `:has()`.
    - **Pattern:** `div:has(label:has-text("Reference")) input`
    - **Logic:** "[REDACTED_BY_SCRIPT]'Reference'[REDACTED_BY_SCRIPT]"
    - **Why:** This works regardless of how many `div`s are nested between the label and the input.
17.4. **Priority:**
    1. `input[aria-label*="Reference"]` (If accessible)
    2. `div:has(label:has-text("Reference")) input` (The Vertical Form Solver)
    3. `input[name*="Reference"]` (If stable)


#### Rule 18: The Law of Post-Search Boundaries (CRITICAL STOP CONDITION)
18.1. Your task is ENTIRELY finished the exact moment the search is submitted and the page begins loading the results.
18.2. You **MUST NOT** include instructions to click on search results in a list.
18.3. You **MUST NOT** include instructions to click on a 'Documents' tab. If the portal requires clicking a documents tab later, the orchestrator handles that natively via `document_tab_selector`.
18.4. You **MUST NOT** try to download files.
18.5. If the last command is submitting the search form (via `CLICK_ELEMENT` or `PRESS_KEY`), the ONLY valid instruction to follow it is a final `WAIT_FOR_SELECTOR`. DO NOTHING ELSE.


### YOUR TASK ###
Analyze the failure, the screenshot, the Architect'[REDACTED_BY_SCRIPT]"Enter".
Respond ONLY with a single JSON object containing one key: `instructions`.

--- HTML AT POINT OF FAILURE ---
{{ report.failure_context_html }}
"""


_ADAPTIVE_RESULTS_PROMPT_TEMPLATE = """
You are an expert web scraping analyst for UK Planning Portals (Idox, Northgate, Agile). 
Your task is to analyze the **Search Results** page to create blueprints for extracting the list of Planning Applications.

### VISUAL CUES
Look for a list or table containing:
1.  **Reference Numbers** (e.g., "22/01234/FUL", "121281")
2.  **Addresses**
3.  **Descriptions** (e.g., "[REDACTED_BY_SCRIPT]")

### MISSION
Analyze the provided HTML and Screenshot to produce:
1.  A `results_page_selectors` blueprint to extract data from the results list.
2.  A `final_wait_instruction` to confirm the results have loaded.

### DATA MAPPING (CRITICAL)
- `listing_container_selector`: The CSS selector for the repeating item. 
    - **IDOX Hint:** Often `#searchresults > li` or `.searchresult`.
    - **Northgate Hint:** Often `table.data > tr`.
- `listing_link_blueprint`: The selector for the `<a>` tag linking to the details.
- `listing_address_blueprint`: The selector for the text containing the **Reference Number**.
- **SYNTAX RULE:** You **MUST NOT** use `:contains()`. It is not valid CSS. Use `:has-text("...")` instead.

### OMNIBUS CLAUSE (SINGLE RESULT DETECTION)
If you cannot find a list of results, you must check if the page is a **Single Application Summary** (e.g., has tabs like "Details", "Comments").
- **IF YES:** Return a valid selector for the "Summary" tab or the main heading as the `listing_container_selector` so the pipeline knows we are done.
- **IF NO (CRITICAL):** If the page looks like a **Search Form** (has input fields, "Search" button) or a **Homepage**, return `null` for `listing_container_selector`. Do NOT select the page title (e.g. "Search Applications") as a result container.

### OUTPUT SCHEMA
Respond with a single JSON object.

Example:
```json
{
  "final_wait_instruction": {
    "command": "WAIT_FOR_SELECTOR",
    "params": { "selector": "#searchresults" },
    "description": "Wait for results."
  },
  "results_page_selectors": {
    "[REDACTED_BY_SCRIPT]": "li.searchresult",
    "listing_link_blueprint": {
      "selector": "a",
      "extraction_method": "ATTRIBUTE",
      "attribute_name": "href"
    },
    "[REDACTED_BY_SCRIPT]": {
      "selector": "p.spec",
      "extraction_method": "TEXT"
    },
    "[REDACTED_BY_SCRIPT]": null
  }
}
```

--- LIVE RESULTS PAGE HTML ---
{{ captured_html }}
"""

_DETAIL_PAGE_PROMPT_TEMPLATE = """
You are an expert web scraping analyst for UK Planning Portals. Your task is to analyze the **Application Details** page.

### MISSION ###
1.  **Tab/Accordion Navigation:** Identify the selectors for the "Documents", "Comments", and "Constraints" sections.
    - **CRITICAL FOR ACCORDIONS:** If the data is inside a collapsible section (common in Idox/Northgate), you **MUST** select the **Visible Trigger** (Header, Link, or Button) that expands the section.
    - **NEVER** select the hidden content container itself (e.g., `div.collapse`, `div.panel-body`).
    - **CORRECT:** `a[href='#collapse-documents']`, `h3.panel-title`, `li.active > a`.
    - **INCORRECT:** `#collapse-documents` (This is the hidden div).

2.  **The Interstitial Trap (CRITICAL):**
    - Sometimes, clicking the "Documents" tab does not show the list immediately. Instead, it shows a button saying "[REDACTED_BY_SCRIPT]" or "[REDACTED_BY_SCRIPT]" (Idox/SwiftLG pattern).
    - If you see this pattern, you MUST provide the selector for that button as `document_interstitial_link_selector`.

3.  **Key Data:** Create extraction blueprints for:
    - `application_status`: The status text (e.g., "Decided", "Pending").
    - `decision`: The decision value (e.g., "Approved", "Refused").
    - `proposal`: The description of works.
    - `address`: The site address.

### OUTPUT SCHEMA ###
Respond with a JSON object. Keys MUST be snake_case.

Example:
```json
{
  "tabs": {
    "[REDACTED_BY_SCRIPT]": "a[href='#documents']",
    "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
    "comments_tab_selector": "#tab_consultations",
    "constraints_tab_selector": "#tab_constraints"
  },

  "data_blueprints": {
    "application_status": { "selector": "span.status", "extraction_method": "TEXT" },
    "decision": { "selector": "#decision_text", "extraction_method": "TEXT" },
    "proposal": { "selector": ".description", "extraction_method": "TEXT" }
  }
}
```

--- HTML OF THE APPLICATION PAGE ---
{{ html_content }}
"""


_RESULTS_CORRECTION_PROMPT = """
You are an expert web scraping analyst. The `results_page_processing` blueprint you created previously FAILED VALIDATION. It could not correctly identify elements on the results page.

### MISSION ###
- Analyze the provided screenshot and HTML of the live property search results page.
- Create a new, corrected `results_page_processing` JSON object.

### BLUEPRINT SCHEMA ###
```json
{
  "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
  "listing_link_blueprint": {
    "selector": "a.property-link",
    "extraction_method": "ATTRIBUTE",
    "attribute_name": "href"
  },
  "[REDACTED_BY_SCRIPT]": {
    "selector": "h2.property-address",
    "extraction_method": "TEXT"
  }
}
```

### YOUR TASK ###
Respond ONLY with the corrected `results_page_processing` JSON object.

--- HTML OF RESULTS PAGE (AT POINT OF FAILURE) ---
{{ report.failure_context_html }}
"""

_DETAIL_CORRECTION_PROMPT = """
You are an expert web data extraction analyst. The `detail_page_processing` blueprints you created FAILED VALIDATION on a live property detail page.

### MISSION ###
- Analyze the provided screenshot and HTML of the property detail page.
- DISCOVER all available, high-value data points (e.g., Price, Address, Bedrooms, etc.).
- Generate a new, complete, and robust `detail_page_processing` JSON object.
- The keys of the object MUST be snake_cased. The values MUST be valid `ExtractionBlueprint` objects.

### BLUEPRINT SCHEMA ###
```json
{
  "selector": "css-selector-for-the-element",
  "extraction_method": "TEXT",
  "attribute_name": null
}
```

### YOUR TASK & OUTPUT FORMAT ###
Respond ONLY with a single JSON object that is the dictionary of the blueprints themselves. DO NOT wrap it in a parent key.

**CORRECT Response Format:**
```json
{
  "address": { "...blueprint..." },
  "price": { "...blueprint..." }
}
```

**INCORRECT Response Format (DO NOT DO THIS):**
```json
{
  "[REDACTED_BY_SCRIPT]": {
    "address": { "...blueprint..." },
    "price": { "...blueprint..." }
  }
}
```

--- HTML OF DETAIL PAGE (AT POINT OF FAILURE) ---
{{ report.failure_context_html }}
"""
@final
class StrategistAgent:
    """[REDACTED_BY_SCRIPT]"""

    def __init__(self, gemini_service: GeminiService):
        self.gemini_service = gemini_service
        self.jinja_env = jinja2.Environment(autoescape=True)
        self.jinja_env.filters['tojson'] = lambda data, indent=2: json.dumps(data, indent=indent)
        
        # Templates
        self.probe_prompt_template = self.jinja_env.from_string(_PROBE_PLAN_PROMPT_TEMPLATE)
        self.adaptive_prompt_template = self.jinja_env.from_string(_ADAPTIVE_RESULTS_PROMPT_TEMPLATE)
        self.detail_prompt_template = self.jinja_env.from_string(_DETAIL_PAGE_PROMPT_TEMPLATE)
        self.navigation_correction_template = self.jinja_env.from_string(_NAVIGATION_CORRECTION_PROMPT)
        self.results_correction_template = self.jinja_env.from_string(_RESULTS_CORRECTION_PROMPT)
        self.detail_correction_template = self.jinja_env.from_string(_DETAIL_CORRECTION_PROMPT)
        
        # --- DOCUMENT HARVESTING PROMPT ---
        self._document_scraping_template = self.jinja_env.from_string("""
        You are an expert Data Engineer specializing in tabular data extraction. 
        You are looking at the **Documents Tab** of a Planning Application.

        ### MISSION
        Create a precise, battle-hardened JSON blueprint to extract every document.

        ### VISUAL ANALYSIS & EDGE CASE DEFENSE
        1. **The Container Trap:** 
           - You MUST identify the repeating element (e.g., `table#documents tbody tr`).
           - **CRITICAL:** You MUST exclude the Header Row (`thead tr`) and Pagination Row from your selector. 
           - *Bad Selector:* `tr` (Selects headers and footers).
           - *Good Selector:* `table.docs > tbody > tr` or `div.document-row`.

        2. **The Link Ambiguity:**
           - A row may handle multiple links ("View", "Measure", "Comment").
           - You MUST select the link that actually opens the file.
           - *Heuristic:* Look for buttons with text "View", "Download", icons of PDFs, or `href` attributes ending in `.pdf`.
           - *Forbidden:* Do not select "Measure" or "Zoom" tools.

        3. **Batch Handling (The "Select All" Paradox):**
           - **Row Checkboxes:** Must be inside the `document_container_selector`. Target the specific `<input type="checkbox">`.
           - **Header Checkbox:** The "Select All" checkbox is usually in the table header, OUTSIDE the document container.
           - **Download Button:** Often labeled "Download Selected", "Apply" (next to a filter action), or represented by a "Briefcase" icon.

        4. **Pagination (The Hidden List):**
           - Look for "Next", ">", ">>", or "Next Page".
           - Preferred selector targets the "Next" button specifically (e.g., `.pager .next`).

        ### DATA MAPPING RULES
        - `document_link_blueprint`: This URL serves as the unique ID for the document. It must be unique per row.
        - `document_date_blueprint`: Extract the text date (e.g., "12/05/2023").
        - `document_type_blueprint`: Extract the description/type (e.g., "Decision Notice").
        - **FORBIDDEN FALLBACK:** You MUST NOT use `"body"` or `""` as a fallback selector. If you cannot find a valid row container, the extraction fails. Do not guess.

        ### OUTPUT SCHEMA (STRICT JSON)
        ```json
        {
          "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
          "[REDACTED_BY_SCRIPT]": { "selector": "[REDACTED_BY_SCRIPT]", "extraction_method": "ATTRIBUTE", "attribute_name": "href" },
          "[REDACTED_BY_SCRIPT]": { "selector": "td:nth-child(4)", "extraction_method": "TEXT" },
          "[REDACTED_BY_SCRIPT]": { "selector": "td:nth-child(2)", "extraction_method": "TEXT" },
          "[REDACTED_BY_SCRIPT]": { "next_page_selector": ".pager a.next" },
          "batch_processing": {
            "checkbox_selector": "td.select-col input", 
            "select_all_selector": "[REDACTED_BY_SCRIPT]",
            "[REDACTED_BY_SCRIPT]": "input#btnDownload"
          }
        }
        ```
        *Note: Set `pagination_blueprint` or `batch_processing` to null if controls are not visible.*

        --- DOCUMENT LIST HTML ---
        {{ html_content }}
        """)

        # New Template for Global Search Fallback
        self._global_search_template = self.jinja_env.from_string("""
        You are a Web Navigation Specialist. You are currently on the Homepage of a Local Council.

        ### THE MISSION
        1. Locate the main site search input (often at the top right, or behind a magnifying glass icon).
        2. Create a plan to type "view planning" into this box.
        3. Submit the search (Click Search or Press Enter).

        ### THE LAW (ABSOLUTE)
        1. **Cookie Pre-emption:** If there is a cookie banner, dismiss it first.
        2. **Input Value:** The value for `FILL_INPUT` MUST be exactly "view planning".
        3. **Outcome:** The final command MUST be `WAIT_FOR_NAVIGATION` (as search usually reloads the page).
        4. **Selector Syntax:** You **MUST NOT** use `:contains()`. It is illegal. Use `:has-text("...")` instead.
        5. **The Law of Stable Selectors:** You **MUST NOT** use IDs that look generated or dynamic (e.g., `id="c_3919..."`, `id="guid-..."`, `id="ember..."`). Instead, use stable attributes like `name`, `placeholder`, `aria-label`, or combine `form` classes with tags (e.g., `form.search-form input`).
        6. **The Law of Implicit Submission:** If the search button is an icon (e.g., magnifying glass) or ambiguous, your plan **MUST** use `PRESS_KEY` with "Enter" on the input field instead of `CLICK_ELEMENT`. This is much more robust and avoids layout obstruction issues.

        ### COMMAND SCHEMA (STRICT)
        Each instruction must match this format EXACTLY:
        - `GOTO_URL`: `{"command": "GOTO_URL", "params": {"url": "..."}}`
        - `FILL_INPUT`: `{"command": "FILL_INPUT", "params": {"selector": "...", "value": "..."}}`
        - `CLICK_ELEMENT`: `{"command": "CLICK_ELEMENT", "params": {"selector": "..."}}`
        - `PRESS_KEY`: `{"command": "PRESS_KEY", "params": {"selector": "...", "key": "Enter"}}`
        - `WAIT_FOR_SELECTOR`: `{"command": "WAIT_FOR_SELECTOR", "params": {"selector": "...", "state": "visible"}}`
        - `WAIT_FOR_NAVIGATION`: `{"command": "WAIT_FOR_NAVIGATION", "params": {}}`

        ### OUTPUT SCHEMA
        Respond with a JSON object: `{"instructions": [...]}`.

        --- HOMEPAGE HTML ---
        {{ homepage_html }}
        """)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.command_schema_json = json.dumps(COMMAND_SCHEMA, indent=2)

    async def generate_global_search_plan(self, homepage_url: str, homepage_html: str) -> DraftPlan:
        """Generates a plan to use the site's internal search engine."""
        self.logger.info(f"[REDACTED_BY_SCRIPT]")
        
        prompt = self._global_search_template.render(homepage_html=homepage_html)
        response_text = await self.gemini_service.generate_content(prompt)
        plan_data = self._parse_llm_json_response(response_text)
        
        if 'instructions' not in plan_data:
             raise StrategyError("[REDACTED_BY_SCRIPT]'instructions'.")

        instructions = plan_data['instructions']

        # SYSTEM OVERRIDE: Enforce GOTO_URL as Step 0.
        # The ValidationAgent spawns a fresh page (about:blank). We cannot trust the AI to remember this.
        if not instructions or instructions[0].get("command") != "GOTO_URL":
            self.logger.warning("[REDACTED_BY_SCRIPT]")
            instructions.insert(0, {
                "command": "GOTO_URL",
                "params": {"url": homepage_url},
                "description": "[REDACTED_BY_SCRIPT]"
            })
        
        # Verify the URL matches if it WAS provided (edge case check)
        elif instructions[0].get("command") == "GOTO_URL":
             # Force alignment to the trusted artifact URL
             instructions[0]["params"]["url"] = homepage_url

        self._validate_instructions_against_schema(instructions, context="[REDACTED_BY_SCRIPT]")

        # Return a DraftPlan (some fields are dummy placeholders as we are in discovery mode)
        return DraftPlan(
            instructions=instructions,
            source_triage_result=None, # Not applicable
            target_address="view planning",
            target_description="[REDACTED_BY_SCRIPT]"
        )

    async def generate_planning_search_plan(
        self, 
        search_page_url: str, 
        search_page_html: str, 
        target_ref_id: str,
        breadcrumbs: list[str] = [],
        previous_failures: list[ValidationReport] = []
    ) -> DraftPlan:
        """
        Phase 2: Generates the plan with full multimodal context (Breadcrumbs + Failure History).
        """
        self.logger.info(f"[REDACTED_BY_SCRIPT]")

        # 1. Compile Error Logs from previous failures
        failure_text = ""
        if previous_failures:
            failure_text = "[REDACTED_BY_SCRIPT]"
            for i, f in enumerate(previous_failures):
                failure_text += f"[REDACTED_BY_SCRIPT]"

        # 2. Render Prompt
        prompt = self.probe_prompt_template.render(
            domain=urlparse(search_page_url).netloc,
            target_address=target_ref_id,
            homepage_html="", 
            searchpage_html=search_page_html + failure_text # Inject text logs
        )

        # 3. Load Images (Breadcrumbs + Failure Screenshots)
        context_images = []
        
        # A. Breadcrumbs (Success History)
        for path in breadcrumbs:
            try:
                if path and Path(path).exists():
                    context_images.append(Image.open(path))
            except Exception as e:
                self.logger.warning(f"[REDACTED_BY_SCRIPT]")

        # B. Failures (Error Context)
        for f in previous_failures:
            try:
                if f.failure_screenshot_path and Path(f.failure_screenshot_path).exists():
                    context_images.append(Image.open(f.failure_screenshot_path))
            except Exception as e:
                self.logger.warning(f"[REDACTED_BY_SCRIPT]")

        # Execute
        response_text = await self.gemini_service.generate_content(prompt, images=context_images)
        plan_data = self._parse_llm_json_response(response_text)

        if 'instructions' not in plan_data:
             raise StrategyError("[REDACTED_BY_SCRIPT]'instructions'.")

        instructions = plan_data['instructions']
        is_navigation_only = plan_data.get('is_navigation_only', False)

        # Ensure that at least one FILL_INPUT contains the reference ID if it's the final search page
        if not is_navigation_only:
            has_reference_input = False
            has_any_fill = False
            for instr in instructions:
                if instr.get("command") == "FILL_INPUT":
                    has_any_fill = True
                    if instr.get("params", {}).get("value") == "{{REFERENCE_ID}}":
                        has_reference_input = True
            
            if not has_reference_input:
                if has_any_fill:
                    self.logger.warning("[REDACTED_BY_SCRIPT]'{{REFERENCE_ID}}'[REDACTED_BY_SCRIPT]")
                    for instr in instructions:
                        if instr.get("command") == "FILL_INPUT":
                            instr["params"]["value"] = "{{REFERENCE_ID}}"
                            break
                else:
                    self.logger.warning("[REDACTED_BY_SCRIPT]")
                    is_navigation_only = True

        # Anti-Hallucination: Verify no GOTO_URL violates domain boundaries
        search_domain = urlparse(search_page_url).netloc
        for instr in instructions:
            if instr.get("command") == "GOTO_URL":
                target_url = instr.get("params", {}).get("url", "")
                target_domain = urlparse(target_url).netloc
                if target_domain and target_domain != search_domain and not target_domain.endswith(".idox.com") and not target_domain.endswith("[REDACTED_BY_SCRIPT]") and "acolnetcgi" not in target_url.lower():
                    raise StrategyError(f"[REDACTED_BY_SCRIPT]")

        self._validate_instructions_against_schema(instructions, context="[REDACTED_BY_SCRIPT]")

        return DraftPlan(
            instructions=instructions,
            source_triage_result=None, 
            target_address=target_ref_id,
            target_description=f"[REDACTED_BY_SCRIPT]",
            is_navigation_only=is_navigation_only
        )

    def _diagnose_timeout_failure(self, html_content: str) -> tuple[str, str | None]:
        """[REDACTED_BY_SCRIPT]"""
        # REFINED KEYWORDS: Removed 'privacy', 'cookie', 'agree' to prevent false positives from footers.
        OBSTRUCTION_KEYWORDS = [
            'cookiebot', 'onetrust', 'termly', 'civic-cookie', 'consent-banner', 
            'modal-dialog', 'overlay', 'gdpr-banner', 'cookie-notice'
        ]
        
        html_lower = html_content.lower()
        found_keywords = [kw for kw in OBSTRUCTION_KEYWORDS if kw in html_lower]

        if found_keywords:
            notes = (
                "[REDACTED_BY_SCRIPT]"
                f"({', '[REDACTED_BY_SCRIPT]"
                "[REDACTED_BY_SCRIPT]'Accept All', 'I Agree')."
            )
            self.logger.warning("[REDACTED_BY_SCRIPT]")
            return "[REDACTED_BY_SCRIPT]", notes
        else:
            self.logger.info("[REDACTED_BY_SCRIPT]")
            return "TIMEOUT", None

            
    def _validate_instructions_against_schema(self, instructions: list[dict[str, Any]], context: str):
        """
        Validates a list of instructions against the canonical COMMAND_SCHEMA.
        Raises StrategyError on failure.
        
        Args:
            instructions: The list of instruction dictionaries to validate.
            context: A string describing the operation (e.g., "probe plan generation") for clear error logging.
        """
        try:
            validate(instance=instructions, schema=COMMAND_SCHEMA)
            self.logger.info(f"Plan for '{context}'[REDACTED_BY_SCRIPT]")
        except JsonSchemaValidationError as e:
            self.logger.error(f"[REDACTED_BY_SCRIPT]'{context}'[REDACTED_BY_SCRIPT]")
            # This is a non-recoverable failure of the AI's core contract.
            raise StrategyError(f"Generated plan for '{context}'[REDACTED_BY_SCRIPT]")


    async def generate_probe_plan(self, triage_result: TriageResult, target_address: str, target_description: str) -> DraftPlan:
        self.logger.info("[REDACTED_BY_SCRIPT]")
        cat_to_url = {v: k for k, v in triage_result.full_classification.items()}
        homepage_url = cat_to_url.get("HOMEPAGE")
        searchpage_url = cat_to_url.get("SEARCH_PAGE")

        if not homepage_url:
            raise StrategyError("No 'HOMEPAGE'[REDACTED_BY_SCRIPT]")

        homepage_html = await self._get_contextual_html(homepage_url, triage_result)
        searchpage_html = await self._get_contextual_html(searchpage_url, triage_result) if searchpage_url else None

        prompt = self.probe_prompt_template.render(
            domain=triage_result.domain,
            target_address=target_address,
            homepage_html=homepage_html,
            searchpage_html=searchpage_html
        )
        response_text = await self.gemini_service.generate_content(prompt)
        plan_data = self._parse_llm_json_response(response_text)

        if 'instructions' not in plan_data:
             raise StrategyError("[REDACTED_BY_SCRIPT]'instructions'.")

        instructions = plan_data['instructions']

        # --- PROJECT GENESIS DIRECTIVE: The Programmatic Judge ---
        # We can no longer trust the AI. We must programmatically enforce the placeholder.
        # This Judge ensures that the plan is reusable and not overfitted to the test data.
        placeholder_found_in_plan = False
        for instruction in instructions:
            if instruction.get("command") == "FILL_INPUT":
                value = instruction.get("params", {}).get("value", "")
                
                # Heuristic: if the AI fills an input with the test reference ID, auto-correct it.
                if value and value.strip() == target_address.strip(): # target_address holds the ref_id in this context
                     self.logger.warning(f"[REDACTED_BY_SCRIPT]'{value}'[REDACTED_BY_SCRIPT]")
                     instruction["params"]["value"] = "{{REFERENCE_ID}}"

                if instruction["params"].get("value") == "{{REFERENCE_ID}}":
                    placeholder_found_in_plan = True

        # Final check: A valid probe plan MUST have at least one placeholder.
        if not placeholder_found_in_plan:
            raise StrategyError("[REDACTED_BY_SCRIPT]")
        
        self._validate_instructions_against_schema(instructions, context="probe plan generation")

        return DraftPlan(
            instructions=instructions,
            source_triage_result=triage_result,
            target_address=target_address,
            target_description=target_description
        )

    async def correct_navigation_plan(self, report: ValidationReport) -> list[dict]:
        self.logger.info("[REDACTED_BY_SCRIPT]")
        data = await self._run_base_correction(report, self.navigation_correction_template)
        
        if 'instructions' not in data:
            raise StrategyError("[REDACTED_BY_SCRIPT]'instructions' key.")
        
        # JUDGEMENT PROTOCOL: Verify before returning.
        instructions = data['instructions']
        self._validate_instructions_against_schema(instructions, context="[REDACTED_BY_SCRIPT]")
        return instructions

    # --- Methods below this line are considered stable ---

    async def _run_base_correction(self, report: ValidationReport, template: jinja2.Template) -> dict:
        image = None
        if report.failure_screenshot_path:
            try:
                image = Image.open(report.failure_screenshot_path)
            except Exception as e:
                self.logger.warning(f"[REDACTED_BY_SCRIPT]")
        
        prompt = template.render(report=report)
        # FIX: Updated argument name to 'images' to match GeminiService signature
        response_text = await self.gemini_service.generate_content(prompt, images=image)
        return self._parse_llm_json_response(response_text)

    async def generate_adaptive_components(self, probe_result: ProbeResult) -> tuple[dict, list[dict]]:
        self.logger.info("[REDACTED_BY_SCRIPT]")
        if not probe_result.captured_screenshot_path or not probe_result.captured_html:
            raise StrategyError("[REDACTED_BY_SCRIPT]")

        image = Image.open(probe_result.captured_screenshot_path)
        
        prompt = self.adaptive_prompt_template.render(
            captured_html=probe_result.captured_html
        )

        # FIX: Updated argument name to 'images'
        response_text = await self.gemini_service.generate_content(prompt, images=image)
        adaptive_data = self._parse_llm_json_response(response_text)

        if 'final_wait_instruction' not in adaptive_data or 'results_page_selectors' not in adaptive_data:
            raise StrategyError("[REDACTED_BY_SCRIPT]")
        
        final_instruction = [adaptive_data['final_wait_instruction']]
        results_processing = adaptive_data['results_page_selectors']

        # CRITICAL CHECK: Ensure we actually got a dict, not None
        if not results_processing:
             raise StrategyError("AI returned 'null'[REDACTED_BY_SCRIPT]")
        
        return results_processing, final_instruction
    
    async def get_detail_page_selectors(self, triage_result: TriageResult) -> dict[str, Any]:
        self.logger.info("[REDACTED_BY_SCRIPT]")
        cat_to_url = {v: k for k, v in triage_result.full_classification.items()}
        
        detail_page_url = cat_to_url.get("DETAIL_PAGE")
        if not detail_page_url:
            self.logger.warning("No 'DETAIL_PAGE'[REDACTED_BY_SCRIPT]")
            return {}

        html_content = await self._get_contextual_html(detail_page_url, triage_result)
        prompt = self.detail_prompt_template.render(html_content=html_content)
        
        response_text = await self.gemini_service.generate_content(prompt)
        
        try:
            data = self._parse_llm_json_response(response_text)
            validated_blueprints = DetailPageProcessing.model_validate(data)
            self.logger.info(f"[REDACTED_BY_SCRIPT]")
            return validated_blueprints.model_dump()
        except Exception as e:
            raise StrategyError(f"[REDACTED_BY_SCRIPT]") from e

    async def generate_document_blueprints(self, html_content: str, image: Image.Image | None) -> dict[str, Any]:
        """[REDACTED_BY_SCRIPT]"""
        self.logger.info("[REDACTED_BY_SCRIPT]")
        prompt = self._document_scraping_template.render(html_content=html_content)
        
        # FIX: Updated argument name to 'images'
        response_text = await self.gemini_service.generate_content(prompt, images=image)
        data = self._parse_llm_json_response(response_text)
        
        # We return the raw dict here; validation happens in the orchestrator/data contract layer
        return data


    async def _get_contextual_html(self, url: str | None, triage_result: TriageResult) -> str:
        if not url: return ""
        url_to_path_map = {meta.url: meta.path for meta in triage_result.source_metadata}
        file_path_str = url_to_path_map.get(url)
        if not file_path_str:
            self.logger.warning(f"[REDACTED_BY_SCRIPT]")
            return ""
        
        self.logger.info(f"[REDACTED_BY_SCRIPT]")
        try:
            return Path(file_path_str).read_text(encoding='utf-8', errors='ignore')
        except FileNotFoundError:
            self.logger.error(f"[REDACTED_BY_SCRIPT]")
            return ""

    def _parse_llm_json_response(self, response_text: str) -> dict[str, Any]:
        self.logger.debug(f"[REDACTED_BY_SCRIPT]")
        try:
            match = re.search(r"[REDACTED_BY_SCRIPT]", response_text, re.DOTALL)
            json_text = match.group(1).strip() if match else response_text.strip()
            data = json.loads(json_text)
            
            # --- AUTO-CORRECTION: Handle array-only responses ---
            if isinstance(data, list) and all(isinstance(i, dict) and "command" in i for i in data):
                self.logger.warning("[REDACTED_BY_SCRIPT]'instructions': ...}")
                data = {"instructions": data}
                
            if not isinstance(data, dict):
                raise StrategyError(f"[REDACTED_BY_SCRIPT]")
            return data
        except json.JSONDecodeError as e:
            # If parsing fails, check if the text contains a refusal or error explanation
            lower_text = response_text.lower()
            if any(x in lower_text for x in ["unable to", "cannot provide", "not present", "valid json"]):
                self.logger.warning(f"[REDACTED_BY_SCRIPT]")
            raise StrategyError(f"[REDACTED_BY_SCRIPT]") from e
        except AttributeError as e:
            raise StrategyError(f"[REDACTED_BY_SCRIPT]") from e

    async def correct_results_page_blueprint(self, report: ValidationReport) -> dict:
        self.logger.info("[REDACTED_BY_SCRIPT]")
        return await self._run_base_correction(report, self.results_correction_template)

    async def correct_detail_page_blueprints(self, report: ValidationReport) -> dict:
        self.logger.info("[REDACTED_BY_SCRIPT]")
        return await self._run_base_correction(report, self.detail_correction_template)