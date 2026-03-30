# ARCHITECTURAL DIRECTIVE: AMARYLLIS-AD-003
**Date:** 2026-02-05
**Status:** IMPLEMENTED
**Target:** `cognitive_profiler` (Strategist & Orchestrator)

## 1. Objective
Strengthen the Cognitive Profiler's ability to navigate complex "Document Access" patterns on LPA websites, specifically addressing the "Interstitial Link" trap where document lists are not immediately visible upon clicking a tab.

## 2. The Implementation Plan

### Phase 1: The Strategist (Brain)
*   [x] **Action:** Updated `_DETAIL_PAGE_PROMPT_TEMPLATE` in `strategist_agent.py`.
    *   *Logic:* Added `document_interstitial_link_selector` to the output schema.
    *   *Prompt:* Explicitly instructed the LLM to look for "View Associated Documents" buttons/links if the tab doesn't reveal a table.

### Phase 2: The Orchestrator (Body)
*   [x] **Action:** Updated Phase 4.1 (Runtime Logic) in `orchestrator.py`.
    *   *Logic:* After clicking the Documents tab, the system now checks for `interstitial_selector`. If present, it executes a secondary navigation step.
*   [x] **Action:** Updated Phase 5 (Synthesis Logic) in `orchestrator.py`.
    *   *Logic:* The `ExecutionPlan` (`final_instructions`) now records this extra step, ensuring the `UPLC Actor` can reproduce the journey.

## 3. Scenarios & Risk Register (The "Other Scenarios")

The user asked: *"What other scenarios might it encounter?"* 
We have identified the following high-probability edge cases for future hardening:

### A. The "New Window" Trap
*   **Scenario:** Clicking "View Documents" opens a popup window or a new tab (`target="_blank"`).
*   **Risk:** The agent might continue waiting on the *original* tab for a table that never appears.
*   **Mitigation:** The `ValidationAgent` handles new pages, but the `Strategist` prompt should explicitly detect `target="_blank"` and set a flag (e.g., `"opens_new_window": true`) so the Orchestrator knows to switch context.

### B. The "Disclaimer Wall" (Secondary Gate)
*   **Scenario:** Accessing the documents page triggers a *second* Copyright/Terms of Use banner, distinct from the one at the start of the session.
*   **Risk:** The agent sees the banner, not the expected table, and fails validation.
*   **Mitigation:** The "Law of Pre-emption" (Rule 2) needs to be applied to the Document Page transition as well.

### C. The "External Legacy" Split
*   **Scenario:** The Planning Portal says: *"For applications before 2010, please visit our [Archive System]."*
*   **Risk:** The documents are on a completely different domain (e.g., `microfiche.council.gov.uk`).
*   **Mitigation:** The prompt needs to detect "Archive" links and classify them as valid document sources.

### D. The "Master PDF" Anomaly
*   **Scenario:** There is no list of documents. There is just one massive PDF link labeled "Combined Scanned Files".
*   **Risk:** The `document_container_selector` (expecting a list/table) will fail.
*   **Mitigation:** The `DocumentPageProcessing` schema needs a fallback for `single_file_download`.

### E. The "Accordion Nesting"
*   **Scenario:** Documents are grouped by type (Drawings, Forms, Correspondence) in foldable accordions.
*   **Risk:** The agent sees the headers but not the files.
*   **Mitigation:** The `Strategist` must identify if *recursive expansion* is needed (Click Tab -> Click "Drawings" -> Scrape).

## 4. Verification Protocol
1.  **Test Case:** `Adur & Worthing` (Idox system with Interstitial Link).
2.  **Expected Log:**
    ```text
    INFO: Found Documents Tab: #tab_documents. Navigating...
    INFO: Interstitial Link Detected: a.view-docs. Navigating deeper...
    INFO: Generating Document Blueprints...
    ```
3.  **Artifact:** The resulting `ExecutionPlan` JSON should contain the extra `CLICK_ELEMENT` instruction.
