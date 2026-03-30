# Wisteria Pipeline: Data Contracts

## 1.0 Data Collector -> Cognitive Profiler Contract

This document defines the mandatory data contract between the upstream "Data Collector" actor and this "Cognitive Profiler" actor.

### 1.1 Data Storage Architecture

The Data Collector **MUST** use its default storage locations for different data types:

-   **HTML Content:** All scraped HTML content **MUST** be saved to the actor's **Default Dataset**.
-   **Screenshots:** All captured screenshots **MUST** be saved to the actor's **Default Key-Value Store (KVS)**.

### 1.2 Dataset Item Structure (for HTML)

Each item pushed to the Dataset **MUST** be a JSON object with the following keys:
-   `url` (string): The full, canonical URL of the scraped page.
-   `html` (string): The complete HTML content of the page.

### 1.3 Key-Value Store Record Structure (for Screenshots)

-   The **Record Key** **MUST** be a sanitized, filesystem-safe representation of the page's URL (e.g., a slug). This key must be derivable from the `url` field in the Dataset to allow for correlation.
-   The **Record Value** **MUST** be the raw, binary content of the screenshot (e.g., JPEG or PNG).
-   The **Record Content-Type** **MUST** be set to the appropriate image mime type (e.g., `image/jpeg`).