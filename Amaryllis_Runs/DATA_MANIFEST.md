# AMARYLLIS DATA LAKE MANIFEST

## Storage Location
**Root Path:** `C:/Users/brand/Desktop/renewables` (Mounted in Workspace)

## 1. Grid Infrastructure (Technical & Network Capacity)
*   **Path:** `/national gird (middle)`
    *   *Files:* `distribution_substation_locations.csv`, `dfes_energy_projections.csv`, `network_opportunity_map_headroom.csv`
    *   *Contents:* High-level National Grid asset mapping and Future Energy Scenarios (DFES).
*   **Path:** `/ukpowernetworks (south east)`
    *   *Files:* `Embedded Capacity Register.geojson`, `Primary Site Transformers.geojson`, `Grid and Primary Sites.geojson`
    *   *Contents:* Raw distribution-level connectivity for the UKPN license area, including overhead line (OHL) voltages (132kV, 33kV, HV, LV).
*   **Path:** `/future dno`
    *   *Files:* Regional folders for Northern Powergrid (NPG), SSEN (South), and SP Energy (North West).
    *   *Contents:* Expansion data for multi-DNO model coverage.
*   **Path:** `/NG_data` & `/UKPN_data`
    *   *Files:* `L1_UKPN_Substation_Capacity.geoparquet`, `DNO_NG_L1_Primary_Substations_FaultEnriched.geoparquet`
    *   *Contents:* **Processed Artifacts.** Cleaned, topological joins of grid assets used as model inputs.

## 2. Environmental & Land Use (Statutory Constraints)
*   **Path:** `/geopackage`
    *   *Files:* `SSSI_L1.gpkg`, `Ancient_Woodland_England.geojson`, `AONB_L1.gpkg`, `ALC_L1.gpkg`
    *   *Contents:* The "Hard Constraints" layer. Contains SSSI (Sites of Special Scientific Interest), Areas of Outstanding Natural Beauty, and the critical Agricultural Land Classification (ALC) grades.
*   **Path:** `/data/L0_artifacts/dno_ng`
    *   *Files:* `DNO_NG_L0_132KV_cable.geoparquet`, `DNO_NG_L0_33KV_substation.geoparquet`
    *   *Contents:* Standardized geometry files for spatial intersection queries.

## 3. Socio-Economic (The Human Terrain & NIMBY Profiling)
*   **Path:** `/geospatial/lsoa-level`
    *   *Files:* `LSOA_L1_Master_SocioEconomic.csv`, `AHAH_V4.csv` (Access to Healthy Assets & Hazards), `Rural_Urban_Classification.csv`
    *   *Contents:* Demographic features (Age, Education, Wealth) mapped to 2021 LSOA boundaries.
*   **Path:** `/QUAL`
    *   *Files:* `Propensity to object/`, `support clusters.txt`, `live sentiment recorder.txt`
    *   *Contents:* Qualitative research data and screenshots used to calibrate the `nimby_coefficient` and sentiment analysis logic.
*   **Path:** `/new geospatial data/lsoa`
    *   *Files:* `COUNT OF HOUSE AGES - LSOA.csv`, `Broadband yearly.csv`
    *   *Contents:* Specialized demographic proxies used to measure "Community Sophistication" and likely objection volume.

## 4. Planning Intelligence (LPA Performance & Policy)
*   **Path:** `/LPA`
    *   *Files:* `PS1_data_202506.csv`, `PS2_data_202506.csv`
    *   *Contents:* Official UK Government statistics on Local Planning Authority (LPA) speed and decision quality.
*   **Path:** `/lpa_plans`
    *   *Files:* `Cornwall.json`, `Cotswold.json`, `Bassetlaw.json`
    *   *Contents:* Scraped and structured Local Plans containing specific policy wording and renewable energy receptiveness.
*   **Path:** `/new geospatial data/area`
    *   *Files:* `local-planning-authority.geojson`, `parish.geojson`
    *   *Contents:* Administrative boundary data for spatial joins between land parcels and governing bodies.

## 5. Pipeline, Logic & Model Artifacts
*   **Path:** `/scripts/production/executors`
    *   *Files:* `exec_001_dfes_headroom.py` through `exec_070_clean_header.py`
    *   *Contents:* The modular feature engineering pipeline. Each `exec` file creates a specific feature layer for the Amaryllis model.
*   **Path:** `/Amaryllis_Runs`
    *   *Files:* `Amaryllis_InferenceReady_Final.csv`, `checkpoint_exec_...csv`
    *   *Contents:* Versioned outputs of the full data fusion process. `Amaryllis_L49_Imputed_Production.csv` is the primary training set.
*   **Path:** `/amaryllis/cognitive_profiler`
    *   *Files:* `orchestrator.py`, `agents/strategist_agent.py`
    *   *Contents:* The Phase II Intelligence Layer—LLM-powered agents that synthesize the GIS and text data into actionable planning risk reports.

## 6. Historical Source of Truth
*   **Path:** `/` (Root)
    *   *Files:* `all_solar_applications_and_dates.csv`, `REPD_Amaryllis_L1.csv`
    *   *Contents:* The ground-truth training labels containing project names, capacities (MW), coordinates, and `planning_duration_days`.