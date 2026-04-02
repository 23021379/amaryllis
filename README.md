# Amaryllis: A Predictive Intelligence Engine for Solar Farm Planning Approval
[![Status](https://img.shields.io/badge/status-in%20development-orange)](https://github.com/brandonhenson/amaryllis)

**Amaryllis is a data-driven model designed to predict the acceptance duration of ground-mounted solar farm planning submissions by UK Local Planning Authorities (LPAs). By analyzing vast geospatial and demographic datasets, it provides a quantitative forecast to help accelerate the renewable energy planning process.**

---

### Table of Contents
1.  [The Core Problem](#the-core-problem)
2.  [Key Features](#key-features)
3.  [Methodology: A "Naive" but Accurate Approach](#methodology-a-naive-but-accurate-approach)
4.  [Data Sources](#data-sources)
5.  [Performance & Limitations](#performance--limitations)
6.  [Future Roadmap](#plans-for-the-future)

---

### The Core Problem
The UK's Net Zero 2050 goal requires an unprecedented acceleration in renewable energy deployment. A major bottleneck in this process is the variable and often lengthy time it takes for a solar farm proposal to gain planning approval from its corresponding LPA. Amaryllis was built to address this challenge by forecasting these timelines based on a project's intrinsic and extrinsic features.

### Key Features
Amaryllis is trained to perform two primary predictive tasks:
*   **Approval Likelihood:** Predicts whether a proposal will be accepted based on key quantitative features (e.g., proposed capacity, coordinates).
*   **Time-to-Acceptance (TTA) Modelling:** More importantly, predicts how long it will take for an accepted proposal to receive its official approval.

### Methodology: A "Naive" but Accurate Approach

Amaryllis is currently "naive," meaning it learns statistical correlations from quantitative data without understanding the qualitative, real-world nuances behind them.

> **Example:** Amaryllis uses geospatial and planning data to know a proposed site is 1km from a heritage site. It has learned that, statistically, proximity to heritage sites increases the time-to-acceptance. However, it does not know if the site is in the direct line-of-sight of the heritage site or if it will harm its *visual amenity*. It cannot be 100% certain of a statutory objection, but it correctly identifies the increased risk and timeline based on historical data patterns.

amaryllis has a comprehensive, but naive*, understanding of how distances to grid infrastructure, general infrastructure, protected lands and areas, and areas of certain demographics impact the time it takes for an LPA to accept a solar farm planning submission.  amaryllis has a strong understanding of how local demographics, community sentiment, and political views can affect the time it takes for an LPA to accept a solar farm planning submission.  amaryllis is naive, but accurate.
Naive refers to the fact that amaryllis learns how the distance to POIs, and their attributes affect the time it takes for an LPA to accept a solar farm’s propsal. amaryllis does not yet take into account the real-world, qualitative, high variance nuances surrounding these POIs.  

amaryllis uses geospatial and planning data from government agencies and network operators, amaryllis knows that the proposed site of a submission is 1km from a heritage site, but it does not know if this proposed site is within direct line of sight of the heritage site. amaryllis does not yet know whether or not the proposed site will harm the visual amenity of the heritage site, and so amaryllis cannot be 100% certain that the proposed site will recieve statutory objection from the heritage site or anyone else impacted.  amaryllis has learned that, generally speaking, being within 1km of a heritage site slows down the submission’s time to acceptance. This generalisation applies to 90%+ of proposed sites, but it will not apply to all of them.


### Data Sources
The model is built on a rich, multi-modal dataset fused from a wide array of official UK sources:
* **1. Planning & Outcome Data**
  * Renewable Energy Planning Database (REPD) – The master list of all renewable projects >150kW.
  * Planning Application Statistics (PS1/PS2/CPS1/CPS2) – LPA performance metrics.
* **2. Grid Infrastructure** 
  * UK Power Networks (UKPN) 
  * National Grid Electricity Distribution
* **3. Ecological & Landscape**
  * Natural England Open Data
  * Historic England
  * Defra (Department for Environment, Food & Rural Affairs)
* **4. Topography & Physical Geography**
  * Ordnance Survey (OS) Data Hub
  * Copernicus Land Monitoring Service.
  * OpenStreetMap (via Geofabrik):
  * Buildings, Landuse, Natural, Places, Railways (used for proximity/density features).
* **5. Socio-Economic**
  * Office for National Statistics (ONS):
  * Census 2021 Data (Demographics).
  * Output Area Classification (OAC).
  * Rural-Urban Classification (RUC).
  * Consumer Data Research Centre (CDRC):
  * Access to Healthy Assets and Hazards (AHAH).
  * Internet User Classification (IUC).
  * Ministry of Housing, Communities & Local Government:
  * Indices of Deprivation (IMD).


### Performance & Limitations
The performance of Amaryllis is entirely dependent on the volume of historical training data available.
*   **Data-Dependent Accuracy:** Certain strata, defined by farm capacity (MW), have more training examples and therefore yield higher prediction accuracy.
*   **Performance Degradation:** There is a scarcity of historical data for very large solar farms (30+ MW). As a result, the model's performance and confidence degrade when forecasting for proposals in this upper stratum.

### Plans for the future:

Using qualitative data: This requires using an llm-powered crawler that traverses different LPA websites to figure out how their website layout works, and how to input a known planning submission into their search program and download the relevant documents. 
All PPI must be removed!
Documents will need to be tokenised into a tabular format: there will likely be hundreds to thousands of unique tokens, each one signifying whether a certain phrase was mentioned, such as: "the proposed site is close to some woodland and will require a wildlife assessment". This specific phrase would hten require an O&M to contact the relevant agency and await their response, and as such this would slow down the planning process by a few days/weeks. Tokens like these will massively boost performance, as the model will know what different LPAs and their planning officers look for in planning submissions, what assessments/reports they usually call for, etc.

The actual plans submitted by the O&M can be analysed to determine what included information caused an officer to call for the reports. If the officer mentioned any missing information, the tokens generated by the analysis can be used to determine if this is true. On top of this, using the quantitative tokens previously mentioned, we can create a program that analyses future proposed plans to ensure all required information is inside the reports/documents to prevent any slowdowns caused by missing info. It can also be used to remove any information that might not be required that could cause an officer to call for an unnecessary report. For example if the proposed site isn't near a woodland, but the report says 'the proposed site is 1km from a woodland', the officer may just call for a wildlife assessment just because it was brought up.

Commuity notes can be analysed too (after PPI has been removed). There may be specific areas in the country with a high NIMBY sentiment. There may be areas where there are organised protest groups. The final document issued by the planning officer takes into account these community notes, so we can analyse them to see what the planning officers usually take into account. For future plans, if a community note contains a tagged issue, the plan can be pre-emptively updated. 

This qualitative step will not only massively improve accuracy of the TTA model, but it will act as a pre-emptive intelligence engine for any O&M team! This will massively speed up the time it takes for solar farm proposals to get accepted, making the Net Zero 2050 goal more achieveable. 
