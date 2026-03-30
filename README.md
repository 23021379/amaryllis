amaryllis has a comprehensive, but naive*, understanding of how distances to grid infrastructure, general infrastructure, protected lands and areas, and areas of certain demographics impact the time it takes for an LPA to accept a solar farm planning submission.  amaryllis has a strong understanding of how local demographics, community sentiment, and political views can affect the time it takes for an LPA to accept a solar farm planning submission.  amaryllis is naive, but accurate.
Naive refers to the fact that amaryllis learns how the distance to POIs, and their attributes affect the time it takes for an LPA to accept a solar farm’s propsal. amaryllis does not yet take into account the real-world, qualitative, high variance nuances surrounding these POIs.  

amaryllis uses geospatial and planning data from government agencies and network operators, amaryllis knows that the proposed site of a submission is 1km from a heritage site, but it does not know if this proposed site is within direct line of sight of the heritage site. amaryllis does not yet know whether or not the proposed site will harm the visual amenity of the heritage site, and so amaryllis cannot be 100% certain that the proposed site will recieve statutory objection from the heritage site or anyone else impacted.  amaryllis has learned that, generally speaking, being within 1km of a heritage site slows down the submission’s time to acceptance. This generalisation applies to 90%+ of proposed sites, but it will not apply to all of them.

amaryllis performs differently on different strata.
amaryllis was trained to predict whether or not a proposal will be accepted based on a few key quantitative features: proposed capacity, proposed coordinates, etc.
amaryllis was also, more importantly, trained to predict how long it would take for a proposed site to receive acceptance from its corresponding LPA based on the same key features.

There are only so many solar farm proposals that could be used to train amaryllis, and so the performance of amaryllis is entirely dependent on this available data.
Certain strata (defined by farm capacity) have more training data, and so naturally they perform more accurately. 
Unfortunately, there are not a lot of 30+ MW solar farms, and so performance degrades after this point.

amaryllis performs differently on different strata.
amaryllis was trained to predict whether or not a proposal will be accepted based on a few key quantitative features: proposed capacity, proposed coordinates, etc.
amaryllis was also, more importantly, trained to predict how long it would take for a proposed site to receive acceptance from its corresponding LPA based on the same key features.

There are only so many solar farm proposals that could be used to train amaryllis, and so the performance of amaryllis is entirely dependent on this available data.
Certain strata (defined by farm capacity) have more training data, and so naturally they perform more accurately. 
Unfortunately, there are not a lot of 30+ MW solar farms, and so performance degrades after this point.

All sources used: 1. Planning & Outcome Data
Renewable Energy Planning Database (REPD) – The master list of all renewable projects >150kW.
Planning Application Statistics (PS1/PS2/CPS1/CPS2) – LPA performance metrics.
2. Grid Infrastructure 
UK Power Networks (UKPN) 
National Grid Electricity Distribution
3. Ecological & Landscape
Natural England Open Data
Historic England
Defra (Department for Environment, Food & Rural Affairs)
4. Topography & Physical Geography
Ordnance Survey (OS) Data Hub
Copernicus Land Monitoring Service.
OpenStreetMap (via Geofabrik):
Buildings, Landuse, Natural, Places, Railways (used for proximity/density features).
5. Socio-Economic
Office for National Statistics (ONS):
Census 2021 Data (Demographics).
Output Area Classification (OAC).
Rural-Urban Classification (RUC).
Consumer Data Research Centre (CDRC):
Access to Healthy Assets and Hazards (AHAH).
Internet User Classification (IUC).
Ministry of Housing, Communities & Local Government:
Indices of Deprivation (IMD).
