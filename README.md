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


Plans for the future:

Using qualitative data: This requires using an llm-powered crawler that traverses different LPA websites to figure out how their website layout works, and how to input a known planning submission into their search program and download the relevant documents. 
All PPI must be removed!
Documents will need to be tokenised into a tabular format: there will likely be hundreds to thousands of unique tokens, each one signifying whether a certain phrase was mentioned, such as: "the proposed site is close to some woodland and will require a wildlife assessment". This specific phrase would hten require an O&M to contact the relevant agency and await their response, and as such this would slow down the planning process by a few days/weeks. Tokens like these will massively boost performance, as the model will know what different LPAs and their planning officers look for in planning submissions, what assessments/reports they usually call for, etc.

The actual plans submitted by the O&M can be analysed to determine what included information caused an officer to call for the reports. If the officer mentioned any missing information, the tokens generated by the analysis can be used to determine if this is true. On top of this, using the quantitative tokens previously mentioned, we can create a program that analyses future proposed plans to ensure all required information is inside the reports/documents to prevent any slowdowns caused by missing info. It can also be used to remove any information that might not be required that could cause an officer to call for an unnecessary report. For example if the proposed site isn't near a woodland, but the report says 'the proposed site is 1km from a woodland', the officer may just call for a wildlife assessment just because it was brought up.

Commuity notes can be analysed too (after PPI has been removed). There may be specific areas in the country with a high NIMBY sentiment. There may be areas where there are organised protest groups. The final document issued by the planning officer takes into account these community notes, so we can analyse them to see what the planning officers usually take into account. For future plans, if a community note contains a tagged issue, the plan can be pre-emptively updated. 

This qualitative step will not only massively improve accuracy of the TTA model, but it will act as a pre-emptive intelligence engine for any O&M team! This will massively speed up the time it takes for solar farm proposals to get accepted, making the Net Zero 2050 goal more achieveable. 
