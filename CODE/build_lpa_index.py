import pandas as pd
import json
import re
import time
import random
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from requests.exceptions import HTTPError
import requests

try:
    from duckduckgo_search import DDGS
except ImportError:
    sys.exit("CRITICAL: 'duckduckgo_search' is required.")

try:
    from googlesearch import search as google_search
except ImportError:
    sys.exit("CRITICAL: 'googlesearch-python' is required.")

# --- CONFIGURATION ---
INPUT_CSV = Path(r"[REDACTED_BY_SCRIPT]")
OUTPUT_JSON = Path("[REDACTED_BY_SCRIPT]")

# Vendor Fingerprints (Regex)
VENDORS = {
    "IDOX": [
        r"/publicaccess/", 
        r"/online-applications/",
        r"/idoxpa-web/"
    ],
    "NORTHGATE": [
        r"[REDACTED_BY_SCRIPT]", 
        r"/MVM/", 
        r"/generic/",
        r"[REDACTED_BY_SCRIPT]",
        r"/generichandler"
    ],
    "AGILE": [
        r"/agile_planning/"
    ],
    "OCELLA": [
        r"/web-planning/"
    ],
    # Catch-all for councils using standard CMS pages as portals (e.g., Worcester)
    "CMS_LANDING": [
        r"[REDACTED_BY_SCRIPT]",
        r"[REDACTED_BY_SCRIPT]",
        r"[REDACTED_BY_SCRIPT]",
        r"/planning/find-a-planning-application"
    ]
}

# MODIFIED: Search Wrapper (Google Fallback Tweak)
def get_search_results(query):
    results = []
    
    # 1. DuckDuckGo
    try:
        with DDGS() as ddgs:
            ddg_gen = ddgs.text(query, region='uk-en', max_results=24)
            results = list(ddg_gen)
            if results: return results
    except Exception:
        pass 
    
    # 2. Google Fallback (Polite)
    if not results:
        try:
            # REPAIR: Removed 'pause' arg which crashes specific library versions.
            # We handle the rate-limit sleep manually before the call.
            time.sleep(3.0) 
            g_gen = google_search(query, num_results=24, lang="en")
            results = [{'href': url} for url in g_gen]
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]", end=" ")
            pass
            
    return results

def sanitize_text(text):
    """
    Surgical string cleaning to remove invisible control characters 
    (tabs, non-breaking spaces) that corrupt search queries.
    """
    if not isinstance(text, str):
        return ""
    # Remove control chars (including \t, \n) and non-breaking spaces
    text = re.sub(r'[\x00-\x1F\x7F\xa0]', ' ', text)
    return " ".join(text.split())

# MODIFIED: Domain & Token Logic
def is_valid_council_domain(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # 1. Blacklist
    blacklist = [
        "planning.org.uk", "www.gov.uk", "nationaltrust.org.uk", 
        "facebook.com", "twitter.com", "instagram.com", "linkedin.com"
    ]
    if any(b in domain for b in blacklist):
        return False

    # 2. Whitelist (Expanded for Wales & NI)
    valid_tlds = [
        ".gov.uk", ".org.uk", ".gov.scot", ".gov.wales", 
        ".gov.je", ".gov.im", ".llyw.cymru" # Welsh Government TLD
    ]
    
    # 3. Special Override for Northern Ireland Central Portal
    if "[REDACTED_BY_SCRIPT]" in domain:
        return True
        
    return any(tld in domain for tld in valid_tlds)

def identify_vendor(url):
    for vendor, patterns in VENDORS.items():
        for pattern in patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return vendor
    return "UNKNOWN"


# MODIFIED: Token Logic
# REASONING: 
# - Explicitly replacing hyphens/dots with spaces BEFORE removing special chars.
# - Added 'st', 'saint' to ignore list.
# - Debug prints enabled in validation.

def extract_significant_tokens(lpa_name):
    # 1. Delimiter normalization
    lpa_name = lpa_name.replace('-', ' ').replace('.', ' ')
    
    # 2. Clean
    clean = re.sub(r'[^\w\s]', '', lpa_name.lower())
    
    # 3. Stopword Filtering
    ignore = {
        'council', 'borough', 'city', 'district', 'of', 'and', 'the', 
        'royal', 'london', 'st', 'saint', 'metropolitan',
        'town', 'parish', 'community'  # Added Tier 2/3 identifiers
    }
    
    words = clean.split()
    filtered_words = [w for w in words if w not in ignore]
    
    # FAIL-SAFE: If filtering removed EVERYTHING (e.g. "City of London"), 
    # revert to the original words to ensure we have something to match.
    if not filtered_words and words:
        filtered_words = words

    acronym = "".join([w[0] for w in filtered_words if w]) if filtered_words else ""
    
    return set(filtered_words), acronym

def validate_domain_against_lpa(url, lpa_name):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # OVERRIDE 1: Vendor Fingerprint
    if identify_vendor(url) != "UNKNOWN":
        return True
        
    # OVERRIDE 2: Northern Ireland Central Portal
    # If we are looking for an NI council and find the NI portal, accept it.
    if "[REDACTED_BY_SCRIPT]" in domain:
        return True

    # 3. Sociological Filtering (Town/Parish Trap)
    # Mandate: Distinguish between LPA (District) and Consultee (Town/Parish).
    # Rejects 'stroudtown.gov.uk' for 'Stroud' unless LPA name explicitly contains 'Town'.
    clean_domain_stem = domain.replace(".gov.uk", "").replace(".org.uk", "")
    tier_markers = ["town", "parish", "community", "village"]
    
    for marker in tier_markers:
        # Check if domain implies lower tier (e.g., 'stroudtown', 'cheveningparish')
        # AND the requested LPA name does NOT contain that marker.
        # This blocks 'stroudtown.gov.uk' (ends with town) from matching 'Stroud'.
        if (clean_domain_stem.endswith(marker) or f"{marker}council" in clean_domain_stem) and marker not in lpa_name.lower():
            # FAIL-SAFE: Check if it'[REDACTED_BY_SCRIPT]'town')
            return False

    # Standard Logic
    GEO_ABBREVIATIONS = get_abbreviations_map()
    tokens, acronym = extract_significant_tokens(lpa_name)
    
    expanded_tokens = set(tokens)
    for t in tokens:
        if t in GEO_ABBREVIATIONS:
            expanded_tokens.add(GEO_ABBREVIATIONS[t])

    for token in expanded_tokens:
        if len(token) >= 3 and token in domain:
            return True
            
    if len(acronym) >= 2 and domain.startswith(acronym):
        return True
        
    return False

# MODIFIED: Abbreviation Map & Probe Logic
def get_abbreviations_map():
    # SURGICAL MAPPINGS FOR REMAINING 24
    return {
        # Standard Counties
        "hertfordshire": "herts",
        "hampshire": "hants",
        "nottinghamshire": "notts",
        "leicestershire": "leics",
        "northamptonshire": "northants",
        "oxfordshire": "oxon",
        "wiltshire": "wilts",
        "clackmannanshire": "clacks",
        "dunbartonshire": "dunbarton",
        "monmouthshire": "monmouth",
        "shropshire": "salop",
        "warwickshire": "warks",
        "cambridgeshire": "cambs",
        "[REDACTED_BY_SCRIPT]": "dumgal",
        "yorkshire": "yorks",
        "lancashire": "lancs",
        "staffordshire": "staffs",
        "lincolnshire": "lincs",
        "buckinghamshire": "bucks",
        "berkshire": "berks",
        
        # The Final Stubborn Set
        "welwyn hatfield": "welhat",
        "[REDACTED_BY_SCRIPT]": "tmbc",
        "[REDACTED_BY_SCRIPT]": "scambs",
        "forest of dean": "fdean",
        "na h-eileanan siar": "cne-siar", # Western Isles
        "isle of wight": "iow",
        "king's lynn and west norfolk": "west-norfolk",
        "[REDACTED_BY_SCRIPT]": "opdc",
        "south antrim": "antrim", # Often shares with Newtownabbey
        "derry city and strabane": "derrystrabane",
        "fermanagh and omagh": "fermanaghomagh",
        "mid ulster": "midulstercouncil",
        "newry, mourne and down": "newrymournedown",
        
        # Phase 2 Additions (Fixing the "63 Missing")
        "south gloucestershire": "southglos",
        "telford and wrekin": "telford",
        "[REDACTED_BY_SCRIPT]": "blackburn",
        "bath and north east somerset": "bathnes",
        "bedford": "bedford",
        "[REDACTED_BY_SCRIPT]": "centralbedfordshire",
        "cheshire east": "cheshireeast",
        "cheshire west and chester": "cheshirewestandchester",
        "[REDACTED_BY_SCRIPT]": "eastriding",
        "[REDACTED_BY_SCRIPT]": "nelincs",
        "north lincolnshire": "northlincs",
        "redcar and cleveland": "redcar-cleveland",
        "southend-on-sea": "southend",
        "stockton-on-tees": "stockton",
        "stoke-on-trent": "stoke",
        "west berkshire": "westberks",
        "windsor and maidenhead": "rbwm"
    }
def clean_authority_name(raw_name):
    clean = sanitize_text(raw_name)
    
    # PATCH: Fix known CSV typos and Source irregularities
    typo_map = {
        "Talbort": "Talbot",
        "Burnely": "Burnley",
        "Norwich North": "Norwich"
    }
    for bad, good in typo_map.items():
        if bad in clean:
            clean = clean.replace(bad, good)

    # Normalize "City of X" / "Royal Borough of X" -> "X"
    prefixes = ["City of ", "Royal Borough of ", "London Borough of "]
    for p in prefixes:
        if clean.startswith(p):
            clean = clean.replace(p, "")
            
    # Handle "Bristol, City of" suffix inversion
    if ", City of" in clean:
        clean = clean.replace(", City of", "")
            
    clean = clean.replace(" Borough Council", "") \
                 .replace(" District Council", "") \
                 .replace(" City Council", "") \
                 .replace(" Council", "") \
                 .replace(",", "") \
                 .strip()
    return clean


def generate_probe_candidates(lpa_name):
    """
    Generates potential official domain names AND common deep-link paths.
    Includes Multi-Regional TLDs (Scotland/Wales/Islands) and Joint Authority logic.
    """
    domains = []
    clean_base = lpa_name.lower().replace(" ", "").replace(".", "").replace("-", "")
    
    # 1. TLD Expansion (The "Devolution" Patch)
    # Covers UK, Scotland, Wales, Jersey, Isle of Man
    tlds = [".gov.uk", ".gov.scot", ".gov.wales", ".llyw.cymru", ".gov.je", ".gov.im"]
    
    # Variant A: Compressed (northayrshire)
    for tld in tlds:
        domains.append(f"[REDACTED_BY_SCRIPT]")
        domains.append(f"[REDACTED_BY_SCRIPT]")

    # Variant B: Hyphenated (north-ayrshire)
    # Only relevant if the original name had spaces or hyphens
    if " " in lpa_name or "-" in lpa_name:
        hyphen_base = lpa_name.lower().replace(" ", "-").replace(".", "")
        for tld in tlds:
            domains.append(f"[REDACTED_BY_SCRIPT]")
            domains.append(f"[REDACTED_BY_SCRIPT]")

    # 2. Abbreviated variations with TLDs
    abbrev_map = get_abbreviations_map()
    lower_name = lpa_name.lower()
    
    for full, short in abbrev_map.items():
        if full in lower_name:
            abbrev_name = lower_name.replace(full, short).replace(" ", "").replace("-", "")
            for tld in tlds:
                domains.append(f"[REDACTED_BY_SCRIPT]")

    # 3. Special Handling for "City of X" / "X City" inversion
    # Ensures "glasgow" is tried for "Glasgow City"
    if "city" in lower_name:
        city_base = lower_name.replace("city", "").strip().replace(" ", "")
        domains.append(f"[REDACTED_BY_SCRIPT]")
        domains.append(f"[REDACTED_BY_SCRIPT]")

    # 3. Path Expansion (The "Worcester Fix")
    # Proactively probe common landing pages to capture councils with CMS fronts.
    common_paths = [
        "", # Root
        "/planning",
        "/planning-applications",
        "[REDACTED_BY_SCRIPT]",
        "/planning/planning-applications",
        "[REDACTED_BY_SCRIPT]",
        # Specific fix for Worcester / Worthing / Adur style CMS paths
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]"
    ]
    
    final_candidates = []
    for domain in list(set(domains)):
        for path in common_paths:
            final_candidates.append(f"{domain}{path}")
            
    return final_candidates

def probe_url(url):
    """
    Checks if a URL exists and is accessible. Returns the FINAL URL after redirects.
    Includes robust headers and extended timeout to handle slow gov.uk servers.
    """
    try:
        headers = {
            'User-Agent': '[REDACTED_BY_SCRIPT]',
            'Accept': '[REDACTED_BY_SCRIPT]',
            'Accept-Language': '[REDACTED_BY_SCRIPT]'
        }
        # Increased timeout to 8.0s for slow council servers
        response = requests.get(url, timeout=8.0, allow_redirects=True, headers=headers)
        
        # Accept 200 OK. 
        # Note: Some portals return 403 on automated probes but still exist. 
        # We rely on 200 for high certainty.
        if response.status_code == 200:
            return response.url
    except Exception:
        return None
    return None


# MODIFIED: Main Build Loop
# REASONING: Logic updated to strictly target 'failed' entries in the existing JSON, preserving success states.

def get_manual_overrides():
    """
    Returns a dictionary of LPA Names -> Exact URLs for entities that:
    1. Are National Bodies (not Councils).
    2. Use centralized portals (Northern Ireland).
    3. Have irregular domain acronyms (e.g., RBKC, RCT).
    """
    ni_portal = "[REDACTED_BY_SCRIPT]"
    return {
        # Northern Ireland Cluster (Centralized Portal)
        "[REDACTED_BY_SCRIPT]": ni_portal,
        "Fermanagh and Omagh": ni_portal,
        "Mid Ulster": ni_portal,
        "[REDACTED_BY_SCRIPT]": ni_portal,
        "South Antrim": ni_portal, # Often Antrim and Newtownabbey
        "[REDACTED_BY_SCRIPT]": ni_portal,
        "Ards and North Down": ni_portal,
        "Belfast": ni_portal,
        "[REDACTED_BY_SCRIPT]": ni_portal,
        "[REDACTED_BY_SCRIPT]": ni_portal,
        "[REDACTED_BY_SCRIPT]": ni_portal,
        "[REDACTED_BY_SCRIPT]": ni_portal,

        # National Infrastructure & Special Bodies
        "DECC (S36)": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "Marine Scotland": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "Crown Estate": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",

        # Irregular Domains (Acronyms & Hyphens)
        "Rhondda Cynon Taf": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "Barking and Dagenham": "[REDACTED_BY_SCRIPT]",
        "North Kesteven": "[REDACTED_BY_SCRIPT]",
        "Na h-Eileanan Siar": "[REDACTED_BY_SCRIPT]",
        "Brighton and Hove": "[REDACTED_BY_SCRIPT]",
        "Newcastle upon Tyne": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        
        # Phase 3: Clearing the "Missing 35"
        "Worthing": "[REDACTED_BY_SCRIPT]",
        "Lewes": "[REDACTED_BY_SCRIPT]",
        "East Staffordshire": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "North West Leicestershire": "[REDACTED_BY_SCRIPT]",
        "Oadby and Wigston": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "South Staffordshire": "[REDACTED_BY_SCRIPT]",
        "South Holland": "[REDACTED_BY_SCRIPT]",
        "Welwyn Hatfield": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "Neath Port Talbot": "[REDACTED_BY_SCRIPT]",
        "Gwynedd": "[REDACTED_BY_SCRIPT]",
        "Isle of Man": "https://www.gov.im/",
        "Jersey": "https://www.gov.je/",
        "Bristol, City of": "[REDACTED_BY_SCRIPT]",
        "Glasgow City": "[REDACTED_BY_SCRIPT]",
        "Perth and Kinross": "[REDACTED_BY_SCRIPT]",
        "County Durham": "[REDACTED_BY_SCRIPT]",
        "North Hertfordshire": "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]": "[REDACTED_BY_SCRIPT]",
        "Nottingham": "[REDACTED_BY_SCRIPT]",
        "Burnely": "[REDACTED_BY_SCRIPT]",  # Force fix for the typo entry
        "Norwich North": "[REDACTED_BY_SCRIPT]", # Mapping constituency to Council
        "Redditch": "[REDACTED_BY_SCRIPT]",

        # Phase 4: The Final 5 (Absolute Override)
        "Boston": "[REDACTED_BY_SCRIPT]",
        "Horsham": "[REDACTED_BY_SCRIPT]",
        "North Ayrshire": "[REDACTED_BY_SCRIPT]",
        "Worcester": "[REDACTED_BY_SCRIPT]",
        
        # Redundant keys for Newry to bypass comma/cleaning ambiguity
        "[REDACTED_BY_SCRIPT]": ni_portal,
        "[REDACTED_BY_SCRIPT]": ni_portal,
        "[REDACTED_BY_SCRIPT]": ni_portal
    }


def build_registry():
    if not INPUT_CSV.exists():
        print(f"[REDACTED_BY_SCRIPT]")
        return

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[REDACTED_BY_SCRIPT]")
    
    # 0. Load Overrides
    manual_overrides = get_manual_overrides()

    # 1. Load Source Data
    try:
        df = pd.read_csv(INPUT_CSV, encoding='cp1252')
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, encoding='latin1')
    
    # Sanitize source
    df['Planning Authority'] = df['Planning Authority'].apply(sanitize_text)
    all_lpas = sorted(df['Planning Authority'].dropna().unique())
    
    # 2. Load Existing Registry
    registry = []
    processed_map = {} # Map raw_name -> entry object
    
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, 'r') as f:
                registry = json.load(f)
                # Index existing entries for quick lookup
                for entry in registry:
                    processed_map[entry['lpa_name_raw']] = entry
            print(f"[REDACTED_BY_SCRIPT]")
        except json.JSONDecodeError:
            print("[REDACTED_BY_SCRIPT]")
            
    # 3. Identify Targets (Missing OR Null URL)
    # We rebuild the list to maintain alphabetical order, but only flag 'targets' for processing
    final_list = []
    targets = []
    
    for lpa_raw in all_lpas:
        # If it exists, use it. If it's missing, create blank.
        entry = processed_map.get(lpa_raw, {
            "lpa_name_raw": lpa_raw,
            "lpa_name_clean": clean_authority_name(lpa_raw),
            "portal_url": None,
            "vendor": "UNKNOWN",
            "verification_status": "AUTO_GENERATED"
        })
        
        # Check if needs processing
        if entry['portal_url'] is None:
            targets.append(entry)
        
        final_list.append(entry)

    print(f"[REDACTED_BY_SCRIPT]")

    if not targets:
        print("[REDACTED_BY_SCRIPT]")
        return

    # 4. Processing Loop
    save_interval = 5
    
    # We iterate over the 'targets' list, but update the objects inside 'final_list' (by reference)
    for i, entry in enumerate(targets):
        clean_name = entry['lpa_name_clean']
        if not clean_name: continue

        print(f"[REDACTED_BY_SCRIPT]", end=" ", flush=True)

        found_root_url = None
        detected_vendor = "UNKNOWN"
        valid_candidate_found = False

        # STEP 0: Manual Override Check (The "Nuclear" Option)
        # Sanitized name matching
        if clean_name in manual_overrides:
            found_root_url = manual_overrides[clean_name]
            detected_vendor = "MANUAL_OVERRIDE"
            valid_candidate_found = True
            print(f"[REDACTED_BY_SCRIPT]")

        # STEP 1: Direct Domain Probe (Bypass Search Engines)
        # Only proceed if override didn't catch it
        if not valid_candidate_found:
             probe_candidates = generate_probe_candidates(clean_name)
        # We try to guess the URL. If it resolves and matches validation, we skip Google entirely.
        probe_candidates = generate_probe_candidates(clean_name)
        for candidate in probe_candidates:
            final_url = probe_url(candidate)
            if final_url:
                if is_valid_council_domain(final_url) and validate_domain_against_lpa(final_url, clean_name):
                    parsed = urlparse(final_url)
                    found_root_url = f"[REDACTED_BY_SCRIPT]"
                    detected_vendor = identify_vendor(final_url)
                    valid_candidate_found = True
                    print(f"[REDACTED_BY_SCRIPT]")
                    break
        
        # STEP 2: Search Engine Fallback (Only if Probe failed)
        if not valid_candidate_found:
            search_queries = [
                f"[REDACTED_BY_SCRIPT]", 
                f"[REDACTED_BY_SCRIPT]",
                f"[REDACTED_BY_SCRIPT]",
                f"[REDACTED_BY_SCRIPT]", 
                f"[REDACTED_BY_SCRIPT]",
                f"[REDACTED_BY_SCRIPT]",
                f"[REDACTED_BY_SCRIPT]"
            ]

            for q_idx, query in enumerate(search_queries):
                if valid_candidate_found: break

                # ... (Standard search logic from previous iteration) ...
                # ... (Use existing get_search_results logic) ...
                # Copying the previous loop structure here for completeness context
                if q_idx > 0: print(f"..({q_idx+1})..", end=" ")
                try:
                    time.sleep(random.uniform(2.0, 4.0))
                    results = get_search_results(query)
                    if results:
                        for res in results:
                            raw_url = res.get('href', '')
                            if not raw_url: continue
                            if is_valid_council_domain(raw_url):
                                if validate_domain_against_lpa(raw_url, clean_name):
                                    full_url_parsed = urlparse(raw_url)
                                    full_clean_url = urlunparse((full_url_parsed.scheme, full_url_parsed.netloc, full_url_parsed.path, '', '', ''))
                                    detected_vendor = identify_vendor(full_clean_url)
                                    found_root_url = f"[REDACTED_BY_SCRIPT]"
                                    valid_candidate_found = True
                                    print(f"[REDACTED_BY_SCRIPT]")
                                    break
                    if not valid_candidate_found and q_idx == len(search_queries) - 1:
                        print("-> FAILED.")
                except Exception as e:
                    print(f"[Err: {e}]", end=" ")

        entry['portal_url'] = found_root_url
        entry['vendor'] = detected_vendor
        
        if (i + 1) % save_interval == 0:
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(final_list, f, indent=4)

    # Final Save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_list, f, indent=4)
    
    print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    build_registry()