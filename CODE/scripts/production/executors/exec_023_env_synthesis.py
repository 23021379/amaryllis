import logging
import numpy as np

# --- Executor Configuration ---
NULL_SENTINEL = -1.0
EPSILON = 1e-6 # To prevent division by zero

def execute(state: dict) -> dict:
    """[REDACTED_BY_SCRIPT]"""
    try:
        features = state['features']
        
        aw_dist = features.get('aw_dist_to_nearest_m', np.inf)
        aonb_dist = features.get('aonb_dist_to_nearest_m', np.inf)

        # Minimum distance to any constraint
        min_dist = min(aw_dist, aonb_dist)
        features['[REDACTED_BY_SCRIPT]'] = min_dist if min_dist != np.inf else NULL_SENTINEL

        # Dominant constraint type
        if aw_dist <= aonb_dist and aw_dist != np.inf:
            features['[REDACTED_BY_SCRIPT]'] = 'AncientWoodland'
        elif aonb_dist < aw_dist and aonb_dist != np.inf:
            features['[REDACTED_BY_SCRIPT]'] = 'AONB'
        else:
            features['[REDACTED_BY_SCRIPT]'] = 'None'

        # Total constraint count in 2km
        aw_count_2km = features.get('aw_count_in_2km', 0)
        aonb_count_2km = features.get('aonb_count_in_2km', 0) # Assumes AONB executor calculated this
        features['[REDACTED_BY_SCRIPT]'] = aw_count_2km + aonb_count_2km

        # --- SIC-GRID-01: kVA per MW-electric Ratio ---
        # This calculation is now architecturally sound, as it consumes features
        # guaranteed to exist from exec_011 and the initial site data load.
        total_kva = features.get('[REDACTED_BY_SCRIPT]', 0.0)
        site_capacity_mwelec = state.get('[REDACTED_BY_SCRIPT]', 0.0)

        if site_capacity_mwelec > 0:
            # Convert site capacity from MW to kVA assuming a power factor of 1 for ratio consistency
            site_capacity_kva = site_capacity_mwelec * 1000
            ratio = total_kva / (site_capacity_kva + EPSILON)
            features['[REDACTED_BY_SCRIPT]'] = np.nan_to_num(ratio, nan=NULL_SENTINEL)
        else:
            features['[REDACTED_BY_SCRIPT]'] = NULL_SENTINEL

    except Exception as e:
        error_msg = f"[REDACTED_BY_SCRIPT]"
        logging.debug(error_msg)
        state['errors'].append(error_msg)
        # Ensure schema consistency on failure
        features.setdefault('[REDACTED_BY_SCRIPT]', NULL_SENTINEL)

    return state