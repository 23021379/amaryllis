"""
AD-AM-INF-01: Decision Dossier Schema

Defines the structured data schema for the Amaryllis Decision Dossier.
Each hex cell will receive a comprehensive dossier containing:
- Identity and location data
- Headline risk assessment
- Uncertainty metrics (confidence intervals, catastrophe risk, error regimes)
- Explainability engine outputs (SHAP-driven feature impacts)
- Contextual benchmarks (geographic and typological)
- All simulation details

Author: Amaryllis Decision Intelligence Team
Date: 2025-11-21
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

@dataclass
class Identity:
    """[REDACTED_BY_SCRIPT]"""
    hex_id: str
    centroid_lon: float
    centroid_lat: float


@dataclass
class HeadlineRiskAssessment:
    """
    The primary, actionable prediction for the optimal development scenario.
    This is the "headline number" that site managers will use for initial screening.
    """
    optimal_capacity_mw: int
    predicted_duration_days: float
    duration_risk_category: str  # "Green", "Amber", or "Red"
    duration_risk_percentile: float  # Percentile rank vs all other hexes


@dataclass
class UncertaintyMetrics:
    """
    Quantification of prediction uncertainty using multiple protocols:
    - P10/P90 confidence intervals
    - Catastrophe risk (probability of extreme delay)
    - Error regime probabilities from specialist ensemble
    """
    confidence_interval_days: List[float]  # [p10, p90]
    catastrophe_risk_prob: float  # Probability of extreme (>100 day) error
    error_regime_probabilities: Dict[str, float]  # {under, accurate, over}


@dataclass
class FeatureDriver:
    """
    A single feature's impact on the prediction, derived from SHAP values.
    Converted to human-interpretable format with day-impact and description.
    """
    feature: str  # Raw feature name
    impact: str  # e.g., "+55 days" or "-15 days"
    description: str  # Human-readable explanation


@dataclass
class ExplainabilityEngine:
    """
    Instance-specific explainability outputs:
    - Why this specific prediction?
    - What are the dominant risk factors?
    - What are the top feature drivers?
    """
    dominant_risk_factor: str  # e.g., "SIC_GRID_POLICY", "COHORT_LPA_ALL"
    narrative_summary: str  # Template-based narrative explanation
    top_drivers_for_optimal_scenario: List[FeatureDriver]  # Top 5-7 features


@dataclass
class AccuracyBenchmark:
    """[REDACTED_BY_SCRIPT]"""
    description: str
    rmse_days: float
    mae_days: float
    sample_count: Optional[int] = None


@dataclass
class TypologicalBenchmark:
    """
    Model performance on projects with similar risk profiles
    (same project regime cluster)
    """
    description: str
    project_regime_id: int
    rmse_days: float
    mae_days: float
    sample_count: Optional[int] = None


@dataclass
class ContextualBenchmarks:
    """
    Provides context on how well the model typically performs on:
    1. Geographically similar sites (25-NN)
    2. Typologically similar projects (same regime)
    """
    local_geographic_accuracy: AccuracyBenchmark
    typological_accuracy: TypologicalBenchmark


@dataclass
class SimulationDetail:
    """[REDACTED_BY_SCRIPT]"""
    sim_capacity_mw: int
    predicted_duration_days: float


@dataclass
class DecisionDossier:
    """
    The complete Decision Dossier for a single hex cell.
    This is the core deliverable of AD-AM-INF-01.
    """
    identity: Identity
    headline_risk_assessment: HeadlineRiskAssessment
    uncertainty_metrics: UncertaintyMetrics
    explainability_engine: ExplainabilityEngine
    contextual_benchmarks: ContextualBenchmarks
    simulation_details: List[SimulationDetail]
    
    def to_geojson_properties(self) -> dict:
        """
        Convert the dossier to a flat dictionary for GeoJSON properties.
        Flattens all nested structures for GIS compatibility (e.g. QGIS/ArcGIS).
        """
        props = {}
        
        # 1. Identity
        props['hex_id'] = self.identity.hex_id
        props['centroid_lon'] = self.identity.centroid_lon
        props['centroid_lat'] = self.identity.centroid_lat
        
        # 2. Headline Risk
        props['optimal_capacity_mw'] = self.headline_risk_assessment.optimal_capacity_mw
        props['[REDACTED_BY_SCRIPT]'] = self.headline_risk_assessment.[REDACTED_BY_SCRIPT]
        props['[REDACTED_BY_SCRIPT]'] = self.headline_risk_assessment.[REDACTED_BY_SCRIPT]
        props['[REDACTED_BY_SCRIPT]'] = self.headline_risk_assessment.[REDACTED_BY_SCRIPT]
        
        # 3. Uncertainty Metrics
        # Flatten list [p10, p90]
        if self.uncertainty_metrics.confidence_interval_days and len(self.uncertainty_metrics.confidence_interval_days) >= 2:
            props['[REDACTED_BY_SCRIPT]'] = self.uncertainty_metrics.confidence_interval_days[0]
            props['[REDACTED_BY_SCRIPT]'] = self.uncertainty_metrics.confidence_interval_days[1]
        else:
            props['[REDACTED_BY_SCRIPT]'] = None
            props['[REDACTED_BY_SCRIPT]'] = None
            
        props['[REDACTED_BY_SCRIPT]'] = self.uncertainty_metrics.[REDACTED_BY_SCRIPT]
        
        # Flatten dictionary error regimes
        for regime, prob in self.uncertainty_metrics.error_regime_probabilities.items():
            props[f'error_prob_{regime}'] = prob
            
        # 4. Explainability
        props['[REDACTED_BY_SCRIPT]'] = self.explainability_engine.[REDACTED_BY_SCRIPT]
        props['narrative_summary'] = self.explainability_engine.narrative_summary
        
        # Flatten top drivers list (take top 3 for brevity in GIS)
        for i, driver in enumerate(self.explainability_engine.top_drivers_for_optimal_scenario[:3]):
            prefix = f"driver_{i+1}"
            props[f'{prefix}_feature'] = driver.feature
            props[f'{prefix}_impact'] = driver.impact
            props[f'{prefix}_desc'] = driver.description
            
        # 5. Contextual Benchmarks
        # Geographic
        geo = self.contextual_benchmarks.local_geographic_accuracy
        props['geo_bench_rmse'] = geo.rmse_days
        props['geo_bench_mae'] = geo.mae_days
        props['geo_bench_count'] = geo.sample_count
        
        # Typological
        typo = self.contextual_benchmarks.typological_accuracy
        props['[REDACTED_BY_SCRIPT]'] = typo.project_regime_id
        props['typo_bench_rmse'] = typo.rmse_days
        props['typo_bench_mae'] = typo.mae_days
        
        # 6. Simulation Details (Pivot to columns)
        # We expect specific capacities, so we'll map them dynamically
        for sim in self.simulation_details:
            cap = sim.sim_capacity_mw
            # Create column like 'sim_50mw_days'
            props[f'sim_{cap}mw_days'] = sim.predicted_duration_days
            
        return props
    
    def get_risk_category_color(self) -> str:
        """[REDACTED_BY_SCRIPT]"""
        colors = {
            "Green": "#10b981",   # Favorable timeline
            "Amber": "#f59e0b",   # Moderate risk
            "Red": "#ef4444"      # High risk / long timeline
        }
        return colors.get(self.headline_risk_assessment.duration_risk_category, "#6b7280")


# --- Risk Category Thresholds ---
RISK_THRESHOLDS = {
    "Green": (0, 250),      # 0-250 days: Favorable
    "Amber": (250, 350),    # 250-350 days: Moderate
    "Red": (350, float('inf'))  # 350+ days: High risk
}

def calculate_risk_category(duration_days: float) -> str:
    """[REDACTED_BY_SCRIPT]"""
    for category, (low, high) in RISK_THRESHOLDS.items():
        if low <= duration_days < high:
            return category
    return "Red"  # Default to Red for safety
