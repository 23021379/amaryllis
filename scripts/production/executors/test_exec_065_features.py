import sys
import os
import logging
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

# Mock geopandas and shapely BEFORE importing exec_065
mock_gpd = MagicMock()
sys.modules['geopandas'] = mock_gpd
sys.modules['shapely'] = MagicMock()
sys.modules['shapely.geometry'] = MagicMock()

# Mock sklearn
mock_sklearn = MagicMock()
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.neighbors'] = MagicMock()

# We need to mock KDTree specifically because it's imported as: from sklearn.neighbors import KDTree
# So we need to make sure that import works.
# The easiest way is to mock sys.modules['sklearn.neighbors'] and give it a KDTree attribute.
mock_neighbors = MagicMock()
sys.modules['sklearn.neighbors'] = mock_neighbors

class MockKDTree:
    def __init__(self, data):
        self.data = data
        
    def query(self, X, k=1):
        # Simple 1D distance mock for testing
        # Assume X is (1, 2) and data is (N, 2)
        # Just return dummy distances and indices
        # Dists: 0, 5000, 10000 (meters)
        # Indices: 0, 1, 2
        n_samples = len(X)
        dists = np.array([[0.0, 5000.0, 10000.0][:k]] * n_samples)
        indices = np.array([[0, 1, 2][:k]] * n_samples)
        return dists, indices
        
    def query_radius(self, X, r):
        # Return indices within radius
        # If r=10000, return [0, 1, 2]
        n_samples = len(X)
        return np.array([np.array([0, 1, 2])] * n_samples, dtype=object)

mock_neighbors.KDTree = MockKDTree

# Add scripts directory to path
sys.path.append(r'[REDACTED_BY_SCRIPT]')
import exec_065_historical_precedent as exec_065

# Mock logging
logging.basicConfig(level=logging.INFO)

def test_workload_features():
    print("[REDACTED_BY_SCRIPT]")
    # Create dummy PS1/CPS1 data
    # We need to mock the file reading, but since the functions take paths, 
    # we might need to monkeypatch pd.read_csv or just create temporary files.
    # Easier to monkeypatch pd.read_csv for this test script.
    
    original_read_csv = pd.read_csv
    
    def mock_read_csv(path, **kwargs):
        if 'PS1' in path:
            return pd.DataFrame({
                'lpanm': ['LPA1', 'LPA1'],
                'f_year': ['2020', '2021'],
                '[REDACTED_BY_SCRIPT]': ['100', '120'],
                '[REDACTED_BY_SCRIPT]': ['10', '12']
            })
        elif 'CPS1' in path:
            return pd.DataFrame({
                'lpanm': ['LPA1', 'LPA1'],
                'quarter': ['2022 Q1', '2022 Q2'],
                '[REDACTED_BY_SCRIPT]': ['30', '40'], # Total 70 for 2022
                '[REDACTED_BY_SCRIPT]': ['3', '4']
            })
        elif 'PS2' in path:
             return pd.DataFrame({
                'lpanm': ['LPA1'],
                '[REDACTED_BY_SCRIPT]': ['8'],
                '[REDACTED_BY_SCRIPT]': ['10'],
                '[REDACTED_BY_SCRIPT]': ['2'],
                '[REDACTED_BY_SCRIPT]': ['3'],
                '[REDACTED_BY_SCRIPT]': ['1'],
                '[REDACTED_BY_SCRIPT]': ['2']
                # Total comm granted: 3, Total comm decisions: 5 -> Rate: 0.6
            })
        elif 'CPS2' in path:
            return pd.DataFrame({
                'lpanm': ['LPA1', 'LPA1', 'LPA1', 'LPA1'],
                'granted_or_refused': ['Granted', 'Refused', 'Granted', 'Granted'],
                'date_received': ['01/01/2023', '01/01/2023', '01/01/2023', '01/01/2023'],
                'date_dispatched': ['11/01/2023', '21/01/2023', '31/01/2023', '15/01/2023'],
                # Days: 10, 20, 30, 14
                # Granted: 10, 30, 14 -> Mean: 18
                # Refused: 20 -> Mean: 20
                # Ratio: 18/20 = 0.9
                # P90 of [10, 14, 20, 30] -> 27
                'type_of_scheme': ['Major Dwellings', 'Minor Dwellings', 'Major Industrial', 'Minor Retail']
            })
        elif 'REPD' in path: # Legacy
             return pd.DataFrame({
                'planning_authority': ['LPA1', 'LPA1', 'LPA2'],
                'permission_granted': ['1', '0', '1'],
                '[REDACTED_BY_SCRIPT]': ['100', '200', '150'],
                'easting': [500000, 505000, 510000], # 5km away, 10km away
                'northing': [100000, 100000, 100000]
            })
        return pd.DataFrame()

    pd.read_csv = mock_read_csv
    
    # Mock os.path.exists
    original_exists = os.path.exists
    os.path.exists = lambda path: True
    
    try:
        # Test Module 1
        df_workload = exec_065.process_ps1_cps1('PS1', 'CPS1')
        print("Workload Result:")
        print("Columns:", df_workload.columns.tolist())
        print(df_workload)
        print("Index:", df_workload.index)
        
        # Expected:
        # 2020: 100, 2021: 120, 2022: 70
        # Avg: (100+120+70)/3 = 96.66
        # Total: 290
        # Trend: 2020->100, 2021->120, 2022->70. Slope approx -15.
        
        if 'lpa1' not in df_workload.index:
            print("WARNING: 'lpa1'[REDACTED_BY_SCRIPT]", df_workload.index.tolist())
            
        if '[REDACTED_BY_SCRIPT]' not in df_workload.columns:
             print("ERROR: 'lpa_avg_yearly_workload'[REDACTED_BY_SCRIPT]", df_workload.columns.tolist())
        
        assert np.isclose(df_workload.loc['lpa1', '[REDACTED_BY_SCRIPT]'], 96.666, atol=0.1)
        assert df_workload.loc['lpa1', 'lpa_total_experience'] == 290
        
        # Test Module 2 (PS2)
        df_ps2 = exec_065.process_ps2('PS2')
        print("\nPS2 Result:")
        print(df_ps2.loc['lpa1'])
        assert df_ps2.loc['lpa1', 'lpa_major_commercial_approval_rate'] == 0.6
        
        # Test Module 3 (CPS2)
        df_cps2 = exec_065.process_cps2('CPS2')
        print("\nCPS2 Result:")
        print(df_cps2.loc['lpa1'])
        # Days: 10, 20, 30, 14. Sorted: 10, 14, 20, 30.
        # P90: 0.9 * (4-1) + 1 = 3.7th index (interpolated). 
        # numpy quantile linear: 10 + (30-10)*0.9 = 28? No.
        # Pandas quantile default is linear.
        # 0.9 quantile of [10, 14, 20, 30] is 27.0
        assert np.isclose(df_cps2.loc['lpa1', '[REDACTED_BY_SCRIPT]'], 27.0, atol=1.0)
        assert df_cps2.loc['lpa1', '[REDACTED_BY_SCRIPT]'] == 0.9

        # Test Spatial
        # Mock Master GDF
        # We need an object that has .geometry.x and .geometry.y
        # And can be indexed.
        
        class MockPoint:
            def __init__(self, x, y):
                self.x = x
                self.y = y
                
        class MockGeoSeries:
            def __init__(self, points):
                self.points = points
                self.x = np.array([p.x for p in points])
                self.y = np.array([p.y for p in points])
                
        class MockGeoDataFrame(pd.DataFrame):
            def __init__(self, *args, **kwargs):
                geometry = kwargs.pop('geometry', None)
                super().__init__(*args, **kwargs)
                if geometry:
                    self.geometry = MockGeoSeries(geometry)
                    
        # Mock gpd.GeoDataFrame to return our MockGeoDataFrame
        mock_gpd.GeoDataFrame = MockGeoDataFrame
        mock_gpd.points_from_xy = lambda x, y: [MockPoint(xi, yi) for xi, yi in zip(x, y)]

        # Create Master GDF
        master_gdf = MockGeoDataFrame(
            {'hex_id': ['A'], 'lpa_join_key': ['lpa1']}, 
            geometry=[MockPoint(500000, 100000)]
        )
        master_gdf.set_index('hex_id', inplace=True)
        
        # We also need to mock gpd.GeoSeries.from_wkt or points_from_xy in the exec script
        # The exec script uses: gpd.GeoDataFrame(df_legacy, geometry=gpd.points_from_xy(...))
        # Our mock_gpd.points_from_xy handles this.
        
        lpa_stats, knn_features = exec_065.process_legacy_spatial(master_gdf, 'REPD')
        print("\nKNN Features:")
        print(knn_features.iloc[0])
        
        # Legacy points:
        # 1. (500000, 100000) - Dist 0. Outcome 1.
        # 2. (505000, 100000) - Dist 5km. Outcome 0.
        # 3. (510000, 100000) - Dist 10km. Outcome 1.
        
        # Nearby (10km radius):
        # Point 1 (0km) -> Yes
        # Point 2 (5km) -> Yes
        # Point 3 (10km) -> Yes (inclusive? usually)
        # If inclusive, count = 3. Approval = (1+0+1)/3 = 0.66
        
        # KNN (K=15 -> all 3)
        # Dists: 0, 5, 10
        # Avg Dist: 5
        # Inverse Weighted Approval:
        # W1 = 1/epsilon (huge)
        # W2 = 1/5 = 0.2
        # W3 = 1/10 = 0.1
        # Weighted Avg dominated by W1 -> Outcome 1.
        
        assert knn_features.iloc[0]['nearby_legacy_count'] >= 2
        assert np.isclose(knn_features.iloc[0]['[REDACTED_BY_SCRIPT]'], 1.0, atol=0.01) # Dominated by 0 distance
        
        print("\nALL TESTS PASSED")
        
    finally:
        pd.read_csv = original_read_csv
        os.path.exists = original_exists

if __name__ == "__main__":
    test_workload_features()
