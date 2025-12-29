import numpy as np
import vwcd

def test_vwcd_basic():
    # Generate synthetic data with a clear change point
    # Segment 1: mean 0, Segment 2: mean 10
    np.random.seed(42)
    s1 = np.random.normal(0, 1, 50)
    s2 = np.random.normal(10, 1, 50)
    X = np.concatenate([s1, s2])
    
    CP, M0, S0, elapsed = vwcd.vwcd(X)
    
    # Check if a change point was detected near index 50
    assert len(CP) > 0
    # It might detect at 49 or 50 or 51 depending on windowing
    assert any(45 <= cp <= 55 for cp in CP)

def test_get_segments():
    X = np.array([1, 1, 1, 10, 10, 10])
    CP = [2] # Change point at index 2 (between 1 and 10)
    
    segments = vwcd.get_segments(X, CP)
    
    assert len(segments) == 2
    assert segments[0]['mean'] == 1.0
    assert segments[1]['mean'] == 10.0
    assert segments[0]['start_index'] == 0
    assert segments[0]['end_index'] == 2
    assert segments[1]['start_index'] == 3
    assert segments[1]['end_index'] == 5

def test_vwcd_empty_or_small():
    X = np.array([1.0, 1.1, 1.2])
    CP, M0, S0, elapsed = vwcd.vwcd(X, w=4) # w > len(X)
    assert len(CP) == 0
    assert len(M0) == 1
