#!/usr/bin/env python3
"""
Test operator gains calculation from neuron activation.
Verifies that spike counts per row map to normalized operator gains.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/jaakkop/Documents/koodii/neuronSeqPerform')

from config import ROWS, COLS
from main import _get_operator_gains_from_neurons

def test_operator_gains():
    """Test that neuron spike counts map to normalized operator gains."""
    print("[Test] Operator gains from neuron activation...")
    
    # Create spike pattern: varying activity per row per column
    accumulated = np.zeros((ROWS, COLS), dtype=bool)
    
    # Column 0: only row 0 fires (10 times)
    accumulated[0, 0] = True
    
    # Column 1: rows 0,1 fire (row 0: 5 times, row 1: 10 times)
    accumulated[0, 1] = True
    accumulated[1, 1] = True
    
    # Column 2: rows 1,3,4 fire
    accumulated[1, 2] = True
    accumulated[3, 2] = True
    accumulated[4, 2] = True
    
    # Column 3: all rows fire equally
    accumulated[:, 3] = True
    
    gains = _get_operator_gains_from_neurons(accumulated)
    
    print(f"\nGains shape: {gains.shape} (should be {(COLS, ROWS)})")
    print("\nColumn → Row activities → Normalized gains:")
    print("-" * 70)
    
    success = True
    for col in range(min(4, COLS)):
        row_activity = np.sum(accumulated[:, col])
        print(f"\nCol {col}: {row_activity} total spikes")
        
        for row in range(ROWS):
            is_active = "●" if accumulated[row, col] else " "
            gain_val = gains[col, row]
            print(f"  Row {row} {is_active}: gain={gain_val:.3f}")
            
            # Verify: if no spikes in column, gain should be 0
            if row_activity == 0 and gain_val != 0.0:
                print(f"    ERROR: Expected 0 gain for inactive column")
                success = False
            
            # Verify: if any spikes in column, at least one gain should be 1.0
            if row_activity > 0 and np.max(gains[col, :]) != 1.0:
                print(f"    ERROR: Expected max gain 1.0 for active column")
                success = False
    
    return success

def test_gain_normalization():
    """Test that gains are always normalized to 0-1 range."""
    print("\n[Test] Gain normalization...")
    
    # Create pattern with multiple spikes per row
    accumulated = np.zeros((ROWS, COLS), dtype=bool)
    
    # Make a column with only one row active
    accumulated[4, 5] = True
    
    gains = _get_operator_gains_from_neurons(accumulated)
    
    # Check that max gain for this column is exactly 1.0
    col_gains = gains[5, :]
    max_gain = np.max(col_gains)
    
    print(f"\nColumn 5 gains: {col_gains}")
    print(f"Max gain: {max_gain:.3f} (should be 1.0)")
    
    success = max_gain == 1.0 and np.all(col_gains >= 0.0) and np.all(col_gains <= 1.0)
    return success

if __name__ == "__main__":
    import pygame
    pygame.init()  # Needed for config import
    
    success1 = test_operator_gains()
    success2 = test_gain_normalization()
    
    overall = success1 and success2
    print(f"\n{'='*70}")
    print(f"Result: {'PASS ✓' if overall else 'FAIL ✗'}")
    exit(0 if overall else 1)
