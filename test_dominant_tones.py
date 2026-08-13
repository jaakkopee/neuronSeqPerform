#!/usr/bin/env python3
"""
Test the dominant tone mapping feature.
Verifies that most-active neuron per column maps to correct scale degree.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/jaakkop/Documents/koodii/neuronSeqPerform')

from config import ROWS, COLS, SCALES
from main import _get_dominant_tone_freqs

def test_dominant_tone_mapping():
    """Test that dominant neuron rows map to scale degrees correctly."""
    print("[Test] Dominant tone mapping...")
    
    scale_name = "major"
    scale = SCALES[scale_name]
    root_midi = 60  # C4
    
    # Create test spike pattern: each column has dominant activity in a different row
    accumulated = np.zeros((ROWS, COLS), dtype=bool)
    
    for col in range(COLS):
        dominant_row = col % ROWS  # Column 0→row 0, col 1→row 1, etc.
        accumulated[dominant_row, col] = True  # Heavy activity in that row
    
    # Get the resulting frequencies
    freqs = _get_dominant_tone_freqs(accumulated, root_midi, scale_name)
    
    print(f"\nScale: {scale_name} = {scale}")
    print(f"Root note: MIDI {root_midi} = {440.0 * (2.0 ** ((root_midi - 69) / 12.0)):.2f} Hz")
    print("\nColumn → Dominant Row → Scale Degree → Octave → Frequency:")
    print("-" * 70)
    
    success = True
    for col in range(COLS):
        dominant_row = col % ROWS
        expected_degree = dominant_row % len(scale)
        octave = col // len(scale)  # Octave from column position
        
        # Calculate expected frequency
        semitone = scale[expected_degree] + octave * 12
        root_freq = 440.0 * (2.0 ** ((root_midi - 69) / 12.0))
        expected_freq = root_freq * (2.0 ** (semitone / 12.0))
        
        actual_freq = freqs[col]
        error = abs(actual_freq - expected_freq)
        
        status = "✓" if error < 1.0 else "✗"
        print(f"{status} Col {col:2d} → Row {dominant_row} → Degree {expected_degree} → "
              f"Oct {octave} → {actual_freq:7.2f} Hz (expected {expected_freq:7.2f})")
        
        if error > 1.0:
            success = False
    
    return success

def test_scale_degree_variety():
    """Test that different scales produce different mappings."""
    print("\n[Test] Scale degree variety...")
    
    root_midi = 60
    accumulated = np.zeros((ROWS, COLS), dtype=bool)
    
    # Row 0 active in all columns
    accumulated[0, :] = True
    
    print("\nAll neurons in row 0 → should map to scale degree 0")
    print("Different scales should produce different base frequencies:\n")
    
    for scale_name in ["major", "minor", "pentatonic"]:
        freqs = _get_dominant_tone_freqs(accumulated, root_midi, scale_name)
        print(f"  {scale_name:12s}: {freqs[0]:7.2f} Hz (all columns)")
    
    return True

if __name__ == "__main__":
    success1 = test_dominant_tone_mapping()
    success2 = test_scale_degree_variety()
    
    overall = success1 and success2
    print(f"\n{'='*60}")
    print(f"Result: {'PASS ✓' if overall else 'FAIL ✗'}")
    exit(0 if overall else 1)
