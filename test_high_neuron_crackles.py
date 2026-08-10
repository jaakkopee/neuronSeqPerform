#!/usr/bin/env python3
"""
Test extreme spike bursts to verify slow envelope attack prevents crackles.
Simulates scenario where more neurons = more simultaneous operator firings.
"""

import numpy as np
from model.lif_network import LIFNetwork
from model.fm_synth import make_synth
from config import SAMPLE_RATE, BUFFER_SIZE, COLS, ROWS

def test_high_neuron_spike_burst():
    """Test maximum spike burst (all operators fire simultaneously)."""
    print("[Test] Initializing network and synth...")
    network = LIFNetwork(rows=ROWS, cols=COLS, neuron_count=8192)  # Max neurons
    synth = make_synth()
    
    print("[Test] Creating massive spike burst (all operators active)...")
    # Create spike pattern: ALL operators fire in column 0
    massive_spikes = np.ones((ROWS, COLS), dtype=bool)
    massive_spikes[:, 1:] = False  # Only column 0
    
    audio_buffers = []
    
    # First, warm up with moderate activity
    print("  Phase 1: Ramp up")
    for step in range(4):
        network.step()
        gradual_spikes = np.zeros((ROWS, COLS), dtype=bool)
        gradual_spikes[:step*8, 0] = True  # Progressive increase
        
        synth_spikes = network.get_synth_spikes(ROWS, COLS, spike_array=gradual_spikes)
        synth.trigger_and_activate(synth_spikes, 0)
        synth.set_active_step(0)
        
        for _ in range(12):
            audio = synth.generate(BUFFER_SIZE)
            audio_buffers.append(audio)
    
    # Then hit with maximum burst
    print("  Phase 2: Maximum spike burst")
    for step in range(8):
        network.step()
        synth_spikes = network.get_synth_spikes(ROWS, COLS, spike_array=massive_spikes)
        synth.trigger_and_activate(synth_spikes, 0)
        synth.set_active_step(0)
        
        audio = synth.generate(BUFFER_SIZE)
        audio_buffers.append(audio)
    
    full_audio = np.concatenate(audio_buffers)
    total_samples = len(full_audio)
    
    print(f"\n=== Analysis ===")
    print(f"Generated {len(audio_buffers)} audio buffers ({total_samples} total samples)")
    
    # Check for crackles: look for high-frequency content (rapid amplitude changes)
    # Crackles have high derivative energy
    diffs = np.abs(np.diff(full_audio))
    
    # Find periods of high derivative energy (crackling)
    smooth_diffs = np.convolve(diffs, np.ones(256) / 256, mode='valid')
    crackle_threshold = np.std(diffs) * 2.0
    crackle_regions = np.where(smooth_diffs > crackle_threshold)[0]
    
    if len(crackle_regions) > 0:
        print(f"⚠ Potential crackles detected in {len(crackle_regions)} regions")
        print(f"   Max derivative: {np.max(diffs):.4f}")
        print(f"   Mean derivative: {np.mean(diffs):.4f}")
        print(f"   Threshold: {crackle_threshold:.4f}")
    else:
        print("✓ No crackle regions detected (slow envelope attack working)")
    
    # Overall stats
    rms = np.sqrt(np.mean(full_audio ** 2))
    peak = np.max(np.abs(full_audio))
    dc_offset = np.mean(full_audio)
    
    print(f"\nAudio statistics:")
    print(f"  RMS level: {rms:.4f}")
    print(f"  Peak level: {peak:.4f}")
    print(f"  DC offset: {dc_offset:.6f}")
    
    # Clipping check
    if np.any(np.abs(full_audio) > 0.95):
        print(f"  ⚠ Severe clipping detected ({np.sum(np.abs(full_audio) > 0.95)} samples > 0.95)")
    
    return len(crackle_regions) == 0

if __name__ == "__main__":
    success = test_high_neuron_spike_burst()
    print(f"\n{'PASS' if success else 'FAIL'} - Crackle test")
    exit(0 if success else 1)
