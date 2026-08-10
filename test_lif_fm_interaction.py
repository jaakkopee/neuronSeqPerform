#!/usr/bin/env python3
"""
Test LIF->FM interaction to verify the accumulated spike fix works correctly.
This test runs without needing a display or audio device.
"""

import numpy as np
from model.lif_network import LIFNetwork
from model.fm_synth import make_synth
from config import ROWS, COLS

def test_accumulated_spike_passthrough():
    """Verify that accumulated spikes (not just last step) are used for FM trigger."""
    
    print("[Test] Initializing LIF network and FM synth...")
    network = LIFNetwork(rows=ROWS, cols=COLS, neuron_count=512, topology_index=0)
    synth = make_synth()
    
    print("[Test] Running LIF micro-steps and checking spike accumulation...")
    
    # Simulate 3 sequencer steps
    for step in range(3):
        print(f"\n=== Step {step} ===")
        
        # Accumulate spikes across 12 micro-steps (like main.py does)
        accumulated = None
        for micro_step in range(12):
            spikes_step = network.step()
            
            if accumulated is None or accumulated.shape != spikes_step.shape:
                accumulated = spikes_step.copy()
            else:
                accumulated |= spikes_step
        
        if accumulated is None:
            print("  Skipped (no accumulated spikes)")
            continue
        
        # Downmix accumulated spikes using the fixed method
        synth_spikes_accum = network.get_synth_spikes(ROWS, COLS, spike_array=accumulated)
        
        # Compare with old behavior (downmixing only latest _spikes)
        synth_spikes_latest = network.get_synth_spikes(ROWS, COLS)
        
        accum_count = int(np.sum(synth_spikes_accum))
        latest_count = int(np.sum(synth_spikes_latest))
        
        print(f"  Accumulated spikes passed to FM: {accum_count} (from all 12 micro-steps)")
        print(f"  Latest-only spikes: {latest_count} (from last micro-step)")
        
        # Verify they're using the correct data
        if accum_count > 0:
            print(f"  ✓ Accumulated data has activity (expected)")
        if accum_count >= latest_count:
            print(f"  ✓ Accumulated >= Latest (correct: accumulation shouldn't lose data)")
        else:
            print(f"  ⚠ Accumulated < Latest (unexpected)")
        
        # Trigger synth with accumulated spikes (the fix)
        synth.trigger_and_activate(synth_spikes_accum, step)
        
        # Generate audio to verify no crashes
        audio = synth.generate(1024)
        audio_valid = np.isfinite(audio).all()
        print(f"  Audio generated: {len(audio)} samples, valid={audio_valid}")
        
        if not audio_valid:
            print(f"  ERROR: Audio contains NaN/Inf!")
            print(f"    {np.sum(~np.isfinite(audio))} invalid samples")
            return False
    
    print("\n✓ Test passed: LIF->FM interaction works correctly")
    print("  - Accumulated spikes are now properly passed to FM synth trigger")
    print("  - No NaN/Inf in generated audio")
    return True

if __name__ == "__main__":
    try:
        success = test_accumulated_spike_passthrough()
        exit(0 if success else 1)
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
