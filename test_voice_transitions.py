#!/usr/bin/env python3
"""
Aggressive glitch detection test focusing on voice transitions.
Tests rapid switching between voices to check if smooth crossfade eliminates clicks.
"""

import numpy as np
from model.lif_network import LIFNetwork
from model.fm_synth import make_synth
from config import SAMPLE_RATE, BUFFER_SIZE, COLS, ROWS

def test_rapid_voice_switches():
    """Test rapid voice transitions with spike bursts."""
    print("[Test] Initializing network and synth...")
    network = LIFNetwork(rows=ROWS, cols=COLS, topology_index=0)
    synth = make_synth()
    
    # Create spike pattern that alternates columns every step
    accumulated_spikes = np.zeros((ROWS, COLS), dtype=bool)
    
    print("[Test] Running rapid voice switching test (16 steps x 12 micro-steps)...")
    
    audio_buffers = []
    total_samples = 0
    
    for step in range(16):
        # Switch active voice each step
        active_col = step % COLS
        
        # Create spike burst in active column
        accumulated_spikes[:] = False
        if step % 2 == 0:  # Only spike on even steps for clarity
            accumulated_spikes[:, active_col] = True
        
        # Run 12 micro-steps per sequencer step (as in main.py)
        for micro in range(12):
            network.step()
        
        # Get synth spikes and trigger
        synth_spikes = network.get_synth_spikes(ROWS, COLS, spike_array=accumulated_spikes)
        synth.trigger_and_activate(synth_spikes, active_col)
        synth.set_active_step(step)
        
        # Generate audio for this step
        audio = synth.generate(BUFFER_SIZE)
        audio_buffers.append(audio)
        total_samples += len(audio)
    
    # Concatenate all buffers
    full_audio = np.concatenate(audio_buffers)
    
    # Analysis
    print(f"\n=== Analysis ===")
    print(f"Generated {len(audio_buffers)} audio buffers ({total_samples} total samples)")
    
    # Check for glitches at voice transition boundaries
    glitches_found = []
    window_size = BUFFER_SIZE // 4  # Check first quarter of each buffer
    
    for buf_idx in range(1, len(audio_buffers)):
        start_sample = buf_idx * BUFFER_SIZE
        end_sample = start_sample + window_size
        
        if end_sample <= len(full_audio):
            buf_segment = full_audio[start_sample:end_sample]
            diffs = np.abs(np.diff(buf_segment))
            max_diff = np.max(diffs)
            
            # Check for amplitude jumps (discontinuities)
            if max_diff > 0.2:  # Threshold for click detection
                glitches_found.append({
                    'buffer': buf_idx,
                    'max_diff': max_diff,
                    'sample': start_sample + np.argmax(diffs)
                })
    
    if glitches_found:
        print(f"⚠ Found {len(glitches_found)} potential glitches at voice transitions:")
        for g in glitches_found[:5]:
            print(f"   Buffer {g['buffer']}: max amplitude jump = {g['max_diff']:.4f}")
    else:
        print("✓ No amplitude discontinuities detected at voice transitions")
    
    # Overall statistics
    rms = np.sqrt(np.mean(full_audio ** 2))
    peak = np.max(np.abs(full_audio))
    dc_offset = np.mean(full_audio)
    
    print(f"\nAudio statistics:")
    print(f"  RMS level: {rms:.4f}")
    print(f"  Peak level: {peak:.4f}")
    print(f"  DC offset: {dc_offset:.6f}")
    
    # Check for NaN/inf
    if np.any(np.isnan(full_audio)):
        print("  ✗ NaN values detected!")
        return False
    if np.any(np.isinf(full_audio)):
        print("  ✗ Inf values detected!")
        return False
    
    return len(glitches_found) == 0

if __name__ == "__main__":
    success = test_rapid_voice_switches()
    exit(0 if success else 1)
