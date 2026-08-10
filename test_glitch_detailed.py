#!/usr/bin/env python3
"""
Detailed glitch detection test - simulates sequencer and audio callback timing.
"""

import numpy as np
import time
from model.lif_network import LIFNetwork
from model.fm_synth import make_synth
from config import ROWS, COLS, SAMPLE_RATE, BUFFER_SIZE

def test_sequencer_audio_timing():
    """Simulate realistic sequencer + audio callback timing."""
    
    print("[Test] Initializing network and synth...")
    network = LIFNetwork(rows=ROWS, cols=COLS, neuron_count=512, topology_index=0)
    synth = make_synth()
    
    # Simulate 4 sequencer steps
    num_steps = 4
    sample_count = 0
    max_samples = int(SAMPLE_RATE * 2)  # 2 seconds of audio
    
    audio_frames = []
    glitch_flags = []
    
    print(f"[Test] Running {num_steps} sequencer steps (~{max_samples / SAMPLE_RATE:.1f}s of audio)...\n")
    
    for step_idx in range(num_steps):
        print(f"=== Sequencer Step {step_idx} ===")
        
        # LIF micro-steps (main thread)
        accumulated = None
        for micro_step in range(12):
            spikes_step = network.step()
            if accumulated is None or accumulated.shape != spikes_step.shape:
                accumulated = spikes_step.copy()
            else:
                accumulated |= spikes_step
        
        if accumulated is None:
            print("  Skipped (no spikes)")
            continue
        
        # Downmix and trigger (main thread)
        synth_spikes = network.get_synth_spikes(ROWS, COLS, spike_array=accumulated)
        synth.trigger_and_activate(synth_spikes, step_idx % COLS)
        
        spikes_active = int(np.sum(synth_spikes))
        print(f"  Triggered column {step_idx % COLS} with {spikes_active} spikes")
        
        # Simulate audio generation (audio callback)
        # In real app, this happens asynchronously, but we'll generate it synchronously here
        num_callbacks = int(SAMPLE_RATE * 0.5 / BUFFER_SIZE)  # 500ms worth of callbacks
        
        for cb_idx in range(num_callbacks):
            try:
                samples = synth.generate(BUFFER_SIZE)
                
                # Check for glitches
                has_nan = not np.isfinite(samples).all()
                has_dc_offset = np.abs(np.mean(samples)) > 0.5
                has_large_jump = np.max(np.abs(np.diff(samples))) > 0.9
                
                if has_nan or has_dc_offset or has_large_jump:
                    glitch = "NaN" if has_nan else ("DC_OFFSET" if has_dc_offset else "JUMP")
                    glitch_flags.append((step_idx, cb_idx, glitch, samples))
                    if has_nan:
                        print(f"    ⚠ Callback {cb_idx}: NaN detected!")
                    if has_dc_offset:
                        mean_val = np.mean(samples)
                        print(f"    ⚠ Callback {cb_idx}: Large DC offset ({mean_val:.3f})")
                    if has_large_jump:
                        max_jump = np.max(np.abs(np.diff(samples)))
                        print(f"    ⚠ Callback {cb_idx}: Large amplitude jump ({max_jump:.3f})")
                
                audio_frames.append(samples)
                sample_count += len(samples)
                
            except Exception as e:
                print(f"    ✗ Callback {cb_idx}: Exception: {e}")
                glitch_flags.append((step_idx, cb_idx, "EXCEPTION", str(e)))
                return False
    
    # Analyze results
    print(f"\n=== Analysis ===")
    print(f"Generated {len(audio_frames)} audio buffers ({sample_count} total samples)")
    
    if glitch_flags:
        print(f"\n⚠ Found {len(glitch_flags)} potential glitches:")
        for step, cb, glitch_type, detail in glitch_flags[:10]:  # Show first 10
            print(f"  - Step {step}, callback {cb}: {glitch_type}")
            if isinstance(detail, str):
                print(f"    Details: {detail}")
        if len(glitch_flags) > 10:
            print(f"  ... and {len(glitch_flags) - 10} more")
        return False
    else:
        print("✓ No glitches detected (NaN, DC offset, or large jumps)")
        
        # Concatenate and do additional analysis
        all_audio = np.concatenate(audio_frames)
        rms = np.sqrt(np.mean(all_audio ** 2))
        peak = np.max(np.abs(all_audio))
        
        print(f"  RMS level: {rms:.4f}")
        print(f"  Peak level: {peak:.4f}")
        print(f"  DC offset: {np.mean(all_audio):.6f}")
        
        return True

if __name__ == "__main__":
    try:
        success = test_sequencer_audio_timing()
        exit(0 if success else 1)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
