#!/usr/bin/env python3
"""Check which backend (Metal GPU or Python fallback) is being used."""

from model.fm_synth import make_synth
from model.lif_network import LIFNetwork
from config import ROWS, COLS

print("=" * 60)
print("BACKEND DIAGNOSTIC")
print("=" * 60)

# Check LIF backend
print("\n[LIF Network]")
lif = LIFNetwork(rows=ROWS, cols=COLS)
if hasattr(lif, '_core') and lif._core is not None:
    print("  ✓ Using Metal-accelerated LIFNetwork (native backend)")
else:
    print("  ⚠ Using NumPy fallback for LIFNetwork (no GPU acceleration)")

# Check FM backend
print("\n[FM Synth]")
synth = make_synth()
synth_class = type(synth).__name__
if synth_class == "NativeFMSynth":
    print("  ✓ Using Metal-accelerated FMSynth (native backend)")
elif synth_class == "FMSynth":
    print("  ⚠ Using pure-Python FMSynth fallback (no GPU acceleration)")
else:
    print(f"  ? Using {synth_class}")

print("\n" + "=" * 60)
print("If both show fallback, glitches may be from the Python synth.")
print("The voice crossfade fix has just been applied to both backends.")
print("=" * 60)
