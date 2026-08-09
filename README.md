# NeuronSeqPerform

A live-performance sequencer driven by a **Leaky Integrate-and-Fire (LIF) neural network** whose rhythm and density model the role of **central pattern generators (CPGs)** in the limbic system. Spike events are routed directly into an FM synthesiser, turning biological firing patterns into music.

---

## Philosophy: LIF Neurons as Limbic CPGs

### Leaky Integrate-and-Fire neurons

The LIF model is the simplest biologically plausible neuron. Each neuron accumulates input current and leaks charge over a membrane time constant τ. When the membrane potential crosses a threshold *v_thresh* it fires a spike and resets:

```
τ · dv/dt = -(v - v_rest) + I_total
```

If `v ≥ v_thresh`:  fire, reset `v = v_rest`, enter refractory period.

The three tuneable parameters — **threshold**, **time constant τ**, and **synaptic weight scale** — map directly to the controls available at performance time.

### Central pattern generators in the limbic system

Central pattern generators are recurrently connected neural circuits capable of producing **autonomous, rhythmic output** without rhythmic sensory input. In vertebrates they underpin locomotion, breathing, and — relevantly here — the **rhythmic oscillations of limbic structures** (hippocampus, amygdala, cingulate cortex) that modulate emotional tone, arousal, and the sense of groove that underlies music perception.

NeuronSeqPerform simulates this principle:

| Biological concept | Implementation |
|---|---|
| CPG burst rhythm | LIF micro-steps fire 1–64 times per sequencer tick; accumulated spikes gate FM voices |
| Limbic oscillation frequency | Tempo (40–200 BPM) × LIF steps per tick controls burst rate |
| Recurrent excitation | Synaptic weight matrix (Ring / FullyConnected / SmallWorld …) |
| Tonic drive / arousal | Global external drive knob biases all neurons toward or away from threshold |
| Refractoriness / silence | Refractory period (`refractory_t = 5 ms`) prevents re-entrant runaway firing |
| Population size scaling | Neuron count 128–4096; more neurons → smoother, more complex spike patterns |

### Network topologies

Each topology produces a characteristic rhythmic texture:

| Name | Character |
|---|---|
| **Ring** | Travelling wave — phase-locked, metronomic |
| **FullyConnected** | Dense excitation — full, busy patterns |
| **Feedforward** | Directed cascade — accents and rolls |
| **SparseRandom** | Irregular bursts — polyrhythmic, unpredictable |
| **SmallWorld** | Hub-driven clusters — swing and syncopation |

### From spikes to sound

Every sequencer tick the spike matrix is pooled into an 8 × 16 operator grid (8 FM operators × 16 presets). Operators whose column is the current step receive a gate trigger; their carrier frequency is drawn from a musical scale seeded at startup. Modulation index, ratio, decay, and active operator count are all MIDI-controllable in real time, so the timbre morphs continuously with the network dynamics.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | ≥ 3.9 |
| numpy | ≥ 1.24 |
| pygame | ≥ 2.5 |
| mido | ≥ 1.3 |
| python-rtmidi | ≥ 1.5 |
| sounddevice | ≥ 0.4.6 |

Optional (macOS only, dramatically faster):

| Dependency | Notes |
|---|---|
| pybind11 | For building the Metal-accelerated native extension |
| Xcode command-line tools | `xcode-select --install` |

---

## Build

### 1 — Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — (macOS only) Build the Metal-accelerated native extension

The native backend runs the LIF network and FM synthesis on Apple GPU/Metal, giving a significant performance boost for large neuron counts.

```bash
pip install pybind11 setuptools
cd synth_native
python setup.py build_ext --inplace
cd ..
```

If the build succeeds you will see `synth_native/_fm_synth.*.so`. If it fails or you are on Linux/Windows the app automatically falls back to the pure-NumPy backend.

---

## Configuration

Edit `config.py` before launching to match your system:

```python
MIDI_PORT_NAME = "MPD218"   # substring match; "" = first available port
MASTER_TEMPO   = 120.0      # default BPM
SAMPLE_RATE    = 44100
BUFFER_SIZE    = 1024       # increase if you hear audio glitches
```

To list available MIDI ports without starting the full application:

```bash
python scan_midi.py
```

---

## Running

```bash
python main.py
```

A pygame window opens showing the 8 × 16 neuron/operator matrix. Cells brighten on spike. The status bar at the bottom shows the active topology, neuron count, current scale, and the last received MIDI message.

Window and session controls:

| Key | Action |
|---|---|
| F11 or Alt+Enter | Toggle fullscreen |
| Esc | Exit fullscreen (if active), otherwise quit |
| 0–9 | Launch scene slot 0–9 |
| Shift+0–9 | Save current state to scene slot 0–9 |
| A | Toggle anti-lock controller |
| - / = | Anti-lock strength down/up |

Quit with the **window close button** or **Escape**.

---

## MIDI Mapping — Akai MPD218

The controller is divided into **three knob banks (A/B/C)** and **three pad banks (A/B/C)**. All knob CCs are on **channel 1 (0-indexed: 0)**. All pad notes are on **channel 10 (0-indexed: 9)**.

### Knob Bank A — Performance Controls

| Knob | CC | Parameter | Range | Notes |
|---|---|---|---|---|
| K1 | 3 | Master Tempo | 40 – 200 BPM | |
| K2 | 9 | Master Volume | 0.0 – 1.0 | |
| K3 | 12 | Active Operator Pairs | 1 – 4 | Texture density |
| K4 | 13 | Envelope Decay Speed | 0.0 – 1.0 | 0 = sustain, 1 = staccato |
| K5 | 14 | FM Modulation Index | 0.0 – 3.0 | Clean ↔ dense spectrum |
| K6 | 15 | Quantization Strength | 0.0 – 1.0 | 0 = loose, 1 = locked |

### Knob Bank B — Network Controls

| Knob | CC | Parameter | Range | Notes |
|---|---|---|---|---|
| K1 | 16 | Firing Threshold | 0.1 – 2.0 | Lower = more spikes |
| K2 | 17 | Time Constant τ | 5 – 100 ms | Shorter = faster integration |
| K3 | 18 | Synaptic Weight Scale | 0.0 – 3.0 | Gain of inter-neuron coupling |
| K4 | 19 | Global External Drive | 0.0 – 1.0 | Internally mapped to LIF drive 0.25 – 3.0 |
| K5 | 20 | Swing Amount | 0.0 – 0.5 | Odd-step delay fraction |
| K6 | 21 | FM Ratio Scale | 0.5 – 2.0 | Harmonic ↔ inharmonic timbre |

### Knob Bank C — Extras

| Knob | CC | Parameter | Range / Options | Notes |
|---|---|---|---|---|
| K1 | 22 | Aftertouch Target | 0–22 (see below) | Routes channel pressure to a parameter |
| K2 | 23 | LIF Topology | 0–4 (direct select) | Ring / FullyConnected / Feedforward / SparseRandom / SmallWorld |
| K3 | 24 | Neuron Count | 0–5 (index) | Steps: 128 / 256 / 512 / 1024 / 2048 / 4096 |
| K4 | 25 | LIF Steps per Tick | 1 – 64 | Micro-steps per sequencer step |
| K5 | 26 | Root Note | C–B (direct) | 12 pitch classes |
| K6 | 27 | Scale | 0–9 (direct) | First 10 scales (see scale list below) |

#### Aftertouch targets (CC 22 value → parameter)

| CC 22 value | Target parameter |
|---|---|
| 0 | Firing Threshold |
| 1 | Time Constant τ |
| 2 | Global Drive |
| 3 | Tempo |
| 4 | Quantization Strength |
| 5 | Swing Amount |
| 6 | Master Volume |
| 7 | FM Modulation Index |
| 8 | Active Operator Pairs |
| 9 | Envelope Decay Speed |
| 10 | Heterogeneity |
| 11 | Inhibitory Ratio |
| 12 | Inhibitory Gain |
| 13 | Delay Spread |
| 14 | Delay Jitter |
| 15 | Adaptation Strength |
| 16 | Adaptation Decay |
| 17 | Noise Amount |
| 18 | Noise Color (white/pink split at 0.5) |
| 19 | Spatial Noise |
| 20 | Anti-lock Enable |
| 21 | Anti-lock Strength |
| 22 | Scene Morph Time (0–8 s) |

### Pad Bank A — Root Note Selection (notes 36–51, channel 10)

Pads select the **root pitch class** (C–D♯ mapped from the 16 pads). Velocity applies a temporary drive boost to the network.

| Pads | Notes | Function |
|---|---|---|
| Pad 1–16 | 36–51 | Set root note (pitch class of pad, octave 4) |

### Pad Bank B — Scale / Mode Selection (notes 52–67, channel 10)

Each pad selects one of the 10 available scales:

| Pad | Note | Scale |
|---|---|---|
| 1 | 52 | chromatic |
| 2 | 53 | major |
| 3 | 54 | minor |
| 4 | 55 | dorian |
| 5 | 56 | phrygian |
| 6 | 57 | lydian |
| 7 | 58 | mixolydian |
| 8 | 59 | locrian |
| 9 | 60 | pentatonic |
| 10 | 61 | blues |
| 11–16 | 62–67 | — (unassigned) |

### Pad Bank C — Network Functions (notes 68–83, channel 10)

| Pad | Note | Function |
|---|---|---|
| 1 | 68 | Randomise synaptic weights |
| 2 | 69 | Reset membrane potentials |
| 3 | 70 | +4 LIF steps per tick (denser patterns) |
| 4 | 71 | −4 LIF steps per tick (sparser patterns) |
| 5 | 72 | Drive boost (momentary) |
| 6 | 73 | Halve weight scale |
| 7 | 74 | Previous topology |
| 8 | 75 | Next topology |
| 9 | 76 | Decrease neuron count (one step) |
| 10 | 77 | Increase neuron count (one step) |
| 11 | 78 | Heterogeneity down |
| 12 | 79 | Heterogeneity up |
| 13 | 80 | Inhibitory ratio down |
| 14 | 81 | Inhibitory ratio up |
| 15 | 82 | Delay spread down |
| 16 | 83 | Delay spread up |

### Channel Pressure (Aftertouch)

Applies to whichever parameter is currently selected by **CC 22**. Useful for expressive real-time modulation without occupying a knob — press harder on any pad to sweep the target parameter.

Noise parameters are controlled through this route:

| Aftertouch target | Effect |
|---|---|
| noise_amount | Continuous 0.0–1.0 |
| noise_color | <0.5 = white, >=0.5 = pink |
| spatial_noise | Continuous 0.0–1.0 |

Note: with anti-lock enabled, the controller may also nudge `noise_amount` during collapse recovery.

---

## Available Scales

| Name | Intervals (semitones) |
|---|---|
| chromatic | 0 1 2 3 4 5 6 7 8 9 10 11 |
| major | 0 2 4 5 7 9 11 |
| minor | 0 2 3 5 7 8 10 |
| dorian | 0 2 3 5 7 9 10 |
| phrygian | 0 1 3 5 7 8 10 |
| lydian | 0 2 4 6 7 9 11 |
| mixolydian | 0 2 4 5 7 9 10 |
| locrian | 0 1 3 5 6 8 10 |
| pentatonic | 0 2 4 7 9 |
| blues | 0 3 5 6 7 10 |

---

## Architecture Overview

```
main.py
├── model/lif_network.py     LIF simulation (NumPy fallback + Metal native)
├── model/fm_synth.py        FM voice engine
├── control/midi_handler.py  MPD218 MIDI receiver (daemon thread)
├── view/matrix_view.py      Pygame display
├── config.py                All tuneable constants
└── synth_native/            Pybind11 Metal extension (macOS optional)
    ├── LIFNetworkNative.mm
    ├── FMSynthMetal.mm
    └── bindings.cpp
```

Each sequencer tick:
1. Run N LIF micro-steps → accumulate spike matrix
2. Pool spike matrix into 8 × 16 operator grid
3. Gate FM voices from spike grid at the current step column
4. Update pygame display
5. Advance step pointer (tempo + swing + quantisation jitter)

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No audio | sounddevice can't open device | Set `BUFFER_SIZE` higher in `config.py`; check system audio permissions |
| No MIDI | Port name mismatch | Run `python scan_midi.py` and update `MIDI_PORT_NAME` in `config.py` |
| Native extension not loading | macOS only / build not run | Run the `synth_native/setup.py` build step; app continues with NumPy fallback |
| Silence despite spikes | Threshold too high or drive too low | Turn down CC 16 or up CC 19 |
| Runaway firing / clipping | Weight scale or drive too high | Turn down CC 18 or CC 19 |
