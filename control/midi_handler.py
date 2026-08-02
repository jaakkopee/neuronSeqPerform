"""
MIDI input handler (runs in a daemon thread).

MPD218 knob layout  (all CCs 0-127, channel 0)
────────────────────────────────────────────────
  BANK A  ── performance controls ─────────────────────────────────────────────
  CC  3   K1  master tempo              (40 – 200 BPM)
  CC  9   K2  master volume             (0.0 – 1.0)
  CC 12   K3  active operator pairs     (1 – 4)        texture density
  CC 13   K4  envelope decay speed      (0.0 – 1.0)    sustain ↔ staccato
  CC 14   K5  FM modulation index       (0.0 – 3.0)    clean ↔ dense spectrum
  CC 15   K6  quantization strength     (0.0 – 1.0)

  BANK B  ── network controls ─────────────────────────────────────────────────
  CC 16   K1  network firing threshold  (0.3 – 2.0)
  CC 17   K2  network time constant τ   (5 – 100 ms)
  CC 18   K3  synaptic weight scale     (0.0 – 3.0)
  CC 19   K4  global external drive     (0.0 – 1.0)
  CC 20   K5  swing amount              (0.0 – 0.5)
  CC 21   K6  FM ratio scale            (0.5 – 2.0)

  BANK C  ── extras ────────────────────────────────────────────────────────────
  CC 22   K1  aftertouch target selector

Note On  (MPD218 pads, channel 9)
────────────────────────────────
  Bank A  notes 36-51  → root note  (pitch class of pad, octave 4)
           velocity    → temporary drive boost
  Bank B  notes 52-67  → scale / mode  (one pad per scale)
  Bank C  notes 68-83  → network functions
           68  randomise weights
           69  reset potentials
           70  +4 LIF steps/tick  (denser patterns)
           71  -4 LIF steps/tick  (sparser patterns)
           72  randomise drive
           73  invert weights

Aftertouch / Channel Pressure
───────────────────────────────
  Applies to the parameter selected by CC 22.
"""

import threading
import numpy as np
import mido

from config import MIDI_CHANNEL, SCALES, SCALE_NAMES, PAD_BANK_A, PAD_BANK_B, PAD_BANK_C

_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def _note_name(midi: int) -> str:
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


class MIDIHandler:
    AFTERTOUCH_TARGETS = [
        "threshold", "tau", "drive", "tempo",
        "quantize",  "swing", "master_vol", "mod_index_scale",
        "active_pairs", "decay_speed",
    ]

    # ── construction ──────────────────────────────────────────────────────────
    def __init__(self, network, synth, config_state: dict):
        self._network      = network
        self._synth        = synth
        self._cfg          = config_state
        self._port         = None
        self._running      = False
        self._thread       = None

        # aftertouch routing
        self.aftertouch_target = "threshold"

        # scale / tonality state (readable from main loop for display)
        self.root_note  = 60          # C4
        self.scale_idx  = 1           # major
        self.scale_name = SCALE_NAMES[1]

        # internal: track absolute weight scale to avoid cumulative drift
        self._weight_scale = 1.0

    # ── port management ───────────────────────────────────────────────────────
    def open_port(self, port_name: str = None) -> bool:
        available = mido.get_input_names()
        if not available:
            print("[MIDI] No input ports found – running without MIDI.")
            return False
        target = None
        if port_name:
            # Case-insensitive substring search
            needle = port_name.lower()
            for name in available:
                if needle in name.lower():
                    target = name
                    break
        if target is None:
            target = available[0]
            if port_name:
                print(f"[MIDI] '{port_name}' not found. Available: {available}")
                print(f"[MIDI] Falling back to: {target}")
        self._port = mido.open_input(target)
        print(f"[MIDI] Opened: {target}")
        return True

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True, name="midi-rx")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._port:
            self._port.close()

    # ── main receive loop ─────────────────────────────────────────────────────
    def _run(self) -> None:
        while self._running and self._port:
            for msg in self._port.iter_pending():
                self._dispatch(msg)

    def _dispatch(self, msg) -> None:
        ch = MIDI_CHANNEL
        if msg.type == "control_change" and msg.channel == ch:
            self._on_cc(msg.control, msg.value)
        elif msg.type == "note_on" and msg.velocity > 0 and msg.channel == ch:
            self._on_note_on(msg.note, msg.velocity)
        elif msg.type in ("aftertouch", "polytouch") and msg.channel == ch:
            val = msg.value if msg.type == "aftertouch" else msg.value
            self._on_aftertouch(val)

    # ── CC handler ────────────────────────────────────────────────────────────
    def _on_cc(self, cc: int, value: int) -> None:
        n = value / 127.0   # normalised 0-1

        # ── Bank A : performance ───────────────────────────────────────────────
        if cc == 3:    # K1  tempo
            v = 40.0 + n * 160.0
            self._cfg["tempo"] = v

        elif cc == 9:  # K2  master volume
            self._synth.master_volume = n
            self._cfg["master_volume"] = n

        elif cc == 12:  # K3  active operator pairs (texture density)
            pairs = 1 + round(n * 3)
            self._synth.set_active_pairs(pairs)
            self._cfg["active_pairs"] = pairs

        elif cc == 13:  # K4  envelope decay speed (sustain ↔ staccato)
            self._synth.set_decay_speed(n)
            self._cfg["decay_speed"] = n

        elif cc == 14:  # K5  FM modulation index (timbre brightness)
            v = n * 3.0
            self._synth.set_mod_index_scale(v)
            self._cfg["mod_index_scale"] = v

        elif cc == 15:  # K6  quantization strength
            self._cfg["quantization_strength"] = n

        # ── Bank B : network ──────────────────────────────────────────────────
        elif cc == 16:  # K1  network firing threshold
            v = 0.3 + n * 1.7
            self._network.set_threshold(v)
            self._cfg["threshold"] = v

        elif cc == 17:  # K2  network time constant τ
            v = 5.0 + n * 95.0
            self._network.set_tau(v)
            self._cfg["tau"] = v

        elif cc == 18:  # K3  synaptic weight scale
            v = n * 3.0
            self._weight_scale = v
            self._network.set_weight_scale(v)
            self._cfg["weight_scale"] = v

        elif cc == 19:  # K4  global drive
            self._network.set_global_drive(n)
            self._cfg["drive_n"] = n

        elif cc == 20:  # K5  swing amount
            v = n * 0.5
            self._cfg["swing_amount"] = v

        elif cc == 21:  # K6  FM ratio scale
            v = 0.5 + n * 1.5
            self._synth.set_ratio_scale(v)
            self._cfg["ratio_scale"] = v

        # ── Bank C : extras ───────────────────────────────────────────────────
        elif cc == 22:  # K1  aftertouch target selector
            idx = round(n * (len(self.AFTERTOUCH_TARGETS) - 1))
            self.aftertouch_target = self.AFTERTOUCH_TARGETS[idx]
            self._cfg["aftertouch_target"] = self.aftertouch_target
            print(f"[MIDI] Aftertouch target → {self.aftertouch_target}")

    # ── Note On handler ───────────────────────────────────────────────────────
    def _on_note_on(self, note: int, velocity: int) -> None:
        lo_a, hi_a = PAD_BANK_A
        lo_b, hi_b = PAD_BANK_B
        lo_c, hi_c = PAD_BANK_C

        if lo_a <= note <= hi_a:
            # ── Bank A : root note (pitch class → octave 4) ───────────────────
            pitch_class    = (note - lo_a) % 12
            self.root_note = 60 + pitch_class       # C4=60 … B4=71
            # Velocity: harder hit → stronger drive burst
            drive_n = 0.4 + (velocity / 127.0) * 0.6
            self._network.set_global_drive(drive_n)
            self._cfg["drive_n"] = drive_n
            self._apply_tonality()

        elif lo_b <= note <= hi_b:
            # ── Bank B : scale / mode selection ──────────────────────────────
            idx = note - lo_b                       # 0-15
            if idx < len(SCALE_NAMES):
                self.scale_idx  = idx
                self.scale_name = SCALE_NAMES[idx]
            self._apply_tonality()

        elif lo_c <= note <= hi_c:
            # ── Bank C : network functions ────────────────────────────────────
            self._network_function(note - lo_c, velocity)

    def _apply_tonality(self) -> None:
        """Recompute 16 preset frequencies from current root_note + scale_name."""
        scale     = SCALES[self.scale_name]
        root_freq = 440.0 * (2.0 ** ((self.root_note - 69) / 12.0))
        freqs     = []
        for i in range(16):
            degree   = i % len(scale)
            octave   = i // len(scale)
            semitone = scale[degree] + octave * 12
            freqs.append(root_freq * (2.0 ** (semitone / 12.0)))
        self._synth.set_all_base_freqs(freqs)
        self._network.frequencies = self._synth.get_operator_frequencies()
        self._cfg["root_note"]  = self.root_note
        self._cfg["scale_name"] = self.scale_name
        print(f"[MIDI] Root {_note_name(self.root_note)}  Scale {self.scale_name}")

    def _network_function(self, func: int, velocity: int) -> None:
        """Execute a pad-triggered network function (Bank C, func = note - 68)."""
        n_neurons = len(self._network.external_drive)
        vel_n     = velocity / 127.0

        if func == 0:    # randomise weights
            self._network.randomize_weights()
            print("[MIDI] Weights randomised")
        elif func == 1:  # reset potentials
            self._network.v[:]          = 0.0
            self._network.refractory[:] = 0.0
            print("[MIDI] Network potentials reset")
        elif func == 2:  # +4 LIF steps
            steps = min(64, self._cfg.get("lif_steps", 12) + 4)
            self._cfg["lif_steps"] = steps
            print(f"[MIDI] LIF steps → {steps}")
        elif func == 3:  # -4 LIF steps
            steps = max(1, self._cfg.get("lif_steps", 12) - 4)
            self._cfg["lif_steps"] = steps
            print(f"[MIDI] LIF steps → {steps}")
        elif func == 4:  # randomise drive (velocity scales upper bound)
            hi = 1.0 + vel_n * 1.5
            self._network.external_drive[:] = np.random.uniform(1.0, hi, n_neurons).astype(np.float32)
            self._cfg["drive_n"] = (1.0 + hi) / 2.0 / 2.0  # approx mid for display
            print(f"[MIDI] Drive randomised  hi={hi:.2f}")
        elif func == 5:  # invert weights
            self._network.weights *= -1
            print("[MIDI] Weights inverted")
        # funcs 6-15 spare

    # ── Aftertouch handler ────────────────────────────────────────────────────
    def _on_aftertouch(self, value: int) -> None:
        n = value / 127.0
        t = self.aftertouch_target

        if t == "threshold":
            self._network.set_threshold(0.3 + n * 1.7)
        elif t == "tau":
            self._network.set_tau(5.0 + n * 95.0)
        elif t == "drive":
            self._network.set_global_drive(n)
        elif t == "tempo":
            self._cfg["tempo"] = 40.0 + n * 160.0
        elif t == "quantize":
            self._cfg["quantization_strength"] = n
        elif t == "swing":
            self._cfg["swing_amount"] = n * 0.5
        elif t == "master_vol":
            self._synth.master_volume = n
        elif t == "mod_index_scale":
            self._synth.set_mod_index_scale(n * 3.0)
        elif t == "active_pairs":
            pairs = 1 + round(n * 3)
            self._synth.set_active_pairs(pairs)
            self._cfg["active_pairs"] = pairs
        elif t == "decay_speed":
            self._synth.set_decay_speed(n)
            self._cfg["decay_speed"] = n
