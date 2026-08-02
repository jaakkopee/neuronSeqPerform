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

Note On
────────
  Note  → root pitch (MIDI note number)
  Vel   → scale / mode selector (0-127 mapped across SCALE_NAMES)

Aftertouch / Channel Pressure
───────────────────────────────
  Applies to the parameter selected by CC 22.
"""

import threading
import mido

from config import MIDI_CHANNEL, SCALES, SCALE_NAMES


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
            self._cfg["tempo"] = 40.0 + n * 160.0

        elif cc == 9:  # K2  master volume
            self._synth.master_volume = n

        elif cc == 12:  # K3  active operator pairs (texture density)
            pairs = 1 + round(n * 3)
            self._synth.set_active_pairs(pairs)
            self._cfg["active_pairs"] = pairs

        elif cc == 13:  # K4  envelope decay speed (sustain ↔ staccato)
            self._synth.set_decay_speed(n)
            self._cfg["decay_speed"] = n

        elif cc == 14:  # K5  FM modulation index (timbre brightness)
            self._synth.set_mod_index_scale(n * 3.0)

        elif cc == 15:  # K6  quantization strength
            self._cfg["quantization_strength"] = n

        # ── Bank B : network ──────────────────────────────────────────────────
        elif cc == 16:  # K1  network firing threshold
            self._network.set_threshold(0.3 + n * 1.7)

        elif cc == 17:  # K2  network time constant τ
            self._network.set_tau(5.0 + n * 95.0)

        elif cc == 18:  # K3  synaptic weight scale
            self._weight_scale = n * 3.0
            self._network.set_weight_scale(self._weight_scale)

        elif cc == 19:  # K4  global drive
            self._network.set_global_drive(n)

        elif cc == 20:  # K5  swing amount
            self._cfg["swing_amount"] = n * 0.5

        elif cc == 21:  # K6  FM ratio scale
            self._synth.set_ratio_scale(0.5 + n * 1.5)

        # ── Bank C : extras ───────────────────────────────────────────────────
        elif cc == 22:  # K1  aftertouch target selector
            idx = round(n * (len(self.AFTERTOUCH_TARGETS) - 1))
            self.aftertouch_target = self.AFTERTOUCH_TARGETS[idx]
            self._cfg["aftertouch_target"] = self.aftertouch_target
            print(f"[MIDI] Aftertouch target → {self.aftertouch_target}")

    # ── Note On handler  (tonality / modality) ────────────────────────────────
    def _on_note_on(self, note: int, velocity: int) -> None:
        self.root_note = note
        idx = round((velocity / 127.0) * (len(SCALE_NAMES) - 1))
        self.scale_idx  = idx
        self.scale_name = SCALE_NAMES[idx]
        scale           = SCALES[self.scale_name]

        root_freq = 440.0 * (2.0 ** ((note - 69) / 12.0))

        # Map 16 presets across scale degrees (repeat over octaves as needed)
        freqs = []
        for i in range(16):
            degree  = i % len(scale)
            octave  = i // len(scale)
            semitone = scale[degree] + octave * 12
            freqs.append(root_freq * (2.0 ** (semitone / 12.0)))

        self._synth.set_all_base_freqs(freqs)
        # Refresh network display frequencies
        self._network.frequencies = self._synth.get_operator_frequencies()
        self._cfg["scale_name"] = self.scale_name
        self._cfg["root_note"]  = note
        print(f"[MIDI] Root={note}  Scale={self.scale_name}")

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
