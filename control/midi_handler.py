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
    CC 16   K1  network firing threshold  (0.1 – 2.0)
  CC 17   K2  network time constant τ   (5 – 100 ms)
  CC 18   K3  synaptic weight scale     (0.0 – 3.0)
  CC 19   K4  global external drive     (0.0 – 1.0)
  CC 20   K5  swing amount              (0.0 – 0.5)
  CC 21   K6  FM ratio scale            (0.5 – 2.0)

  BANK C  ── extras ────────────────────────────────────────────────────────────
    CC 22   K1  aftertouch target selector
    CC 23   K2  LIF topology direct select     (Ring..SmallWorld)
    CC 24   K3  LIF neuron count direct select  (128..4096)
    CC 25   K4  LIF steps per tick              (1..64)
    CC 26   K5  root note direct select         (C..B)
    CC 27   K6  scale direct select             (first 10 scales)

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
           72  drive boost
           73  halve weight scale
           74  topology previous
           75  topology next
           76  neuron count down
           77  neuron count up
           78  heterogeneity down
           79  heterogeneity up

Aftertouch / Channel Pressure
───────────────────────────────
  Applies to the parameter selected by CC 22.
"""

import threading
import time
import mido

from config import MIDI_CHANNEL, MIDI_NOTE_CHANNEL, SCALES, SCALE_NAMES, PAD_BANK_A, PAD_BANK_B, PAD_BANK_C

_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def _note_name(midi: int) -> str:
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


class MIDIHandler:
    AFTERTOUCH_TARGETS = [
        "threshold", "tau", "drive", "tempo",
        "quantize",  "swing", "master_vol", "mod_index_scale",
        "active_pairs", "decay_speed", "heterogeneity",
    ]
    NEURON_COUNT_STEPS = [128, 256, 512, 1024, 2048, 4096]
    CC_LABELS = {
        3: "Tempo", 9: "Volume", 12: "Pairs", 13: "Decay", 14: "FMmod", 15: "Quant",
        16: "Thresh", 17: "Tau", 18: "Weight", 19: "Drive", 20: "Swing", 21: "Ratio",
        22: "AT Target", 23: "Topology", 24: "Neurons", 25: "LIF Steps", 26: "Root", 27: "Scale",
    }
    CC_META = {
        3: ("A", 0), 9: ("A", 1), 12: ("A", 2), 13: ("A", 3), 14: ("A", 4), 15: ("A", 5),
        16: ("B", 0), 17: ("B", 1), 18: ("B", 2), 19: ("B", 3), 20: ("B", 4), 21: ("B", 5),
        22: ("C", 0), 23: ("C", 1), 24: ("C", 2), 25: ("C", 3), 26: ("C", 4), 27: ("C", 5),
    }
    BANK_C_PAD_LABELS = [
        "RndW", "Reset", "+Step", "-Step", "Boost", "HalfW",
        "Topo-", "Topo+", "N--", "N++",
        "Het-", "Het+", "---", "---", "---", "---",
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
        self._cfg.setdefault("noteon_flash", {})
        self._cfg.setdefault("last_midi", None)
        self._sync_heterogeneity_cfg()

        # quick static audit: ensure all main runtime parameters are reachable
        # via knobs, pads, or aftertouch routing.
        self._audit_controller_coverage()

    def _set_last_cc(self, cc: int) -> None:
        label = self.CC_LABELS.get(cc, f"CC{cc}")
        bank, slot = self.CC_META.get(cc, ("?", -1))
        self._cfg["last_midi"] = {
            "kind": "cc",
            "cc": cc,
            "bank": bank,
            "slot": slot,
            "label": label,
            "timestamp": time.monotonic(),
        }

    def _set_last_pad(self, note: int, bank: str, slot: int, label: str) -> None:
        self._cfg["last_midi"] = {
            "kind": "pad",
            "note": note,
            "bank": bank,
            "slot": slot,
            "label": label,
            "timestamp": time.monotonic(),
        }

    def _audit_controller_coverage(self) -> None:
        required = {
            "tempo", "master_volume", "active_pairs", "decay_speed",
            "mod_index_scale", "quantization_strength", "threshold", "tau",
            "weight_scale", "drive_n", "swing_amount", "ratio_scale",
            "aftertouch_target", "topology_index", "neuron_count",
            "lif_steps", "root_note", "scale_name", "heterogeneity",
        }

        knob_controls = {
            "tempo", "master_volume", "active_pairs", "decay_speed",
            "mod_index_scale", "quantization_strength", "threshold", "tau",
            "weight_scale", "drive_n", "swing_amount", "ratio_scale",
            "aftertouch_target", "topology_index", "neuron_count",
            "lif_steps", "root_note", "scale_name",
        }
        pad_controls = {"root_note", "scale_name", "lif_steps", "topology_index", "neuron_count"}
        aftertouch_controls = {
            "threshold", "tau", "drive_n", "tempo", "quantization_strength",
            "swing_amount", "master_volume", "mod_index_scale", "active_pairs", "decay_speed",
            "heterogeneity",
        }

        exposed = knob_controls | pad_controls | aftertouch_controls
        missing = sorted(required - exposed)
        self._cfg["controller_coverage_ok"] = (len(missing) == 0)
        self._cfg["controller_coverage_missing"] = missing

        if missing:
            print(f"[MIDI] Controller mapping missing: {missing}")
        else:
            print(f"[MIDI] Controller mapping OK ({len(required)} parameters reachable)")

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
        # CCs and aftertouch come from the knob channel (0)
        # Pad note-ons come from the drum channel (9) on the MPD218
        if msg.type == "control_change" and msg.channel == MIDI_CHANNEL:
            self._on_cc(msg.control, msg.value)
        elif msg.type == "note_on" and msg.velocity > 0 and msg.channel == MIDI_NOTE_CHANNEL:
            self._on_note_on(msg.note, msg.velocity)
        elif msg.type in ("aftertouch", "polytouch") and msg.channel in (MIDI_CHANNEL, MIDI_NOTE_CHANNEL): self._on_aftertouch(msg.value)

    # ── CC handler ────────────────────────────────────────────────────────────
    def _on_cc(self, cc: int, value: int) -> None:
        n = value / 127.0   # normalised 0-1
        if cc in self.CC_LABELS:
            self._set_last_cc(cc)

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
            v = 0.1 + n * 1.9
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

        elif cc == 23:  # K2  direct topology select
            idx = round(n * 4.0)
            self._network.set_topology_index(idx)
            self._cfg["topology_index"] = self._network.topology_index
            self._cfg["topology_name"] = self._network.topology_name()
            print(f"[MIDI] Topology → {self._cfg['topology_name']}")

        elif cc == 24:  # K3  direct neuron count select
            idx = round(n * (len(self.NEURON_COUNT_STEPS) - 1))
            idx = max(0, min(idx, len(self.NEURON_COUNT_STEPS) - 1))
            ncount = self._network.set_neuron_count(self.NEURON_COUNT_STEPS[idx])
            self._cfg["neuron_count"] = ncount
            print(f"[MIDI] Neurons → {ncount}")

        elif cc == 25:  # K4  direct LIF steps select
            steps = 1 + round(n * 63.0)
            self._cfg["lif_steps"] = steps
            print(f"[MIDI] LIF steps → {steps}")

        elif cc == 26:  # K5  direct root note select (pitch class)
            pitch_class = round(n * 11.0)
            self.root_note = 60 + pitch_class
            self._apply_tonality()

        elif cc == 27:  # K6  direct scale select
            idx = round(n * (len(SCALE_NAMES) - 1))
            idx = max(0, min(idx, len(SCALE_NAMES) - 1))
            self.scale_idx = idx
            self.scale_name = SCALE_NAMES[idx]
            self._apply_tonality()

    # ── Note On handler ───────────────────────────────────────────────────────
    def _on_note_on(self, note: int, velocity: int) -> None:
        lo_a, hi_a = PAD_BANK_A
        lo_b, hi_b = PAD_BANK_B
        lo_c, hi_c = PAD_BANK_C

        # Record timestamp for flash indicator (view reads this)
        self._cfg["noteon_flash"][note] = time.monotonic()

        if lo_a <= note <= hi_a:
            slot = note - lo_a
            label = _NOTE_NAMES[slot % 12]
            self._set_last_pad(note, "A", slot, label)
            # ── Bank A : root note (pitch class → octave 4) ───────────────────
            pitch_class    = (note - lo_a) % 12
            self.root_note = 60 + pitch_class       # C4=60 … B4=71
            # Velocity: harder hit → stronger drive burst
            drive_n = 0.4 + (velocity / 127.0) * 0.6
            self._network.set_global_drive(drive_n)
            self._cfg["drive_n"] = drive_n
            self._apply_tonality()

        elif lo_b <= note <= hi_b:
            slot = note - lo_b
            label = SCALE_NAMES[slot] if slot < len(SCALE_NAMES) else "---"
            self._set_last_pad(note, "B", slot, label)
            # ── Bank B : scale / mode selection ──────────────────────────────
            idx = note - lo_b                       # 0-15
            if idx < len(SCALE_NAMES):
                self.scale_idx  = idx
                self.scale_name = SCALE_NAMES[idx]
            self._apply_tonality()

        elif lo_c <= note <= hi_c:
            slot = note - lo_c
            label = self.BANK_C_PAD_LABELS[slot] if slot < len(self.BANK_C_PAD_LABELS) else "---"
            self._set_last_pad(note, "C", slot, label)
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
        vel_n     = velocity / 127.0

        if func == 0:    # randomise weights
            self._network.randomize_weights()
            print("[MIDI] Weights randomised")
        elif func == 1:  # reset potentials
            self._network.reset_state()
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
            hi = 0.6 + vel_n * 0.4
            self._network.set_global_drive(hi)
            self._cfg["drive_n"] = hi
            print(f"[MIDI] Drive randomised  hi={hi:.2f}")
        elif func == 5:  # invert weights
            self._network.set_weight_scale(max(0.0, self._weight_scale * 0.5))
            self._weight_scale = max(0.0, self._weight_scale * 0.5)
            self._cfg["weight_scale"] = self._weight_scale
            print(f"[MIDI] Weight scale halved → {self._weight_scale:.2f}")
        elif func == 6:  # previous topology
            idx = self._network.cycle_topology(-1)
            self._cfg["topology_index"] = idx
            self._cfg["topology_name"] = self._network.topology_name()
            print(f"[MIDI] Topology → {self._cfg['topology_name']}")
        elif func == 7:  # next topology
            idx = self._network.cycle_topology(+1)
            self._cfg["topology_index"] = idx
            self._cfg["topology_name"] = self._network.topology_name()
            print(f"[MIDI] Topology → {self._cfg['topology_name']}")
        elif func == 8:  # smaller neuron count
            ncount = self._network.nudge_neuron_count_step(-1)
            self._cfg["neuron_count"] = ncount
            print(f"[MIDI] Neurons → {ncount}")
        elif func == 9:  # larger neuron count
            ncount = self._network.nudge_neuron_count_step(+1)
            self._cfg["neuron_count"] = ncount
            print(f"[MIDI] Neurons → {ncount}")
        elif func == 10:  # heterogeneity down
            v = max(0.0, float(self._cfg.get("heterogeneity", 0.0)) - 0.08)
            self._network.set_heterogeneity(v)
            self._sync_heterogeneity_cfg()
            print(f"[MIDI] Heterogeneity → {v:.2f}")
        elif func == 11:  # heterogeneity up
            v = min(1.0, float(self._cfg.get("heterogeneity", 0.0)) + 0.08)
            self._network.set_heterogeneity(v)
            self._sync_heterogeneity_cfg()
            print(f"[MIDI] Heterogeneity → {v:.2f}")
        # funcs 12-15 spare

    def _sync_heterogeneity_cfg(self) -> None:
        state = self._network.heterogeneity_state()
        self._cfg.update(state)

    # ── Aftertouch handler ────────────────────────────────────────────────────
    def _on_aftertouch(self, value: int) -> None:
        n = value / 127.0
        t = self.aftertouch_target

        if t == "threshold":
            self._network.set_threshold(0.1 + n * 1.9)
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
        elif t == "heterogeneity":
            self._network.set_heterogeneity(n)
            self._sync_heterogeneity_cfg()
