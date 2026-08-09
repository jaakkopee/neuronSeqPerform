"""
NeuronSeqPerform – main entry point.

Loop flow (each sequencer tick)
───────────────────────────────
  1. Run N LIF micro-steps  →  spike matrix
  2. Couple spikes into FM synth envelopes
  3. Update view state (frequencies, potentials, spikes)
  4. Advance to next step column  (tempo + swing + quantization jitter)
  5. Draw frame via pygame
  6. Repeat

Audio runs asynchronously in a sounddevice callback thread.
MIDI runs in a separate daemon thread.
"""

import sys
import time
from collections import deque
import numpy as np

import sounddevice as sd
import pygame

from config import (
    SAMPLE_RATE, BUFFER_SIZE, COLS, ROWS,
    MASTER_TEMPO, QUANTIZATION_STRENGTH, SWING_AMOUNT,
    SCALES, SCALE_NAMES, MIDI_PORT_NAME,
)
from model.lif_network import LIFNetwork
from model.fm_synth    import make_synth
from control.midi_handler import MIDIHandler
from view.matrix_view     import MatrixView


# ── helpers ────────────────────────────────────────────────────────────────────
def _build_initial_freqs(root_midi: int = 60, scale_name: str = "major") -> list:
    """Map 16 preset slots onto a scale starting from root_midi."""
    scale     = SCALES[scale_name]
    root_freq = 440.0 * (2.0 ** ((root_midi - 69) / 12.0))
    freqs     = []
    for i in range(COLS):
        degree   = i % len(scale)
        octave   = i // len(scale)
        semitone = scale[degree] + octave * 12
        freqs.append(root_freq * (2.0 ** (semitone / 12.0)))
    return freqs


def _compute_synchrony_index(spike_history: deque[np.ndarray]) -> float:
    """Return a bounded synchrony score in [0, 1] from recent spike frames.

    Uses a population-coupling metric:
      S = var_t(mean_n x_tn) / mean_n(var_t(x_tn))
    where x_tn is binary activity for neuron n at step t.
    """
    if len(spike_history) < 2:
        return 0.0

    x = np.stack(spike_history, axis=0).astype(np.float32, copy=False)  # (T, N)
    pop_rate = x.mean(axis=1)
    pop_var = float(np.var(pop_rate))
    mean_neuron_var = float(np.mean(np.var(x, axis=0)))

    if mean_neuron_var <= 1e-8:
        # Degenerate case (flat activity): use instant unanimity as fallback.
        p = float(x[-1].mean())
        return float(np.clip(1.0 - 4.0 * p * (1.0 - p), 0.0, 1.0))

    return float(np.clip(pop_var / mean_neuron_var, 0.0, 1.0))


def _compute_spike_entropy(spike_history: deque[np.ndarray]) -> float:
    """Return normalized binary entropy in [0, 1] over a rolling spike window.

    For each pooled-cell channel n, estimate firing probability p_n over time and
    compute H(p_n). The reported metric is mean_n H(p_n), where H is base-2
    binary entropy with maximum 1 bit at p_n = 0.5.
    """
    if len(spike_history) < 2:
        return 0.0

    x = np.stack(spike_history, axis=0).astype(np.float32, copy=False)  # (T, N)
    p = np.mean(x, axis=0)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    h = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return float(np.clip(np.mean(h), 0.0, 1.0))


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── shared runtime config (written by MIDI, read by main loop + view) ─────
    config_state = {
        # Bank A
        "tempo":                  MASTER_TEMPO,
        "master_volume":          0.5,
        "active_pairs":           2,
        "decay_speed":            0.5,
        "mod_index_scale":        0.25,
        "quantization_strength":  QUANTIZATION_STRENGTH,
        # Bank B
        "threshold":              1.0,
        "tau":                    20.0,
        "weight_scale":           1.0,
        "drive_n":                0.5,
        "swing_amount":           SWING_AMOUNT,
        "ratio_scale":            1.0,
        # Bank B / status
        "scale_name":             "major",
        "root_note":              60,
        "aftertouch_target":      "threshold",
        "lif_steps":              12,
        "topology_index":         0,
        "topology_name":          "Ring",
        "neuron_count":           512,
        "last_midi":              None,
        "controller_coverage_ok": False,
        "controller_coverage_missing": [],
        "noteon_flash":            {},   # note -> monotonic timestamp
        "synchrony_index":         0.0,
        "spike_entropy":           0.0,
    }

    # ── model ──────────────────────────────────────────────────────────────────
    network = LIFNetwork()
    synth   = make_synth()

    config_state["topology_index"] = network.topology_index
    config_state["topology_name"] = network.topology_name()
    config_state["neuron_count"] = network.neuron_count

    # Seed with a musical scale
    synth.set_all_base_freqs(_build_initial_freqs(60, "major"))
    network.frequencies = synth.get_operator_frequencies()

    # ── view ───────────────────────────────────────────────────────────────────
    view = MatrixView()

    # ── MIDI ───────────────────────────────────────────────────────────────────
    midi = MIDIHandler(network, synth, config_state)
    midi_ok = midi.open_port(MIDI_PORT_NAME)
    if midi_ok:
        midi.start()

    # ── audio stream ───────────────────────────────────────────────────────────
    def _audio_cb(outdata, frames, time_info, status):
        samples = synth.generate(frames)           # float32 (frames,)
        outdata[:, 0] = samples
        if outdata.shape[1] > 1:
            outdata[:, 1] = samples

    audio_ok = False
    stream   = None
    try:
        stream = sd.OutputStream(
            samplerate = SAMPLE_RATE,
            blocksize  = BUFFER_SIZE,
            channels   = 2,
            dtype      = "float32",
            callback   = _audio_cb,
            latency    = "high",   # more internal buffering = fewer underruns
        )
        stream.start()
        audio_ok = True
        print("[Audio] Stream started.")
    except Exception as exc:
        print(f"[Audio] Could not open stream: {exc}")

    # ── sequencer state ────────────────────────────────────────────────────────
    current_step   = 0
    synth.set_active_step(current_step)
    last_step_time = time.monotonic()
    sync_history   = deque(maxlen=32)

    # ── main loop ──────────────────────────────────────────────────────────────
    running = True
    while running:
        # ── pygame events ─────────────────────────────────────────────────────
        running = view.handle_events()

        now = time.monotonic()

        # ── timing ────────────────────────────────────────────────────────────
        tempo  = config_state["tempo"]
        # 16th-note duration in seconds
        step_dur = 60.0 / tempo / 4.0

        # Swing: odd steps arrive slightly late
        swing = config_state["swing_amount"]
        swing_offset = step_dur * swing if current_step % 2 == 1 else 0.0

        # Quantization: add timing jitter inversely proportional to quant strength
        quant = config_state["quantization_strength"]
        jitter = (1.0 - quant) * step_dur * 0.15 * (np.random.rand() - 0.5)

        fire_at = last_step_time + step_dur + swing_offset + jitter

        if now >= fire_at:
            # ── advance to next step ──────────────────────────────────────────
            current_step = (current_step + 1) % COLS

            # ── LIF micro-steps: accumulate spikes across all sub-steps ───────
            accumulated = None
            for _ in range(config_state["lif_steps"]):
                spikes_step = network.step()
                # MIDI can change neuron count/topology at any time, which can
                # change LIF grid shape between micro-steps. Restart accumulation
                # on shape changes to avoid broadcast errors.
                if accumulated is None or accumulated.shape != spikes_step.shape:
                    accumulated = spikes_step.copy()
                else:
                    accumulated |= spikes_step

            if accumulated is None:
                continue

            synth_spikes = network.get_synth_spikes(ROWS, COLS)
            sync_history.append(synth_spikes.reshape(-1).astype(np.float32, copy=False))
            config_state["synchrony_index"] = _compute_synchrony_index(sync_history)
            config_state["spike_entropy"] = _compute_spike_entropy(sync_history)

            # ── atomically activate voice + gate env from spikes ──────────────
            synth.trigger_and_activate(synth_spikes, current_step)

            # ── update view state ─────────────────────────────────────────────
            spikes     = accumulated
            potentials = network.get_potentials()
            # Refresh display frequencies (may have changed via MIDI)
            network.frequencies = synth.get_operator_frequencies()

            # propagate aftertouch target to config_state for display
            config_state["aftertouch_target"] = midi.aftertouch_target if midi_ok else "threshold"
            config_state["topology_index"] = network.topology_index
            config_state["topology_name"] = network.topology_name()
            config_state["neuron_count"] = network.neuron_count

            view.update(
                spikes       = spikes,
                potentials   = potentials,
                frequencies  = network.frequencies,
                current_step = current_step,
                config_state = config_state,
            )

            # keep last_step_time on the ideal grid to avoid drift
            last_step_time = fire_at

        # ── draw ──────────────────────────────────────────────────────────────
        view.draw()
        view.tick()

    # ── shutdown ───────────────────────────────────────────────────────────────
    if midi_ok:
        midi.stop()
    if audio_ok and stream is not None:
        stream.stop()
        stream.close()
    pygame.quit()
    print("Bye.")


if __name__ == "__main__":
    main()
