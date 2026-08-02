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
import numpy as np

import sounddevice as sd
import pygame

from config import (
    SAMPLE_RATE, BUFFER_SIZE, COLS,
    MASTER_TEMPO, QUANTIZATION_STRENGTH, SWING_AMOUNT,
    SCALES, SCALE_NAMES, MIDI_PORT_NAME,
)
from model.lif_network import LIFNetwork
from model.fm_synth    import FMSynth
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


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── shared runtime config (written by MIDI, read by main loop + view) ─────
    config_state = {
        "tempo":                  MASTER_TEMPO,
        "quantization_strength":  QUANTIZATION_STRENGTH,
        "swing_amount":           SWING_AMOUNT,
        "scale_name":             "major",
        "root_note":              60,
        "aftertouch_target":      "threshold",
    }

    # ── model ──────────────────────────────────────────────────────────────────
    network = LIFNetwork()
    synth   = FMSynth()

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
            latency    = "low",
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

    # Number of LIF micro-steps executed per sequencer tick
    LIF_STEPS_PER_TICK = 12

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
            # ── LIF network step(s) ───────────────────────────────────────────
            for _ in range(LIF_STEPS_PER_TICK):
                spikes = network.step()

            # ── couple spikes → FM envelopes ──────────────────────────────────
            synth.update_neuron_activations(spikes)

            # ── update view state ─────────────────────────────────────────────
            potentials = network.get_potentials()
            # Refresh display frequencies (may have changed via MIDI)
            network.frequencies = synth.get_operator_frequencies()

            # propagate aftertouch target to config_state for display
            config_state["aftertouch_target"] = midi.aftertouch_target if midi_ok else "threshold"

            view.update(
                spikes       = spikes,
                potentials   = potentials,
                frequencies  = network.frequencies,
                current_step = current_step,
                config_state = config_state,
            )

            # ── advance step ──────────────────────────────────────────────────
            current_step = (current_step + 1) % COLS
            synth.set_active_step(current_step)
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
