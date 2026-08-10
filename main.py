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
from typing import Any
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
from control.diversity_controller import DiversityController
from control.midi_handler import MIDIHandler
from control.scene_manager import SceneManager, SCENE_PARAM_KEYS
from view.matrix_view     import MatrixView


SCENE_KEY_TO_SLOT = {
    pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
    pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8, pygame.K_9: 9,
    pygame.K_KP0: 0, pygame.K_KP1: 1, pygame.K_KP2: 2, pygame.K_KP3: 3, pygame.K_KP4: 4,
    pygame.K_KP5: 5, pygame.K_KP6: 6, pygame.K_KP7: 7, pygame.K_KP8: 8, pygame.K_KP9: 9,
}


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


def _capture_scene_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    """Capture only scene-relevant keys from runtime config."""
    snapshot: dict[str, Any] = {}
    for key in SCENE_PARAM_KEYS:
        if key in cfg:
            snapshot[key] = cfg[key]
    return snapshot


def _apply_scene_updates(
    updates: dict[str, Any],
    network: LIFNetwork,
    synth,
    midi: MIDIHandler | None,
    cfg: dict[str, Any],
) -> None:
    """Apply scene-driven parameter updates to model, synth, and runtime state."""
    if not updates:
        return

    def _set_cfg_float(key: str) -> None:
        if key in updates:
            cfg[key] = float(updates[key])

    for k in ("tempo", "master_volume", "decay_speed", "mod_index_scale", "quantization_strength", "weight_scale", "drive_n", "swing_amount", "ratio_scale", "scene_morph_time"):
        _set_cfg_float(k)

    if "master_volume" in updates:
        synth.master_volume = float(cfg["master_volume"])

    if "active_pairs" in updates:
        pairs = int(round(float(updates["active_pairs"])))
        pairs = max(1, min(4, pairs))
        synth.set_active_pairs(pairs)
        cfg["active_pairs"] = pairs

    if "decay_speed" in updates:
        synth.set_decay_speed(float(cfg["decay_speed"]))
    if "mod_index_scale" in updates:
        synth.set_mod_index_scale(float(cfg["mod_index_scale"]))
    if "ratio_scale" in updates:
        synth.set_ratio_scale(float(cfg["ratio_scale"]))

    if "topology_index" in updates:
        network.set_topology_index(int(round(float(updates["topology_index"]))))
    if "neuron_count" in updates:
        ncount = network.set_neuron_count(int(round(float(updates["neuron_count"]))))
        cfg["neuron_count"] = ncount

    if "threshold" in updates:
        network.set_threshold(float(updates["threshold"]))
    if "tau" in updates:
        network.set_tau(float(updates["tau"]))
    if "weight_scale" in updates:
        network.set_weight_scale(float(cfg["weight_scale"]))
    if "drive_n" in updates:
        network.set_global_drive(float(cfg["drive_n"]))

    if "heterogeneity" in updates:
        network.set_heterogeneity(float(updates["heterogeneity"]))
    if "inhibitory_ratio" in updates:
        network.set_inhibitory_ratio(float(updates["inhibitory_ratio"]))
    if "inhibitory_gain" in updates:
        network.set_inhibitory_gain(float(updates["inhibitory_gain"]))
    if "excitatory_scale" in updates:
        network.set_excitatory_scale(float(updates["excitatory_scale"]))
    if "inhibitory_scale" in updates:
        network.set_inhibitory_scale(float(updates["inhibitory_scale"]))
    if "delay_spread_steps" in updates:
        network.set_delay_spread_steps(int(round(float(updates["delay_spread_steps"]))))
    if "delay_jitter" in updates:
        network.set_delay_jitter(float(updates["delay_jitter"]))

    if "adaptation_strength" in updates:
        network.set_adaptation_strength(float(updates["adaptation_strength"]))
    if "adaptation_decay" in updates:
        network.set_adaptation_decay(float(updates["adaptation_decay"]))
    if "noise_amount" in updates:
        network.set_noise_amount(float(updates["noise_amount"]))
    if "noise_color" in updates:
        network.set_noise_color(updates["noise_color"])
    if "spatial_noise" in updates:
        network.set_spatial_noise(float(updates["spatial_noise"]))

    if "lif_steps" in updates:
        cfg["lif_steps"] = int(max(1, min(64, round(float(updates["lif_steps"])))))

    root = int(cfg.get("root_note", 60))
    if "root_note" in updates:
        root = int(round(float(updates["root_note"])))
        root = max(0, min(127, root))
        cfg["root_note"] = root

    scale_name = str(cfg.get("scale_name", "major"))
    if "scale_name" in updates:
        candidate = str(updates["scale_name"])
        if candidate in SCALES:
            scale_name = candidate
            cfg["scale_name"] = candidate

    if "root_note" in updates or "scale_name" in updates:
        synth.set_all_base_freqs(_build_initial_freqs(root, scale_name))
        network.frequencies = synth.get_operator_frequencies()
        if midi is not None:
            midi.root_note = root
            midi.scale_name = scale_name
            if scale_name in SCALE_NAMES:
                midi.scale_idx = SCALE_NAMES.index(scale_name)

    cfg["topology_index"] = network.topology_index
    cfg["topology_name"] = network.topology_name()
    cfg["neuron_count"] = network.neuron_count
    cfg.update(network.heterogeneity_state())
    cfg.update(network.phase2_state())
    cfg.update(network.phase3_state())


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
        "heterogeneity":          0.0,
        "hetero_seed":            1337,
        "threshold_spread":       0.0,
        "tau_spread":             0.0,
        "refractory_spread":      0.0,
        "drive_spread":           0.0,
        "inhibitory_ratio":       0.18,
        "inhibitory_gain":        1.0,
        "excitatory_scale":       1.0,
        "inhibitory_scale":       1.0,
        "delay_spread_steps":     0,
        "delay_jitter":           0.0,
        "adaptation_strength":    0.0,
        "adaptation_decay":       0.93,
        "noise_amount":           0.0,
        "noise_color":            "white",
        "noise_color_index":      0,
        "spatial_noise":          0.0,
        "last_midi":              None,
        "controller_coverage_ok": False,
        "controller_coverage_missing": [],
        "noteon_flash":            {},   # note -> monotonic timestamp
        "synchrony_index":         0.0,
        "spike_entropy":           0.0,
        "active_ratio":            0.0,
        "active_ratio_ma":         0.0,
        "scene_index":             0,
        "scene_name":              "Calm lattice",
        "scene_morph_time":        1.6,
        "scene_morph_progress":    1.0,
        "scene_morph_active":      False,
        "scene_strategy_name":     "static",
        "anti_lock_enabled":       False,
        "anti_lock_strength":      0.45,
        "anti_lock_collapse_score": 0.0,
        "anti_lock_last_reason":   "idle",
        "anti_lock_last_magnitude": 0.0,
        "anti_lock_last_nudge_time": -1.0,
    }

    # ── model ──────────────────────────────────────────────────────────────────
    network = LIFNetwork()
    synth   = make_synth()

    network.set_heterogeneity_seed(int(config_state["hetero_seed"]))
    network.set_heterogeneity(float(config_state["heterogeneity"]))
    network.set_inhibitory_ratio(float(config_state["inhibitory_ratio"]))
    network.set_inhibitory_gain(float(config_state["inhibitory_gain"]))
    network.set_excitatory_scale(float(config_state["excitatory_scale"]))
    network.set_inhibitory_scale(float(config_state["inhibitory_scale"]))
    network.set_delay_spread_steps(int(config_state["delay_spread_steps"]))
    network.set_delay_jitter(float(config_state["delay_jitter"]))
    network.set_adaptation_strength(float(config_state["adaptation_strength"]))
    network.set_adaptation_decay(float(config_state["adaptation_decay"]))
    network.set_noise_amount(float(config_state["noise_amount"]))
    network.set_noise_color(config_state["noise_color"])
    network.set_spatial_noise(float(config_state["spatial_noise"]))

    config_state["topology_index"] = network.topology_index
    config_state["topology_name"] = network.topology_name()
    config_state["neuron_count"] = network.neuron_count
    config_state.update(network.heterogeneity_state())
    config_state.update(network.phase2_state())
    config_state.update(network.phase3_state())

    scene_manager = SceneManager()
    diversity_controller = DiversityController()
    config_state["scene_index"] = scene_manager.active_slot
    config_state["scene_name"] = scene_manager.scene_name()

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
    active_ratio_history = deque(maxlen=16)

    def _handle_scene_hotkeys(event: pygame.event.Event) -> bool:
        if event.key == pygame.K_a:
            enabled = not bool(config_state.get("anti_lock_enabled", False))
            config_state["anti_lock_enabled"] = enabled
            print(f"[AntiLock] {'ON' if enabled else 'OFF'}")
            return True

        if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            v = max(0.0, float(config_state.get("anti_lock_strength", 0.45)) - 0.05)
            config_state["anti_lock_strength"] = v
            print(f"[AntiLock] strength -> {v:.2f}")
            return True

        if event.key in (pygame.K_EQUALS, pygame.K_KP_PLUS):
            v = min(1.0, float(config_state.get("anti_lock_strength", 0.45)) + 0.05)
            config_state["anti_lock_strength"] = v
            print(f"[AntiLock] strength -> {v:.2f}")
            return True

        slot = SCENE_KEY_TO_SLOT.get(event.key)
        if slot is None:
            return False

        snapshot = _capture_scene_snapshot(config_state)
        now = time.monotonic()

        if event.mod & pygame.KMOD_SHIFT:
            name = scene_manager.save_scene(slot, snapshot)
            print(f"[Scene] Saved slot {slot} -> {name}")
            return True

        morph_time = float(config_state.get("scene_morph_time", 0.0))
        name = scene_manager.launch_scene(slot, snapshot, now, morph_time)
        print(f"[Scene] Launch slot {slot} ({name})  morph={morph_time:.2f}s")
        return True

    # ── main loop ──────────────────────────────────────────────────────────────
    running = True
    while running:
        # ── pygame events ─────────────────────────────────────────────────────
        running = view.handle_events(_handle_scene_hotkeys)

        now = time.monotonic()
        scene_updates, scene_meta = scene_manager.tick(now)
        if scene_updates:
            _apply_scene_updates(scene_updates, network, synth, midi if midi_ok else None, config_state)
        config_state.update(scene_meta)

        anti_updates, anti_meta = diversity_controller.tick(now, config_state)
        if anti_updates:
            _apply_scene_updates(anti_updates, network, synth, midi if midi_ok else None, config_state)
        config_state.update(anti_meta)

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

            # Downmix accumulated spikes (all micro-steps) instead of just the latest step.
            # This ensures FM synth trigger matches the LIF's cumulative activity.
            synth_spikes = network.get_synth_spikes(ROWS, COLS, spike_array=accumulated)
            sync_history.append(synth_spikes.reshape(-1).astype(np.float32, copy=False))
            config_state["synchrony_index"] = _compute_synchrony_index(sync_history)
            config_state["spike_entropy"] = _compute_spike_entropy(sync_history)

            active_ratio = float(np.mean(synth_spikes.astype(np.float32, copy=False)))
            active_ratio_history.append(active_ratio)
            config_state["active_ratio"] = active_ratio
            config_state["active_ratio_ma"] = float(np.mean(active_ratio_history))

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
            config_state.update(network.heterogeneity_state())
            config_state.update(network.phase2_state())
            config_state.update(network.phase3_state())

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
