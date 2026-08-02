"""
FM synthesiser  –  16 voices (presets / columns) × 8 operators each.

Pairing algorithm
─────────────────
  Pair k :  Op 2k  (carrier)  ←  modulated by  Op 2k+1  (modulator)
  k ∈ {0, 1, 2, 3}

Neuron coupling
───────────────
  neuron_env[op, col]  is an amplitude envelope per (operator, voice) cell.
  On a spike it jumps toward 1.0; between spikes it decays exponentially.
  Carriers  (even op rows)  use a slower decay  → sustain character.
  Modulators (odd op rows)  use a faster decay  → transient timbre burst.

Thread safety
─────────────
  The audio callback runs in a high-priority OS thread.  All state shared
  between the callback and the main/MIDI threads is protected by self._lock.
"""

import threading
import numpy as np

from config import SAMPLE_RATE, COLS, ROWS, NUM_OPERATORS
from model.presets import PRESETS


class FMSynth:
    # ── construction ──────────────────────────────────────────────────────────
    def __init__(self):
        self._sr   = SAMPLE_RATE
        self._lock = threading.Lock()

        # ── preset parameters  (COLS × NUM_OPERATORS) ─────────────────────────
        self._ratios      = np.zeros((COLS, NUM_OPERATORS), np.float64)
        self._levels      = np.zeros((COLS, NUM_OPERATORS), np.float64)
        self._mod_indices = np.zeros((COLS, NUM_OPERATORS), np.float64)

        # ── runtime scalers ───────────────────────────────────────────────────
        self._mod_index_scale   = 0.25   # default: light FM (0 = clean sine, 3 = dense)
        self._ratio_scale       = 1.0
        # ── texture controls ──────────────────────────────────────────────────
        self._active_pairs      = 2      # carrier-mod pairs rendered (1-4)
        # Per-audio-buffer decay factors (computed by set_decay_speed).
        # Carrier ops (even rows) sustain longer; modulator ops (odd) shorter.
        self._per_buf_carrier   = 0.9286  # speed=0.5 default
        self._per_buf_mod       = 0.8700  # speed=0.5 default

        # ── voice base frequencies  (one per column / preset) ─────────────────
        self._base_freqs = np.full(COLS, 440.0, np.float64)

        # ── derived frequencies  (COLS × NUM_OPERATORS) ───────────────────────
        self._frequencies = np.zeros((COLS, NUM_OPERATORS), np.float64)

        # ── oscillator phase accumulators  (COLS × NUM_OPERATORS) ─────────────
        self._phases = np.zeros((COLS, NUM_OPERATORS), np.float64)

        # ── neuron-driven amplitude envelope  (NUM_OPERATORS × COLS) ──────────
        # Zero at start; trigger_and_activate() gates it from spike state.
        self._env = np.zeros((NUM_OPERATORS, COLS), np.float64)

        # ── per-voice crossfade gain (enables click-free voice transitions) ───
        # Ramped linearly within each audio buffer: 0.0 = silent, 1.0 = full.
        self._voice_gain   = np.zeros(COLS, np.float64)

        # ── which voice is active (desired target for crossfade) ──────────────
        self._active_step  = 0
        self._voice_active = np.zeros(COLS, bool)
        self._voice_active[0] = True

        self.master_volume = 0.5

        self._load_presets()

    # ── preset loading ────────────────────────────────────────────────────────
    def _load_presets(self) -> None:
        for col, preset in enumerate(PRESETS):
            self._ratios[col]      = preset["ratios"]
            self._levels[col]      = preset["levels"]
            self._mod_indices[col] = preset["mod_indices"]
        self._recompute_freqs()

    def _recompute_freqs(self) -> None:
        """Recalculate all operator frequencies from base freqs + ratios."""
        for col in range(COLS):
            self._frequencies[col] = self._base_freqs[col] * self._ratios[col] * self._ratio_scale

    # ── frequency control ─────────────────────────────────────────────────────
    def set_all_base_freqs(self, freqs) -> None:
        with self._lock:
            self._base_freqs[:] = freqs
            self._recompute_freqs()

    def set_base_freq(self, col: int, freq: float) -> None:
        with self._lock:
            self._base_freqs[col] = freq
            self._frequencies[col] = freq * self._ratios[col] * self._ratio_scale

    # ── step / trigger control ────────────────────────────────────────────────
    def set_active_step(self, step: int) -> None:
        with self._lock:
            self._voice_active[:] = False
            self._voice_active[step] = True
            self._active_step = step

    def trigger_column(self, accumulated_spikes: np.ndarray, col: int) -> None:
        """Gate env for a column from spike state (env decay runs in generate)."""
        with self._lock:
            self._env[:, col] = accumulated_spikes[:, col].astype(np.float64)

    def trigger_and_activate(self, accumulated_spikes: np.ndarray, col: int) -> None:
        """
        Atomically switch the active voice AND gate the envelope from spikes.
        Using one lock acquisition ensures the audio callback never sees a
        mismatch between voice_active and env state, eliminating click sources.
        """
        with self._lock:
            self._voice_active[:] = False
            self._voice_active[col] = True
            self._active_step = col
            self._env[:, col] = accumulated_spikes[:, col].astype(np.float64)

    # ── display info ──────────────────────────────────────────────────────────
    def get_operator_frequencies(self) -> np.ndarray:
        """
        Return operator frequencies as (NUM_OPERATORS, COLS) = (ROWS, COLS)
        suitable for display in the matrix view.
        """
        with self._lock:
            return self._frequencies.T.copy().astype(np.float32)  # (COLS, OPS).T

    # ── FM parameter control ──────────────────────────────────────────────────
    def set_mod_index_scale(self, scale: float) -> None:
        with self._lock:
            self._mod_index_scale = float(scale)

    def set_active_pairs(self, n: int) -> None:
        """Set how many carrier-modulator pairs are rendered (1-4)."""
        with self._lock:
            self._active_pairs = max(1, min(4, int(n)))

    def set_decay_speed(self, speed: float) -> None:
        """
        speed 0.0 → slow decay (sustained, long tail)
        speed 1.0 → fast decay (staccato, percussive)

        Values are per-audio-buffer (512/44100 ≈11.6 ms).
        At default 120 BPM (10.8 buffers/step):
          speed 0 → carrier keeps ~99% amplitude per step
          speed 1 → carrier drops to ~19% by next step
        """
        s = max(0.0, min(1.0, float(speed)))
        with self._lock:
            self._per_buf_carrier = 0.9953 - s * (0.9953 - 0.862)
            self._per_buf_mod     = 0.9900 - s * (0.9900 - 0.750)

    def set_ratio_scale(self, scale: float) -> None:
        with self._lock:
            self._ratio_scale = float(scale)
            self._recompute_freqs()

    # ── audio generation (called from sounddevice callback thread) ────────────
    def generate(self, frames: int) -> np.ndarray:
        """
        Render `frames` mono samples.  Returns float32 array of length `frames`.
        """
        with self._lock:
            output = np.zeros(frames, np.float64)
            n      = np.arange(frames, dtype=np.float64)

            for col in range(COLS):
                # ── voice crossfade gain ───────────────────────────────────
                # Ramp linearly within the buffer: active→1.0, inactive→0.0.
                # One buffer (~11.6 ms) is enough for a click-free transition.
                gain_start = self._voice_gain[col]
                gain_end   = 1.0 if self._voice_active[col] else 0.0
                # Cap change to ±1.0 per buffer (i.e. full transition per buf)
                gain_end = np.clip(gain_end,
                                   gain_start - 1.0,
                                   gain_start + 1.0)
                self._voice_gain[col] = gain_end

                if gain_start == 0.0 and gain_end == 0.0:
                    continue                          # fully silent, skip

                gain_ramp = np.linspace(gain_start, gain_end, frames)

                voice    = np.zeros(frames, np.float64)
                n_active = 0

                for pair in range(self._active_pairs):
                    c_op = pair * 2
                    m_op = pair * 2 + 1

                    f_car = self._frequencies[col, c_op]
                    f_mod = self._frequencies[col, m_op]
                    if f_car <= 0.0:
                        continue

                    # Env ramp within buffer: smooth decay from start to end.
                    # This eliminates the per-buffer amplitude step.
                    env_c0 = self._env[c_op, col]
                    env_m0 = self._env[m_op, col]
                    env_c1 = env_c0 * self._per_buf_carrier
                    env_m1 = env_m0 * self._per_buf_mod

                    level_ramp = (self._levels[col, c_op]
                                  * np.linspace(env_c0, env_c1, frames))
                    mi_ramp    = (self._mod_indices[col, m_op]
                                  * self._mod_index_scale
                                  * np.linspace(env_m0, env_m1, frames))

                    phi_car_inc = 2.0 * np.pi * f_car / self._sr
                    phi_mod_inc = (2.0 * np.pi * f_mod / self._sr
                                   if f_mod > 0 else 0.0)

                    phi_mod = self._phases[col, m_op] + phi_mod_inc * n
                    phi_car = self._phases[col, c_op] + phi_car_inc * n

                    voice += level_ramp * np.sin(phi_car + mi_ramp * np.sin(phi_mod))
                    n_active += 1

                    self._phases[col, c_op] = ((self._phases[col, c_op]
                                                + phi_car_inc * frames)
                                               % (2.0 * np.pi))
                    self._phases[col, m_op] = ((self._phases[col, m_op]
                                                + phi_mod_inc * frames)
                                               % (2.0 * np.pi))

                    # Commit decayed env values after generating the ramp
                    self._env[c_op, col] = env_c1
                    self._env[m_op, col] = env_m1

                if n_active > 0:
                    voice /= n_active
                output += voice * gain_ramp

            # soft limiting + master volume
            output = np.tanh(output * 0.7) * self.master_volume
            return output.astype(np.float32)
