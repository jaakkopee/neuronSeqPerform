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
        self._mod_index_scale = 1.0
        self._ratio_scale     = 1.0

        # ── voice base frequencies  (one per column / preset) ─────────────────
        self._base_freqs = np.full(COLS, 440.0, np.float64)

        # ── derived frequencies  (COLS × NUM_OPERATORS) ───────────────────────
        self._frequencies = np.zeros((COLS, NUM_OPERATORS), np.float64)

        # ── oscillator phase accumulators  (COLS × NUM_OPERATORS) ─────────────
        self._phases = np.zeros((COLS, NUM_OPERATORS), np.float64)

        # ── neuron-driven amplitude envelope  (NUM_OPERATORS × COLS) ──────────
        # indexed as [operator_row, preset_col]
        # Seeded at 0.5 so audio is audible from the very first step;
        # spikes drive it up, silence lets it decay.
        self._env = np.full((NUM_OPERATORS, COLS), 0.5, np.float64)

        # ── which voice is active (only active voice is rendered) ─────────────
        self._active_step = 0
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

    # ── step control ──────────────────────────────────────────────────────────
    def set_active_step(self, step: int) -> None:
        with self._lock:
            self._voice_active[:] = False
            self._voice_active[step] = True
            self._active_step = step

    # ── neuron activation coupling ────────────────────────────────────────────
    def update_neuron_activations(self, spikes: np.ndarray) -> None:
        """
        spikes : bool array (ROWS, COLS) from LIF network.
        Rows map to operators; columns map to presets.
        """
        with self._lock:
            spikes_f = spikes.astype(np.float64)          # (ROWS, COLS)

            # Carrier rows (0,2,4,6) → slower decay (sustain character)
            # Modulator rows (1,3,5,7) → faster decay (transient timbre)
            decay = np.where(np.arange(NUM_OPERATORS) % 2 == 0, 0.997, 0.96)
            decay = decay[:, np.newaxis]                   # broadcast over cols

            self._env *= decay
            self._env = np.where(spikes_f == 1.0,
                                 np.minimum(1.0, self._env + 0.4),
                                 self._env)

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

            for col in range(COLS):
                if not self._voice_active[col]:
                    continue

                voice = np.zeros(frames, np.float64)
                n_active = 0

                for pair in range(4):
                    c_op = pair * 2      # carrier operator index
                    m_op = pair * 2 + 1  # modulator operator index

                    f_car = self._frequencies[col, c_op]
                    f_mod = self._frequencies[col, m_op]

                    if f_car <= 0.0:
                        continue

                    level_car = self._levels[col, c_op] * self._env[c_op, col]
                    mi        = (self._mod_indices[col, m_op]
                                 * self._env[m_op, col]
                                 * self._mod_index_scale)

                    # --- phase arrays -----------------------------------------
                    phi_car_inc = 2.0 * np.pi * f_car / self._sr
                    phi_mod_inc = 2.0 * np.pi * f_mod / self._sr if f_mod > 0 else 0.0

                    n = np.arange(frames, dtype=np.float64)
                    phi_mod = self._phases[col, m_op] + phi_mod_inc * n
                    phi_car = self._phases[col, c_op] + phi_car_inc * n

                    sample = level_car * np.sin(phi_car + mi * np.sin(phi_mod))
                    voice += sample
                    n_active += 1

                    # advance phase accumulators
                    self._phases[col, c_op] = (self._phases[col, c_op]
                                               + phi_car_inc * frames) % (2.0 * np.pi)
                    self._phases[col, m_op] = (self._phases[col, m_op]
                                               + phi_mod_inc * frames) % (2.0 * np.pi)

                if n_active > 0:
                    voice /= n_active
                output += voice

            # soft limiting + master volume
            output = np.tanh(output * 0.7) * self.master_volume
            return output.astype(np.float32)
