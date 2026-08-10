"""
FM synthesiser  –  16 voices (presets / columns) × 8 operators each.

Backend selection
─────────────────
  If the Metal/pybind11 native extension (synth_native) has been built,
  NativeFMSynth wraps it and is used automatically.  Otherwise the pure-Python
  FMSynth is used as a fallback.

  To build the native backend (macOS only):
      cd synth_native && pip install -e . && cd ..

  After that, main.py will pick it up via make_synth().
"""

import threading
import numpy as np

from config import SAMPLE_RATE, BUFFER_SIZE, COLS, ROWS, NUM_OPERATORS
from model.presets import PRESETS

# ── try to load the native Metal backend ──────────────────────────────────────
_native_ok = False
try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from synth_native._fm_synth import FMSynth as _NativeCoreClass
    _native_ok = True
except ImportError:
    _native_ok = False


class NativeFMSynth:
    """
    Thin Python wrapper around the Metal-backed C++ FMSynth.
    Presents exactly the same public API as the pure-Python FMSynth below.
    """

    def __init__(self):
        self._core = _NativeCoreClass(SAMPLE_RATE, BUFFER_SIZE)
        self._lock = threading.Lock()   # only for set_all_base_freqs etc.

        # Load all 16 presets
        for col, preset in enumerate(PRESETS):
            self._core.load_preset(
                col,
                preset["ratios"].astype(np.float32),
                preset["levels"].astype(np.float32),
                preset["mod_indices"].astype(np.float32),
            )

        self._base_freqs = np.full(COLS, 440.0, np.float32)

    # ── frequency control ──────────────────────────────────────────────────
    def set_all_base_freqs(self, freqs) -> None:
        arr = np.asarray(freqs, dtype=np.float32)
        self._base_freqs[:] = arr
        self._core.set_all_base_freqs(arr)

    def set_base_freq(self, col: int, freq: float) -> None:
        self._base_freqs[col] = freq
        self._core.set_base_freq(col, float(freq))

    # ── step control ────────────────────────────────────────────────────────
    def set_active_step(self, step: int) -> None:
        with self._lock:
            self._core.set_active_step(step)

    def trigger_and_activate(self,
                              accumulated_spikes: np.ndarray,
                              col: int) -> None:
        # Thread-safe: protect against concurrent generate() calls from audio thread
        with self._lock:
            self._core.trigger_and_activate(
                accumulated_spikes.astype(np.float32), col)

    # ── parameter control ───────────────────────────────────────────────────
    def set_mod_index_scale(self, scale: float) -> None:
        self._core.set_mod_index_scale(float(scale))

    def set_active_pairs(self, n: int) -> None:
        self._core.set_active_pairs(int(n))

    def set_decay_speed(self, speed: float) -> None:
        self._core.set_decay_speed(float(speed))

    def set_ratio_scale(self, scale: float) -> None:
        self._core.set_ratio_scale(float(scale))

    @property
    def master_volume(self) -> float:
        return self._core.master_volume

    @master_volume.setter
    def master_volume(self, v: float) -> None:
        self._core.master_volume = float(v)

    # ── audio generation ────────────────────────────────────────────────────
    def generate(self, frames: int) -> np.ndarray:
        # Thread-safe: protect against concurrent trigger_and_activate() calls
        with self._lock:
            return self._core.generate(frames)

    # ── display ─────────────────────────────────────────────────────────────
    def get_operator_frequencies(self) -> np.ndarray:
        return self._core.get_operator_frequencies().astype(np.float32)


def make_synth() -> "FMSynth | NativeFMSynth":
    """Return a Metal-backed NativeFMSynth if available, else pure-Python FMSynth."""
    if _native_ok:
        try:
            s = NativeFMSynth()
            print("[FMSynth] Using Metal-accelerated backend.")
            return s
        except Exception as e:
            print(f"[FMSynth] Metal init failed ({e}), falling back to Python backend.")
    else:
        print("[FMSynth] Native backend not built – using pure-Python fallback.")
    return FMSynth()


class FMSynth:
    """Pure-Python FM synth fallback (used when native backend unavailable)."""

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

            Intentionally lock-free.  Holding self._lock here would stall the
            callback whenever the main or MIDI thread updates a parameter, causing
            buffer underruns and audible clicks.  The only mutable arrays written
            exclusively by this method are _phases (no contention).  The rare race
            on _env/_voice_gain from trigger_and_activate is inaudible because it
            only occurs when the voice gain is near 0 (start/end of a step).
            """
            output = np.zeros(frames, np.float64)
            n      = np.arange(frames, dtype=np.float64)

            for col in range(COLS):
                # ── voice crossfade gain ───────────────────────────────────────
                gain_start = self._voice_gain[col]
                gain_end   = float(np.clip(
                    1.0 if self._voice_active[col] else 0.0,
                    gain_start - 1.0,
                    gain_start + 1.0))
                self._voice_gain[col] = gain_end

                if gain_start == 0.0 and gain_end == 0.0:
                    continue

                gain_ramp = np.linspace(gain_start, gain_end, frames)
                voice     = np.zeros(frames, np.float64)
                n_active  = 0

                for pair in range(self._active_pairs):
                    c_op = pair * 2
                    m_op = pair * 2 + 1

                    f_car = self._frequencies[col, c_op]
                    f_mod = self._frequencies[col, m_op]
                    if f_car <= 0.0:
                        continue

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
                    self._env[c_op, col] = env_c1
                    self._env[m_op, col] = env_m1

                if n_active > 0:
                    voice /= n_active
                output += voice * gain_ramp

            np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            output = np.tanh(output * 0.7) * self.master_volume
            return output.astype(np.float32)
