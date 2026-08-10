"""
Leaky Integrate-and-Fire network with optional native Metal backend.

This wrapper keeps a musically stable 16-column layout and supports dynamic
neuron counts/topologies inspired by vjay_ace:
  Ring, FullyConnected, Feedforward, SparseRandom, SmallWorld.
"""

from __future__ import annotations

from collections import deque
import math
import threading
import numpy as np

from config import ROWS as SYNTH_ROWS, COLS as SYNTH_COLS


TOPOLOGY_NAMES = [
    "Ring",
    "FullyConnected",
    "Feedforward",
    "SparseRandom",
    "SmallWorld",
]

NEURON_COUNT_STEPS = [128, 256, 512, 1024, 2048, 4096]


_native_ok = False
try:
    from synth_native._fm_synth import LIFNetwork as _NativeLIFCore

    _native_ok = True
except ImportError:
    _native_ok = False


class LIFNetwork:
    def __init__(
        self,
        rows: int = SYNTH_ROWS,
        cols: int = SYNTH_COLS,
        neuron_count: int | None = None,
        topology_index: int = 0,
    ):
        self.cols = max(4, int(cols))
        self._state_lock = threading.RLock()
        self.neuron_count = int(neuron_count) if neuron_count is not None else max(512, rows * cols)
        self.topology_index = int(np.clip(topology_index, 0, len(TOPOLOGY_NAMES) - 1))

        self._threshold = 1.0
        self._tau = 20.0
        self._weight_scale = 1.0
        self._global_drive = 1.25
        self._base_refractory_ms = 5.0

        # Phase 1 heterogeneity controls.
        self._heterogeneity = 0.0
        self._heterogeneity_seed = 1337
        self._threshold_spread = 0.0
        self._tau_spread = 0.0
        self._refractory_spread = 0.0
        self._drive_spread = 0.0

        # Phase 2 E/I and delay controls.
        self._inhibitory_ratio = 0.18
        self._inhibitory_gain = 1.0
        self._excitatory_scale = 1.0
        self._inhibitory_scale = 1.0
        self._delay_spread_steps = 0
        self._delay_jitter = 0.0

        # Phase 3 adaptation and noise controls.
        self._adaptation_strength = 0.0
        self._adaptation_decay = 0.93
        self._noise_amount = 0.0
        self._noise_color = "white"
        self._spatial_noise = 0.0

        self._native = None
        if _native_ok:
            try:
                self._native = _NativeLIFCore(self.neuron_count, self.cols)
                self._native.set_topology(self.topology_index)
                self._native.set_threshold(self._threshold)
                self._native.set_tau(self._tau)
                self._native.set_weight_scale(self._weight_scale)
                self._native.set_global_drive(self._global_drive)
            except Exception as exc:
                print(f"[LIF] Native backend init failed ({exc}); using NumPy fallback.")
                self._native = None

        self.rows = max(1, math.ceil(self.neuron_count / self.cols))
        self._spikes = np.zeros((self.rows, self.cols), dtype=bool)
        self._potentials = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.frequencies = np.zeros((SYNTH_ROWS, SYNTH_COLS), dtype=np.float32)
        self._h_threshold = np.zeros(self.neuron_count, dtype=np.float32)
        self._h_tau = np.zeros(self.neuron_count, dtype=np.float32)
        self._h_refractory = np.zeros(self.neuron_count, dtype=np.float32)
        self._h_drive = np.zeros(self.neuron_count, dtype=np.float32)
        self._native_drive_offsets = np.zeros(self.neuron_count, dtype=np.float32)
        self._inhibitory_mask = np.zeros(self.neuron_count, dtype=bool)
        self._delay_steps = np.zeros(self.neuron_count, dtype=np.int16)
        self._spike_history: deque[np.ndarray] = deque(maxlen=1)
        self._zero_spikes = np.zeros(self.neuron_count, dtype=np.float32)
        self._adaptation_current = np.zeros(self.neuron_count, dtype=np.float32)
        self._noise_state = np.zeros(self.neuron_count, dtype=np.float32)
        self._spatial_profile = np.zeros(self.neuron_count, dtype=np.float32)
        self._noise_rng = np.random.default_rng(self._heterogeneity_seed + self.neuron_count * 41)

        # Fallback-state fields.
        self.v = np.zeros(self.neuron_count, np.float32)
        self.v_rest = np.zeros(self.neuron_count, np.float32)
        self.v_thresh = np.full(self.neuron_count, self._threshold, np.float32)
        self.tau = np.full(self.neuron_count, self._tau, np.float32)
        self.refractory = np.zeros(self.neuron_count, np.float32)
        self.refractory_t = np.full(self.neuron_count, self._base_refractory_ms, np.float32)
        self.dt = 1.0
        self.spikes = np.zeros(self.neuron_count, dtype=bool)
        self.external_drive = np.full(self.neuron_count, self._global_drive, np.float32)
        self._base_weights = np.zeros((self.neuron_count, self.neuron_count), np.float32)
        self.weights = np.zeros((self.neuron_count, self.neuron_count), np.float32)

        if self._native is None:
            self._seed_fallback_state()
        self._rebuild_fallback_weights()

        self._regenerate_heterogeneity_profiles()
        self._regenerate_population_roles()
        self._regenerate_delay_profile()
        self._regenerate_spatial_profile()
        self._reset_spike_history()
        self._apply_heterogeneity()

    # ---------------------------------------------------------------------
    # Public API used by the rest of the app
    # ---------------------------------------------------------------------
    def step(self) -> np.ndarray:
        with self._state_lock:
            if self._native is not None:
                delayed_src = self._delayed_source_activity()
                syn_overlay = self._compute_syn_overlay(delayed_src)
                runtime_term = self._phase3_runtime_term()
                drive_vec = np.clip(self._native_drive_offsets + syn_overlay + runtime_term, -2.5, 3.5).astype(np.float32)
                self._native.set_external_drive(np.ascontiguousarray(drive_vec, dtype=np.float32))

                self._native.step()
                self._spikes = self._native.get_spikes() > 0.5
                self._potentials = self._native.get_potentials().astype(np.float32)
                self.rows, self.cols = self._spikes.shape
                self.neuron_count = int(self._native.neuron_count())
                self._push_spike_history(self._spikes.reshape(-1)[: self.neuron_count])
                self._update_adaptation(self._spikes.reshape(-1)[: self.neuron_count])
                return self._spikes

            delayed_src = self._delayed_source_activity()
            i_syn = self._compute_syn_overlay(delayed_src)
            runtime_term = self._phase3_runtime_term()
            i_tot = i_syn + self.external_drive + runtime_term

            in_ref = self.refractory > 0.0
            dv = (-(self.v - self.v_rest) + i_tot) * (self.dt / np.maximum(self.tau, 1e-4))
            self.v += dv
            self.v[in_ref] = self.v_rest[in_ref]

            self.spikes = self.v >= self.v_thresh
            self.v[self.spikes] = self.v_rest[self.spikes]
            self.refractory[self.spikes] = self.refractory_t[self.spikes]
            self.refractory = np.maximum(0.0, self.refractory - self.dt)
            self._push_spike_history(self.spikes)
            self._update_adaptation(self.spikes)

            self._pack_fallback_views()
            return self._spikes

    def get_potentials(self) -> np.ndarray:
        if self._native is not None:
            return self._potentials
        self._pack_fallback_views()
        return self._potentials

    def get_synth_spikes(self, target_rows: int = SYNTH_ROWS, target_cols: int = SYNTH_COLS, spike_array: np.ndarray | None = None) -> np.ndarray:
        """Downmix/pool the dynamic LIF field into synth operator grid size.
        
        If spike_array is provided, downmix that instead of self._spikes.
        This allows downmixing accumulated spikes across multiple micro-steps.
        """
        src = spike_array if spike_array is not None else self._spikes
        src_rows, src_cols = src.shape
        out = np.zeros((target_rows, target_cols), dtype=bool)

        if src_rows == 0 or src_cols == 0:
            return out

        # Column mapping by nearest neighbour keeps sequencing behavior coherent.
        col_map = np.floor(np.linspace(0, src_cols - 1, target_cols)).astype(int)

        # Row max-pooling keeps transient spikes from disappearing in compression.
        edges = np.linspace(0, src_rows, target_rows + 1).astype(int)
        for tr in range(target_rows):
            r0 = edges[tr]
            r1 = max(r0 + 1, edges[tr + 1])
            r1 = min(r1, src_rows)
            if r0 >= src_rows:
                continue
            band = src[r0:r1, :]
            pooled = np.any(band, axis=0)
            out[tr, :] = pooled[col_map]

        return out

    def set_threshold(self, value: float) -> None:
        v = float(value)
        self._threshold = v
        if self._native is not None:
            self._native.set_threshold(v)
        self._apply_heterogeneity()

    def set_tau(self, value: float) -> None:
        v = float(value)
        self._tau = v
        if self._native is not None:
            self._native.set_tau(v)
        self._apply_heterogeneity()

    def set_weight_scale(self, scale: float) -> None:
        v = float(scale)
        self._weight_scale = v
        if self._native is not None:
            self._native.set_weight_scale(v)
        self._refresh_effective_weights()

    def set_global_drive(self, value: float) -> None:
        # Keep MIDI semantics: normalized 0..1 mapped to a wider LIF drive range.
        drive = 0.25 + float(value) * 2.75
        self._global_drive = drive
        if self._native is not None:
            self._native.set_global_drive(drive)
        self._apply_heterogeneity()

    def set_neuron_drive(self, row: int, col: int, value: float) -> None:
        idx = row * self.cols + col
        if idx < 0 or idx >= self.neuron_count:
            return
        if self._native is not None:
            self._native.set_neuron_drive(row, col, float(value))
            return
        self.external_drive[idx] = float(value)

    def randomize_weights(self) -> None:
        if self._native is not None:
            self._native.randomize_weights()
        self._rebuild_fallback_weights()

    def reset_state(self) -> None:
        if self._native is not None:
            self._native.reset_state()
            self._native.step()
            self._spikes = self._native.get_spikes() > 0.5
            self._potentials = self._native.get_potentials().astype(np.float32)
            self._reset_spike_history()
            self._push_spike_history(self._spikes.reshape(-1)[: self.neuron_count])
            return
        self._seed_fallback_state()
        self.refractory[:] = 0.0
        self.spikes[:] = False
        self._reset_spike_history()
        self._pack_fallback_views()

    def set_topology_index(self, index: int) -> None:
        with self._state_lock:
            idx = int(np.clip(index, 0, len(TOPOLOGY_NAMES) - 1))
            self.topology_index = idx
            if self._native is not None:
                self._native.set_topology(idx)
            self._regenerate_heterogeneity_profiles()
            self._regenerate_population_roles()
            self._regenerate_delay_profile()
            self._regenerate_spatial_profile()
            self._reset_spike_history()
            self._rebuild_fallback_weights()
            self._apply_heterogeneity()

    def cycle_topology(self, direction: int = 1) -> int:
        idx = (self.topology_index + int(np.sign(direction))) % len(TOPOLOGY_NAMES)
        self.set_topology_index(idx)
        return idx

    def set_neuron_count(self, count: int) -> int:
        with self._state_lock:
            count = max(64, int(count))
            self.neuron_count = count

            if self._native is not None:
                self._native.set_neuron_count(count)
                self.rows = int(self._native.rows())
                self.cols = int(self._native.cols())
                self.neuron_count = int(self._native.neuron_count())
            else:
                self.rows = max(1, math.ceil(self.neuron_count / self.cols))

            self._spikes = np.zeros((self.rows, self.cols), dtype=bool)
            self._potentials = np.zeros((self.rows, self.cols), dtype=np.float32)
            self._h_threshold = np.zeros(self.neuron_count, dtype=np.float32)
            self._h_tau = np.zeros(self.neuron_count, dtype=np.float32)
            self._h_refractory = np.zeros(self.neuron_count, dtype=np.float32)
            self._h_drive = np.zeros(self.neuron_count, dtype=np.float32)
            self._native_drive_offsets = np.zeros(self.neuron_count, dtype=np.float32)
            self._inhibitory_mask = np.zeros(self.neuron_count, dtype=bool)
            self._delay_steps = np.zeros(self.neuron_count, dtype=np.int16)
            self._zero_spikes = np.zeros(self.neuron_count, dtype=np.float32)
            self._adaptation_current = np.zeros(self.neuron_count, dtype=np.float32)
            self._noise_state = np.zeros(self.neuron_count, dtype=np.float32)
            self._spatial_profile = np.zeros(self.neuron_count, dtype=np.float32)
            self._noise_rng = np.random.default_rng(self._heterogeneity_seed + self.neuron_count * 41)

            self.v = np.zeros(self.neuron_count, np.float32)
            self.v_rest = np.zeros(self.neuron_count, np.float32)
            self.v_thresh = np.full(self.neuron_count, self._threshold, np.float32)
            self.tau = np.full(self.neuron_count, self._tau, np.float32)
            self.refractory = np.zeros(self.neuron_count, np.float32)
            self.refractory_t = np.full(self.neuron_count, self._base_refractory_ms, np.float32)
            self.spikes = np.zeros(self.neuron_count, dtype=bool)
            self.external_drive = np.full(self.neuron_count, self._global_drive, np.float32)
            self._base_weights = np.zeros((self.neuron_count, self.neuron_count), np.float32)
            self.weights = np.zeros((self.neuron_count, self.neuron_count), np.float32)
            if self._native is None:
                self._seed_fallback_state()
            self._rebuild_fallback_weights()
            self._regenerate_heterogeneity_profiles()
            self._regenerate_population_roles()
            self._regenerate_delay_profile()
            self._regenerate_spatial_profile()
            self._reset_spike_history()
            self._apply_heterogeneity()
            return self.neuron_count

    def nudge_neuron_count_step(self, delta: int) -> int:
        current = self.neuron_count
        idx = min(range(len(NEURON_COUNT_STEPS)), key=lambda i: abs(NEURON_COUNT_STEPS[i] - current))
        idx = int(np.clip(idx + delta, 0, len(NEURON_COUNT_STEPS) - 1))
        return self.set_neuron_count(NEURON_COUNT_STEPS[idx])

    def topology_name(self) -> str:
        return TOPOLOGY_NAMES[self.topology_index]

    def set_threshold_spread(self, value: float) -> None:
        self._threshold_spread = float(np.clip(value, 0.0, 1.5))
        self._apply_heterogeneity()

    def set_tau_spread(self, value: float) -> None:
        self._tau_spread = float(np.clip(value, 0.0, 1.5))
        self._apply_heterogeneity()

    def set_refractory_spread(self, value: float) -> None:
        self._refractory_spread = float(np.clip(value, 0.0, 1.5))
        self._apply_heterogeneity()

    def set_drive_spread(self, value: float) -> None:
        self._drive_spread = float(np.clip(value, 0.0, 1.5))
        self._apply_heterogeneity()

    def set_heterogeneity(self, value: float) -> None:
        v = float(np.clip(value, 0.0, 1.0))
        self._heterogeneity = v
        self._threshold_spread = 0.95 * v
        self._tau_spread = 0.85 * v
        self._refractory_spread = 0.75 * v
        self._drive_spread = 1.10 * v
        self._apply_heterogeneity()

    def set_heterogeneity_seed(self, seed: int) -> None:
        with self._state_lock:
            self._heterogeneity_seed = int(seed)
            self._regenerate_heterogeneity_profiles()
            self._regenerate_population_roles()
            self._regenerate_delay_profile()
            self._regenerate_spatial_profile()
            self._reset_spike_history()
            if self._native is None:
                self._seed_fallback_state()
            self._noise_rng = np.random.default_rng(self._heterogeneity_seed + self.neuron_count * 41)
            self._rebuild_fallback_weights()
            self._apply_heterogeneity()

    def heterogeneity_state(self) -> dict:
        return {
            "heterogeneity": float(self._heterogeneity),
            "hetero_seed": int(self._heterogeneity_seed),
            "threshold_spread": float(self._threshold_spread),
            "tau_spread": float(self._tau_spread),
            "refractory_spread": float(self._refractory_spread),
            "drive_spread": float(self._drive_spread),
        }

    def set_inhibitory_ratio(self, value: float) -> None:
        with self._state_lock:
            self._inhibitory_ratio = float(np.clip(value, 0.0, 0.9))
            self._regenerate_population_roles()
            self._refresh_effective_weights()

    def set_inhibitory_gain(self, value: float) -> None:
        self._inhibitory_gain = float(np.clip(value, 0.0, 3.0))
        self._refresh_effective_weights()

    def set_excitatory_scale(self, value: float) -> None:
        self._excitatory_scale = float(np.clip(value, 0.0, 3.0))
        self._refresh_effective_weights()

    def set_inhibitory_scale(self, value: float) -> None:
        self._inhibitory_scale = float(np.clip(value, 0.0, 3.0))
        self._refresh_effective_weights()

    def set_delay_spread_steps(self, value: int) -> None:
        with self._state_lock:
            self._delay_spread_steps = int(np.clip(value, 0, 12))
            self._regenerate_delay_profile()
            self._reset_spike_history()

    def set_delay_jitter(self, value: float) -> None:
        with self._state_lock:
            self._delay_jitter = float(np.clip(value, 0.0, 1.0))
            self._regenerate_delay_profile()
            self._reset_spike_history()

    def phase2_state(self) -> dict:
        return {
            "inhibitory_ratio": float(self._inhibitory_ratio),
            "inhibitory_gain": float(self._inhibitory_gain),
            "excitatory_scale": float(self._excitatory_scale),
            "inhibitory_scale": float(self._inhibitory_scale),
            "delay_spread_steps": int(self._delay_spread_steps),
            "delay_jitter": float(self._delay_jitter),
        }

    def set_adaptation_strength(self, value: float) -> None:
        with self._state_lock:
            self._adaptation_strength = float(np.clip(value, 0.0, 3.0))

    def set_adaptation_decay(self, value: float) -> None:
        with self._state_lock:
            self._adaptation_decay = float(np.clip(value, 0.70, 0.999))

    def set_noise_amount(self, value: float) -> None:
        with self._state_lock:
            self._noise_amount = float(np.clip(value, 0.0, 1.0))

    def set_noise_color(self, value: str | int | float) -> None:
        with self._state_lock:
            if isinstance(value, str):
                mode = value.lower().strip()
                self._noise_color = "pink" if mode in {"pink", "p"} else "white"
            else:
                self._noise_color = "pink" if float(value) >= 0.5 else "white"

    def set_spatial_noise(self, value: float) -> None:
        with self._state_lock:
            self._spatial_noise = float(np.clip(value, 0.0, 1.0))

    def phase3_state(self) -> dict:
        return {
            "adaptation_strength": float(self._adaptation_strength),
            "adaptation_decay": float(self._adaptation_decay),
            "noise_amount": float(self._noise_amount),
            "noise_color": self._noise_color,
            "noise_color_index": 1 if self._noise_color == "pink" else 0,
            "spatial_noise": float(self._spatial_noise),
        }

    # ------------------------------------------------------------------
    # NumPy fallback internals
    # ------------------------------------------------------------------
    def _regenerate_heterogeneity_profiles(self) -> None:
        n = self.neuron_count
        if n <= 0:
            self._h_threshold = np.zeros(0, dtype=np.float32)
            self._h_tau = np.zeros(0, dtype=np.float32)
            self._h_refractory = np.zeros(0, dtype=np.float32)
            self._h_drive = np.zeros(0, dtype=np.float32)
            return

        rng = np.random.default_rng(self._heterogeneity_seed + n * 13 + self.topology_index * 101)

        def make_axis() -> np.ndarray:
            v = rng.standard_normal(n).astype(np.float32)
            v -= float(np.mean(v))
            std = float(np.std(v))
            if std > 1e-6:
                v /= std
            v = np.clip(v, -2.0, 2.0) * 0.5
            return v

        self._h_threshold = make_axis()
        self._h_tau = make_axis()
        self._h_refractory = make_axis()
        self._h_drive = make_axis()

    def _regenerate_population_roles(self) -> None:
        n = self.neuron_count
        if n <= 0:
            self._inhibitory_mask = np.zeros(0, dtype=bool)
            return

        n_inh = int(round(self._inhibitory_ratio * n))
        n_inh = int(np.clip(n_inh, 0, max(0, n - 1)))
        rng = np.random.default_rng(self._heterogeneity_seed + n * 29 + self.topology_index * 211)
        mask = np.zeros(n, dtype=bool)
        if n_inh > 0:
            idx = rng.permutation(n)[:n_inh]
            mask[idx] = True
        self._inhibitory_mask = mask

    def _regenerate_delay_profile(self) -> None:
        n = self.neuron_count
        if n <= 0:
            self._delay_steps = np.zeros(0, dtype=np.int16)
            return

        spread = int(max(0, self._delay_spread_steps))
        if spread == 0:
            self._delay_steps = np.zeros(n, dtype=np.int16)
            return

        rng = np.random.default_rng(self._heterogeneity_seed + n * 31 + self.topology_index * 223)
        base = rng.integers(0, spread + 1, size=n).astype(np.float32)
        if self._delay_jitter > 0.0:
            noise = rng.normal(0.0, self._delay_jitter * max(1.0, spread * 0.75), size=n).astype(np.float32)
            base = np.rint(base + noise)
        self._delay_steps = np.clip(base, 0, spread).astype(np.int16)

    def _reset_spike_history(self) -> None:
        hist_len = max(1, int(self._delay_spread_steps) + 1)
        self._spike_history = deque(maxlen=hist_len)
        zero = np.zeros(self.neuron_count, dtype=np.float32)
        self._zero_spikes = zero
        self._adaptation_current = np.zeros(self.neuron_count, dtype=np.float32)
        self._noise_state = np.zeros(self.neuron_count, dtype=np.float32)
        for _ in range(hist_len):
            self._spike_history.append(zero.copy())

    def _delayed_source_activity(self) -> np.ndarray:
        if self.neuron_count <= 0:
            return np.zeros(0, dtype=np.float32)

        expected_len = max(1, int(self._delay_spread_steps) + 1)
        if len(self._spike_history) == 0 or self._spike_history.maxlen != expected_len:
            self._reset_spike_history()
        elif self._spike_history[-1].shape[0] != self.neuron_count:
            self._reset_spike_history()

        if expected_len == 1:
            return self._spike_history[-1]

        out = np.zeros(self.neuron_count, dtype=np.float32)
        hist = list(self._spike_history)
        for delay in range(expected_len):
            mask = self._delay_steps == delay
            if not np.any(mask):
                continue
            src = hist[-1 - delay]
            if src.shape[0] != self.neuron_count:
                self._reset_spike_history()
                return self._spike_history[-1]
            out[mask] = src[mask]
        return out

    def _compute_syn_overlay(self, delayed_src: np.ndarray) -> np.ndarray:
        """Shape-safe synaptic overlay used by both native and fallback stepping."""
        n = int(self.neuron_count)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        if self.weights.shape != (n, n) or self._base_weights.shape != (n, n):
            self._rebuild_fallback_weights()

        src = np.asarray(delayed_src, dtype=np.float32).reshape(-1)
        if src.size != n:
            fixed = np.zeros(n, dtype=np.float32)
            ncopy = min(n, src.size)
            fixed[:ncopy] = src[:ncopy]
            src = fixed

        return self.weights @ src

    def _push_spike_history(self, spikes: np.ndarray) -> None:
        if self.neuron_count <= 0:
            return
        arr = np.asarray(spikes, dtype=np.float32).reshape(-1)
        if arr.size < self.neuron_count:
            tmp = np.zeros(self.neuron_count, dtype=np.float32)
            tmp[: arr.size] = arr
            arr = tmp
        elif arr.size > self.neuron_count:
            arr = arr[: self.neuron_count]
        self._spike_history.append(arr.copy())

    def _refresh_effective_weights(self) -> None:
        if self.neuron_count <= 0:
            self.weights = np.zeros((0, 0), dtype=np.float32)
            return

        if self._inhibitory_mask.shape[0] != self.neuron_count:
            self._regenerate_population_roles()

        src_scale = np.where(
            self._inhibitory_mask,
            -self._inhibitory_scale * self._inhibitory_gain,
            self._excitatory_scale,
        ).astype(np.float32)

        scaled = (self._base_weights * self._weight_scale).astype(np.float32)
        self.weights = scaled * src_scale[np.newaxis, :]

    def _regenerate_spatial_profile(self) -> None:
        n = self.neuron_count
        if n <= 0:
            self._spatial_profile = np.zeros(0, dtype=np.float32)
            return

        rows = max(1, self.rows)
        cols = max(1, self.cols)
        yy = np.arange(rows, dtype=np.float32).reshape(-1, 1)
        xx = np.arange(cols, dtype=np.float32).reshape(1, -1)

        # Structured low-frequency field for clustered modulation.
        field = (
            np.sin(2.0 * np.pi * yy / max(2.0, rows * 0.7))
            + np.cos(2.0 * np.pi * xx / max(2.0, cols * 0.8))
        ).astype(np.float32)

        rng = np.random.default_rng(self._heterogeneity_seed + n * 37 + self.topology_index * 241)
        field += 0.15 * rng.standard_normal(field.shape).astype(np.float32)

        flat = field.reshape(-1)[:n]
        flat -= float(np.mean(flat))
        std = float(np.std(flat))
        if std > 1e-6:
            flat /= std
        self._spatial_profile = np.clip(flat, -2.0, 2.0).astype(np.float32)

    def _phase3_runtime_term(self) -> np.ndarray:
        n = self.neuron_count
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        if self._noise_state.shape[0] != n:
            self._noise_state = np.zeros(n, dtype=np.float32)
        if self._adaptation_current.shape[0] != n:
            self._adaptation_current = np.zeros(n, dtype=np.float32)
        if self._spatial_profile.shape[0] != n:
            self._regenerate_spatial_profile()

        white = self._noise_rng.standard_normal(n).astype(np.float32)
        if self._noise_color == "pink":
            self._noise_state = (0.985 * self._noise_state + 0.15 * white).astype(np.float32)
            noise = self._noise_state
        else:
            noise = white

        noise_term = (self._noise_amount * 0.8) * noise
        spatial_term = (self._spatial_noise * 0.9) * self._spatial_profile
        adapt_term = self._adaptation_current

        return np.clip(noise_term + spatial_term - adapt_term, -2.5, 2.5).astype(np.float32)

    def _update_adaptation(self, spikes: np.ndarray) -> None:
        n = self.neuron_count
        if n <= 0:
            return
        s = np.asarray(spikes, dtype=np.float32).reshape(-1)
        if s.size < n:
            tmp = np.zeros(n, dtype=np.float32)
            tmp[: s.size] = s
            s = tmp
        elif s.size > n:
            s = s[:n]

        self._adaptation_current = (
            self._adaptation_current * self._adaptation_decay
            + self._adaptation_strength * s
        ).astype(np.float32)
        self._adaptation_current = np.clip(self._adaptation_current, 0.0, 4.0)

    def _apply_heterogeneity(self) -> None:
        if self.neuron_count <= 0:
            return

        # Spread controls are centered around base scalar parameters.
        thresh_vec = np.clip(
            self._threshold * (1.0 + 0.70 * self._threshold_spread * self._h_threshold),
            0.05,
            3.0,
        ).astype(np.float32)
        tau_vec = np.clip(
            self._tau * (1.0 + 0.75 * self._tau_spread * self._h_tau),
            1.0,
            200.0,
        ).astype(np.float32)
        refr_vec = np.clip(
            self._base_refractory_ms * (1.0 + 0.80 * self._refractory_spread * self._h_refractory),
            0.0,
            40.0,
        ).astype(np.float32)

        drive_offsets = (0.90 * self._drive_spread * self._h_drive).astype(np.float32)

        if self._native is not None:
            # Native backend currently supports scalar threshold/tau/refractory.
            # Spread controls are projected into per-neuron drive offsets so the
            # performer still gets diversity from all heterogeneity knobs.
            drive_offsets = drive_offsets + (
                -0.35 * self._threshold_spread * self._h_threshold
                -0.20 * self._tau_spread * self._h_tau
                -0.15 * self._refractory_spread * self._h_refractory
            ).astype(np.float32)
            self._native_drive_offsets = np.clip(drive_offsets, -1.5, 1.5).astype(np.float32)
            return

        self.v_thresh[:] = thresh_vec
        self.tau[:] = tau_vec
        self.refractory_t[:] = refr_vec
        self.external_drive[:] = np.clip(self._global_drive + drive_offsets, 0.0, 3.0)

    def _seed_fallback_state(self) -> None:
        rng = np.random.default_rng(self._heterogeneity_seed + self.neuron_count * 17 + self.topology_index * 193)
        self.v[:] = rng.uniform(0.02, 0.22, self.neuron_count).astype(np.float32)
        self.refractory[:] = 0.0
        self.spikes[:] = False

    def _rebuild_fallback_weights(self) -> None:
        n = self.neuron_count
        w = np.zeros((n, n), dtype=np.float32)
        rng = np.random.default_rng(self._heterogeneity_seed + 42 + self.topology_index * 97 + n)

        if self.topology_index == 0:  # Ring
            for i in range(n):
                for hop in (1, 2, 4):
                    w[i, (i + hop) % n] = 0.18 / hop
                    w[i, (i - hop + n) % n] = 0.18 / hop

        elif self.topology_index == 1:  # FullyConnected
            w[:] = 0.12 / np.sqrt(float(n))
            np.fill_diagonal(w, 0.0)

        elif self.topology_index == 2:  # Feedforward
            layers = 4
            layer_size = max(1, n // layers)
            for i in range(n):
                src_layer = min(i // layer_size, layers - 1)
                j0 = (src_layer + 1) * layer_size
                j1 = min((src_layer + 2) * layer_size, n)
                if j0 < n:
                    w[i, j0:j1] = 0.20

        elif self.topology_index == 3:  # SparseRandom
            mask = rng.random((n, n)) < 0.10
            np.fill_diagonal(mask, False)
            w[mask] = rng.uniform(0.05, 0.22, mask.sum()).astype(np.float32)

        else:  # SmallWorld
            for i in range(n):
                for hop in (1, 2, 3):
                    target = (i + hop) % n
                    if rng.random() < 0.05:
                        target = int(rng.integers(0, n))
                    w[i, target] = 0.16 / hop
                for _ in range(2):
                    w[i, int(rng.integers(0, n))] = float(rng.uniform(0.05, 0.22))

        self._base_weights = w
        self._refresh_effective_weights()

    def _pack_fallback_views(self) -> None:
        padded = self.rows * self.cols
        spikes_flat = np.zeros(padded, dtype=bool)
        pots_flat = np.zeros(padded, dtype=np.float32)

        spikes_flat[: self.neuron_count] = self.spikes
        pots_flat[: self.neuron_count] = np.clip(self.v / np.maximum(self.v_thresh, 1e-4), 0.0, 1.0)

        self._spikes = spikes_flat.reshape(self.rows, self.cols)
        self._potentials = pots_flat.reshape(self.rows, self.cols)
