"""
Leaky Integrate-and-Fire neuron network  (8 rows × 16 columns = 128 neurons).

Simulation is vectorised with NumPy.  All mutable arrays are plain NumPy arrays;
callers that share this object across threads must coordinate their own locking.
"""

import numpy as np
from config import ROWS, COLS


class LIFNetwork:
    # ── construction ──────────────────────────────────────────────────────────
    def __init__(self, rows: int = ROWS, cols: int = COLS):
        self.rows = rows
        self.cols = cols
        n = rows * cols

        # ── membrane state ────────────────────────────────────────────────────
        self.v            = np.random.uniform(0.0, 0.3, n).astype(np.float32)
        self.v_rest       = np.zeros(n, np.float32)
        self.v_thresh     = np.ones(n,  np.float32)         # default threshold = 1.0
        self.tau          = np.full(n, 20.0, np.float32)    # membrane time constant (ms)
        self.r_mem        = np.ones(n,  np.float32)         # membrane resistance
        self.refractory   = np.zeros(n, np.float32)         # remaining refractory time (ms)
        self.refractory_t = np.full(n, 5.0, np.float32)    # absolute refractory period (ms)
        self.dt           = 1.0                              # simulation time step (ms)

        # ── spike output ──────────────────────────────────────────────────────
        self.spikes       = np.zeros(n, bool)               # current spike vector

        # ── synaptic weights ──────────────────────────────────────────────────
        self._weight_scale = 1.0
        self._base_weights = self._make_random_weights(n)
        self.weights       = self._base_weights.copy()

        # ── external drive ────────────────────────────────────────────────────
        # Drive is drawn above threshold so neurons fire spontaneously;
        # heterogeneous values produce varied firing rates across the network.
        self.external_drive = np.random.uniform(1.1, 1.8, n).astype(np.float32)

        # ── display frequencies (set by FMSynth) ─────────────────────────────
        self.frequencies  = np.zeros((rows, cols), np.float32)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _make_random_weights(n: int) -> np.ndarray:
        """Sparse random weight matrix, ~10 % connectivity."""
        mask = np.random.rand(n, n) < 0.10
        np.fill_diagonal(mask, False)
        w = np.zeros((n, n), np.float32)
        w[mask] = np.random.randn(mask.sum()).astype(np.float32) * 0.25
        return w

    # ── simulation step ───────────────────────────────────────────────────────
    def step(self) -> np.ndarray:
        """Advance by one dt.  Returns spike matrix (ROWS, COLS)."""
        I_syn = self.weights @ self.spikes.astype(np.float32)
        I     = I_syn + self.external_drive

        in_ref = self.refractory > 0.0

        dv = (-(self.v - self.v_rest) + self.r_mem * I) * (self.dt / self.tau)
        self.v += dv
        self.v[in_ref] = self.v_rest[in_ref]

        self.spikes = self.v >= self.v_thresh
        self.v[self.spikes] = self.v_rest[self.spikes]
        self.refractory[self.spikes] = self.refractory_t[self.spikes]
        self.refractory = np.maximum(0.0, self.refractory - self.dt)

        return self.spikes.reshape(self.rows, self.cols)

    def get_potentials(self) -> np.ndarray:
        """Return normalised membrane potentials as (ROWS, COLS) in [0, 1]."""
        return np.clip(
            self.v.reshape(self.rows, self.cols) / self.v_thresh.reshape(self.rows, self.cols),
            0.0, 1.0
        )

    # ── parameter control (called from MIDI / main thread) ────────────────────
    def set_threshold(self, value: float) -> None:
        self.v_thresh[:] = float(value)

    def set_tau(self, value: float) -> None:
        self.tau[:] = float(value)

    def set_weight_scale(self, scale: float) -> None:
        self._weight_scale = float(scale)
        self.weights = self._base_weights * self._weight_scale

    def set_global_drive(self, value: float) -> None:
        # value is normalised 0-1 from MIDI; map to a musically useful drive range
        # (0 → below threshold, 1 → well above threshold for fast firing)
        self.external_drive[:] = 0.5 + float(value) * 1.5   # 0.5 – 2.0

    def set_neuron_drive(self, row: int, col: int, value: float) -> None:
        self.external_drive[row * self.cols + col] = float(value)

    def randomize_weights(self) -> None:
        self._base_weights = self._make_random_weights(self.rows * self.cols)
        self.weights = self._base_weights * self._weight_scale
