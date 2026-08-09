"""Phase-5 anti-lock controller for runtime diversity correction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class _Clamp:
    low: float
    high: float


class DiversityController:
    """Monitors collapse metrics and issues bounded corrective nudges."""

    _CLAMPS = {
        "threshold": _Clamp(0.1, 2.0),
        "drive_n": _Clamp(0.0, 3.0),
        "heterogeneity": _Clamp(0.0, 1.0),
        "inhibitory_gain": _Clamp(0.0, 3.0),
        "delay_jitter": _Clamp(0.0, 1.0),
        "adaptation_strength": _Clamp(0.0, 3.0),
        "noise_amount": _Clamp(0.0, 1.0),
    }

    def __init__(
        self,
        synchrony_threshold: float = 0.72,
        entropy_threshold: float = 0.38,
        active_low: float = 0.10,
        active_high: float = 0.72,
        cooldown_seconds: float = 0.80,
    ) -> None:
        self._sync_threshold = float(synchrony_threshold)
        self._entropy_threshold = float(entropy_threshold)
        self._active_low = float(active_low)
        self._active_high = float(active_high)
        self._cooldown_seconds = float(cooldown_seconds)

        self._last_nudge_at = -1e9
        self._last_reason = "idle"
        self._last_magnitude = 0.0

    def tick(self, now: float, cfg: dict) -> tuple[dict, dict]:
        """Return (updates, meta) for anti-lock intervention if needed."""
        enabled = bool(cfg.get("anti_lock_enabled", False))
        strength = max(0.0, min(1.0, float(cfg.get("anti_lock_strength", 0.0))))

        sync = max(0.0, min(1.0, float(cfg.get("synchrony_index", 0.0))))
        entropy = max(0.0, min(1.0, float(cfg.get("spike_entropy", 0.0))))
        active = max(0.0, min(1.0, float(cfg.get("active_ratio_ma", cfg.get("active_ratio", 0.0)))))

        sync_score = 0.0
        if sync > self._sync_threshold:
            sync_score = (sync - self._sync_threshold) / max(1e-6, 1.0 - self._sync_threshold)

        entropy_score = 0.0
        if entropy < self._entropy_threshold:
            entropy_score = (self._entropy_threshold - entropy) / max(1e-6, self._entropy_threshold)

        overactive_score = 0.0
        if active > self._active_high:
            overactive_score = (active - self._active_high) / max(1e-6, 1.0 - self._active_high)

        underactive_score = 0.0
        if active < self._active_low:
            underactive_score = (self._active_low - active) / max(1e-6, self._active_low)

        collapse_score = max(0.0, 0.65 * sync_score + 0.35 * entropy_score)

        meta = {
            "anti_lock_enabled": enabled,
            "anti_lock_strength": strength,
            "anti_lock_collapse_score": float(collapse_score),
            "anti_lock_last_reason": self._last_reason,
            "anti_lock_last_magnitude": float(self._last_magnitude),
            "anti_lock_last_nudge_time": float(self._last_nudge_at),
        }

        if not enabled or strength <= 1e-6:
            return {}, meta

        elapsed = float(now) - self._last_nudge_at
        if elapsed < self._cooldown_seconds:
            return {}, meta

        intensity = max(collapse_score, overactive_score, underactive_score) * strength
        if intensity < 0.08:
            return {}, meta

        updates: dict[str, float] = {}
        reason = "desync"

        threshold = float(cfg.get("threshold", 1.0))
        drive = float(cfg.get("drive_n", 0.5))

        if overactive_score > 0.0:
            reason = "overactive"
            updates["threshold"] = threshold + 0.28 * intensity
            updates["drive_n"] = drive - 0.22 * intensity
        elif underactive_score > 0.0:
            reason = "underactive"
            updates["threshold"] = threshold - 0.18 * intensity
            updates["drive_n"] = drive + 0.20 * intensity
        else:
            # Neutral drive nudge to avoid staying in a narrow lock basin.
            updates["drive_n"] = drive - 0.04 * intensity

        updates["heterogeneity"] = float(cfg.get("heterogeneity", 0.0)) + 0.18 * intensity
        updates["inhibitory_gain"] = float(cfg.get("inhibitory_gain", 1.0)) + 0.20 * intensity
        updates["delay_jitter"] = float(cfg.get("delay_jitter", 0.0)) + 0.16 * intensity
        updates["adaptation_strength"] = float(cfg.get("adaptation_strength", 0.0)) + 0.22 * intensity
        updates["noise_amount"] = float(cfg.get("noise_amount", 0.0)) + 0.14 * intensity

        for key, value in list(updates.items()):
            clamp = self._CLAMPS.get(key)
            if clamp is not None:
                updates[key] = max(clamp.low, min(clamp.high, float(value)))

        self._last_nudge_at = float(now)
        self._last_reason = reason
        self._last_magnitude = float(intensity)

        meta["anti_lock_last_reason"] = self._last_reason
        meta["anti_lock_last_magnitude"] = self._last_magnitude
        meta["anti_lock_last_nudge_time"] = self._last_nudge_at

        return updates, meta
