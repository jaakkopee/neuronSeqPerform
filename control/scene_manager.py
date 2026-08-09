"""Scene recall, morphing, and lightweight strategy modulation for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
from typing import Any

NEURON_COUNT_STEPS = [128, 256, 512, 1024, 2048, 4096]

SCENE_PARAM_KEYS = [
    "tempo",
    "master_volume",
    "active_pairs",
    "decay_speed",
    "mod_index_scale",
    "quantization_strength",
    "threshold",
    "tau",
    "weight_scale",
    "drive_n",
    "swing_amount",
    "ratio_scale",
    "topology_index",
    "neuron_count",
    "lif_steps",
    "root_note",
    "scale_name",
    "heterogeneity",
    "inhibitory_ratio",
    "inhibitory_gain",
    "excitatory_scale",
    "inhibitory_scale",
    "delay_spread_steps",
    "delay_jitter",
    "adaptation_strength",
    "adaptation_decay",
    "noise_amount",
    "noise_color",
    "spatial_noise",
    "scene_morph_time",
]


@dataclass(slots=True)
class ScenePreset:
    slot: int
    name: str
    targets: dict[str, Any]
    modulators: list[dict[str, Any]] = field(default_factory=list)
    clamps: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(slots=True)
class _MorphState:
    slot: int
    start_time: float
    duration: float
    start_targets: dict[str, Any]
    end_targets: dict[str, Any]


class SceneManager:
    """Owns scene preset state, launch/save behavior, and morph progression."""

    DISCRETE_KEYS = {
        "topology_index",
        "neuron_count",
        "lif_steps",
        "root_note",
        "scale_name",
        "noise_color",
    }

    def __init__(self, presets: list[ScenePreset] | None = None) -> None:
        preset_list = presets if presets is not None else make_default_scenes()
        self._scenes: dict[int, ScenePreset] = {int(p.slot): p for p in preset_list}
        if 0 not in self._scenes:
            raise ValueError("Scene slot 0 must exist")

        self._active_slot = 0
        self._strategy_started_at = 0.0
        self._morph: _MorphState | None = None
        self._queued_launch_state: dict[str, Any] | None = None

    def scene_name(self, slot: int | None = None) -> str:
        idx = self._active_slot if slot is None else int(max(0, min(9, slot)))
        return self._scenes.get(idx, self._scenes[0]).name

    @property
    def active_slot(self) -> int:
        return self._active_slot

    def launch_scene(self, slot: int, start_state: dict[str, Any], now: float, morph_time: float) -> str:
        idx = int(max(0, min(9, slot)))
        if idx not in self._scenes:
            idx = 0

        start_targets = self._filter_targets(start_state)
        end_targets = copy.deepcopy(self._scenes[idx].targets)
        duration = max(0.0, float(morph_time))

        if duration <= 1e-4:
            self._queued_launch_state = end_targets
            self._active_slot = idx
            self._strategy_started_at = now
            self._morph = None
            return self._scenes[idx].name

        self._morph = _MorphState(
            slot=idx,
            start_time=float(now),
            duration=duration,
            start_targets=start_targets,
            end_targets=end_targets,
        )
        return self._scenes[idx].name

    def save_scene(self, slot: int, snapshot: dict[str, Any]) -> str:
        idx = int(max(0, min(9, slot)))
        base = self._scenes.get(idx)
        if base is None:
            base = copy.deepcopy(self._scenes[0])
            base.slot = idx

        base.targets = self._filter_targets(snapshot)
        # User-saved scenes default to static recall for predictable A/B workflow.
        base.modulators = []
        base.name = f"User {idx}"
        self._scenes[idx] = base
        return base.name

    def tick(self, now: float) -> tuple[dict[str, Any], dict[str, Any]]:
        updates: dict[str, Any] = {}
        morph_progress = 1.0
        morph_active = False

        display_slot = self._active_slot

        if self._queued_launch_state is not None:
            updates.update(copy.deepcopy(self._queued_launch_state))
            self._queued_launch_state = None

        if self._morph is not None:
            m = self._morph
            elapsed = max(0.0, float(now) - m.start_time)
            morph_progress = 1.0 if m.duration <= 1e-9 else max(0.0, min(1.0, elapsed / m.duration))
            morph_active = morph_progress < 1.0
            display_slot = m.slot

            updates.update(self._interpolate_targets(m.start_targets, m.end_targets, morph_progress))

            if not morph_active:
                self._active_slot = m.slot
                self._strategy_started_at = float(now)
                self._morph = None
                display_slot = self._active_slot

        active_scene = self._scenes.get(display_slot, self._scenes[0])

        if not morph_active:
            strategy = self._strategy_targets(active_scene, now)
            if strategy:
                updates.update(strategy)

        self._apply_clamps(active_scene, updates)

        meta = {
            "scene_index": int(display_slot),
            "scene_name": active_scene.name,
            "scene_morph_progress": float(morph_progress),
            "scene_morph_active": bool(morph_active),
            "scene_strategy_name": self._strategy_name(active_scene),
        }
        return updates, meta

    def _filter_targets(self, source: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in SCENE_PARAM_KEYS:
            if key in source:
                out[key] = copy.deepcopy(source[key])
        return out

    def _interpolate_targets(self, start: dict[str, Any], end: dict[str, Any], alpha: float) -> dict[str, Any]:
        out: dict[str, Any] = {}
        keys = set(start) | set(end)

        for key in keys:
            a = start.get(key, end.get(key))
            b = end.get(key, a)

            if key in self.DISCRETE_KEYS:
                out[key] = b if alpha >= 1.0 else a
                continue

            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                out[key] = float(a) + (float(b) - float(a)) * alpha
            else:
                out[key] = b if alpha >= 1.0 else a

        return out

    def _strategy_targets(self, scene: ScenePreset, now: float) -> dict[str, Any]:
        if not scene.modulators:
            return {}

        t = max(0.0, float(now) - self._strategy_started_at)
        out: dict[str, Any] = {}

        for mod in scene.modulators:
            param = str(mod.get("param", "")).strip()
            if not param:
                continue
            base = scene.targets.get(param)
            if not isinstance(base, (int, float)):
                continue

            freq_hz = max(0.0, float(mod.get("freq_hz", 0.0)))
            depth = float(mod.get("depth", 0.0))
            phase = float(mod.get("phase", 0.0))
            wave = str(mod.get("wave", "sine")).lower()
            osc = self._wave_sample(wave, t, freq_hz, phase)
            out[param] = float(base) + depth * osc

        return out

    def _wave_sample(self, wave: str, t: float, freq_hz: float, phase: float) -> float:
        if freq_hz <= 1e-9:
            return 0.0

        if wave == "square":
            return 1.0 if math.sin(2.0 * math.pi * freq_hz * t + phase) >= 0.0 else -1.0

        if wave == "triangle":
            phase_cycles = phase / (2.0 * math.pi)
            cycle = (t * freq_hz + phase_cycles) % 1.0
            return 2.0 * abs(2.0 * cycle - 1.0) - 1.0

        return math.sin(2.0 * math.pi * freq_hz * t + phase)

    def _apply_clamps(self, scene: ScenePreset, values: dict[str, Any]) -> None:
        for key, (vmin, vmax) in scene.clamps.items():
            if key not in values:
                continue
            raw = values[key]
            if not isinstance(raw, (int, float)):
                continue
            values[key] = max(float(vmin), min(float(vmax), float(raw)))

        if "active_pairs" in values and isinstance(values["active_pairs"], (int, float)):
            values["active_pairs"] = int(max(1, min(4, round(values["active_pairs"]))))

        if "lif_steps" in values and isinstance(values["lif_steps"], (int, float)):
            values["lif_steps"] = int(max(1, min(64, round(values["lif_steps"]))))

        if "topology_index" in values and isinstance(values["topology_index"], (int, float)):
            values["topology_index"] = int(max(0, min(4, round(values["topology_index"]))))

        if "neuron_count" in values and isinstance(values["neuron_count"], (int, float)):
            requested = int(round(values["neuron_count"]))
            values["neuron_count"] = min(NEURON_COUNT_STEPS, key=lambda n: abs(n - requested))

        if "root_note" in values and isinstance(values["root_note"], (int, float)):
            values["root_note"] = int(max(48, min(84, round(values["root_note"]))))

    def _strategy_name(self, scene: ScenePreset) -> str:
        if not scene.modulators:
            return "static"
        return "+".join(str(m.get("param", "mod")) for m in scene.modulators)


def make_default_scenes() -> list[ScenePreset]:
    base = {
        "tempo": 120.0,
        "master_volume": 0.5,
        "active_pairs": 2,
        "decay_speed": 0.5,
        "mod_index_scale": 0.25,
        "quantization_strength": 0.85,
        "threshold": 1.0,
        "tau": 20.0,
        "weight_scale": 1.0,
        "drive_n": 0.7,
        "swing_amount": 0.08,
        "ratio_scale": 1.0,
        "topology_index": 0,
        "neuron_count": 512,
        "lif_steps": 12,
        "root_note": 60,
        "scale_name": "major",
        "heterogeneity": 0.22,
        "inhibitory_ratio": 0.18,
        "inhibitory_gain": 1.0,
        "excitatory_scale": 1.0,
        "inhibitory_scale": 1.0,
        "delay_spread_steps": 2,
        "delay_jitter": 0.12,
        "adaptation_strength": 0.35,
        "adaptation_decay": 0.93,
        "noise_amount": 0.12,
        "noise_color": "white",
        "spatial_noise": 0.18,
        "scene_morph_time": 1.6,
    }

    clamps = {
        "tempo": (40.0, 200.0),
        "master_volume": (0.0, 1.0),
        "decay_speed": (0.0, 1.0),
        "mod_index_scale": (0.0, 3.0),
        "quantization_strength": (0.0, 1.0),
        "threshold": (0.1, 2.0),
        "tau": (5.0, 100.0),
        "weight_scale": (0.0, 3.0),
        "drive_n": (0.0, 3.0),
        "swing_amount": (0.0, 0.5),
        "ratio_scale": (0.5, 2.0),
        "heterogeneity": (0.0, 1.0),
        "inhibitory_ratio": (0.0, 0.9),
        "inhibitory_gain": (0.0, 3.0),
        "excitatory_scale": (0.0, 3.0),
        "inhibitory_scale": (0.0, 3.0),
        "delay_spread_steps": (0.0, 12.0),
        "delay_jitter": (0.0, 1.0),
        "adaptation_strength": (0.0, 3.0),
        "adaptation_decay": (0.70, 0.999),
        "noise_amount": (0.0, 1.0),
        "spatial_noise": (0.0, 1.0),
        "scene_morph_time": (0.0, 8.0),
    }

    def scene(slot: int, name: str, *, mods: list[dict[str, Any]] | None = None, **overrides: Any) -> ScenePreset:
        targets = dict(base)
        targets.update(overrides)
        return ScenePreset(
            slot=slot,
            name=name,
            targets=targets,
            modulators=mods or [],
            clamps=dict(clamps),
        )

    return [
        scene(0, "Calm lattice", drive_n=0.55, threshold=1.08, heterogeneity=0.16, noise_amount=0.05, delay_spread_steps=1),
        scene(1, "Traveling wave", topology_index=0, drive_n=0.78, threshold=0.92, delay_spread_steps=5, delay_jitter=0.20,
              mods=[{"param": "drive_n", "freq_hz": 0.10, "depth": 0.12}, {"param": "swing_amount", "freq_hz": 0.05, "depth": 0.06, "wave": "triangle"}]),
        scene(2, "Pulsing islands", topology_index=4, drive_n=0.74, heterogeneity=0.52, spatial_noise=0.62, noise_amount=0.22,
              mods=[{"param": "threshold", "freq_hz": 0.16, "depth": 0.14}, {"param": "noise_amount", "freq_hz": 0.11, "depth": 0.10}]),
        scene(3, "Sparse sparks", topology_index=3, drive_n=0.44, threshold=1.28, tau=30.0, heterogeneity=0.34, noise_amount=0.10,
              mods=[{"param": "drive_n", "freq_hz": 0.08, "depth": 0.08, "wave": "triangle"}]),
        scene(4, "Burst storm", topology_index=1, neuron_count=1024, drive_n=1.30, threshold=0.72, tau=12.0, weight_scale=1.45,
              heterogeneity=0.72, delay_spread_steps=7, noise_amount=0.28, adaptation_strength=0.22,
              mods=[{"param": "drive_n", "freq_hz": 0.22, "depth": 0.20}, {"param": "threshold", "freq_hz": 0.22, "depth": 0.16, "phase": 3.14159}]),
        scene(5, "Predator prey", topology_index=4, drive_n=0.86, threshold=0.95, inhibitory_ratio=0.36, inhibitory_gain=1.9,
              excitatory_scale=1.15, inhibitory_scale=1.2, delay_spread_steps=4,
              mods=[{"param": "inhibitory_gain", "freq_hz": 0.09, "depth": 0.45}, {"param": "drive_n", "freq_hz": 0.09, "depth": 0.16, "phase": 3.14159}]),
        scene(6, "Small-world drift", topology_index=4, drive_n=0.70, threshold=1.02, heterogeneity=0.48, delay_spread_steps=6,
              delay_jitter=0.34, tau=24.0, mods=[{"param": "delay_jitter", "freq_hz": 0.07, "depth": 0.18}]),
        scene(7, "Edge-of-chaos", topology_index=3, neuron_count=1024, drive_n=0.98, threshold=0.88, tau=16.0, weight_scale=1.3,
              heterogeneity=0.74, inhibitory_ratio=0.24, inhibitory_gain=1.2, delay_spread_steps=8,
              adaptation_strength=0.38, noise_amount=0.26, spatial_noise=0.42,
              mods=[{"param": "noise_amount", "freq_hz": 0.12, "depth": 0.16}, {"param": "adaptation_strength", "freq_hz": 0.10, "depth": 0.20}]),
        scene(8, "Polyrhythm mesh", topology_index=2, drive_n=0.82, threshold=0.96, tau=18.0, swing_amount=0.28,
              quantization_strength=0.45, lif_steps=24, heterogeneity=0.58, delay_spread_steps=5,
              mods=[{"param": "tempo", "freq_hz": 0.03, "depth": 14.0}, {"param": "drive_n", "freq_hz": 0.15, "depth": 0.12, "wave": "triangle"}]),
        scene(9, "Freeze then shatter", topology_index=1, drive_n=0.60, threshold=1.42, tau=28.0, heterogeneity=0.40,
              inhibitory_gain=1.6, delay_spread_steps=2, adaptation_strength=0.66, noise_amount=0.32, noise_color="pink", spatial_noise=0.54,
              mods=[{"param": "threshold", "freq_hz": 0.06, "depth": 0.44, "wave": "square"}, {"param": "noise_amount", "freq_hz": 0.13, "depth": 0.24}]),
    ]
