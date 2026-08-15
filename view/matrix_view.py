"""Dynamic neuron-grid renderer with Metal-accelerated GPU rendering."""

from __future__ import annotations

import colorsys
import numpy as np
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, SCALE_NAMES
from view.matrix_metal import MatrixViewMetalRenderer


BG_TOP = (10, 14, 28)
BG_BOT = (6, 9, 18)
PANEL_BG = (12, 18, 36)
PANEL_BORDER = (38, 56, 98)
TEXT_MAIN = (230, 238, 255)
TEXT_SUB = (140, 164, 210)
STEP_HIGHLIGHT = (250, 220, 120)

TOPOLOGY_NAMES = ["Ring", "FullyConnected", "Feedforward", "SparseRandom", "SmallWorld"]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

KNOB_CHEATSHEET = {
    "A": [(3, "Tempo"), (9, "Volume"), (12, "Pairs"), (13, "Decay"), (14, "FMmod"), (15, "Quant")],
    "B": [(16, "Thresh"), (17, "Tau"), (18, "Weight"), (19, "Drive"), (20, "Swing"), (21, "Ratio")],
    "C": [(22, "ATTgt*"), (23, "Topo"), (24, "Neurons"), (25, "Steps"), (26, "Root"), (27, "Scale")],
}
PAD_CHEATSHEET_A = [NOTE_NAMES[i % 12] for i in range(16)]
PAD_CHEATSHEET_B = [
    (SCALE_NAMES[i][:6] if i < len(SCALE_NAMES) else "---")
    for i in range(10)
] + ["---", "Oct-", "Oct+", "---", "---", "---"]  # 16 total: pads 0-9 scales, 11-12 octave
PAD_CHEATSHEET_C = [
    "RndW", "Reset", "+Step", "-Step", "Boost", "HalfW", "Topo-", "Topo+",
    "N--", "N++", "Het-", "Het+", "I--", "I++", "Dly-", "Dly+",
]


def _hsv_to_rgb255(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


class MatrixView:
    def __init__(self):
        pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()

        default_w = SCREEN_WIDTH
        default_h = SCREEN_HEIGHT
        try:
            display_info = pygame.display.Info()
            default_w = min(SCREEN_WIDTH, max(800, int(display_info.current_w * 0.94)))
            default_h = min(SCREEN_HEIGHT, max(620, int(display_info.current_h * 0.9)))
        except pygame.error:
            # Keep configured defaults if display info is unavailable.
            pass
        self._windowed_size = (default_w, default_h)
        self._is_fullscreen = False
        self._screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
        pygame.display.set_caption("NeuronSeqPerform")
        self._clock = pygame.time.Clock()

        self._fn_small = pygame.font.SysFont("monospace", 12)
        self._fn_medium = pygame.font.SysFont("monospace", 14)
        self._fn_title = pygame.font.SysFont("monospace", 20, bold=True)

        self.spikes = np.zeros((8, 16), dtype=bool)
        self.potentials = np.zeros((8, 16), dtype=np.float32)
        self.frequencies = np.zeros((8, 16), dtype=np.float32)
        self.current_step = 0
        self.config_state: dict = {}

        self._fade = np.zeros((8, 16), dtype=np.float32)
        self._pad_fade: dict[int, float] = {}
        
        # Initialize Metal renderer for GPU-accelerated grid rendering
        self._metal_renderer = MatrixViewMetalRenderer(8, 16, cell_width=24, gap_width=2)
        self._grid_texture = None
        self._grid_surface = None

    def _toggle_fullscreen(self) -> None:
        if self._is_fullscreen:
            self._screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
            self._is_fullscreen = False
            return

        self._windowed_size = self._screen.get_size()
        self._screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self._is_fullscreen = True

    def update(self, spikes, potentials, frequencies, current_step, config_state):
        self.spikes = np.asarray(spikes, dtype=bool)
        self.potentials = np.asarray(potentials, dtype=np.float32)
        self.frequencies = np.asarray(frequencies, dtype=np.float32)
        self.current_step = int(current_step)
        self.config_state = config_state

        if self._fade.shape != self.spikes.shape:
            # Spike grid is the authoritative visual shape during live resizes.
            self._fade = np.zeros(self.spikes.shape, dtype=np.float32)
            # Recreate Metal renderer for new grid dimensions
            try:
                self._metal_renderer = MatrixViewMetalRenderer(
                    self.spikes.shape[0], self.spikes.shape[1], 
                    cell_width=24, gap_width=2
                )
            except Exception as e:
                print(f"[MatrixView] Failed to recreate renderer: {e}")
        
        self._fade = np.where(self.spikes, 1.0, self._fade).astype(np.float32)

    def draw(self) -> None:
        self._draw_background()
        self._draw_title()
        self._draw_grid()
        self._draw_status_panel()
        pygame.display.flip()

        self._fade *= 0.82
        for k in list(self._pad_fade):
            self._pad_fade[k] *= 0.8
            if self._pad_fade[k] < 0.01:
                del self._pad_fade[k]

    def _draw_background(self) -> None:
        win_w, win_h = self._screen.get_size()
        h = win_h
        for y in range(h):
            t = y / max(1, h - 1)
            c = (
                int(BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * t),
                int(BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * t),
                int(BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * t),
            )
            pygame.draw.line(self._screen, c, (0, y), (win_w, y))

    def _draw_title(self) -> None:
        title = self._fn_title.render("NeuronSeqPerform  |  LIF Metal Grid", True, TEXT_MAIN)
        self._screen.blit(title, (20, 12))

    def _draw_grid(self) -> None:
        rows, cols = self.spikes.shape
        win_w, win_h = self._screen.get_size()

        x = 20
        y = 48
        w = win_w - 40
        h = int(win_h * 0.56)

        pygame.draw.rect(self._screen, (9, 16, 34), (x, y, w, h), border_radius=10)
        pygame.draw.rect(self._screen, PANEL_BORDER, (x, y, w, h), width=1, border_radius=10)

        if rows <= 0 or cols <= 0:
            return

        cw = w / cols
        ch = h / rows

        # Step column highlight always aligns to sequencer columns (0..15).
        step_col = int(np.clip(self.current_step, 0, cols - 1))
        hx = x + int(step_col * cw)
        pygame.draw.rect(
            self._screen,
            (38, 48, 78),
            (hx, y + 2, max(2, int(cw)), h - 4),
            border_radius=6,
        )

        # Use Metal renderer to render grid cells to texture
        threshold = self.config_state.get("threshold", 1.0)
        try:
            # Render grid with Metal GPU acceleration
            texture_data = self._metal_renderer.render(
                self.potentials, 
                self.spikes.astype(np.uint8),
                threshold
            )
            
            # Convert NumPy array to Pygame surface and display
            # texture_data shape is (height, width, 4) with RGBA channels
            tex_h, tex_w = texture_data.shape[:2]
            # Use frombuffer with RGBA format to create surface from byte data
            self._grid_surface = pygame.image.frombuffer(
                texture_data.tobytes(),
                (tex_w, tex_h),
                'RGBA'
            )
            self._grid_surface = pygame.transform.scale(self._grid_surface, (w, h))
            self._screen.blit(self._grid_surface, (x, y))
            
        except Exception as e:
            # Fallback to CPU rendering if Metal fails
            print(f"[MatrixView] Metal render failed ({e}), falling back to CPU")
            self._draw_grid_cpu(x, y, w, h, rows, cols)
        
        # Thin lattice lines make dense topologies readable even with 4k neurons.
        line_c = (24, 30, 52)
        for c in range(1, cols):
            lx = x + int(c * cw)
            pygame.draw.line(self._screen, line_c, (lx, y), (lx, y + h), 1)
        for r in range(1, rows):
            ly = y + int(r * ch)
            pygame.draw.line(self._screen, line_c, (x, ly), (x + w, ly), 1)

        pygame.draw.rect(self._screen, STEP_HIGHLIGHT, (hx, y + h - 5, max(2, int(cw)), 4), border_radius=2)
    
    def _draw_grid_cpu(self, x: int, y: int, w: int, h: int, rows: int, cols: int) -> None:
        """CPU-based fallback grid rendering."""
        cw = w / cols
        ch = h / rows
        
        for r in range(rows):
            for c in range(cols):
                pot = float(self.potentials[r, c]) if r < self.potentials.shape[0] and c < self.potentials.shape[1] else 0.0
                pot = max(0.0, min(1.0, pot))
                fade = float(self._fade[r, c]) if r < self._fade.shape[0] and c < self._fade.shape[1] else 0.0
                is_spike = bool(self.spikes[r, c])

                # Row hue drift + potential brightness gives colorful "activity fabric".
                hue = ((r / max(1, rows)) * 0.82 + (c / max(1, cols)) * 0.18 + pot * 0.15) % 1.0
                sat = 0.75
                val = 0.18 + pot * 0.72
                base = _hsv_to_rgb255(hue, sat, val)

                if is_spike or fade > 0.01:
                    boost = min(1.0, fade)
                    spike_col = _hsv_to_rgb255((hue + 0.08) % 1.0, 0.35, 1.0)
                    col = tuple(int(base[i] + (spike_col[i] - base[i]) * boost) for i in range(3))
                else:
                    col = base

                rx = x + int(c * cw) + 1
                ry = y + int(r * ch) + 1
                rw = max(1, int(cw) - 2)
                rh = max(1, int(ch) - 2)
                pygame.draw.rect(self._screen, col, (rx, ry, rw, rh), border_radius=2)

    def _draw_status_panel(self) -> None:
        win_w, win_h = self._screen.get_size()
        y = int(win_h * 0.64)
        x = 20
        w = win_w - 40
        h = win_h - y - 18

        pygame.draw.rect(self._screen, PANEL_BG, (x, y, w, h), border_radius=10)
        pygame.draw.rect(self._screen, PANEL_BORDER, (x, y, w, h), width=1, border_radius=10)

        cfg = self.config_state
        rows, cols = self.spikes.shape

        line1 = (
            f"Topology: {cfg.get('topology_name', 'Ring')}   "
            f"Neurons: {cfg.get('neuron_count', rows * cols)}   "
            f"Grid: {rows}x{cols}   "
            f"Step: {self.current_step + 1}/{cols}"
        )
        line2 = (
            f"Tempo: {cfg.get('tempo', 120.0):.1f} BPM   "
            f"Threshold: {cfg.get('threshold', 1.0):.2f}   "
            f"Tau: {cfg.get('tau', 20.0):.1f} ms   "
            f"Weight: {cfg.get('weight_scale', 1.0):.2f}   "
            f"Drive: {cfg.get('drive_n', 0.5):.2f}   "
            f"Het: {cfg.get('heterogeneity', 0.0):.2f}"
        )
        line3 = (
            f"Scale: {cfg.get('scale_name', 'major')}   "
            f"Root: {cfg.get('root_note', 60)}   "
            f"Oct: {cfg.get('octave_offset', 0):+d}   "
            f"LIF steps/tick: {cfg.get('lif_steps', 12)}   "
            f"AT target: {cfg.get('aftertouch_target', 'threshold')}"
        )
        line4 = (
            f"Het spread Thr/Tau/Ref/Drv: "
            f"{cfg.get('threshold_spread', 0.0):.2f}/"
            f"{cfg.get('tau_spread', 0.0):.2f}/"
            f"{cfg.get('refractory_spread', 0.0):.2f}/"
            f"{cfg.get('drive_spread', 0.0):.2f}   "
            f"E/I r:{cfg.get('inhibitory_ratio', 0.18):.2f} g:{cfg.get('inhibitory_gain', 1.0):.2f} "
            f"xE:{cfg.get('excitatory_scale', 1.0):.2f} xI:{cfg.get('inhibitory_scale', 1.0):.2f}   "
            f"Dly:{cfg.get('delay_spread_steps', 0)} jit:{cfg.get('delay_jitter', 0.0):.2f}   "
            f"Seed:{cfg.get('hetero_seed', 1337)}"
        )
        line5 = (
            f"Adapt str:{cfg.get('adaptation_strength', 0.0):.2f} "
            f"dec:{cfg.get('adaptation_decay', 0.93):.3f}   "
            f"Noise amt:{cfg.get('noise_amount', 0.0):.2f} "
            f"mode:{cfg.get('noise_color', 'white')}   "
            f"Spatial:{cfg.get('spatial_noise', 0.0):.2f}"
        )

        self._screen.blit(self._fn_medium.render(line1, True, TEXT_MAIN), (x + 12, y + 12))
        self._screen.blit(self._fn_small.render(line2, True, TEXT_SUB), (x + 12, y + 38))
        self._screen.blit(self._fn_small.render(line3, True, TEXT_SUB), (x + 12, y + 58))
        self._screen.blit(self._fn_small.render(line4, True, TEXT_SUB), (x + 12, y + 76))
        self._screen.blit(self._fn_small.render(line5, True, TEXT_SUB), (x + 12, y + 94))
        self._draw_scene_status(x + 12, y + 112, w - 24, cfg)

        top_idx = int(cfg.get("topology_index", 0))
        metric_w = 220
        metric_x = x + w - 12 - metric_w
        metric_y = y + 136
        legend_w = max(220, metric_x - (x + 12) - 8)

        self._draw_topology_legend(x + 12, y + 136, legend_w, 54, top_idx)
        self._draw_metric_panel(metric_x, metric_y, metric_w, 54, cfg)

        cheat_y = y + 194
        hints_y = y + h - 16
        cheat_h = max(40, hints_y - 8 - cheat_y)
        self._draw_controller_cheatsheet(x + 12, cheat_y, w - 24, cheat_h, cfg)

        hints = "MPD218: full 3x16 pads + 3x6 knobs shown below. *AT target includes heterogeneity + phase-2 + phase-3 controls."
        self._screen.blit(self._fn_small.render(hints, True, (116, 184, 232)), (x + 12, hints_y))

    def _draw_scene_status(self, x: int, y: int, w: int, cfg: dict) -> None:
        scene_idx = int(cfg.get("scene_index", 0))
        scene_name = str(cfg.get("scene_name", "-"))
        strategy = str(cfg.get("scene_strategy_name", "static"))
        morph = float(np.clip(float(cfg.get("scene_morph_progress", 1.0)), 0.0, 1.0))
        morph_active = bool(cfg.get("scene_morph_active", False))

        title = f"Scene {scene_idx}: {scene_name}"
        if strategy:
            title += f"  [{strategy}]"
        self._screen.blit(self._fn_small.render(title, True, TEXT_MAIN), (x, y))

        bar_w = min(220, max(120, w // 4))
        bx = x + w - bar_w
        by = y + 2
        pygame.draw.rect(self._screen, (30, 44, 70), (bx, by, bar_w, 10), border_radius=3)
        fill_w = max(1, int(bar_w * morph))
        fill_c = (250, 214, 120) if morph_active else (138, 236, 166)
        pygame.draw.rect(self._screen, fill_c, (bx, by, fill_w, 10), border_radius=3)
        morph_txt = f"Morph {int(round(morph * 100.0)):3d}%"
        self._screen.blit(self._fn_small.render(morph_txt, True, TEXT_SUB), (bx, by + 12))

    def _draw_metric_panel(self, x: int, y: int, w: int, h: int, cfg: dict) -> None:
        pygame.draw.rect(self._screen, (14, 24, 44), (x, y, w, h), border_radius=6)
        pygame.draw.rect(self._screen, (66, 96, 148), (x, y, w, h), width=1, border_radius=6)

        metrics = [
            ("Sync", float(cfg.get("synchrony_index", 0.0)), (250, 214, 120)),
            ("Ent", float(cfg.get("spike_entropy", 0.0)), (124, 214, 255)),
            ("Act", float(cfg.get("active_ratio", 0.0)), (138, 236, 166)),
        ]

        row_h = 16
        bar_x = x + 42
        bar_w = w - 86
        for i, (name, value, color) in enumerate(metrics):
            v = max(0.0, min(1.0, value))
            ry = y + 3 + i * row_h

            self._screen.blit(self._fn_small.render(name, True, TEXT_SUB), (x + 6, ry + 2))
            pygame.draw.rect(self._screen, (30, 44, 70), (bar_x, ry + 4, bar_w, 8), border_radius=3)
            fill_w = max(1, int(bar_w * v))
            pygame.draw.rect(self._screen, color, (bar_x, ry + 4, fill_w, 8), border_radius=3)
            val = self._fn_small.render(f"{v:.2f}", True, TEXT_MAIN)
            self._screen.blit(val, (x + w - 36, ry + 2))

    def _draw_controller_cheatsheet(self, x: int, y: int, w: int, h: int, cfg: dict) -> None:
        last = cfg.get("last_midi") or {}
        kind = last.get("kind")
        bank = last.get("bank")
        slot = int(last.get("slot", -1)) if isinstance(last.get("slot", -1), int) else -1

        if h <= 40:
            return

        # Knob cheat-sheet (3 rows x 6)
        knob_h = int(h * 0.45)
        knob_h = max(28, min(72, knob_h))
        if h - knob_h - 6 < 24:
            knob_h = max(24, h - 6 - 24)
        self._draw_knob_cheatsheet(x, y, w, knob_h, kind, bank, slot)

        # Pad cheat-sheet (3 rows x 16)
        pad_y = y + knob_h + 6
        pad_h = max(24, h - knob_h - 6)
        self._draw_pad_cheatsheet(x, pad_y, w, pad_h, kind, bank, slot)

        if last:
            cc = last.get("cc")
            note = last.get("note")
            label = last.get("label", "")
            if kind == "cc":
                info = f"Last: Knob Bank {bank}{slot + 1}  CC{cc}  {label}"
            elif kind == "pad":
                info = f"Last: Pad Bank {bank}{slot + 1:02d}  Note {note}  {label}"
            else:
                info = "Last: -"
            self._screen.blit(self._fn_small.render(info, True, (196, 216, 250)), (x, y - 14))

    def _draw_knob_cheatsheet(self, x: int, y: int, w: int, h: int, kind: str, bank: str, slot: int) -> None:
        banks = ["A", "B", "C"]
        sel_y_nudge = 6
        row_gap = 4
        row_h = max(8, int((h - row_gap * 2) / 3))
        for ri, b in enumerate(banks):
            ry = y + ri * (row_h + row_gap)
            items = KNOB_CHEATSHEET[b]
            gap = 4
            tile_w = max(52, int((w - gap * 5) / 6))
            for ci, (cc, lbl) in enumerate(items):
                rx = x + ci * (tile_w + gap)
                selected = (kind == "cc" and bank == b and slot == ci)
                draw_ry = ry + (sel_y_nudge if selected else 0)
                fill = (36, 54, 86) if selected else (14, 22, 42)
                border = STEP_HIGHLIGHT if selected else (50, 72, 112)
                pygame.draw.rect(self._screen, fill, (rx, draw_ry, tile_w, row_h), border_radius=4)
                pygame.draw.rect(self._screen, border, (rx, draw_ry, tile_w, row_h), width=1, border_radius=4)
                txt = f"{b}{ci+1} CC{cc} {lbl}"
                surf = self._fn_small.render(txt, True, TEXT_MAIN if selected else TEXT_SUB)
                self._screen.blit(surf, (rx + 4, draw_ry + max(1, row_h // 2 - 6)))

    def _draw_pad_cheatsheet(self, x: int, y: int, w: int, h: int, kind: str, bank: str, slot: int) -> None:
        rows = [
            ("A", PAD_CHEATSHEET_A),
            ("B", PAD_CHEATSHEET_B),
            ("C", PAD_CHEATSHEET_C),
        ]
        row_gap = 3
        row_h = max(8, int((h - row_gap * 2) / 3))
        gap = 2
        tile_w = max(24, int((w - gap * 15) / 16))

        for ri, (b, labels) in enumerate(rows):
            ry = y + ri * (row_h + row_gap)
            for ci in range(16):
                rx = x + ci * (tile_w + gap)
                selected = (kind == "pad" and bank == b and slot == ci)
                fill = (42, 62, 90) if selected else (13, 19, 34)
                border = STEP_HIGHLIGHT if selected else (42, 60, 96)
                pygame.draw.rect(self._screen, fill, (rx, ry, tile_w, row_h), border_radius=3)
                pygame.draw.rect(self._screen, border, (rx, ry, tile_w, row_h), width=1, border_radius=3)

                if ci == 0:
                    bank_lbl = self._fn_small.render(b, True, (140, 184, 240))
                    self._screen.blit(bank_lbl, (rx + 2, ry + 1))

                label = labels[ci]
                txt = self._fn_small.render(label, True, TEXT_MAIN if selected else TEXT_SUB)
                self._screen.blit(txt, txt.get_rect(center=(rx + tile_w // 2, ry + row_h // 2)))

    def _draw_topology_legend(self, x: int, y: int, w: int, h: int, selected_idx: int) -> None:
        title = self._fn_small.render("Topology Legend", True, TEXT_SUB)
        self._screen.blit(title, (x, y))

        tile_y = y + 14
        tile_h = h - 22
        gap = 6
        n = len(TOPOLOGY_NAMES)
        tile_w = max(40, int((w - gap * (n - 1)) / n))

        for i, name in enumerate(TOPOLOGY_NAMES):
            tx = x + i * (tile_w + gap)
            selected = (i == selected_idx)
            fill = (24, 36, 64) if selected else (16, 22, 40)
            border = STEP_HIGHLIGHT if selected else (48, 70, 110)
            pygame.draw.rect(self._screen, fill, (tx, tile_y, tile_w, tile_h), border_radius=6)
            pygame.draw.rect(self._screen, border, (tx, tile_y, tile_w, tile_h), width=1, border_radius=6)

            icon_rect = pygame.Rect(tx + 6, tile_y + 6, tile_w - 12, max(10, tile_h - 28))
            self._draw_topology_icon(i, icon_rect, border)

            label = self._fn_small.render(name[:7], True, TEXT_MAIN if selected else TEXT_SUB)
            self._screen.blit(label, label.get_rect(center=(tx + tile_w // 2, tile_y + tile_h - 10)))

    def _draw_topology_icon(self, kind: int, rect: pygame.Rect, color: tuple[int, int, int]) -> None:
        cx, cy = rect.centerx, rect.centery
        rw, rh = rect.width, rect.height

        def node(px: int, py: int) -> None:
            pygame.draw.circle(self._screen, color, (px, py), 2)

        if kind == 0:  # Ring
            pts = [
                (cx - rw // 4, cy),
                (cx - rw // 8, cy - rh // 3),
                (cx + rw // 8, cy - rh // 3),
                (cx + rw // 4, cy),
                (cx + rw // 8, cy + rh // 3),
                (cx - rw // 8, cy + rh // 3),
            ]
            for i in range(len(pts)):
                pygame.draw.line(self._screen, color, pts[i], pts[(i + 1) % len(pts)], 1)
                node(*pts[i])

        elif kind == 1:  # FullyConnected
            pts = [
                (cx - rw // 4, cy - rh // 3),
                (cx + rw // 4, cy - rh // 3),
                (cx - rw // 4, cy + rh // 3),
                (cx + rw // 4, cy + rh // 3),
            ]
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    pygame.draw.line(self._screen, color, pts[i], pts[j], 1)
            for p in pts:
                node(*p)

        elif kind == 2:  # Feedforward
            cols = [cx - rw // 4, cx, cx + rw // 4]
            ys = [cy - rh // 4, cy + rh // 4]
            for c0, c1 in zip(cols[:-1], cols[1:]):
                for y0 in ys:
                    for y1 in ys:
                        pygame.draw.line(self._screen, color, (c0, y0), (c1, y1), 1)
            for c in cols:
                for yy in ys:
                    node(c, yy)

        elif kind == 3:  # SparseRandom
            pts = [
                (cx - rw // 4, cy - rh // 4),
                (cx - rw // 8, cy + rh // 5),
                (cx + rw // 10, cy - rh // 6),
                (cx + rw // 4, cy + rh // 6),
                (cx + rw // 6, cy - rh // 3),
            ]
            edges = [(0, 1), (1, 2), (2, 3), (0, 4)]
            for a, b in edges:
                pygame.draw.line(self._screen, color, pts[a], pts[b], 1)
            for p in pts:
                node(*p)

        else:  # SmallWorld
            pts = [
                (cx - rw // 4, cy),
                (cx - rw // 8, cy - rh // 3),
                (cx + rw // 8, cy - rh // 3),
                (cx + rw // 4, cy),
                (cx + rw // 8, cy + rh // 3),
                (cx - rw // 8, cy + rh // 3),
            ]
            for i in range(len(pts)):
                pygame.draw.line(self._screen, color, pts[i], pts[(i + 1) % len(pts)], 1)
            pygame.draw.line(self._screen, color, pts[0], pts[3], 1)
            pygame.draw.line(self._screen, color, pts[1], pts[4], 1)
            for p in pts:
                node(*p)

    def handle_events(self, keydown_handler=None) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                handled = False
                if keydown_handler is not None:
                    handled = bool(keydown_handler(event))
                if handled:
                    continue
                if event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and (event.mod & pygame.KMOD_ALT):
                    self._toggle_fullscreen()
                elif event.key == pygame.K_ESCAPE:
                    if self._is_fullscreen:
                        self._toggle_fullscreen()
                    else:
                        return False
            if event.type == pygame.VIDEORESIZE and not self._is_fullscreen:
                resized_w = max(800, int(event.w))
                resized_h = max(620, int(event.h))
                self._windowed_size = (resized_w, resized_h)
                self._screen = pygame.display.set_mode(self._windowed_size, pygame.RESIZABLE)
        return True

    def tick(self) -> None:
        self._clock.tick(FPS)
