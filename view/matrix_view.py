"""Dynamic neuron-grid renderer with filled, colorful rectangular cells."""

from __future__ import annotations

import colorsys
import numpy as np
import pygame

from config import SCREEN_WIDTH, SCREEN_HEIGHT, FPS


BG_TOP = (10, 14, 28)
BG_BOT = (6, 9, 18)
PANEL_BG = (12, 18, 36)
PANEL_BORDER = (38, 56, 98)
TEXT_MAIN = (230, 238, 255)
TEXT_SUB = (140, 164, 210)
STEP_HIGHLIGHT = (250, 220, 120)

TOPOLOGY_NAMES = ["Ring", "FullyConnected", "Feedforward", "SparseRandom", "SmallWorld"]


def _hsv_to_rgb255(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


class MatrixView:
    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
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

    def update(self, spikes, potentials, frequencies, current_step, config_state):
        self.spikes = np.asarray(spikes, dtype=bool)
        self.potentials = np.asarray(potentials, dtype=np.float32)
        self.frequencies = np.asarray(frequencies, dtype=np.float32)
        self.current_step = int(current_step)
        self.config_state = config_state

        if self._fade.shape != self.spikes.shape:
            self._fade = np.zeros_like(self.potentials, dtype=np.float32)
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
        h = SCREEN_HEIGHT
        for y in range(h):
            t = y / max(1, h - 1)
            c = (
                int(BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * t),
                int(BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * t),
                int(BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * t),
            )
            pygame.draw.line(self._screen, c, (0, y), (SCREEN_WIDTH, y))

    def _draw_title(self) -> None:
        title = self._fn_title.render("NeuronSeqPerform  |  LIF Metal Grid", True, TEXT_MAIN)
        self._screen.blit(title, (20, 12))

    def _draw_grid(self) -> None:
        rows, cols = self.spikes.shape

        x = 20
        y = 48
        w = SCREEN_WIDTH - 40
        h = int(SCREEN_HEIGHT * 0.66)

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

        # Thin lattice lines make dense topologies readable even with 4k neurons.
        line_c = (24, 30, 52)
        for c in range(1, cols):
            lx = x + int(c * cw)
            pygame.draw.line(self._screen, line_c, (lx, y), (lx, y + h), 1)
        for r in range(1, rows):
            ly = y + int(r * ch)
            pygame.draw.line(self._screen, line_c, (x, ly), (x + w, ly), 1)

        pygame.draw.rect(self._screen, STEP_HIGHLIGHT, (hx, y + h - 5, max(2, int(cw)), 4), border_radius=2)

    def _draw_status_panel(self) -> None:
        y = int(SCREEN_HEIGHT * 0.74)
        x = 20
        w = SCREEN_WIDTH - 40
        h = SCREEN_HEIGHT - y - 18

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
            f"Drive: {cfg.get('drive_n', 0.5):.2f}"
        )
        line3 = (
            f"Scale: {cfg.get('scale_name', 'major')}   "
            f"Root: {cfg.get('root_note', 60)}   "
            f"LIF steps/tick: {cfg.get('lif_steps', 12)}   "
            f"AT target: {cfg.get('aftertouch_target', 'threshold')}"
        )

        self._screen.blit(self._fn_medium.render(line1, True, TEXT_MAIN), (x + 12, y + 12))
        self._screen.blit(self._fn_small.render(line2, True, TEXT_SUB), (x + 12, y + 38))
        self._screen.blit(self._fn_small.render(line3, True, TEXT_SUB), (x + 12, y + 58))

        top_idx = int(cfg.get("topology_index", 0))
        self._draw_topology_legend(x + 12, y + 84, w - 24, 92, top_idx)

        hints = "Bank C knobs: CC23 topology  CC24 neurons  CC25 lif-steps  | Pads 74-77 mirror these"
        self._screen.blit(self._fn_small.render(hints, True, (116, 184, 232)), (x + 12, y + h - 22))

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

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def tick(self) -> None:
        self._clock.tick(FPS)
