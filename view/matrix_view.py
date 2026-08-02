"""
Pygame-based 8 × 16 neuron-matrix visualiser.

Layout
──────
  • 8 rows (operators)  × 16 columns (presets / steps)
  • Each cell contains a circle representing one neuron.
  • Inside the circle:
      – top  line : note name derived from the operator frequency
      – lower line: frequency in Hz / kHz
  • Carrier rows (even) are drawn in teal;  modulator rows (odd) in amber.
  • The currently playing step column is highlighted.
  • Membrane potential is shown as a radial fill inside the circle.
  • A spike causes a brief white-yellow flash that decays smoothly.
  • Bottom panel shows live tempo, quantize, swing, scale and step.
"""

import math
import numpy as np
import pygame

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS,
    ROWS, COLS,
    MARGIN_LEFT, MARGIN_TOP,
    CELL_W, CELL_H, NEURON_RADIUS,
)


# ── colour palette ─────────────────────────────────────────────────────────────
BG          = ( 12,  14,  22)
GRID        = ( 35,  38,  58)
STEP_GLOW   = ( 55,  65, 105)
NEURON_BG   = ( 22,  25,  40)
CARRIER_C   = ( 40, 190, 145)   # teal  – even rows
MODULATOR_C = (200, 110,  45)   # amber – odd rows
SPIKE_C     = (255, 250, 130)
TEXT_BRIGHT = (230, 235, 255)
TEXT_DIM    = (110, 120, 150)
OUTLINE_DEF = ( 50,  55,  80)
TITLE_C     = (180, 210, 255)
INFO_C      = (140, 155, 190)


# ── helpers ────────────────────────────────────────────────────────────────────
_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def _freq_to_note(freq: float) -> str:
    if freq <= 0:
        return "---"
    midi   = 12.0 * math.log2(freq / 440.0) + 69.0
    n      = round(midi)
    name   = _NOTE_NAMES[n % 12]
    octave = n // 12 - 1
    return f"{name}{octave}"

def _freq_label(freq: float) -> str:
    if freq <= 0:
        return ""
    return f"{freq:.0f}Hz" if freq < 1000 else f"{freq/1000:.2f}k"

def _lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ── main class ─────────────────────────────────────────────────────────────────
class MatrixView:
    def __init__(self):
        pygame.init()
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("NeuronSeqPerform")
        self._clock  = pygame.time.Clock()

        # fonts
        self._fn_small  = pygame.font.SysFont("monospace", 10)
        self._fn_medium = pygame.font.SysFont("monospace", 12)
        self._fn_large  = pygame.font.SysFont("monospace", 14, bold=True)
        self._fn_title  = pygame.font.SysFont("monospace", 19, bold=True)

        # Precompute cell centres
        self._cx = np.array(
            [MARGIN_LEFT + c * CELL_W + CELL_W // 2 for c in range(COLS)], int)
        self._cy = np.array(
            [MARGIN_TOP  + r * CELL_H + CELL_H // 2 for r in range(ROWS)], int)

        # Display state (updated each sequencer tick)
        self.spikes       = np.zeros((ROWS, COLS), bool)
        self.potentials   = np.zeros((ROWS, COLS), np.float32)
        self.frequencies  = np.zeros((ROWS, COLS), np.float32)
        self.current_step = 0
        self.config_state : dict = {}

        # Per-cell spike fade  (decays each draw frame)
        self._fade = np.zeros((ROWS, COLS), np.float32)

        # Surface for the static grid background (rebuilt once)
        self._bg_surf = self._build_bg()

    # ── background surface (drawn once) ────────────────────────────────────────
    def _build_bg(self) -> pygame.Surface:
        surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        surf.fill(BG)

        for r in range(ROWS):
            for c in range(COLS):
                cx, cy = int(self._cx[c]), int(self._cy[r])
                pygame.draw.circle(surf, NEURON_BG,   (cx, cy), NEURON_RADIUS)
                pygame.draw.circle(surf, OUTLINE_DEF, (cx, cy), NEURON_RADIUS, 1)
        return surf

    # ── state update (called from main loop on each sequencer step) ────────────
    def update(self, spikes, potentials, frequencies, current_step, config_state):
        self.spikes       = spikes
        self.potentials   = potentials
        self.frequencies  = frequencies
        self.current_step = current_step
        self.config_state = config_state
        # Inject spike energy into fade
        self._fade = np.where(spikes, 1.0, self._fade).astype(np.float32)

    # ── draw one frame ──────────────────────────────────────────────────────────
    def draw(self) -> None:
        self._screen.blit(self._bg_surf, (0, 0))

        self._draw_step_column()
        self._draw_neurons()
        self._draw_column_labels()
        self._draw_row_labels()
        self._draw_title()
        self._draw_info_panel()

        pygame.display.flip()

        # Decay fade after drawing
        self._fade *= 0.75

    # ── sub-draw routines ───────────────────────────────────────────────────────
    def _draw_step_column(self) -> None:
        c  = self.current_step
        cx = int(self._cx[c])
        x  = cx - CELL_W // 2 + 2
        y  = MARGIN_TOP - 4
        w  = CELL_W - 4
        h  = ROWS * CELL_H + 8
        pygame.draw.rect(self._screen, STEP_GLOW, (x, y, w, h), border_radius=6)

    def _draw_neurons(self) -> None:
        for r in range(ROWS):
            for c in range(COLS):
                self._draw_neuron(r, c)

    def _draw_neuron(self, r: int, c: int) -> None:
        cx     = int(self._cx[c])
        cy     = int(self._cy[r])
        freq   = float(self.frequencies[r, c])
        pot    = float(self.potentials[r, c])
        fade   = float(self._fade[r, c])

        base_c = CARRIER_C if r % 2 == 0 else MODULATOR_C
        fill_c = _lerp_color(base_c, SPIKE_C, fade)

        # Filled arc representing membrane potential (radius scaled by potential)
        inner_r = max(1, int(NEURON_RADIUS * pot))
        if inner_r > 0:
            pygame.draw.circle(self._screen, fill_c, (cx, cy), inner_r)

        # Outer ring flashes white on spike
        ring_c = SPIKE_C if self.spikes[r, c] else OUTLINE_DEF
        pygame.draw.circle(self._screen, ring_c, (cx, cy), NEURON_RADIUS, 2)

        # Frequency text centred inside the circle
        note_s = _freq_to_note(freq)
        hz_s   = _freq_label(freq)

        surf_note = self._fn_small.render(note_s, True, TEXT_BRIGHT)
        surf_hz   = self._fn_small.render(hz_s,   True, TEXT_DIM)

        self._screen.blit(surf_note,
                          surf_note.get_rect(center=(cx, cy - 6)))
        self._screen.blit(surf_hz,
                          surf_hz.get_rect(center=(cx, cy + 7)))

    def _draw_column_labels(self) -> None:
        for c in range(COLS):
            lbl  = self._fn_small.render(str(c + 1), True, TEXT_DIM)
            rect = lbl.get_rect(center=(int(self._cx[c]), MARGIN_TOP - 14))
            self._screen.blit(lbl, rect)

    def _draw_row_labels(self) -> None:
        labels = ["C0","M0","C1","M1","C2","M2","C3","M3"]
        for r in range(ROWS):
            color = CARRIER_C if r % 2 == 0 else MODULATOR_C
            lbl   = self._fn_small.render(labels[r], True, color)
            self._screen.blit(lbl, (4, int(self._cy[r]) - 5))

    def _draw_title(self) -> None:
        surf = self._fn_title.render("NeuronSeqPerform", True, TITLE_C)
        self._screen.blit(surf, (MARGIN_LEFT, 10))

    def _draw_info_panel(self) -> None:
        y    = MARGIN_TOP + ROWS * CELL_H + 14
        cfg  = self.config_state

        tempo  = cfg.get("tempo",                  120.0)
        quant  = cfg.get("quantization_strength",  0.85)
        swing  = cfg.get("swing_amount",           0.0)
        scale  = cfg.get("scale_name",             "major")
        root   = cfg.get("root_note",              60)
        at_tgt = cfg.get("aftertouch_target",      "threshold")
        step   = self.current_step + 1

        note_name = _NOTE_NAMES[root % 12]

        segments = [
            f"Tempo {tempo:6.1f} BPM",
            f"Quant {quant:.2f}",
            f"Swing {swing:.2f}",
            f"Scale {note_name} {scale}",
            f"AT → {at_tgt}",
            f"Step {step:2d}/{COLS}",
        ]

        x = MARGIN_LEFT
        dx = (SCREEN_WIDTH - 2 * MARGIN_LEFT) // len(segments)
        for seg in segments:
            surf = self._fn_medium.render(seg, True, INFO_C)
            self._screen.blit(surf, (x, y))
            x += dx

    # ── event / tick ───────────────────────────────────────────────────────────
    def handle_events(self) -> bool:
        """Process pygame events.  Returns False when the window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def tick(self) -> None:
        self._clock.tick(FPS)
