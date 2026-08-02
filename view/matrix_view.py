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
        cfg   = self.config_state
        y     = MARGIN_TOP + ROWS * CELL_H + 8
        bw    = (SCREEN_WIDTH - 2 * MARGIN_LEFT) // 6   # block width

        # Parameter tables: (cc_label, short_name, config_key, lo, hi)
        bank_a = [
            ("CC3",  "Tempo",   "tempo",                 40.0,  200.0),
            ("CC9",  "Volume",  "master_volume",          0.0,    1.0),
            ("CC12", "Pairs",   "active_pairs",           1.0,    4.0),
            ("CC13", "Decay",   "decay_speed",            0.0,    1.0),
            ("CC14", "FMmod",   "mod_index_scale",        0.0,    3.0),
            ("CC15", "Quant",   "quantization_strength",  0.0,    1.0),
        ]
        bank_b = [
            ("CC16", "Thresh",  "threshold",    0.3,  2.0),
            ("CC17", "Tau",     "tau",          5.0, 100.0),
            ("CC18", "Wt.Sc",   "weight_scale", 0.0,  3.0),
            ("CC19", "Drive",   "drive_n",      0.0,  1.0),
            ("CC20", "Swing",   "swing_amount", 0.0,  0.5),
            ("CC21", "Ratio",   "ratio_scale",  0.5,  2.0),
        ]

        self._draw_bank_header("BANK A  performance", y, CARRIER_C)
        y += 14
        for i, (cc, name, key, lo, hi) in enumerate(bank_a):
            val  = cfg.get(key, lo)
            norm = max(0.0, min(1.0, (float(val) - lo) / max(hi - lo, 1e-9)))
            self._draw_param_block(MARGIN_LEFT + i * bw, y, bw - 3, 40,
                                   cc, name, self._fmt(key, val), norm, CARRIER_C)

        y += 45
        self._draw_bank_header("BANK B  network", y, MODULATOR_C)
        y += 14
        for i, (cc, name, key, lo, hi) in enumerate(bank_b):
            val  = cfg.get(key, lo)
            norm = max(0.0, min(1.0, (float(val) - lo) / max(hi - lo, 1e-9)))
            self._draw_param_block(MARGIN_LEFT + i * bw, y, bw - 3, 40,
                                   cc, name, self._fmt(key, val), norm, MODULATOR_C)

        y += 45
        # Status line
        root      = cfg.get("root_note", 60)
        note_name = _NOTE_NAMES[root % 12]
        scale     = cfg.get("scale_name", "major")
        at_tgt    = cfg.get("aftertouch_target", "threshold")
        step      = self.current_step + 1
        lif_steps = cfg.get("lif_steps", 12)
        status    = (f"Step {step:2d}/{COLS}    "
                     f"Scale: {note_name} {scale}    "
                     f"LIF {lif_steps} steps/tick    "
                     f"CC22: AT \u2192 {at_tgt}")
        surf = self._fn_medium.render(status, True, TEXT_DIM)
        self._screen.blit(surf, (MARGIN_LEFT, y))

    # ── helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _fmt(key: str, val) -> str:
        if key == "tempo":          return f"{val:.0f}BPM"
        if key == "master_volume":  return f"{val:.2f}"
        if key == "active_pairs":   return f"{int(val)}/4"
        if key == "decay_speed":    return f"{val:.2f}"
        if key == "mod_index_scale":return f"{val:.2f}"
        if key == "quantization_strength": return f"{val:.2f}"
        if key == "threshold":      return f"{val:.2f}"
        if key == "tau":            return f"{val:.0f}ms"
        if key == "weight_scale":   return f"{val:.2f}"
        if key == "drive_n":        return f"{val:.2f}"
        if key == "swing_amount":   return f"{val:.2f}"
        if key == "ratio_scale":    return f"{val:.2f}"
        return str(val)

    def _draw_bank_header(self, text: str, y: int, color) -> None:
        surf = self._fn_small.render(text, True, color)
        self._screen.blit(surf, (MARGIN_LEFT, y))
        x1 = MARGIN_LEFT + surf.get_width() + 8
        x2 = SCREEN_WIDTH - MARGIN_LEFT
        pygame.draw.line(self._screen, (38, 42, 62), (x1, y + 5), (x2, y + 5))

    def _draw_param_block(self, x: int, y: int, w: int, h: int,
                          cc_lbl: str, name: str, val_str: str,
                          norm: float, color) -> None:
        # Background
        pygame.draw.rect(self._screen, (16, 18, 30), (x, y, w, h), border_radius=3)
        # CC label (top-left, tiny dim)
        s = self._fn_small.render(cc_lbl, True, (60, 68, 100))
        self._screen.blit(s, (x + 3, y + 2))
        # Parameter name
        s = self._fn_medium.render(name, True, color)
        self._screen.blit(s, (x + 3, y + 13))
        # Value (right-aligned, same row as name)
        s = self._fn_small.render(val_str, True, TEXT_BRIGHT)
        self._screen.blit(s, (x + w - s.get_width() - 3, y + 15))
        # Bar background
        by = y + h - 7
        pygame.draw.rect(self._screen, (38, 42, 60), (x + 3, by, w - 6, 4), border_radius=2)
        # Bar fill
        fw = max(0, int((w - 6) * norm))
        if fw > 0:
            pygame.draw.rect(self._screen, color, (x + 3, by, fw, 4), border_radius=2)

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
