# ── Network ───────────────────────────────────────────────────────────────────
ROWS = 8          # operators / network rows
COLS = 16         # presets   / network columns

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 44100
BUFFER_SIZE   = 512
CHANNELS      = 2

# ── Timing ────────────────────────────────────────────────────────────────────
MASTER_TEMPO            = 120.0   # BPM
QUANTIZATION_STRENGTH   = 0.85    # 0.0 (free) → 1.0 (locked to grid)
SWING_AMOUNT            = 0.0     # 0.0 (straight) → 0.5 (full swing)

# ── Display ───────────────────────────────────────────────────────────────────
SCREEN_WIDTH  = 1280
SCREEN_HEIGHT = 760
FPS           = 60
MARGIN_LEFT   = 40
MARGIN_TOP    = 52
CELL_W        = (SCREEN_WIDTH  - 2 * MARGIN_LEFT) // COLS   # 75
CELL_H        = 70
NEURON_RADIUS = 27

# ── MIDI ──────────────────────────────────────────────────────────────────────
MIDI_CHANNEL   = 0           # 0-indexed
MIDI_PORT_NAME = "MPD218"    # substring match – set to "" to use first port

# ── Scales (semitone offsets from root) ───────────────────────────────────────
SCALES = {
    "chromatic"   : [0,1,2,3,4,5,6,7,8,9,10,11],
    "major"       : [0,2,4,5,7,9,11],
    "minor"       : [0,2,3,5,7,8,10],
    "dorian"      : [0,2,3,5,7,9,10],
    "phrygian"    : [0,1,3,5,7,8,10],
    "lydian"      : [0,2,4,6,7,9,11],
    "mixolydian"  : [0,2,4,5,7,9,10],
    "locrian"     : [0,1,3,5,6,8,10],
    "pentatonic"  : [0,2,4,7,9],
    "blues"       : [0,3,5,6,7,10],
}
SCALE_NAMES = list(SCALES.keys())

# ── FM synth ──────────────────────────────────────────────────────────────────
NUM_OPERATORS = 8
NUM_PRESETS   = 16
