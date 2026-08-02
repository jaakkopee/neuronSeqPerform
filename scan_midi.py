"""
MPD218 MIDI scanner  –  move every knob, hit every pad, press Ctrl-C when done.
Detects CC messages and Note Ons (with frequencies).
Output is written to midioutput.txt in the current directory.
"""

import mido
import sys
import math
from collections import defaultdict
from datetime import datetime

OUTPUT_FILE = "midioutput.txt"
PORT_HINT   = "MPD218"

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def midi_to_freq(note):
    return 440.0 * (2.0 ** ((note - 69) / 12.0))

def midi_to_name(note):
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"

# ── find port ─────────────────────────────────────────────────────────────────
available = mido.get_input_names()
print(f"Available MIDI ports: {available}")

port_name = next((p for p in available if PORT_HINT.lower() in p.lower()), None)
if port_name is None:
    if not available:
        sys.exit("No MIDI ports found.")
    port_name = available[0]
    print(f"'{PORT_HINT}' not found – using: {port_name}")
else:
    print(f"Using: {port_name}")

print("\nMove every knob and hit every pad.  Press Ctrl-C when done.\n")

# ── capture ───────────────────────────────────────────────────────────────────
log        = []
cc_seen    = defaultdict(set)   # cc  -> set of values
note_seen  = defaultdict(set)   # note -> set of velocities

def handle(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    extra = ""
    if msg.type == "note_on" and msg.velocity > 0:
        extra = f"  [{midi_to_name(msg.note)}  {midi_to_freq(msg.note):.2f} Hz]"
    line = f"{ts}  {msg}{extra}"
    print(line)
    log.append(line)
    if msg.type == "control_change":
        cc_seen[msg.control].add(msg.value)
    elif msg.type == "note_on" and msg.velocity > 0:
        note_seen[msg.note].add(msg.velocity)

try:
    with mido.open_input(port_name) as port:
        for msg in port:
            handle(msg)
except KeyboardInterrupt:
    pass

# ── write output ──────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w") as f:
    f.write(f"MPD218 MIDI scan  –  {datetime.now()}\n")
    f.write(f"Port: {port_name}\n")
    f.write("=" * 60 + "\n\n")

    f.write("=== ALL MESSAGES ===\n")
    for line in log:
        f.write(line + "\n")

    f.write("\n=== CC SUMMARY (cc: min..max) ===\n")
    for cc in sorted(cc_seen):
        vals = cc_seen[cc]
        f.write(f"  CC {cc:3d}  range {min(vals):3d} – {max(vals):3d}"
                f"  ({len(vals)} distinct values)\n")

    f.write("\n=== NOTE ON SUMMARY (note: name  freq  vel range) ===\n")
    for note in sorted(note_seen):
        vels = note_seen[note]
        freq = midi_to_freq(note)
        name = midi_to_name(note)
        f.write(f"  Note {note:3d}  {name:4s}  {freq:8.2f} Hz"
                f"  vel {min(vels):3d} – {max(vels):3d}"
                f"  ({len(vels)} hits)\n")

print(f"\nWritten to {OUTPUT_FILE}"
      f"  ({len(log)} messages, {len(cc_seen)} CCs, {len(note_seen)} notes found)")
