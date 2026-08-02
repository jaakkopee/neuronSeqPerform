"""
MPD218 CC scanner  –  move every knob/pad on your controller, then press Ctrl-C.
Output is written to midioutput.txt in the current directory.
"""

import mido
import signal
import sys
from collections import defaultdict
from datetime import datetime

OUTPUT_FILE = "midioutput.txt"
PORT_HINT   = "MPD218"

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

print("\nMove every knob and pad on the MPD218.  Press Ctrl-C when done.\n")

# ── capture ───────────────────────────────────────────────────────────────────
log   = []          # (msg_type, channel, cc_or_note, value)
seen  = defaultdict(set)   # cc -> set of values seen

def handle(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"{ts}  {msg}"
    print(line)
    log.append(line)
    if msg.type == "control_change":
        seen[msg.control].add(msg.value)

try:
    with mido.open_input(port_name) as port:
        for msg in port:
            handle(msg)
except KeyboardInterrupt:
    pass

# ── write output ──────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, "w") as f:
    f.write(f"MPD218 CC scan  –  {datetime.now()}\n")
    f.write(f"Port: {port_name}\n")
    f.write("=" * 60 + "\n\n")

    f.write("=== ALL MESSAGES ===\n")
    for line in log:
        f.write(line + "\n")

    f.write("\n=== CC SUMMARY (cc: min..max) ===\n")
    for cc in sorted(seen):
        vals = seen[cc]
        f.write(f"  CC {cc:3d}  range {min(vals):3d} – {max(vals):3d}"
                f"  ({len(vals)} distinct values)\n")

print(f"\nWritten to {OUTPUT_FILE}  ({len(log)} messages, {len(seen)} CCs found)")
