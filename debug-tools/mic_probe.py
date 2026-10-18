"""Independent microphone probe, run beside the app on the same device as config.yaml.

Logs two kinds of failure, so a journal line can be compared with the
`Audio stall` and `Decoded ... level` lines of WhisperWriter:

- STALL: no callback for longer than GAP_SECONDS (the stream stopped).
- DEAD SIGNAL: frames keep arriving but carry no voice, either a constant DC
  offset (|mean| above DC_LIMIT, typical of a muted codec switch) or digital
  silence (std below SILENCE_STD). A quiet room is neither: it keeps a std of
  a few dozen int16 units around a mean of about zero.

A heartbeat line with mean, std and level is printed every HEARTBEAT_SECONDS.
"""
import sys
import threading
import time
from datetime import datetime

import numpy as np
import sounddevice as sd

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "Default Source"
SAMPLE_RATE = 16000
BLOCK = 480
GAP_SECONDS = 1.0
WINDOW_SECONDS = 1.0
HEARTBEAT_SECONDS = 30
DC_LIMIT = 2000  # int16 units, about -24 dBFS of pure offset
SILENCE_STD = 1.0

lock = threading.Lock()
window = []
last = time.monotonic()


def log(message):
    print(f"{datetime.now():%H:%M:%S} {message}", flush=True)


def callback(indata, frames, time_info, status):
    global last
    last = time.monotonic()
    if status:
        log(f"status: {status}")
    with lock:
        window.append(indata[:, 0].copy())


def classify(mean, std):
    if abs(mean) > DC_LIMIT:
        return "DEAD SIGNAL (DC offset)"
    if std < SILENCE_STD:
        return "DEAD SIGNAL (digital silence)"
    return "ok"


with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                    blocksize=BLOCK, device=DEVICE, callback=callback):
    log(f"probe open on {DEVICE!r}")
    stalled = False
    state = "ok"
    window_start = time.monotonic()
    last_heartbeat = window_start
    while True:
        time.sleep(0.25)
        now = time.monotonic()

        gap = now - last
        if gap > GAP_SECONDS and not stalled:
            stalled = True
            log(f"PROBE STALL: no frames for {gap:.1f}s")
        elif gap <= GAP_SECONDS and stalled:
            stalled = False
            log("PROBE RESUMED")

        if now - window_start < WINDOW_SECONDS:
            continue
        with lock:
            samples = np.concatenate(window).astype(np.float64) if window else None
            window.clear()
        window_start = now
        if samples is None:
            continue

        mean, std = float(samples.mean()), float(samples.std())
        level = 20 * np.log10(max(np.sqrt(np.mean(samples ** 2)), 1.0) / 32768.0)
        new_state = classify(mean, std)
        if new_state != state:
            state = new_state
            log(f"{state}: mean {mean:.0f}, std {std:.1f}, level {level:.0f} dBFS")
        elif now - last_heartbeat >= HEARTBEAT_SECONDS:
            log(f"heartbeat: mean {mean:.0f}, std {std:.1f}, level {level:.0f} dBFS")
        if now - last_heartbeat >= HEARTBEAT_SECONDS:
            last_heartbeat = now
