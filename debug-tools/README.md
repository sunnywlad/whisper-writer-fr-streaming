# Debug tools

Standalone scripts for diagnosing the microphone, kept outside the app.

## mic_probe.py

Opens its own `sounddevice` stream beside WhisperWriter and logs what the
microphone really delivers, to tell a stalled stream from a dead signal.

    venv/bin/python debug-tools/mic_probe.py ["Default Source"]

| Line | Meaning |
|---|---|
| `PROBE STALL` / `PROBE RESUMED` | No callback for over 1 s: the stream stopped. |
| `DEAD SIGNAL (DC offset)` | Frames arrive but hold a constant offset (|mean| > 2000). The codec's capture path is off, for example `Internal Mic Switch`. |
| `DEAD SIGNAL (digital silence)` | Frames arrive but are all zeros. |
| `heartbeat` | Mean, std and level of the last second, every 30 s. |

A healthy quiet room reads a mean near 0 and a std of a few dozen; speech reads
a std in the thousands. When the signal is dead, check `amixer -c 0 cget
numid=38` and turn it on with `amixer -c 0 cset numid=38 on`.

## Findings so far

- **Only "A" typed, nothing at exit.** The microphone delivered a constant DC
  offset, not audio. The VAD read it as one click and the decoder returned
  " A". Cause: the codec's `Internal Mic Switch` was off. The `Decoded` log line
  shows it as a `level` near -9 to -21 dBFS on a 0.81 s block, with no
  `Audio stall`.
- **Audio stalling mid-session.** The PortAudio callback goes silent for
  seconds. `streaming_asr.py` logs `Audio stall` / `Audio resumed` and reopens
  the stream up to 3 times. It is a mitigation, not a fix.
- **Hang after suspend.** `Stopping capture` is logged with no `Microphone
  closed` after it. Still under observation.
