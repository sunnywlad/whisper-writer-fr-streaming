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
