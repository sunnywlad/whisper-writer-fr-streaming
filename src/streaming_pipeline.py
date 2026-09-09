"""Gapless producer/consumer dictation pipeline.

The legacy :class:`result_thread.ResultThread` is serial: it records one speech
segment, closes the microphone, transcribes, types, then reopens the microphone.
Everything spoken while it transcribes and types is lost.

This module keeps a single :class:`sounddevice.InputStream` open for the whole
dictation session:

* a **producer** thread reads 30 ms frames, runs WebRTC VAD exactly as the
  legacy path did, and on every speech -> silence boundary snapshots the
  accumulated audio as a segment, pushes it onto a FIFO and keeps reading with
  no gap;
* a **consumer** thread (this ``QThread``) pops segments in order, transcribes
  them one at a time, and emits ``resultSignal`` so the main thread does the
  typing.

A single consumer is what guarantees the typed text comes out in the order it
was spoken. The queue is unbounded on purpose: if transcription falls behind,
latency grows but no audio is ever dropped.
"""

import queue
import threading
import time
import traceback
from collections import deque

import numpy as np
import sounddevice as sd
import webrtcvad
from PyQt5.QtCore import QThread, QMutex, pyqtSignal

from transcription import transcribe
from utils import ConfigManager

# WebRTC VAD only accepts 10, 20 or 30 ms frames. The legacy path used 30 ms.
FRAME_DURATION_MS = 30

# Frames skipped once at session start so the activation key click is not
# mistaken for speech. Unlike the legacy path this happens per session, not per
# segment, because the stream is never reopened.
KEY_CLICK_SKIP_SECONDS = 0.15

# Frames of silence kept as lead-in when a buffer holding no speech at all is
# dropped, so the first syllable of the next word is not clipped.
SILENCE_PREROLL_FRAMES = 10

# Pushed by the producer when it exits, so the consumer knows the session is
# over only after it has drained every real segment ahead of it in the FIFO.
_END_OF_STREAM = object()


class StreamingResultThread(QThread):
    """Consumer thread that owns a persistent-microphone producer thread.

    Signals:
        statusSignal: 'recording', 'transcribing', 'idle' or 'error'.
        resultSignal: one transcribed segment, emitted in capture order.
    """

    statusSignal = pyqtSignal(str)
    resultSignal = pyqtSignal(str)

    def __init__(self, local_model=None):
        """
        :param local_model: Local transcription model (if applicable)
        """
        super().__init__()
        self.local_model = local_model
        self.sample_rate = None

        self._segments = queue.Queue()
        self._stop_producer = threading.Event()
        self._abort = threading.Event()
        self._producer = None

        # Guarded by self.mutex.
        self.mutex = QMutex()
        self._pending_samples = 0
        self._backlog_warned = False
        self._last_status = None

    # ------------------------------------------------------------------
    # Public API (mirrors ResultThread so main.py can treat both alike)
    # ------------------------------------------------------------------

    def stop(self):
        """Stop capturing, then let the consumer drain the queue.

        Non-blocking on purpose: the caller is the UI thread, and the queued
        results still have to be typed by that very thread while the consumer
        finishes. Blocking here would freeze the typing until the drain ended.
        """
        if self._stop_producer.is_set():
            ConfigManager.console_print('Already stopping; draining the backlog.')
            return
        ConfigManager.console_print('Stopping capture. Draining the transcription backlog...')
        self._stop_producer.set()

    def stop_recording(self):
        """Alias of :meth:`stop`, kept for API parity with ResultThread."""
        self.stop()

    def abort(self, wait_ms=5000):
        """Hard shutdown for application exit: drop the backlog and join."""
        self._abort.set()
        self._stop_producer.set()
        self._segments.put(_END_OF_STREAM)
        if self.isRunning():
            if not self.wait(wait_ms):
                ConfigManager.console_print('Streaming pipeline did not stop in time.')

    def is_stopping(self):
        """True once a stop has been requested and the queue is draining."""
        return self._stop_producer.is_set()

    def pending_seconds(self):
        """Seconds of captured audio still waiting to be transcribed."""
        self.mutex.lock()
        try:
            return self._pending_samples / float(self.sample_rate or 16000)
        finally:
            self.mutex.unlock()

    # ------------------------------------------------------------------
    # Consumer side (runs in this QThread)
    # ------------------------------------------------------------------

    def run(self):
        """Start the producer, then transcribe its segments in order."""
        try:
            recording_options = ConfigManager.get_config_section('recording_options')
            self.sample_rate = int(recording_options.get('sample_rate') or 16000)

            self._producer = threading.Thread(target=self._produce,
                                              name='ww-audio-producer',
                                              daemon=True)
            self._producer.start()
            self._emit_status('recording')
            self._consume()
        except Exception:
            traceback.print_exc()
            self._emit_status('error')
        finally:
            self._stop_producer.set()
            if self._producer is not None:
                self._producer.join(timeout=2.0)
                if self._producer.is_alive():
                    ConfigManager.console_print('Audio producer did not stop within 2s.')
            self._emit_status('idle')

    def _consume(self):
        """Pop segments FIFO and transcribe them one at a time."""
        while True:
            try:
                segment = self._segments.get(timeout=0.1)
            except queue.Empty:
                if self._abort.is_set():
                    break
                # Safety net: the producer died without pushing the sentinel.
                if (self._stop_producer.is_set() and self._producer is not None
                        and not self._producer.is_alive()):
                    break
                continue

            if segment is _END_OF_STREAM:
                break

            pending_seconds, warning = self._update_backlog(-len(segment))
            if warning:
                ConfigManager.console_print(warning)

            if self._abort.is_set():
                break

            self._emit_status('transcribing')
            audio_seconds = len(segment) / float(self.sample_rate)
            start_time = time.time()
            try:
                result = transcribe(segment, self.local_model)
            except Exception:
                traceback.print_exc()
                self._emit_status('error')
                continue
            elapsed = time.time() - start_time
            ConfigManager.console_print(
                f'Segment of {audio_seconds:.2f}s transcribed in {elapsed:.2f}s '
                f'({pending_seconds:.1f}s still pending). Post-processed line: {result}'
            )

            if self._abort.is_set():
                break

            if result and result.strip():
                self.resultSignal.emit(result)

            if not self._stop_producer.is_set():
                self._emit_status('recording')

    # ------------------------------------------------------------------
    # Producer side (runs in a plain daemon thread)
    # ------------------------------------------------------------------

    def _produce(self):
        """Hold one InputStream open and cut it into segments at every pause."""
        recording_options = ConfigManager.get_config_section('recording_options')
        sample_rate = self.sample_rate
        frame_size = int(sample_rate * (FRAME_DURATION_MS / 1000.0))

        silence_duration_ms = recording_options.get('silence_duration') or 900
        silence_frames = max(1, int(silence_duration_ms / FRAME_DURATION_MS))
        min_duration_ms = recording_options.get('min_duration') or 100

        max_segment_seconds = recording_options.get('max_segment_duration') or 0
        max_segment_frames = int(max_segment_seconds * 1000 / FRAME_DURATION_MS) if max_segment_seconds else 0
        # A buffer holding no speech is trimmed even when the safety cut is
        # disabled, so an idle session cannot grow it without bound.
        silence_cap_frames = max_segment_frames or int(60 * 1000 / FRAME_DURATION_MS)

        frames_to_skip = int(KEY_CLICK_SKIP_SECONDS * sample_rate / frame_size)

        vad = webrtcvad.Vad(2)  # VAD aggressiveness: 0 to 3, 3 being the most aggressive
        speech_detected = False
        silent_frame_count = 0
        segment_frames = []

        # Unbounded: the producer loop is the only consumer of this buffer and
        # does almost no work per frame, so it never falls behind the callback.
        sample_buffer = deque()
        data_ready = threading.Event()

        def audio_callback(indata, frames, time_info, status):
            if status:
                ConfigManager.console_print(f'Audio callback status: {status}')
            sample_buffer.extend(indata[:, 0].tolist())
            data_ready.set()

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16',
                                blocksize=frame_size,
                                device=recording_options.get('sound_device'),
                                callback=audio_callback):
                ConfigManager.console_print('Microphone open. Recording continuously...')

                while not self._stop_producer.is_set():
                    if not data_ready.wait(timeout=0.1):
                        continue
                    data_ready.clear()

                    while len(sample_buffer) >= frame_size:
                        frame = np.array([sample_buffer.popleft() for _ in range(frame_size)],
                                         dtype=np.int16)
                        segment_frames.append(frame)

                        # Do not run VAD on the key-click frames, but keep them.
                        if frames_to_skip > 0:
                            frames_to_skip -= 1
                            continue

                        if vad.is_speech(frame.tobytes(), sample_rate):
                            silent_frame_count = 0
                            if not speech_detected:
                                ConfigManager.console_print('Speech detected.')
                                speech_detected = True
                        else:
                            silent_frame_count += 1

                        at_pause = speech_detected and silent_frame_count > silence_frames
                        too_long = bool(max_segment_frames) and len(segment_frames) >= max_segment_frames

                        if at_pause or (too_long and speech_detected):
                            if too_long and not at_pause:
                                ConfigManager.console_print(
                                    f'No pause after {max_segment_seconds}s; cutting the segment.'
                                )
                            self._push_segment(segment_frames, sample_rate, min_duration_ms)
                            segment_frames = []
                            speech_detected = False
                            silent_frame_count = 0
                        elif not speech_detected and len(segment_frames) >= silence_cap_frames:
                            # Nothing but silence so far: drop it instead of
                            # sending a silent segment to Whisper, which would
                            # only invite a hallucinated line.
                            segment_frames = segment_frames[-SILENCE_PREROLL_FRAMES:]

                # Flush the tail captured before the stop request, but only if it
                # actually holds speech.
                if speech_detected:
                    self._push_segment(segment_frames, sample_rate, min_duration_ms)
        except Exception:
            traceback.print_exc()
        finally:
            self._segments.put(_END_OF_STREAM)
            ConfigManager.console_print('Microphone closed.')

    def _push_segment(self, frames, sample_rate, min_duration_ms):
        """Queue one segment, unless it is shorter than min_duration."""
        if not frames:
            return

        audio_data = np.concatenate(frames)
        duration_ms = len(audio_data) / float(sample_rate) * 1000.0

        if duration_ms < min_duration_ms:
            ConfigManager.console_print(
                f'Segment discarded: {duration_ms:.0f} ms is under the {min_duration_ms} ms minimum.'
            )
            return

        # Account before queueing so the pending counter is never negative.
        pending_seconds, warning = self._update_backlog(len(audio_data))
        self._segments.put(audio_data)
        ConfigManager.console_print(
            f'Segment queued: {duration_ms / 1000:.2f}s of audio, {pending_seconds:.1f}s pending.'
        )
        if warning:
            ConfigManager.console_print(warning)

    # ------------------------------------------------------------------
    # Shared state helpers
    # ------------------------------------------------------------------

    def _update_backlog(self, delta_samples):
        """Adjust the pending-audio counter; return (seconds, warning or None)."""
        threshold = ConfigManager.get_config_value('recording_options',
                                                   'backlog_warning_seconds') or 15
        warning = None
        self.mutex.lock()
        try:
            self._pending_samples = max(0, self._pending_samples + delta_samples)
            pending_seconds = self._pending_samples / float(self.sample_rate or 16000)
            if pending_seconds > threshold and not self._backlog_warned:
                self._backlog_warned = True
                warning = (f'WARNING: {pending_seconds:.1f}s of speech is waiting to be '
                           f'transcribed (over the {threshold}s threshold). Text will lag '
                           f'behind, but nothing is dropped.')
            elif pending_seconds <= threshold and self._backlog_warned:
                self._backlog_warned = False
        finally:
            self.mutex.unlock()
        return pending_seconds, warning

    def _emit_status(self, status):
        """Emit statusSignal only when the status actually changes."""
        self.mutex.lock()
        try:
            changed = status != self._last_status
            if changed:
                self._last_status = status
        finally:
            self.mutex.unlock()
        if changed:
            self.statusSignal.emit(status)
