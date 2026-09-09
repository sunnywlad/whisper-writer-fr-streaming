"""Streaming dictation core built on whisper_streaming's LocalAgreement-2.

The first continuous-mode attempt (``streaming_pipeline.py``, now removed) kept
the microphone open but cut the audio into VAD segments and transcribed each one
in isolation. Every cut was final, so a cut landing mid-phrase produced a closed
sentence with a premature period, and greedy decoding on a truncated fragment
could fall into a repetition loop that nothing could revise.

This module keeps the persistent microphone and replaces everything downstream:

* a **producer** thread holds one :class:`sounddevice.InputStream` open for the
  whole session and pushes 30 ms frames onto a FIFO. It runs WebRTC VAD purely
  as a *silence gate*, never as a cut: it declines to forward audio while the
  user has been silent for a while, so the decoder's rolling buffer does not
  fill with silence. Speech, its lead-in and its trailing silence are always
  forwarded intact;
* a **consumer** thread (this ``QThread``) appends that audio to a single
  :class:`OnlineASRProcessor` and calls ``process_iter()``. LocalAgreement-2
  re-decodes the growing buffer and commits only the prefix on which two
  successive decodes agree; the uncertain tail is held back. Nothing is ever
  typed that a later decode could have wanted to revise, so there are no
  premature periods and a transient repetition loop is never committed.

One consumer means the confirmed text comes out in the order it was spoken. The
queue is unbounded on purpose: if the CPU falls behind, latency grows and the
status window says so, but no audio is ever dropped.
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

from transcription import decode_temperatures
from utils import ConfigManager
from vendor.whisper_online import FasterWhisperASR, OnlineASRProcessor

# Whisper is trained on 16 kHz mono audio and whisper_streaming hard-codes that
# rate. The streaming path therefore ignores recording_options.sample_rate.
SAMPLE_RATE = 16000

# WebRTC VAD only accepts 10, 20 or 30 ms frames.
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)

# Audio kept ahead of the first speech frame so no syllable is clipped.
PREROLL_FRAMES = 10  # 300 ms

# Frames skipped once at session start so the activation key click is not
# mistaken for speech. Per session, not per segment: the stream never reopens.
KEY_CLICK_SKIP_SECONDS = 0.15

# When no new audio has arrived for this long, run one more process_iter on the
# unchanged buffer. Two decodes of identical audio agree, so this is what
# confirms the tail of an utterance at a natural pause instead of holding it
# hostage until the user presses the stop key.
SETTLE_IDLE_SECONDS = 0.4

# Pushed by the producer when it exits, after every real frame ahead of it.
_END_OF_STREAM = object()


class InjectedFasterWhisperASR(FasterWhisperASR):
    """whisper_streaming's faster-whisper backend, with our model injected.

    Upstream ``FasterWhisperASR.load_model`` builds its own ``WhisperModel`` and
    hard-codes ``device="cuda", compute_type="float16"``. WhisperWriter already
    holds one CPU int8 model, and 8 GB of RAM has no room for a second, so
    ``load_model`` is overridden to hand back the instance we were given. That
    is the whole point of this subclass; keeping it here rather than editing the
    vendored file leaves that file diffable against upstream.

    ``transcribe`` is overridden too, for the decode options upstream hard-codes:
    a temperature *fallback list* rather than a bare 0.0 (see
    ``transcription.decode_temperatures``), the configured beam size, and the
    user's vocabulary prompt prepended to the processor's rolling context.
    """

    def __init__(self, model, lan, initial_prompt=None):
        # Set before super().__init__, which calls load_model().
        self._injected_model = model
        self._user_prompt = (initial_prompt or '').strip()
        self._decode_options = self._build_decode_options()
        super().__init__(lan=lan)

    @staticmethod
    def _build_decode_options():
        local_options = ConfigManager.get_config_section('model_options', 'local') or {}
        common_options = ConfigManager.get_config_section('model_options', 'common') or {}
        return {
            'beam_size': local_options.get('beam_size') or 5,
            'condition_on_previous_text': bool(
                local_options.get('condition_on_previous_text')),
            # The repetition-loop guard. A scalar temperature leaves faster-whisper
            # no way to notice a degenerate decode and retry it; the fallback list
            # plus the two thresholds below make it re-roll at a higher temperature.
            'temperature': decode_temperatures(common_options.get('temperature')),
            'compression_ratio_threshold': 2.4,   # faster-whisper default
            'log_prob_threshold': -1.0,           # faster-whisper default
        }

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        """Return the already-loaded model instead of building a second one."""
        return self._injected_model

    def transcribe(self, audio, init_prompt=""):
        # init_prompt is whisper_streaming's rolling context: the committed text
        # that has already scrolled out of the audio buffer. The user's
        # vocabulary prompt goes in front of it so proper nouns keep their
        # spelling ("Hermes", "Anthropic", "Solidity") for the whole session.
        prompt = ' '.join(p for p in (self._user_prompt, init_prompt or '') if p).strip()
        segments, _info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=prompt or None,
            word_timestamps=True,  # required: ts_words() reads segment.words
            **self._decode_options,
            **self.transcribe_kargs)
        return list(segments)


class StreamingASRThread(QThread):
    """Consumer thread owning a persistent microphone and one OnlineASRProcessor.

    Signals:
        statusSignal: 'listening', 'listening:<seconds behind>', 'idle', 'error'.
        resultSignal: one block of newly *confirmed* text, in spoken order.
    """

    statusSignal = pyqtSignal(str)
    resultSignal = pyqtSignal(str)

    def __init__(self, local_model=None):
        """
        :param local_model: the already-loaded faster_whisper.WhisperModel.
        """
        super().__init__()
        self.local_model = local_model

        self._audio_queue = queue.Queue()
        self._stop_producer = threading.Event()
        self._abort = threading.Event()
        self._producer = None

        # Guarded by self.mutex.
        self.mutex = QMutex()
        self._pending_samples = 0
        self._backlog_warned = False
        self._last_status = None
        self._emitted_any = False

    # ------------------------------------------------------------------
    # Public API (mirrors ResultThread so main.py can treat both alike)
    # ------------------------------------------------------------------

    def stop(self):
        """Stop capturing, then let the consumer drain what was already heard.

        Non-blocking on purpose: the caller is the UI thread, and it is that
        very thread that has to type the drained text. Blocking here would
        freeze the typing until the drain ended.
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
        self._audio_queue.put(_END_OF_STREAM)
        if self.isRunning():
            if not self.wait(wait_ms):
                ConfigManager.console_print('Streaming ASR did not stop in time.')

    def is_stopping(self):
        """True once a stop has been requested and the backlog is draining."""
        return self._stop_producer.is_set()

    def pending_seconds(self):
        """Seconds of captured audio not yet handed to a finished decode."""
        self.mutex.lock()
        try:
            return self._pending_samples / float(SAMPLE_RATE)
        finally:
            self.mutex.unlock()

    # ------------------------------------------------------------------
    # Consumer side (runs in this QThread)
    # ------------------------------------------------------------------

    def run(self):
        """Build the ASR exactly once, start the producer, then stream."""
        try:
            recording_options = ConfigManager.get_config_section('recording_options')
            configured_rate = int(recording_options.get('sample_rate') or SAMPLE_RATE)
            if configured_rate != SAMPLE_RATE:
                ConfigManager.console_print(
                    f'Streaming mode requires {SAMPLE_RATE} Hz; ignoring the configured '
                    f'{configured_rate} Hz.')

            common_options = ConfigManager.get_config_section('model_options', 'common') or {}
            language = common_options.get('language') or 'auto'
            trimming_seconds = recording_options.get('buffer_trimming_seconds') or 12

            # Exactly one ASR object and one processor per session. The model
            # inside it is the one main.py loaded at startup.
            asr = InjectedFasterWhisperASR(
                model=self.local_model,
                lan=language,
                initial_prompt=common_options.get('initial_prompt'))
            online = OnlineASRProcessor(
                asr,
                tokenizer=None,  # "segment" trimming never touches the tokenizer
                buffer_trimming=('segment', trimming_seconds))
            online.init()
            ConfigManager.console_print(
                f'Streaming ASR ready (language={language}, '
                f'buffer trimming at {trimming_seconds}s).')

            self._producer = threading.Thread(target=self._produce,
                                              name='ww-audio-producer',
                                              daemon=True)
            self._producer.start()
            self._emit_status('listening')
            self._consume(online)
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

    def _consume(self, online):
        """Feed the processor and emit whatever LocalAgreement confirms.

        Cadence: whisper_streaming suggests calling ``process_iter()`` about
        once a second. On a 4-core CPU with no GPU, ``small`` int8 decodes
        slower than real time, so a fixed 1 s tick would only queue decodes
        behind each other. Instead the loop is self-pacing: it takes whatever
        audio piled up while the previous decode ran, subject to a floor of
        ``min_chunk_seconds``. When the machine keeps up, the floor sets the
        cadence; when it does not, each decode simply takes a bigger bite and
        latency grows in place of work being dropped.
        """
        recording_options = ConfigManager.get_config_section('recording_options')
        min_chunk_seconds = float(recording_options.get('min_chunk_seconds') or 2.0)
        min_chunk_samples = int(min_chunk_seconds * SAMPLE_RATE)
        # Upper bound on a single insert. Without it, a session that has fallen
        # a minute behind would hand the decoder that whole minute at once: one
        # colossal decode, no text meanwhile, and a rolling buffer far past the
        # trimming threshold. Biting it keeps each pass near the size the
        # trimming setting was chosen for, and keeps text flowing while draining.
        max_bite_samples = max(
            min_chunk_samples,
            int((recording_options.get('buffer_trimming_seconds') or 12) * SAMPLE_RATE))

        collected = deque()
        collected_samples = 0
        producer_done = False
        settled = True
        last_audio_at = time.time()

        while not self._abort.is_set():
            # --- collect everything already waiting, in one pass -----------
            try:
                item = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                item = None
            batch = [] if item is None else [item]
            while True:
                try:
                    batch.append(self._audio_queue.get_nowait())
                except queue.Empty:
                    break

            for chunk in batch:
                if chunk is _END_OF_STREAM:
                    producer_done = True
                    continue
                collected.append(chunk)
                collected_samples += len(chunk)
                last_audio_at = time.time()
                settled = False

            # --- decide whether this pass decodes --------------------------
            if self._abort.is_set():
                break
            if collected_samples >= min_chunk_samples:
                decode_now = True
            elif producer_done:
                # Draining: decode the tail however short it is, then stop.
                decode_now = collected_samples > 0
            elif not settled and time.time() - last_audio_at >= SETTLE_IDLE_SECONDS:
                # The user paused. Re-decoding the unchanged buffer makes the
                # two hypotheses agree, which confirms the tail of the phrase
                # instead of holding it until the session ends.
                decode_now = True
                settled = True
            else:
                decode_now = False

            if not decode_now:
                if producer_done:
                    break
                self._report_backlog()
                continue

            bite = []
            consumed_samples = 0
            while collected and consumed_samples < max_bite_samples:
                frame = collected.popleft()
                bite.append(frame)
                consumed_samples += len(frame)
            collected_samples -= consumed_samples
            audio = np.concatenate(bite) if bite else np.empty(0, dtype=np.int16)

            # Report before decoding, not only after. The bite about to be
            # decoded is still audio the user has spoken and not yet seen, and a
            # slow decode is exactly when the status window has something worth
            # saying. Reporting only afterwards would show the lag at its
            # lowest, right after it was discounted.
            self._report_backlog()

            if not self._step(online, audio):
                break
            self._settle_backlog(consumed_samples)

            if producer_done and not collected and self._audio_queue.empty():
                break

        if self._abort.is_set():
            ConfigManager.console_print('Streaming ASR aborted; backlog dropped.')
            return

        # Nothing spoken before the stop may be lost: flush the held-back tail.
        try:
            tail = online.finish()
        except Exception:
            traceback.print_exc()
            return
        self._emit_confirmed(tail, final=True)

    def _step(self, online, audio):
        """Insert one block of audio and emit whatever it confirms.

        Returns False if the caller should stop looping.
        """
        if len(audio):
            online.insert_audio_chunk(audio.astype(np.float32) / 32768.0)

        started = time.time()
        try:
            confirmed = online.process_iter()
        except Exception:
            traceback.print_exc()
            self._emit_status('error')
            return True  # a bad decode is not a reason to end the session
        elapsed = time.time() - started

        audio_seconds = len(audio) / float(SAMPLE_RATE)
        ConfigManager.console_print(
            f'Decoded {audio_seconds:.2f}s of new audio in {elapsed:.2f}s; '
            f'confirmed: {confirmed[2]!r}')

        if self._abort.is_set():
            return False
        self._emit_confirmed(confirmed)
        return True

    def _emit_confirmed(self, confirmed, final=False):
        """Emit the text half of a (start, end, text) tuple, if there is any.

        Deliberately *not* ``post_process_transcription``. That function is
        written for a whole utterance: it strips the text and re-adds a trailing
        space. Applied to a confirmed fragment it would destroy the spacing
        whisper_streaming already encodes. faster-whisper emits each word with
        its own leading space and whisper_streaming joins words with "", so
        successive fragments concatenate correctly on their own, including when
        a split lands inside a word ("l'" + "agent"). Stripping each fragment
        and appending a space would insert one there.

        So intra-session spacing is left exactly as the model produced it: only
        the session's first fragment is left-trimmed, and the configured
        trailing space is added once, after the final one, to separate this
        dictation from the next.
        """
        text = confirmed[2] if confirmed and len(confirmed) > 2 else ''
        post_processing = ConfigManager.get_config_section('post_processing')

        if text and post_processing.get('remove_capitalization'):
            text = text.lower()
        if text and not self._emitted_any:
            text = text.lstrip()

        if final:
            if post_processing.get('remove_trailing_period') and text.endswith('.'):
                text = text[:-1]
            if post_processing.get('add_trailing_space') and (text or self._emitted_any):
                text += ' '

        if not text:
            return
        self._emitted_any = True
        self.resultSignal.emit(text)

    # ------------------------------------------------------------------
    # Producer side (runs in a plain daemon thread)
    # ------------------------------------------------------------------

    def _produce(self):
        """Hold one InputStream open and forward speech, gating pure silence.

        The VAD here is a gate, never a knife. It decides whether a frame is
        worth sending to the decoder, and it is the only thing standing between
        a long silence and a rolling buffer full of nothing. It never ends a
        segment: cutting is what the old pipeline did, and what made every cut
        permanent.
        """
        recording_options = ConfigManager.get_config_section('recording_options')
        silence_duration_ms = recording_options.get('silence_duration') or 900
        silence_frames = max(1, int(silence_duration_ms / FRAME_DURATION_MS))

        vad = webrtcvad.Vad(2)  # 0 to 3, 3 being the most aggressive
        frames_to_skip = int(KEY_CLICK_SKIP_SECONDS * 1000 / FRAME_DURATION_MS)
        speech_active = False
        silent_frame_count = 0
        preroll = []

        sample_buffer = bytearray()
        buffer_lock = threading.Lock()
        data_ready = threading.Event()

        def audio_callback(indata, frames, time_info, status):
            if status:
                ConfigManager.console_print(f'Audio callback status: {status}')
            with buffer_lock:
                sample_buffer.extend(indata.tobytes())
            data_ready.set()

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                                blocksize=FRAME_SIZE,
                                device=recording_options.get('sound_device'),
                                callback=audio_callback):
                ConfigManager.console_print('Microphone open. Listening continuously...')

                frame_bytes = FRAME_SIZE * 2  # int16
                while not self._stop_producer.is_set():
                    if not data_ready.wait(timeout=0.1):
                        continue
                    data_ready.clear()

                    while True:
                        with buffer_lock:
                            if len(sample_buffer) < frame_bytes:
                                break
                            raw = bytes(sample_buffer[:frame_bytes])
                            del sample_buffer[:frame_bytes]

                        frame = np.frombuffer(raw, dtype=np.int16)

                        if frames_to_skip > 0:
                            frames_to_skip -= 1
                            continue

                        is_speech = vad.is_speech(raw, SAMPLE_RATE)
                        if is_speech:
                            silent_frame_count = 0
                            if not speech_active:
                                speech_active = True
                                ConfigManager.console_print('Speech detected.')
                                for held in preroll:
                                    self._push(held)
                                preroll = []
                        else:
                            silent_frame_count += 1

                        if speech_active:
                            # Trailing silence is forwarded too: Whisper needs it
                            # to know the word ended.
                            self._push(frame)
                            if silent_frame_count > silence_frames:
                                speech_active = False
                                ConfigManager.console_print('Pause; gating the microphone.')
                        else:
                            preroll.append(frame)
                            if len(preroll) > PREROLL_FRAMES:
                                preroll.pop(0)
        except Exception:
            traceback.print_exc()
        finally:
            self._audio_queue.put(_END_OF_STREAM)
            ConfigManager.console_print('Microphone closed.')

    def _push(self, frame):
        """Queue one frame of audio and account for it in the backlog."""
        self.mutex.lock()
        try:
            self._pending_samples += len(frame)
        finally:
            self.mutex.unlock()
        self._audio_queue.put(frame)

    # ------------------------------------------------------------------
    # Shared state helpers
    # ------------------------------------------------------------------

    def _settle_backlog(self, consumed_samples):
        """Drop consumed audio from the backlog, then report the new figure."""
        self.mutex.lock()
        try:
            self._pending_samples = max(0, self._pending_samples - consumed_samples)
        finally:
            self.mutex.unlock()
        self._report_backlog()

    def _report_backlog(self):
        """Warn once when the backlog crosses the threshold, and set the status."""
        threshold = ConfigManager.get_config_value('recording_options',
                                                   'backlog_warning_seconds') or 15
        pending = self.pending_seconds()

        warning = None
        self.mutex.lock()
        try:
            if pending > threshold and not self._backlog_warned:
                self._backlog_warned = True
                warning = (f'WARNING: {pending:.1f}s of speech is waiting to be '
                           f'transcribed (over the {threshold}s threshold). Text will '
                           f'lag behind, but nothing is dropped.')
            elif pending <= threshold and self._backlog_warned:
                self._backlog_warned = False
        finally:
            self.mutex.unlock()

        if warning:
            ConfigManager.console_print(warning)

        # 'listening' while the microphone is open, 'draining' once it is not.
        # Either way the lag is appended when it is worth mentioning, because a
        # slow drain is exactly when the user wants to know how much is left.
        state = 'draining' if self._stop_producer.is_set() else 'listening'
        self._emit_status(f'{state}:{int(pending)}' if pending >= 3 else state)

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
