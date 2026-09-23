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

from parakeet_asr import ParakeetASR, is_parakeet
from transcription import decode_temperatures
from utils import ConfigManager
from vendor.whisper_online import FasterWhisperASR, HypothesisBuffer, OnlineASRProcessor

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


def _norm_word(word):
    """A word as LocalAgreement should compare it: case and punctuation ignored."""
    return ''.join(ch for ch in word.lower() if ch.isalnum())


class NormalizedHypothesisBuffer(HypothesisBuffer):
    """Upstream's HypothesisBuffer, comparing words case- and punctuation-blind.

    Upstream compares word strings exactly. Two decodes that hear the same word
    but punctuate it differently ("une." at a window edge, "une" once the next
    words are audible; "le" / "Le") then fail to agree, so LocalAgreement holds
    the word back, and the n-gram check that drops re-decoded words fails too,
    which types them twice ("une. une tokenomique", "le Le Wagon"). The logic
    below is upstream's ``insert`` and ``flush`` with only the comparisons
    normalised; the committed text is the newer decode's, punctuation included.
    """

    def insert(self, new, offset):
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1 and self.commited_in_buffer:
                cn, nn = len(self.commited_in_buffer), len(self.new)
                for i in range(1, min(min(cn, nn), 5) + 1):
                    c = ' '.join(_norm_word(w[2]) for w in self.commited_in_buffer[-i:])
                    tail = ' '.join(_norm_word(w[2]) for w in self.new[:i])
                    if c == tail:
                        del self.new[:i]
                        break

    def flush(self):
        commit = []
        while self.new and self.buffer:
            na, nb, nt = self.new[0]
            if _norm_word(nt) != _norm_word(self.buffer[0][2]):
                break
            commit.append((na, nb, nt))
            self.last_commited_word = nt
            self.last_commited_time = nb
            self.buffer.pop(0)
            self.new.pop(0)
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit


class NormalizedOnlineASRProcessor(OnlineASRProcessor):
    """OnlineASRProcessor whose hypothesis buffer is a NormalizedHypothesisBuffer."""

    def init(self, offset=None):
        super().init(offset)
        buffer = NormalizedHypothesisBuffer()
        buffer.last_commited_time = self.buffer_time_offset
        self.transcript_buffer = buffer


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
            # Candidates sampled per fallback step. faster-whisper defaults to 5,
            # so one degenerate pass through the full ladder cost up to 26
            # decodes: the 2 s -> 23 s spikes seen in the logs. One candidate
            # per step keeps the retry, at a fraction of the price.
            'best_of': local_options.get('best_of') or 1,
            'condition_on_previous_text': bool(
                local_options.get('condition_on_previous_text')),
            # The repetition-loop guard. A scalar temperature leaves faster-whisper
            # no way to notice a degenerate decode and retry it; the fallback list
            # plus the two thresholds below make it re-roll at a higher temperature.
            # Every other rung only (0.0, 0.4, 0.8): LocalAgreement already keeps
            # a transient loop from being committed, so three tries are enough.
            'temperature': decode_temperatures(common_options.get('temperature'))[::2],
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
            # inside it is the one main.py loaded at startup: a faster-whisper
            # model or a Parakeet recognizer, per model_options.local.engine.
            if is_parakeet(self.local_model):
                asr = ParakeetASR(self.local_model)
            else:
                asr = InjectedFasterWhisperASR(
                    model=self.local_model,
                    lan=language,
                    initial_prompt=common_options.get('initial_prompt'))
            online = NormalizedOnlineASRProcessor(
                asr,
                tokenizer=None,  # "segment" trimming never touches the tokenizer
                buffer_trimming=('segment', trimming_seconds))
            online.init()
            ConfigManager.console_print(
                f'Streaming ASR ready ({type(asr).__name__}, language={language}, '
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

        # Wall-clock backstop. LocalAgreement only confirms a word once two
        # successive decodes agree on it; at a natural pause the settle pass
        # below is meant to supply that second decode, but if the preceding
        # decode trimmed the buffer (segment completion) the premise is broken
        # and the tail can stay invisible until finish() (F9). If the queue has
        # been empty and nothing new has been confirmed for this long, force the
        # current unconfirmed tail out and carry on without ending the session.
        tail_flush_seconds = float(recording_options.get('tail_flush_seconds') or 4.0)

        # Hard ceiling on the rolling buffer. Upstream trims only on a completed
        # segment, and a long French sentence often stays one segment, so the
        # buffer grew unchecked and every pass re-decoded all of it.
        max_buffer_samples = int(
            float(recording_options.get('max_buffer_seconds') or 12) * SAMPLE_RATE)

        collected = deque()
        collected_samples = 0
        producer_done = False
        settled = True
        last_audio_at = time.time()
        # Last time the user saw forward motion: audio taken in, or text put
        # out. The backstop is driven by this, not by raw silence.
        last_progress_at = time.time()
        # Samples in the processor's rolling buffer right after the last decode.
        # A later decode that leaves fewer means a segment was completed and the
        # buffer trimmed, which is exactly what strands the tail.
        last_decode_buffer_samples = self._buffer_samples(online)

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
                last_progress_at = last_audio_at
                # New audio re-arms the settle pass unconditionally.
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
                # Backstop: the queue has drained, nothing is buffered locally,
                # and no audio or text has moved for tail_flush_seconds. The
                # settle pass has had its chance and the tail is still held.
                # Force it out (commit the pending hypothesis on the current
                # buffer) without ending the session, then keep listening. A
                # forced word may be one a later decode would have revised;
                # that is preferred to text that never appears.
                if (not collected and self._audio_queue.empty()
                        and time.time() - last_progress_at >= tail_flush_seconds):
                    forced = self._force_flush_tail(online)
                    if forced and len(forced) > 2 and forced[2]:
                        ConfigManager.console_print(
                            f'Tail-flush: no progress for {tail_flush_seconds:.1f}s; '
                            f'forcing out {forced[2]!r}')
                        self._emit_confirmed(forced)
                    # Reset the clock whether or not there was anything to
                    # flush, so this retries at most once per interval.
                    last_progress_at = time.time()
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

            ok, confirmed_text = self._step(online, audio)
            if not ok:
                break

            forced = self._enforce_buffer_cap(online, max_buffer_samples)
            if forced and forced[2]:
                ConfigManager.console_print(
                    f'Buffer cap: nothing confirmed in the buffer; forcing out '
                    f'{forced[2]!r}')
                self._emit_confirmed(forced)
                confirmed_text = True

            # Re-arm the settle pass after a decode that either confirmed text
            # or trimmed the buffer. Without this, `settled` latches True after
            # one pass and no later tail is ever re-confirmed; a trim is the
            # case the single settle pass silently gets wrong.
            buffer_now = self._buffer_samples(online)
            trimmed = buffer_now < last_decode_buffer_samples
            last_decode_buffer_samples = buffer_now
            if confirmed_text or trimmed:
                settled = False
                last_progress_at = time.time()

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

        Returns ``(keep_going, confirmed_text)``: ``keep_going`` is False when
        the caller should stop looping, ``confirmed_text`` True when this pass
        committed any new text.
        """
        if len(audio):
            online.insert_audio_chunk(audio.astype(np.float32) / 32768.0)

        started = time.time()
        try:
            confirmed = online.process_iter()
        except Exception:
            traceback.print_exc()
            self._emit_status('error')
            return True, False  # a bad decode is not a reason to end the session
        elapsed = time.time() - started

        audio_seconds = len(audio) / float(SAMPLE_RATE)
        buffer_seconds = self._buffer_samples(online) / float(SAMPLE_RATE)
        ConfigManager.console_print(
            f'Decoded {audio_seconds:.2f}s of new audio in {elapsed:.2f}s '
            f'(buffer {buffer_seconds:.1f}s); confirmed: {confirmed[2]!r}')

        if self._abort.is_set():
            return False, False
        self._emit_confirmed(confirmed)
        text = confirmed[2] if confirmed and len(confirmed) > 2 else ''
        return True, bool(text and text.strip())

    @staticmethod
    def _buffer_samples(online):
        """Length of the processor's rolling audio buffer, 0 if it has none."""
        return len(getattr(online, 'audio_buffer', ()))

    @classmethod
    def _enforce_buffer_cap(cls, online, max_samples):
        """Trim the rolling buffer once it exceeds ``max_samples``.

        The cut lands at the end of the last confirmed word, never inside audio
        whose words have not been typed yet. If the buffer holds no confirmed
        word at all, the pending hypothesis is forced out first so there is one
        to cut at. Only when even that is empty (nothing recognisable in the
        whole buffer) is the oldest audio dropped, keeping the newest half.

        Returns the forced ``(beg, end, text)`` tuple for the caller to emit,
        or None when nothing had to be forced.
        """
        buffer_samples = cls._buffer_samples(online)
        if buffer_samples <= max_samples:
            return None

        offset = online.buffer_time_offset
        buffer_end = offset + buffer_samples / float(SAMPLE_RATE)
        max_seconds = max_samples / float(SAMPLE_RATE)
        forced = None
        # Force when there is no confirmed word to cut at, or when cutting at
        # it would still leave the buffer over the ceiling.
        if (not online.commited or online.commited[-1][1] <= offset
                or buffer_end - online.commited[-1][1] > max_seconds):
            forced = cls._force_flush_tail(online)

        if online.commited and online.commited[-1][1] > offset:
            cut_at = online.commited[-1][1]
        else:
            cut_at = buffer_end - max_seconds / 2
        online.chunk_at(cut_at)
        ConfigManager.console_print(
            f'Buffer cap: {buffer_samples / SAMPLE_RATE:.1f}s buffer trimmed at '
            f'{cut_at:.2f}s, {cls._buffer_samples(online) / SAMPLE_RATE:.1f}s left.')
        return forced

    @staticmethod
    def _drop_recommitted_prefix(commited, pending, max_ngram=5):
        """Strip leading pending words that repeat the end of the committed text.

        A re-decode often starts its hypothesis on the last word already typed
        ("... quelque chose." then " chose. Je veux ..."). Upstream's normal
        commit path drops such an overlap in ``HypothesisBuffer.insert``, but
        only while the matching words are still in ``commited_in_buffer``; after
        a trim they are gone, and a forced flush would type the word twice.
        Same rule as upstream (longest matching n-gram up to 5 words), compared
        against the full committed list and ignoring case and punctuation.
        """
        if not commited or not pending:
            return pending

        for n in range(min(max_ngram, len(commited), len(pending)), 0, -1):
            tail = [_norm_word(w[2]) for w in commited[-n:]]
            head = [_norm_word(w[2]) for w in pending[:n]]
            if tail == head and any(tail):
                return pending[n:]
        return pending

    @staticmethod
    def _force_flush_tail(online):
        """Commit the pending (unconfirmed) hypothesis without ending the session.

        Mirrors what ``HypothesisBuffer.flush`` does when LocalAgreement agrees:
        the held words move into the committed set and the hypothesis buffer is
        cleared, so they are neither re-emitted by a later ``process_iter`` nor
        emitted again by ``finish()``. The rolling audio buffer and its time
        offset are left untouched, so decoding simply continues.

        Returns a ``(beg, end, text)`` tuple like ``process_iter``.
        """
        tb = getattr(online, 'transcript_buffer', None)
        pending = list(getattr(tb, 'buffer', []) or []) if tb is not None else []
        pending = StreamingASRThread._drop_recommitted_prefix(online.commited, pending)
        if not pending:
            if tb is not None:
                tb.buffer = []
            return (None, None, '')
        try:
            online.commited.extend(pending)
            tb.commited_in_buffer.extend(pending)
            tb.last_commited_time = pending[-1][1]
            tb.last_commited_word = pending[-1][2]
            tb.buffer = []
            return online.to_flush(pending)
        except Exception:
            traceback.print_exc()
            return (None, None, '')

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
