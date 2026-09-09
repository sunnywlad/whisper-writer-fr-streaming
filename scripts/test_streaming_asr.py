"""Headless harness for src/streaming_asr.py.

The GUI, the microphone and faster-whisper are all unavailable in a Linux/WSL
checkout, so this script stubs out PyQt5, sounddevice, webrtcvad and the
whisper_streaming backend, then pushes a synthetic int16 signal through the real
producer / consumer code.

It cannot judge transcription quality. What it does assert is the mechanical
contract the streaming rewrite has to honour:

1. the microphone stream is opened once per session and closed once, never in
   the middle of it;
2. the ASR object and its processor are built exactly once per session, and the
   model handed to them is the one instance the application already loaded;
3. every confirmed tuple the decoder returns is emitted exactly once, in order,
   with no gaps, even when decoding is far slower than capture;
4. every sample the producer queued reaches insert_audio_chunk: no lost audio;
5. finish()'s held-back tail is always emitted on stop, even with a backlog
   still pending;
6. the backlog warning fires once the pending audio passes
   backlog_warning_seconds;
7. stop() returns immediately and lets the backlog drain; abort() returns
   promptly and drops it;
8. the decode options carry a temperature *fallback ladder*, not a bare scalar,
   on both the streaming and the legacy path;
9. the injected backend still fits the genuine vendored whisper_streaming API,
   not merely the mock of it used by the threading checks above.

Run with the system Python (the project venv is Windows-only):

    python3 scripts/test_streaming_asr.py
"""

import os
import sys
import threading
import time
import types

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, 'src')


# ----------------------------------------------------------------------
# Stub: PyQt5.QtCore
# ----------------------------------------------------------------------

class _BoundSignal:
    """Minimal signal. Slots are called inline, on the emitting thread."""

    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in list(self._slots):
            slot(*args)


class pyqtSignal:
    """Descriptor handing out one _BoundSignal per instance, like PyQt does."""

    def __init__(self, *types_):
        self._name = None

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        signals = obj.__dict__.setdefault('_stub_signals', {})
        return signals.setdefault(self._name, _BoundSignal())


class QThread:
    """threading.Thread wearing a QThread face."""

    def __init__(self):
        self._stub_thread = None

    def start(self):
        self._stub_thread = threading.Thread(target=self.run, daemon=True)
        self._stub_thread.start()

    def run(self):
        pass

    def isRunning(self):
        return self._stub_thread is not None and self._stub_thread.is_alive()

    def wait(self, msecs=None):
        if self._stub_thread is None:
            return True
        self._stub_thread.join(None if msecs is None else msecs / 1000.0)
        return not self._stub_thread.is_alive()


class QMutex:
    def __init__(self):
        self._lock = threading.RLock()

    def lock(self):
        self._lock.acquire()

    def unlock(self):
        self._lock.release()


def _install_pyqt_stub():
    qtcore = types.ModuleType('PyQt5.QtCore')
    qtcore.QThread = QThread
    qtcore.QMutex = QMutex
    qtcore.pyqtSignal = pyqtSignal
    pyqt5 = types.ModuleType('PyQt5')
    pyqt5.QtCore = qtcore
    sys.modules['PyQt5'] = pyqt5
    sys.modules['PyQt5.QtCore'] = qtcore


# ----------------------------------------------------------------------
# Stub: webrtcvad
# ----------------------------------------------------------------------

SPEECH_THRESHOLD = 1000


class _StubVad:
    """Deterministic stand-in: a loud frame means speech."""

    def __init__(self, aggressiveness=2):
        self.aggressiveness = aggressiveness

    def is_speech(self, frame_bytes, sample_rate):
        frame = np.frombuffer(frame_bytes, dtype=np.int16)
        return bool(np.abs(frame).max() > SPEECH_THRESHOLD)


def _install_webrtcvad_stub():
    module = types.ModuleType('webrtcvad')
    module.Vad = _StubVad
    sys.modules['webrtcvad'] = module


# ----------------------------------------------------------------------
# Stub: sounddevice
# ----------------------------------------------------------------------

class _Mic:
    """Scripted microphone shared by the stub InputStream."""

    signal = np.zeros(0, dtype=np.int16)
    speed = 30.0
    opens = 0
    closes = 0
    exhausted = threading.Event()

    @classmethod
    def load(cls, signal, speed=30.0):
        cls.signal = signal
        cls.speed = speed
        cls.opens = 0
        cls.closes = 0
        cls.exhausted = threading.Event()


class _StubInputStream:
    def __init__(self, **kwargs):
        self.samplerate = kwargs['samplerate']
        self.blocksize = kwargs['blocksize']
        self.callback = kwargs['callback']
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        _Mic.opens += 1
        self._thread = threading.Thread(target=self._feed, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        _Mic.closes += 1
        return False

    def _feed(self):
        signal = _Mic.signal
        period = (self.blocksize / float(self.samplerate)) / _Mic.speed
        index = 0
        while not self._stop.is_set() and index < len(signal):
            block = signal[index:index + self.blocksize]
            index += self.blocksize
            if len(block) < self.blocksize:
                block = np.concatenate(
                    [block, np.zeros(self.blocksize - len(block), dtype=np.int16)])
            self.callback(block.reshape(-1, 1), self.blocksize, None, None)
            time.sleep(period)
        _Mic.exhausted.set()


def _install_sounddevice_stub():
    module = types.ModuleType('sounddevice')
    module.InputStream = _StubInputStream
    sys.modules['sounddevice'] = module


# ----------------------------------------------------------------------
# Stub: the heavy imports transcription.py pulls in
#
# transcription.py itself is NOT stubbed: the harness exercises the real
# decode_temperatures and post_process_transcription.
# ----------------------------------------------------------------------

class _StubWhisperModel:
    """Stands in for faster_whisper.WhisperModel. Never actually decodes."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def transcribe(self, *args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError('the streaming path must not call the model directly')


def _install_heavy_import_stubs():
    faster_whisper = types.ModuleType('faster_whisper')
    faster_whisper.WhisperModel = _StubWhisperModel
    sys.modules['faster_whisper'] = faster_whisper

    if 'soundfile' not in sys.modules:
        soundfile = types.ModuleType('soundfile')
        soundfile.write = lambda *a, **k: None
        sys.modules['soundfile'] = soundfile

    openai = types.ModuleType('openai')
    openai.OpenAI = lambda *a, **k: None
    sys.modules['openai'] = openai


# ----------------------------------------------------------------------
# Stub: vendor.whisper_online
#
# FasterWhisperASR is replaced by a stand-in with upstream's ASRBase shape, so
# InjectedFasterWhisperASR is exercised for real (its load_model override, its
# decode options) without a model on disk. OnlineASRProcessor is replaced by a
# scripted fake returning known confirmed tuples.
# ----------------------------------------------------------------------

DECODE_DELAY = [0.0]


class _MockFasterWhisperASR:
    """Same constructor contract as upstream ASRBase / FasterWhisperASR."""

    sep = ""
    instances = []

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=None):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan == 'auto' else lan
        self.model = self.load_model(modelsize, cache_dir, model_dir)
        _MockFasterWhisperASR.instances.append(self)

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        raise AssertionError('the subclass must override load_model')

    def ts_words(self, segments):
        return []

    def segments_end_ts(self, res):
        return []


class _MockHypBuffer:
    """The two fields streaming_asr._force_flush_tail reads and rewrites."""

    def __init__(self):
        self.buffer = []
        self.commited_in_buffer = []
        self.last_commited_time = 0
        self.last_commited_word = None


class _MockOnlineASRProcessor:
    """Scripted LocalAgreement stand-in.

    Default mode: process_iter() hands back a fresh 'text-N' every call, except
    every third call which returns the empty tuple, so the consumer's "nothing
    confirmed this pass" path is exercised too. That numbering is what lets the
    test prove exactly-once, in-order delivery without knowing how many passes
    ran.

    HOLD_TAIL mode (opt-in, for the tail-flush tests): each utterance (a run of
    decodes on real signal, ended by silence) leaves exactly one word held in
    ``transcript_buffer.buffer``. It is confirmed only when the *next* utterance
    begins; a settle pass on silence never confirms it. So the final word of the
    last utterance stays invisible until the wall-clock backstop forces it out.
    TRIM_AT_UTTERANCE simulates a mid-session ``chunk_completed_segment``
    shrinking the rolling buffer when that utterance's first decode runs.
    """

    TAIL_TEXT = 'the-held-back-tail'
    HOLD_TAIL = False
    TRIM_AT_UTTERANCE = None
    instances = []

    def __init__(self, asr, tokenizer=None, buffer_trimming=None, logfile=None):
        self.asr = asr
        self.tokenizer = tokenizer
        self.buffer_trimming = buffer_trimming
        self.inserted_samples = 0
        self.init_calls = 0
        self.iter_calls = 0
        self.finish_calls = 0
        self._lock = threading.Lock()
        # Surface the real OnlineASRProcessor exposes and that the force-flush
        # reaches into.
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.commited = []
        self.transcript_buffer = _MockHypBuffer()
        self._new_audio = False
        self._in_speech = False
        self._utterance = 0
        _MockOnlineASRProcessor.instances.append(self)

    def init(self, offset=None):
        self.init_calls += 1

    def insert_audio_chunk(self, audio):
        assert audio.dtype == np.float32, f'expected float32, got {audio.dtype}'
        if len(audio):
            assert np.abs(audio).max() <= 1.0, 'audio must be normalised to [-1, 1]'
        with self._lock:
            self.inserted_samples += len(audio)
            self.audio_buffer = np.append(self.audio_buffer, audio)
            # Only actual signal re-arms the held tail. Trailing silence frames
            # (all zeros, forwarded by the VAD gate after speech stops) must
            # leave the tail stuck, which is the real-world stall this models.
            if len(audio) and float(np.abs(audio).max()) > 1e-4:
                self._new_audio = True

    def to_flush(self, sents, sep=None, offset=0):
        if sep is None:
            sep = getattr(self.asr, 'sep', '')
        text = sep.join(s[2] for s in sents)
        if not sents:
            return (None, None, '')
        return (offset + sents[0][0], offset + sents[-1][1], text)

    def process_iter(self):
        time.sleep(DECODE_DELAY[0])
        with self._lock:
            self.iter_calls += 1
            n = self.iter_calls

        if not _MockOnlineASRProcessor.HOLD_TAIL:
            if n % 3 == 0:
                return (None, None, '')
            return (float(n), float(n) + 1.0, f'text-{n}')

        # --- HOLD_TAIL -------------------------------------------------------
        with self._lock:
            fresh = self._new_audio
            self._new_audio = False

        tb = self.transcript_buffer

        if not fresh:
            # Silence, incl. the settle pass: no spontaneous agreement, so the
            # held word stays invisible. The wall-clock backstop must rescue it.
            self._in_speech = False
            return (None, None, '')

        if self._in_speech:
            # Same utterance continuing: keep refining the one held word, commit
            # nothing.
            return (None, None, '')

        # A new utterance begins. The previous utterance's held word now gets
        # its confirming decode; a brand-new held word takes its place.
        self._in_speech = True
        self._utterance += 1

        confirmed = list(tb.buffer)
        tb.buffer = []
        if confirmed:
            self.commited.extend(confirmed)
            tb.commited_in_buffer.extend(confirmed)

        seq = self._utterance
        tb.buffer = [(float(seq), float(seq) + 1.0, f'w{seq}')]

        if _MockOnlineASRProcessor.TRIM_AT_UTTERANCE == seq:
            keep = len(self.audio_buffer) // 4
            self.audio_buffer = (self.audio_buffer[-keep:] if keep
                                 else self.audio_buffer[:0])

        return self.to_flush(confirmed)

    def finish(self):
        with self._lock:
            self.finish_calls += 1
        tb = self.transcript_buffer
        if _MockOnlineASRProcessor.HOLD_TAIL and tb.buffer:
            tail = self.to_flush(list(tb.buffer))
            tb.buffer = []
            return tail
        return (0.0, 1.0, self.TAIL_TEXT)


def _install_vendor_stub():
    whisper_online = types.ModuleType('vendor.whisper_online')
    whisper_online.FasterWhisperASR = _MockFasterWhisperASR
    whisper_online.OnlineASRProcessor = _MockOnlineASRProcessor
    vendor = types.ModuleType('vendor')
    vendor.whisper_online = whisper_online
    sys.modules['vendor'] = vendor
    sys.modules['vendor.whisper_online'] = whisper_online


# ----------------------------------------------------------------------
# Signal building
# ----------------------------------------------------------------------

def build_signal(burst_durations_ms, sample_rate, silence_ms, lead_in_ms=300,
                 amplitude=6000):
    """Build speech bursts separated by silence long enough to close the gate."""
    silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.int16)
    parts = [np.zeros(int(sample_rate * lead_in_ms / 1000), dtype=np.int16)]

    half_period = max(1, sample_rate // 400)  # ~200 Hz square wave
    for duration_ms in burst_durations_ms:
        count = int(sample_rate * duration_ms / 1000)
        ramp = np.arange(count)
        burst = np.where((ramp // half_period) % 2 == 0,
                         amplitude, -amplitude).astype(np.int16)
        parts.append(burst)
        parts.append(silence)

    return np.concatenate(parts)


# ----------------------------------------------------------------------
# Harness plumbing
# ----------------------------------------------------------------------

LOG = []
CONSOLE = []


def _load_real_vendored_module():
    """Import src/vendor/whisper_online.py under a private name.

    The stub below replaces `vendor.whisper_online` for the threading tests, but
    one check has to run against the genuine upstream classes: that is the only
    way to catch the vendored API drifting away from what InjectedFasterWhisperASR
    assumes. Loading it here, before the stub is installed, keeps both available.
    """
    import importlib.util

    path = os.path.join(SRC_DIR, 'vendor', 'whisper_online.py')
    spec = importlib.util.spec_from_file_location('real_whisper_online', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_stubs_and_import():
    _install_pyqt_stub()
    _install_webrtcvad_stub()
    _install_sounddevice_stub()
    _install_heavy_import_stubs()
    real_vendor = _load_real_vendored_module()
    _install_vendor_stub()

    sys.path.insert(0, SRC_DIR)
    from utils import ConfigManager
    ConfigManager.initialize(schema_path=os.path.join(SRC_DIR, 'config_schema.yaml'))

    # Capture console output so the backlog warning can be asserted on.
    def _recording_console_print(cls, message):
        CONSOLE.append(str(message))

    ConfigManager.console_print = classmethod(_recording_console_print)

    import streaming_asr
    import transcription
    return ConfigManager, streaming_asr, transcription, real_vendor


def configure(config_manager, overrides=None):
    """Set the recording/model options this harness depends on.

    :param overrides: {(section, key, ...): value} applied over the defaults.
    """
    defaults = {
        ('recording_options', 'sample_rate'): 16000,
        ('recording_options', 'silence_duration'): 300,
        ('recording_options', 'min_chunk_seconds'): 0.5,
        ('recording_options', 'buffer_trimming_seconds'): 12,
        ('recording_options', 'backlog_warning_seconds'): 15,
        ('recording_options', 'tail_flush_seconds'): 4,
        ('recording_options', 'sound_device'): None,
        ('model_options', 'common', 'language'): 'fr',
        ('model_options', 'common', 'temperature'): 0.0,
        ('model_options', 'common', 'initial_prompt'): 'Hermes, Anthropic, Solidity.',
        ('model_options', 'local', 'beam_size'): 1,
        ('model_options', 'local', 'condition_on_previous_text'): False,
        ('post_processing', 'remove_trailing_period'): False,
        ('post_processing', 'add_trailing_space'): True,
        ('post_processing', 'remove_capitalization'): False,
    }
    defaults.update(overrides or {})
    for keys, value in defaults.items():
        config_manager.set_config_value(value, *keys)


def reset_state():
    LOG.clear()
    CONSOLE.clear()
    _MockFasterWhisperASR.instances.clear()
    _MockOnlineASRProcessor.instances.clear()
    _MockOnlineASRProcessor.HOLD_TAIL = False
    _MockOnlineASRProcessor.TRIM_AT_UTTERANCE = None
    DECODE_DELAY[0] = 0.0


def make_thread(streaming_asr, model):
    """A StreamingASRThread that records what its producer queued."""

    class _CountingThread(streaming_asr.StreamingASRThread):
        pushed_samples = 0

        def _push(self, frame):
            self.pushed_samples += len(frame)
            super()._push(frame)

    thread = _CountingThread(model)
    thread.resultSignal.connect(LOG.append)
    return thread


def assert_contiguous_texts(emitted, tail_text):
    """The emitted list must be text-1, text-2, ... with no gap and no repeat."""
    assert emitted, 'nothing was emitted at all'
    assert emitted[-1].strip() == tail_text, (
        f'the finish() tail must come last, got {emitted[-1]!r}')

    body = [text.strip() for text in emitted[:-1]]
    assert len(set(body)) == len(body), f'a confirmed block was emitted twice: {body}'

    numbers = []
    for text in body:
        assert text.startswith('text-'), f'unexpected emitted text {text!r}'
        numbers.append(int(text.split('-')[1]))
    assert numbers == sorted(numbers), f'confirmed text came out of order: {numbers}'

    # process_iter returns the empty tuple every third call; those must not be
    # emitted, and every other call must be.
    expected = [n for n in range(1, max(numbers) + 1) if n % 3 != 0]
    assert numbers == expected, (
        f'confirmed blocks were dropped or duplicated: got {numbers}, want {expected}')


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_slow_decode_emits_every_confirmation_in_order(config_manager, streaming_asr):
    """Audio keeps arriving while decoding lags. Nothing is lost or reordered."""
    reset_state()
    configure(config_manager)

    model = _StubWhisperModel()
    # ~12 s of speech played at 8x, decoded at 0.3 s a pass: the decoder lags
    # behind the microphone for the whole session, not only at the drain.
    _Mic.load(build_signal([1000] * 10, 16000, silence_ms=500), speed=8.0)
    DECODE_DELAY[0] = 0.3

    thread = make_thread(streaming_asr, model)
    thread.start()
    assert _Mic.exhausted.wait(timeout=30), 'the stub microphone never drained'

    started = time.time()
    thread.stop()
    stop_elapsed = time.time() - started
    assert stop_elapsed < 0.1, f'stop() blocked for {stop_elapsed:.2f}s'

    assert thread.wait(30000), 'the consumer never finished draining'

    processor = _MockOnlineASRProcessor.instances[0]
    assert_contiguous_texts(LOG, _MockOnlineASRProcessor.TAIL_TEXT)
    assert processor.finish_calls == 1, (
        f'finish() must run exactly once, ran {processor.finish_calls}')
    assert processor.inserted_samples == thread.pushed_samples, (
        f'lost audio: producer queued {thread.pushed_samples} samples, '
        f'{processor.inserted_samples} reached the decoder')
    assert _Mic.opens == 1 and _Mic.closes == 1, (
        f'the microphone must open and close once, got {_Mic.opens}/{_Mic.closes}')
    print(f'  ok  in-order, exactly-once: {len(LOG) - 1} confirmed blocks + tail; '
          f'{processor.inserted_samples} samples fed, none lost')


def test_asr_built_once_with_the_injected_model(config_manager, streaming_asr):
    """One ASR, one processor, and the model is the instance we passed in."""
    reset_state()
    configure(config_manager)

    model = _StubWhisperModel()
    _Mic.load(build_signal([400] * 3, 16000, silence_ms=500), speed=40.0)

    thread = make_thread(streaming_asr, model)
    thread.start()
    assert _Mic.exhausted.wait(timeout=30)
    thread.stop()
    assert thread.wait(30000)

    assert len(_MockFasterWhisperASR.instances) == 1, (
        f'the ASR was built {len(_MockFasterWhisperASR.instances)} times, want 1')
    assert len(_MockOnlineASRProcessor.instances) == 1, (
        f'the processor was built {len(_MockOnlineASRProcessor.instances)} times, want 1')

    asr = _MockFasterWhisperASR.instances[0]
    assert asr.model is model, 'the injected model was not the one used'
    assert asr.original_language == 'fr', f'language not wired: {asr.original_language}'
    assert 'Hermes' in asr._user_prompt, 'the vocabulary prompt was not wired'

    processor = _MockOnlineASRProcessor.instances[0]
    assert processor.buffer_trimming == ('segment', 12), (
        f'buffer trimming not wired: {processor.buffer_trimming}')
    assert processor.tokenizer is None, 'segment trimming must not need a tokenizer'
    print('  ok  one ASR, one processor, injected model, prompt and language wired')


def test_streaming_decode_options_use_the_temperature_ladder(config_manager,
                                                             streaming_asr):
    """The repetition-loop guard: a fallback ladder, never a bare scalar."""
    reset_state()
    configure(config_manager)

    options = streaming_asr.InjectedFasterWhisperASR._build_decode_options()
    temperature = options['temperature']
    assert isinstance(temperature, list) and len(temperature) > 1, (
        f'temperature must be a fallback ladder, got {temperature!r}')
    assert temperature[0] == 0.0 and temperature[-1] == 1.0, (
        f'unexpected ladder {temperature!r}')
    assert options['compression_ratio_threshold'] == 2.4
    assert options['log_prob_threshold'] == -1.0
    print(f'  ok  streaming decode temperature ladder {temperature}')


def test_legacy_path_temperature_ladder(transcription):
    """The same guard on the legacy ResultThread path."""
    assert transcription.decode_temperatures(0.0) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert transcription.decode_temperatures(None) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ladder = transcription.decode_temperatures(0.2)
    assert ladder[0] == 0.2 and ladder[-1] <= 1.0 and len(ladder) > 1, ladder
    assert transcription.decode_temperatures([0.0, 0.5]) == [0.0, 0.5]
    print('  ok  legacy path widens a scalar temperature into a ladder')


def test_backlog_warning_fires(config_manager, streaming_asr):
    """Pending audio past the threshold warns, and the status says so."""
    reset_state()
    configure(config_manager, {('recording_options', 'backlog_warning_seconds'): 2})

    statuses = []
    model = _StubWhisperModel()
    _Mic.load(build_signal([2000] * 4, 16000, silence_ms=500), speed=40.0)
    DECODE_DELAY[0] = 1.0  # a decoder far slower than the microphone

    thread = make_thread(streaming_asr, model)
    thread.statusSignal.connect(statuses.append)
    thread.start()
    assert _Mic.exhausted.wait(timeout=30)
    # Keep the microphone nominally open while the decoder works through the
    # pile, so the lag is reported as a 'listening' state and not only as a
    # drain after the stop.
    deadline = time.time() + 15
    while time.time() < deadline and not any(s.startswith('listening:') for s in statuses):
        time.sleep(0.05)
    thread.stop()
    assert thread.wait(60000)

    warnings = [line for line in CONSOLE if line.startswith('WARNING:')]
    assert warnings, f'no backlog warning was printed; console was {CONSOLE[-5:]}'
    behind = [s for s in statuses if s.startswith('listening:')]
    assert behind, f'the status never reported a lag; statuses were {statuses}'
    assert 'listening' in statuses, 'the session never reported plain listening'
    assert any(s.split(':')[0] == 'draining' for s in statuses), (
        f'the stop phase never reported draining; statuses were {statuses}')
    assert statuses[-1] == 'idle', f'the session must end idle, got {statuses[-1]}'
    print(f'  ok  backlog warning fired; status reached {sorted(set(behind))[-1]!r}')


def test_finish_tail_emitted_even_with_a_backlog(config_manager, streaming_asr):
    """Stop with work still queued: the tail must still be typed."""
    reset_state()
    configure(config_manager)

    model = _StubWhisperModel()
    _Mic.load(build_signal([1500] * 3, 16000, silence_ms=400), speed=50.0)
    DECODE_DELAY[0] = 0.4

    thread = make_thread(streaming_asr, model)
    thread.start()
    # Stop while the mic is still feeding, so a real backlog exists.
    time.sleep(0.6)
    thread.stop()
    assert thread.wait(60000), 'the drain never completed'

    processor = _MockOnlineASRProcessor.instances[0]
    assert LOG, 'nothing was emitted'
    assert LOG[-1].strip() == _MockOnlineASRProcessor.TAIL_TEXT, (
        f'the tail was not emitted last: {LOG[-3:]}')
    assert processor.finish_calls == 1
    assert processor.inserted_samples == thread.pushed_samples, (
        'audio captured before the stop was dropped instead of decoded')
    assert thread.pending_seconds() == 0.0, (
        f'{thread.pending_seconds():.2f}s still counted as pending after the drain')
    print('  ok  stop() drains the backlog and emits the finish() tail')


def test_abort_returns_promptly_and_drops_the_backlog(config_manager, streaming_asr):
    """App exit must not wait for a slow decoder to work through the queue."""
    reset_state()
    configure(config_manager)

    model = _StubWhisperModel()
    _Mic.load(build_signal([3000] * 3, 16000, silence_ms=400), speed=60.0)
    DECODE_DELAY[0] = 0.5

    thread = make_thread(streaming_asr, model)
    thread.start()
    time.sleep(1.0)
    emitted_before = len(LOG)

    started = time.time()
    thread.abort(wait_ms=5000)
    elapsed = time.time() - started
    assert elapsed < 4.0, f'abort() took {elapsed:.2f}s'
    assert not thread.isRunning(), 'the thread was still running after abort()'

    processor = _MockOnlineASRProcessor.instances[0]
    assert processor.finish_calls == 0, 'abort() must not flush the tail'
    assert len(LOG) - emitted_before <= 1, 'abort() kept draining the backlog'
    assert _Mic.closes == 1, 'the microphone was left open after abort()'
    print(f'  ok  abort() returned in {elapsed:.2f}s, backlog dropped, mic closed')


def _load_streaming_asr_against_real_vendor(real_vendor):
    """Load a second copy of streaming_asr bound to the genuine vendored module.

    The copy the other tests use was imported while `vendor.whisper_online` was
    stubbed, so its base class is the mock. Rather than mutate that module, this
    swaps the real module into sys.modules just long enough to execute a fresh
    copy, then puts the stub back.
    """
    import importlib.util

    stub = sys.modules['vendor.whisper_online']
    sys.modules['vendor.whisper_online'] = real_vendor
    sys.modules['vendor'].whisper_online = real_vendor
    try:
        path = os.path.join(SRC_DIR, 'streaming_asr.py')
        spec = importlib.util.spec_from_file_location('streaming_asr_real', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules['vendor.whisper_online'] = stub
        sys.modules['vendor'].whisper_online = stub


def test_subclass_fits_the_real_vendored_api(config_manager, _stubbed_streaming_asr,
                                             real_vendor):
    """Drive the genuine upstream OnlineASRProcessor with our injected backend.

    Everything else here runs against a mock of whisper_streaming. This check is
    what proves the mock is not lying: if a future re-vendor changes the
    constructor, the load_model signature or the transcribe contract, this fails
    rather than the application failing on the user's machine.
    """
    reset_state()
    configure(config_manager)

    class _RecordingModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, **kwargs):
            self.calls.append(kwargs)
            return iter([]), None

    injected = _load_streaming_asr_against_real_vendor(
        real_vendor).InjectedFasterWhisperASR
    assert injected.__mro__[1] is real_vendor.FasterWhisperASR, (
        'InjectedFasterWhisperASR no longer subclasses the vendored backend')

    model = _RecordingModel()
    asr = injected(model=model, lan='fr', initial_prompt='Hermes, Anthropic, Solidity.')
    assert asr.model is model, 'the vendored ASRBase loaded a model of its own'
    assert asr.original_language == 'fr'

    online = real_vendor.OnlineASRProcessor(asr, tokenizer=None,
                                            buffer_trimming=('segment', 12))
    online.init()
    for _ in range(2):
        online.insert_audio_chunk(np.zeros(16000, dtype=np.float32))
        confirmed = online.process_iter()
        assert isinstance(confirmed, tuple) and len(confirmed) == 3, confirmed
    tail = online.finish()
    assert isinstance(tail, tuple) and len(tail) == 3, tail

    assert model.calls, 'the processor never reached the backend'
    kwargs = model.calls[0]
    assert kwargs['word_timestamps'] is True, 'ts_words() needs word timestamps'
    assert isinstance(kwargs['temperature'], list) and len(kwargs['temperature']) > 1
    assert kwargs['language'] == 'fr'
    assert 'Hermes' in kwargs['initial_prompt']
    print('  ok  real upstream processor drives the injected backend end to end')


def test_fragment_spacing_is_left_to_the_model(config_manager, streaming_asr):
    """Fragments are typed verbatim; only the session's edges are adjusted.

    faster-whisper attaches a leading space to each word and whisper_streaming
    joins words with "", so consecutive fragments already concatenate correctly,
    including across a split inside a word. Stripping each fragment and adding a
    trailing space, as the legacy whole-utterance post-processing does, would
    insert a space in the middle of "l'agent".
    """
    reset_state()
    configure(config_manager)

    thread = make_thread(streaming_asr, _StubWhisperModel())
    thread._emit_confirmed((0.0, 1.0, " l'"))
    thread._emit_confirmed((1.0, 2.0, 'agent'))
    thread._emit_confirmed((2.0, 3.0, ' Hermes'))
    thread._emit_confirmed((3.0, 4.0, ''))          # nothing confirmed: no emit
    thread._emit_confirmed((4.0, 5.0, ' arrive'), final=True)

    typed = ''.join(LOG)
    assert typed == "l'agent Hermes arrive ", f'bad spacing: {typed!r}'
    assert LOG[0] == "l'", f'the first fragment must be left-trimmed: {LOG[0]!r}'
    assert LOG[1] == 'agent', f'a mid-word split must not gain a space: {LOG[1]!r}'
    assert LOG[-1].endswith(' '), 'the session must end with the configured space'
    print(f'  ok  fragments concatenate verbatim: {typed!r}')


def _wait_for(predicate, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_stuck_tail_is_force_flushed_when_idle(config_manager, streaming_asr):
    """Speak, then go quiet WITHOUT pressing stop: the tail must still appear.

    LocalAgreement holds the last word until a second decode agrees with it.
    At a natural pause that second decode never comes, so before this fix the
    word stayed invisible until finish() (F9). The wall-clock backstop must
    force it out within tail_flush_seconds.
    """
    reset_state()
    configure(config_manager, {
        ('recording_options', 'tail_flush_seconds'): 2,
        ('recording_options', 'min_chunk_seconds'): 0.2,
        ('recording_options', 'silence_duration'): 600,
    })
    _MockOnlineASRProcessor.HOLD_TAIL = True

    model = _StubWhisperModel()
    _Mic.load(build_signal([1200], 16000, silence_ms=900), speed=3.0)

    thread = make_thread(streaming_asr, model)
    thread.start()
    assert _Mic.exhausted.wait(timeout=15), 'the stub microphone never drained'

    # Deliberately no stop()/finish(): only the backstop can deliver the tail.
    flushed = _wait_for(lambda: bool(LOG), timeout=6)
    thread.abort()

    processor = _MockOnlineASRProcessor.instances[0]
    assert flushed, (
        f'the held tail was never force-flushed while idle; console tail '
        f'{CONSOLE[-5:]}')
    assert [s.strip() for s in LOG] == ['w1'], f'unexpected forced tail {LOG!r}'
    assert processor.finish_calls == 0, 'the backstop must not end the session'
    assert any('Tail-flush' in line for line in CONSOLE), (
        'the force-flush was not logged')
    print(f'  ok  idle tail force-flushed without stop(): {LOG!r}')


def test_stuck_tail_flushes_again_after_a_midsession_trim(config_manager,
                                                          streaming_asr):
    """The latching-`settled` regression: a trim must not disable later flushes.

    The single settle pass assumed an unchanged buffer and latched `settled`
    after running once. A mid-session segment completion trims the buffer, the
    assumption breaks, and every later tail stayed stuck until F9. After the
    fix `settled` re-arms on a trim and the backstop still fires regardless.
    """
    reset_state()
    configure(config_manager, {
        ('recording_options', 'tail_flush_seconds'): 2,
        ('recording_options', 'min_chunk_seconds'): 0.2,
        ('recording_options', 'silence_duration'): 600,
    })
    _MockOnlineASRProcessor.HOLD_TAIL = True
    # The 2nd utterance's first decode completes a segment and trims the buffer.
    _MockOnlineASRProcessor.TRIM_AT_UTTERANCE = 2

    model = _StubWhisperModel()
    _Mic.load(build_signal([1200, 1200], 16000, silence_ms=900), speed=3.0)

    thread = make_thread(streaming_asr, model)
    thread.start()
    assert _Mic.exhausted.wait(timeout=15), 'the stub microphone never drained'

    # 'w2' is the tail left unconfirmed *after* the trim. If `settled` stayed
    # latched and the backstop were absent, it would never be emitted.
    flushed = _wait_for(lambda: any(s.strip() == 'w2' for s in LOG), timeout=8)
    thread.abort()

    processor = _MockOnlineASRProcessor.instances[0]
    assert flushed, (
        f'the post-trim tail never flushed; LOG={LOG!r} console={CONSOLE[-6:]}')
    assert [s.strip() for s in LOG][:2] == ['w1', 'w2'], (
        f'confirmed then forced text is wrong: {LOG!r}')
    assert processor.finish_calls == 0, 'the backstop must not end the session'
    print(f'  ok  tail still flushes after a mid-session trim: {LOG!r}')


def main():
    (config_manager, streaming_asr, transcription,
     real_vendor) = _install_stubs_and_import()

    tests = [
        ('subclass fits the real vendored whisper_streaming API',
         lambda: test_subclass_fits_the_real_vendored_api(
             config_manager, streaming_asr, real_vendor)),
        ('slow decode emits every confirmation in order',
         lambda: test_slow_decode_emits_every_confirmation_in_order(
             config_manager, streaming_asr)),
        ('ASR built once with the injected model',
         lambda: test_asr_built_once_with_the_injected_model(
             config_manager, streaming_asr)),
        ('streaming decode uses the temperature ladder',
         lambda: test_streaming_decode_options_use_the_temperature_ladder(
             config_manager, streaming_asr)),
        ('legacy path uses the temperature ladder',
         lambda: test_legacy_path_temperature_ladder(transcription)),
        ('fragment spacing is left to the model',
         lambda: test_fragment_spacing_is_left_to_the_model(
             config_manager, streaming_asr)),
        ('backlog warning fires',
         lambda: test_backlog_warning_fires(config_manager, streaming_asr)),
        ('finish() tail emitted even with a backlog',
         lambda: test_finish_tail_emitted_even_with_a_backlog(
             config_manager, streaming_asr)),
        ('abort() returns promptly and drops the backlog',
         lambda: test_abort_returns_promptly_and_drops_the_backlog(
             config_manager, streaming_asr)),
        ('a stuck tail is force-flushed when the session goes idle',
         lambda: test_stuck_tail_is_force_flushed_when_idle(
             config_manager, streaming_asr)),
        ('a stuck tail still flushes after a mid-session trim',
         lambda: test_stuck_tail_flushes_again_after_a_midsession_trim(
             config_manager, streaming_asr)),
    ]

    failures = 0
    for name, test in tests:
        print(f'- {name}')
        try:
            test()
        except AssertionError as error:
            failures += 1
            print(f'  FAIL  {error}')
        except Exception as error:  # noqa: BLE001 - the harness reports, never raises
            failures += 1
            print(f'  ERROR {type(error).__name__}: {error}')
            import traceback
            traceback.print_exc()

    print()
    if failures:
        print(f'{failures} of {len(tests)} checks failed.')
        return 1
    print(f'All {len(tests)} checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
