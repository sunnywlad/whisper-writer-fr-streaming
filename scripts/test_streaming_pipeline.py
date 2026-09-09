"""Headless harness for src/streaming_pipeline.py.

The GUI, the microphone and faster-whisper are all unavailable in a Linux/WSL
checkout, so this script stubs out PyQt5, sounddevice, webrtcvad and
transcription, then pushes a synthetic int16 signal through the real producer /
consumer code.

It asserts the three properties the streaming rewrite exists for:

1. the microphone stream is opened once per session and closed once, never
   between two segments;
2. every captured segment is emitted exactly once and in capture order, even
   when transcription is far slower than capture;
3. stopping the session drains the backlog before the thread exits, while
   aborting drops it and returns promptly.

Run with the system Python (the project venv is Windows-only):

    python3 scripts/test_streaming_pipeline.py
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
    """Deterministic stand-in: loud frame means speech."""

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
# Stub: transcription
# ----------------------------------------------------------------------

TRANSCRIBE_DELAY = [0.0]
TRANSCRIBE_CALLS = []


def _mock_transcribe(audio_data, local_model=None):
    """Sleep, then read the segment identity back out of the amplitude."""
    time.sleep(TRANSCRIBE_DELAY[0])
    amplitude = int(np.abs(audio_data).max())
    TRANSCRIBE_CALLS.append(amplitude)
    return f'seg-{amplitude} '


def _install_transcription_stub():
    module = types.ModuleType('transcription')
    module.transcribe = _mock_transcribe
    sys.modules['transcription'] = module


# ----------------------------------------------------------------------
# Signal building
# ----------------------------------------------------------------------

def build_signal(burst_durations_ms, sample_rate, silence_ms, lead_in_ms=300,
                 first_amplitude=5000, amplitude_step=100):
    """Build speech bursts separated by silence.

    Each burst carries its own amplitude, so the mocked transcriber can name the
    segment it was handed and the test can check ordering exactly.

    :return: (int16 signal, list of amplitudes in order)
    """
    silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.int16)
    parts = [np.zeros(int(sample_rate * lead_in_ms / 1000), dtype=np.int16)]
    amplitudes = []

    half_period = max(1, sample_rate // 400)  # ~200 Hz square wave
    for index, duration_ms in enumerate(burst_durations_ms):
        amplitude = first_amplitude + index * amplitude_step
        count = int(sample_rate * duration_ms / 1000)
        ramp = np.arange(count)
        burst = np.where((ramp // half_period) % 2 == 0, amplitude, -amplitude).astype(np.int16)
        parts.append(burst)
        parts.append(silence)
        amplitudes.append(amplitude)

    return np.concatenate(parts), amplitudes


# ----------------------------------------------------------------------
# Harness plumbing
# ----------------------------------------------------------------------

LOG = []


def _install_stubs_and_import():
    _install_pyqt_stub()
    _install_webrtcvad_stub()
    _install_sounddevice_stub()
    _install_transcription_stub()

    sys.path.insert(0, SRC_DIR)
    from utils import ConfigManager
    ConfigManager.initialize(schema_path=os.path.join(SRC_DIR, 'config_schema.yaml'))

    def capturing_console_print(message):
        LOG.append(str(message))

    ConfigManager.console_print = staticmethod(capturing_console_print)

    import streaming_pipeline
    return ConfigManager, streaming_pipeline


def configure(ConfigManager, **options):
    """Pin every recording option the pipeline reads, so the run is hermetic."""
    defaults = {
        'sample_rate': 16000,
        'silence_duration': 700,
        'min_duration': 100,
        'max_segment_duration': 0,
        'backlog_warning_seconds': 15,
        'sound_device': None,
    }
    defaults.update(options)
    for key, value in defaults.items():
        ConfigManager.set_config_value(value, 'recording_options', key)


class Collector:
    """Records what the pipeline emits, as the main thread would."""

    def __init__(self, thread):
        self.results = []
        self.statuses = []
        thread.resultSignal.connect(self.results.append)
        thread.statusSignal.connect(self.statuses.append)


FAILURES = []


def check(condition, message):
    if condition:
        print(f'  ok   {message}')
    else:
        print(f'  FAIL {message}')
        FAILURES.append(message)


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------

def scenario_gapless_with_slow_transcription(ConfigManager, streaming_pipeline):
    """Capture 6 bursts while transcription runs far slower than capture."""
    print('\n[1] gapless capture, transcription slower than speech')
    LOG.clear()
    TRANSCRIBE_CALLS.clear()
    configure(ConfigManager, silence_duration=700, backlog_warning_seconds=3)

    sample_rate = 16000
    bursts = [600, 900, 700, 1100, 800, 1000]
    signal, amplitudes = build_signal(bursts, sample_rate, silence_ms=900)
    _Mic.load(signal, speed=30.0)

    TRANSCRIBE_DELAY[0] = 0.35  # ~6x slower than the 30x-accelerated capture

    thread = streaming_pipeline.StreamingResultThread(local_model=None)
    collector = Collector(thread)
    thread.start()

    # Let the whole scripted signal be captured, then stop the session.
    check(_Mic.exhausted.wait(timeout=30), 'the scripted signal was fully fed')
    peak_backlog = thread.pending_seconds()
    thread.stop()

    finished = thread.wait(30000)
    check(finished, 'the consumer thread finished after stop()')

    expected = [f'seg-{amplitude} ' for amplitude in amplitudes]
    check(collector.results == expected,
          f'{len(expected)} segments emitted once, in capture order '
          f'(got {len(collector.results)})')
    if collector.results != expected:
        print(f'       expected {expected}')
        print(f'       got      {collector.results}')

    check(len(set(TRANSCRIBE_CALLS)) == len(TRANSCRIBE_CALLS),
          'no segment was transcribed twice')
    check(_Mic.opens == 1 and _Mic.closes == 1,
          f'the microphone stream was opened once and closed once '
          f'(opens={_Mic.opens}, closes={_Mic.closes})')
    check(peak_backlog > 1.0,
          f'a real backlog built up while capture continued ({peak_backlog:.1f}s pending)')
    check(any('WARNING' in line and 'waiting to be transcribed' in line for line in LOG),
          'the backlog warning was printed')
    check(thread.pending_seconds() == 0.0, 'the queue was drained to zero')
    check(collector.statuses[0] == 'recording' and collector.statuses[-1] == 'idle',
          f'status went recording -> ... -> idle (got {collector.statuses})')


def scenario_max_segment_cut(ConfigManager, streaming_pipeline):
    """A monologue with no pause at all is cut by the safety knob."""
    print('\n[2] monologue with no pause is cut at max_segment_duration')
    LOG.clear()
    TRANSCRIBE_CALLS.clear()
    configure(ConfigManager, silence_duration=700, max_segment_duration=3)

    sample_rate = 16000
    signal, _ = build_signal([12000], sample_rate, silence_ms=900)
    _Mic.load(signal, speed=60.0)

    TRANSCRIBE_DELAY[0] = 0.02

    thread = streaming_pipeline.StreamingResultThread(local_model=None)
    collector = Collector(thread)
    thread.start()

    check(_Mic.exhausted.wait(timeout=30), 'the scripted signal was fully fed')
    thread.stop()
    check(thread.wait(30000), 'the consumer thread finished after stop()')

    cut_notices = [line for line in LOG if 'No pause after' in line]
    check(len(collector.results) >= 4,
          f'the 12s monologue produced at least 4 segments (got {len(collector.results)})')
    check(len(cut_notices) >= 3,
          f'the safety cut fired and said so (got {len(cut_notices)} notices)')
    check(all('discarded' not in line for line in LOG),
          'no segment was discarded as too short')


def scenario_abort_drops_backlog(ConfigManager, streaming_pipeline):
    """abort() must return promptly and not wait for the backlog."""
    print('\n[3] abort() drops the backlog instead of draining it')
    LOG.clear()
    TRANSCRIBE_CALLS.clear()
    configure(ConfigManager, silence_duration=700)

    sample_rate = 16000
    bursts = [700] * 8
    signal, amplitudes = build_signal(bursts, sample_rate, silence_ms=900)
    _Mic.load(signal, speed=60.0)

    TRANSCRIBE_DELAY[0] = 0.5

    thread = streaming_pipeline.StreamingResultThread(local_model=None)
    collector = Collector(thread)
    thread.start()

    check(_Mic.exhausted.wait(timeout=30), 'the scripted signal was fully fed')
    backlog = thread.pending_seconds()
    started = time.time()
    thread.abort(wait_ms=5000)
    elapsed = time.time() - started

    check(backlog > 2.0, f'a backlog existed at abort time ({backlog:.1f}s pending)')
    check(not thread.isRunning(), 'the thread stopped on abort')
    check(elapsed < 3.0, f'abort returned promptly ({elapsed:.2f}s)')
    check(len(collector.results) < len(amplitudes),
          f'the backlog was dropped rather than transcribed '
          f'({len(collector.results)} of {len(amplitudes)} typed)')
    check(_Mic.opens == 1 and _Mic.closes == 1,
          f'the microphone stream was still opened and closed exactly once '
          f'(opens={_Mic.opens}, closes={_Mic.closes})')


def scenario_short_segment_discarded(ConfigManager, streaming_pipeline):
    """min_duration still applies, per segment."""
    print('\n[4] min_duration discards a too-short segment, keeps the rest')
    LOG.clear()
    TRANSCRIBE_CALLS.clear()
    configure(ConfigManager, silence_duration=700, min_duration=1500)

    sample_rate = 16000
    # 1st and 3rd bursts land under the 1500 ms floor once silence is added,
    # the 2nd one clears it.
    bursts = [200, 2000, 200]
    signal, amplitudes = build_signal(bursts, sample_rate, silence_ms=900)
    _Mic.load(signal, speed=60.0)

    TRANSCRIBE_DELAY[0] = 0.02

    thread = streaming_pipeline.StreamingResultThread(local_model=None)
    collector = Collector(thread)
    thread.start()

    check(_Mic.exhausted.wait(timeout=30), 'the scripted signal was fully fed')
    thread.stop()
    check(thread.wait(30000), 'the consumer thread finished after stop()')

    check(collector.results == [f'seg-{amplitudes[1]} '],
          f'only the long segment survived (got {collector.results})')
    check(sum('discarded' in line for line in LOG) == 2,
          'both short segments were reported as discarded')


def main():
    ConfigManager, streaming_pipeline = _install_stubs_and_import()

    scenario_gapless_with_slow_transcription(ConfigManager, streaming_pipeline)
    scenario_max_segment_cut(ConfigManager, streaming_pipeline)
    scenario_abort_drops_backlog(ConfigManager, streaming_pipeline)
    scenario_short_segment_discarded(ConfigManager, streaming_pipeline)

    print()
    if FAILURES:
        print(f'{len(FAILURES)} check(s) failed:')
        for failure in FAILURES:
            print(f'  - {failure}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
