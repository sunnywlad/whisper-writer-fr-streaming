"""NVIDIA Parakeet TDT 0.6B v3 backend (sherpa-onnx, int8, CPU).

Why a second engine: on a 4-core laptop CPU, Whisper ``small`` re-encodes a
fixed 30 s window on every streaming pass (~2 s of encoder alone), so the
streaming decoder barely keeps pace with speech. Parakeet is a transducer: its
cost scales with the audio actually in the buffer, and on the same machine it
decodes about three times faster than real time, with better French accuracy
than ``small``.

:class:`ParakeetASR` exposes the three methods whisper_streaming's
``OnlineASRProcessor`` calls (``transcribe``, ``ts_words``,
``segments_end_ts``), so the LocalAgreement loop in ``streaming_asr`` runs
unchanged on top of it. It has no equivalent of Whisper's ``initial_prompt``.
"""

import os

import numpy as np

from utils import ConfigManager

SAMPLE_RATE = 16000

DEFAULT_MODEL_DIR = os.path.join('models', 'sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8')

# Silence added on both sides of every window. Without it a transducer can
# return nothing at all, or a truncated sentence, on a window that ends
# abruptly (measured: an empty result on a 7.8 s window that decodes fine
# 0.5 s shorter).
PAD_SECONDS = 0.5

# A word ending this close to the end of the window was cut by the window, not
# by the speaker. Its punctuation is the model reacting to the audio stopping
# ("une. une tokenomique"), so it is dropped; the next pass, which hears what
# follows, decides.
WINDOW_EDGE_SECONDS = 0.3

_PUNCTUATION = '.,?!;:…'


def create_parakeet_recognizer():
    """Load the sherpa-onnx recognizer described by model_options.local."""
    import sherpa_onnx

    local_options = ConfigManager.get_config_section('model_options', 'local') or {}
    model_dir = local_options.get('parakeet_model_dir') or DEFAULT_MODEL_DIR
    threads = local_options.get('cpu_threads') or 4
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f'Parakeet model not found in {model_dir!r}. Download '
            'sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2 from the sherpa-onnx '
            '"asr-models" release and extract it there.')

    options = dict(
        encoder=os.path.join(model_dir, 'encoder.int8.onnx'),
        decoder=os.path.join(model_dir, 'decoder.int8.onnx'),
        joiner=os.path.join(model_dir, 'joiner.int8.onnx'),
        tokens=os.path.join(model_dir, 'tokens.txt'),
        model_type='nemo_transducer',
        num_threads=threads)
    options.update(_hotword_options(local_options, model_dir))

    ConfigManager.console_print(f'Loading Parakeet from {model_dir} ({threads} threads, '
                                f'{options.get("decoding_method", "greedy_search")})...')
    return sherpa_onnx.OfflineRecognizer.from_transducer(**options)


def _hotword_options(local_options, model_dir):
    """Decode-time vocabulary biasing: Parakeet's stand-in for initial_prompt.

    The hotwords file is plain text, one term per line. sherpa-onnx tokenises
    it itself when given ``modeling_unit='bpe'`` and a SentencePiece-style
    ``bpe.vocab``; without those it falls back to a CJK splitter that tears
    "▁Le" into "▁" + "Le" and silently skips most terms. ``bpe.vocab`` is
    derived once from NVIDIA's ``tokenizer.json`` (a BPE model: a piece's score
    is minus its rank). Hotwords need beam search; with none configured the
    faster greedy search is kept.
    """
    hotwords_file = local_options.get('parakeet_hotwords_file')
    if not hotwords_file:
        return {}
    if not os.path.isfile(hotwords_file):
        ConfigManager.console_print(f'Hotwords file {hotwords_file!r} not found; greedy search.')
        return {}

    bpe_vocab = os.path.join(model_dir, 'bpe.vocab')
    if not os.path.isfile(bpe_vocab):
        tokenizer_json = os.path.join(model_dir, 'tokenizer.json')
        if not os.path.isfile(tokenizer_json):
            ConfigManager.console_print(
                f'Hotwords need {tokenizer_json} (from huggingface.co/nvidia/'
                'parakeet-tdt-0.6b-v3); greedy search without hotwords.')
            return {}
        import json
        with open(tokenizer_json, encoding='utf-8') as f:
            vocab = json.load(f)['model']['vocab']
        with open(bpe_vocab, 'w', encoding='utf-8') as f:
            for piece, rank in sorted(vocab.items(), key=lambda item: item[1]):
                f.write(f'{piece}\t{-rank}\n')

    return dict(decoding_method='modified_beam_search',
                max_active_paths=4,
                hotwords_file=hotwords_file,
                hotwords_score=float(local_options.get('parakeet_hotwords_score') or 2.0),
                modeling_unit='bpe',
                bpe_vocab=bpe_vocab)


def is_parakeet(model):
    """True for a sherpa-onnx recognizer, False for a faster-whisper model."""
    return type(model).__module__.startswith('sherpa_onnx')


class ParakeetASR:
    """Parakeet behind the backend interface OnlineASRProcessor expects."""

    # Words carry their own leading space, exactly like faster-whisper's.
    sep = ''

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self._window_seconds = 0.0

    def _decode(self, audio):
        pad = np.zeros(int(PAD_SECONDS * SAMPLE_RATE), dtype=np.float32)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.concatenate([pad, audio.astype(np.float32), pad]))
        self.recognizer.decode_stream(stream)
        return stream.result

    def transcribe(self, audio, init_prompt=''):
        """Decode one window. ``init_prompt`` is accepted and ignored."""
        self._window_seconds = len(audio) / float(SAMPLE_RATE)
        return self._decode(audio)

    def transcribe_text(self, audio):
        """Whole-utterance transcription, for the non-streaming modes."""
        return self._decode(audio).text.strip()

    def ts_words(self, result):
        """Group sub-word tokens into ``(start, end, ' word')`` triples.

        Two timing rules matter, both learned from real recordings:

        * a word ends where its last token ends (start + TDT duration), never
          where the next token starts. Across a pause that would stretch the
          word over the silence, and the buffer cap would cut the next phrase
          away with it;
        * a punctuation token never moves a word's end. The model can emit the
          "." when it decides the sentence is over, which may be seconds later
          (seen: "Voilà" at 37.3 s, its "." at 41.9 s).
        """
        tokens = list(result.tokens)
        starts = [t - PAD_SECONDS for t in result.timestamps]
        durations = list(getattr(result, 'durations', []) or []) or [0.08] * len(tokens)

        # [start, end, text, time of the word's last punctuation token]
        words = []
        for token, start, duration in zip(tokens, starts, durations):
            piece = token.replace('▁', ' ')
            is_punctuation = not any(ch.isalnum() for ch in piece)
            if is_punctuation and words:
                # French typography gives "?" its own word ("▁?"). Kept as its
                # own word, it carries a late timestamp and, stripped at a
                # window edge, leaves a bare space behind. Attach it instead,
                # space included, without touching the word's end.
                words[-1][2] += piece
                words[-1][3] = start
                continue
            if piece.startswith(' ') or not words:
                words.append([start, start + duration,
                              piece if piece.startswith(' ') else ' ' + piece, None])
                continue
            words[-1][2] += piece
            words[-1][1] = start + duration

        # The last word's punctuation is dropped when either the word or the
        # punctuation itself sits at the window edge: that is the model
        # reacting to the audio stopping, not the speaker ending a sentence.
        edge = self._window_seconds - WINDOW_EDGE_SECONDS
        if words and (words[-1][1] >= edge or (words[-1][3] or 0) >= edge):
            words[-1][2] = words[-1][2].rstrip(_PUNCTUATION + ' ')
        return [tuple(w[:3]) for w in words if any(ch.isalnum() for ch in w[2])]

    def segments_end_ts(self, result):
        """A transducer has no segments; sentence ends stand in for them."""
        return [end for _, end, word in self.ts_words(result)
                if word.rstrip()[-1:] in '.?!']
