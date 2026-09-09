import io
import os
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from openai import OpenAI

from utils import ConfigManager

def create_local_model():
    """
    Create a local model using the faster-whisper library.
    """
    ConfigManager.console_print('Creating local model...')
    local_model_options = ConfigManager.get_config_section('model_options')['local']
    compute_type = local_model_options['compute_type']
    model_path = local_model_options.get('model_path')
    cpu_threads = local_model_options.get('cpu_threads', 0) or 0

    if compute_type == 'int8':
        device = 'cpu'
        ConfigManager.console_print('Using int8 quantization, forcing CPU usage.')
    else:
        device = local_model_options['device']

    try:
        if model_path:
            ConfigManager.console_print(f'Loading model from: {model_path}')
            model = WhisperModel(model_path,
                                 device=device,
                                 compute_type=compute_type,
                                 cpu_threads=cpu_threads,
                                 download_root=None)  # Prevent automatic download
        else:
            model = WhisperModel(local_model_options['model'],
                                 device=device,
                                 compute_type=compute_type,
                                 cpu_threads=cpu_threads)
    except Exception as e:
        ConfigManager.console_print(f'Error initializing WhisperModel: {e}')
        ConfigManager.console_print('Falling back to CPU.')
        model = WhisperModel(model_path or local_model_options['model'],
                             device='cpu',
                             compute_type=compute_type,
                             cpu_threads=cpu_threads,
                             download_root=None if model_path else None)

    ConfigManager.console_print('Local model created.')
    return model

def decode_temperatures(configured):
    """Turn a configured temperature into faster-whisper's fallback ladder.

    A bare ``temperature=0.0`` makes decoding purely greedy, and greedy decoding
    on an awkward audio chunk is what sends Whisper into a repetition loop
    ("le plus grand, le plus grand, ..."). faster-whisper can detect such a
    degenerate decode, via ``compression_ratio_threshold`` and
    ``log_prob_threshold``, and retry it at a higher temperature, but only when
    it is given a *sequence* of temperatures to fall back through. A scalar
    leaves it nowhere to go, so it emits the loop.

    A configured scalar is therefore widened into the standard ladder starting
    at that value. A sequence the user set explicitly is passed through.
    """
    if isinstance(configured, (list, tuple)):
        return list(configured) or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    start = float(configured or 0.0)
    ladder = [round(start + 0.2 * step, 2) for step in range(6)]
    return [t for t in ladder if t <= 1.0] or [start]

def transcribe_local(audio_data, local_model=None):
    """
    Transcribe an audio file using a local model.
    """
    if not local_model:
        local_model = create_local_model()
    model_options = ConfigManager.get_config_section('model_options')

    # Convert int16 to float32
    audio_data_float = audio_data.astype(np.float32) / 32768.0

    response = local_model.transcribe(audio=audio_data_float,
                                      language=model_options['common']['language'],
                                      initial_prompt=model_options['common']['initial_prompt'],
                                      condition_on_previous_text=model_options['local']['condition_on_previous_text'],
                                      # Repetition-loop guard: a fallback ladder,
                                      # never a bare scalar. See decode_temperatures.
                                      temperature=decode_temperatures(
                                          model_options['common']['temperature']),
                                      compression_ratio_threshold=2.4,  # faster-whisper default
                                      log_prob_threshold=-1.0,          # faster-whisper default
                                      vad_filter=model_options['local']['vad_filter'],
                                      beam_size=model_options['local'].get('beam_size', 5),)
    return ''.join([segment.text for segment in list(response[0])])

def transcribe_api(audio_data):
    """
    Transcribe an audio file using the OpenAI API.
    """
    model_options = ConfigManager.get_config_section('model_options')
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY') or None,
        base_url=model_options['api']['base_url'] or 'https://api.openai.com/v1'
    )

    # Convert numpy array to WAV file
    byte_io = io.BytesIO()
    sample_rate = ConfigManager.get_config_section('recording_options').get('sample_rate') or 16000
    sf.write(byte_io, audio_data, sample_rate, format='wav')
    byte_io.seek(0)

    response = client.audio.transcriptions.create(
        model=model_options['api']['model'],
        file=('audio.wav', byte_io, 'audio/wav'),
        language=model_options['common']['language'],
        prompt=model_options['common']['initial_prompt'],
        temperature=model_options['common']['temperature'],
    )
    return response.text

def post_process_transcription(transcription):
    """
    Apply post-processing to the transcription.
    """
    transcription = transcription.strip()
    post_processing = ConfigManager.get_config_section('post_processing')
    if post_processing['remove_trailing_period'] and transcription.endswith('.'):
        transcription = transcription[:-1]
    if post_processing['add_trailing_space']:
        transcription += ' '
    if post_processing['remove_capitalization']:
        transcription = transcription.lower()

    return transcription

def transcribe(audio_data, local_model=None):
    """
    Transcribe audio date using the OpenAI API or a local model, depending on config.
    """
    if audio_data is None:
        return ''

    if ConfigManager.get_config_value('model_options', 'use_api'):
        transcription = transcribe_api(audio_data)
    else:
        transcription = transcribe_local(audio_data, local_model)

    return post_process_transcription(transcription)

