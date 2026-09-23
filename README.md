# <img src="./assets/ww-logo.png" alt="WhisperWriter icon" width="25" height="25"> WhisperWriter

![version](https://img.shields.io/badge/version-1.0.1-blue)

<p align="center">
    <img src="./assets/ww-demo-image-02.gif" alt="WhisperWriter demo gif" width="340" height="136">
</p>

**Update (2024-05-28):** I've just merged in a major rewrite of WhisperWriter! We've migrated from using `tkinter` to using `PyQt5` for the UI, added a new settings window for configuration, a new continuous recording mode, support for a local API, and more! Please be patient as I work out any bugs that may have been introduced in the process. If you encounter any problems, please [open a new issue](https://github.com/savbell/whisper-writer/issues)!

WhisperWriter is a small speech-to-text app that uses [OpenAI's Whisper model](https://openai.com/research/whisper) to auto-transcribe recordings from a user's microphone to the active window.

Once started, the script runs in the background and waits for a keyboard shortcut to be pressed (`ctrl+shift+space` by default). When the shortcut is pressed, the app starts recording from your microphone. There are four recording modes to choose from:
- `continuous` (default): Recording will stop after a long enough pause in your speech. The app will transcribe the text and then start recording again. To stop listening, press the keyboard shortcut again.
- `voice_activity_detection`: Recording will stop after a long enough pause in your speech. Recording will not start until the keyboard shortcut is pressed again.
- `press_to_toggle` Recording will stop when the keyboard shortcut is pressed again. Recording will not start until the keyboard shortcut is pressed again.
- `hold_to_record` Recording will continue until the keyboard shortcut is released. Recording will not start until the keyboard shortcut is held down again.

You can change the keyboard shortcut (`activation_key`) and recording mode in the [Configuration Options](#configuration-options). While recording and transcribing, a small status window is displayed that shows the current stage of the process (but this can be turned off). Once the transcription is complete, the transcribed text will be automatically written to the active window.

The transcription can either be done locally through the [faster-whisper Python package](https://github.com/SYSTRAN/faster-whisper/) or through a request to [OpenAI's API](https://platform.openai.com/docs/guides/speech-to-text). By default, the app will use a local model, but you can change this in the [Configuration Options](#configuration-options). If you choose to use the API, you will need to either provide your OpenAI API key or change the base URL endpoint.

**Fun fact:** Almost the entirety of the initial release of the project was pair-programmed with [ChatGPT-4](https://openai.com/product/gpt-4) and [GitHub Copilot](https://github.com/features/copilot) using VS Code. Practically every line, including most of this README, was written by AI. After the initial prototype was finished, WhisperWriter was used to write a lot of the prompts as well!

## This fork: French streaming dictation on a CPU laptop

[Français](#français) · [English](#english)

### Français

Cette version vise un usage précis : dicter en français, en continu, sur un
portable sans GPU (Intel i5-10210U, 4 cœurs, 8 Go), avec du jargon technique.
Le texte s'écrit au fil de la parole, sans attendre la fin de la phrase.

#### Ce qui diffère d'upstream

- **Streaming réel** (`recording_mode: continuous`, `use_streaming: true`) : le
  micro reste ouvert, et l'algorithme LocalAgreement de
  [whisper_streaming](https://github.com/ufal/whisper_streaming) ne tape que les
  mots sur lesquels deux décodages successifs s'accordent.
- **Second moteur, Parakeet TDT 0.6B v3** (`engine: parakeet`), via
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) en int8. Whisper reste
  disponible (`engine: whisper`).
- **Hotwords** : le vocabulaire de `hotwords.txt` est favorisé pendant le
  décodage. C'est l'équivalent de l'`initial_prompt` de Whisper pour Parakeet.
- Correctifs de stabilité et de vitesse, détaillés ci-dessous.

#### Démarche

Le point de départ était une application qui gelait ou devenait très lente.
Chaque étape a été mesurée avant d'être retenue.

**1. Diagnostic à partir des journaux** (`journalctl --user`, 509 décodages réels)

- Le décodage était plus lent que la parole : 1,27 s de calcul par seconde
  d'audio. Le retard s'accumulait donc inévitablement.
- Le « gel » était en fait la vidange de ce retard après l'arrêt (F9).
- Des pics de 2 s à 23 s pour le même morceau venaient du repli en
  température de faster-whisper. Avec `best_of=5`, une passe ratée pouvait
  coûter jusqu'à 26 décodages.
- Whisper encode toujours une fenêtre de 30 s : 2,15 s de coût fixe par passe,
  même pour 0,7 s d'audio nouveau. Avec 8 threads, c'est pire (4,5 s) : on
  garde 4 threads.
- Le buffer glissant ne se coupait que sur un segment terminé, et une longue
  phrase française n'en produit souvent aucun.

**2. Premiers correctifs, sur Whisper**

- `best_of: 1` et trois températures au lieu de six : pics supprimés (pire
  passe 4,5 s au lieu de 23 s).
- `max_buffer_seconds` : plafond du buffer, coupé au dernier mot confirmé, sans
  jamais retaper un mot déjà écrit.
- Key listener : le clavier virtuel de dotool est ignoré, et les codes de
  touche inconnus ne font plus perdre un appui sur F9.
- Résultat : plus stable, mais toujours au rythme de la parole au mieux. La
  limite était le coût fixe de Whisper.

**3. Comparaison de trois moteurs** sur des enregistrements réels, avec le même
filtre de silence et la même horloge simulée que l'application :

| | Whisper small | SimulStreaming (hybride) | Parakeet v3 |
|---|---|---|---|
| Calcul / seconde d'audio | 0,93 à 1,35 | 1,23 | 0,5 à 0,7 |
| Retard médian | 8,6 à 19 s | 6,2 s | 3 à 4 s |

SimulStreaming (via WhisperLiveKit) garde l'encodeur rapide mais décode en
PyTorch : plus lent ici. Parakeet l'emporte, parce que son coût suit l'audio
réel au lieu d'une fenêtre fixe de 30 s.

**4. Intégration de Parakeet** (`src/parakeet_asr.py`). Le passage en streaming
d'un modèle hors ligne a demandé quatre règles, toutes tirées d'erreurs
observées :

- un mot se termine avec son dernier token, pas au début du mot suivant (sinon
  il s'étire sur un silence et le plafond coupe la phrase suivante) ;
- la ponctuation ne repousse jamais la fin d'un mot, parce que le modèle peut
  poser le « . » plusieurs secondes plus tard ;
- 0,5 s de silence entoure chaque fenêtre, sans quoi une fenêtre qui s'arrête
  net peut rendre un texte vide ;
- LocalAgreement compare les mots sans tenir compte de la casse ni de la
  ponctuation, ce qui supprime des doublons comme « une. une ».

**5. Jargon.** Deux correcteurs placés après la transcription ont été testés et
écartés :

- un petit LLM (Qwen2.5 1,5B : invente des phrases ; 3B : juste, mais 5 à 11 s
  par phrase) ;
- un modèle de ponctuation seule (rapide, mais 2 Go de RAM et quelques
  erreurs).

La solution retenue agit pendant le décodage : les hotwords de sherpa-onnx,
avec le tokenizer de NVIDIA. Un biais de 2,0 corrige Alyra, TradFi, dApp,
Anthropic et Hermes. À 3,0, le modèle invente des sigles au début des sessions.
Les sigles courts (MRN, IA) sont donc limités à 1,0.

#### Installation de Parakeet

```bash
pip install -r requirements.txt   # ajoute sherpa-onnx
mkdir -p models && cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
tar xjf sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
# tokenizer, requis pour les hotwords (bpe.vocab est généré au premier lancement)
curl -L -o sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokenizer.json \
  https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/tokenizer.json
```

Créez ensuite `hotwords.txt` à la racine (ignoré par git), un terme par ligne.
Un score optionnel se met en fin de ligne :

```text
Claude Code
Alyra
MRN :1.0
```

#### Nouvelles options

| Option | Rôle | Valeur retenue |
|---|---|---|
| `local.engine` | `whisper` ou `parakeet` ; seul le moteur choisi est chargé | `parakeet` |
| `local.parakeet_model_dir` | dossier du modèle | `models/sherpa-onnx-…` |
| `local.parakeet_hotwords_file` | vocabulaire à favoriser ; vide = pas de biais | `hotwords.txt` |
| `local.parakeet_hotwords_score` | force du biais | `2.0` |
| `local.best_of` | candidats par repli de température (Whisper) | `1` |
| `recording.max_buffer_seconds` | plafond du buffer glissant | `20` |
| `recording.min_chunk_seconds` / `buffer_trimming_seconds` | cadence et découpe, réglées pour Parakeet | `2.0` / `10` |

Avec `engine: whisper`, revenez à `min_chunk_seconds: 3.0`,
`buffer_trimming_seconds: 5` et `max_buffer_seconds: 12`.

#### Limites connues

- Un doublon occasionnel aux jonctions entre fenêtres, surtout en début de
  session.
- Les noms composés collés sortent parfois en deux mots (« Whisper Writer »).
- `scripts/test_streaming_asr.py` teste la mécanique (14 vérifications), pas la
  qualité de transcription.

### English

This version targets one use case: continuous French dictation on a laptop with
no GPU (Intel i5-10210U, 4 cores, 8 GB), with technical jargon. Text is typed
as you speak, without waiting for the end of the sentence.

#### What differs from upstream

- **True streaming** (`recording_mode: continuous`, `use_streaming: true`): the
  microphone stays open, and the LocalAgreement algorithm from
  [whisper_streaming](https://github.com/ufal/whisper_streaming) only types the
  words two successive decodes agree on.
- **A second engine, Parakeet TDT 0.6B v3** (`engine: parakeet`), through
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) in int8. Whisper remains
  available (`engine: whisper`).
- **Hotwords**: the vocabulary in `hotwords.txt` is favoured at decode time.
  It is Parakeet's equivalent of Whisper's `initial_prompt`.
- Stability and speed fixes, detailed below.

#### Approach

The starting point was an app that froze or became very slow. Every step was
measured before it was kept.

**1. Diagnosis from the logs** (`journalctl --user`, 509 real decodes)

- Decoding was slower than speech: 1.27 s of compute per second of audio, so
  the lag could only grow.
- The "freeze" was the backlog draining after the stop key (F9).
- Spikes from 2 s to 23 s on the same chunk came from faster-whisper's
  temperature fallback. With `best_of=5`, one failed pass could cost up to 26
  decodes.
- Whisper always encodes a 30 s window: a fixed 2.15 s per pass, even for
  0.7 s of new audio. Eight threads are worse (4.5 s), so four are kept.
- The rolling buffer was only trimmed on a completed segment, which a long
  French sentence often never produces.

**2. First fixes, on Whisper**

- `best_of: 1` and three temperatures instead of six: spikes gone (worst pass
  4.5 s instead of 23 s).
- `max_buffer_seconds`: a buffer ceiling, cut at the last confirmed word,
  never retyping a word already written.
- Key listener: dotool's virtual keyboard is ignored, and unknown key codes no
  longer swallow an F9 press.
- Result: more stable, but at best level with speech. The limit was Whisper's
  fixed cost.

**3. Three engines compared** on real recordings, with the app's own silence
gate and a simulated clock:

| | Whisper small | SimulStreaming (hybrid) | Parakeet v3 |
|---|---|---|---|
| Compute / second of audio | 0.93 to 1.35 | 1.23 | 0.5 to 0.7 |
| Median lag | 8.6 to 19 s | 6.2 s | 3 to 4 s |

SimulStreaming (through WhisperLiveKit) keeps the fast encoder but decodes in
PyTorch, which is slower here. Parakeet wins because its cost follows the real
audio instead of a fixed 30 s window.

**4. Integrating Parakeet** (`src/parakeet_asr.py`). Streaming an offline model
took four rules, each drawn from an observed failure:

- a word ends with its last token, not at the start of the next word (else it
  stretches over a pause and the buffer cap cuts the next sentence away);
- punctuation never moves a word's end, since the model may emit the "."
  seconds later;
- 0.5 s of silence pads each window, otherwise a window that ends abruptly can
  decode to nothing;
- LocalAgreement compares words ignoring case and punctuation, which removes
  duplicates such as "une. une".

**5. Jargon.** Two post-transcription correctors were tried and rejected:

- a small LLM (Qwen2.5 1.5B invents sentences; 3B is right but takes 5 to 11 s
  per sentence);
- a punctuation-only model (fast, but 2 GB of RAM and a few errors).

The chosen fix works during decoding: sherpa-onnx hotwords, with NVIDIA's
tokenizer. A bias of 2.0 fixes Alyra, TradFi, dApp, Anthropic and Hermes. At
3.0 the model invents acronyms at the start of sessions, so short acronyms
(MRN, IA) are capped at 1.0.

#### Installing Parakeet

```bash
pip install -r requirements.txt   # adds sherpa-onnx
mkdir -p models && cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
tar xjf sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
# tokenizer, required for hotwords (bpe.vocab is generated on first launch)
curl -L -o sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokenizer.json \
  https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/tokenizer.json
```

Then create `hotwords.txt` at the repository root (git-ignored), one term per
line, with an optional score at the end of the line:

```text
Claude Code
Alyra
MRN :1.0
```

#### New options

| Option | Purpose | Chosen value |
|---|---|---|
| `local.engine` | `whisper` or `parakeet`; only the chosen engine is loaded | `parakeet` |
| `local.parakeet_model_dir` | model folder | `models/sherpa-onnx-…` |
| `local.parakeet_hotwords_file` | vocabulary to favour; empty = no bias | `hotwords.txt` |
| `local.parakeet_hotwords_score` | bias strength | `2.0` |
| `local.best_of` | candidates per temperature fallback (Whisper) | `1` |
| `recording.max_buffer_seconds` | rolling buffer ceiling | `20` |
| `recording.min_chunk_seconds` / `buffer_trimming_seconds` | cadence and trimming, tuned for Parakeet | `2.0` / `10` |

With `engine: whisper`, go back to `min_chunk_seconds: 3.0`,
`buffer_trimming_seconds: 5` and `max_buffer_seconds: 12`.

#### Known limitations

- An occasional duplicate at window joins, mostly at session start.
- Glued compound names sometimes come out as two words ("Whisper Writer").
- `scripts/test_streaming_asr.py` checks the mechanics (14 checks), not
  transcription quality.

## Getting Started

### Prerequisites
Before you can run this app, you'll need to have the following software installed:

- Git: [https://git-scm.com/downloads](https://git-scm.com/downloads)
- Python `3.11`: [https://www.python.org/downloads/](https://www.python.org/downloads/)

If you want to run `faster-whisper` on your GPU, you'll also need to install the following NVIDIA libraries:

- [cuBLAS for CUDA 12](https://developer.nvidia.com/cublas)
- [cuDNN 8 for CUDA 12](https://developer.nvidia.com/cudnn)

<details>
<summary>More information on GPU execution</summary>

The below was taken directly from the [`faster-whisper` README](https://github.com/SYSTRAN/faster-whisper?tab=readme-ov-file#gpu):

**Note:** The latest versions of `ctranslate2` support CUDA 12 only. For CUDA 11, the current workaround is downgrading to the `3.24.0` version of `ctranslate2` (This can be done with `pip install --force-reinsall ctranslate2==3.24.0`).

There are multiple ways to install the NVIDIA libraries mentioned above. The recommended way is described in the official NVIDIA documentation, but we also suggest other installation methods below.

#### Use Docker

The libraries (cuBLAS, cuDNN) are installed in these official NVIDIA CUDA Docker images: `nvidia/cuda:12.0.0-runtime-ubuntu20.04` or `nvidia/cuda:12.0.0-runtime-ubuntu22.04`.

#### Install with `pip` (Linux only)

On Linux these libraries can be installed with `pip`. Note that `LD_LIBRARY_PATH` must be set before launching Python.

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12

export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

**Note**: Version 9+ of `nvidia-cudnn-cu12` appears to cause issues due its reliance on cuDNN 9 (Faster-Whisper does not currently support cuDNN 9). Ensure your version of the Python package is for cuDNN 8.

#### Download the libraries from Purfview's repository (Windows & Linux)

Purfview's [whisper-standalone-win](https://github.com/Purfview/whisper-standalone-win) provides the required NVIDIA libraries for Windows & Linux in a [single archive](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs). Decompress the archive and place the libraries in a directory included in the `PATH`.

</details>

### Installation
To set up and run the project, follow these steps:

#### 1. Clone the repository:

```
git clone https://github.com/savbell/whisper-writer
cd whisper-writer
```

#### 2. Create a virtual environment and activate it:

```
python -m venv venv

# For Linux and macOS:
source venv/bin/activate

# For Windows:
venv\Scripts\activate
```

#### 3. Install the required packages:

```
pip install -r requirements.txt
```

#### 4. Run the Python code:

```
python run.py
```

#### 5. Configure and start WhisperWriter:
On first run, a Settings window should appear. Once configured and saved, another window will open. Press "Start" to activate the keyboard listener. Press the activation key (`ctrl+shift+space` by default) to start recording and transcribing to the active window.

### Configuration Options

WhisperWriter uses a configuration file to customize its behaviour. To set up the configuration, open the Settings window:

<p align="center">
    <img src="./assets/ww-settings-demo.gif" alt="WhisperWriter Settings window demo gif" width="350" height="350">
</p>

#### Model Options
- `use_api`: Toggle to choose whether to use the OpenAI API or a local Whisper model for transcription. (Default: `false`)
- `common`: Options common to both API and local models.
  - `language`: The language code for the transcription in [ISO-639-1 format](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes). (Default: `null`)
  - `temperature`: Controls the randomness of the transcription output. Lower values make the output more focused and deterministic. (Default: `0.0`)
  - `initial_prompt`: A string used as an initial prompt to condition the transcription. More info: [OpenAI Prompting Guide](https://platform.openai.com/docs/guides/speech-to-text/prompting). (Default: `null`)

- `api`: Configuration options for the OpenAI API. See the [OpenAI API documentation](https://platform.openai.com/docs/api-reference/audio/create?lang=python) for more information.
  - `model`: The model to use for transcription. Currently, only `whisper-1` is available. (Default: `whisper-1`)
  - `base_url`: The base URL for the API. Can be changed to use a local API endpoint, such as [LocalAI](https://localai.io/). (Default: `https://api.openai.com/v1`)
  - `api_key`: Your API key for the OpenAI API. Required for non-local API usage. (Default: `null`)

- `local`: Configuration options for the local Whisper model.
  - `model`: The model to use for transcription. The larger models provide better accuracy but are slower. See [available models and languages](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages). (Default: `base`)
  - `device`: The device to run the local Whisper model on. Use `cuda` for NVIDIA GPUs, `cpu` for CPU-only processing, or `auto` to let the system automatically choose the best available device. (Default: `auto`)
  - `compute_type`: The compute type to use for the local Whisper model. [More information on quantization here](https://opennmt.net/CTranslate2/quantization.html). (Default: `default`)
  - `condition_on_previous_text`: Set to `true` to use the previously transcribed text as a prompt for the next transcription request. (Default: `true`)
  - `vad_filter`: Set to `true` to use [a voice activity detection (VAD) filter](https://github.com/snakers4/silero-vad) to remove silence from the recording. (Default: `false`)
  - `model_path`: The path to the local Whisper model. If not specified, the default model will be downloaded. (Default: `null`)

#### Recording Options
- `activation_key`: The keyboard shortcut to activate the recording and transcribing process. Separate keys with a `+`. (Default: `ctrl+shift+space`)
- `input_backend`: The input backend to use for detecting key presses. `auto` will try to use the best available backend. (Default: `auto`)
- `recording_mode`: The recording mode to use. Options include `continuous` (auto-restart recording after pause in speech until activation key is pressed again), `voice_activity_detection` (stop recording after pause in speech), `press_to_toggle` (stop recording when activation key is pressed again), `hold_to_record` (stop recording when activation key is released). (Default: `continuous`)
- `sound_device`: The numeric index of the sound device to use for recording. To find device numbers, run `python -m sounddevice`. (Default: `null`)
- `sample_rate`: The sample rate in Hz to use for recording. (Default: `16000`)
- `silence_duration`: The duration in milliseconds to wait for silence before stopping the recording. (Default: `900`)
- `min_duration`: The minimum duration in milliseconds for a recording to be processed. Recordings shorter than this will be discarded. (Default: `100`)

#### Post-processing Options
- `writing_key_press_delay`: The delay in seconds between each key press when writing the transcribed text. (Default: `0.005`)
- `remove_trailing_period`: Set to `true` to remove the trailing period from the transcribed text. (Default: `false`)
- `add_trailing_space`: Set to `true` to add a space to the end of the transcribed text. (Default: `true`)
- `remove_capitalization`: Set to `true` to convert the transcribed text to lowercase. (Default: `false`)
- `input_method`: The method to use for simulating keyboard input. (Default: `pynput`)

#### Miscellaneous Options
- `print_to_terminal`: Set to `true` to print the script status and transcribed text to the terminal. (Default: `true`)
- `hide_status_window`: Set to `true` to hide the status window during operation. (Default: `false`)
- `noise_on_completion`: Set to `true` to play a noise after the transcription has been typed out. (Default: `false`)

If any of the configuration options are invalid or not provided, the program will use the default values.

## Known Issues

You can see all reported issues and their current status in our [Issue Tracker](https://github.com/savbell/whisper-writer/issues). If you encounter a problem, please [open a new issue](https://github.com/savbell/whisper-writer/issues/new) with a detailed description and reproduction steps, if possible.

## Roadmap
Below are features I am planning to add in the near future:
- [x] Restructuring configuration options to reduce redundancy
- [x] Update to use the latest version of the OpenAI API
- [ ] Additional post-processing options:
  - [ ] Simple word replacement (e.g. "gonna" -> "going to" or "smiley face" -> "😊")
  - [ ] Using GPT for instructional post-processing
- [x] Updating GUI
- [ ] Creating standalone executable file

Below are features not currently planned:
- [ ] Pipelining audio files

Implemented features can be found in the [CHANGELOG](CHANGELOG.md).

## Contributing

Contributions are welcome! I created this project for my own personal use and didn't expect it to get much attention, so I haven't put much effort into testing or making it easy for others to contribute. If you have ideas or suggestions, feel free to [open a pull request](https://github.com/savbell/whisper-writer/pulls) or [create a new issue](https://github.com/savbell/whisper-writer/issues/new). I'll do my best to review and respond as time allows.

## Credits

- [OpenAI](https://openai.com/) for creating the Whisper model and providing the API. Plus [ChatGPT](https://chat.openai.com/), which was used to write a lot of the initial code for this project.
- [Guillaume Klein](https://github.com/guillaumekln) for creating the [faster-whisper Python package](https://github.com/SYSTRAN/faster-whisper).
- All of our [contributors](https://github.com/savbell/whisper-writer/graphs/contributors)!

## License

This project is licensed under the GNU General Public License. See the [LICENSE](LICENSE) file for details.
