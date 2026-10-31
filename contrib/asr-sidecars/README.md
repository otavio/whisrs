# ASR sidecars

`whisrs` can call any local HTTP ASR service through the generic
`asr-sidecar` backend. Sidecars accept WAV audio as multipart form data and
return a JSON response with a `text` field.

## Contract

Endpoint:

```text
POST /transcribe
```

Multipart fields:

- `file`: WAV audio
- `model`: model identifier
- `language`: ISO 639-1 language code; left out when the language is `auto`
- `hotwords`: optional vocabulary/context prompt

Response:

```json
{ "text": "transcribed text" }
```

This is the OpenAI `/v1/audio/transcriptions` shape, so an OpenAI-compatible
transcription server needs no wrapper: point `[asr-sidecar] url` at its full
URL. The path is whatever you configure, `/transcribe` is only the default.
The one divergence is the prompt, which whisrs sends as `hotwords` rather than
`prompt`, so a strict OpenAI-compatible server ignores `[general] prompt`.

## Implementations

- [`moonshine`](moonshine/): Useful Sensors Moonshine in a Python server
  (`server.py`), English only.
- [`nemo-speech`](nemo-speech/): NVIDIA Nemotron 3.5 ASR Streaming and
  Parakeet TDT v3 in `nemo-speech serve`, one prebuilt binary with no Python.
- [`parakeet`](parakeet/): NVIDIA Parakeet TDT v3 through NeMo in a Python
  server (`server.py`).
- [`vibevoice`](vibevoice/): Microsoft VibeVoice-ASR through Transformers in a
  Python server (`server.py`).

## Recommended choices

- `moonshine`: fastest lightweight CPU/low-memory English sidecar.
- `nemo-speech`: local CPU sidecar with no Python (Nemotron for live
  dictation and recordings, Parakeet for batch dictation).
- `parakeet`: recommended local GPU sidecar for high-quality multilingual
  dictation.
- `vibevoice`: heavier long-form transcription sidecar.

| Sidecar | Default model | Best for |
|---|---|---|
| `moonshine` | `UsefulSensors/moonshine-base` | Fast lightweight English dictation |
| `nemo-speech` | the loaded model (`model = ""`) | Live and batch dictation and recordings on CPU, no Python |
| `parakeet` | `nvidia/parakeet-tdt-0.6b-v3` | High-quality local GPU dictation |
| `vibevoice` | `microsoft/VibeVoice-ASR-HF` | Long-form local transcription experiments |

Each sidecar has its own README with installation and GPU notes and the
whisrs config; the Python ones also ship it as a `config.toml.example`.

## AMD ROCm note

For AMD/ROCm, PyTorch still uses `cuda:0` device spelling, but NVIDIA CUDA
driver libraries are not present. NeMo's CUDA graph decoder expects
`libcuda.so.1`, so the Parakeet sidecar disables that decoder by default. Do
not pass `--use-cuda-graph-decoder` on ROCm.
