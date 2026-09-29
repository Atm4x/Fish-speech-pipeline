# Fish-Speech-Lib 0.1.1

A Python interface to the Fish Speech text-to-speech and voice-cloning pipeline.
Use it from your application without running the Fish Speech web UI.

Upstream project: [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech).

## Release notes

### 0.1.1 — September 30, 2026

- **MultiGPU support:** run independent `FishSpeech` instances on different GPUs
  in the same process by selecting `cuda:0`, `cuda:1`, and so on.
- Model tensors, KV caches, sampling and decoder operations follow the selected
  device. Each instance owns its request queue and worker lifecycle.
- `torch.compile` runs inside the selected CUDA device context, including lazy
  compilation and CUDA graph replay.
- Request-local sampling accepts `seed` without changing the process-wide Torch RNG.
- Added `close()` and context-manager support, with cleanup if initialization fails.
- Removed source-tree root setup and import-time environment changes.
- Device setup diagnostics use a compact JSON snapshot in the system temp folder.
- Added `pyproject.toml` release tooling and CI for Python 3.10–3.12 on Windows and Linux.

### 0.1.0.1 — April 24, 2025

Initial PyPI package for the Fish Speech inference wrapper.

## Requirements

- Python 3.10 or newer. CI covers Python 3.10, 3.11 and 3.12.
- PyTorch and matching torchaudio installed for your accelerator.
- NVIDIA CUDA is recommended for GPU inference. CPU and MPS device routing are
  also available; performance and compilation support depend on the backend.
- Enough memory on each GPU for its own text-semantic model, decoder and caches.

## Installation

Install PyTorch and torchaudio for your platform using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/), then:

```bash
pip install fish-speech-lib
```

To install the current release branch before its PyPI publication:

```bash
pip install "git+https://github.com/Atm4x/Fish-speech-pipeline.git@releases"
```

For local development, run `pip install -e .` from the repository root.

## Basic usage

```python
import soundfile as sf
from fish_speech_lib import FishSpeech

with FishSpeech(device="cuda:0", compile_model=False) as tts:
    sample_rate, audio = tts(text="Hello, world!", max_new_tokens=450)
    sf.write("output.wav", audio, sample_rate)
```

The result is `(sample_rate, audio)`, with audio returned as a NumPy array.
Use `close()` when managing the instance without a `with` block.

## MultiGPU

Each instance uses one device. To distribute work across multiple GPUs, create
one instance per GPU and schedule requests in your application:

```python
from concurrent.futures import ThreadPoolExecutor
import soundfile as sf
from fish_speech_lib import FishSpeech

with FishSpeech(device="cuda:0") as first, FishSpeech(device="cuda:1") as second:
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(first, text="First request.", max_new_tokens=450),
            pool.submit(second, text="Second request.", max_new_tokens=450),
        ]
        for index, job in enumerate(jobs):
            sample_rate, audio = job.result()
            sf.write(f"output-{index}.wav", audio, sample_rate)
```

Use explicit device indices to control placement. A bare `"cuda"` selects the
first visible GPU, `cuda:0`. Indices refer to the CUDA devices visible to your
process. Each GPU needs memory for a complete instance; request scheduling is
handled by the calling application.

An unavailable explicit GPU raises an error. Bare `"cuda"` or `"mps"` can fall
back to CPU when that accelerator is unavailable.

## Voice cloning

```python
from fish_speech_lib import FishSpeech

with FishSpeech(device="cuda:0") as tts:
    sample_rate, audio = tts(
        text="This is an example of voice cloning.",
        reference_audio="reference.wav",
        reference_audio_text="The text spoken in the reference audio.",
        max_new_tokens=1000,
        seed=123,
    )
```

## Configuration

| Constructor option | Default | Purpose |
|---|---|---|
| `device` | `"cuda"` | `"cuda:N"`, `"cpu"` or `"mps"` |
| `half` | `False` | Use half precision for the text-semantic model |
| `compile_model` | `False` | Compile the token decode path |
| `llama_checkpoint_path` | `"checkpoints/fish-speech-1.5"` | Text-semantic checkpoint directory |
| `decoder_checkpoint_path` | `"checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"` | Decoder checkpoint |

Generation options include `top_p=0.7`, `temperature=0.7`,
`repetition_penalty=1.2`, `max_new_tokens=1024`, `chunk_length=200`,
`seed=None` and `use_memory_cache=True`.

## Compilation and diagnostics

Set `compile_model=True` to use `torch.compile`. Initialization warms up the
compiled decode path; the initial call can take substantially longer than later
inference. Keep model instances alive to reuse their compiled paths and caches.

When a compiler is configured, a compact snapshot is saved to
`fish_speech_devices_<PID>.json` in `tempfile.gettempdir()` (`%TEMP%` on Windows).
It stores the latest compiler setup per device and is replaced atomically.
The snapshot is written during setup, with no per-token event log or console echo.

## Model downloads

If the default Fish Speech 1.5 checkpoints are missing, the library downloads
them from `fishaudio/fish-speech-1.5` on Hugging Face.

## Development and releases

CI runs tests on Windows/Linux with Python 3.10–3.12 and builds wheel + sdist.
The `releases` branch contains release candidates. Publishing is controlled by
the repository variable `PYPI_PUBLISH_ENABLED`.
See [release setup and version commands](.github/RELEASING.md).

## License

Source code is distributed under [Apache License 2.0](LICENSE).
See [NOTICE](NOTICE) for attribution to Fish Audio and this project.

Downloaded Fish Speech 1.5 model weights are separate assets under
CC BY-NC-SA 4.0. The source license does not change the weights' license.

Original Fish Speech code: Copyright 2024–2025 Fish Audio Authors.
Fish-Speech-Lib modifications: Copyright 2025–2026 Atm4x.
