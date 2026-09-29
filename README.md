# Fish-Speech-Lib

Fish-Speech-Lib provides a Python interface to the Fish Speech inference pipeline for text-to-speech and voice cloning, without requiring the Fish Speech web UI.

Upstream project: [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)

## Requirements

- Python 3.10 or newer.
- PyTorch installed separately for the accelerator you intend to use.
- CUDA is recommended for NVIDIA GPUs; CPU and MPS are supported by the library's device routing, although performance and feature support vary by backend.

## Installation

Install PyTorch for your platform first, then install the library:

```bash
pip install fish_speech_lib
```

When running directly from the source tree, add the repository root to `PYTHONPATH` or `sys.path`. A `.project-root` marker is not required.

## Basic usage

```python
from fish_speech_lib import FishSpeech
import soundfile as sf

with FishSpeech(device="cuda", compile_model=False) as tts:
    sample_rate, audio = tts(
        text="Hello, world!",
        max_new_tokens=450,
    )

sf.write("output.wav", audio, sample_rate)
```

### Device selection

`device` accepts normal PyTorch device strings. In a multi-GPU process, pass an explicit CUDA index when placement matters:

```python
tts = FishSpeech(device="cuda:1")
```

A bare `"cuda"` selects the first visible CUDA device (`cuda:0`). Explicit indices such as `"cuda:1"` are preserved throughout the library's model, cache, sampling, and decoder paths.

## Voice cloning

```python
sample_rate, audio = tts(
    text="This is an example of voice cloning.",
    reference_audio="path/to/reference.wav",
    reference_audio_text="The text spoken in the reference audio.",
    max_new_tokens=1000,
)
```

## `torch.compile`

Set `compile_model=True` to compile the token decode path with `torch.compile`. Compilation is lazy, so the first inference can take substantially longer than later calls. CUDA compilation is invoked inside the selected model device context; the library does not remap `CUDA_VISIBLE_DEVICES` or install process-wide Torch/Inductor monkey patches.

## Model downloads

If the default Fish Speech 1.5 checkpoints are missing, the library downloads them from `fishaudio/fish-speech-1.5` on Hugging Face.

## License

The source code in this repository is distributed under the Apache License 2.0. See [`LICENSE`](LICENSE). Files derived from Fish Speech retain Fish Audio attribution, and files modified by this project carry modification notices.

The Fish Speech 1.5 model weights downloaded from Hugging Face are separate assets and are licensed separately under CC BY-NC-SA 4.0. The repository's Apache-2.0 license does not relicense those model weights.

## Copyright

- Original Fish Speech code: Copyright 2024-2025 Fish Audio Authors.
- Fish-Speech-Lib modifications: Copyright 2025-2026 Atm4x.
