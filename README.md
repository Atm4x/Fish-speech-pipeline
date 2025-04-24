# **Fish-Speech-Lib** 0.1.0

Original project: [Fish-Speech](https://github.com/fishaudio/fish-speech)
<hr/>

***Fish-Speech-Lib*** is a Python library that provides a simple interface to the Fish-Speech pipeline, allowing you to generate high-quality speech with voice cloning capabilities without requiring a web UI.

Pytorch with CUDA or MPS is required to get Fish-Speech-Lib working.

**It may contain bugs. Report an issue in case of error.**

## Prerequisites

You must have **Python>=3.10** installed.

You must have **CUDA or MPS** support for your GPU (MPS is not fully tested yet).

## **Installation**
1) Install pytorch **with CUDA or MPS support** here: https://pytorch.org/get-started/locally/

2) Then, install Fish-Speech-Lib using pip install:
```
pip install git+https://github.com/Atm4x/Fish-speech-pipeline#egg=fish_speech_lib
```

3) Finally, create a `.project-root` file in the root directory of your project.

## Usage

Fish-Speech-Lib provides a class called `FishSpeech`. There are a few parameters that are optional:

`device` - Device to run on: "cuda" (default), "cpu", or "mps"

`half` - Whether to use half-precision (FP16) (default is False)

`compile_model` - Whether to use torch.compile for optimization (Not tested, needs to have CUDA toolkit installed)(default is False)

`llama_checkpoint_path` - Path to LLaMA model (default is "checkpoints/fish-speech-1.5")

`decoder_checkpoint_path` - Path to decoder model (default is "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth")

`streaming` - Enable streaming mode (DOESN'T WORK RN) (default is False)

To use the model, first make an instance of FishSpeech:

```python
from fish_speech_lib.inference import FishSpeech
import soundfile as sf

# Initialize model
tts = FishSpeech(
    device="cuda",
    half=False,
    compile_model=False
)
```

And the final step is calling the model to generate speech:

```python 
sample_rate, audio_data = tts(text="Hello, world!", max_new_tokens=450)
sf.write("output.wav", audio_data, sample_rate, format='WAV')
```

Parameters for the `tts()` function:

`text` - Text to be synthesized (required)

`reference_audio` - Path to reference audio for voice cloning (optional, default is None)

`reference_audio_text` - Text spoken in the reference audio (optional, default is "")

`top_p` - Top-p sampling parameter (optional, default is 0.7)

`temperature` - Temperature for sampling (optional, default is 0.7)

`repetition_penalty` - Repetition penalty (optional, default is 1.2)

`max_new_tokens` - Maximum number of tokens to generate (optional, default is 1024)

`chunk_length` - Length of iterative prompt in words (optional, default is 200)

`seed` - Random seed for reproducibility (optional, default is None)

`use_memory_cache` - Use memory cache for reference audio (optional, default is True)

## Example of usage
A simple example for generating speech:

```python
from fish_speech_lib.inference import FishSpeech
import soundfile as sf

# Initialize model
tts = FishSpeech(device="cuda")

# Generate speech
sample_rate, audio_data = tts(text="Hello, world!", max_new_tokens=450)

# Save the audio
sf.write("output.wav", audio_data, sample_rate, format='WAV')
```

## Voice Cloning Example

```python
# Generate speech with voice cloning
sample_rate, audio_data = tts(
    "This is an example of voice cloning with Fish-Speech.",
    reference_audio="path/to/reference.wav",
    reference_audio_text="The text that is spoken in the reference audio.",
    max_new_tokens=1000,
    chunk_length=1000
)

sf.write("cloned_voice.wav", audio_data, sample_rate, format='WAV')
```

## Model Downloading

The library automatically downloads the required models from the Hugging Face Hub if they are not found locally. The models are downloaded to the specified checkpoint paths.

## Exceptions

No exceptions found rn, but if you encounter any, please report them.

## License

The code within this repository (`Fish-Speech-Lib`) is licensed under the **Apache License 2.0**. You can find a copy of the license in the `LICENSE` file.

**IMPORTANT NOTE ON MODEL USAGE:**
The pre-trained models automatically downloaded and utilized by this library originate from the Fish-Speech project and are licensed separately under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.

*   **This means the models CANNOT be used for any commercial purposes.**
*   Please review the full CC BY-NC-SA 4.0 license terms here: [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Disclaimer

Users of this library are solely responsible for ensuring their usage complies with all applicable laws and ethical standards. Do not use this tool for illegal or harmful purposes. The developers of this fork are not liable for any misuse.

## Copyright

*   Original Work (Fish-Speech): Copyright (c) 2024 Fish Audio Authors
*   Modifications in this Fork (Fish-Speech-Lib): Copyright (c) 2025 Atm4x

## Authors
[Atm4x](https://github.com/Atm4x)

Based on [Fish-Speech](https://github.com/fishaudio/fish-speech)