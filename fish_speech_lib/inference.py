# Copyright 2025 Atm4x (Apache License 2.0)
# See LICENSE file for details.

import torch
from pathlib import Path
from typing import Union, Optional
import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from .fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
from .fish_speech.utils.file import audio_to_bytes

class FishSpeech15:
    """Пайплайн для моделей версии 1.5 (Firefly + Tiktoken)"""
    def __init__(
        self,
        device: str = "cuda",
        half: bool = False,
        compile_model: bool = False,
        llama_checkpoint_path: str = "checkpoints/fish-speech-1.5",
        decoder_checkpoint_path: str = "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        streaming: bool = False,
    ):
        self.device = self._resolve_device(device)
        self.precision = torch.half if half else torch.bfloat16
        self.compile_model = compile_model
        self.streaming = streaming
        self.llama_checkpoint_path = llama_checkpoint_path
        self.decoder_checkpoint_path = decoder_checkpoint_path

        self._load_or_download_models()

        from .fish_speech.inference_engine import TTSInferenceEngine
        from .fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from .fish_speech.models.vqgan.inference import load_model as load_decoder_model

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.llama_checkpoint_path,
            device=self.device,
            precision=self.precision,
            compile=self.compile_model,
        )
        decoder_model = load_decoder_model(
            checkpoint_path=self.decoder_checkpoint_path, device=self.device
        )
        self.engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=self.compile_model,
        )

    def _download_models(self):
        repo_id = "fishaudio/fish-speech-1.5"
        model_files = ["model.pth", "tokenizer.tiktoken", "config.json", "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"]
        for file in model_files:
            file_path = Path(self.llama_checkpoint_path) / file
            if not file_path.exists():
                hf_hub_download(repo_id=repo_id, filename=file, local_dir=self.llama_checkpoint_path, local_dir_use_symlinks=False)

    def _load_or_download_models(self):
        local_llama_path = Path(self.llama_checkpoint_path)
        if local_llama_path.exists() and (local_llama_path / "model.pth").exists():
            return
        self._download_models()

    def _resolve_device(self, device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available(): return "cpu"
        if device == "mps" and not torch.backends.mps.is_available(): return "cpu"
        return device

    @torch.no_grad()
    def __call__(
        self, text: str, reference_audio: Union[str, Path, bytes, None] = None, reference_audio_text: str = "", 
        *, top_p: float = 0.7, temperature: float = 0.7, repetition_penalty: float = 1.2, 
        max_new_tokens: int = 1024, chunk_length: int = 200, seed: Optional[int] = None, use_memory_cache: bool = True
    ) -> tuple[int, np.ndarray]:
        
        references = []
        if reference_audio:
            audio_bytes = audio_to_bytes(reference_audio) if not isinstance(reference_audio, bytes) else reference_audio
            references.append(ServeReferenceAudio(audio=audio_bytes, text=reference_audio_text))
            
        request = ServeTTSRequest(
            text=text, references=references, max_new_tokens=max_new_tokens, chunk_length=chunk_length, 
            top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature, seed=seed, 
            streaming=False, normalize=True, use_memory_cache="on" if use_memory_cache else "off"
        )

        for result in self.engine.inference(request):
            if result.code == "final":
                return result.audio
            elif result.code == "error":
                raise result.error
        raise RuntimeError("Failed to generate audio.")

class FishSpeechS2:
    """Пайплайн для моделей архитектуры S1/S2 (DAC + Qwen Tokenizer)"""
    def __init__(
        self,
        device: str = "cuda",
        half: bool = True,  # Рекомендуется True для S2
        compile_model: bool = False,
        llama_checkpoint_path: str = "checkpoints/s2-pro", 
        decoder_checkpoint_path: str = "checkpoints/s2-pro/codec.pth",
    ):
        self.device = self._resolve_device(device)
        self.precision = torch.half if half else torch.bfloat16
        self.compile_model = compile_model
        self.llama_checkpoint_path = llama_checkpoint_path
        self.decoder_checkpoint_path = decoder_checkpoint_path

        self._load_or_download_models()

        from .fish_speech.inference_engine.s1_s2 import TTSInferenceEngine as TTSInferenceEngineS2
        from .fish_speech.models.text2semantic.s1_s2.inference import launch_thread_safe_queue as launch_queue_s2
        from .fish_speech.models.dac.inference import load_model as load_dac_model

        llama_queue = launch_queue_s2(
            checkpoint_path=self.llama_checkpoint_path,
            device=self.device,
            precision=self.precision,
            compile=self.compile_model,
        )
        
        decoder_model = load_dac_model(
            config_name=None, 
            checkpoint_path=self.decoder_checkpoint_path, 
            device=self.device
        )
        
        self.engine = TTSInferenceEngineS2(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self.precision,
            compile=self.compile_model,
        )

    def _load_or_download_models(self):
        local_llama_path = Path(self.llama_checkpoint_path)
        if local_llama_path.exists() and (local_llama_path / "config.json").exists() and (local_llama_path / "codec.pth").exists():
            return
        
        print(f"S2 Models not found locally. Downloading from huggingface to {self.llama_checkpoint_path}...")
        snapshot_download(
            repo_id="fishaudio/s2-pro", 
            local_dir=self.llama_checkpoint_path, 
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", ".git*"]
        )

    def _resolve_device(self, device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available(): return "cpu"
        if device == "mps" and not torch.backends.mps.is_available(): return "cpu"
        return device

    @torch.no_grad()
    def __call__(
        self, text: str, reference_audio: Union[str, Path, bytes, None] = None, reference_audio_text: str = "", 
        *, top_p: float = 0.7, temperature: float = 0.7, repetition_penalty: float = 1.2, 
        max_new_tokens: int = 1024, chunk_length: int = 200, seed: Optional[int] = None, use_memory_cache: bool = True
    ) -> tuple[int, np.ndarray]:
        
        references = []
        if reference_audio:
            audio_bytes = audio_to_bytes(reference_audio) if not isinstance(reference_audio, bytes) else reference_audio
            references.append(ServeReferenceAudio(audio=audio_bytes, text=reference_audio_text))
        
        request = ServeTTSRequest(
            text=text, references=references, max_new_tokens=max_new_tokens, chunk_length=chunk_length, 
            top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature, seed=seed, 
            streaming=False, normalize=True, use_memory_cache="on" if use_memory_cache else "off"
        )

        for result in self.engine.inference(request):
            if result.code == "final":
                return result.audio
            elif result.code == "error":
                raise result.error
        raise RuntimeError("Failed to generate audio via S2.")

# Для обратной совместимости оставляем старое имя для 1.5
FishSpeech = FishSpeech15