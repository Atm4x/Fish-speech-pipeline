# Copyright 2025 Atm4x (Apache License 2.0)
# See LICENSE file for details.

import torch
from pathlib import Path
from typing import Union, Optional
from contextlib import nullcontext

import torchaudio
import librosa
import numpy as np
import io

from .fish_speech.models.text2semantic import inference as text2semantic_inference
from .fish_speech.text.clean import clean_text
from .fish_speech.utils.file import audio_to_bytes
from .fish_speech.tokenizer import IM_END_TOKEN
from .fish_speech.conversation import (
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from huggingface_hub import hf_hub_download

def load_models(
    llama_checkpoint_path: str,
    decoder_checkpoint_path: str,
    decoder_config_name: str,
    device: str,
    precision: torch.dtype,
    compile_model: bool,
):
    """Загружает модели LLAMA и VQ-GAN."""

    llama_model, decode_one_token = text2semantic_inference.load_model(
        llama_checkpoint_path,
        device,
        precision,
        compile=compile_model,
        is_agent=False,  # Важно: is_agent=False для обычной модели
    )
    # We set max_seq_len to a large value to avoid having the model re-compiled
    # each time the length changes.  We have key/value cache,
    # the actual sequence length will be at most max_length in the config.
    with torch.device(device):
        llama_model.setup_caches(
            max_batch_size=1,
            max_seq_len=llama_model.config.max_seq_len,
            dtype=next(llama_model.parameters()).dtype,
        )

    tokenizer = llama_model.tokenizer
    config = llama_model.config

    # Загрузка VQGAN модели
    from fish_speech.models.vqgan.inference import load_model as load_vqgan

    vqgan_model = load_vqgan(decoder_config_name, decoder_checkpoint_path, device=device)

    if precision == torch.half:
        vqgan_model = vqgan_model.half()  # Применяем half() к VQGAN
    vqgan_model.eval()


    return llama_model, decode_one_token, tokenizer, config, vqgan_model  # Возвращаем vqgan


def process_text(text: str) -> str:
    """Очищает и нормализует текст."""
    return clean_text(text)