# Copyright 2025-2026 Atm4x
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from .fish_speech.device import DeviceLike, resolve_device
from .fish_speech.models.text2semantic import inference as text2semantic_inference
from .fish_speech.models.vqgan.inference import load_model as load_vqgan
from .fish_speech.text.clean import clean_text


def load_models(
    llama_checkpoint_path: str,
    decoder_checkpoint_path: str,
    decoder_config_name: str | None,
    device: DeviceLike,
    precision: torch.dtype,
    compile_model: bool,
):
    """Load the text-semantic model and VQ-GAN decoder on one explicit device.

    ``decoder_config_name`` is retained for backward compatibility; the bundled
    Fish Speech 1.5 decoder uses the library's built-in Firefly configuration.
    """
    del decoder_config_name
    resolved = resolve_device(device, fallback_to_cpu=True)

    llama_model, decode_one_token = text2semantic_inference.load_model(
        llama_checkpoint_path,
        resolved,
        precision,
        compile=compile_model,
        is_agent=False,
    )
    llama_model.setup_caches(
        max_batch_size=1,
        max_seq_len=llama_model.config.max_seq_len,
        dtype=next(llama_model.parameters()).dtype,
    )

    tokenizer = llama_model.tokenizer
    config = llama_model.config

    vqgan_model = load_vqgan(decoder_checkpoint_path, device=resolved)
    if precision == torch.float16:
        vqgan_model = vqgan_model.half()
    vqgan_model.eval()

    return llama_model, decode_one_token, tokenizer, config, vqgan_model


def process_text(text: str) -> str:
    """Clean and normalize text using the bundled Fish Speech text pipeline."""
    return clean_text(text)
