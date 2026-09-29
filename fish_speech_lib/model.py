# Copyright 2025-2026 Atm4x
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from .fish_speech.device import DeviceLike, resolve_device
from .fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from .fish_speech.models.vqgan.inference import load_model as load_vqgan_model


class TTSConfigLike(Protocol):
    device: DeviceLike
    half: bool
    compile: bool
    llama_checkpoint_path: str | Path | None
    decoder_checkpoint_path: str | Path | None


class ModelManager:
    """Load and own the text-semantic worker and VQ-GAN decoder."""

    def __init__(self, config: TTSConfigLike):
        self.config = config
        self.vqgan_model = None
        self.llama_queue = None
        self.device = resolve_device(config.device, fallback_to_cpu=True)
        self.precision = torch.float16 if config.half else torch.bfloat16
        self.load_models()

    def load_models(self) -> None:
        """Load the models configured for this manager."""
        if (
            self.config.llama_checkpoint_path is None
            or self.config.decoder_checkpoint_path is None
        ):
            raise ValueError(
                "Both llama_checkpoint_path and decoder_checkpoint_path must be provided."
            )

        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.config.llama_checkpoint_path,
            device=self.device,
            precision=self.precision,
            compile=self.config.compile,
        )
        self.vqgan_model = load_vqgan_model(
            checkpoint_path=self.config.decoder_checkpoint_path,
            device=self.device,
        )
        self.vqgan_model.eval()

    def get_models(self):
        """Return the loaded text-semantic worker and VQ-GAN model."""
        if self.llama_queue is None or self.vqgan_model is None:
            raise RuntimeError("Models have not been loaded. Call load_models() first.")
        return self.llama_queue, self.vqgan_model
