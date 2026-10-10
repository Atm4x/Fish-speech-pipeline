# Copyright 2025-2026 Atm4x
# SPDX-License-Identifier: Apache-2.0

from .config import FIREFLY_GAN_VQ_CONFIG
from .inference import FishSpeech
from .fish_speech.models.vqgan.modules.firefly import FireflyArchitecture

__all__ = ["FishSpeech", "FIREFLY_GAN_VQ_CONFIG", "FireflyArchitecture"]
__version__ = "0.1.1.1"
