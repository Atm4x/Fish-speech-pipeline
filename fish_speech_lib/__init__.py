from fish_speech_lib.inference import *
from fish_speech_lib.config import FIREFLY_GAN_VQ_CONFIG 
import os
os.environ["HYDRA_FULL_ERROR"] = "1"

from fish_speech_lib.fish_speech.models.vqgan.modules.firefly import FireflyArchitecture


__all__ = ["FishSpeech"]
__version__ = "0.1.0"  # Указывай версию