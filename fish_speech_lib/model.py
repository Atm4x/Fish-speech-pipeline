import torch
from fish_speech.models.text2semantic.inference import (
    load_model as load_text2semantic_model,
    launch_thread_safe_queue
)
from fish_speech.models.vqgan.inference import load_model as load_vqgan_model
from .config import TTSConfig  # Import the config

class ModelManager:
    """
    Manages the loading and initialization of the text-to-semantic and VQ-GAN models.
    """

    def __init__(self, config: TTSConfig):
        self.config = config
        self.text2semantic_model = None
        self.vqgan_model = None
        self.llama_queue = None
        self.device = (
            "cuda"
            if torch.cuda.is_available() and config.device.startswith("cuda")
            else config.device
        )
        self.precision = torch.half if config.half else torch.bfloat16
        self.load_models()

    def load_models(self):
        """Loads the necessary models based on the configuration."""

        if self.config.llama_checkpoint_path is None or self.config.decoder_checkpoint_path is None:
            raise ValueError(
                "Both llama_checkpoint_path and decoder_checkpoint_path must be provided in the config."
            )

        self.llama_queue = launch_thread_safe_queue(
            checkpoint_path=self.config.llama_checkpoint_path,
            device=self.device,
            precision=self.precision,
            compile=self.config.compile,
        )

        self.vqgan_model = load_vqgan_model(
            config_name=self.config.decoder_config_name,
            checkpoint_path=self.config.decoder_checkpoint_path,
            device=self.device,
        )
        #Warm up
        self.vqgan_model.eval()


    def get_models(self):
        """Returns the loaded models."""
        if self.llama_queue is None or self.vqgan_model is None:
            raise RuntimeError("Models have not been loaded. Call load_models() first.")
        return self.llama_queue, self.vqgan_model