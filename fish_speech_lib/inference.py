import torch
from pathlib import Path
from typing import Union, Optional
import numpy as np

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.vqgan.inference import load_model as load_decoder_model
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
from fish_speech.utils.file import audio_to_bytes, read_ref_text
from huggingface_hub import hf_hub_download  # Импортируем

class FishSpeech:
    def __init__(
        self,
        device: str = "cuda",
        half: bool = False,
        compile_model: bool = False,
        llama_checkpoint_path: str = "checkpoints/fish-speech-1.5",  # Пути по умолчанию
        decoder_checkpoint_path: str = "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        decoder_config_name: str = "firefly_gan_vq",
        streaming: bool = False, # Добавили в init
    ):
        """
        Инициализирует модели FishSpeech.

        Args:
            device: "cuda" (по умолчанию), "cpu", или "mps".
            half: Использовать ли FP16 (half-precision).
            compile_model: Использовать ли torch.compile для ускорения.
            llama_checkpoint_path: Путь к чекпоинту LLAMA модели.
            decoder_checkpoint_path: Путь к чекпоинту декодера (VQ-GAN).
            decoder_config_name: Имя конфигурации декодера (из Hydra).
            streaming: Включить стриминг
        """
        self.device = self._resolve_device(device)
        self.precision = torch.half if half else torch.bfloat16
        self.compile_model = compile_model
        self.streaming = streaming
        self.llama_checkpoint_path = llama_checkpoint_path
        self.decoder_checkpoint_path = decoder_checkpoint_path

        # Проверяем наличие файлов и загружаем модели
        self._load_or_download_models()

        # Инициализация TTSInferenceEngine (уже после загрузки моделей)
        self.engine = self._initialize_engine(
            self.llama_checkpoint_path,
            self.decoder_checkpoint_path,
            decoder_config_name,
            self.device,
            self.precision,
            self.compile_model,
        )
    def _download_models(self):
      from huggingface_hub import hf_hub_download
      default_local_path = "checkpoints/fish-speech-1.5"

      # Hugging Face repo ID
      repo_id = "fishaudio/fish-speech-1.5"

      # Model files to download
      model_files = [
          "model.pth",
          "tokenizer.tiktoken",
          "config.json",
          "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
      ]

      # Download using the function
      for file in model_files:
        file_path = Path(default_local_path) / file
        if not file_path.exists():
          print(f"{file} does not exist, downloading from Hugging Face repository...")
          hf_hub_download(
          repo_id=repo_id,
          filename=file,
          local_dir=default_local_path,
          local_dir_use_symlinks=False,
        )
        else:
           print(f"{file} already exists, skipping download.")


    def _load_or_download_models(self):
        """Загружает модели из локальной директории или скачивает с HuggingFace Hub."""
        local_llama_path = Path(self.llama_checkpoint_path)
        local_decoder_path = Path(self.decoder_checkpoint_path)
        config_path = local_llama_path / "config.json"
        tokenizer_path = local_llama_path / "tokenizer.tiktoken"

        # Проверяем наличие основных файлов
        if (
            local_llama_path.exists()
            and local_decoder_path.exists()
            and config_path.exists()
            and tokenizer_path.exists()
        ):

            print("Loading models from local directory.")
            return # Модели уже есть

        print("Local models not found, downloading from Hugging Face Hub...")
        self._download_models()



    def _resolve_device(self, device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU.")
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, using CPU.")
            return "cpu"
        return device

    def _initialize_engine(
        self,
        llama_checkpoint_path,
        decoder_checkpoint_path,
        decoder_config_name,
        device,
        precision,
        compile_model,
    ):
        """Инициализирует и возвращает TTSInferenceEngine."""
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=llama_checkpoint_path,
            device=device,
            precision=precision,
            compile=compile_model,
        )
        decoder_model = load_decoder_model(
            config_name=decoder_config_name, checkpoint_path=decoder_checkpoint_path, device=device
        )
        if precision == torch.half:
            decoder_model = decoder_model.half()

        engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=compile_model,
        )

        # "Прогрев" модели.  Делаем это *один раз* при инициализации.
        engine.inference(
            ServeTTSRequest(
                text="test",
                references=[],
                max_new_tokens=10,  # Небольшое значение для прогрева
            )
        )
        return engine

    @torch.no_grad()
    def __call__(
        self,
        text: str,
        reference_audio: Union[str, Path, bytes, None] = None,
        reference_audio_text: str = "",
        *,
        top_p: float = 0.7,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 1024,
        chunk_length: int = 200,
        seed: Optional[int] = 42,  # Фиксированный seed по умолчанию
        use_memory_cache: bool = True,  # Добавили параметр use_memory_cache
    ) -> tuple[int, np.ndarray]:
        """
        Генерирует речь по тексту.

        Args:
            text: Текст для озвучивания.
            reference_audio: Путь к референсному аудио (str или Path),
                аудиофайл в байтах, или None.  Если указан путь,
                то reference_audio_text игнорируется.
            reference_audio_text: Текст для референсного аудио (используется,
                только если reference_audio -- bytes).
            top_p: Параметр top_p для сэмплирования.
            temperature: Параметр temperature для сэмплирования.
            repetition_penalty: Параметр repetition_penalty для сэмплирования.
            max_new_tokens: Максимальное количество генерируемых токенов.
            seed:  Seed для воспроизводимости.
            chunk_length: Длина итеративного промпта (слов).
            use_memory_cache: Использовать ли кэширование референсных аудио.

        Returns:
            (sample_rate, waveform) - кортеж с частотой дискретизации и numpy-массивом аудио.
        """

        # Создаем ServeTTSRequest
        if reference_audio is not None:
            if isinstance(reference_audio, (str, Path)):
                #  reference_audio это путь к файлу
                audio_bytes = audio_to_bytes(reference_audio)
                # Если есть .lab файл, читаем текст из него, иначе ""
                lab_file = Path(reference_audio).with_suffix(".lab")
                reference_text = read_ref_text(str(lab_file)) if lab_file.exists() else ""
                references = [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]
            elif isinstance(reference_audio, bytes):
                # reference_audio это байты, используем reference_audio_text
                references = [
                    ServeReferenceAudio(audio=reference_audio, text=reference_audio_text)
                ]
            else:
                raise TypeError(
                    "reference_audio must be str, Path, bytes, or None"
                )
        else:
            references = []


        request = ServeTTSRequest(
            text=text,
            references=references,  # Используем список references
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            streaming=self.streaming,  # Стриминг
            normalize=True,
            use_memory_cache="on" if use_memory_cache else "off", # Добавлено
            reference_id=None,  # reference_id  не используем, используем references
        )

        if seed is not None:
            torch.manual_seed(seed) # Применяем seed
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed(seed)

        # Вызываем inference у TTSInferenceEngine
        result_generator = self.engine.inference(request)
        final_result = None
        for result in result_generator:
            if result.code == "header":
                pass
            elif result.code == "segment":
                pass
            elif result.code == "final":
                final_result = result
                break
            elif result.code == "error":
                raise result.error

        if final_result is None or final_result.audio is None:
            raise RuntimeError("Failed to generate audio.")

        sample_rate, audio_data = final_result.audio
        return sample_rate, audio_data