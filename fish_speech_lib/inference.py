import torch
from pathlib import Path
from typing import Union, Optional
import numpy as np
import gc # Для сборки мусора
import torch._dynamo # Для сброса кэша
import threading # Для управления потоком
import queue # Для типа очереди
from loguru import logger # Для логирования

# Импортируем нужные компоненты
from .fish_speech.inference_engine import TTSInferenceEngine
# Убедимся, что импортируем модифицированную функцию
from .fish_speech.models.text2semantic.inference import launch_thread_safe_queue, GenerateRequest, WrappedGenerateResponse
from .fish_speech.models.vqgan.inference import load_model as load_decoder_model
from .fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
from .fish_speech.utils.file import audio_to_bytes, read_ref_text
from huggingface_hub import hf_hub_download

class FishSpeech:
    def __init__(
        self,
        device: str = "cuda",
        half: bool = False,
        compile_model: bool = False,
        llama_checkpoint_path: str = "checkpoints/fish-speech-1.5",
        decoder_checkpoint_path: str = "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        # streaming убран из init, так как он не используется для __call__
    ):
        """
        Инициализирует модели FishSpeech.

        Args:
            device: "cuda" (по умолчанию), "cpu", или "mps".
            half: Использовать ли FP16 (half-precision).
            compile_model: Использовать ли torch.compile при первой инициализации.
            llama_checkpoint_path: Путь к чекпоинту LLAMA модели.
            decoder_checkpoint_path: Путь к чекпоинту декодера (VQ-GAN).
        """
        self.device = self._resolve_device(device)
        self.half = half
        self.precision = torch.half if half else torch.bfloat16
        self._compile_model = compile_model # Внутреннее хранение настройки
        self.llama_checkpoint_path = llama_checkpoint_path
        self.decoder_checkpoint_path = decoder_checkpoint_path

        # Атрибуты для хранения компонентов (инициализируются в _setup_components)
        self.llama_queue: Optional[queue.Queue] = None
        self.llama_thread: Optional[threading.Thread] = None
        self.decoder_model = None
        self.engine: Optional[TTSInferenceEngine] = None

        logger.info(f"Initializing FishSpeech with compile_model={self._compile_model}")
        # Проверяем наличие файлов и загружаем модели, если нужно
        self._load_or_download_models()
        # Настраиваем компоненты
        self._setup_components()
        logger.info("FishSpeech initialized successfully.")


    def _download_models(self):
        # ... (код скачивания не изменился) ...
        from huggingface_hub import hf_hub_download
        default_local_path = "checkpoints/fish-speech-1.5"
        repo_id = "fishaudio/fish-speech-1.5"
        model_files = [
            "model.pth", "tokenizer.tiktoken", "config.json",
            "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
        ]
        for file in model_files:
            file_path = Path(default_local_path) / file
            if not file_path.exists():
                logger.info(f"Downloading {file} from Hugging Face repository...")
                hf_hub_download(
                    repo_id=repo_id, filename=file,
                    local_dir=default_local_path, local_dir_use_symlinks=False,
                )
            else:
                logger.info(f"{file} already exists, skipping download.")


    def _load_or_download_models(self):
        # ... (код проверки/скачивания не изменился) ...
        local_llama_path = Path(self.llama_checkpoint_path)
        local_decoder_path = Path(self.decoder_checkpoint_path)
        config_path = local_llama_path / "config.json"
        tokenizer_path = local_llama_path / "tokenizer.tiktoken"
        if (local_llama_path.exists() and local_decoder_path.exists() and
                config_path.exists() and tokenizer_path.exists()):
            logger.info("Found models in local directory.")
            return
        logger.info("Local models not found or incomplete.")
        self._download_models()

    def _resolve_device(self, device: str) -> str:
        # ... (код не изменился) ...
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU.")
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU.")
            return "cpu"
        logger.info(f"Using device: {device}")
        return device

    def _setup_components(self):
        """Инициализирует очередь LLAMA, поток, декодер и движок TTS."""
        logger.info(f"Setting up components with compile_model={self._compile_model}...")

        # 1. Запускаем поток LLAMA
        try:
            self.llama_queue, self.llama_thread = launch_thread_safe_queue(
                checkpoint_path=self.llama_checkpoint_path,
                device=self.device,
                precision=self.precision,
                compile=self._compile_model, # Используем текущую настройку
            )
            # Проверяем, жив ли поток после инициализации (на случай ошибки внутри worker)
            if not self.llama_thread.is_alive():
                 raise RuntimeError("LLAMA worker thread failed to initialize or start.")
            logger.info("LLAMA queue and worker thread started.")
        except Exception as e:
            logger.error(f"Failed to launch LLAMA thread: {e}")
            # Очищаем, если что-то успело создаться
            self.llama_queue = None
            self.llama_thread = None
            raise # Перевыбрасываем ошибку

        # 2. Загружаем декодер
        try:
            self.decoder_model = load_decoder_model(
                checkpoint_path=self.decoder_checkpoint_path, device=self.device
            )
            # Применение half precision к декодеру, если нужно
            if self.precision == torch.half:
                 logger.info("Applying half precision to decoder model.")
                 # self.decoder_model = self.decoder_model.half()
            self.decoder_model.eval() # Убедимся, что модель в режиме eval
            logger.info("Decoder model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load decoder model: {e}")
            # Нужно остановить поток LLAMA перед выходом
            self.close() # Вызываем close для очистки
            raise

        # 3. Создаем движок TTS
        try:
            self.engine = TTSInferenceEngine(
                llama_queue=self.llama_queue,
                decoder_model=self.decoder_model,
                precision=self.precision,
                compile=self._compile_model, # Передаем настройку в движок
            )
            logger.info("TTSInferenceEngine created.")
        except Exception as e:
            logger.error(f"Failed to create TTSInferenceEngine: {e}")
            self.close()
            raise

        # 4. "Прогрев" модели (необязательно делать каждый раз, но безопасно)
        logger.info("Warming up the TTS engine...")
        try:
            # Используем генератор и просто проходим по нему
            # чтобы убедиться, что первый вызов прошел
            _ = list(self.engine.inference(
                ServeTTSRequest(
                    text="warmup", # Короткий текст для прогрева
                    references=[],
                    max_new_tokens=10, # Минимальное количество токенов
                    chunk_length=10,
                    top_p=0.7,
                    temperature=0.7,
                    repetition_penalty=1.0,
                    seed=42,
                    streaming=False,
                    normalize=False,
                    use_memory_cache="off"
                )
            ))
            logger.info("TTS engine warmup complete.")
        except Exception as e:
            logger.error(f"Error during TTS engine warmup: {e}")
            # Не фатально, но стоит залогировать
            # self.close() # Не вызываем close здесь, возможно, движок еще рабочий

    def set_compile_mode(self, compile_model: bool):
        """
        Динамически изменяет режим компиляции модели LLAMA.
        Пересоздает необходимые компоненты.
        """
        if self._compile_model == compile_model:
            logger.info(f"Compile mode is already set to {compile_model}. No changes needed.")
            return

        logger.warning(f"Changing compile mode from {self._compile_model} to {compile_model}. This will re-initialize components.")

        # 1. Корректно завершаем работу текущих компонентов
        self.close()
        logger.info("Previous components closed.")

        # 2. Сбрасываем кэш Dynamo перед новой инициализацией (особенно важно при False -> True)
        logger.info("Resetting torch._dynamo cache...")
        try:
            torch._dynamo.reset()
            logger.info("torch._dynamo cache reset successfully.")
        except Exception as e:
            logger.error(f"Failed to reset torch._dynamo cache: {e}. Proceeding, but issues might occur.")

        # 3. Обновляем настройку
        self._compile_model = compile_model

        # 4. Пересоздаем компоненты с новой настройкой
        logger.info("Re-initializing components with new compile mode...")
        try:
            self._setup_components()
            logger.info("Components re-initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to re-initialize components after changing compile mode: {e}")
            # Важно сбросить состояние, если инициализация не удалась
            self._compile_model = not compile_model # Вернуть старое значение? Или None? Лучше None
            self._compile_model = None # Указываем, что состояние невалидно
            self.engine = None
            self.decoder_model = None
            self.llama_queue = None
            self.llama_thread = None
            raise # Передаем ошибку выше

    def close(self):
        """Останавливает фоновый поток и освобождает ресурсы."""
        logger.info("Closing FishSpeech instance...")
        if self.llama_queue is not None:
            try:
                logger.debug("Sending shutdown signal to LLAMA worker thread...")
                self.llama_queue.put(None) # Сигнал потоку для завершения
            except Exception as e:
                logger.warning(f"Error sending shutdown signal to LLAMA queue: {e}")

        if self.llama_thread is not None and self.llama_thread.is_alive():
            logger.debug(f"Waiting for LLAMA worker thread ({self.llama_thread.name}) to join...")
            try:
                self.llama_thread.join(timeout=5.0) # Ждем завершения потока (с таймаутом)
                if self.llama_thread.is_alive():
                    logger.warning("LLAMA worker thread did not join within timeout.")
                else:
                    logger.info("LLAMA worker thread joined successfully.")
            except Exception as e:
                 logger.warning(f"Error joining LLAMA worker thread: {e}")

        # Удаляем ссылки на компоненты
        self.engine = None
        self.decoder_model = None
        self.llama_queue = None
        self.llama_thread = None
        logger.debug("Component references removed.")

        # Очистка памяти
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
        logger.info("FishSpeech instance closed.")

    # Делаем compile_model свойством для удобства чтения (но не записи напрямую)
    @property
    def compile_model(self) -> bool:
        return self._compile_model

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
        seed: Optional[int] = None,
        use_memory_cache: bool = True,
    ) -> tuple[int, np.ndarray]:
        """
        Генерирует речь по тексту.
        (Args и Returns не изменились)
        """
        if self.engine is None:
             raise RuntimeError("FishSpeech instance is not properly initialized or has been closed.")

        # Создаем ServeTTSRequest
        references = []
        if reference_audio:
            try:
                references = self._get_reference_audio(reference_audio, reference_audio_text)
            except Exception as e:
                logger.error(f"Failed to process reference audio: {e}")
                # Продолжаем без референса или выбрасываем ошибку? Пока продолжаем.
                references = []


        request = ServeTTSRequest(
            text=text,
            references=references,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            streaming=False, # Streaming пока не поддерживается через __call__
            normalize=True,
            use_memory_cache="on" if use_memory_cache else "off",
            reference_id=None,
        )

        # Вызываем inference у TTSInferenceEngine
        try:
            result_generator = self.engine.inference(request)
            final_result = None
            # Обрабатываем результат генератора
            for result in result_generator:
                if result.code == "final":
                    final_result = result
                    break
                elif result.code == "error":
                    logger.error(f"Error received from inference engine: {result.error}")
                    raise result.error # Перевыбрасываем ошибку движка
                # Другие коды (header, segment) игнорируем в не-стриминговом вызове

            if final_result is None or final_result.audio is None:
                # Это может случиться, если генератор завершился без final или с ошибкой
                raise RuntimeError("Failed to generate audio or received no final result.")

            sample_rate, audio_data = final_result.audio
            return sample_rate, audio_data

        except Exception as e:
            logger.exception(f"An error occurred during inference call: {e}")
            # Возможно, стоит попытаться восстановить состояние? Пока просто перевыбрасываем.
            raise # Передаем ошибку выше


    def _get_reference_audio(self, reference_audio: Union[str, Path, bytes], reference_text: str) -> list:
        """Внутренний метод для получения байтов референсного аудио."""
        if isinstance(reference_audio, bytes):
            audio_bytes = reference_audio
        elif isinstance(reference_audio, (str, Path)):
            path = Path(reference_audio)
            if not path.exists():
                raise FileNotFoundError(f"Reference audio file not found: {path}")
            with open(path, "rb") as audio_file:
                audio_bytes = audio_file.read()
        else:
            raise TypeError("reference_audio must be a file path (str/Path) or bytes.")

        # TODO: Добавить чтение текста из .lab файла, если reference_text пустой?
        # Пока используем только переданный reference_text.
        if not reference_text:
             logger.warning("Reference audio provided, but reference_audio_text is empty. Quality may be affected.")


        return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]

    def __enter__(self):
        # Позволяет использовать 'with FishSpeech(...) as tts:'
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Гарантирует вызов close при выходе из блока 'with'
        self.close()

    def __del__(self):
        # Попытка очистки при удалении объекта сборщиком мусора
        # Менее надежно, чем явный вызов close() или использование 'with'
        logger.debug(f"__del__ called for FishSpeech instance {id(self)}")
        self.close()
