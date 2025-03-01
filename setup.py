from setuptools import setup, find_packages

setup(
    name='fish_speech_lib',
    version='0.1.0',
    packages=find_packages(),  #  ВАЖНО: find_packages()
    install_requires=[
        "numpy<=1.26.4",
        "transformers>=4.45.2",
        "datasets==2.18.0",
        "lightning>=2.1.0",
        "hydra-core>=1.3.2",
        "tensorboard>=2.14.1",
        "natsort>=8.4.0",
        "einops>=0.7.0",
        "librosa>=0.10.1",
        "rich>=13.5.3",
        "gradio>5.0.0", # ТУТ ОШИБКА
        "wandb>=0.15.11",
        "grpcio>=1.58.0",
        "kui>=1.6.0",
        "uvicorn>=0.30.0",
        "loguru>=0.6.0",
      # "loralib>=0.1.2", # Убрал, т.к. больше не нужно
        "pyrootutils>=1.0.4",
      # "vector_quantize_pytorch==1.14.24",
        "resampy>=0.4.3",
      # "einx[torch]==0.2.2", # Удалил, т.к. конфликтовало
        "zstandard>=0.22.0",
        "pydub",
        "pyaudio",
        "faster-whisper",
      # "modelscope==1.17.1", # Убрал
      # "funasr==1.1.5", # Убрал
      # "opencc-python-reimplemented==0.1.7", # Убрал
        "silero-vad",
        "ormsgpack",
        "tiktoken>=0.8.0",
        "pydantic==2.9.2",
        "cachetools",
        "huggingface-hub"
    ],
    author='Atm4x',
    description='Fish Speech pipeline as library so you don\'t need to webui.',
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Или другую лицензию
        'Operating System :: OS Independent',
    ],
    # scripts=['fish_speech_lib/inference.py'], # УДАЛИТЬ! Не нужно
    python_requires='>=3.8',  # Укажи минимальную версию Python
)