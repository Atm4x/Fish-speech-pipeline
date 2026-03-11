import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import soundfile as sf
import torch
import librosa
from fish_speech_lib.inference import FishSpeechS2

def main():
    # Настройки
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Инициализация FishSpeech S2...")
    tts = FishSpeechS2(
        device=device,
        half=not torch.cuda.is_bf16_supported(),
        llama_checkpoint_path="checkpoints/s2-pro",
        decoder_checkpoint_path="checkpoints/s2-pro/codec.pth"
    )

    # ВАЖНО: Обрезаем слишком длинное аудио для скорости!
    reference_audio_path = "CrazyMita.wav"
    processed_ref_path = "CrazyMita_short.wav"
    
    if os.path.exists(reference_audio_path):
        # Если аудио длиннее 15 секунд, берем только первые 15
        y, sr = librosa.load(reference_audio_path, sr=None)
        if len(y) > sr * 11:
            print("⚠️ Референс слишком длинный (41 сек). Обрезаю до 15 сек для ускорения...")
            sf.write(processed_ref_path, y[:sr*11], sr)
            ref_audio = processed_ref_path
        else:
            ref_audio = reference_audio_path
        
        ref_text = "Привет, я Мита. Как тебя зовут? Рада знакомству. Последний раз я торопилась и оставила тут бардак. Поможешь мне? Ух, спасибо большое." # Желательно покороче
    else:
        ref_audio = None
        ref_text = ""

    print("Генерация... (теперь должно быть быстрее)")
    
    # Текст для синтеза
    text_to_speak = "Привет! Теперь я использую полный конфиг кодека, и мой голос должен стать чистым."

    sample_rate, audio_data = tts(
        text=text_to_speak,
        reference_audio=ref_audio,
        reference_audio_text=ref_text,
        max_new_tokens=512, # Уменьшил для теста
        chunk_length=200
    )

    output_filename = "output_s2_fixed.wav"
    sf.write(output_filename, audio_data, sample_rate, format='WAV')
    print(f"\n✅ Готово! Проверь {output_filename}")

if __name__ == "__main__":
    main()