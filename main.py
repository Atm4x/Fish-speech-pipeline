import os
import soundfile as sf
from fish_speech_lib.inference import FishSpeechS2

def main():
    print("Инициализация FishSpeech S2 (первый запуск может занять время из-за скачивания s2-pro)...")
    
    # Инициализируем пайплайн S2
    # Обрати внимание: для S2 крайне рекомендуется half=True (FP16/BF16)
    tts = FishSpeechS2(
        device="cuda",
        half=True, 
        llama_checkpoint_path="checkpoints/s2-pro",
        decoder_checkpoint_path="checkpoints/s2-pro/codec.pth"
    )

    print("Модели успешно загружены. Генерация аудио...")

    # Базовый текст для генерации
    text_to_speak = "Привет! Я работаю на новой архитектуре S2, это прорыв в генерации речи."
    
    # ----------------------------------------------------
    # Твои данные для клонирования голоса (если есть)
    # Если референса нет - будет сгенерирован случайный (Zero-Shot)
    # ----------------------------------------------------
    reference_audio_path = "CrazyMita.wav" # <-- Поменяй на свой путь к файлу
    reference_text = "Привет, я Мита. Как тебя зовут? Рада знакомству. Последний раз я торопилась и оставила тут бардак. Поможешь мне? Ух, спасибо большое. Привет. Я тут проголодалась. Хочу кушать. Купишь мне ингредиенты для куриного супа? Ого, то, что нужно. А теперь давай приготовим это вместе? Привет. Я собиралась сегодня купить дешёвый телевизор. Он маленький и с антенной. Я хотела бы его с собой взять, когда пойду на прогулку. Помни, что ты можешь поиграть в игры, где ты получишь деньги, поиграешь, заработаешь деньги, а потом купишь мне телевизор? Ура! Теперь у тебя достаточно денег, чтобы купить мне телевизор!"
    
    if os.path.exists(reference_audio_path):
        print(f"Использую референс: {reference_audio_path}")
        ref_audio = reference_audio_path
        ref_text = reference_text
    else:
        print(f"Референс '{reference_audio_path}' не найден. Генерирую без клонирования (Zero-shot).")
        ref_audio = None
        ref_text = ""

    # Вызов генерации (inference)
    sample_rate, audio_data = tts(
        text=text_to_speak,
        reference_audio=ref_audio,
        reference_audio_text=ref_text,
        max_new_tokens=1024,
        chunk_length=200
    )

    # Сохраняем результат
    output_filename = "output_s2.wav"
    sf.write(output_filename, audio_data, sample_rate, format='WAV')
    print(f"\n✅ Успех! Аудио сохранено в {output_filename}")

if __name__ == "__main__":
    main()