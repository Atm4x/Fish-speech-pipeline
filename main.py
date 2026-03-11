import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import shlex
import soundfile as sf
import torch
import librosa

from fish_speech_lib.inference import FishSpeechS2

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory


def trim_audio_to_file(in_path: str, max_seconds: float | None):
    """
    Если max_seconds задан и файл длиннее — обрезает и сохраняет рядом.
    Возвращает путь к файлу (исходный или обрезанный).
    """
    if not in_path or not os.path.exists(in_path):
        return None

    if not max_seconds or max_seconds <= 0:
        return in_path

    y, sr = librosa.load(in_path, sr=None)
    max_len = int(sr * max_seconds)

    if len(y) <= max_len:
        return in_path

    base, ext = os.path.splitext(in_path)
    out_path = f"{base}_short_{max_seconds:g}s.wav"
    sf.write(out_path, y[:max_len], sr)
    return out_path


def print_help():
    print(
        "\nКоманды:\n"
        "  /help                         - помощь\n"
        "  /show                         - показать текущие настройки\n"
        "  /ref <путь_к_wav>             - установить референс (обрежется по /trim)\n"
        "  /trim <секунды|0>             - установить лимит обрезки референса (0 = не обрезать)\n"
        "  /retrim                       - заново применить текущий /trim к текущему референсу\n"
        "  /reftext <текст>              - установить текст референса\n"
        "  /clearref                     - убрать референс (будет обычный TTS)\n"
        "  /tokens <int>                 - max_new_tokens\n"
        "  /chunk <int>                  - chunk_length\n"
        "  exit | quit | q               - выход\n"
        "\nПодсказка: пути с пробелами берите в кавычки:\n"
        '  /ref "C:\\Voices\\Crazy Mita.wav"\n'
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Инициализация FishSpeech S2...")
    tts = FishSpeechS2(
        device=device,
        half=not torch.cuda.is_bf16_supported(),
        llama_checkpoint_path="checkpoints/s2-pro",
        decoder_checkpoint_path="checkpoints/s2-pro/codec.pth",
    )

    # Параметры/состояние
    ref_original = None
    ref_audio = None
    ref_text = ""
    trim_seconds = 11.0

    max_new_tokens = 512
    chunk_length = 200

    # Prompt с историей (стрелки ↑/↓)
    session = PromptSession(history=FileHistory(".tts_history.txt"))

    print("\nГотово. Пиши текст для озвучки или /help.\n")

    i = 1
    while True:
        try:
            line = session.prompt("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not line:
            continue

        low = line.lower().strip()
        if low in {"exit", "quit", "q"}:
            break

        # Команды
        if line.startswith("/"):
            # Windows-friendly split (поддержка кавычек)
            try:
                parts = shlex.split(line, posix=False)
            except ValueError as e:
                print(f"Ошибка разбора команды: {e}")
                continue

            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in {"/help", "/h"}:
                print_help()
                continue

            if cmd == "/show":
                print("\nТекущие настройки:")
                print(f"  ref_original: {ref_original}")
                print(f"  ref_audio:    {ref_audio}")
                print(f"  trim_seconds: {trim_seconds}")
                print(f"  ref_text:     {ref_text[:120]}{'...' if len(ref_text) > 120 else ''}")
                print(f"  max_new_tokens: {max_new_tokens}")
                print(f"  chunk_length:   {chunk_length}\n")
                continue

            if cmd == "/ref":
                if not args:
                    print("Использование: /ref <путь_к_wav>")
                    continue
                path = " ".join(args).strip('"')
                if not os.path.exists(path):
                    print(f"Файл не найден: {path}")
                    continue
                ref_original = path
                ref_audio = trim_audio_to_file(ref_original, trim_seconds)
                if ref_audio != ref_original:
                    print(f"✅ Референс установлен и обрезан: {ref_audio}")
                else:
                    print(f"✅ Референс установлен: {ref_audio}")
                continue

            if cmd == "/trim":
                if not args:
                    print("Использование: /trim <секунды|0>")
                    continue
                try:
                    trim_seconds = float(args[0])
                except ValueError:
                    print("Нужно число. Пример: /trim 11   или  /trim 0")
                    continue

                if trim_seconds <= 0:
                    print("✅ Обрезка отключена (/trim 0).")
                else:
                    print(f"✅ Обрезка установлена: {trim_seconds:g} сек.")

                # Автопереобработка текущего референса
                if ref_original and os.path.exists(ref_original):
                    ref_audio = trim_audio_to_file(ref_original, trim_seconds)
                    print(f"↻ Применил обрезку к референсу: {ref_audio}")
                continue

            if cmd == "/retrim":
                if not ref_original:
                    print("Референс не задан. Используйте /ref <путь>")
                    continue
                if not os.path.exists(ref_original):
                    print(f"Исходный референс не найден: {ref_original}")
                    continue
                ref_audio = trim_audio_to_file(ref_original, trim_seconds)
                print(f"↻ Готово: {ref_audio}")
                continue

            if cmd == "/reftext":
                if not args:
                    ref_text = ""
                    print("✅ ref_text очищен.")
                else:
                    # сохраняем как есть (после команды)
                    ref_text = line[len("/reftext"):].lstrip()
                    print("✅ ref_text установлен.")
                continue

            if cmd == "/clearref":
                ref_original = None
                ref_audio = None
                print("✅ Референс отключён.")
                continue

            if cmd == "/tokens":
                if not args:
                    print("Использование: /tokens <int>")
                    continue
                try:
                    max_new_tokens = int(args[0])
                    print(f"✅ max_new_tokens = {max_new_tokens}")
                except ValueError:
                    print("Нужно целое число.")
                continue

            if cmd == "/chunk":
                if not args:
                    print("Использование: /chunk <int>")
                    continue
                try:
                    chunk_length = int(args[0])
                    print(f"✅ chunk_length = {chunk_length}")
                except ValueError:
                    print("Нужно целое число.")
                continue

            print("Неизвестная команда. /help")
            continue

        # Обычный синтез текста
        print("Генерация...")
        try:
            sample_rate, audio_data = tts(
                text=line,
                reference_audio=ref_audio,
                reference_audio_text=ref_text,
                max_new_tokens=max_new_tokens,
                chunk_length=chunk_length,
            )
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            continue

        out_name = f"tts_{i:04d}_{int(time.time())}.wav"
        sf.write(out_name, audio_data, sample_rate, format="WAV")
        print(f"✅ Сохранено: {out_name}\n")
        i += 1


if __name__ == "__main__":
    main()