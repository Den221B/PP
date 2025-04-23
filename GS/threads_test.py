import sounddevice as sd
import numpy as np
import webrtcvad
import whisper
import queue
import collections
import time
import threading
from scipy.io.wavfile import write as write_wav

print(sd.query_devices())

# Параметры аудио
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # мс
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
BUFFER_DURATION = 1  # секунд
thrash_hold = 0.4

# Очередь аудио
audio_queue = queue.Queue()

# Инициализация VAD
vad = webrtcvad.Vad(0)  # чувствительность: 0 — наименее чувствительный

# Загрузка Whisper на GPU
print("Загрузка Whisper...")
asr_model = whisper.load_model("medium", device="cuda")
print("Whisper готов!")


def asr_worker(audio_data):
    timestamp = int(time.time())
    filename = f"speech_{timestamp}.wav"
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    write_wav(filename, SAMPLE_RATE, audio_np)

    print(f"🧠 Распознаю {filename}...")
    result = asr_model.transcribe(filename)
    print(f"📄 Результат: {result['text']}\n")



def callback(indata, frames, time_info, status):
    audio_queue.put(bytes(indata))


def main_loop():
    ring_buffer = collections.deque(maxlen=int(BUFFER_DURATION * 1000 / FRAME_DURATION))
    triggered = False
    voiced_frames = []

    print("🎙 Ожидание речи (Ctrl+C для выхода)...")

    while True:
        frame = audio_queue.get()

        if not triggered:
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            ring_buffer.append(frame)
            if sum(vad.is_speech(f, SAMPLE_RATE) for f in ring_buffer) > 0.9 * ring_buffer.maxlen:
                print("🎤 Речь обнаружена!")
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)

            if sum(vad.is_speech(f, SAMPLE_RATE) for f in ring_buffer) < thrash_hold * ring_buffer.maxlen:
                triggered = False

                audio_bytes = b"".join(voiced_frames)
                threading.Thread(target=asr_worker, args=(audio_bytes,), daemon=True).start()

                voiced_frames = []
                ring_buffer.clear()


def run():
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype='int16',
        channels=1,
        callback=callback
    ):
        main_loop()


if __name__ == "__main__":
    run()
