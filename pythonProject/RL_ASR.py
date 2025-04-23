import sounddevice as sd
import numpy as np
import whisper
import scipy.io.wavfile as wav
import time

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.5
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
THRESHOLD = 50000
SILENCE_TIMEOUT = 1.5


print("Загрузка Whisper...")
model = whisper.load_model("medium", device="cuda")
print("Whisper готов!")

audio_buffer = []
last_voice_time = None


def process_block(indata):
    global audio_buffer, last_voice_time

    volume = np.linalg.norm(indata)
    now = time.time()

    if volume > THRESHOLD:
        print("🎙 Говоришь...")
        audio_buffer.extend(indata)
        last_voice_time = now
    else:
        if audio_buffer and last_voice_time and (now - last_voice_time > SILENCE_TIMEOUT):
            print("🧠 Обработка...")
            audio_np = np.array(audio_buffer, dtype=np.int16)
            filename = "speech.wav"
            wav.write(filename, SAMPLE_RATE, audio_np)

            result = model.transcribe(filename)
            print("📄 Результат:", result["text"], "\n")

            audio_buffer = []
            last_voice_time = None


def main():
    print("🎧 Ожидание речи (Ctrl+C для выхода)...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while True:
            block = stream.read(BLOCK_SIZE)[0].flatten()
            process_block(block)


if __name__ == "__main__":
    main()
