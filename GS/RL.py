import sounddevice as sd
import numpy as np
import webrtcvad
import whisper
import queue
import collections
import time
from scipy.io.wavfile import write as write_wav
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path


SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
BUFFER_DURATION = 1


print("Load models...")
asr_model = whisper.load_model("medium", device="cuda")
encoder = VoiceEncoder()
print("Whisper + VoiceEncoder are ready!")


audio_queue = queue.Queue()
vad = webrtcvad.Vad(2)  # sensitive
speaker_db = {}


def save_wav(frames, filename="output.wav"):
    audio = np.concatenate(frames)
    write_wav(filename, SAMPLE_RATE, audio)


def callback(indata, frames, time_info, status):
    audio_queue.put(bytes(indata))


def identify_speaker(wav_path):
    wav = preprocess_wav(Path(wav_path))
    embedding = encoder.embed_utterance(wav)

    if not speaker_db:
        speaker_id = "speaker_1"
        speaker_db[speaker_id] = embedding
        return speaker_id

    best_score = -1
    best_id = None
    for spk_id, emb in speaker_db.items():
        similarity = np.inner(embedding, emb)
        if similarity > best_score:
            best_score = similarity
            best_id = spk_id

    if best_score > 0.75:
        return best_id
    else:
        new_id = f"speaker_{len(speaker_db) + 1}"
        speaker_db[new_id] = embedding
        return new_id


def main_loop():
    ring_buffer = collections.deque(maxlen=int(BUFFER_DURATION * 1000 / FRAME_DURATION))
    triggered = False
    voiced_frames = []

    print("🎙 Wait for speaker (Ctrl+C to escape)...")

    while True:
        frame = audio_queue.get()

        if not triggered:
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            ring_buffer.append(frame)
            if sum(vad.is_speech(f, SAMPLE_RATE) for f in ring_buffer) > 0.9 * ring_buffer.maxlen:
                print("🎤 Speach detected!")
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)

            if sum(vad.is_speech(f, SAMPLE_RATE) for f in ring_buffer) < 0.4 * ring_buffer.maxlen:
                triggered = False
                filename = f"speech_{int(time.time())}.wav"
                audio_bytes = b"".join(voiced_frames)
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                save_wav([audio_np], filename)

                print("🧠 Speaker recognition and detection...")
                result = asr_model.transcribe(filename)
                speaker_id = identify_speaker(filename)

                print(f"🧑 {speaker_id}: {result['text']}\n")

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
