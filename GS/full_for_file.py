import os
import json
import torchaudio
import whisper
from pyannote.audio import Pipeline
from pathlib import Path
from pydub import AudioSegment
from .utils import reduce_noise  # Общие утилиты

def process_audio_file(
    input_path: str,
    output_json: str = "transcript.json",
    hf_token: str = "YOUR_HF_TOKEN"
) -> None:
    # Шумоподавление для всего файла
    print("Applying noise reduction...")
    audio = AudioSegment.from_file(input_path)
    samples = np.array(audio.get_array_of_samples())
    cleaned = reduce_noise(samples.astype(np.float32), audio.frame_rate)
    cleaned_audio = AudioSegment(
        cleaned.astype(np.int16).tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=2,
        channels=1
    )

    # Сохраняем очищенный файл
    temp_path = "temp_cleaned.wav"
    cleaned_audio.export(temp_path, format="wav")

    # Диаризация
    print("Running diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=hf_token
    )
    diarization = pipeline(temp_path)

    # ASR
    model = whisper.load_model("medium", device="cuda")
    results = []

    print("Processing segments...")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Извлекаем сегмент
        waveform, sr = torchaudio.load(temp_path)
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        segment = waveform[:, start_sample:end_sample]

        # Сохраняем временный файл
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            torchaudio.save(f.name, segment, sr)

            # Транскрипция
            result = model.transcribe(f.name, language="ru")
            results.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "text": result["text"]
            })

    # Сохраняем результаты
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    # Очистка
    os.remove(temp_path)
    print(f"Done! Results saved to {output_json}")

if __name__ == "__main__":
    import sys
    process_audio_file(sys.argv[1])