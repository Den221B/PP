import os
from pydub import AudioSegment
from pyannote.audio import Pipeline
import whisper
import torchaudio
import tempfile



def combine_audios(file_list, output_path):
    combined = AudioSegment.empty()
    for file in file_list:
        audio = AudioSegment.from_file(file)
        combined += audio
    combined.export(output_path, format="wav")


print("Загрузка моделей...")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="token")

asr_model = whisper.load_model("base")
combined_audio_path = "Tests/3French.wav"


print("Выполнение диаризации...")
diarization = diarization_pipeline(combined_audio_path)


def extract_segment(path, start, end):
    waveform, sr = torchaudio.load(path)
    start_frame = int(start * sr)
    end_frame = int(end * sr)
    segment = waveform[:, start_frame:end_frame]


    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(f.name, segment, sr)
        return f.name


print("Распознавание речи по сегментам...")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segment_path = extract_segment(combined_audio_path, turn.start, turn.end)
    result = asr_model.transcribe(segment_path)
    print(f"[{turn.start:.1f}s - {turn.end:.1f}s] {speaker}: {result['text']}")
    os.remove(segment_path)
