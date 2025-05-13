import sounddevice as sd
import numpy as np
import webrtcvad
import whisper
import queue
import collections
import time
import threading
import json
import noisereduce as nr
from scipy.io.wavfile import write as write_wav
from resemblyzer import VoiceEncoder, preprocess_wav  # Убедитесь, что resemblyzer установлен
from pathlib import Path
import torch
import os
import warnings  # Для подавления предупреждений

# Подавить FutureWarning от torch.load, так как мы не контролируем код библиотек
# которые могут использовать старый API torch.load
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# Конфигурация
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
BUFFER_FRAMES = int(1.0 * 1000 / FRAME_DURATION)
RESEMBLE_TARGET_SAMPLING_RATE = 16000
POST_SPEECH_FRAMES = 15
VAD_AGGRESSIVENESS = 3
SIMILARITY_THRESHOLD = 0.70

TRANSCRIPT_FILE = "transcript.json"
AUDIO_SAVE_DIR = "recorded_segments"
Path(AUDIO_SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Инициализация моделей
print("Initializing models...")
try:
    # Используйте CPU, если CUDA недоступна или вызывает проблемы
    device_type = "cuda" if torch.cuda.is_available() else "cpu"  # Добавил torch для проверки
    print(f"Using device: {device_type} for Whisper model")
    asr_model = whisper.load_model("tiny", device=device_type)
    encoder = VoiceEncoder(device=device_type)  # Resemblyzer также может использовать device
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    print("Models ready!")
except Exception as e:
    print(f"Error initializing models: {e}")
    print("Please ensure you have PyTorch, Whisper, Resemblyzer and their dependencies installed.")
    print("If using CUDA, ensure CUDA toolkit is compatible with PyTorch version.")
    exit()

# Глобальные состояния
audio_queue = queue.Queue()
speaker_db = {}
transcript = []
transcript_lock = threading.Lock()
last_saved_transcript_len = 0


def reduce_noise_segment(audio_segment_np: np.ndarray) -> np.ndarray:
    """
    Удаляет шум из сегмента аудио (numpy array int16).
    """
    if audio_segment_np.dtype != np.int16:
        raise ValueError(f"Audio data must be int16 numpy array, got {audio_segment_np.dtype}")


    processed_audio_np = audio_segment_np.copy()  # Работаем с копией
    if processed_audio_np.ndim == 2:
        if processed_audio_np.shape[1] == 1:  # Моно, но в формате (N, 1)
            print(f"Debug: Reshaping audio from {processed_audio_np.shape} to 1D for noise reduction.")
            processed_audio_np = processed_audio_np.squeeze()
        else:  # Стерео или многоканальный
            print(
                f"Warning: multichannel audio_segment_np with shape {processed_audio_np.shape} detected. Taking first channel for noise reduction.")
            processed_audio_np = processed_audio_np[:, 0]

    if processed_audio_np.ndim != 1:
        raise ValueError(
            f"Audio data for noise reduction must be 1D, but got shape {processed_audio_np.shape} after processing.")

    audio_float = processed_audio_np.astype(np.float32) / 32768.0

    print(f"Debug: Shape of audio_float passed to nr.reduce_noise: {audio_float.shape}, dtype: {audio_float.dtype}")

    try:
        reduced_noise_float = nr.reduce_noise(
            y=audio_float,
            sr=SAMPLE_RATE,
            stationary=False,
            prop_decrease=0.8,

        )
    except Exception as e_nr:
        print(f"ERROR during noise reduction: {e_nr}")
        print(f"Input audio_float shape: {audio_float.shape}, dtype: {audio_float.dtype}")
        import traceback
        traceback.print_exc()
        print("Skipping noise reduction for this segment.")
        reduced_noise_int16 = (audio_float * 32768.0).astype(np.int16)
        return reduced_noise_int16

    reduced_noise_int16 = (reduced_noise_float * 32768.0).astype(np.int16)
    return reduced_noise_int16


def save_wav(audio_data_np: np.ndarray, filename: str) -> None:
    write_wav(filename, SAMPLE_RATE, audio_data_np)


def update_transcript_file() -> None:
    global last_saved_transcript_len
    with transcript_lock:
        if not transcript:
            return
        if len(transcript) > last_saved_transcript_len:
            try:
                with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
                    json.dump(transcript, f, indent=2, ensure_ascii=False)
                print(f"Transcript updated and saved to: {TRANSCRIPT_FILE} ({len(transcript)} entries)")
                last_saved_transcript_len = len(transcript)
            except Exception as e:
                print(f"Error saving transcript: {e}")


# def identify_speaker(wav_path: str) -> str:
#     try:
#         # Resemblyzer может быть чувствителен к очень коротким аудио.
#         # Минимальная длина аудио для Resemblyzer около 1.6 секунд.
#         # Если файл короче, эмбеддинг может быть плохим.
#         # preprocess_wav также нормализует громкость.
#         wav = preprocess_wav(Path(wav_path))
#         if len(wav) < encoder.sampling_rate * 0.5:  # Например, минимум 0.5 секунды
#             print(
#                 f"Warning: Preprocessed wav is too short for reliable speaker embedding: {len(wav) / encoder.sampling_rate:.2f}s for {wav_path}")
#             # Можно вернуть специальный ID или предыдущего спикера, если есть контекст
#             return "unknown_speaker_short_audio"
#         embedding = encoder.embed_utterance(wav)
#     except Exception as e:
#         print(f"Error during speaker embedding for {wav_path}: {e}")
#         return "unknown_speaker_emb_error"
#
#     best_score = -1
#     best_id = None
#
#     if not speaker_db:
#         new_id = "speaker_1"
#         speaker_db[new_id] = embedding
#         print(f"New speaker identified: {new_id}")
#         return new_id
#
#     for spk_id, emb_list in speaker_db.items():
#         current_emb = emb_list
#         similarity = np.inner(embedding, current_emb)
#         if similarity > best_score:
#             best_score = similarity
#             best_id = spk_id
#
#     if best_id is not None and best_score > SIMILARITY_THRESHOLD:
#         # Обновляем эмбеддинг (простое усреднение или экспоненциальное сглаживание)
#         # speaker_db[best_id] = (speaker_db[best_id] * 0.8 + embedding * 0.2)
#         # Для начала можно не обновлять, а просто использовать первый эмбеддинг,
#         # или хранить список эмбеддингов и сравнивать со средним.
#         # Простое обновление:
#         speaker_db[best_id] = (speaker_db[best_id] + embedding) / 2.0
#         print(f"Identified as existing {best_id} with score {best_score:.2f}")
#         return best_id
#     else:
#         new_id = f"speaker_{len(speaker_db) + 1}"
#         speaker_db[new_id] = embedding
#         print(f"New speaker identified: {new_id} (best score for existing was {best_score:.2f})")
#         return new_id

def identify_speaker(wav_path: str) -> str:
    processed_wav_data = None  # Для отладки
    try:
        print(f"Debug SpeakerID: Processing file {wav_path}")
        path_obj = Path(wav_path)
        if not path_obj.exists():
            print(f"Error SpeakerID: File not found at {wav_path}")
            return "unknown_speaker_file_not_found"
        if path_obj.stat().st_size == 0:
            print(f"Error SpeakerID: File {wav_path} is empty.")
            return "unknown_speaker_empty_file"

        # Этап 1: Предобработка аудио
        try:
            processed_wav_data = preprocess_wav(path_obj)  # path_obj, а не строка wav_path
        except Exception as e_preprocess:
            print(f"Error SpeakerID: preprocess_wav failed for {wav_path}. Error: {e_preprocess}")
            import traceback
            traceback.print_exc()
            return "unknown_speaker_preprocess_error"

        # Проверка результата preprocess_wav
        if processed_wav_data is None or len(processed_wav_data) == 0:
            print(
                f"Warning SpeakerID: Preprocessed wav is empty or None for {wav_path}. Original size: {path_obj.stat().st_size}")
            return "unknown_speaker_empty_preprocessed_wav"

        print(f"Debug SpeakerID: Preprocessed wav shape: {processed_wav_data.shape}, dtype: {processed_wav_data.dtype}, min: {np.min(processed_wav_data):.2f}, max: {np.max(processed_wav_data):.2f}")

        min_duration_seconds = 1.0
        min_duration_samples = int(RESEMBLE_TARGET_SAMPLING_RATE * min_duration_seconds)
        if len(processed_wav_data) < min_duration_samples:
            print(f"Warning SpeakerID: Preprocessed wav is too short ({len(processed_wav_data)/RESEMBLE_TARGET_SAMPLING_RATE:.2f}s) for speaker embedding: {wav_path}. Min required: {min_duration_seconds}s")
            return "unknown_speaker_short_audio"

        # Этап 2: Получение эмбеддинга
        embedding = None
        try:
            embedding = encoder.embed_utterance(processed_wav_data)
        except Exception as e_embed:
            print(f"Error SpeakerID: embed_utterance failed for {wav_path}. Error: {e_embed}")
            print(
                f"Debug SpeakerID: Data passed to embed_utterance - shape: {processed_wav_data.shape}, dtype: {processed_wav_data.dtype}, any NaNs: {np.isnan(processed_wav_data).any()}, any Infs: {np.isinf(processed_wav_data).any()}")
            import traceback
            traceback.print_exc()
            return "unknown_speaker_emb_error"  # Это ваша текущая ошибка

        if embedding is None:
            print(f"Error SpeakerID: embed_utterance returned None for {wav_path}")
            return "unknown_speaker_emb_returned_none"

    except Exception as e_outer:
        # Общий обработчик, если что-то пошло не так до специфических try-except
        print(f"Error SpeakerID: General error during speaker identification for {wav_path}. Error: {e_outer}")
        import traceback
        traceback.print_exc()
        return "unknown_speaker_general_error"

    # Этап 3: Сравнение с базой данных спикеров (остается как было, но теперь вызывается только если эмбеддинг получен)
    best_score = -1.0  # Инициализируем float
    best_id = None

    if not speaker_db:
        new_id = "speaker_1"
        speaker_db[new_id] = embedding  # Сохраняем непосредственно эмбеддинг
        print(f"New speaker identified: {new_id} (from file {Path(wav_path).name})")
        return new_id

    for spk_id, known_embedding in speaker_db.items():
        try:
            similarity = np.inner(embedding, known_embedding)
        except Exception as e_sim:
            print(f"Error SpeakerID: Failed to calculate similarity for speaker {spk_id}. Error: {e_sim}")
            print(f"Current embedding shape: {embedding.shape}, Known embedding shape: {known_embedding.shape}")
            continue  # Пропускаем этого спикера, если не можем посчитать схожесть

        if similarity > best_score:
            best_score = similarity
            best_id = spk_id

    if best_id is not None and best_score >= SIMILARITY_THRESHOLD:  # Используем >=
        # Обновляем эмбеддинг (экспоненциальное сглаживание или простое усреднение)
        speaker_db[best_id] = (speaker_db[best_id].astype(np.float32) * 0.8 + embedding.astype(np.float32) * 0.2)
        print(f"Identified as existing {best_id} with score {best_score:.2f} (file: {Path(wav_path).name})")
        return best_id
    else:
        new_id = f"speaker_{len(speaker_db) + 1}"
        speaker_db[new_id] = embedding
        print(
            f"New speaker identified: {new_id} (best score for existing was {best_score:.2f if best_id else 'N/A'}) (file: {Path(wav_path).name})")
        return new_id


def process_audio_segment_thread(frames_list: list, segment_start_time: float) -> None:
    try:
        if not frames_list:
            print("Skipping empty audio segment.")
            return

        segment_audio_np = np.concatenate(frames_list)  # Форма будет (N, 1)
        segment_duration_actual = len(segment_audio_np) / SAMPLE_RATE
        print(f"Debug: Original segment_audio_np shape: {segment_audio_np.shape}, dtype: {segment_audio_np.dtype}")

        cleaned_audio_np = reduce_noise_segment(segment_audio_np.copy())  # Передаем копию

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename_no_ext = f"segment_{timestamp_str}_{int(segment_start_time)}"
        cleaned_filename = Path(AUDIO_SAVE_DIR) / f"{filename_no_ext}_cleaned.wav"

        save_wav(cleaned_audio_np, str(cleaned_filename))  # cleaned_audio_np должен быть (N,)
        print(f"Saved cleaned audio to: {cleaned_filename}")

        # ASR
        transcription_result = asr_model.transcribe(str(cleaned_filename), language="ru", fp16=(device_type == "cuda"))
        text = transcription_result["text"].strip()
        print(f"Transcription: '{text}'")

        speaker_id = "unknown"
        if text:
            speaker_id = identify_speaker(str(cleaned_filename))
        else:
            print("No text transcribed, skipping speaker identification.")

        entry = {
            "start_time_abs": segment_start_time,
            "end_time_abs": segment_start_time + segment_duration_actual,
            "duration": segment_duration_actual,
            "speaker": speaker_id,
            "text": text,
            "cleaned_audio_file": str(cleaned_filename)
        }
        with transcript_lock:
            transcript.append(entry)
            transcript.sort(key=lambda x: x["start_time_abs"])
        update_transcript_file()

        print(
            f"\nSegment Processed: [{speaker_id} at {time.strftime('%H:%M:%S', time.localtime(segment_start_time))}]: {text}\n")

    except Exception as e:
        print(f"Error in process_audio_segment_thread: {str(e)}")
        import traceback
        traceback.print_exc()


def audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    if status:
        print(f"Audio callback status: {status}")
    audio_queue.put(indata.copy())


def main_loop():
    ring_buffer = collections.deque(maxlen=BUFFER_FRAMES)
    post_speech_activity_buffer = collections.deque(maxlen=POST_SPEECH_FRAMES)
    active_speech_frames = []
    is_speech_active = False
    segment_start_time = 0

    VAD_START_THRESHOLD_RATIO = 0.7
    VAD_END_THRESHOLD_RATIO = 0.85

    print(f"🎙️  Listening... (Ctrl+C to stop)")
    print(f"Frame size: {FRAME_SIZE} samples ({FRAME_DURATION}ms)")
    print(f"VAD ring buffer: {BUFFER_FRAMES} frames ({BUFFER_FRAMES * FRAME_DURATION / 1000:.2f}s)")
    print(
        f"VAD post speech buffer: {POST_SPEECH_FRAMES} frames ({POST_SPEECH_FRAMES * FRAME_DURATION / 1000:.2f}s for silence)")

    while True:
        try:
            current_frame_np = audio_queue.get(timeout=0.1)
            current_frame_bytes = current_frame_np.tobytes()

            is_current_frame_speech = False
            try:
                if len(current_frame_bytes) == FRAME_SIZE * 2 * 1:  # channels=1
                    is_current_frame_speech = vad.is_speech(current_frame_bytes, SAMPLE_RATE)
                else:
                    print(
                        f"Warning: Incorrect frame size for VAD: {len(current_frame_bytes)}, expected {FRAME_SIZE * 2}")
            except Exception as e_vad:
                print(f"VAD error: {e_vad}. Frame len: {len(current_frame_bytes)}")
                is_current_frame_speech = False

            if not is_speech_active:
                ring_buffer.append(is_current_frame_speech)
                # Сохраняем фреймы в `active_speech_frames` сразу, чтобы не потерять начало речи,
                # если буфер `ring_buffer` будет содержать аудио-фреймы.
                # В текущей реализации `ring_buffer` хранит bool, так что аудио-фреймы из него мы не берем.
                # Начало речи может быть немного потеряно (до BUFFER_FRAMES).
                # Для улучшения можно хранить аудио в ring_buffer.
                # Пока что добавляем current_frame_np в active_speech_frames только после детекции.

                if len(ring_buffer) == BUFFER_FRAMES:
                    num_speech_frames_in_ring = sum(ring_buffer)
                    if num_speech_frames_in_ring >= VAD_START_THRESHOLD_RATIO * BUFFER_FRAMES:
                        print(
                            f"\n🔊 Speech detected! ({num_speech_frames_in_ring}/{BUFFER_FRAMES} voiced frames in buffer)")
                        is_speech_active = True
                        # Время начала сегмента - это текущее время минус длительность буфера обнаружения
                        # плюс небольшой запас, чтобы учесть, что речь могла начаться внутри буфера.
                        # Это примерное время.
                        segment_start_time = time.time() - (len(ring_buffer) * FRAME_DURATION / 1000.0)

                        active_speech_frames.clear()
                        # Если бы ring_buffer хранил аудио, мы бы добавили его сюда.
                        # Например: active_speech_frames.extend(list(audio_ring_buffer))
                        # Сейчас начинаем с текущего фрейма, который "подтвердил" начало речи.
                        active_speech_frames.append(current_frame_np)
                        post_speech_activity_buffer.clear()
            else:
                active_speech_frames.append(current_frame_np)
                post_speech_activity_buffer.append(is_current_frame_speech)

                if len(post_speech_activity_buffer) == POST_SPEECH_FRAMES:
                    num_silent_frames_in_post = POST_SPEECH_FRAMES - sum(post_speech_activity_buffer)
                    if num_silent_frames_in_post >= VAD_END_THRESHOLD_RATIO * POST_SPEECH_FRAMES:
                        print(
                            f"🎤 Silence detected, processing segment... ({num_silent_frames_in_post}/{POST_SPEECH_FRAMES} silent frames)")
                        is_speech_active = False

                        # Убираем "хвост" тишины из active_speech_frames
                        # frames_to_process = active_speech_frames[:-POST_SPEECH_FRAMES] # Это удалит последние POST_SPEECH_FRAMES фреймов
                        # Более аккуратно: найти последний речевой фрейм в post_speech_activity_buffer и обрезать по нему.
                        # Пока оставим простой вариант: обрабатываем все накопленные фреймы.
                        # Если POST_SPEECH_FRAMES это действительно тишина, она будет обработана, но может быть не нужна.
                        # Однако, если последний фрейм в post_speech_activity_buffer был речью, но среднее значение низкое,
                        # мы можем обрезать речь.
                        # Просто копируем все active_speech_frames:
                        frames_to_process = list(active_speech_frames)  # Создаем копию

                        if frames_to_process:
                            threading.Thread(
                                target=process_audio_segment_thread,
                                args=(frames_to_process, segment_start_time),
                                daemon=True
                            ).start()
                        else:
                            print("Segment too short after considering silence tail, skipping.")

                        active_speech_frames.clear()
                        ring_buffer.clear()
                        post_speech_activity_buffer.clear()

        except queue.Empty:
            if not is_speech_active and transcript:
                update_transcript_file()
            continue
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    print("Final transcript save...")
    update_transcript_file()
    print("Done.")


def run():
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype='int16',
        channels=1,  # Моно
        callback=audio_callback
    )
    with stream:
        main_loop()


if __name__ == "__main__":
    run()