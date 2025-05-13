import noisereduce as nr
import numpy as np

def reduce_noise(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    return nr.reduce_noise(
        y=audio.astype(np.float32),
        sr=sr,
        stationary=True,
        use_tensorflow=True,
        n_fft=1024,
        hop_length=256
    )

def save_audio(audio: np.ndarray, path: str, sr: int = 16000) -> None:
    from scipy.io.wavfile import write
    write(path, sr, audio.astype(np.int16))