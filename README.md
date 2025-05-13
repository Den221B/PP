# Real-time Speech Processing System

## Features
- Real-time voice activity detection
- Noise reduction using spectral gating
- Speaker identification using voice embeddings
- Live transcription with Whisper
- File processing with speaker diarization
- JSON transcript output

## Installation
```bash
conda create -n speech python=3.9
conda activate speech
pip install -r requirements.txt
```

## Usage

Real-time processing:

```bash
python GS/RL.py
```

File processing:

```bash
python GS/full_for_file.py input_audio.wav
```

## Configuration

Edit `SAMPLE_RATE`, `VAD_AGGRESSIVENESS` in RL.py for hardware tuning


