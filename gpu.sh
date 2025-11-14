#!/bin/bash
# gpu.sh - Verify CUDA/cuDNN + faster_whisper GPU Setup (Fixed Generator)

uv run python -c "
import torch
from faster_whisper import WhisperModel
from pydub import AudioSegment
import io

print('CUDA available:', torch.cuda.is_available())
print('GPU name:', torch.cuda.get_device_name(0))
print('cuDNN version:', torch.backends.cudnn.version())

# Load tiny model on GPU
model = WhisperModel('tiny', device='cuda', compute_type='float16')
print('Model initialized on GPU (no errors = loaded)')

# Create valid 1s silence WAV in memory (16kHz mono)
silence = AudioSegment.silent(duration=1000).set_frame_rate(16000).set_channels(1)
wav_buffer = io.BytesIO()
silence.export(wav_buffer, format='wav')
wav_buffer.seek(0)

# Transcribe dummy WAV (checks full runtime/GPU ops)
segments_gen, info = model.transcribe(wav_buffer, language='en', beam_size=1)
segments = list(segments_gen)  # Consume generator to list
print('GPU inference OK:', len(segments) >= 0, f'(Segments: {len(segments)}, Lang: {info.language})')

print('✅ GPU setup verified - ready for duber!')
"