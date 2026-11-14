"""python -m kokoro_malay "Selamat pagi." -o out.wav [--model path.onnx] [--speed 1.0]"""
import argparse
import wave

import numpy as np

from . import SAMPLE_RATE, KokoroMalay

ap = argparse.ArgumentParser()
ap.add_argument('text')
ap.add_argument('-o', '--out', default='out.wav')
ap.add_argument('--model')
ap.add_argument('--speed', type=float, default=1.0)
A = ap.parse_args()
y = KokoroMalay(A.model)(A.text, speed=A.speed)
pcm = (np.clip(y, -1, 1) * 32767).astype('<i2')
with wave.open(A.out, 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SAMPLE_RATE)
    w.writeframes(pcm.tobytes())
print(f'{len(y) / SAMPLE_RATE:.2f} s -> {A.out}')
