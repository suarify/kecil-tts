"""Generate am_michael reference clips with Kokoro (for ChatterboxVC voice conversion).
Writes kaggle/ds_ref/michael_en/*.wav at 24 kHz."""
import os

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault('PYTHONUTF8', '1')
os.makedirs('kaggle/ds_ref/michael_en', exist_ok=True)

from kokoro import KPipeline  # noqa: E402

pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')   # American English
SENTS = [
    'The weather this morning is calm and clear, with a light breeze from the north.',
    'She placed the documents on the table and asked everyone to review them carefully.',
    'Our team will meet again on Thursday to discuss the delivery schedule.',
    'It is important to speak slowly and clearly when explaining the details.',
    'Please confirm the appointment before the end of the working day.',
]
parts = []
for i, text in enumerate(SENTS):
    for _, _, audio in pipe(text, voice='am_michael', speed=1.0):
        a = audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
        sf.write(f'kaggle/ds_ref/michael_en/m{i + 1:02d}.wav', a, 24000)
        parts.append(a)
        print(f'm{i + 1:02d}.wav', round(len(a) / 24000, 2), 's')
ref = np.concatenate(parts)
sf.write('kaggle/ds_ref/michael_en/ref_michael.wav', ref, 24000)
print('combined reference:', round(len(ref) / 24000, 2), 's')
