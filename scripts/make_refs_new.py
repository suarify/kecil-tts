"""Generate am_fenrir and af_nicole reference clips (for the two new voices)."""
import os

import numpy as np
import soundfile as sf

os.environ.setdefault('PYTHONUTF8', '1')
OUT = 'runs/refs_new'
os.makedirs(OUT, exist_ok=True)

from kokoro import KPipeline  # noqa: E402

TEXT = ('The morning air is cool and the streets are quiet. We will begin the meeting after a short break, '
        'so please take your seats and check the documents on the table.')
pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
for voice in ('af_nicole', 'am_fenrir'):
    chunks = [a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a) for _, _, a in pipe(TEXT, voice=voice)]
    y = np.concatenate(chunks)
    sf.write(f'{OUT}/ref_{voice}.wav', y, 24000)
    print(voice, round(len(y) / 24000, 2), 's')
