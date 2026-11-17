"""Generate short samples of candidate Kokoro voices for a possible third Kecil voice."""
import os

import numpy as np
import soundfile as sf

os.environ.setdefault('PYTHONUTF8', '1')
OUT = 'runs/listen/voices'
os.makedirs(OUT, exist_ok=True)

from kokoro import KPipeline  # noqa: E402

TEXT = ('Good morning. This is a short sample so you can hear the tone of this voice before we use it '
        'to build a Malay model. The weather today is calm, with a little sunlight in the afternoon.')
VOICES = ['af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'bf_emma', 'am_puck', 'am_fenrir', 'bm_george']

pipes = {'a': KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M'),
         'b': KPipeline(lang_code='b', repo_id='hexgrad/Kokoro-82M')}
for v in VOICES:
    pipe = pipes[v[0]]
    try:
        chunks = [a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a) for _, _, a in pipe(TEXT, voice=v)]
        y = np.concatenate(chunks)
        sf.write(f'{OUT}/{v}.wav', y, 24000)
        print(v, round(len(y) / 24000, 1), 's')
    except Exception as e:
        print('FAIL', v, type(e).__name__, str(e)[:80])
