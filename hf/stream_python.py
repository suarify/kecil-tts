"""Streaming example: play each piece of audio as it is synthesised, show time to first audio.

pip install onnxruntime numpy huggingface_hub sounddevice
pip install https://huggingface.co/lseanlon/suaradee-v0.2/resolve/main/dist/suaradee-0.2.0-py3-none-any.whl
"""
import time

import sounddevice as sd

from suaradee import SAMPLE_RATE, Suaradee

TEXT = ('Selamat pagi, boleh saya bantu encik hari ini? Jumlah bayaran akaun anda ialah RM1,250.50. '
        'Terima kasih kerana menghubungi kami, sila tunggu sebentar.')

tts = Suaradee.from_pretrained('lseanlon/suaradee-v0.2')       # downloads the 8.5 MB int8 model once

started = time.perf_counter()
first = None
pieces = tts.stream(TEXT)                                       # generator: one float32 array per piece
stream = sd.OutputStream(SAMPLE_RATE, channels=1, dtype='float32')
stream.start()
for chunk in pieces:
    if first is None:
        first = time.perf_counter() - started
        print(f'first audio after {first:.2f} s')
    stream.write(chunk)
print(f'total {time.perf_counter() - started:.2f} s, {len(tts.pieces(TEXT))} pieces')
stream.close()
