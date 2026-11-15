"""pip install onnxruntime numpy huggingface_hub soundfile
   pip install https://huggingface.co/lseanlon/suaradee-v0.1/resolve/main/dist/suaradee-0.1.0-py3-none-any.whl"""
import soundfile as sf

from suaradee import SAMPLE_RATE, Suaradee

tts = Suaradee.from_pretrained('lseanlon/suaradee-v0.1')      # downloads the 8.5 MB int8 model once
audio = tts('Selamat pagi, boleh saya bantu encik hari ini?')
sf.write('hello.wav', audio, SAMPLE_RATE)
print(tts.phonemize('Jumlah bayaran anda ialah RM1,250.50.'))
