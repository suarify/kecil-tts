"""Suaradee v0.2 — Malay text to speech. Runs the 8.5 MB ONNX model on CPU with onnxruntime."""
import time

import gradio as gr
import numpy as np

from suaradee import SAMPLE_RATE, Suaradee

tts = Suaradee.from_pretrained('Suarify/kecil-tts-yati-v0.4')

EXAMPLES = [
    'Selamat pagi, boleh saya bantu encik hari ini?',
    'Jumlah bayaran anda ialah RM1,250.50.',
    'Boleh tolong confirm appointment saya pada hari Isnin?',
    'Terima kasih kerana menghubungi kami, sila tunggu sebentar.',
    'Kod OTP anda ialah 482913.',
    'Temu janji tuan dijadualkan pada hari Khamis, pukul empat petang.',
]


def speak(text, speed):
    text = (text or '').strip()
    if not text:
        raise gr.Error('Type some Malay text first.')
    t0 = time.perf_counter()
    y = tts(text, speed=float(speed))
    took = time.perf_counter() - t0
    sec = len(y) / SAMPLE_RATE
    info = f'{sec:.2f} s of audio in {took:.2f} s ({sec / max(took, 1e-6):.1f}x real time, 1 CPU thread)'
    return (SAMPLE_RATE, (np.clip(y, -1, 1) * 32767).astype(np.int16)), ' | '.join(tts.phonemize(text)), info


with gr.Blocks(title='Suaradee v0.2') as demo:
    gr.Markdown('# Suaradee v0.2\nAn 8.5 MB Malay text-to-speech model. First release: understandable, a little muffled, '
                'and roughly one sentence in ten has a wrong word. [Model card](https://huggingface.co/Suarify/kecil-tts-yati-v0.4)')
    text = gr.Textbox(label='Teks (Malay)', lines=3, value=EXAMPLES[0])
    speed = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label='Speed')
    btn = gr.Button('Speak', variant='primary')
    audio = gr.Audio(label='Audio', type='numpy')
    ph = gr.Textbox(label='Phonemes', interactive=False)
    info = gr.Textbox(label='Timing', interactive=False)
    gr.Examples(EXAMPLES, inputs=text)
    btn.click(speak, [text, speed], [audio, ph, info])

if __name__ == '__main__':
    demo.launch()
