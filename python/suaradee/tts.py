"""Suaradee: Malay text -> phonemes -> one ONNX graph (8.5 MB) -> 24 kHz audio. Needs onnxruntime and numpy."""
import os
import re

import numpy as np
import onnxruntime as ort

from .ms_g2p import G2P, VOCAB, chunk
from .ms_norm import normalize

SAMPLE_RATE = 24000
REPO = 'lseanlon/suaradee-v0.1'
MAX_PHONEMES = 400          # the student was trained on single sentences of at most 400 phonemes
G2P_OPTIONS = dict(stress=False, r='r', diph='glide', final_a='a')   # must match how the training rows were made
SENT_END = re.compile(r'(?<=[.!?…])\s+|\n+')


class Suaradee:
    """Suaradee(model_path)("Selamat pagi.") -> float32 numpy array at 24 kHz."""

    def __init__(self, model_path=None, threads=1):
        model_path = model_path or os.environ.get('SUARADEE_MODEL') or self._find_model()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
        self.g2p = G2P(**G2P_OPTIONS)

    @staticmethod
    def _find_model():
        """A model.onnx next to the package, else the int8 file from the Hugging Face repo (cached by huggingface_hub)."""
        local = os.path.join(os.path.dirname(__file__), 'model.onnx')
        if os.path.exists(local):
            return local
        from huggingface_hub import hf_hub_download
        return hf_hub_download(os.environ.get('SUARADEE_REPO', REPO), 'onnx/model_quantized.onnx')

    @classmethod
    def from_pretrained(cls, repo_id=None, threads=1):
        from huggingface_hub import hf_hub_download
        return cls(hf_hub_download(repo_id or os.environ.get('SUARADEE_REPO', REPO), 'onnx/model_quantized.onnx'), threads=threads)

    def phonemize(self, text):
        """List of phoneme strings, each short enough for one model call."""
        out = []
        for sent in SENT_END.split(normalize(text)):
            sent = sent.strip()
            if not sent:
                continue
            ph = self.g2p(sent)
            if not any(c.isalpha() and c in VOCAB for c in ph):      # punctuation alone is not speech
                continue
            parts = [ph] if len(ph) <= MAX_PHONEMES else chunk(ph, MAX_PHONEMES)
            out.extend(parts)
        return out

    def ids(self, phonemes):
        return np.array([self.g2p.tokens(phonemes)], dtype=np.int64)

    def __call__(self, text, speed=1.0, pause=0.12):
        pieces = self.phonemize(text)
        if not pieces:
            return np.zeros(0, dtype=np.float32)
        gap = np.zeros(int(pause * SAMPLE_RATE), dtype=np.float32)
        audio = []
        for p in pieces:
            y = self.session.run(None, {'input_ids': self.ids(p), 'speed': np.array([speed], dtype=np.float32)})[0][0]
            audio += [y.astype(np.float32), gap]
        return np.concatenate(audio[:-1])
