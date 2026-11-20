"""KecilTTS: Malay text -> phonemes -> one ONNX graph (8.5 MB) -> 24 kHz audio. Needs onnxruntime and numpy."""
import os
import re

import numpy as np
import onnxruntime as ort

from .ms_g2p import G2P, VOCAB, chunk
from .ms_norm import normalize

SAMPLE_RATE = 24000
REPO = 'Suarify/kecil-tts-yati-v0.4'
MAX_PHONEMES = 400          # the student was trained on single sentences of at most 400 phonemes
FIRST_MAX = 60              # streaming: the first piece is at most this many phonemes, so sound starts early
LATER_MAX = 160             # streaming: every later piece is at most this many phonemes
G2P_OPTIONS = dict(stress=False, r='r', diph='glide', final_a='a')   # must match how the training rows were made
SENT_END = re.compile(r'(?<=[.!?…])\s+|\n+')


class KecilTTS:
    """KecilTTS(model_path)("Selamat pagi.") -> float32 numpy array at 24 kHz."""

    def __init__(self, model_path=None, threads=1):
        model_path = model_path or os.environ.get('SUARADEE_MODEL') or self._find_model()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3          # the exported graph has a few unused constants; ONNX Runtime warns about them
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

    @staticmethod
    def _split_commas(phonemes, first_limit, later_limit):
        """Cut a phoneme string at commas only: the first piece at most first_limit symbols, later pieces at
        most later_limit. A piece with no comma inside its limit is kept whole, so piece boundaries of
        comma-free texts never change."""
        out = []
        rest = phonemes.strip()
        first = True
        while True:
            limit = first_limit if first else later_limit
            if len(rest) <= limit:
                out.append(rest)
                return out
            cut = rest.rfind(',', 0, limit + 1)
            if cut < 0:
                out.append(rest)
                return out
            out.append(rest[:cut].strip())
            rest = rest[cut:].strip()
            first = False
            if not rest:
                return out

    def pieces(self, text, first_max=FIRST_MAX):
        """Phoneme pieces for streaming: sentences as phonemize() makes them, with pieces that exceed the
        streaming limit cut at commas."""
        out = []
        for ph in self.phonemize(text):
            out.extend(self._split_commas(ph, first_max, LATER_MAX))
        return out

    def stream(self, text, speed=1.0, pause=0.12, first_max=FIRST_MAX):
        """Yield one float32 array per piece (sample rate SAMPLE_RATE); a pause-length gap is yielded
        between pieces. The consumer gets the first piece before the rest is synthesised."""
        gap = np.zeros(int(pause * SAMPLE_RATE), dtype=np.float32)
        emitted = False
        for p in self.pieces(text, first_max=first_max):
            if not any(c.isalpha() and c in VOCAB for c in p):      # punctuation alone is not speech
                continue
            y = self.session.run(None, {'input_ids': self.ids(p), 'speed': np.array([speed], dtype=np.float32)})[0][0]
            if emitted:
                yield gap.copy()
            yield y.astype(np.float32)
            emitted = True

    def __call__(self, text, speed=1.0, pause=0.12):
        audio = list(self.stream(text, speed=speed, pause=pause))
        return np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
