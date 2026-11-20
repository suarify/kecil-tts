"""Backwards-compatible shim: use `from kecil_tts import SAMPLE_RATE, KecilTTS` instead."""
from kecil_tts import SAMPLE_RATE, KecilTTS

Suaradee = KecilTTS  # old name still works
__all__ = ["KecilTTS", "Suaradee", "SAMPLE_RATE"]
