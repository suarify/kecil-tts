"""Runtime tests. Set KOKORO_MALAY_MODEL to an exported int8 .onnx file, then:  pytest python/tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kokoro_malay import SAMPLE_RATE, KokoroMalay  # noqa: E402

MODEL = os.environ.get('KOKORO_MALAY_MODEL')
pytestmark = pytest.mark.skipif(not MODEL or not os.path.exists(MODEL), reason='KOKORO_MALAY_MODEL not set')


@pytest.fixture(scope='module')
def tts():
    return KokoroMalay(MODEL)


def test_sentence(tts):
    y = tts('Selamat pagi, boleh saya bantu encik hari ini?')
    sec = len(y) / SAMPLE_RATE
    assert y.dtype == np.float32 and np.isfinite(y).all()
    assert 1.5 < sec < 6.0 and np.abs(y).max() > 0.01


def test_length_tracks_phonemes(tts):
    ph = tts.phonemize('Terima kasih kerana menghubungi kami.')[0]
    y = tts('Terima kasih kerana menghubungi kami.')
    assert 0.5 < (len(y) / SAMPLE_RATE) / (0.08 * len(ph)) < 2.0


def test_empty_and_symbols(tts):
    assert len(tts('')) == 0
    assert len(tts('   ...  ')) == 0


def test_long_paragraph(tts):
    text = ' '.join(['Saya akan menghantar dokumen itu melalui emel esok.'] * 25)
    assert len(text) > 1200
    y = tts(text)
    assert len(y) / SAMPLE_RATE > 20


def test_numbers_and_codeswitch(tts):
    assert len(tts('Jumlah bayaran anda ialah RM1,250.50.')) > 0
    assert len(tts('Boleh tolong confirm appointment saya pada hari Isnin?')) > 0


def test_speed(tts):
    slow = len(tts('Selamat pagi semua.', speed=0.8))
    fast = len(tts('Selamat pagi semua.', speed=1.25))
    assert slow > fast
