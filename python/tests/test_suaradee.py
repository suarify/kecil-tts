"""Runtime tests. Set SUARADEE_MODEL to an exported int8 .onnx file, then:  pytest python/tests"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from suaradee import SAMPLE_RATE, Suaradee  # noqa: E402

MODEL = os.environ.get('SUARADEE_MODEL')
pytestmark = pytest.mark.skipif(not MODEL or not os.path.exists(MODEL), reason='SUARADEE_MODEL not set')


@pytest.fixture(scope='module')
def tts():
    return Suaradee(MODEL)


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


def test_stream_matches_call_structure(tts):
    # The model has a random-noise head, so separate runs differ in samples; the invariant is the
    # piece structure and the total length (and __call__ is stream() concatenated, by construction).
    text = 'Selamat pagi, boleh saya bantu encik? Jumlah bayaran akaun anda ialah RM1,250.50, sila semak.'
    pieces = tts.pieces(text)
    ys = list(tts.stream(text))
    assert len(ys) == 2 * len(pieces) - 1          # piece audio + gap between pieces
    assert sum(a.shape[-1] for a in ys) == len(tts(text))
    assert all(a.dtype == np.float32 for a in ys)


def test_stream_first_piece_short(tts):
    text = 'Selamat pagi encik, saya nak tanya pasal akaun simpanan bersama pihak bank ini, boleh atau tidak?'
    first = next(iter(tts.stream(text)))
    assert len(tts.pieces(text)[0]) <= 60          # at most first_max phonemes
    assert len(first) / SAMPLE_RATE < 1.5          # the first sound comes early


def test_stream_pieces_commafree_unchanged(tts):
    text = 'Saya akan menghantar dokumen itu melalui emel esok pagi kepada pihak pentadbiran utama syarikat.'
    assert tts.pieces(text) == tts.phonemize(text)   # no commas, so no new boundaries


def test_stream_long_piece_cut_at_commas(tts):
    ph = tts.phonemize('Selamat pagi, boleh saya bantu encik hari ini?')[0]
    assert len(ph) <= 60 or tts.pieces('Selamat pagi, boleh saya bantu encik hari ini?') != [ph]


def test_stream_empty(tts):
    assert list(tts.stream('')) == []
    assert len(tts('')) == 0
