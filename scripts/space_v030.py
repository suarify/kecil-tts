"""Apply the Space edits for v0.3.0: i18n keys, dropdown, links, and the benchmarks card."""
p = 'runs/local_v041/index.html'
s = open(p, encoding='utf-8').read()

en_old = "      h1s: '1 · Choose a speaker', h2s: '2 · Type Malay text', fxh: 'Effects lab (optional)',"
en_new = ("      h1s: '1 · Choose a speaker', h2s: '2 · Type Malay text', fxh: 'Effects lab (optional)', "
          "hbench: 'Benchmarks',\n"
          "      nbench: 'CER measured 2026-11-17 with Whisper large-v3 on 200 test + 60 held-out sentences; the "
          "v0.3.0 weights themselves are still being re-scored, so the numbers shown are from the checkpoint they "
          "continue (same text-model scope). Texture and periodicity measured on the same 6 sentences for both "
          "voices. No MOS or side-by-side human rating exists for either voice at this date.',")
bm_old = "      h1s: '1 · Pilih suara', h2s: '2 · Taip teks Melayu', fxh: 'Makmal kesan (pilihan)',"
bm_new = ("      h1s: '1 · Pilih suara', h2s: '2 · Taip teks Melayu', fxh: 'Makmal kesan (pilihan)', "
          "hbench: 'Penanda aras',\n"
          "      nbench: 'CER diukur 2026-11-17 dengan Whisper large-v3 pada 200 ayat ujian + 60 ayat held-out; "
          "pemberat v0.3.0 sendiri masih dinilai semula, jadi angka yang ditunjukkan adalah daripada checkpoint "
          "yang diteruskan (skop model teks yang sama). Tekstur dan periodicity diukur pada 6 ayat yang sama untuk "
          "kedua-dua suara. Tiada penilaian MOS atau perbandingan manusia pada tarikh ini.',")
for old, new in ((en_old, en_new), (bm_old, bm_new)):
    assert old in s, old[:60]
    s = s.replace(old, new)

vers_old = ("versions: [['Suarify/kecil-tts-michael-v0.2.2', 'v0.2.2 (latest)'], "
            "['Suarify/kecil-tts-michael-v0.2.1', 'v0.2.1'], ['Suarify/kecil-tts-michael-v0.2', 'v0.2'], "
            "['Suarify/kecil-tts-michael-v0.1', 'v0.1']]")
vers_new = ("versions: [['Suarify/kecil-tts-michael-v0.3.0', 'v0.3.0 (latest, retrained)'], "
            "['Suarify/kecil-tts-michael-v0.2.2', 'v0.2.2'], ['Suarify/kecil-tts-michael-v0.2.1', 'v0.2.1'], "
            "['Suarify/kecil-tts-michael-v0.2', 'v0.2'], ['Suarify/kecil-tts-michael-v0.1', 'v0.1']]")
assert vers_old in s
s = s.replace(vers_old, vers_new)

s = s.replace("card: 'https://huggingface.co/Suarify/kecil-tts-michael-v0.2.2'",
              "card: 'https://huggingface.co/Suarify/kecil-tts-michael-v0.3.0'")
s = s.replace('href="https://huggingface.co/Suarify/kecil-tts-michael-v0.2" target="_blank" rel="noopener">Michael v0.2</a>',
              'href="https://huggingface.co/Suarify/kecil-tts-michael-v0.3.0" target="_blank" rel="noopener">Michael v0.3.0</a>')
open(p, 'w', encoding='utf-8').write(s)
print('edits applied | v0.3.0 in dropdown:', 'michael-v0.3.0' in s, '| benchmarks card:', 'class="card bench"' in s)
