// npm i @huggingface/transformers   (Node 18+); run from the repo root:  node examples/node_example.mjs
import { writeFileSync } from 'node:fs';
import { Suaradee, SAMPLE_RATE } from '../js/suaradee.js';
import { G2P } from '../js/g2p/ms_g2p.js';
import { readFileSync } from 'node:fs';

const load = (f) => JSON.parse(readFileSync(new URL('../js/g2p/' + f, import.meta.url), 'utf8'));
const g2p = new G2P({ vocab: load('vocab.json'), lexE: load('lexicon_e.json'), lexX: load('lexicon_exceptions.json'), lexEn: load('lexicon_en.json') },
  { stress: false, r: 'r', diph: 'glide' });
const tts = await Suaradee.load({ model: 'Suarify/kecil-tts-yati-v0.4', g2p });
const { audio } = await tts.speak('Selamat pagi, boleh saya bantu encik hari ini?');
const pcm = Buffer.alloc(44 + audio.length * 2);
pcm.write('RIFF', 0); pcm.writeUInt32LE(36 + audio.length * 2, 4); pcm.write('WAVEfmt ', 8); pcm.writeUInt32LE(16, 16); pcm.writeUInt16LE(1, 20);
pcm.writeUInt16LE(1, 22); pcm.writeUInt32LE(SAMPLE_RATE, 24); pcm.writeUInt32LE(SAMPLE_RATE * 2, 28); pcm.writeUInt16LE(2, 32); pcm.writeUInt16LE(16, 34);
pcm.write('data', 36); pcm.writeUInt32LE(audio.length * 2, 40);
audio.forEach((x, i) => pcm.writeInt16LE(Math.round(Math.max(-1, Math.min(1, x)) * 32767), 44 + i * 2));
writeFileSync('hello.wav', pcm);
console.log('hello.wav written');
