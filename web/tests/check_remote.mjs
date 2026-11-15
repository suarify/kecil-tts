// Load the published model from the Hub with transformers.js and synthesize one sentence.
//   node web/tests/check_remote.mjs [repo]
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { G2P } from '../g2p/ms_g2p.js';
import { Suaradee, G2P_OPTIONS } from '../suaradee.js';

const here = dirname(fileURLToPath(import.meta.url));
const repo = process.argv[2] || 'Suarify/suaradee-v0.3';
const load = (f) => JSON.parse(readFileSync(join(here, '..', 'g2p', f), 'utf8'));
const g2p = new G2P({ vocab: load('vocab.json'), lexE: load('lexicon_e.json'), lexX: load('lexicon_exceptions.json'), lexEn: load('lexicon_en.json') }, G2P_OPTIONS);

const t = Date.now();
const tts = await Suaradee.load({ model: repo, g2p });
console.log('loaded', repo, 'in', ((Date.now() - t) / 1000).toFixed(1), 's on', tts.device);
const { audio } = await tts.speak('Selamat pagi, boleh saya bantu encik hari ini?');
const sec = audio.length / 24000;
let s = 0;
for (let i = 0; i < audio.length; i++) s += audio[i] * audio[i];
console.log(`audio ${sec.toFixed(2)} s rms ${Math.sqrt(s / audio.length).toFixed(3)}`);
console.log(sec > 1 && sec < 15 ? 'REMOTE OK' : 'REMOTE BAD');
