// JS vs Python parity: node g2p/tests/parity.mjs
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalize } from '../ms_norm.js';
import { G2P } from '../ms_g2p.js';

const here = dirname(fileURLToPath(import.meta.url));
const load = (f) => JSON.parse(readFileSync(join(here, '..', f), 'utf8'));
const lex = { vocab: load('vocab.json'), lexE: load('lexicon_e.json'), lexX: load('lexicon_exceptions.json'), lexEn: load('lexicon_en.json') };

const rows = readFileSync(join(here, 'golden.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const cache = new Map();
let normBad = 0, phBad = 0;
for (const r of rows) {
  const key = JSON.stringify(r.opts);
  if (!cache.has(key)) cache.set(key, new G2P(lex, r.opts));
  const n = normalize(r.text);
  if (n !== r.norm) {
    if (normBad++ < 10) console.log('NORM MISMATCH\n  text:', r.text, '\n  py:  ', r.norm, '\n  js:  ', n);
    continue;
  }
  const ph = cache.get(key).phonemize(n);
  if (ph !== r.ph) {
    if (phBad++ < 10) console.log('G2P MISMATCH', key, '\n  norm:', n, '\n  py:  ', r.ph, '\n  js:  ', ph);
  }
}
const ok = rows.length - normBad - phBad;
console.log(`parity: ${ok}/${rows.length} identical (normaliser mismatches ${normBad}, g2p mismatches ${phBad})`);
process.exit(normBad + phBad === 0 ? 0 : 1);
