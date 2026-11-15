// node web/tests/node_parity.mjs  — transformers.js in Node against the Python runtime's ids and loudness envelope.
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { G2P } from '../g2p/ms_g2p.js';
import { normalize } from '../g2p/ms_norm.js';
import { Suaradee, G2P_OPTIONS, SAMPLE_RATE } from '../suaradee.js';

const here = dirname(fileURLToPath(import.meta.url));
const load = (f) => JSON.parse(readFileSync(join(here, '..', 'g2p', f), 'utf8'));
const g2p = new G2P({ vocab: load('vocab.json'), lexE: load('lexicon_e.json'), lexX: load('lexicon_exceptions.json'), lexEn: load('lexicon_en.json') }, G2P_OPTIONS);
const modelDir = resolve(here, '..', '..', 'export', 'ms_bella_8m');
const ref = JSON.parse(readFileSync(join(here, 'ref.json'), 'utf8'));

const tts = await Suaradee.load({ model: 'ms_bella_8m', g2p, local: true, modelRoot: resolve(here, '..', '..', 'export') + '/', dtype: process.env.DTYPE || 'q8' });
let bad = 0;
for (const r of ref) {
  const ph = tts.phonemize(r.text)[0];
  const ids = tts.ids(ph);
  const sameIds = JSON.stringify(ids) === JSON.stringify(r.ids);
  const y = await tts.synth(ph);
  const env = [];
  for (let i = 0; i + 2400 < y.length; i += 2400) { let s = 0; for (let k = 0; k < 2400; k++) s += y[i + k] ** 2; env.push(Math.sqrt(s / 2400)); }
  const n = Math.min(env.length, r.env.length);
  const a = env.slice(0, n), b = r.env.slice(0, n);
  const ma = a.reduce((x, v) => x + v, 0) / n, mb = b.reduce((x, v) => x + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) ** 2; sbb += (b[i] - mb) ** 2; }
  const corr = sab / Math.sqrt(saa * sbb);
  const lenOk = Math.abs(y.length - r.samples) <= 0.02 * r.samples;
  const ok = sameIds && lenOk && corr >= 0.95;
  if (!ok) bad++;
  console.log(`${ok ? 'ok ' : 'BAD'} ids ${sameIds ? 'same' : 'DIFFER'} | samples ${y.length} vs ${r.samples} | envelope corr ${corr.toFixed(3)} | ${r.text.slice(0, 40)}`);
}
const { audio } = await tts.speak(ref[3].text);
console.log(`speak(): ${(audio.length / SAMPLE_RATE).toFixed(2)} s`);
console.log(bad === 0 ? 'NODE PARITY PASS' : `NODE PARITY FAIL (${bad})`);
process.exit(bad === 0 ? 0 : 1);
