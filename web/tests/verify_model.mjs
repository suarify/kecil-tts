// Verify a candidate ONNX export under the web stack (transformers.js + onnxruntime-node, q8):
// the same graph ops WebGPU/WASM use, so a pass here means the browser demo will load it.
//   node web/tests/verify_model.mjs <model folder under export/>
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readFileSync } from 'node:fs';
import { G2P } from '../g2p/ms_g2p.js';
import { Suaradee, G2P_OPTIONS } from '../suaradee.js';

const here = dirname(fileURLToPath(import.meta.url));
const name = process.argv[2] || 'ms_clean_check';
const load = (f) => JSON.parse(readFileSync(join(here, '..', 'g2p', f), 'utf8'));
const g2p = new G2P({ vocab: load('vocab.json'), lexE: load('lexicon_e.json'), lexX: load('lexicon_exceptions.json'), lexEn: load('lexicon_en.json') }, G2P_OPTIONS);

const tts = await Suaradee.load({
  model: name,
  g2p,
  local: true,
  modelRoot: resolve(here, '..', '..', 'export') + '/',
  dtype: process.env.DTYPE || 'q8',
});
let bad = 0;
for (const text of [
  'Pesanan cik akan dihantar ke Seremban esok pagi.',
  'Driver dah on the way ke Cyberjaya.',
  'Kod pengesahan anda ialah 591670.',
  'Okey, baik. Ya, betul.',
]) {
  const ph = tts.phonemize(text)[0];
  const y = await tts.synth(ph);
  const sec = y.length / 24000;
  let s = 0;
  for (let i = 0; i < y.length; i++) s += y[i] * y[i];
  const rms = Math.sqrt(s / y.length);
  let peak = 0;
  for (let i = 0; i < y.length; i++) peak = Math.max(peak, Math.abs(y[i]));
  const ok = sec > 0.5 && sec < 20 && rms > 0.01 && peak < 1.5 && Number.isFinite(rms);
  if (!ok) bad++;
  console.log(`${ok ? 'ok ' : 'BAD'} ${sec.toFixed(2)} s  rms ${rms.toFixed(3)}  peak ${peak.toFixed(2)}  ${text.slice(0, 45)}`);
}
console.log(bad === 0 ? 'MODEL OK (web stack)' : `${bad} failures`);
process.exit(bad === 0 ? 0 : 1);
