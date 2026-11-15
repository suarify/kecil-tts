// Streaming example (Node 18+): write each 16-bit PCM piece to stdout as it is synthesised, so
//   node examples/stream_node.mjs | ffplay -f s16le -ar 24000 -
// starts playing before the rest is generated.
import { Suaradee, SAMPLE_RATE } from '../js/suaradee.js';
import { G2P } from '../js/g2p/ms_g2p.js';
import { readFileSync } from 'node:fs';

const load = (f) => JSON.parse(readFileSync(new URL('../js/g2p/' + f, import.meta.url), 'utf8'));
const g2p = new G2P({ vocab: load('vocab.json'), lexE: load('lexicon_e.json'), lexX: load('lexicon_exceptions.json'), lexEn: load('lexicon_en.json') },
  { stress: false, r: 'r', diph: 'glide' });
const tts = await Suaradee.load({ model: 'lseanlon/suaradee-v0.2', g2p });

const TEXT = 'Selamat pagi, boleh saya bantu encik hari ini? Jumlah bayaran akaun anda ialah RM1,250.50. ' +
  'Terima kasih kerana menghubungi kami, sila tunggu sebentar.';

const started = performance.now();
// The wave header is not known up front when streaming; write a header that claims a large size,
// which ffplay and most players accept.
const header = Buffer.alloc(44);
header.write('RIFF', 0); header.writeUInt32LE(0x7fffff24, 4); header.write('WAVE', 8);
header.write('fmt ', 12); header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20);
header.writeUInt16LE(1, 22); header.writeUInt32LE(SAMPLE_RATE, 24);
header.writeUInt32LE(SAMPLE_RATE * 2, 28); header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34);
header.write('data', 36); header.writeUInt32LE(0x7fffff00, 40);
process.stdout.write(header);

let first = null, samples = 0;
for await (const chunk of tts.stream(TEXT)) {
  if (first === null) {
    first = (performance.now() - started) / 1000;
    process.stderr.write(`first audio after ${first.toFixed(2)} s\n`);
  }
  const pcm = Buffer.alloc(chunk.length * 2);
  chunk.forEach((x, i) => pcm.writeInt16LE(Math.round(Math.max(-1, Math.min(1, x)) * 32767), i * 2));
  process.stdout.write(pcm);
  samples += chunk.length;
}
process.stderr.write(`total ${((performance.now() - started) / 1000).toFixed(2)} s, ${tts.pieces(TEXT).length} pieces\n`);
