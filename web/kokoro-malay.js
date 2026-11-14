// Malay text -> phonemes (ms_norm + ms_g2p, same code as Python) -> 8M ONNX model via transformers.js -> 24 kHz audio.
//
//   import { KokoroMalay } from './kokoro-malay.js';
//   const tts = await KokoroMalay.load({ modelDir: './export/ms_bella_8m', g2pBase: './g2p/' });
//   const { audio, sampleRate } = await tts.speak('Selamat pagi, boleh saya bantu?');
//
// The graph has two inputs, input_ids (int64 [1,T]) and speed (float32 [1]); one voice, no style input.
import { StyleTextToSpeech2Model, Tensor, env } from '@huggingface/transformers';
import { normalize } from './g2p/ms_norm.js';
import { G2P, chunk, loadG2P } from './g2p/ms_g2p.js';

export const SAMPLE_RATE = 24000;
const MAX_PHONEMES = 400;                       // the student was trained on single sentences of at most 400 phonemes
export const G2P_OPTIONS = { stress: false, r: 'r', diph: 'glide' };   // must match how the training rows were made

export function splitSentences(text) {
  return text.split(/(?<=[.!?…])\s+|\n+/).map((s) => s.trim()).filter(Boolean);
}

export class KokoroMalay {
  constructor(model, g2p) { this.model = model; this.g2p = g2p; }

  // g2p: either an already built G2P instance or { base } to fetch the lexicons (browser) .
  static async load({ modelDir, g2p, g2pBase = './g2p/', dtype = 'q8', device = typeof window === 'undefined' ? 'cpu' : 'wasm', local = false } = {}) {
    if (local) { env.allowLocalModels = true; env.allowRemoteModels = false; env.localModelPath = modelDir.replace(/[^/\\]+[/\\]?$/, ''); }
    const name = local ? modelDir.replace(/[/\\]+$/, '').split(/[/\\]/).pop() : modelDir;
    const model = await StyleTextToSpeech2Model.from_pretrained(name, { dtype, device });
    return new KokoroMalay(model, g2p || (await loadG2P(g2pBase, G2P_OPTIONS)));
  }

  phonemize(text) {
    const out = [];
    for (const sent of splitSentences(normalize(text))) {
      const ph = this.g2p.phonemize(sent);
      if (!/\p{L}/u.test(ph)) continue;         // punctuation alone is not speech
      out.push(...(ph.length <= MAX_PHONEMES ? [ph] : chunk(ph, MAX_PHONEMES)));
    }
    return out;
  }

  ids(phonemes) { return this.g2p.tokens(phonemes); }

  async synth(phonemes, speed = 1.0) {
    const ids = this.ids(phonemes);
    const input_ids = new Tensor('int64', BigInt64Array.from(ids, (v) => BigInt(v)), [1, ids.length]);
    const out = await this.model({ input_ids, speed: new Tensor('float32', new Float32Array([speed]), [1]) });
    return out.waveform.data;
  }

  async speak(text, { speed = 1.0, pause = 0.12 } = {}) {
    const parts = [];
    let total = 0;
    const gap = new Float32Array(Math.round(pause * SAMPLE_RATE));
    for (const p of this.phonemize(text)) {
      const y = await this.synth(p, speed);
      if (parts.length) { parts.push(gap); total += gap.length; }
      parts.push(y); total += y.length;
    }
    const audio = new Float32Array(total);
    let o = 0;
    for (const p of parts) { audio.set(p, o); o += p.length; }
    return { audio, sampleRate: SAMPLE_RATE };
  }
}
