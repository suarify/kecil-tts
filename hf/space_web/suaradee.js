// Suaradee: Malay text -> phonemes (ms_norm + ms_g2p) -> 8 MB ONNX model through transformers.js -> 24 kHz audio.
//
//   import { Suaradee } from './suaradee.js';
//   const tts = await Suaradee.load({ model: 'lseanlon/suaradee-v0.1', g2pBase: './g2p/' });   // WebGPU if available, else WASM
//   const { audio, sampleRate } = await tts.speak('Selamat pagi, boleh saya bantu?');
//
// The graph has two inputs, input_ids (int64 [1,T]) and speed (float32 [1]); one voice, no style input.
import { StyleTextToSpeech2Model, Tensor, env } from '@huggingface/transformers';
import { normalize } from './g2p/ms_norm.js';
import { chunk, loadG2P } from './g2p/ms_g2p.js';

export const SAMPLE_RATE = 24000;
const MAX_PHONEMES = 400;                       // trained on single sentences of at most 400 phonemes
export const G2P_OPTIONS = { stress: false, r: 'r', diph: 'glide' };   // must match how the training rows were made

export function splitSentences(text) {
  return text.split(/(?<=[.!?…])\s+|\n+/).map((s) => s.trim()).filter(Boolean);
}

export class Suaradee {
  constructor(model, g2p, device) { this.model = model; this.g2p = g2p; this.device = device; }

  // model: Hugging Face repo id, or a folder name under modelRoot when local is true.
  // device: 'auto' (WebGPU when the browser has it and it works, else WASM), 'webgpu', 'wasm' or 'cpu' (Node).
  static async load({ model = 'lseanlon/suaradee-v0.1', g2p, g2pBase = './g2p/', dtype = 'q8', device = 'auto', local = false, modelRoot, sessionOptions } = {}) {
    if (local) { env.allowLocalModels = true; env.allowRemoteModels = false; env.localModelPath = modelRoot; }
    const inNode = typeof window === 'undefined';
    const tries = device !== 'auto' ? [device] : inNode ? ['cpu'] : (typeof navigator !== 'undefined' && navigator.gpu ? ['webgpu', 'wasm'] : ['wasm']);
    const rules = g2p || (await loadG2P(g2pBase, G2P_OPTIONS));
    let lastErr;
    for (const d of tries) {
      try {
        const m = await StyleTextToSpeech2Model.from_pretrained(model, { dtype, device: d, ...(sessionOptions ? { session_options: sessionOptions } : {}) });
        const inst = new Suaradee(m, rules, d);
        await inst.synth(inst.phonemize('Selamat pagi.')[0]);       // warm-up: surfaces unsupported operators now, not on first use
        return inst;
      } catch (e) { lastErr = e; console.warn(`Suaradee: device ${d} failed`, e); }
    }
    throw lastErr;
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
