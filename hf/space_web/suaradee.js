// Suaradee: Malay text -> phonemes (ms_norm + ms_g2p) -> 8 MB ONNX model through transformers.js -> 24 kHz audio.
//
//   import { Suaradee } from './suaradee.js';
//   const tts = await Suaradee.load({ model: 'lseanlon/suaradee-v0.3', g2pBase: './g2p/' });   // WebGPU if available, else WASM
//   const { audio, sampleRate } = await tts.speak('Selamat pagi, boleh saya bantu?');
//
// The graph has two inputs, input_ids (int64 [1,T]) and speed (float32 [1]); one voice, no style input.
import { StyleTextToSpeech2Model, Tensor, env } from '@huggingface/transformers';
import { normalize } from './g2p/ms_norm.js';
import { chunk, loadG2P } from './g2p/ms_g2p.js';

export const SAMPLE_RATE = 24000;
const MAX_PHONEMES = 400;                       // trained on single sentences of at most 400 phonemes
export const FIRST_MAX = 60;                    // streaming: first piece at most this many phonemes, sound starts early
export const LATER_MAX = 160;                   // streaming: every later piece at most this many phonemes
export const G2P_OPTIONS = { stress: false, r: 'r', diph: 'glide' };   // must match how the training rows were made

export function splitSentences(text) {
  return text.split(/(?<=[.!?…])\s+|\n+/).map((s) => s.trim()).filter(Boolean);
}

export class Suaradee {
  constructor(model, g2p, device) { this.model = model; this.g2p = g2p; this.device = device; }

  // model: Hugging Face repo id, or a folder name under modelRoot when local is true.
  // device: 'auto' (WebGPU when the browser has it and it works, else WASM), 'webgpu', 'wasm' or 'cpu' (Node).
  static async load({ model = 'lseanlon/suaradee-v0.3', g2p, g2pBase = './g2p/', dtype = 'q8', device = 'auto', local = false, modelRoot, sessionOptions } = {}) {
    if (local) { env.allowLocalModels = true; env.allowRemoteModels = false; env.localModelPath = modelRoot; }
    const inNode = typeof window === 'undefined';
    // Embedded pages (e.g. the Hugging Face Space iframe) are not cross-origin isolated, so
    // SharedArrayBuffer is unavailable and ORT's threaded WASM cannot start. Single-threaded WASM
    // needs no SAB and is the only backend an embed can use when WebGPU is denied.
    if (!inNode && typeof self !== 'undefined' && !self.crossOriginIsolated && env.backends?.onnx?.wasm) {
      env.backends.onnx.wasm.numThreads = 1;
    }
    // navigator.gpu can exist while the page is denied an adapter (embedded frames), so ask for one.
    const webgpuOk = async () => {
      if (inNode || typeof navigator === 'undefined' || !navigator.gpu) return false;
      try { return !!(await navigator.gpu.requestAdapter()); } catch { return false; }
    };
    const tries = device !== 'auto' ? [device] : inNode ? ['cpu'] : ((await webgpuOk()) ? ['webgpu', 'wasm'] : ['wasm']);
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

  // Cut a phoneme string at commas only: first piece at most firstLimit symbols, later pieces at most
  // laterLimit. A piece with no comma inside its limit is kept whole, so comma-free texts never change.
  static splitCommas(phonemes, firstLimit, laterLimit) {
    const out = [];
    let rest = phonemes.trim();
    let first = true;
    for (;;) {
      const limit = first ? firstLimit : laterLimit;
      if (rest.length <= limit) { out.push(rest); return out; }
      const cut = rest.lastIndexOf(',', limit);
      if (cut < 0) { out.push(rest); return out; }
      out.push(rest.slice(0, cut).trim());
      rest = rest.slice(cut + 1).trim();
      first = false;
      if (!rest) return out;
    }
  }

  // Phoneme pieces for streaming: sentences as phonemize() makes them, with pieces that exceed the
  // streaming limit cut at commas.
  pieces(text, firstMax = FIRST_MAX) {
    const out = [];
    for (const ph of this.phonemize(text)) out.push(...Suaradee.splitCommas(ph, firstMax, LATER_MAX));
    return out;
  }

  // Yields one Float32Array per piece at SAMPLE_RATE; a pause-length gap between pieces. The consumer
  // gets the first piece before the rest is synthesised.
  async *stream(text, { speed = 1.0, pause = 0.12, firstMax = FIRST_MAX } = {}) {
    const gap = new Float32Array(Math.round(pause * SAMPLE_RATE));
    let emitted = false;
    for (const p of this.pieces(text, firstMax)) {
      if (!/\p{L}/u.test(p)) continue;          // punctuation alone is not speech
      if (emitted) yield gap.slice();
      yield await this.synth(p, speed);
      emitted = true;
    }
  }

  async speak(text, opts = {}) {
    const parts = [];
    let total = 0;
    for await (const y of this.stream(text, opts)) { parts.push(y); total += y.length; }
    const audio = new Float32Array(total);
    let o = 0;
    for (const p of parts) { audio.set(p, o); o += p.length; }
    return { audio, sampleRate: SAMPLE_RATE };
  }
}
