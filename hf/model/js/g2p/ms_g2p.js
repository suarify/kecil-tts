// Malay G2P for Kokoro (kokoro-malay-bella). Line-for-line port of ms_g2p.py.
//
//   import { G2P, loadG2P } from './ms_g2p.js';
//   const g2p = await loadG2P('./g2p/');          // browser: fetches the JSON lexicons
//   const ph = g2p.phonemize('Selamat pagi');
//   const ids = g2p.tokens(ph);                    // [0, ..., 0]

const LETTERS = {
  a: 'ˈA', b: 'bˈi', c: 'sˈi', d: 'dˈi', e: 'ˈi', f: 'ˈɛf', g: 'ʤˈi',
  h: 'ˈAʧ', i: 'ˈI', j: 'ʤˈA', k: 'kˈA', l: 'ˈɛl', m: 'ˈɛm', n: 'ˈɛn',
  o: 'ˈO', p: 'pˈi', q: 'kjˈu', r: 'ˈɑr', s: 'ˈɛs', t: 'tˈi', u: 'jˈu',
  v: 'vˈi', w: 'dˈʌbəljˌu', x: 'ˈɛks', y: 'wˈI', z: 'zˈɛd',
};

const FUNCTION_WORDS = new Set([
  'di', 'ke', 'dari', 'daripada', 'yang', 'dan', 'atau', 'untuk', 'dengan', 'pada',
  'kepada', 'akan', 'pun', 'lah', 'kah', 'tah', 'si', 'sang', 'para', 'oleh', 'bagi',
  'ia', 'nya', 'ini', 'itu', 'tu', 'ni', 'je', 'la', 'kan', 'tak',
]);

const PREFIX_OUTER = [
  ['meng', ['', 'k']], ['meny', ['s']], ['mem', ['', 'p']], ['men', ['', 't']], ['me', ['']],
  ['peng', ['', 'k']], ['peny', ['s']], ['pem', ['', 'p']], ['pen', ['', 't']],
  ['per', ['']], ['pel', ['']], ['pe', ['']],
  ['ber', ['']], ['bel', ['']], ['be', ['']], ['ter', ['']], ['te', ['']],
  ['di', ['']], ['ke', ['']], ['se', ['']],
];
const PREFIX_INNER = ['per', 'ber', 'ter', 'ke', 'se', 'pe'];
const SUF_DERIV = ['kan', 'an', 'i'];
const SUF_ENCL = ['nya', 'lah', 'kah', 'tah', 'ku', 'mu'];
const SUFFIX_STRINGS = new Set(['']);
for (const d of ['', ...SUF_DERIV]) for (const e of ['', ...SUF_ENCL]) SUFFIX_STRINGS.add(d + e);

const VOWEL_LETTERS = 'aeéiou';
const NUCLEI = new Set('aeəiouAIOWY');
const SINGLE = {
  a: 'a', 'é': 'e', e: 'ə', i: 'i', o: 'o', u: 'u',
  b: 'b', c: 'ʧ', d: 'd', f: 'f', g: 'ɡ', h: 'h', j: 'ʤ', k: 'k',
  l: 'l', m: 'm', n: 'n', p: 'p', q: 'k', s: 's', t: 't', v: 'v',
  w: 'w', y: 'j', z: 'z',
};
const DIGRAPHS = {
  ng: 'ŋ', ny: 'ɲ', sy: 'ʃ', sh: 'ʃ', kh: 'x', gh: 'ɣ', ch: 'ʧ',
  th: 't', dh: 'd', ph: 'f', dz: 'z',
};
const DIPHTHONGS = { ai: ['aj', 'I'], au: ['aw', 'W'], oi: ['oj', 'Y'], 'éi': ['ej', 'A'] };
const PUNCT = ',.!?;:…—';
const TOKEN_RE = /[A-Za-zé']+|[,.!?;:…—]/g;

const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
const isVowelLetter = (c) => c !== '' && VOWEL_LETTERS.includes(c);

export function stripAccents(text) {
  return text.normalize('NFD').replace(/\p{Mn}/gu, '').normalize('NFC');
}

export class G2P {
  constructor({ vocab, lexE = {}, lexX = {}, lexEn = {} }, { stress = true, r = 'r', diph = 'glide' } = {}) {
    this.vocab = vocab;
    this.lexE = lexE;
    this.lexX = lexX;
    this.lexEn = lexEn;
    this.stress = stress;
    this.r = r;
    this.diph = diph === 'glide' ? 0 : 1;
  }

  *_prefixSplits(w) {
    yield ['', w, 0];
    for (const [p, restores] of PREFIX_OUTER) {
      if (!w.startsWith(p) || w.length <= p.length) continue;
      const rest = w.slice(p.length);
      for (const rs of restores) {
        yield [p, rs + rest, rs.length];
        if (rs === '') {
          for (const q of PREFIX_INNER) {
            if (rest.startsWith(q) && rest.length > q.length) yield [p + q, rest.slice(q.length), 0];
          }
        }
      }
    }
  }

  _decompose(w) {
    let best = null;
    let bestFree = null;
    const better = (c, b) => b === null || c[0] > b[0] || (c[0] === b[0] && c[1] > b[1]);
    for (const encl of ['', ...SUF_ENCL]) {
      if (encl && !w.endsWith(encl)) continue;
      const w1 = w.slice(0, w.length - encl.length);
      for (const der of ['', ...SUF_DERIV]) {
        if (der && !w1.endsWith(der)) continue;
        const w2 = w1.slice(0, w1.length - der.length);
        const suffix = der + encl;
        for (const [prefix, root, drop] of this._prefixSplits(w2)) {
          if (root.length < 2) continue;
          const nAff = prefix.length + suffix.length;
          const cand = [root.length, -nAff, prefix, root, drop, suffix];
          if (has(this.lexE, root)) {
            if (better(cand, best)) best = cand;
          } else if (!root.includes('e') && prefix &&
                     [...root].filter((c) => VOWEL_LETTERS.includes(c)).length >= 2) {
            if (better(cand, bestFree)) bestFree = cand;
          }
        }
      }
    }
    const pick = best || bestFree;
    if (pick === null) return null;
    const [, , prefix, root, drop, suffix] = pick;
    const spelled = has(this.lexE, root) ? this.lexE[root] : root;
    return prefix + spelled.slice(drop) + suffix;
  }

  static _heuristic(w) {
    const idx = [];
    for (let i = 0; i < w.length; i++) if (VOWEL_LETTERS.includes(w[i])) idx.push(i);
    if (!idx.length) return w;
    const last = idx[idx.length - 1];
    const chars = [...w];
    if (w[last] === 'e') {
      const coda = w.slice(last + 1);
      if (coda === '' || (!'mnr'.includes(coda[coda.length - 1]) && coda !== 'ng')) chars[last] = 'é';
    }
    return chars.join('');
  }

  respell(w) {
    if (!w.includes('e')) return w;
    if (has(this.lexE, w)) return this.lexE[w];
    const d = this._decompose(w);
    if (d !== null) return d;
    return G2P._heuristic(w);
  }

  _letters(s) {
    const out = [];
    const n = s.length;
    let i = 0;
    while (i < n) {
      const c = s[i];
      const two = s.slice(i, i + 2);
      if ('aoé'.includes(c) && has(DIPHTHONGS, two) && SUFFIX_STRINGS.has(s.slice(i + 2))) {
        const ph = DIPHTHONGS[two][this.diph];
        G2P._pushVowel(out, ph[0]);
        out.push(...ph.slice(1));
        i += 2;
        continue;
      }
      if (has(DIGRAPHS, two)) {
        out.push(DIGRAPHS[two]);
        i += 2;
        continue;
      }
      if (c === "'") {
        if (out.length && i + 1 < n) out.push('ʔ');
        i += 1;
        continue;
      }
      if (c === 'k') {
        const prevVowel = i > 0 && isVowelLetter(s[i - 1]);
        const nxt = i + 1 < n ? s[i + 1] : '';
        if (prevVowel && (nxt === '' || (!isVowelLetter(nxt) && !"srl'".includes(nxt)))) out.push('ʔ');
        else out.push('k');
        i += 1;
        continue;
      }
      if (c === 'x') {
        if (i === 0) out.push('z');
        else out.push('k', 's');
        i += 1;
        continue;
      }
      if (c === 'r') {
        out.push(this.r);
        i += 1;
        continue;
      }
      if (!has(SINGLE, c)) {
        i += 1;
        continue;
      }
      const ph = SINGLE[c];
      if (VOWEL_LETTERS.includes(c)) G2P._pushVowel(out, ph);
      else out.push(ph);
      i += 1;
    }
    return out;
  }

  static _pushVowel(out, v) {
    const last = out.length ? out[out.length - 1] : '';
    if (last && NUCLEI.has(last) && (last === 'ə' || last === v)) out.push('ʔ');
    out.push(v);
  }

  _stress(ph, w) {
    if (!this.stress || FUNCTION_WORDS.has(w)) return ph;
    const nuc = [];
    ph.forEach((p, i) => { if (NUCLEI.has(p)) nuc.push(i); });
    if (!nuc.length) return ph;
    let k = nuc.length >= 2 ? nuc[nuc.length - 2] : nuc[nuc.length - 1];
    if (nuc.length >= 2 && ph[k] === 'ə') k = nuc[nuc.length - 1];
    return [...ph.slice(0, k), 'ˈ', ...ph.slice(k)];
  }

  word(tok) {
    if (tok.length >= 2 && isUpper(tok) && /^[A-Za-zé]+$/.test(tok) && !has(this.lexX, tok.toLowerCase())) {
      return [...tok.toLowerCase()].filter((c) => has(LETTERS, c)).map((c) => LETTERS[c]).join(' ');
    }
    const w = tok.toLowerCase().replace(/^'+|'+$/g, '');
    if (!w) return '';
    if (has(this.lexX, w)) return this.lexX[w];
    if (has(this.lexEn, w)) return this.lexEn[w];
    return this._stress(this._letters(this.respell(w)), w).join('');
  }

  phonemize(text) {
    text = stripAccents(text).split('-').join(' ');
    let out = '';
    for (const tok of text.match(TOKEN_RE) || []) {
      if (PUNCT.includes(tok)) {
        out = out.replace(/ +$/, '') + tok + ' ';
        continue;
      }
      const ph = this.word(tok);
      if (ph) out += ph + ' ';
    }
    return out.trim();
  }

  tokens(phonemes) {
    const ids = [0];
    for (const c of phonemes) if (has(this.vocab, c)) ids.push(this.vocab[c]);
    ids.push(0);
    return ids;
  }
}

// Python str.isupper(): at least one cased character and no lowercase ones.
function isUpper(s) {
  return s !== s.toLowerCase() && s === s.toUpperCase();
}

export function chunk(phonemes, limit = 510) {
  const out = [];
  let rest = phonemes.trim();
  while (rest.length > limit) {
    let cut = -1;
    for (const marks of ['.!?…', ';:—', ',', ' ']) {
      for (let j = limit - 1; j > Math.floor(limit / 3); j--) {
        if (marks.includes(rest[j])) { cut = j + 1; break; }
      }
      if (cut > 0) break;
    }
    if (cut <= 0) cut = limit;
    out.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trim();
  }
  if (rest) out.push(rest);
  return out;
}

export async function loadG2P(base = './', opts = {}) {
  const get = async (name, optional = false) => {
    const r = await fetch(base + name);
    if (!r.ok) {
      if (optional) return {};
      throw new Error(`failed to load ${name}: ${r.status}`);
    }
    return r.json();
  };
  const [vocab, lexE, lexX, lexEn] = await Promise.all([
    get('vocab.json'), get('lexicon_e.json'), get('lexicon_exceptions.json', true), get('lexicon_en.json', true),
  ]);
  return new G2P({ vocab, lexE, lexX, lexEn }, opts);
}
