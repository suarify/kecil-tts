// Malay text normaliser for kokoro-malay-bella. Line-for-line port of ms_norm.py.
// Regexes avoid the 'u' flag so \d, \w and \b are ASCII, matching Python's re.ASCII.

const ONES = ['kosong', 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'lapan', 'sembilan'];
const MONTHS = ['Januari', 'Februari', 'Mac', 'April', 'Mei', 'Jun', 'Julai', 'Ogos',
  'September', 'Oktober', 'November', 'Disember'];
const MONTH_ABBR = {
  jan: 1, feb: 2, mac: 3, mar: 3, apr: 4, mei: 5, may: 5, jun: 6,
  jul: 7, ogo: 8, aug: 8, sep: 9, okt: 10, oct: 10, nov: 11, dis: 12, dec: 12,
};
const MONTH_RE = 'januari|january|jan|februari|february|feb|mac|march|mar|april|apr|mei|may|' +
  'jun|june|julai|july|jul|ogos|august|aug|september|sept|sep|oktober|october|okt|oct|' +
  'november|nov|disember|december|dis|dec';

const ABBR = [
  ['sdn. bhd.', 'sendirian berhad'], ['sdn bhd', 'sendirian berhad'],
  ['bhd.', 'berhad'], ['sdn.', 'sendirian'],
  ['dr.', 'doktor'], ['en.', 'encik'], ['pn.', 'puan'], ['cik.', 'cik'],
  ['tn.', 'tuan'], ['hj.', 'haji'], ['hjh.', 'hajah'], ['prof.', 'profesor'],
  ['jln.', 'jalan'], ['jln', 'jalan'], ['tmn.', 'taman'], ['tmn', 'taman'],
  ['kg.', 'kampung'], ['lrg.', 'lorong'],
  ['dll.', 'dan lain-lain'], ['dll', 'dan lain-lain'], ['dsb.', 'dan sebagainya'],
  ['spt.', 'seperti'], ['spt', 'seperti'], ['yg', 'yang'], ['dgn', 'dengan'],
  ['utk', 'untuk'], ['kpd', 'kepada'], ['drp', 'daripada'], ['sbb', 'sebab'],
  ['tlg', 'tolong'], ['tq', 'terima kasih'], ['tkasih', 'terima kasih'],
  ['lg', 'lagi'], ['jgn', 'jangan'], ['sy', 'saya'], ['x', 'tak'],
  ['no.', 'nombor'], ['tel.', 'telefon'], ['ptg', 'petang'], ['pg', 'pagi'],
];

const UNITS = [
  ['km/j', 'kilometer sejam'], ['km/h', 'kilometer sejam'],
  ['°c', 'darjah Celsius'], ['km', 'kilometer'], ['kg', 'kilogram'],
  ['cm', 'sentimeter'], ['mm', 'milimeter'], ['ml', 'mililiter'],
  ['mg', 'miligram'], ['g', 'gram'], ['m', 'meter'], ['l', 'liter'],
];

const DIGIT_CONTEXT = '(?:kod|otp|pin|nombor|no\\.?|akaun|telefon|tel\\.?|ic|tac)';
const NUM = '\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?|\\d+(?:\\.\\d+)?';
const TIME_SFX = '(am|pm|a\\.m\\.|p\\.m\\.|pagi|petang|tengah hari|malam)';
const DIGITS = '0123456789';

function sub(pattern, repl, text, ci = false) {
  return text.replace(new RegExp(pattern, ci ? 'gi' : 'g'), repl);
}

// Wrap a Python-style callback: m(i) returns group i or ''.
function cb(fn) {
  return (...args) => {
    let n = args.length - 2;
    if (typeof args[n] !== 'number') n -= 1; // named groups object present
    const groups = args.slice(0, n);
    const m = (i) => groups[i] || '';
    return fn(m);
  };
}

function cardinal(n) {
  if (n < 10) return ONES[n];
  if (n === 10) return 'sepuluh';
  if (n === 11) return 'sebelas';
  if (n < 20) return ONES[n - 10] + ' belas';
  if (n < 100) {
    const t = Math.floor(n / 10), r = n % 10;
    return ONES[t] + ' puluh' + (r === 0 ? '' : ' ' + ONES[r]);
  }
  const scales = [[1e12, 'trilion', 'satu trilion'], [1e9, 'bilion', 'satu bilion'],
    [1e6, 'juta', 'satu juta'], [1000, 'ribu', 'seribu'], [100, 'ratus', 'seratus']];
  for (const [size, word, single] of scales) {
    if (n >= size) {
      const q = Math.floor(n / size), r = n % size;
      const head = q === 1 ? single : cardinal(q) + ' ' + word;
      return head + (r === 0 ? '' : ' ' + cardinal(r));
    }
  }
  throw new Error('cardinal ' + n);
}

function digits(s) {
  const out = [];
  for (const ch of s) {
    if (DIGITS.includes(ch)) out.push(ONES[+ch]);
    else if (ch === '+') out.push('tambah');
  }
  return out.join(' ');
}

function number(s) {
  s = s.replace(/,/g, '');
  const dot = s.indexOf('.');
  if (dot >= 0) {
    const whole = s.slice(0, dot), frac = s.slice(dot + 1);
    return cardinal(parseInt(whole || '0', 10)) + ' perpuluhan ' + digits(frac);
  }
  return cardinal(parseInt(s, 10));
}

function ordinal(n) {
  return n === 1 ? 'pertama' : 'ke' + cardinal(n);
}

function timeWords(h, m, suffix) {
  h = parseInt(h, 10); m = parseInt(m, 10);
  let part = '';
  if (suffix) {
    const sfx = suffix.toLowerCase().replace(/\./g, '');
    if (sfx === 'am') part = 'pagi';
    else if (sfx === 'pm') part = h === 12 ? 'tengah hari' : (h < 7 ? 'petang' : 'malam');
    else part = suffix;
  }
  const words = cardinal(h) + (m === 0 ? '' : ' ' + cardinal(m));
  return words + (part ? ' ' + part : '');
}

function stripAccents(text) {
  return text.normalize('NFD').replace(/\p{Mn}/gu, '').normalize('NFC');
}

const money = cb((m) => {
  const amount = m(2), scale = m(3);
  const unit = m(1).toLowerCase() === 'rm' ? 'ringgit' : 'dolar';
  if (scale) return number(amount) + ' ' + scale.toLowerCase() + ' ' + unit;
  const a = amount.replace(/,/g, '');
  const dot = a.indexOf('.');
  const whole = dot < 0 ? a : a.slice(0, dot);
  const frac = dot < 0 ? '' : a.slice(dot + 1);
  const wholeN = parseInt(whole || '0', 10);
  const sen = frac ? parseInt((frac + '00').slice(0, 2), 10) : 0;
  const parts = [];
  if (wholeN || !sen) parts.push(cardinal(wholeN) + ' ' + unit);
  if (sen) parts.push(cardinal(sen) + ' sen');
  return parts.join(' ');
});

const dateNum = cb((m) => {
  const d = parseInt(m(1), 10), mo = parseInt(m(2), 10), y = m(3);
  if (!(mo >= 1 && mo <= 12 && d >= 1 && d <= 31)) return m(0);
  const yy = y.length === 4 ? parseInt(y, 10) : 2000 + parseInt(y, 10);
  return cardinal(d) + ' ' + MONTHS[mo - 1] + ' ' + cardinal(yy);
});

const dateWord = cb((m) => {
  const mo = MONTH_ABBR[m(2).toLowerCase().slice(0, 3)];
  let out = cardinal(parseInt(m(1), 10)) + ' ' + MONTHS[mo - 1];
  if (m(3)) out += ' ' + cardinal(parseInt(m(3), 10));
  return out;
});

const timeM = cb((m) => {
  const lead = m(1);
  return (lead ? lead : 'pukul ') + timeWords(m(2), m(3), m(4));
});

const numM = cb((m) => {
  const s = m(0);
  const plain = s.replace(/,/g, '');
  const isInt = [...plain].every((c) => DIGITS.includes(c));
  if ((plain.length > 1 && plain[0] === '0' && !plain.includes('.')) || (isInt && plain.length >= 8)) {
    return digits(plain);
  }
  return number(s);
});

function esc(s) {
  return s.replace(/([.*+?^${}()|[\]\\/])/g, '\\$1');
}

export function normalize(text) {
  let t = stripAccents(text);
  t = t.replace(/’/g, "'").replace(/‘/g, "'").replace(/“/g, '"').replace(/”/g, '"');
  t = t.replace(/–/g, '-').replace(/—/g, ' — ');
  t = t.replace(/[\u00a0\u1680\u2000-\u200b\u2028\u2029\u202f\u205f\u3000\ufeff\t\r\n\f\v]/g, ' ');

  t = sub('\\+?\\b6?0\\d{1,2}[- ]?\\d{3,4}[- ]?\\d{3,4}\\b', cb((m) => ' ' + digits(m(0)) + ' '), t);
  t = sub('\\b\\d{6}-\\d{2}-\\d{4}\\b', cb((m) => ' ' + digits(m(0)) + ' '), t);
  t = sub('\\b(' + DIGIT_CONTEXT + ')((?:\\s+(?!RM|USD|US\\$)[A-Za-z]+){0,3}\\s*:?\\s*)(?!RM|USD|US\\$)(\\d[\\d\\- ]*\\d|\\d)\\b(?![.,]\\d)',
    cb((m) => m(1) + m(2) + digits(m(3))), t, true);

  t = sub('\\b(RM|USD|US\\$)\\s?(' + NUM + ')(?:\\s?(juta|bilion|ribu|trilion)\\b)?', money, t, true);
  t = sub('(\\$)\\s?(' + NUM + ')(?:\\s?(juta|bilion|ribu|trilion)\\b)?', money, t, true);

  t = sub('(' + NUM + ')\\s?%', cb((m) => number(m(1)) + ' peratus'), t);

  t = sub('\\b(\\d{1,2})[/-](\\d{1,2})[/-](\\d{4}|\\d{2})\\b', dateNum, t);
  t = sub('\\b(\\d{1,2})\\s+(' + MONTH_RE + ')\\.?(?:\\s+(\\d{4}))?\\b', dateWord, t, true);

  t = sub('(\\b(?:pukul|jam)\\s+)?\\b(\\d{1,2}):(\\d{2})\\s?' + TIME_SFX + '?', timeM, t, true);
  t = sub('(\\b(?:pukul|jam)\\s+)\\b(\\d{1,2})\\.(\\d{2})\\s?' + TIME_SFX + '?', timeM, t, true);
  t = sub('()\\b(\\d{1,2})\\.(\\d{2})\\s?' + TIME_SFX, timeM, t, true);
  t = sub('(\\b(?:pukul|jam)\\s+)?\\b(\\d{1,2})\\s?(am|pm|a\\.m\\.|p\\.m\\.)(?=\\W|$)',
    cb((m) => (m(1) || 'pukul ') + timeWords(m(2), 0, m(3))), t, true);

  t = sub('\\bke-?(\\d+)\\b', cb((m) => ordinal(parseInt(m(1), 10))), t, true);
  t = sub('\\b(' + NUM + ')\\s?-\\s?(' + NUM + ')\\b',
    cb((m) => number(m(1)) + ' hingga ' + number(m(2))), t);

  for (const [u, w] of UNITS) {
    t = sub('(\\d)\\s?' + esc(u) + '(?![a-zA-Z])', cb((m) => m(1) + ' ' + w), t, true);
  }

  t = sub(NUM, numM, t);

  const abbr = ABBR.slice().sort((a, b) => b[0].length - a[0].length);
  for (const [a, w] of abbr) {
    const pat = '(?<![\\w.])' + esc(a) + (a.endsWith('.') ? '(?!\\w)' : '(?![\\w.])');
    t = sub(pat, () => w, t, true);
  }

  t = t.split('&').join(' dan ').split('+').join(' tambah ').split('=').join(' sama dengan ')
    .split('%').join(' peratus ');

  t = sub('\\s+', ' ', t).trim();
  t = sub('\\s+([,.!?;:…])', '$1', t);
  if (/\.$/.test(text.replace(/\s+$/, '')) && t && !'.!?…'.includes(t[t.length - 1])) t += '.';
  return t;
}

export { cardinal, digits, number };
