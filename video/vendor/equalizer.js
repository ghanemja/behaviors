// Vendored copy of equalizer (https://github.com/ghanemja/equalizer), MIT, Copyright (c) 2026 Janelle Ghanem. Renders the video soundtrack.
// equalizer — beats from math equations.
//
// Every sound is an equation of time. A track pairs an equation with a step
// pattern that decides when it fires, plus named parameters (the knobs):
//
//   { name: 'Kick',
//     equation: 'sin(TAU * (base*tau + punch*(1 - exp(-tau*sweep))/sweep)) * exp(-tau*decay)',
//     params: { base: 48, punch: 150, sweep: 30, decay: 7 },
//     pattern: 'x...x...x...x...' }
//
// Equations are parsed by a small whitelisted parser, never run as raw JS, so
// songs shared as links are safe to open.

// --- Expression language ------------------------------------------------------

/** Variables every equation can read. */
export const VARIABLES = {
  t: 'seconds since the song started',
  tau: 'seconds since this note started',
  f: 'frequency of the note in Hz',
  beat: 'beats since the song started',
  step: 'index of the step that triggered this note',
  bar: 'bar number (beatsPerBar beats per bar, 4 by default)',
  r: 'a random number in [0, 1) chosen once per note',
  bpm: 'tempo in beats per minute',
};

const fract = (x) => x - Math.floor(x);
let seed = 1;
const rand = () => {
  // xorshift32: deterministic noise, so a song renders identically every time
  seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
  return ((seed >>> 0) / 4294967296);
};

/** Functions every equation can call. Oscillators take a phase in cycles. */
export const FUNCTIONS = {
  sin: Math.sin, cos: Math.cos, tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan, atan2: Math.atan2,
  sinh: Math.sinh, cosh: Math.cosh, tanh: Math.tanh,
  abs: Math.abs, sign: Math.sign, floor: Math.floor, ceil: Math.ceil, round: Math.round,
  exp: Math.exp, log: Math.log, log2: Math.log2, pow: Math.pow, sqrt: Math.sqrt,
  min: Math.min, max: Math.max,
  fract,
  mod: (a, b) => a - b * Math.floor(a / b),
  clamp: (x, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x)),
  mix: (a, b, k) => a + (b - a) * k,
  // oscillators: phase in cycles (e.g. saw(f*tau)), output in [-1, 1]
  osc: (p) => Math.sin(2 * Math.PI * p),
  saw: (p) => 2 * fract(p) - 1,
  sqr: (p, duty = 0.5) => (fract(p) < duty ? 1 : -1),
  tri: (p) => 1 - 4 * Math.abs(fract(p + 0.25) - 0.5),
  // envelopes
  ad: (tau, a, d) => Math.min(1, tau / Math.max(a, 1e-6)) * Math.exp(-Math.max(0, tau - a) * d),
  // randomness
  noise: () => rand() * 2 - 1,
  hash: (x) => fract(Math.sin(x * 12.9898) * 43758.5453),
};

export const CONSTANTS = { PI: Math.PI, TAU: 2 * Math.PI, E: Math.E };

const RESERVED = new Set(['break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else',
  'export', 'extends', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof', 'new', 'return', 'super', 'switch',
  'this', 'throw', 'try', 'typeof', 'var', 'void', 'while', 'with', 'yield', 'let', 'static', 'enum', 'await', 'implements',
  'package', 'protected', 'interface', 'private', 'public', 'null', 'true', 'false', 'undefined', 'NaN', 'Infinity',
  'eval', 'arguments', 'constructor', 'prototype', '__proto__']);

export class EquationError extends Error {}

// Own-property checks only: `in` would also accept inherited names like "constructor".
const has = (obj, key) => Object.hasOwn(obj, key);

/** A valid parameter name: letters, digits and underscores, not shadowing a built-in. */
export function isParamName(name) {
  return typeof name === 'string' && /^[A-Za-z_][A-Za-z0-9_]{0,31}$/.test(name) && !RESERVED.has(name)
    && !has(FUNCTIONS, name) && !has(CONSTANTS, name) && !has(VARIABLES, name);
}

const TOKEN = /\s*(?:(\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(&&|\|\||[<>=!]=|[-+*/%^(),?:<>!]))/y;

function tokenize(src) {
  const out = [];
  TOKEN.lastIndex = 0;
  while (TOKEN.lastIndex < src.length) {
    if (/^\s*$/.test(src.slice(TOKEN.lastIndex))) break;
    const at = TOKEN.lastIndex;
    const m = TOKEN.exec(src);
    if (!m) throw new EquationError(`Unexpected "${src.slice(at).trim()[0]}" at position ${at + 1}`);
    if (m[1] !== undefined) out.push({ type: 'num', value: m[1] });
    else if (m[2] !== undefined) out.push({ type: 'id', value: m[2] });
    else out.push({ type: 'op', value: m[3] });
  }
  return out;
}

// Recursive-descent parser that emits fully parenthesized JS. Only numbers,
// whitelisted names and math operators can appear in the output.
function toJs(src, names) {
  const toks = tokenize(src);
  let i = 0;
  const peek = () => toks[i];
  const isOp = (v) => peek()?.type === 'op' && peek().value === v;
  const expect = (v) => {
    if (!isOp(v)) throw new EquationError(`Expected "${v}"${peek() ? ` but found "${peek().value}"` : ' at the end'}`);
    i++;
  };

  function ternary() {
    const c = or();
    if (isOp('?')) { i++; const a = ternary(); expect(':'); const b = ternary(); return `(${c} ? ${a} : ${b})`; }
    return c;
  }
  const binary = (next, ops, map = {}) => () => {
    let left = next();
    while (peek()?.type === 'op' && ops.includes(peek().value)) {
      const op = toks[i++].value;
      left = `(${left} ${map[op] ?? op} ${next()})`;
    }
    return left;
  };
  const or = binary(() => and(), ['||']);
  const and = binary(() => cmp(), ['&&']);
  const cmp = binary(() => add(), ['<', '>', '<=', '>=', '==', '!='], { '==': '===', '!=': '!==' });
  const add = binary(() => mul(), ['+', '-']);
  const mul = binary(() => unary(), ['*', '/', '%']);
  function unary() {
    if (isOp('-') || isOp('+') || isOp('!')) { const op = toks[i++].value; return `(${op}${unary()})`; }
    return power();
  }
  function power() {
    const base = primary();
    if (isOp('^')) { i++; return `(${base} ** ${unary()})`; }
    return base;
  }
  function primary() {
    const tok = toks[i++];
    if (!tok) throw new EquationError('The equation ends too early');
    if (tok.type === 'num') return `(${tok.value})`;
    if (tok.type === 'op' && tok.value === '(') { const e = ternary(); expect(')'); return `(${e})`; }
    if (tok.type === 'id') {
      if (isOp('(')) {
        if (!has(FUNCTIONS, tok.value)) throw new EquationError(`Unknown function "${tok.value}"`);
        i++;
        const args = [];
        if (!isOp(')')) { args.push(ternary()); while (isOp(',')) { i++; args.push(ternary()); } }
        expect(')');
        return `${tok.value}(${args.join(', ')})`;
      }
      if (has(CONSTANTS, tok.value) || names.has(tok.value)) return tok.value;
      if (has(FUNCTIONS, tok.value)) throw new EquationError(`"${tok.value}" is a function: call it like ${tok.value}(x)`);
      throw new EquationError(`Unknown name "${tok.value}"`);
    }
    throw new EquationError(`Unexpected "${tok.value}"`);
  }

  const js = ternary();
  if (i < toks.length) throw new EquationError(`Unexpected "${toks[i].value}"`);
  return js;
}

const VAR_NAMES = Object.keys(VARIABLES);
const cache = new Map();

/**
 * Compile an equation into a function (t, tau, f, beat, step, bar, r, bpm, ...params).
 * Throws EquationError with a readable message if the equation is invalid.
 */
export function compile(equation, paramNames = []) {
  if (typeof equation !== 'string' || !equation.trim()) throw new EquationError('Write an equation');
  if (equation.length > 2000) throw new EquationError('That equation is too long (2000 characters max)');
  for (const p of paramNames) if (!isParamName(p)) throw new EquationError(`"${p}" can't be used as a parameter name`);
  const key = equation + '\u0000' + paramNames.join(',');
  if (cache.has(key)) return cache.get(key);
  const js = toJs(equation, new Set([...VAR_NAMES, ...paramNames]));
  const fnNames = Object.keys(FUNCTIONS);
  const factory = new Function('F', 'C',
    `"use strict"; const {${fnNames.join(', ')}} = F; const {${Object.keys(CONSTANTS).join(', ')}} = C;` +
    `return function (${[...VAR_NAMES, ...paramNames].join(', ')}) { return ${js}; };`);
  const fn = factory(FUNCTIONS, CONSTANTS);
  cache.set(key, fn);
  return fn;
}

// --- Notes and patterns -----------------------------------------------------------

const SEMITONES = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

/** "A4" -> 440, "C#3" -> 138.59, "Eb2" -> 77.78. Returns null if not a note. */
export function noteToFreq(name) {
  const m = /^([A-Ga-g])([#b]?)(-?\d)$/.exec(name);
  if (!m) return null;
  const semi = SEMITONES[m[1].toUpperCase()] + (m[2] === '#' ? 1 : m[2] === 'b' ? -1 : 0);
  const midi = semi + 12 * (Number(m[3]) + 1);
  return 440 * 2 ** ((midi - 69) / 12);
}

/**
 * Parse a pattern into steps. Two forms:
 *  - compact, for drums: "x...x..." (x = hit, . = continue, - = stop, | ignored)
 *  - spaced, for notes:  "C3 . E3 . G3+B3 - x ." (notes or chords joined by +)
 * Each step is { hit: false } | { hit: true, notes: [Hz] } | { stop: true }.
 */
export function parsePattern(pattern, defaultFreq = 110) {
  const src = (pattern ?? '').trim();
  if (!src) return [];
  const toks = /\s/.test(src) ? src.split(/\s+/) : [...src.replace(/\|/g, '')];
  return toks.map((tok) => {
    if (tok === '.' || tok === '_') return { hit: false };
    if (tok === '-') return { stop: true };
    if (tok === 'x' || tok === 'X') return { hit: true, notes: [defaultFreq] };
    const notes = tok.split('+').map(noteToFreq);
    if (notes.some((n) => n == null)) throw new EquationError(`"${tok}" isn't a step. Use x, ., -, or notes like C3 or C3+E3`);
    return { hit: true, notes };
  });
}

/** Normalize params given as { name: number } or { name: { value, min, max } }. */
export function paramValues(params = {}) {
  const out = {};
  for (const [k, v] of Object.entries(params)) out[k] = typeof v === 'number' ? v : Number(v?.value ?? 0);
  return out;
}

// --- Rendering ----------------------------------------------------------------------

const FADE_IN = 0.001, FADE_OUT = 0.004, MAX_VOICE = 16;

function onePoleLowpass(buf, hz, sr) {
  const a = 1 - Math.exp(-2 * Math.PI * hz / sr);
  let y = 0;
  for (let i = 0; i < buf.length; i++) { y += a * (buf[i] - y); buf[i] = y; }
}
function onePoleHighpass(buf, hz, sr) {
  const a = 1 - Math.exp(-2 * Math.PI * hz / sr);
  let y = 0;
  for (let i = 0; i < buf.length; i++) { y += a * (buf[i] - y); buf[i] -= y; }
}
function echo(buf, delaySamples, feedback) {
  for (let i = delaySamples; i < buf.length; i++) buf[i] += feedback * buf[i - delaySamples];
}

/** When a track fires: [{ time, step, notes, end }] in seconds. */
export function triggers(track, song, duration) {
  const spb = 60 / (song.bpm ?? 120);
  const rate = track.rate ?? 4;
  const stepDur = spb / rate;
  const steps = parsePattern(track.pattern, noteToFreq(track.note ?? 'A2'));
  const gate = track.gate ? compile(track.gate) : null;
  const hits = [];
  const stops = [];
  if (steps.length) {
    for (let s = 0; s * stepDur < duration - 1e-9; s++) {
      const st = steps[s % steps.length];
      const time = s * stepDur;
      if (st.stop) stops.push(time);
      if (!st.hit) continue;
      if (gate) {
        const beat = time / spb;
        if (!gate(time, 0, 0, beat, s, Math.floor(beat / beatsPerBar(song)), 0, song.bpm ?? 120)) continue;
      }
      hits.push({ time, step: s, notes: st.notes });
    }
  }
  for (const time of track.at ?? []) {
    if (time >= 0 && time < duration) hits.push({ time, step: -1, notes: [noteToFreq(track.note ?? 'A2')] });
  }
  hits.sort((a, b) => a.time - b.time);
  const len = track.len ?? Infinity;
  hits.forEach((h, k) => {
    const next = hits[k + 1]?.time ?? Infinity;
    const stop = stops.find((s) => s > h.time) ?? Infinity;
    h.end = Math.min(next, stop, h.time + len, h.time + MAX_VOICE, duration);
  });
  return hits;
}

/** Render one track (mono, before pan/gain) into a Float32Array. */
export function renderTrack(track, song, { sampleRate = 44100, duration } = {}) {
  const N = Math.ceil(duration * sampleRate);
  const buf = new Float32Array(N);
  const values = paramValues(track.params);
  const names = Object.keys(values);
  const fn = compile(track.equation, names);
  const pv = names.map((n) => values[n]);
  const bpm = song.bpm ?? 120;
  const spb = 60 / bpm;
  const bpb = beatsPerBar(song);
  let voiceIndex = 0;
  for (const hit of triggers(track, song, duration)) {
    const r = FUNCTIONS.hash(voiceIndex++ * 7.31 + 0.5);
    const s0 = Math.round(hit.time * sampleRate);
    const s1 = Math.min(N, Math.round((hit.end + FADE_OUT) * sampleRate));
    const releaseAt = hit.end - hit.time;
    const amp = 1 / Math.sqrt(hit.notes.length);
    for (const f of hit.notes) {
      for (let i = s0; i < s1; i++) {
        const t = i / sampleRate;
        const tau = (i - s0) / sampleRate;
        const beat = t / spb;
        let v = fn(t, tau, f, beat, hit.step, Math.floor(beat / bpb), r, bpm, ...pv);
        if (!Number.isFinite(v)) v = 0;
        let g = amp;
        if (tau < FADE_IN) g *= tau / FADE_IN;
        if (tau > releaseAt) g *= Math.max(0, 1 - (tau - releaseAt) / FADE_OUT);
        buf[i] += Math.max(-4, Math.min(4, v)) * g;
      }
    }
  }
  if (track.hp) onePoleHighpass(buf, track.hp, sampleRate);
  if (track.lp) onePoleLowpass(buf, track.lp, sampleRate);
  if (track.echo) echo(buf, Math.round((track.echoTime ?? 0.75) * spb * sampleRate), Math.min(0.95, track.echo));
  return buf;
}

/** Beats in one bar (the time signature's top number): song.beatsPerBar, 4 by default. */
export function beatsPerBar(song) {
  const b = Math.round(+song?.beatsPerBar);
  return b >= 1 && b <= 16 ? b : 4;
}

/** Length of a song in seconds: `duration`, or `bars` of beatsPerBar beats. */
export function songDuration(song) {
  return song.duration ?? (song.bars ?? 4) * beatsPerBar(song) * 60 / (song.bpm ?? 120);
}

/**
 * Render a whole song to stereo Float32Arrays.
 * options: sampleRate, normalize (peak to about -1 dB), stems (also return each track).
 */
export function renderSong(song, { sampleRate = 44100, normalize = true, stems = false } = {}) {
  seed = (song.seed ?? 1) >>> 0 || 1;
  const duration = songDuration(song);
  const N = Math.ceil(duration * sampleRate);
  const left = new Float32Array(N);
  const right = new Float32Array(N);
  const tracks = song.tracks ?? [];
  const soloing = tracks.some((t) => t.solo);
  const stemOut = [];
  for (const track of tracks) {
    const audible = !track.mute && (!soloing || track.solo);
    if (!audible && !stems) { stemOut.push(null); continue; }
    const buf = renderTrack(track, song, { sampleRate, duration });
    const gain = track.gain ?? 0.8;
    for (let i = 0; i < N; i++) buf[i] *= gain;
    stemOut.push(buf);
    if (!audible) continue;
    const pan = Math.max(-1, Math.min(1, track.pan ?? 0));
    const gl = Math.cos((pan + 1) * Math.PI / 4) * Math.SQRT2;
    const gr = Math.sin((pan + 1) * Math.PI / 4) * Math.SQRT2;
    for (let i = 0; i < N; i++) { left[i] += buf[i] * gl; right[i] += buf[i] * gr; }
  }
  const master = song.master ?? 0.5;  // headroom before the soft clip
  let peak = 0;
  for (let i = 0; i < N; i++) {
    left[i] = Math.tanh(left[i] * master);
    right[i] = Math.tanh(right[i] * master);
    peak = Math.max(peak, Math.abs(left[i]), Math.abs(right[i]));
  }
  if (normalize && peak > 0) {
    const k = 0.89 / peak;
    for (let i = 0; i < N; i++) { left[i] *= k; right[i] *= k; }
  }
  const fadeIn = Math.round((song.fadeIn ?? 0) * sampleRate);
  const fadeOut = Math.round((song.fadeOut ?? 0) * sampleRate);
  for (let i = 0; i < fadeIn && i < N; i++) { left[i] *= i / fadeIn; right[i] *= i / fadeIn; }
  for (let i = 0; i < fadeOut && i < N; i++) { const k = i / fadeOut; left[N - 1 - i] *= k; right[N - 1 - i] *= k; }
  return { left, right, sampleRate, duration, stems: stems ? stemOut : undefined };
}

/** Encode stereo float audio as a 16-bit PCM WAV file. */
export function toWav(left, right, sampleRate) {
  const n = left.length;
  const view = new DataView(new ArrayBuffer(44 + n * 4));
  const str = (o, s) => { for (let k = 0; k < s.length; k++) view.setUint8(o + k, s.charCodeAt(k)); };
  str(0, 'RIFF'); view.setUint32(4, 36 + n * 4, true); str(8, 'WAVE');
  str(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 2, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 4, true); view.setUint16(32, 4, true); view.setUint16(34, 16, true);
  str(36, 'data'); view.setUint32(40, n * 4, true);
  for (let i = 0, o = 44; i < n; i++, o += 4) {
    view.setInt16(o, Math.max(-1, Math.min(1, left[i])) * 32767, true);
    view.setInt16(o + 2, Math.max(-1, Math.min(1, right[i])) * 32767, true);
  }
  return new Uint8Array(view.buffer);
}

// --- Live playback (browser) ---------------------------------------------------------

/**
 * Plays a song on a loop and hot-swaps it when you change it, keeping the
 * playhead where it is.
 *
 *   const player = new Player();
 *   await player.play(song);   // must follow a user gesture
 *   player.update(song2);      // re-renders and crossfades in place
 */
export class Player {
  constructor() {
    this.ctx = null;
    this.source = null;
    this.gain = null;
    this.startedAt = 0;
    this.duration = 0;
    this.stems = [];
    this.analyser = null;
  }

  get playing() { return !!this.source; }

  /** Seconds into the loop. */
  get position() {
    if (!this.source) return 0;
    return ((this.ctx.currentTime - this.startedAt) % this.duration + this.duration) % this.duration;
  }

  async play(song) {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.master = this.ctx.createGain();
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 2048;
      this.master.connect(this.analyser);
      this.analyser.connect(this.ctx.destination);
    }
    await this.ctx.resume();
    this.update(song, true);
  }

  update(song, restart = false) {
    if (!this.ctx || (!this.source && !restart)) return;
    const { left, right, duration, stems } = renderSong(song, { sampleRate: this.ctx.sampleRate, stems: true, normalize: false });
    const buffer = this.ctx.createBuffer(2, left.length, this.ctx.sampleRate);
    buffer.copyToChannel(left, 0);
    buffer.copyToChannel(right, 1);
    const now = this.ctx.currentTime;
    const offset = restart || !this.source ? 0 : this.position % duration;
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.loop = true;
    const g = this.ctx.createGain();
    src.connect(g).connect(this.master);
    const fade = 0.03;
    g.gain.setValueAtTime(0, now);
    g.gain.linearRampToValueAtTime(1, now + fade);
    src.start(now, offset);
    if (this.source) {
      const old = this.source, oldGain = this.gain;
      oldGain.gain.setValueAtTime(1, now);
      oldGain.gain.linearRampToValueAtTime(0, now + fade);
      old.stop(now + fade + 0.01);
    }
    this.source = src;
    this.gain = g;
    this.duration = duration;
    this.startedAt = now - offset;
    this.stems = stems;
  }

  stop() {
    if (!this.source) return;
    this.source.stop();
    this.source = null;
  }

  set volume(v) { if (this.master) this.master.gain.value = v; }
}
