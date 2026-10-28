// The video soundtrack: every sound is an equation of time, rendered with the
// vendored equalizer engine (video/vendor). 150 bpm in a half-time feel, so a
// beat is 0.4 s and a bar is 1.6 s; every scene cut in reel.html sits on a bar
// line. Sound effects come from cues.json, which the reel writes from the
// simulations (`node video/render.js cues video/cues.json`), so they stay in
// sync if the video is retimed. Render with:
//   node video/soundtrack.mjs soundtrack.wav
//
// 0-6.4s     title: soft keys; a filing cabinet, a beer bottle and a candlestick tip over; a riser into the drop
// 6.4s       the drop, on the canned peaches search
// 19.2s      the behavior map sweeps in with sparkles; the lead rests
// 25.6s      results, with the JEPA
// 35.2-38.4s breakdown under the apples; the doorstop apple boings back up
// 38.4s      the beat drops back in as the bug is revealed
// 41.6s      end card; the last chord lands at 43.2s
import cues from './cues.json' with { type: 'json' };

const BAR = 1.6;
const DROP = 6.4, BREAK = [35.2, 38.4], LAST = 43.2, END_CARD = 41.6, MAP = [19.2, 25.6];

// The arrangement is math too.
const DRUMS = `t >= ${DROP} && (t < ${BREAK[0]} || t >= ${BREAK[1]}) && t < ${LAST}`;
const SOFT = `(t >= ${BREAK[0]} && t < ${BREAK[1]} ? 0.6 : 1)`;

// I-vi-ii-V in C, rotated so the drop (bar 4) lands on Dm9.
const PROG = [
  { chord: 'F3+A3+C4+E4', root: 'D2', lead: 'F5 . . A5 . . C6 . . . A5 . E5 . . .' },  // Dm9
  { chord: 'F3+A3+B3+E4', root: 'G1', lead: 'D5 . . F5 . . E5 . . . D5 . B4 . . .' },  // G13
  { chord: 'E3+G3+B3+D4', root: 'C2', lead: 'E5 . . G5 . . B5 . . . A5 . G5 . . .' },  // Cmaj9
  { chord: 'C3+G3+B3+E4', root: 'A1', lead: 'C5 . . E5 . . D5 . . . C5 . . . . .' },  // Am9
];
const bar = (cells) => { const s = Array(16).fill('.'); for (const [k, v] of cells) s[k] = v; return s.join(' '); };
const CHORDS = PROG.map((p) => p.chord).join(' ');

// Cue helpers: drop near-duplicates (a track restarts its voice on every hit), and
// split simultaneous hits across tracks so they ring together.
const dedupe = (ts, gap = 0.06) => [...ts].sort((a, b) => a - b).filter((t, k, s) => k === 0 || t - s[k - 1] >= gap);
const PANS = [[-1, -0.3], [-0.3, 0], [0, 0.3], [0.3, 1.01]];
const lands = PANS.map(([lo, hi]) => ({ pan: (lo + Math.min(hi, 1)) / 2, at: dedupe(cues.lands.filter((l) => l.pan >= lo && l.pan < hi).map((l) => l.t)) }));
const chimes = [[], [], [], []];  // the k-th chime of a quick run goes to voice k, so a row of stay-ups plays an arpeggio
cues.chimes.forEach((t, k, s) => { let n = 0; while (k - n - 1 >= 0 && t - s[k - n - 1] < 0.3) n++; chimes[n % 4].push(t); });
const GROW = 0.4;
const KEYS = '(sin(TAU*f*tau) + 0.4*sin(TAU*2*f*tau)*exp(-4*tau) + 0.15*sin(TAU*3.01*f*tau)*exp(-9*tau)) * exp(-1.2*tau) * (1 + 0.15*sin(TAU*4*t))';

export default {
  title: 'Behaviors video',
  bpm: 150,
  duration: cues.end,
  fadeOut: 1.2,
  tracks: [
    // --- Lo-fi bed ---
    {
      name: 'Crackle',
      equation: 'noise() > 0.996 ? noise() : 0',
      pattern: 'x', rate: 0.25, len: BAR, hp: 2500, gain: 0.25,
    },
    {
      name: 'Pad',
      equation: `(saw(f*tau*0.995) + saw(f*tau*1.005)) / 2 * min(1, tau/0.4) * ${SOFT}`,
      pattern: CHORDS, rate: 0.25, len: BAR, gate: `t < ${LAST}`, lp: 900, gain: 0.22,
    },
    {
      name: 'Intro keys',
      equation: KEYS,
      pattern: CHORDS, rate: 0.25, len: BAR, gate: `t < ${DROP}`, lp: 1100, gain: 0.45,
    },
    {
      name: 'Keys',
      // electric piano: a sine with a fading octave and bell partial, and tremolo
      equation: `${KEYS} * ${SOFT}`,
      pattern: PROG.map((p) => bar([[0, p.chord], [10, p.chord]])).join(' '),
      gate: `t >= ${DROP} && t < ${LAST}`, lp: 2600, gain: 0.4, pan: 0.1,
    },
    {
      name: 'Last chord',
      equation: KEYS.replace('exp(-1.2*tau)', 'exp(-0.8*tau)'),
      pattern: 'E3+G3+B3+D4 E3+G3+B3+D4', rate: 0.25, gate: `t >= ${LAST}`, lp: 2600, gain: 0.55,
    },

    // --- The groove, from the drop ---
    {
      name: 'Kick',
      equation: 'sin(TAU*(50*tau + 7*(1 - exp(-35*tau)))) * exp(-9*tau)',
      pattern: 'x.....x...x.....' + 'x.....x...x..x..', gate: DRUMS, gain: 0.9,
    },
    {
      name: '808',
      equation: 'tanh(1.8*sin(TAU*(f*tau + 1.2*(1 - exp(-30*tau))))) * exp(-1.4*tau) * min(1, tau/0.004)',
      pattern: PROG.map((p) => bar([[0, p.root], [10, p.root]])).join(' '),
      gate: DRUMS, lp: 420, gain: 0.65,
    },
    {
      name: 'Clap',
      equation: 'noise() * (exp(-tau*16) + 0.8*exp(-abs(tau - 0.012)*400) + 0.6*exp(-abs(tau - 0.024)*400))',
      pattern: '........x.......', gate: DRUMS, hp: 900, gain: 0.5,
    },
    {
      name: 'Hat',
      equation: 'noise() * exp(-70*tau) * (fract(beat) < 0.25 ? 1 : 0.6)',
      pattern: 'x.x.x.x.x.x.x.x.' + 'x.x.x.x.x.x.xxxx',
      gate: `t >= ${DROP - 2 * BAR} && t < ${LAST} && (t < ${BREAK[0]} || t >= ${BREAK[1]})`, hp: 8000, gain: 0.2, pan: 0.25,
    },
    {
      name: 'Open hat',
      equation: 'noise() * exp(-16*tau)',
      pattern: '......x.......x.', gate: DRUMS, hp: 6500, gain: 0.26, pan: -0.2,
    },
    {
      name: 'Lead',
      // a marimba-ish pluck with a dotted-eighth echo; it rests under the behavior map
      equation: '(sin(TAU*f*tau) + 0.25*sin(TAU*4*f*tau)*exp(-30*tau)) * exp(-7*tau)',
      pattern: PROG.map((p) => p.lead).join(' '), gate: `${DRUMS} && (t < ${MAP[0]} || t >= ${MAP[1]})`, gain: 0.36, pan: -0.15, echo: 0.3,
    },

    // --- Sound effects, at the video's exact moments ---
    {
      name: 'Touch',
      equation: 'sin(TAU*1700*tau) * exp(-110*tau)',
      at: dedupe(cues.touches, 0.1), len: 0.06, gain: 0.22,
    },
    ...lands.map(({ pan, at }, k) => ({
      name: `Thud ${k + 1}`,
      equation: 'sin(TAU*(70*tau + 4*(1 - exp(-40*tau)))) * exp(-14*tau) + noise() * exp(-45*tau) * 0.5',
      at, len: 0.4, lp: 1400, gain: 0.5, pan,
    })),
    ...chimes.map((at, k) => ({
      name: `Chime ${k + 1}`,
      // a small bell for each object that stays up: E6, G6, B6, D7
      equation: 'sin(TAU*f*tau) * exp(-7*tau) + 0.3*sin(TAU*2.76*f*tau) * exp(-14*tau)',
      at, note: ['E6', 'G6', 'B6', 'D7'][k], len: 0.6, gain: 0.16, pan: -0.3 + 0.2 * k,
    })),
    {
      name: 'Boing',
      // the doorstop apple rocks over and back: a wobbling pitch
      equation: 'sin(TAU*(300*tau - 3.8*cos(TAU*5*tau)*exp(-2.5*tau))) * exp(-2.2*tau) * min(1, tau/0.01)',
      at: cues.boings.map((b) => b.t), len: 1.3, gain: 0.3, pan: cues.boings[0]?.pan ?? 0,
    },
    {
      name: 'Sparkle',
      // a high ping as each wave of dots lands on the map; r gives every ping its own pitch
      equation: 'sin(TAU*f*(1 + 0.5*r)*tau) * exp(-40*tau)',
      at: cues.sparkles ?? [], note: 'C7', len: 0.12, gain: 0.12, pan: 0.2,
    },
    {
      name: 'Pop',
      equation: 'sin(TAU*(380*tau + 16*(1 - exp(-45*tau)))) * exp(-35*tau)',
      at: cues.pops, len: 0.15, gain: 0.36,
    },
    {
      name: 'Whoosh',
      equation: `noise() * (tau < ${GROW} ? (tau/${GROW})^2 : exp(-(tau - ${GROW})*12))`,
      at: cues.cuts.map((c) => c - GROW), len: 0.8, lp: 2600, gain: 0.4,
    },
    {
      name: 'Riser',
      equation: `noise() * min(1, tau/${BAR})^3`,
      at: [DROP - BAR, BREAK[1] - BAR], len: BAR, hp: 2200, gain: 0.28,
    },
    {
      name: 'Crash',
      equation: 'noise() * exp(-2.2*tau)',
      at: [DROP, BREAK[1], END_CARD, LAST], len: BAR, hp: 5000, gain: 0.22,
    },
  ],
};
