// The video soundtrack: every sound is an equation of time, rendered with the
// vendored equalizer engine (video/vendor). 150 bpm in a half-time feel, so a
// beat is 0.4 s and a bar is 1.6 s; every scene cut in reel.html sits on a bar
// line. Render with:
//   node video/soundtrack.mjs soundtrack.wav
//
// 0-3.2s     title: soft keys, three objects pushed over; a riser into the drop
// 3.2s       the drop, on the canned peaches query
// 12.8s      the look-alike pairs
// 19.2-22.4s breakdown under the apples; the doorstop apple boings back up
// 22.4s      the beat drops back in with the bug callout
// 25.6s      end card; the last chord lands at 30.4s

// When objects land after toppling, from the rollouts in data.json (video seconds).
// Panned to where they sit on screen.
const LAND_L = [0.76, 4.89, 14.06, 14.14, 20.85];
const LAND_C = [0.81, 4.81];
const LAND_R = [0.86, 4.89, 14.06, 14.14];
const TOUCH = [0.32, 3.88, 13.36, 19.85];          // the pusher reaches the objects
const POPS = [0.15, 8.8, 16.8, 22.4, 25.6, 26.4, 27.2];  // badge, callouts, end card
const CUTS = [12.8, 19.2, 25.6];                   // a whoosh lands on each cut
const GROW = 0.4;

// The arrangement is math too.
const DRUMS = 't >= 3.2 && (t < 19.2 || t >= 22.4) && t < 30.4';
const SOFT = '(t >= 19.2 && t < 22.4 ? 0.6 : 1)';

// I-vi-ii-V in C, rotated so the drop (bar 2) lands on Dm9.
const CHORDS = ['E3+G3+B3+D4', 'C3+G3+B3+E4', 'F3+A3+C4+E4', 'F3+A3+B3+E4'];
const ROOTS = ['C2', 'A1', 'D2', 'G1'];
const bar = (cells) => { const s = Array(16).fill('.'); for (const [k, v] of cells) s[k] = v; return s.join(' '); };
const LEAD = [
  'E5 . . G5 . . B5 . . . A5 . G5 . . .',
  'C5 . . E5 . . D5 . . . C5 . . . . .',
  'F5 . . A5 . . C6 . . . A5 . E5 . . .',
  'D5 . . F5 . . E5 . . . D5 . B4 . . .',
].join(' ');

export default {
  title: 'Behaviors video',
  bpm: 150,
  duration: 31.6,
  fadeOut: 1.0,
  tracks: [
    // --- Lo-fi bed ---
    {
      name: 'Crackle',
      equation: 'noise() > 0.996 ? noise() : 0',
      pattern: 'x', rate: 0.25, len: 1.6, hp: 2500, gain: 0.25,
    },
    {
      name: 'Pad',
      equation: `(saw(f*tau*0.995) + saw(f*tau*1.005)) / 2 * min(1, tau/0.4) * ${SOFT}`,
      pattern: CHORDS.join(' '), rate: 0.25, len: 1.6, gate: 't < 30.4', lp: 900, gain: 0.22,
    },
    {
      name: 'Intro keys',
      equation: '(sin(TAU*f*tau) + 0.4*sin(TAU*2*f*tau)*exp(-4*tau)) * exp(-0.9*tau) * (1 + 0.15*sin(TAU*4*t))',
      pattern: CHORDS.join(' '), rate: 0.25, len: 1.6, gate: 't < 3.2', lp: 1100, gain: 0.5,
    },
    {
      name: 'Keys',
      // electric piano: a sine with a fading octave and bell partial, and tremolo
      equation: `(sin(TAU*f*tau) + 0.4*sin(TAU*2*f*tau)*exp(-4*tau) + 0.15*sin(TAU*3.01*f*tau)*exp(-9*tau)) * exp(-1.2*tau) * (1 + 0.15*sin(TAU*4*t)) * ${SOFT}`,
      pattern: CHORDS.map((c) => bar([[0, c], [10, c]])).join(' '),
      gate: 't >= 3.2 && t < 30.4', lp: 2600, gain: 0.4, pan: 0.1,
    },
    {
      name: 'Last chord',
      equation: '(sin(TAU*f*tau) + 0.4*sin(TAU*2*f*tau)*exp(-4*tau)) * exp(-0.9*tau) * (1 + 0.15*sin(TAU*4*t))',
      pattern: 'E3+G3+B3+D4 E3+G3+B3+D4', rate: 0.25, gate: 't >= 30.4', lp: 2600, gain: 0.55,
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
      pattern: ROOTS.map((r) => bar([[0, r], [10, r]])).join(' '),
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
      gate: `t >= 1.6 && t < 30.4 && (t < 19.2 || t >= 22.4)`, hp: 8000, gain: 0.2, pan: 0.25,
    },
    {
      name: 'Open hat',
      equation: 'noise() * exp(-16*tau)',
      pattern: '......x.......x.', gate: DRUMS, hp: 6500, gain: 0.26, pan: -0.2,
    },
    {
      name: 'Lead',
      // a marimba-ish pluck with a dotted-eighth echo
      equation: '(sin(TAU*f*tau) + 0.25*sin(TAU*4*f*tau)*exp(-30*tau)) * exp(-7*tau)',
      pattern: LEAD, gate: DRUMS, gain: 0.38, pan: -0.15, echo: 0.3,
    },

    // --- Sound effects, at the video's exact moments ---
    {
      name: 'Touch',
      equation: 'sin(TAU*1700*tau) * exp(-110*tau)',
      at: TOUCH, len: 0.06, gain: 0.25,
    },
    ...[['Thud L', LAND_L, -0.45], ['Thud C', LAND_C, 0], ['Thud R', LAND_R, 0.45]].map(([name, at, pan]) => ({
      name,
      equation: 'sin(TAU*(70*tau + 4*(1 - exp(-40*tau)))) * exp(-14*tau) + noise() * exp(-45*tau) * 0.5',
      at, len: 0.4, lp: 1400, gain: 0.55, pan,
    })),
    {
      name: 'Boing',
      // the doorstop apple rocks to 45 degrees and back: a wobbling pitch
      equation: 'sin(TAU*(300*tau - 3.8*cos(TAU*5*tau)*exp(-2.5*tau))) * exp(-2.2*tau) * min(1, tau/0.01)',
      at: [20.05], len: 1.3, gain: 0.3, pan: 0.45,
    },
    {
      name: 'Pop',
      equation: 'sin(TAU*(380*tau + 16*(1 - exp(-45*tau)))) * exp(-35*tau)',
      at: POPS, len: 0.15, gain: 0.38,
    },
    {
      name: 'Whoosh',
      equation: `noise() * (tau < ${GROW} ? (tau/${GROW})^2 : exp(-(tau - ${GROW})*12))`,
      at: CUTS.map((c) => c - GROW), len: 0.8, lp: 2600, gain: 0.4,
    },
    {
      name: 'Riser',
      equation: 'noise() * min(1, tau/1.6)^3',
      at: [1.6, 20.8], len: 1.6, hp: 2200, gain: 0.28,
    },
    {
      name: 'Crash',
      equation: 'noise() * exp(-2.2*tau)',
      at: [3.2, 22.4, 25.6, 30.4], len: 1.6, hp: 5000, gain: 0.22,
    },
  ],
};
