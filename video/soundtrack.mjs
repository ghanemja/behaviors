// Render the video soundtrack (video/soundtrack.song.js) to a WAV file.
// Usage: node video/soundtrack.mjs soundtrack.wav
import { writeFileSync } from 'node:fs';
import { renderSong, toWav } from './vendor/equalizer.js';
import song from './soundtrack.song.js';

const out = process.argv[2] ?? 'soundtrack.wav';
const { left, right, sampleRate } = renderSong(song, { sampleRate: 44100 });
writeFileSync(out, toWav(left, right, sampleRate));
console.log(`wrote ${out}`);
