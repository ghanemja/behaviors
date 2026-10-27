const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const outDir = process.argv[2];
  const fps = +(process.argv[3] || 30);
  const only = process.argv[4];
  fs.mkdirSync(outDir, { recursive: true });
  const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const page = await browser.newPage({ viewport: { width: 1080, height: 1350 }, deviceScaleFactor: 1 });
  const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await page.goto('http://localhost:8765/reel.html', { waitUntil: 'networkidle', timeout: 90000 });
  const dur = await page.evaluate(() => window.setup());
  if (only) {
    for (const t of only.split(',').map(Number)) {
      await page.evaluate(t => window.renderAt(t), t);
      await page.screenshot({ path: `${outDir}/still_${t}.png` });
    }
  } else {
    const n = Math.round(dur * fps);
    for (let f = 0; f < n; f++) {
      await page.evaluate(t => window.renderAt(t), f / fps);
      await page.screenshot({ path: `${outDir}/f${String(f).padStart(4, '0')}.jpg`, type: 'jpeg', quality: 92 });
      if (f % 100 === 0) console.log('frame', f, '/', n);
    }
  }
  console.log('errors:', JSON.stringify(errs.slice(0, 5)));
  await browser.close();
})();
