// Visitor analytics through GoatCounter (no cookies, so no consent banner).
// Add to a page with one tag:
//   <script src="analytics.js" data-goatcounter="CODE" data-site="NAME"></script>
// CODE is the GoatCounter site (CODE.goatcounter.com); NAME prefixes click events so
// sites sharing one GoatCounter dashboard stay apart.
// Page views are counted by GoatCounter's own script. This file adds click events: every
// link and button click is sent as "NAME click: <label>", and pages can call
// window.track('name') for things that aren't buttons (e.g. clicks on a 3D canvas).
(() => {
  const me = document.currentScript;
  const code = me && me.dataset.goatcounter;
  if (!code) return;
  // GoatCounter's count.js sends to the first script tag with data-goatcounter; take ours off
  // so it finds its own (otherwise it posts to "ghanemja" on this site and nothing is counted).
  me.removeAttribute('data-goatcounter');
  const site = me.dataset.site || location.pathname.split('/')[1] || location.hostname;

  const gc = document.createElement('script');
  gc.async = true;
  gc.src = 'https://gc.zgo.at/count.js';
  gc.dataset.goatcounter = `https://${code}.goatcounter.com/count`;
  document.head.appendChild(gc);

  const queue = [];
  const send = (name) => {
    const g = window.goatcounter;
    if (!g || !g.count) { queue.push(name); return; }
    g.count({ path: `${site} ${name}`.slice(0, 200), title: location.pathname, event: true });
  };
  window.track = send;
  gc.addEventListener('load', () => { while (queue.length) send(queue.shift()); });

  const labelOf = (el) => {
    const text = el.dataset.track || el.getAttribute('aria-label') || el.textContent.replace(/\s+/g, ' ').trim();
    if (text) return text.slice(0, 60);
    const href = el.getAttribute('href');
    return href ? href.replace(/^https?:\/\//, '') : el.id || el.tagName.toLowerCase();
  };
  document.addEventListener('click', (e) => {
    const el = e.target.closest && e.target.closest('a, button, [data-track]');
    if (!el) return;
    const href = el.getAttribute('href');
    const out = href && /^(https?:|mailto:)/.test(href) && !href.startsWith(location.origin) ? ' ↗' : '';
    send(`click: ${labelOf(el)}${out}`);
  }, true);
})();
