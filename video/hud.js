// automix HUD overlaid on the visualizer for the video. window.__hudInit(data); window.__hud(t) per frame.
(() => {
  const css = `
  #hud{position:fixed;inset:0;z-index:100000;pointer-events:none;color:#f2efe8;
       font-family:"SF Pro Display",-apple-system,"Helvetica Neue",sans-serif;-webkit-font-smoothing:antialiased}
  #hud .mono{font-family:"SF Mono",ui-monospace,Menlo,monospace}
  #hud .glass{background:rgba(8,8,10,.42);backdrop-filter:blur(18px) saturate(1.2);-webkit-backdrop-filter:blur(18px) saturate(1.2);
       border:1px solid rgba(255,255,255,.10);border-radius:14px}
  #np{position:absolute;left:56px;bottom:56px;width:560px;padding:22px 26px 20px;transition:opacity .4s}
  #np .brand{display:flex;align-items:center;gap:10px;font:600 13px/1 "SF Mono",Menlo,monospace;letter-spacing:.24em;text-transform:uppercase;color:rgba(242,239,232,.62)}
  #np .brand i{width:8px;height:8px;border-radius:50%;background:#c7a169;box-shadow:0 0 12px #c7a169}
  #np .lbl{margin-top:18px;font:500 11px/1 "SF Mono",Menlo,monospace;letter-spacing:.28em;color:rgba(242,239,232,.5);text-transform:uppercase}
  #np .title{margin-top:8px;font:650 34px/1.12 "SF Pro Display",-apple-system,sans-serif;letter-spacing:-.01em}
  #np .artist{margin-top:6px;font:400 19px/1.3 "SF Pro Text",-apple-system,sans-serif;color:rgba(242,239,232,.72)}
  #np .meta{margin-top:12px;font:500 14px/1 "SF Mono",Menlo,monospace;color:#c7a169;letter-spacing:.06em}
  #tx{position:absolute;left:56px;bottom:56px;width:640px;padding:22px 26px 22px;opacity:0;transition:opacity .45s}
  #tx .row{display:flex;align-items:center;gap:14px;margin-top:14px}
  #tx .deck{flex:1;min-width:0}
  #tx .deck .d{font:600 11px/1 "SF Mono",Menlo,monospace;letter-spacing:.24em;color:rgba(242,239,232,.5)}
  #tx .deck .t{margin-top:7px;font:600 18px/1.2 "SF Pro Display",sans-serif;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  #tx .deck .m{margin-top:5px;font:500 12px/1 "SF Mono",Menlo,monospace;color:rgba(242,239,232,.6)}
  #tx .xf{width:120px;height:2px;background:rgba(255,255,255,.18);position:relative;border-radius:1px}
  #tx .xf b{position:absolute;top:-7px;width:10px;height:16px;margin-left:-5px;border-radius:3px;background:#c7a169;box-shadow:0 0 14px rgba(199,161,105,.8)}
  #tx .tech{font:650 28px/1.1 "SF Pro Display",sans-serif;letter-spacing:-.01em}
  #tx .tech small{font:500 13px/1 "SF Mono",Menlo,monospace;color:#c7a169;letter-spacing:.08em;margin-left:12px;vertical-align:middle}
  #tx .desc{margin-top:8px;font:400 16px/1.35 "SF Pro Text",sans-serif;color:rgba(242,239,232,.78)}
  #tx .bar{margin-top:16px;height:3px;border-radius:2px;background:rgba(255,255,255,.14);overflow:hidden}
  #tx .bar b{display:block;height:100%;width:0;background:linear-gradient(90deg,#c7a169,#f3dcb2)}
  #tl{position:absolute;right:56px;bottom:56px;width:520px;padding:16px 18px 14px}
  #tl .lbl{display:flex;justify-content:space-between;font:500 11px/1 "SF Mono",Menlo,monospace;letter-spacing:.2em;color:rgba(242,239,232,.55);text-transform:uppercase}
  #tl .segs{position:relative;display:flex;height:22px;margin-top:12px;border-radius:4px;overflow:hidden;gap:2px}
  #tl .seg{height:100%;background:rgba(255,255,255,.14);display:flex;align-items:center;justify-content:center;font:600 10px/1 "SF Mono",Menlo,monospace;color:rgba(242,239,232,.7)}
  #tl .seg.tr{background:rgba(199,161,105,.55)}
  #tl .seg.on{background:rgba(255,255,255,.34);color:#fff}
  #tl .seg.tr.on{background:#c7a169}
  #tl .ph{position:absolute;top:-4px;bottom:-4px;width:2px;background:#fff;box-shadow:0 0 8px #fff}
  #mode{position:absolute;left:56px;top:48px;font:500 12px/1 "SF Mono",Menlo,monospace;letter-spacing:.24em;text-transform:uppercase;color:rgba(242,239,232,.55)}
  #mode b{color:#f2efe8;font-weight:600}
  #card{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;opacity:0;background:radial-gradient(ellipse at center,rgba(0,0,0,.25),rgba(0,0,0,.7))}
  #card h1{margin:0;font:700 76px/1 "SF Pro Display",sans-serif;letter-spacing:-.02em}
  #card h1 i{font-style:normal;color:#c7a169;font-weight:300;margin:0 18px}
  #card p{margin:22px 0 0;font:500 16px/1.6 "SF Mono",Menlo,monospace;letter-spacing:.2em;text-transform:uppercase;color:rgba(242,239,232,.7)}
  #card .credits{margin-top:28px;font:400 15px/1.75 "SF Pro Text",sans-serif;letter-spacing:0;text-transform:none;color:rgba(242,239,232,.82)}
  `;
  const el = (tag, attrs = {}, html = '') => { const e = document.createElement(tag); Object.assign(e, attrs); e.innerHTML = html; return e; };
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  const ease = (x) => x * x * (3 - 2 * x);
  let D, root, np, tx, tl, ph, segEls, mode, card;

  window.__hudInit = (data) => {
    D = data;
    document.head.appendChild(el('style', {}, css));
    root = el('div', { id: 'hud' });
    np = el('div', { id: 'np', className: 'glass' });
    tx = el('div', { id: 'tx', className: 'glass' });
    tl = el('div', { id: 'tl', className: 'glass' });
    mode = el('div', { id: 'mode' });
    card = el('div', { id: 'card' });
    root.append(mode, np, tx, tl, card);
    document.body.appendChild(root);
    tl.innerHTML = `<div class="lbl"><span>automix set</span><span class="mono" id="tlt"></span></div><div class="segs"></div>`;
    const segs = tl.querySelector('.segs');
    const total = D.duration;
    segEls = D.markers.map((m) => {
      const s = el('div', { className: 'seg' + (m.type === 'transition' ? ' tr' : '') }, m.type === 'track' ? String(m.index + 1) : '');
      s.style.flex = `0 0 calc(${((m.end_sec - m.start_sec) / total) * 100}% - 2px)`;
      segs.appendChild(s); return s;
    });
    ph = el('div', { className: 'ph' }); segs.appendChild(ph);
  };

  window.__hud = (t) => {
    const ms = D.markers;
    const k = ms.findIndex((m) => t >= m.start_sec && t < m.end_sec);
    const i = k < 0 ? ms.length - 1 : k;
    const m = ms[i];
    segEls.forEach((s, j) => s.classList.toggle('on', j === i));
    ph.style.left = `${(t / D.duration) * 100}%`;
    tl.querySelector('#tlt').textContent = `${fmt(t)} / ${fmt(D.duration)}`;
    const md = D.modes.filter((x) => x.at <= t).pop() || D.modes[0];
    mode.innerHTML = `visualizzer · <b>${md.label}</b>`;
    if (m.type === 'track') {
      const tr = D.tracks[m.index];
      np.innerHTML = `<div class="brand"><i></i>automix</div><div class="lbl">now playing · track ${m.index + 1} of ${D.tracks.length}</div>
        <div class="title">${tr.title}</div><div class="artist">${tr.artist}</div><div class="meta">${Math.round(m.bpm)} BPM · ${m.key}</div>`;
    }
    let txo = 0;
    if (m.type === 'transition') {
      const p = (t - m.start_sec) / Math.max(0.01, m.end_sec - m.start_sec);
      const a = D.tracks[m.from], b = D.tracks[m.to];
      const lib = m.stems ? 'stems' : 'full mix';
      tx.innerHTML = `<div class="np brand" style="display:flex;align-items:center;gap:10px;font:600 13px/1 'SF Mono',Menlo,monospace;letter-spacing:.24em;color:rgba(242,239,232,.62)"><i style="width:8px;height:8px;border-radius:50%;background:#c7a169;box-shadow:0 0 12px #c7a169;display:inline-block"></i>AUTOMIX · TRANSITION</div>
        <div class="tech" style="margin-top:16px">${m.label}<small>${m.bars ? m.bars + ' BARS · ' : ''}${lib.toUpperCase()}</small></div>
        <div class="desc">${m.description || ''}</div>
        <div class="row"><div class="deck"><div class="d">DECK A</div><div class="t">${a.title}</div><div class="m">${Math.round(m.bpm_from)} BPM · ${m.key_from}</div></div>
          <div class="xf"><b style="left:${ease(Math.min(1, Math.max(0, p))) * 100}%"></b></div>
          <div class="deck" style="text-align:right"><div class="d">DECK B</div><div class="t">${b.title}</div><div class="m">${Math.round(m.bpm_to)} BPM · ${m.key_to}</div></div></div>
        <div class="bar"><b style="width:${p * 100}%"></b></div>`;
      txo = Math.min(1, (t - m.start_sec) / 0.6, (m.end_sec - t) / 0.6);
    }
    tx.style.opacity = txo; np.style.opacity = 1 - txo;
    // intro + outro cards
    const intro = t < D.introSec ? Math.min(1, t / 0.8, (D.introSec - t) / 1.2) : 0;
    const outro = t > D.duration - D.outroSec ? Math.min(1, (t - (D.duration - D.outroSec)) / 1.5) : 0;
    const c = Math.max(intro, outro);
    card.style.opacity = c;
    const hudO = 1 - c;
    [np, tx, tl, mode].forEach((e) => (e.style.filter = `opacity(${hudO})`));
    if (intro > 0) card.innerHTML = `<h1>automix<i>×</i>visualizzer</h1><p>an AI DJ mix · ${D.subtitle} · ${D.tracks.length} tracks</p>`;
    else if (outro > 0) card.innerHTML = `<h1>automix<i>×</i>visualizzer</h1><p>mixed by automix · visuals by visualizzer</p>
      <div class="credits">${D.tracks.map((tr) => `${tr.title} — ${tr.artist}`).join('<br>')}<br><span style="color:#c7a169">${D.credit}</span></div>`;
  };
})();
