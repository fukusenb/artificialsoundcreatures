"use strict";
/*
 * Artificial Sound Creatures - インタラクティブ合唱シミュレーター (2 層モデル)
 *
 * 遅い層 (Felix Hess): 内部状態で「鳴くかどうか」を決める
 *   - 自発 spont: 沈黙中に溜まり、鳴くと疲れて減る(種火になる)
 *   - 興奮 exc  : カエルの声を聞くと増加、時間減衰(合唱の持続)
 *   - 抑制 inh  : 環境音(人など)で増加、時間減衰
 *   - 駆動 D = baseline + spont + exc - inh。D>閾値 で CALLING、下回ると SILENT。
 * 速い層 (合原/逆相): 「いつ鳴くか」
 *   - SILENT の間だけ他個体の鳴きとタイミングを聞く。
 *   - SILENT->CALLING の瞬間に、最も大きく聞いた鳴きへ +T/2(逆相)で位相を一度だけセット。
 *   - 鳴き始めたらタイミングは聞かず自走。以降は音の種類で内部状態を更新するだけ。
 */

let FIELD_W = 1000, FIELD_H = 600;
const RASTER_H = 150;
const EPS = 1e-6;

const COLORS = [
  "#0d9488", "#ea580c", "#dc2626", "#2563eb", "#7c3aed", "#16a34a",
  "#db2777", "#ca8a04", "#0891b2", "#c2410c", "#059669", "#9333ea",
];

const state = {
  frogs: [], noises: [], pending: [], callLog: [],
  t: 0, playing: true, drag: null, selected: null,
  params: {
    T: 1.0, spread: 0.05, gate: 0.15, refDist: 150, latency: 0.1, speed: 1.0,
    baseline: 0.15, threshold: 1.0, spontRate: 0.20, fatigue: 0.06,
    excGain: 0.9, excTau: 2.5, inhGain: 2.5, inhTau: 3.0,
    latComp: true, showLinks: true, sound: false, callVol: 0.9,
  },
  audio: null, customBuffer: null,
};

let field, fctx, raster, rctx;

function rand() { return Math.random(); }
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeFrog(x, y) {
  const P = state.params;
  return {
    uid: makeFrog._n = (makeFrog._n || 0) + 1,
    x, y,
    phase: rand(),
    period: Math.max(0.3, P.T * (1 + P.spread * randn())),
    threshold: P.threshold * (1 + 0.15 * randn()),
    active: false,
    spont: rand() * 0.3, exc: 0, inh: 0, D: 0,
    lastHeardTime: -1e9, lastHeardLoud: 0,
    flash: 0,
  };
}
function makeNoise(x, y) {
  return { uid: makeNoise._n = (makeNoise._n || 0) + 1, x, y, level: 0.9, radius: 150 };
}

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

function noiseAt(x, y) {
  let s = 0;
  for (const nz of state.noises) {
    const d = Math.hypot(x - nz.x, y - nz.y);
    s += nz.level / (1 + (d / nz.radius) * (d / nz.radius));
  }
  return s;
}

function step(dt) {
  const P = state.params;
  const frogs = state.frogs;

  for (const f of frogs) {
    // --- 遅い層: 内部状態の更新 ---
    f.exc -= f.exc * (dt / P.excTau);
    f.inh -= f.inh * (dt / P.inhTau);
    f.inh += P.inhGain * noiseAt(f.x, f.y) * dt;
    if (f.active) f.spont = Math.max(0, f.spont - P.fatigue * dt);
    else f.spont = Math.min(1.2, f.spont + P.spontRate * dt);
    f.D = P.baseline + f.spont + f.exc - f.inh;
    f.flash = Math.max(0, f.flash - dt * 3.0);

    // --- 状態遷移 ---
    if (!f.active && f.D > f.threshold) {
      if (state.t - f.lastHeardTime < 3 * f.period) {
        const comp = P.latComp ? P.latency / f.period : 0;
        const dtSince = state.t - f.lastHeardTime;
        f.phase = ((0.5 + comp - dtSince / f.period) % 1 + 1) % 1;   // +T/2 逆相を一発セット
      }
      f.active = true;
    } else if (f.active && f.D < f.threshold) {
      f.active = false;
    }
  }

  // --- 速い層: CALLING の個体だけ発声 ---
  for (let i = 0; i < frogs.length; i++) {
    const f = frogs[i];
    if (!f.active) continue;
    f.phase += dt / f.period;
    if (f.phase >= 1.0) {
      f.phase -= 1.0;
      f.flash = 1.0;
      f.exc += 0.15 * P.excGain;
      state.callLog.push({ t: state.t, idx: i });
      state.pending.push({ applyTime: state.t + P.latency, source: i });
      if (P.sound) playCall(f.x);
    }
  }

  // --- 聞こえた鳴きの処理 ---
  const due = [], still = [];
  for (const ev of state.pending) {
    if (ev.applyTime <= state.t + 1e-9) due.push(ev); else still.push(ev);
  }
  state.pending = still;
  for (const ev of due) {
    const src = frogs[ev.source];
    if (!src) continue;
    for (const l of frogs) {
      if (l === src) continue;
      const d = dist(l, src);
      const heard = 1.0 / (1 + (d / P.refDist) * (d / P.refDist));
      if (heard < P.gate) continue;
      l.exc += P.excGain * heard;                    // カエル声で興奮(全個体)
      if (!l.active && heard >= l.lastHeardLoud * 0.9) {   // SILENT の個体はタイミングを記憶
        l.lastHeardTime = state.t;
        l.lastHeardLoud = Math.max(heard, l.lastHeardLoud * 0.5);
      }
    }
  }
  for (const f of frogs) f.lastHeardLoud *= (1 - dt / 1.5);

  if (state.callLog.length > 4000) {
    const cutoff = state.t - 20;
    state.callLog = state.callLog.filter((c) => c.t >= cutoff);
  }
}

function nearestOf(i) {
  const frogs = state.frogs;
  let best = -1, bd = 1e9;
  for (let j = 0; j < frogs.length; j++) {
    if (i === j) continue;
    const d = dist(frogs[i], frogs[j]);
    if (d < bd) { bd = d; best = j; }
  }
  return best;
}

function antiphaseScore() {
  const frogs = state.frogs;
  let sum = 0, cnt = 0;
  for (let i = 0; i < frogs.length; i++) {
    const j = nearestOf(i);
    if (j < 0) continue;
    if (!frogs[i].active || !frogs[j].active) continue;
    let pd = Math.abs(frogs[i].phase - frogs[j].phase) % 1;
    pd = Math.min(pd, 1 - pd);
    sum += pd / 0.5; cnt++;
  }
  return cnt ? sum / cnt : null;
}

function activeFraction() {
  if (state.frogs.length === 0) return 0;
  return state.frogs.filter((f) => f.active).length / state.frogs.length;
}

// ---- 描画 ------------------------------------------------------------
function render() {
  fctx.clearRect(0, 0, FIELD_W, FIELD_H);
  fctx.fillStyle = "#f4f6f9";
  fctx.fillRect(0, 0, FIELD_W, FIELD_H);

  for (const nz of state.noises) {
    const g = fctx.createRadialGradient(nz.x, nz.y, 2, nz.x, nz.y, nz.radius);
    const a = 0.12 + 0.30 * nz.level;
    g.addColorStop(0, `rgba(220,60,60,${a})`);
    g.addColorStop(1, "rgba(220,60,60,0)");
    fctx.fillStyle = g;
    fctx.beginPath(); fctx.arc(nz.x, nz.y, nz.radius, 0, 2 * Math.PI); fctx.fill();
    fctx.strokeStyle = (state.selected === nz) ? "#111827" : "rgba(200,50,50,0.7)";
    fctx.lineWidth = (state.selected === nz) ? 2 : 1;
    fctx.beginPath(); fctx.arc(nz.x, nz.y, 9, 0, 2 * Math.PI); fctx.stroke();
    fctx.fillStyle = "rgba(200,50,50,0.9)"; fctx.font = "10px system-ui";
    fctx.fillText("環境音/人", nz.x - 24, nz.y - 13);
  }

  if (state.params.showLinks) {
    const seen = new Set();
    for (let i = 0; i < state.frogs.length; i++) {
      const j = nearestOf(i);
      if (j < 0) continue;
      if (!state.frogs[i].active || !state.frogs[j].active) continue;
      const key = i < j ? i + "-" + j : j + "-" + i;
      if (seen.has(key)) continue;
      seen.add(key);
      let pd = Math.abs(state.frogs[i].phase - state.frogs[j].phase) % 1;
      pd = Math.min(pd, 1 - pd) / 0.5;
      const r = Math.round(210 * (1 - pd)), gg = Math.round(170 * pd);
      fctx.strokeStyle = `rgba(${r},${gg},60,0.7)`; fctx.lineWidth = 1.5;
      fctx.beginPath();
      fctx.moveTo(state.frogs[i].x, state.frogs[i].y);
      fctx.lineTo(state.frogs[j].x, state.frogs[j].y);
      fctx.stroke();
    }
  }

  for (let i = 0; i < state.frogs.length; i++) {
    const f = state.frogs[i];
    const col = COLORS[f.uid % COLORS.length];
    if (f.active) {
      if (f.flash > 0) {
        fctx.strokeStyle = `rgba(15,23,42,${0.55 * f.flash})`; fctx.lineWidth = 2;
        fctx.beginPath(); fctx.arc(f.x, f.y, 14 + (1 - f.flash) * 26, 0, 2 * Math.PI); fctx.stroke();
      }
      fctx.strokeStyle = "rgba(0,0,0,0.12)"; fctx.lineWidth = 3;
      fctx.beginPath(); fctx.arc(f.x, f.y, 14, 0, 2 * Math.PI); fctx.stroke();
      fctx.strokeStyle = col; fctx.lineWidth = 3;
      fctx.beginPath();
      fctx.arc(f.x, f.y, 14, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * f.phase); fctx.stroke();
      fctx.fillStyle = col;
      fctx.beginPath(); fctx.arc(f.x, f.y, 9 + 3 * f.flash, 0, 2 * Math.PI); fctx.fill();
    } else {
      // SILENT: 灰色の輪 + 「ためこみ度(D/閾値)」の弧
      const charge = Math.max(0, Math.min(1, f.D / f.threshold));
      fctx.strokeStyle = "rgba(0,0,0,0.15)"; fctx.lineWidth = 3;
      fctx.beginPath(); fctx.arc(f.x, f.y, 14, 0, 2 * Math.PI); fctx.stroke();
      fctx.strokeStyle = "rgba(120,120,120,0.9)"; fctx.lineWidth = 3;
      fctx.beginPath();
      fctx.arc(f.x, f.y, 14, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * charge); fctx.stroke();
      fctx.fillStyle = "rgba(150,150,150,0.5)";
      fctx.beginPath(); fctx.arc(f.x, f.y, 7, 0, 2 * Math.PI); fctx.fill();
    }
    if (state.selected === f) {
      fctx.strokeStyle = "#111827"; fctx.lineWidth = 2;
      fctx.beginPath(); fctx.arc(f.x, f.y, 20, 0, 2 * Math.PI); fctx.stroke();
    }
    fctx.fillStyle = "rgba(0,0,0,0.65)"; fctx.font = "10px system-ui";
    fctx.fillText("#" + i, f.x - 6, f.y + 3);
  }

  drawRaster();
  updateMetric();
}

function drawRaster() {
  rctx.clearRect(0, 0, field.width, RASTER_H);
  rctx.fillStyle = "#f4f6f9"; rctx.fillRect(0, 0, field.width, RASTER_H);
  const W = field.width, win = 20, t0 = state.t - win;
  const n = Math.max(1, state.frogs.length);
  rctx.strokeStyle = "rgba(0,0,0,0.06)";
  for (let s = Math.ceil(t0); s <= state.t; s++) {
    const x = ((s - t0) / win) * W;
    rctx.beginPath(); rctx.moveTo(x, 0); rctx.lineTo(x, RASTER_H); rctx.stroke();
  }
  for (const c of state.callLog) {
    if (c.t < t0 || c.idx >= state.frogs.length) continue;
    const x = ((c.t - t0) / win) * W;
    const y = 16 + (c.idx / Math.max(1, n - 1)) * (RASTER_H - 32);
    const col = COLORS[(state.frogs[c.idx] ? state.frogs[c.idx].uid : c.idx) % COLORS.length];
    rctx.strokeStyle = col; rctx.lineWidth = 2;
    rctx.beginPath(); rctx.moveTo(x, y - 5); rctx.lineTo(x, y + 5); rctx.stroke();
  }
  rctx.fillStyle = "rgba(0,0,0,0.5)"; rctx.font = "11px system-ui";
  rctx.fillText("call raster (last 20 s)  \u2192 time", 8, 14);
}

function updateMetric() {
  const s = antiphaseScore();
  const el = document.getElementById("metric");
  if (s === null) { el.textContent = "—"; el.style.color = "#57606a"; }
  else {
    el.textContent = Math.round(s * 100) + "%";
    el.style.color = `rgb(${Math.round(210 * (1 - s))},${Math.round(160 * s)},60)`;
  }
  const af = document.getElementById("activefrac");
  if (af) af.textContent = Math.round(activeFraction() * 100) + "%";
}

// ---- WebAudio -------------------------------------------------------
function ensureAudio() {
  if (!state.audio) {
    try { state.audio = new (window.AudioContext || window.webkitAudioContext)(); }
    catch (e) { return null; }
  }
  if (state.audio.state === "suspended") state.audio.resume();
  return state.audio;
}
function loadAudioFile(file) {
  const ac = ensureAudio();
  const nameEl = document.getElementById("audioName");
  if (!ac || !file) return;
  if (nameEl) nameEl.textContent = "読み込み中…";
  file.arrayBuffer().then((b) => ac.decodeAudioData(b)).then((dec) => {
    state.customBuffer = dec; if (nameEl) nameEl.textContent = file.name;
  }).catch((e) => { if (nameEl) nameEl.textContent = "読み込み失敗"; console.error(e); });
}
function playCall(x) {
  const ac = ensureAudio(); if (!ac) return;
  const now = ac.currentTime;
  const gain = ac.createGain();
  const pan = ac.createStereoPanner ? ac.createStereoPanner() : null;
  const dest = pan || ac.destination;
  if (pan) { pan.pan.value = Math.max(-1, Math.min(1, (x / FIELD_W) * 2 - 1)); pan.connect(ac.destination); }
  if (state.customBuffer) {
    const src = ac.createBufferSource();
    src.buffer = state.customBuffer;
    src.playbackRate.value = 0.97 + 0.06 * Math.random();
    gain.gain.value = state.params.callVol;
    src.connect(gain); gain.connect(dest); src.start(now);
  } else {
    const osc = ac.createOscillator(), am = ac.createOscillator(), amGain = ac.createGain();
    osc.frequency.value = 2600 + 400 * Math.random();
    am.frequency.value = 45; amGain.gain.value = 0.5;
    am.connect(amGain); amGain.connect(gain.gain);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.25 * state.params.callVol, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.22);
    osc.connect(gain); gain.connect(dest);
    osc.start(now); am.start(now); osc.stop(now + 0.24); am.stop(now + 0.24);
  }
}

// ---- 入力(ドラッグ&ドロップ) --------------------------------------
function pointerPos(evt) {
  const r = field.getBoundingClientRect();
  return { x: (evt.clientX - r.left) * (field.width / r.width),
           y: (evt.clientY - r.top) * (field.height / r.height) };
}
function pick(p) {
  for (const f of state.frogs) if (Math.hypot(f.x - p.x, f.y - p.y) < 16) return { kind: "frog", obj: f };
  for (const nz of state.noises) if (Math.hypot(nz.x - p.x, nz.y - p.y) < 14) return { kind: "noise", obj: nz };
  return null;
}
function onDown(evt) {
  const p = pointerPos(evt);
  const hit = pick(p);
  if (hit) { state.selected = hit.obj; state.drag = { obj: hit.obj, dx: hit.obj.x - p.x, dy: hit.obj.y - p.y }; }
  else state.selected = null;
  refreshInspector();
}
function onMove(evt) {
  if (!state.drag) return;
  const p = pointerPos(evt), o = state.drag.obj;
  o.x = Math.max(0, Math.min(FIELD_W, p.x + state.drag.dx));
  o.y = Math.max(0, Math.min(FIELD_H, p.y + state.drag.dy));
}
function onUp() { state.drag = null; }

// ---- コントロール ----------------------------------------------------
function bindSlider(id, key, fmt) {
  const el = document.getElementById(id);
  const out = document.getElementById(id + "Val");
  const apply = () => {
    state.params[key] = parseFloat(el.value);
    if (out) out.textContent = fmt ? fmt(state.params[key]) : el.value;
  };
  el.addEventListener("input", apply); apply();
}

function setupControls() {
  bindSlider("T", "T", (v) => v.toFixed(2) + " s");
  bindSlider("spread", "spread", (v) => (v * 100).toFixed(0) + " %");
  bindSlider("baseline", "baseline", (v) => v.toFixed(2));
  bindSlider("threshold", "threshold", (v) => v.toFixed(2));
  bindSlider("spontrate", "spontRate", (v) => v.toFixed(2));
  bindSlider("excgain", "excGain", (v) => v.toFixed(2));
  bindSlider("exctau", "excTau", (v) => v.toFixed(1) + " s");
  bindSlider("inhgain", "inhGain", (v) => v.toFixed(2));
  bindSlider("inhtau", "inhTau", (v) => v.toFixed(1) + " s");
  bindSlider("gate", "gate", (v) => v.toFixed(2));
  bindSlider("refdist", "refDist", (v) => v.toFixed(0) + " px");
  bindSlider("latency", "latency", (v) => (v * 1000).toFixed(0) + " ms");
  bindSlider("speed", "speed", (v) => v.toFixed(2) + "x");
  bindSlider("callvol", "callVol", (v) => v.toFixed(2));

  document.getElementById("latcomp").addEventListener("change", (e) => state.params.latComp = e.target.checked);
  document.getElementById("showlinks").addEventListener("change", (e) => state.params.showLinks = e.target.checked);
  document.getElementById("sound").addEventListener("change", (e) => { state.params.sound = e.target.checked; if (e.target.checked) ensureAudio(); });
  document.getElementById("audioFile").addEventListener("change", (e) => { if (e.target.files && e.target.files[0]) loadAudioFile(e.target.files[0]); });
  document.getElementById("btnSynth").addEventListener("click", () => { state.customBuffer = null; document.getElementById("audioName").textContent = "(合成音)"; });

  document.getElementById("btnPlay").addEventListener("click", () => {
    state.playing = !state.playing;
    document.getElementById("btnPlay").textContent = state.playing ? "\u23f8 一時停止" : "\u25b6 再生";
  });
  document.getElementById("btnReset").addEventListener("click", () => {
    for (const f of state.frogs) { f.phase = rand(); f.active = false; f.spont = rand() * 0.3; f.exc = 0; f.inh = 0; f.lastHeardTime = -1e9; f.lastHeardLoud = 0; }
    state.callLog = []; state.pending = []; state.t = 0;
  });
  document.getElementById("btnAddFrog").addEventListener("click", () => {
    state.frogs.push(makeFrog(80 + rand() * (FIELD_W - 160), 80 + rand() * (FIELD_H - 160)));
  });
  document.getElementById("btnAddNoise").addEventListener("click", () => {
    const nz = makeNoise(FIELD_W / 2, FIELD_H / 2); state.noises.push(nz); state.selected = nz; refreshInspector();
  });
  document.getElementById("btnDelete").addEventListener("click", deleteSelected);
  document.getElementById("btnClearNoise").addEventListener("click", () => { state.noises = []; state.selected = null; refreshInspector(); });

  document.getElementById("nzLevel").addEventListener("input", (e) => { if (state.selected && "level" in state.selected) state.selected.level = parseFloat(e.target.value); });
  document.getElementById("nzRadius").addEventListener("input", (e) => { if (state.selected && "radius" in state.selected) state.selected.radius = parseFloat(e.target.value); });
  window.addEventListener("keydown", (e) => { if (e.key === "Backspace" || e.key === "Delete") deleteSelected(); });
}

function deleteSelected() {
  if (!state.selected) return;
  let k = state.frogs.indexOf(state.selected); if (k >= 0) state.frogs.splice(k, 1);
  k = state.noises.indexOf(state.selected); if (k >= 0) state.noises.splice(k, 1);
  state.selected = null; refreshInspector();
}
function refreshInspector() {
  const box = document.getElementById("inspector");
  const sel = state.selected;
  if (sel && "level" in sel) {
    box.style.display = "block";
    document.getElementById("nzLevel").value = sel.level;
    document.getElementById("nzRadius").value = sel.radius;
  } else box.style.display = "none";
}

// ---- メインループ ----------------------------------------------------
let lastReal = 0;
function loop(ts) {
  const realDt = lastReal ? (ts - lastReal) / 1000 : 0;
  lastReal = ts;
  if (state.playing) {
    let simDt = Math.min(0.1, realDt) * state.params.speed;
    const h = 0.005;
    while (simDt > 0) { const d = Math.min(h, simDt); state.t += d; step(d); simDt -= d; }
  }
  render();
  requestAnimationFrame(loop);
}

function sizeCanvas() {
  const stage = field.parentElement;
  const w = Math.max(320, Math.min(1600, stage.clientWidth));
  FIELD_W = Math.round(w); FIELD_H = Math.round(w * 0.56);
  field.width = FIELD_W; field.height = FIELD_H;
  raster.width = FIELD_W; raster.height = RASTER_H;
  for (const f of state.frogs) { f.x = Math.min(f.x, FIELD_W); f.y = Math.min(f.y, FIELD_H); }
  for (const nz of state.noises) { nz.x = Math.min(nz.x, FIELD_W); nz.y = Math.min(nz.y, FIELD_H); }
}

function init() {
  field = document.getElementById("field");
  raster = document.getElementById("raster");
  fctx = field.getContext("2d");
  rctx = raster.getContext("2d");
  sizeCanvas();
  window.addEventListener("resize", sizeCanvas);

  const cx = FIELD_W / 2, cy = FIELD_H / 2, R = Math.min(FIELD_W, FIELD_H) * 0.32;
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * 2 * Math.PI;
    state.frogs.push(makeFrog(cx + R * Math.cos(a), cy + R * Math.sin(a)));
  }

  field.addEventListener("pointerdown", onDown);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);

  setupControls();
  refreshInspector();
  requestAnimationFrame(loop);
}

window.addEventListener("DOMContentLoaded", init);
