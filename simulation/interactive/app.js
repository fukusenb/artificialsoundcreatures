"use strict";
/*
 * Artificial Sound Creatures - インタラクティブ合唱シミュレーター
 *
 * カエル個体(パルス結合位相振動子)を 2 次元フィールドにドラッグ配置し、
 * 環境音源を置いて、逆相同期(交互鳴き)の自己組織化をリアルタイムに観察する。
 * モデルは simulation/frog_chorus.py と同じ規則:
 *   - phase を固有周期 T で進め、phase>=1 で発声・リセット
 *   - 近い個体ほど大きく聞こえ(音量 = A0/(1+(d/d0)^2))、閾値以上のイベントに結合
 *   - 反発 PRC で位相差 0.5(=π, 逆相)が安定化
 * 環境音は「検出のマスキング」として作用し、近くの個体は相手の鳴きを聞き逃しやすくなる。
 */

const FIELD_W = 660, FIELD_H = 480;
const RASTER_H = 150;
const A0 = 1.0;             // 発声音量
const EPS = 1e-6;

const COLORS = [
  "#4fd1c5", "#f6ad55", "#fc8181", "#63b3ed", "#b794f4", "#68d391",
  "#f687b3", "#f6e05e", "#76e4f7", "#ff8a65", "#9ae6b4", "#d6bcfa",
];

// ---- 状態 ------------------------------------------------------------
const state = {
  frogs: [],
  noises: [],
  pending: [],      // {applyTime, source}
  callLog: [],      // {t, idx}
  t: 0,
  playing: true,
  drag: null,       // {kind:'frog'|'noise', obj, dx, dy}
  selected: null,
  params: {
    K: 0.4, T: 1.0, spread: 0.05, gate: 0.15, latency: 0.1,
    refDist: 120, noiseMask: 0.8, speed: 1.0,
    mode: "prc", latComp: true, showLinks: true, sound: false,
  },
  audio: null,
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
  const T = state.params.T * (1 + state.params.spread * randn());
  return {
    uid: makeFrog._n = (makeFrog._n || 0) + 1,
    x, y,
    phase: rand(),
    period: Math.max(0.3, T),
    lastCall: -1e9,
    flash: 0,
  };
}
function makeNoise(x, y) {
  return { uid: makeNoise._n = (makeNoise._n || 0) + 1, x, y, level: 0.6, radius: 110 };
}

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

// ---- モデル 1 ステップ ----------------------------------------------
function step(dt) {
  const P = state.params;
  const frogs = state.frogs;

  // 1) 位相を進める → 発声
  for (let i = 0; i < frogs.length; i++) {
    const f = frogs[i];
    f.phase += dt / f.period;
    f.flash = Math.max(0, f.flash - dt * 3.0);
    if (f.phase >= 1.0) {
      f.phase -= 1.0;
      f.lastCall = state.t;
      f.flash = 1.0;
      state.callLog.push({ t: state.t, idx: i });
      state.pending.push({ applyTime: state.t + P.latency, source: i });
      if (P.sound) playCall(f.x);
    }
  }

  // 2) 遅延キューから「今聞こえる」発声を処理
  const due = [];
  const still = [];
  for (const ev of state.pending) {
    if (ev.applyTime <= state.t + 1e-9) due.push(ev); else still.push(ev);
  }
  state.pending = still;
  for (const ev of due) applyHeardCall(ev.source);

  // ラスターログの間引き(20 秒より古いものを捨てる)
  const cutoff = state.t - 20;
  if (state.callLog.length > 4000) {
    state.callLog = state.callLog.filter((c) => c.t >= cutoff);
  }
}

function localNoise(l) {
  let s = 0;
  for (const nz of state.noises) {
    const d = Math.hypot(l.x - nz.x, l.y - nz.y);
    s += nz.level / (1 + (d / nz.radius) * (d / nz.radius));
  }
  return s;
}

function applyHeardCall(srcIdx) {
  const P = state.params;
  const frogs = state.frogs;
  const src = frogs[srcIdx];
  if (!src) return;
  const d0 = P.refDist;

  // schedule モードは「最も大きく聞こえた 1 体」に絞る(ウィナーテイクオール)
  let candidates = [];
  for (let l = 0; l < frogs.length; l++) {
    if (l === srcIdx) continue;
    const lf = frogs[l];
    if (state.t - lf.lastCall < 0.2) continue;              // 不応期(自己ゲート)
    const d = dist(lf, src);
    const heard = A0 / (1 + (d / d0) * (d / d0));
    if (heard < P.gate) continue;                            // 遠い個体は無視
    // 環境音マスキング: 近くがうるさいほど聞き逃す
    const nz = localNoise(lf);
    const missProb = P.noiseMask * nz / (nz + heard + EPS);
    if (rand() < missProb) continue;
    candidates.push({ l, heard });
  }

  if (P.mode === "schedule") {
    if (candidates.length === 0) return;
    candidates.sort((a, b) => b.heard - a.heard);
    const { l, heard } = candidates[0];
    const lf = frogs[l];
    const comp = P.latComp ? P.latency / lf.period : 0;
    const target = (0.5 + comp) % 1.0;                       // +T/2 後に発声
    const gain = Math.min(1, P.K * heard);
    lf.phase = wrapBlend(lf.phase, target, gain);
  } else {
    // PRC モード: 反発結合を音量で重み付けして各個体に適用
    for (const { l, heard } of candidates) {
      const lf = frogs[l];
      const w = heard / A0;
      const dphi = 0.35 * P.K * w * Math.sin(2 * Math.PI * lf.phase);
      lf.phase = ((lf.phase + dphi) % 1 + 1) % 1;
    }
  }
}

function wrapBlend(a, target, g) {
  let diff = ((target - a + 0.5) % 1 + 1) % 1 - 0.5;         // -0.5..0.5 の最短差
  return ((a + g * diff) % 1 + 1) % 1;
}

// ---- 指標: 最近傍ペアの逆相度 ---------------------------------------
function antiphaseScore() {
  const frogs = state.frogs;
  if (frogs.length < 2) return null;
  let sum = 0, cnt = 0;
  for (let i = 0; i < frogs.length; i++) {
    let best = -1, bd = 1e9;
    for (let j = 0; j < frogs.length; j++) {
      if (i === j) continue;
      const d = dist(frogs[i], frogs[j]);
      if (d < bd) { bd = d; best = j; }
    }
    if (best < 0) continue;
    let pd = Math.abs(frogs[i].phase - frogs[best].phase) % 1;
    pd = Math.min(pd, 1 - pd);        // 0..0.5
    sum += pd / 0.5;                  // 0..1 (1=完全逆相)
    cnt++;
  }
  return cnt ? sum / cnt : null;
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

// ---- 描画 ------------------------------------------------------------
function render() {
  fctx.clearRect(0, 0, FIELD_W, FIELD_H);
  fctx.fillStyle = "#0d1117";
  fctx.fillRect(0, 0, FIELD_W, FIELD_H);

  // 環境音源(影響範囲)
  for (const nz of state.noises) {
    const g = fctx.createRadialGradient(nz.x, nz.y, 2, nz.x, nz.y, nz.radius);
    const a = 0.10 + 0.25 * nz.level;
    g.addColorStop(0, `rgba(220,80,80,${a})`);
    g.addColorStop(1, "rgba(220,80,80,0)");
    fctx.fillStyle = g;
    fctx.beginPath(); fctx.arc(nz.x, nz.y, nz.radius, 0, 2 * Math.PI); fctx.fill();
    fctx.strokeStyle = (state.selected === nz) ? "#fff" : "rgba(220,80,80,0.6)";
    fctx.lineWidth = (state.selected === nz) ? 2 : 1;
    fctx.beginPath(); fctx.arc(nz.x, nz.y, 9, 0, 2 * Math.PI); fctx.stroke();
    fctx.fillStyle = "rgba(220,80,80,0.9)";
    fctx.font = "10px system-ui";
    fctx.fillText("noise", nz.x - 15, nz.y - 13);
  }

  // 最近傍リンク(逆相度で色付け)
  if (state.params.showLinks) {
    const seen = new Set();
    for (let i = 0; i < state.frogs.length; i++) {
      const j = nearestOf(i);
      if (j < 0) continue;
      const key = i < j ? i + "-" + j : j + "-" + i;
      if (seen.has(key)) continue;
      seen.add(key);
      let pd = Math.abs(state.frogs[i].phase - state.frogs[j].phase) % 1;
      pd = Math.min(pd, 1 - pd) / 0.5;      // 0..1
      const r = Math.round(255 * (1 - pd)), gg = Math.round(200 * pd);
      fctx.strokeStyle = `rgba(${r},${gg},90,0.5)`;
      fctx.lineWidth = 1.5;
      fctx.beginPath();
      fctx.moveTo(state.frogs[i].x, state.frogs[i].y);
      fctx.lineTo(state.frogs[j].x, state.frogs[j].y);
      fctx.stroke();
    }
  }

  // カエル個体
  for (let i = 0; i < state.frogs.length; i++) {
    const f = state.frogs[i];
    const col = COLORS[f.uid % COLORS.length];
    // 発声フラッシュ(広がる輪)
    if (f.flash > 0) {
      fctx.strokeStyle = `rgba(255,255,255,${f.flash})`;
      fctx.lineWidth = 2;
      fctx.beginPath();
      fctx.arc(f.x, f.y, 14 + (1 - f.flash) * 26, 0, 2 * Math.PI);
      fctx.stroke();
    }
    // 位相リング
    fctx.strokeStyle = "rgba(255,255,255,0.18)";
    fctx.lineWidth = 3;
    fctx.beginPath(); fctx.arc(f.x, f.y, 14, 0, 2 * Math.PI); fctx.stroke();
    fctx.strokeStyle = col; fctx.lineWidth = 3;
    fctx.beginPath();
    fctx.arc(f.x, f.y, 14, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * f.phase);
    fctx.stroke();
    // 本体
    fctx.fillStyle = col;
    fctx.beginPath(); fctx.arc(f.x, f.y, 9 + 3 * f.flash, 0, 2 * Math.PI); fctx.fill();
    if (state.selected === f) {
      fctx.strokeStyle = "#fff"; fctx.lineWidth = 2;
      fctx.beginPath(); fctx.arc(f.x, f.y, 20, 0, 2 * Math.PI); fctx.stroke();
    }
    fctx.fillStyle = "rgba(255,255,255,0.75)";
    fctx.font = "10px system-ui";
    fctx.fillText("#" + i, f.x - 6, f.y + 3);
  }

  drawRaster();
  updateMetric();
}

function drawRaster() {
  rctx.clearRect(0, 0, field.width, RASTER_H);
  rctx.fillStyle = "#0d1117";
  rctx.fillRect(0, 0, field.width, RASTER_H);
  const W = field.width, win = 20;
  const t0 = state.t - win;
  const n = Math.max(1, state.frogs.length);
  // グリッド
  rctx.strokeStyle = "rgba(255,255,255,0.06)";
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
  rctx.fillStyle = "rgba(255,255,255,0.5)"; rctx.font = "11px system-ui";
  rctx.fillText("call raster (last 20 s)  \u2192 time", 8, 14);
}

function updateMetric() {
  const s = antiphaseScore();
  const el = document.getElementById("metric");
  if (s === null) { el.textContent = "\u2014"; return; }
  const pct = Math.round(s * 100);
  el.textContent = pct + "%";
  el.style.color = `rgb(${Math.round(255 * (1 - s))},${Math.round(200 * s)},90)`;
}

// ---- WebAudio(カエルっぽい鳴き) -----------------------------------
function playCall(x) {
  try {
    const ac = state.audio || (state.audio = new (window.AudioContext || window.webkitAudioContext)());
    const now = ac.currentTime;
    const osc = ac.createOscillator();
    const am = ac.createOscillator();
    const amGain = ac.createGain();
    const gain = ac.createGain();
    const pan = ac.createStereoPanner ? ac.createStereoPanner() : null;
    osc.frequency.value = 2600 + 400 * Math.random();
    am.frequency.value = 45; amGain.gain.value = 0.5;
    am.connect(amGain); amGain.connect(gain.gain);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.25, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.22);
    osc.connect(gain);
    if (pan) { pan.pan.value = Math.max(-1, Math.min(1, (x / FIELD_W) * 2 - 1)); gain.connect(pan); pan.connect(ac.destination); }
    else gain.connect(ac.destination);
    osc.start(now); am.start(now); osc.stop(now + 0.24); am.stop(now + 0.24);
  } catch (e) { /* ignore */ }
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
  if (hit) {
    state.selected = hit.obj;
    state.drag = { kind: hit.kind, obj: hit.obj, dx: hit.obj.x - p.x, dy: hit.obj.y - p.y };
  } else {
    state.selected = null;
  }
  refreshInspector();
}
function onMove(evt) {
  if (!state.drag) return;
  const p = pointerPos(evt);
  const o = state.drag.obj;
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
  bindSlider("K", "K", (v) => v.toFixed(2));
  bindSlider("T", "T", (v) => v.toFixed(2) + " s");
  bindSlider("spread", "spread", (v) => (v * 100).toFixed(0) + " %");
  bindSlider("gate", "gate", (v) => v.toFixed(2));
  bindSlider("latency", "latency", (v) => (v * 1000).toFixed(0) + " ms");
  bindSlider("refdist", "refDist", (v) => v.toFixed(0) + " px");
  bindSlider("noisemask", "noiseMask", (v) => v.toFixed(2));
  bindSlider("speed", "speed", (v) => v.toFixed(2) + "x");

  document.getElementById("mode").addEventListener("change", (e) => {
    state.params.mode = e.target.value;
  });
  document.getElementById("latcomp").addEventListener("change", (e) => {
    state.params.latComp = e.target.checked;
  });
  document.getElementById("showlinks").addEventListener("change", (e) => {
    state.params.showLinks = e.target.checked;
  });
  document.getElementById("sound").addEventListener("change", (e) => {
    state.params.sound = e.target.checked;
    if (e.target.checked && !state.audio) {
      state.audio = new (window.AudioContext || window.webkitAudioContext)();
    }
  });

  document.getElementById("btnPlay").addEventListener("click", () => {
    state.playing = !state.playing;
    document.getElementById("btnPlay").textContent = state.playing ? "\u23f8 一時停止" : "\u25b6 再生";
  });
  document.getElementById("btnReset").addEventListener("click", () => {
    for (const f of state.frogs) f.phase = rand();
    state.callLog = []; state.pending = []; state.t = 0;
  });
  document.getElementById("btnAddFrog").addEventListener("click", () => {
    state.frogs.push(makeFrog(80 + rand() * (FIELD_W - 160), 80 + rand() * (FIELD_H - 160)));
  });
  document.getElementById("btnAddNoise").addEventListener("click", () => {
    const nz = makeNoise(80 + rand() * (FIELD_W - 160), 80 + rand() * (FIELD_H - 160));
    state.noises.push(nz); state.selected = nz; refreshInspector();
  });
  document.getElementById("btnDelete").addEventListener("click", deleteSelected);
  document.getElementById("btnClearNoise").addEventListener("click", () => {
    state.noises = []; if (state.selected && state.noises.indexOf(state.selected) < 0) {} state.selected = null; refreshInspector();
  });

  // インスペクタ(選択中の環境音の level / radius)
  document.getElementById("nzLevel").addEventListener("input", (e) => {
    if (state.selected && "level" in state.selected) state.selected.level = parseFloat(e.target.value);
  });
  document.getElementById("nzRadius").addEventListener("input", (e) => {
    if (state.selected && "radius" in state.selected) state.selected.radius = parseFloat(e.target.value);
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Backspace" || e.key === "Delete") deleteSelected();
  });
}

function deleteSelected() {
  if (!state.selected) return;
  const fi = state.frogs.indexOf(state.selected);
  if (fi >= 0) state.frogs.splice(fi, 1);
  const ni = state.noises.indexOf(state.selected);
  if (ni >= 0) state.noises.splice(ni, 1);
  state.selected = null; refreshInspector();
}

function refreshInspector() {
  const box = document.getElementById("inspector");
  const sel = state.selected;
  if (sel && "level" in sel) {
    box.style.display = "block";
    document.getElementById("nzLevel").value = sel.level;
    document.getElementById("nzRadius").value = sel.radius;
  } else {
    box.style.display = "none";
  }
}

// ---- メインループ ----------------------------------------------------
let lastReal = 0;
function loop(ts) {
  const realDt = lastReal ? (ts - lastReal) / 1000 : 0;
  lastReal = ts;
  if (state.playing) {
    let simDt = Math.min(0.1, realDt) * state.params.speed;
    const h = 0.005;                       // 固定積分刻み
    while (simDt > 0) {
      const d = Math.min(h, simDt);
      state.t += d;
      step(d);
      simDt -= d;
    }
  }
  render();
  requestAnimationFrame(loop);
}

function init() {
  field = document.getElementById("field");
  raster = document.getElementById("raster");
  field.width = FIELD_W; field.height = FIELD_H;
  raster.width = FIELD_W; raster.height = RASTER_H;
  fctx = field.getContext("2d");
  rctx = raster.getContext("2d");

  // 初期配置: 円周上に 6 体 + 環境音 1
  const cx = FIELD_W / 2, cy = FIELD_H / 2, R = 150;
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * 2 * Math.PI;
    state.frogs.push(makeFrog(cx + R * Math.cos(a), cy + R * Math.sin(a)));
  }
  state.noises.push(makeNoise(cx, cy - 40));
  state.noises[0].level = 0.0;   // 最初は無音(邪魔しない)

  field.addEventListener("pointerdown", onDown);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);

  setupControls();
  refreshInspector();
  requestAnimationFrame(loop);
}

window.addEventListener("DOMContentLoaded", init);
