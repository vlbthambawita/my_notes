// Interactive widgets for the "Tsetlin Machine Autoencoders and Word Embeddings" lesson.
// Widgets, each mounted on a <div> with a matching id:
//   #viz-sample – build one training example: coin flip, pick u notes, OR them, mask the target word
//   #viz-mask   – hide one word of a note and let the trained autoencoder's rules vote on it
//   #viz-embed  – train the autoencoder in your browser and explore the word embedding it produces
// All randomness is seeded (mulberry32) and uses only + − × ÷ and comparisons, so the numbers match the
// Python check script exactly. Runs after DOMContentLoaded so window.Viz (assets/js/viz-common.js) exists.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, el, svg, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

  // ---------------------------------------------------------------- Small styles
  const style = document.createElement("style");
  style.textContent = `
    .tae-chips { display: flex; flex-wrap: wrap; gap: 5px; margin: 6px 0; }
    .tae-chip { font-size: 0.68rem; line-height: 1.2; padding: 3px 8px; border-radius: 999px; border: 1.5px solid var(--viz-border); color: var(--viz-muted); background: var(--viz-bg); white-space: nowrap; }
    .tae-chip.on { border-color: var(--viz-data); background: color-mix(in srgb, var(--viz-data) 18%, var(--viz-bg)); color: var(--viz-text); }
    .tae-chip.mask { border-style: dashed; border-color: var(--viz-model); color: var(--viz-model); font-weight: 700; }
    button.tae-chip { cursor: pointer; font-family: inherit; }
    button.tae-chip:disabled { cursor: default; }
    .tae-row { font-size: 0.72rem; color: var(--viz-muted); margin-top: 4px; }
    .tae-row b { color: var(--viz-text); }
    .tae-box { background: var(--viz-bg); border: 1px solid var(--viz-border); border-radius: 6px; padding: 8px 10px; margin-top: 8px; }
    .tae-box h4 { margin: 0 0 4px; font-size: 0.75rem; }
    .tae-cols { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
    .tae-col { flex: 1 1 250px; min-width: 0; background: var(--viz-bg); border: 1px solid var(--viz-border); border-radius: 6px; padding: 8px 10px; }
    .tae-col h4 { margin: 0 0 6px; font-size: 0.75rem; }
    .tae-rule { display: flex; gap: 8px; align-items: flex-start; font-size: 0.72rem; line-height: 1.35; padding: 3px 4px; border-radius: 4px; color: var(--viz-muted); }
    .tae-rule.on { color: var(--viz-text); background: color-mix(in srgb, var(--role) 14%, transparent); }
    .tae-dot { flex: 0 0 auto; width: 10px; height: 10px; margin-top: 3px; border-radius: 50%; border: 2px solid var(--role); }
    .tae-rule.on .tae-dot { background: var(--role); }
    .tae-w { flex: 0 0 auto; font-weight: 700; color: var(--viz-loss); min-width: 2.2em; text-align: right; }
    .tae-txt { flex: 1 1 auto; min-width: 0; overflow-wrap: anywhere; }
    .tae-txt s { color: var(--viz-model); }
    .tae-sub { color: var(--viz-muted); font-size: 0.68rem; }
    .tae-bar { display: inline-block; height: 8px; border-radius: 3px; background: var(--viz-loss); vertical-align: middle; margin-right: 6px; }
    .viz select.tae-sel, .viz input.tae-sel { font: inherit; padding: 2px 6px; border-radius: 6px; border: 1px solid var(--viz-border); background: var(--viz-bg); color: var(--viz-text); }
  `;
  document.head.appendChild(style);

  // ---- core start (mirrors tmae.py; keep the order of random draws identical)
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), a | 1);
      t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const WORDS = ["fever", "cough", "sputum", "chest pain", "breathless", "wheeze",
    "sneezing", "runny nose", "itchy eyes", "nausea", "vomiting", "diarrhoea",
    "stomach ache", "thirst", "frequent urination", "weight loss", "fatigue", "headache"];
  const WI = Object.fromEntries(WORDS.map((w, i) => [w, i]));
  const CONDITIONS = {
    "pneumonia": { "fever": 0.9, "cough": 0.9, "sputum": 0.7, "chest pain": 0.5, "breathless": 0.5, "fatigue": 0.5 },
    "asthma": { "wheeze": 0.9, "breathless": 0.8, "cough": 0.6 },
    "hay fever": { "sneezing": 0.9, "runny nose": 0.9, "itchy eyes": 0.7 },
    "gastroenteritis": { "nausea": 0.8, "vomiting": 0.7, "diarrhoea": 0.8, "stomach ache": 0.7, "fever": 0.3 },
    "diabetes": { "thirst": 0.8, "frequent urination": 0.8, "weight loss": 0.5, "fatigue": 0.7 },
    "migraine": { "headache": 0.95, "nausea": 0.5 },
  };
  const CONDS = Object.keys(CONDITIONS);
  const NOISE = 0.03;
  const M = WORDS.length;

  function makeNotes(n, seed) {
    const rand = mulberry32(seed), notes = [], labels = [];
    while (notes.length < n) {
      const c1 = Math.floor(rand() * CONDS.length), cs = [c1];
      if (rand() < 0.2) {
        const c2 = Math.floor(rand() * (CONDS.length - 1));
        cs.push(c2 < c1 ? c2 : c2 + 1);
      }
      const x = WORDS.map((w) => {
        let p = NOISE;
        for (const c of cs) p = Math.max(p, CONDITIONS[CONDS[c]][w] || 0);
        return rand() < p ? 1 : 0;
      });
      if (x.reduce((a, b) => a + b, 0) === 0) continue;
      notes.push(x);
      labels.push(cs);
    }
    return { notes, labels };
  }

  function indexNotes(notes) {
    const withK = [], withoutK = [];
    for (let k = 0; k < M; k++) {
      withK.push([]); withoutK.push([]);
      notes.forEach((x, d) => (x[k] ? withK[k] : withoutK[k]).push(d));
    }
    return { withK, withoutK };
  }

  // One training example for target word k: coin flip q, then OR together u notes with (q = 1) or without (q = 0) word k
  function drawExample(rand, notes, idx, k, u) {
    const q = rand() < 0.5 ? 1 : 0;
    const pool = q ? idx.withK[k] : idx.withoutK[k];
    const x = new Array(M).fill(0), picked = [];
    for (let a = 0; a < u; a++) {
      const d = pool[Math.floor(rand() * pool.length)];
      picked.push(d);
      const note = notes[d];
      for (let i = 0; i < M; i++) if (note[i]) x[i] = 1;
    }
    return { q, x, picked };
  }

  // The autoencoder: one shared pool of rules (memory per word), one signed integer weight per (word, rule)
  class TMAE {
    constructor(nClauses, T, s, depth, u, seed) {
      this.n = nClauses; this.T = T; this.s = s; this.depth = depth; this.half = Math.floor(depth / 2); this.u = u;
      this.forget = 1 / s;
      this.rand = mulberry32(seed);
      this.mem = Array.from({ length: nClauses }, () => new Array(M).fill(this.half));
      this.W = Array.from({ length: M }, () => Array.from({ length: nClauses }, () => (this.rand() < 0.5 ? 1 : -1)));
      this.active = new Array(nClauses).fill(1);
    }
    fires(j, x, k, training) {
      const r = this.mem[j];
      let empty = true;
      for (let i = 0; i < M; i++) {
        if (r[i] > this.half) {
          empty = false;
          if (i !== k && !x[i]) return 0;               // the masked word k counts as True
        }
      }
      return training || !empty ? 1 : 0;              // empty rules are True while learning, silent when predicting
    }
    update(k, x, q) {
      const { T, half, depth } = this, act = this.active, w = this.W[k];
      const C = [];
      for (let j = 0; j < this.n; j++) C.push(act[j] ? this.fires(j, x, k, true) : 0);
      let v = 0;
      for (let j = 0; j < this.n; j++) v += w[j] * C[j];
      v = Math.max(-T, Math.min(T, v));
      const p = q ? (T - v) / (2 * T) : (T + v) / (2 * T);
      for (let j = 0; j < this.n; j++) {
        if (!act[j] || this.rand() > p) continue;
        const r = this.mem[j];
        const recog = (w[j] >= 0) === (q === 1);
        if (recog) {
          if (C[j]) {                                 // Recognize Feedback
            for (let i = 0; i < M; i++) {
              if (i === k) continue;
              if (x[i]) { if (r[i] < depth) r[i]++; }
              else if (this.rand() <= this.forget && r[i] > 1) r[i]--;
            }
          } else {                                    // Erase Feedback
            for (let i = 0; i < M; i++) {
              if (i === k) continue;
              if (this.rand() <= this.forget && r[i] > 1) r[i]--;
            }
          }
        } else if (C[j]) {                            // Reject Feedback
          for (let i = 0; i < M; i++) if (i !== k && !x[i] && r[i] <= half) r[i]++;
        }
        if (C[j]) w[j] += q ? 1 : -1;                 // weight moves towards the target value
      }
    }
    round(notes, idx) {
      for (let j = 0; j < this.n; j++) {               // TMU: rules whose weights are already large are rarely updated
        let a = 0;
        for (let k = 0; k < M; k++) a += Math.abs(this.W[k][j]);
        a = a / M;
        this.active[j] = this.rand() <= (this.T - Math.min(a, this.T)) / this.T ? 1 : 0;
      }
      const order = WORDS.map((_, i) => i);
      for (let i = M - 1; i > 0; i--) {
        const jj = Math.floor(this.rand() * (i + 1));
        [order[i], order[jj]] = [order[jj], order[i]];
      }
      for (const k of order) {
        const ex = drawExample(this.rand, notes, idx, k, this.u);
        this.update(k, ex.x, ex.q);
      }
    }
    rule(j) { const out = []; for (let i = 0; i < M; i++) if (this.mem[j][i] > this.half) out.push(i); return out; }
    predict(x, k) {
      let v = 0;
      for (let j = 0; j < this.n; j++) v += this.W[k][j] * this.fires(j, x, k, false);
      return { v, y: v >= 0 ? 1 : 0 };
    }
    embedding() { return this.W.map((row) => row.map((w) => Math.max(0, Math.min(this.T, w)))); }
  }

  function cosine(a, b) {
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < a.length; i++) { num += a[i] * b[i]; da += a[i] * a[i]; db += b[i] * b[i]; }
    return da && db ? num / (Math.sqrt(da) * Math.sqrt(db)) : 0;
  }
  // ---- core end

  const DEFAULTS = { n: 24, T: 24, s: 4, depth: 100, u: 2, rounds: 300, seed: 7, dataSeed: 1, nTrain: 600, nTest: 300 };
  let DATA = null;
  function data() {
    if (DATA) return DATA;
    const { notes, labels } = makeNotes(DEFAULTS.nTrain + DEFAULTS.nTest, DEFAULTS.dataSeed);
    const train = notes.slice(0, DEFAULTS.nTrain), test = notes.slice(DEFAULTS.nTrain);
    DATA = { train, test, labels, idx: indexNotes(train) };
    return DATA;
  }
  let defaultModel = null;
  function getDefaultModel() {
    if (defaultModel) return defaultModel;
    const d = data();
    const tm = new TMAE(DEFAULTS.n, DEFAULTS.T, DEFAULTS.s, DEFAULTS.depth, DEFAULTS.u, DEFAULTS.seed);
    for (let r = 0; r < DEFAULTS.rounds; r++) tm.round(d.train, d.idx);
    defaultModel = tm;
    return tm;
  }
  const groupsOf = (i) => CONDS.filter((c) => WORDS[i] in CONDITIONS[c]);

  // ---------------------------------------------------------------- Shared helpers
  function control(parent, html) {
    const div = document.createElement("div");
    div.className = "viz-controls";
    div.innerHTML = html;
    parent.appendChild(div);
    return div;
  }
  function readout(parent) {
    const div = document.createElement("div");
    div.className = "viz-readout";
    div.style.whiteSpace = "pre-line";
    parent.appendChild(div);
    return div;
  }
  function hint(parent, text) {
    const div = document.createElement("div");
    div.className = "viz-hint";
    div.innerHTML = text;
    parent.appendChild(div);
    return div;
  }
  function box(parent) {
    const div = document.createElement("div");
    parent.appendChild(div);
    return div;
  }
  const wordOptions = (sel) => WORDS.map((w) => `<option value="${WI[w]}"${w === sel ? " selected" : ""}>${w}</option>`).join("");
  const chip = (text, cls) => `<span class="tae-chip${cls ? " " + cls : ""}">${text}</span>`;
  const signed = (v) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v);
  // A rule as text, with the masked target word struck through
  function ruleText(lits, k) {
    if (!lits.length) return "(empty rule)";
    return lits.map((i) => (i === k ? `<s title="masked: counts as True">${WORDS[i]}</s>` : WORDS[i])).join(" AND ");
  }

  // ---------------------------------------------------------------- Widget 1: build one training example
  function sampleWidget(root) {
    const ctrl = control(root, `
      <label>target word <select class="tae-sel" name="k">${wordOptions("cough")}</select></label>
      <label>notes per example <b>u</b> <input type="range" name="u" min="1" max="5" step="1" value="2"> <span class="val"></span></label>
      <button data-a="draw">🎲 Draw an example</button><button data-a="reset">Reset</button>`);
    const out = box(root);
    const read = readout(root);
    hint(root, "Blue = the word is in the note. The target word is <b>masked</b> (dashed): the rules never see it, and they are asked to guess whether it was there. The draws are seeded, so Reset replays the same sequence.");
    const kSel = ctrl.querySelector("[name=k]"), uIn = ctrl.querySelector("[name=u]");
    let rand, count, last;

    function reset() { rand = mulberry32(2024); count = 0; last = null; draw(); }
    function drawOne() {
      const d = data(), k = +kSel.value, u = +uIn.value;
      last = { k, u, ...drawExample(rand, d.train, d.idx, k, u) };
      count++;
      draw();
    }
    function avgContext(k, u) {                       // average context words over 2,000 draws (own seed)
      const d = data(), r = mulberry32(99);
      let s1 = 0, n1 = 0, s0 = 0, n0 = 0;
      for (let t = 0; t < 2000; t++) {
        const ex = drawExample(r, d.train, d.idx, k, u);
        const c = ex.x.reduce((a, b) => a + b, 0) - ex.x[k];
        if (ex.q) { s1 += c; n1++; } else { s0 += c; n0++; }
      }
      return [s1 / n1, s0 / n0];
    }
    function draw() {
      ctrl.querySelector(".val").textContent = uIn.value;
      const d = data(), k = +kSel.value;
      const freq = d.idx.withK[k].length;
      if (!last) {
        out.innerHTML = `<div class="tae-box"><div class="tae-row">Press <b>Draw an example</b>. Word <b>${WORDS[k]}</b> appears in ${freq} of the ${d.train.length} training notes (${fmt((100 * freq) / d.train.length, 1)}%), but the coin flip makes it present in half of the examples.</div></div>`;
        read.textContent = " ";
        return;
      }
      const L = last;
      let html = `<div class="tae-box"><h4>1 · Coin flip for <span style="color: var(--viz-model)">${WORDS[L.k]}</span>: ${L.q ? "present (q = 1)" : "absent (q = 0)"}</h4>`;
      html += `<div class="tae-row">so pick ${L.u} note${L.u > 1 ? "s" : ""} at random from the ${L.q ? d.idx.withK[L.k].length : d.idx.withoutK[L.k].length} notes that ${L.q ? "mention" : "do not mention"} <b>${WORDS[L.k]}</b>:</div>`;
      html += `<h4 style="margin-top: 8px">2 · The picked notes</h4>`;
      L.picked.forEach((n) => {
        html += `<div class="tae-row">note #${n + 1}</div><div class="tae-chips">${WORDS.map((w, i) => (d.train[n][i] ? chip(w, i === L.k ? "on mask" : "on") : "")).join("")}</div>`;
      });
      html += `<h4 style="margin-top: 8px">3 · OR them together and mask the target</h4><div class="tae-chips">${WORDS.map((w, i) => (i === L.k ? chip(w + " = ?", "mask") : chip(w, L.x[i] ? "on" : ""))).join("")}</div>`;
      html += `<div class="tae-row">The rules see the ${L.x.reduce((a, b) => a + b, 0) - L.x[L.k]} blue context words and must predict <b>${WORDS[L.k]} = ${L.q}</b>.</div></div>`;
      out.innerHTML = html;
      const [a1, a0] = avgContext(L.k, L.u);
      read.textContent = `example ${count} · target ${WORDS[L.k]} · q = ${L.q} · u = ${L.u} · context words: ${L.x.reduce((a, b) => a + b, 0) - L.x[L.k]}\n` +
        `averaged over 2,000 draws with u = ${L.u}: ${fmt(a1, 2)} context words when ${WORDS[L.k]} is present, ${fmt(a0, 2)} when it is absent`;
    }
    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "draw") drawOne();
      if (a === "reset") reset();
    });
    kSel.addEventListener("change", () => { last = null; draw(); });
    uIn.addEventListener("input", () => { ctrl.querySelector(".val").textContent = uIn.value; });
    uIn.addEventListener("change", () => { last = null; draw(); });
    reset();
  }

  // ---------------------------------------------------------------- Widget 2: predict a masked word
  function maskWidget(root) {
    const ctrl = control(root, `
      <label>hide the word <select class="tae-sel" name="k">${wordOptions("cough")}</select></label>
      <button data-p="fever,sputum,chest pain">fever + sputum + chest pain</button>
      <button data-p="breathless,wheeze">breathless + wheeze</button>
      <button data-p="sneezing,runny nose">sneezing + runny nose</button>
      <button data-p="test">next test note ▶</button>`);
    const chipsDiv = document.createElement("div");
    chipsDiv.className = "tae-chips";
    root.appendChild(chipsDiv);
    const cols = document.createElement("div");
    cols.className = "tae-cols";
    root.appendChild(cols);
    const out = readout(root);
    hint(root, "Click words to put them in the note or take them out. A <s style=\"color: var(--viz-model)\">struck-through</s> word in a rule is the hidden word itself: it counts as True, so the rule effectively uses only its other words. Filled dot = the rule fires on this note and adds its weight to v. Weights between −3 and +3 are summed in but not listed.");
    const kSel = ctrl.querySelector("[name=k]");
    let note = new Array(M).fill(0), truth = null, testPos = -1;

    function setNote(words) { note = WORDS.map((w) => (words.includes(w) ? 1 : 0)); truth = null; draw(); }
    function draw() {
      const tm = getDefaultModel(), k = +kSel.value;
      chipsDiv.innerHTML = WORDS.map((w, i) => (i === k ? `<button class="tae-chip mask" disabled>${w} = ?</button>` : `<button class="tae-chip${note[i] ? " on" : ""}" data-i="${i}">${w}</button>`)).join("");
      const rows = [];
      for (let j = 0; j < tm.n; j++) rows.push({ j, w: tm.W[k][j], on: tm.fires(j, note, k, false), lits: tm.rule(j) });
      const col = (title, list, role) => {
        let html = `<h4 style="color: var(--viz-${role})">${title}</h4>`;
        if (!list.length) html += `<div class="tae-rule">(no rule with a weight this large)</div>`;
        for (const r of list) {
          html += `<div class="tae-rule${r.on ? " on" : ""}" style="--role: var(--viz-${role})"><span class="tae-dot"></span><span class="tae-w">${signed(r.w)}</span><span class="tae-txt">R${r.j + 1}: IF ${ruleText(r.lits, k)}</span></div>`;
        }
        return `<div class="tae-col">${html}</div>`;
      };
      const pos = rows.filter((r) => r.w >= 4).sort((a, b) => b.w - a.w);
      const neg = rows.filter((r) => r.w <= -4).sort((a, b) => a.w - b.w);
      cols.innerHTML = col(`Rules voting “${WORDS[k]} is there”`, pos, "opt") + col(`Rules voting “${WORDS[k]} is not there”`, neg, "grad");
      const { v, y } = tm.predict(note, k);
      const fired = rows.filter((r) => r.on && r.w !== 0);
      let text = `note: ${WORDS.filter((_, i) => note[i] && i !== k).join(", ") || "(no other words)"}\n`;
      text += fired.length ? `firing rules: ${fired.map((r) => `R${r.j + 1} (${signed(r.w)})`).join(" + ")}\n` : "no rule fires\n";
      text += `v = ${signed(v) || "0"} → predicts ${WORDS[k]} ${y ? "PRESENT" : "ABSENT"}${v === 0 ? " (a tie counts as present, as in the paper: x̂ = 1 when v ≥ 0)" : ""}`;
      if (truth !== null) text += ` · in the note it was ${truth ? "present" : "absent"} ${truth === y ? "✓" : "✗"}`;
      out.textContent = text;
    }
    chipsDiv.addEventListener("click", (e) => {
      const i = e.target.dataset.i;
      if (i === undefined) return;
      note[+i] = 1 - note[+i]; truth = null; draw();
    });
    ctrl.addEventListener("click", (e) => {
      const p = e.target.dataset.p;
      if (p === undefined) return;
      if (p === "test") {
        const { test } = data(), k = +kSel.value;
        testPos = (testPos + 1) % test.length;
        note = test[testPos].slice();
        truth = note[k];
        draw();
        out.textContent = `test note ${testPos + 1} of ${test.length}\n` + out.textContent;
        return;
      }
      setNote(p.split(","));
    });
    kSel.addEventListener("change", () => { truth = null; draw(); });
    onThemeChange(draw);
    setNote(["fever", "sputum", "chest pain"]);
  }

  // ---------------------------------------------------------------- Widget 3: train and explore the embedding
  function embedWidget(root) {
    const small = root.clientWidth < 520;
    const ctrl = control(root, `
      <label>notes per example <b>u</b> <input type="range" name="u" min="1" max="5" step="1" value="${DEFAULTS.u}"> <span class="val"></span></label>
      <label>Voting Margin <b>T</b> <input type="range" name="T" min="4" max="64" step="4" value="${DEFAULTS.T}"> <span class="val"></span></label>
      <label>specificity <b>s</b> <input type="range" name="s" min="2" max="10" step="0.5" value="${DEFAULTS.s}"> <span class="val"></span></label>
      <label>seed <input class="tae-sel" type="number" name="seed" value="${DEFAULTS.seed}" min="1" max="999" style="width: 4.5em"></label>
      <label>similarity from <select class="tae-sel" name="src"><option value="tm" selected>TM embedding (rule weights)</option><option value="ppmi">co-occurrence counts (PPMI)</option></select></label>
      <button data-a="defaults">Defaults</button>`);
    const W = small ? 340 : 600, LBL = small ? 96 : 128, TOP = small ? 10 : 92;
    const cell = (W - LBL - (small ? 6 : 52)) / M, H = TOP + cell * M + 8;
    const short = (w) => (small && w === "frequent urination" ? "freq. urination" : w);
    const chart = svg(W, H);
    chart.style.cursor = "pointer";
    root.appendChild(chart);
    const out = readout(root);
    const detail = box(root);
    hint(root, `Each cell is the cosine similarity of two words' embeddings: dark purple = similar. Lines separate the word groups: breathing, hay fever, stomach, diabetes, then fatigue and headache. Click a row (a word) to see its nearest neighbours and the rules its embedding is made of. ${DEFAULTS.n} rules, memory depth ${DEFAULTS.depth}, ${DEFAULTS.rounds} rounds over ${DEFAULTS.nTrain} generated notes.`);
    const val = (name) => +ctrl.querySelector(`[name=${name}]`).value;
    let tm = null, status = "", sel = WI["cough"], run = null;

    // PPMI baseline from the same training notes
    const ppmi = (() => {
      const { train } = data(), N = train.length;
      const f = WORDS.map((_, i) => train.reduce((a, x) => a + x[i], 0));
      return WORDS.map((_, a) => WORDS.map((_, b) => {
        if (a === b) return 0;
        const co = train.reduce((s, x) => s + x[a] * x[b], 0);
        return co ? Math.max(0, Math.log((co * N) / (f[a] * f[b]))) : 0;
      }));
    })();

    function start() {
      if (run) run.cancel = true;
      const cfg = { u: val("u"), T: val("T"), s: val("s"), seed: Math.max(1, Math.round(val("seed")) || 1) };
      const d = data();
      const model = new TMAE(DEFAULTS.n, cfg.T, cfg.s, DEFAULTS.depth, cfg.u, cfg.seed);
      const me = { cancel: false };
      run = me; tm = null;
      let r = 0;
      (function next() {
        if (me.cancel) return;
        const end = Math.min(DEFAULTS.rounds, r + 50);
        for (; r < end; r++) model.round(d.train, d.idx);
        status = r < DEFAULTS.rounds ? `training… round ${r} / ${DEFAULTS.rounds}` : "";
        if (r === DEFAULTS.rounds) tm = model;
        draw();
        if (r < DEFAULTS.rounds) setTimeout(next, 0);
      })();
    }

    function vectors() { return ctrl.querySelector("[name=src]").value === "ppmi" ? ppmi : tm.embedding(); }

    function draw() {
      const c = colors();
      ctrl.querySelectorAll(".val").forEach((s) => { s.textContent = s.parentElement.querySelector("input").value; });
      chart.innerHTML = "";
      if (!tm) {
        el("text", { x: W / 2, y: H / 2, "text-anchor": "middle", style: textStyle }, chart).textContent = status || "scroll here to train";
        out.textContent = status || " ";
        detail.innerHTML = "";
        return;
      }
      const V = vectors();
      const S = V.map((a) => V.map((b) => cosine(a, b)));
      // group colour bands (by first listed condition of each word)
      for (let a = 0; a < M; a++) {
        const y = TOP + a * cell;
        el("text", { x: LBL - 6, y: y + cell / 2 + 3.5, "text-anchor": "end", style: (a === sel ? textStyle + "; font-weight: 700" : "") + "; font-size: " + (small ? 9.5 : 11) + "px" }, chart).textContent = short(WORDS[a]);
        if (!small) {
          const x = LBL + a * cell + cell / 2;
          el("text", { x: 0, y: 0, transform: `translate(${x + 3.5},${TOP - 6}) rotate(-60)`, style: (a === sel ? textStyle + "; font-weight: 700" : "") + "; font-size: 10.5px" }, chart).textContent = WORDS[a];
        }
        for (let b = 0; b < M; b++) {
          const s = a === b ? 1 : Math.max(0, S[a][b]);
          el("rect", { x: LBL + b * cell + 0.5, y: y + 0.5, width: cell - 1, height: cell - 1, fill: a === b ? c.grid : c.loss, "fill-opacity": a === b ? 1 : 0.06 + 0.94 * s, rx: 1.5 }, chart);
        }
      }
      // group separators
      const bounds = [6, 9, 13, 16, 17];   // breathing · hay fever · stomach · diabetes · fatigue · headache
      for (const bnd of bounds) {
        el("line", { x1: LBL, x2: LBL + M * cell, y1: TOP + bnd * cell, y2: TOP + bnd * cell, stroke: c.text, "stroke-width": 0.8, "stroke-opacity": 0.5 }, chart);
        el("line", { x1: LBL + bnd * cell, x2: LBL + bnd * cell, y1: TOP, y2: TOP + M * cell, stroke: c.text, "stroke-width": 0.8, "stroke-opacity": 0.5 }, chart);
      }
      el("rect", { x: 2, y: TOP + sel * cell, width: LBL + M * cell - 2, height: cell, fill: "none", stroke: c.model, "stroke-width": 2, rx: 3 }, chart);

      // nearest-neighbour score
      let good = 0;
      const nn = WORDS.map((_, a) => {
        const order = WORDS.map((_, b) => b).filter((b) => b !== a).sort((p, q) => S[a][q] - S[a][p] || p - q);
        if (groupsOf(a).some((g) => groupsOf(order[0]).includes(g))) good++;
        return order;
      });
      const src = ctrl.querySelector("[name=src]").value === "ppmi" ? "co-occurrence (PPMI)" : "TM embedding";
      out.textContent = `${src}: for ${good} of ${M} words, the nearest neighbour comes from the same condition\n` +
        `${WORDS[sel]} → ${nn[sel].slice(0, 3).map((b) => `${WORDS[b]} ${fmt(S[sel][b], 2)}`).join(" · ")}`;

      // the selected word's embedding: rules with positive weight
      const E = tm.embedding()[sel];
      const list = E.map((w, j) => ({ j, w })).filter((r) => r.w > 0).sort((a, b) => b.w - a.w);
      const maxW = Math.max(1, ...list.map((r) => r.w));
      let html = `<div class="tae-box"><h4>Embedding of <span style="color: var(--viz-model)">${WORDS[sel]}</span>: ${list.length} of ${tm.n} rules have a positive weight</h4>`;
      for (const r of list) {
        html += `<div class="tae-rule on" style="--role: var(--viz-loss)"><span class="tae-w">${r.w}</span><span class="tae-txt"><span class="tae-bar" style="width: ${Math.round((60 * r.w) / maxW)}px"></span>R${r.j + 1}: ${ruleText(tm.rule(r.j), sel)}</span></div>`;
      }
      if (!list.length) html += `<div class="tae-rule">(no rule votes for this word: its embedding is all zeros)</div>`;
      detail.innerHTML = html + "</div>";
    }

    chart.addEventListener("click", (e) => {
      if (!tm) return;
      const b = chart.getBoundingClientRect();
      const y = ((e.clientY - b.top) / b.height) * H;
      const a = Math.floor((y - TOP) / cell);
      if (a >= 0 && a < M) { sel = a; draw(); }
    });
    ctrl.querySelectorAll("input[type=range]").forEach((i) => {
      i.addEventListener("input", () => { i.parentElement.querySelector(".val").textContent = i.value; });
      i.addEventListener("change", start);
    });
    ctrl.querySelector("[name=seed]").addEventListener("change", start);
    ctrl.querySelector("[name=src]").addEventListener("change", draw);
    ctrl.addEventListener("click", (e) => {
      if (e.target.dataset.a === "defaults") {
        for (const k of ["u", "T", "s", "seed"]) ctrl.querySelector(`[name=${k}]`).value = DEFAULTS[k];
        ctrl.querySelector("[name=src]").value = "tm";
        start();
      }
    });
    onThemeChange(draw);
    draw();
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => {
        if (entries.some((en) => en.isIntersecting)) { io.disconnect(); start(); }
      }, { rootMargin: "200px" });
      io.observe(root);
    } else start();
  }

  const mounts = { "viz-sample": sampleWidget, "viz-mask": maskWidget, "viz-embed": embedWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
