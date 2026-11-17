// Interactive widgets for the "Regression with Tsetlin Machines" lesson.
// Widgets, each mounted on a <div> with a matching id:
//   #viz-score    – a points score as rules: every firing rule adds one vote, the vote sum is the score
//   #viz-stairs   – a trained regression TM on a dose curve: interval rules stack up to a staircase
//   #viz-feedback – how the error's sign picks the feedback and its size sets the probability
//   #viz-train    – train a regression TM on the dose curve in your browser
// All randomness is seeded (mulberry32) and uses only + − × ÷ and comparisons, so the numbers match the
// Python check script exactly. Runs after DOMContentLoaded so window.Viz (assets/js/viz-common.js) exists.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

  const style = document.createElement("style");
  style.textContent = `
    .rtm-rules { display: flex; flex-direction: column; gap: 2px; margin-top: 8px; background: var(--viz-bg); border: 1px solid var(--viz-border); border-radius: 6px; padding: 8px 10px; }
    .rtm-rules h4 { margin: 0 0 4px; font-size: 0.75rem; color: var(--viz-model); }
    .rtm-rule { display: flex; gap: 8px; align-items: flex-start; font-size: 0.72rem; line-height: 1.35; padding: 2px 4px; border-radius: 4px; color: var(--viz-muted); }
    .rtm-rule.on { color: var(--viz-text); background: color-mix(in srgb, var(--viz-model) 14%, transparent); }
    .rtm-dot { flex: 0 0 auto; width: 10px; height: 10px; margin-top: 3px; border-radius: 50%; border: 2px solid var(--viz-model); }
    .rtm-rule.on .rtm-dot { background: var(--viz-model); }
    .rtm-n { flex: 0 0 auto; font-weight: 700; color: var(--viz-loss); min-width: 2.2em; }
    .rtm-txt { flex: 1 1 auto; min-width: 0; overflow-wrap: anywhere; }
    .viz select.rtm-sel, .viz input.rtm-sel { font: inherit; padding: 2px 6px; border-radius: 6px; border: 1px solid var(--viz-border); background: var(--viz-bg); color: var(--viz-text); }
    .viz .rtm-checks { display: flex; flex-wrap: wrap; gap: 6px 14px; }
    .viz .rtm-checks label { gap: 4px; }
  `;
  document.head.appendChild(style);

  // ---------------------------------------------------------------- Seeded random numbers
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), a | 1);
      t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------------------------------------------------------------- The regression Tsetlin machine
  // memory[j][k] = position (1 … depth) of literal k in rule j; memorized if > depth / 2.
  // Output: ŷ = yMin + v / T × (yMax − yMin), with v = number of firing rules, clipped to [0, T].
  // Training: target in votes τ = (y − yMin) / (yMax − yMin) × T; error = v − τ.
  //   error < 0 (too low):  each rule picked with probability |error| / T; Recognize if it fires, Erase if not.
  //   error > 0 (too high): each rule picked with probability |error| / T; Reject if it fires.
  // Memorize Value 1, Forget Value 1/s (the book's convention from Ch. 2 on, and TMU's default).
  class RTM {
    constructor(nRules, T, s, depth, seed, nLits, yMin, yMax) {
      this.m = nRules; this.T = T; this.depth = depth; this.half = depth / 2; this.L = nLits;
      this.forget = 1 / s; this.yMin = yMin; this.yMax = yMax;
      this.rand = mulberry32(seed);
      this.memory = Array.from({ length: nRules }, () => new Int16Array(nLits).fill(depth / 2));
    }
    fires(j, lits, training) {
      const rule = this.memory[j];
      let empty = true;
      for (let k = 0; k < this.L; k++) {
        if (rule[k] > this.half) { empty = false; if (!lits[k]) return 0; }
      }
      return training || !empty ? 1 : 0;           // an empty rule is True while learning, silent when predicting
    }
    votes(lits, training) {
      const f = [];
      let v = 0;
      for (let j = 0; j < this.m; j++) { f.push(this.fires(j, lits, training)); v += f[j]; }
      return { v: Math.min(v, this.T), f };
    }
    toVotes(y) { return (y - this.yMin) / (this.yMax - this.yMin) * this.T; }
    fromVotes(v) { return this.yMin + v / this.T * (this.yMax - this.yMin); }
    predict(lits) { return this.fromVotes(this.votes(lits, false).v); }
    update(lits, y) {
      const { v, f } = this.votes(lits, true);
      const err = v - this.toVotes(y);
      if (err === 0) return;
      let p = Math.abs(err) / this.T;               // the error's size sets the feedback probability
      if (p > 1) p = 1;
      const top = this.depth, half = this.half;
      for (let j = 0; j < this.m; j++) {
        if (this.rand() > p) continue;
        const rule = this.memory[j];
        if (err < 0) {                              // prediction too low: make more rules fire
          if (f[j]) {                               // Recognize Feedback
            for (let k = 0; k < this.L; k++) {
              if (lits[k]) { if (rule[k] < top) rule[k]++; }
              else if (this.rand() <= this.forget && rule[k] > 1) rule[k]--;
            }
          } else {                                  // Erase Feedback
            for (let k = 0; k < this.L; k++) if (this.rand() <= this.forget && rule[k] > 1) rule[k]--;
          }
        } else if (f[j]) {                          // prediction too high: Reject Feedback for firing rules
          for (let k = 0; k < this.L; k++) if (!lits[k] && rule[k] <= half) rule[k]++;
        }
      }
    }
    epoch(data) {
      const idx = data.map((_, i) => i);
      for (let i = idx.length - 1; i > 0; i--) {
        const j = Math.floor(this.rand() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      for (const i of idx) this.update(data[i].lits, data[i].y);
    }
    mae(data) {
      let s = 0;
      for (const d of data) s += Math.abs(this.predict(d.lits) - d.y);
      return s / data.length;
    }
    rule(j) {
      const out = [];
      for (let k = 0; k < this.L; k++) if (this.memory[j][k] > this.half) out.push(k);
      return out;
    }
  }

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
  const narrow = (root) => root.clientWidth < 520;
  const f2 = (v) => (Math.abs(v) < 0.005 ? "0.00" : v.toFixed(2));
  function whenVisible(root, fn) {
    if (!("IntersectionObserver" in window)) { fn(); return; }
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
    }, { rootMargin: "200px" });
    io.observe(root);
  }

  // ================================================================ Data 1: an invented dose–response curve
  const fTrue = (x) => 12 * x / (1.5 + x) - 0.09 * x * x;   // benefit rises, plateaus, then falls (invented)
  const THRESH = Array.from({ length: 19 }, (_, i) => 0.5 * (i + 1));   // "dose ≥ 0.5", …, "dose ≥ 9.5"
  const K = THRESH.length;
  function thermo(x) {
    const b = THRESH.map((t) => (x >= t ? 1 : 0));
    return b.concat(b.map((v) => 1 - v));          // literals: dose ≥ t …, then NOT dose ≥ t …
  }
  function makeCurve(seed, n, sd) {
    const r = mulberry32(seed), out = [];
    for (let i = 0; i < n; i++) {
      const x = 10 * r();
      let z = 0;
      for (let q = 0; q < 12; q++) z += r();
      z -= 6;                                       // roughly normal noise (sum of 12 uniforms)
      const y = fTrue(x) + sd * z;
      out.push({ x, y, lits: thermo(x) });
    }
    return out;
  }
  const CURVE = { train: makeCurve(1, 300, 0.6), test: makeCurve(2, 300, 0.6) };
  CURVE.yMin = Math.min(...CURVE.train.map((d) => d.y));
  CURVE.yMax = Math.max(...CURVE.train.map((d) => d.y));
  CURVE.floor = CURVE.test.reduce((s, d) => s + Math.abs(fTrue(d.x) - d.y), 0) / CURVE.test.length;
  const CDEF = { m: 40, T: 20, s: 2.5, depth: 100, epochs: 60, seed: 1 };

  // A thermometer rule → the dose interval [lo, hi) where it fires (null if it can never fire)
  function interval(lits) {
    let lo = 0, hi = 10;
    for (const k of lits) { if (k < K) lo = Math.max(lo, THRESH[k]); else hi = Math.min(hi, THRESH[k - K]); }
    return lo < hi ? [lo, hi] : null;
  }
  // Group a trained model's rules: distinct intervals with counts, plus empty / never-firing counts
  function intervals(tm) {
    const map = new Map();
    let empty = 0, never = 0;
    for (let j = 0; j < tm.m; j++) {
      const r = tm.rule(j);
      if (!r.length) { empty++; continue; }
      const I = interval(r);
      if (!I) { never++; continue; }
      const key = I.join("-");
      if (map.has(key)) map.get(key).n++; else map.set(key, { lo: I[0], hi: I[1], n: 1 });
    }
    const list = [...map.values()].sort((a, b) => a.lo - b.lo || a.hi - b.hi);
    return { list, empty, never };
  }
  const doseText = (I) => (I.lo === 0 ? `dose < ${I.hi}` : I.hi === 10 ? `dose ≥ ${I.lo}` : `${I.lo} ≤ dose < ${I.hi}`);

  let curveModel = null;
  function getCurveModel() {
    if (curveModel) return curveModel;
    const tm = new RTM(CDEF.m, CDEF.T, CDEF.s, CDEF.depth, CDEF.seed, 2 * K, CURVE.yMin, CURVE.yMax);
    for (let e = 0; e < CDEF.epochs; e++) tm.epoch(CURVE.train);
    curveModel = tm;
    return tm;
  }

  // Draw data, true curve and a model's staircase into an SVG area
  function drawCurvePanel(g, c, x, y, tm, opts) {
    for (const d of opts.points) el("circle", { cx: x(d.x), cy: y(d.y), r: 2, fill: c.data, "fill-opacity": 0.35 }, g);
    let d = "";
    for (let i = 0; i <= 200; i++) { const xv = i / 20; d += (i ? "L" : "M") + x(xv).toFixed(1) + "," + y(fTrue(xv)).toFixed(1); }
    el("path", { d, fill: "none", stroke: c.text, "stroke-width": 1.5, "stroke-dasharray": "5 4", "stroke-opacity": 0.6 }, g);
    if (!tm) return;
    // the staircase only changes at thresholds, so evaluate it once per 0.5-wide cell
    let p = "";
    for (let i = 0; i < 20; i++) {
      const a = i * 0.5, b = a + 0.5, yy = y(tm.predict(thermo(a + 0.25)));
      p += (i ? "L" : "M") + x(a).toFixed(1) + "," + yy.toFixed(1) + "L" + x(b).toFixed(1) + "," + yy.toFixed(1);
    }
    el("path", { d: p, fill: "none", stroke: c.model, "stroke-width": 2.5, "stroke-linejoin": "round" }, g);
  }

  // ================================================================ Data 2: a simulated points score
  // Modelled on CHA2DS2-VASc (stroke risk in atrial fibrillation): 1 point each for heart failure,
  // hypertension, diabetes, vascular disease, female sex and age 65–74; 2 points for prior stroke/TIA and age ≥ 75.
  const COND = ["heart failure", "hypertension", "diabetes", "prior stroke/TIA", "vascular disease", "female"];
  const PREV = [0.2, 0.6, 0.25, 0.15, 0.2, 0.45];
  const AGE_T = [50, 55, 60, 65, 70, 75, 80, 85];
  const NS = COND.length + AGE_T.length;            // 14 Boolean features, 28 literals
  const scoreOf = (b, age) => b[0] + b[1] + b[2] + 2 * b[3] + b[4] + b[5] + (age >= 75 ? 2 : age >= 65 ? 1 : 0);
  function scoreLits(b, age) {
    const x = b.concat(AGE_T.map((a) => (age >= a ? 1 : 0)));
    return x.concat(x.map((v) => 1 - v));
  }
  function patients(seed, n) {
    const r = mulberry32(seed), out = [];
    for (let i = 0; i < n; i++) {
      const b = PREV.map((p) => (r() < p ? 1 : 0));
      const age = 45 + 45 * r();
      out.push({ b, age, y: scoreOf(b, age), lits: scoreLits(b, age) });
    }
    return out;
  }
  // Plain-English rule over the score literals (age literals merged into one range)
  function scoreRuleText(lits) {
    if (!lits.length) return "(empty rule)";
    const parts = [];
    let lo = -1, hi = 1e9;
    for (const k of lits) {
      const f = k % NS, neg = k >= NS;
      if (f < COND.length) parts.push((neg ? "NOT " : "") + COND[f]);
      else if (neg) hi = Math.min(hi, AGE_T[f - COND.length]); else lo = Math.max(lo, AGE_T[f - COND.length]);
    }
    if (lo >= 0 && hi < 1e9) parts.push(lo < hi ? `age ${lo}–${hi - 1}` : "age (impossible range)");
    else if (lo >= 0) parts.push(`age ≥ ${lo}`);
    else if (hi < 1e9) parts.push(`age < ${hi}`);
    return parts.join(" AND ");
  }
  const scoreRawText = (lits) => lits.map((k) => (k >= NS ? "NOT " : "") + (k % NS < COND.length ? COND[k % NS] : `age ≥ ${AGE_T[k % NS - COND.length]}`)).join(" AND ");
  const lit = (name) => COND.indexOf(name);
  const ageLit = (a) => COND.length + AGE_T.indexOf(a);
  const PUBLISHED = [[lit("heart failure")], [lit("hypertension")], [lit("diabetes")], [lit("prior stroke/TIA")], [lit("prior stroke/TIA")],
    [lit("vascular disease")], [lit("female")], [ageLit(65)], [ageLit(75)]];
  const SDEF = { m: 9, T: 9, s: 2, depth: 100, epochs: 40, seed: 1 };
  let scoreModel = null;
  function getScoreModel() {
    if (scoreModel) return scoreModel;
    const train = patients(11, 600), test = patients(12, 600);
    const tm = new RTM(SDEF.m, SDEF.T, SDEF.s, SDEF.depth, SDEF.seed, 2 * NS, 0, 9);
    let exactAt = null;
    for (let e = 0; e < SDEF.epochs; e++) {
      tm.epoch(train);
      if (exactAt === null && tm.mae(test) === 0) exactAt = e + 1;
    }
    scoreModel = { tm, train, test, mae: tm.mae(test), exactAt };
    return scoreModel;
  }

  // ---------------------------------------------------------------- Widget 1: a points score as rules
  function scoreWidget(root) {
    const ctrl = control(root, `<label>rules <select class="rtm-sel" name="set">
        <option value="pub">published points table</option>
        <option value="learned">learned by a regression TM</option></select></label>`);
    const pick = control(root, `<div class="rtm-checks">${COND.map((c, i) => `<label><input type="checkbox" name="c${i}"> ${c}</label>`).join("")}</div>
      <label>age <input type="range" name="age" min="45" max="90" step="1" value="78"> <span class="val"></span></label>
      <button data-a="next">next simulated patient ▶</button>`);
    const list = document.createElement("div");
    list.className = "rtm-rules";
    root.appendChild(list);
    const small = narrow(root);
    const BW = small ? 340 : 560;
    const bar = svg(BW, 78);
    bar.style.marginTop = "10px";
    root.appendChild(bar);
    const out = readout(root);
    hint(root, "Filled dot = the rule's condition is True for this patient, so it adds one vote. Here T = 9 and the output range is 0–9 points, so each vote is worth exactly one point. Hover a learned rule to see its raw literals.");
    const setSel = ctrl.querySelector("[name=set]");
    const boxes = COND.map((_, i) => pick.querySelector(`[name=c${i}]`));
    const ageIn = pick.querySelector("[name=age]");
    let truth = null, pos = -1;
    boxes[1].checked = true; boxes[5].checked = true;

    function rules() {
      if (setSel.value === "pub") return PUBLISHED;
      const { tm } = getScoreModel();
      return Array.from({ length: tm.m }, (_, j) => tm.rule(j));
    }
    function draw() {
      const c = colors();
      const b = boxes.map((x) => (x.checked ? 1 : 0)), age = +ageIn.value;
      pick.querySelector(".val").textContent = age;
      const lits = scoreLits(b, age);
      const rs = rules();
      let v = 0, html = `<h4>${setSel.value === "pub" ? "Points table as rules (each +1 vote)" : "Rules learned from 600 simulated patients (each +1 vote)"}</h4>`;
      rs.forEach((r, j) => {
        const on = r.length > 0 && r.every((k) => lits[k] === 1);
        if (on) v++;
        html += `<div class="rtm-rule${on ? " on" : ""}" title="${scoreRawText(r) || "(empty)"}"><span class="rtm-dot"></span><span class="rtm-n">R${j + 1}</span><span class="rtm-txt">IF ${scoreRuleText(r)} THEN +1</span></div>`;
      });
      list.innerHTML = html;
      v = Math.min(v, 9);
      bar.innerHTML = "";
      const x = linear(0, 9, 24, BW - 24);
      for (let i = 0; i < 9; i++) {
        el("rect", { x: x(i) + 1.5, y: 22, width: x(1) - x(0) - 3, height: 20, rx: 3, fill: i < v ? c.loss : c.grid, "fill-opacity": i < v ? 0.85 : 1 }, bar);
      }
      for (let i = 0; i <= 9; i++) el("text", { x: x(i), y: 56, "text-anchor": "middle" }, bar).textContent = i;
      el("text", { x: 24, y: 14 }, bar).textContent = "votes v (out of T = 9)";
      el("text", { x: x(v), y: 72, "text-anchor": "middle", style: "fill: var(--viz-loss); font-weight: 700; font-size: 12px" }, bar).textContent = `v = ${v}`;
      const s = scoreOf(b, age);
      let text = `votes v = ${v} → output = 0 + ${v} / 9 × (9 − 0) = ${v} points · points-table score: ${s} ${v === s ? "✓" : "✗"}`;
      if (truth !== null) text = `simulated test patient ${pos + 1} of 600 · ` + text;
      if (setSel.value === "learned") {
        const M = getScoreModel();
        text += `\nlearned model on all 600 test patients: mean absolute error ${fmt(M.mae, 3)} (exact from epoch ${M.exactAt} of ${SDEF.epochs})`;
      }
      out.textContent = text;
    }
    setSel.addEventListener("change", draw);
    pick.addEventListener("input", () => { truth = null; draw(); });
    pick.addEventListener("click", (e) => {
      if (e.target.dataset.a !== "next") return;
      const { test } = getScoreModel();
      pos = (pos + 1) % test.length;
      const p = test[pos];
      p.b.forEach((v, i) => { boxes[i].checked = !!v; });
      ageIn.value = String(Math.floor(p.age));
      truth = p.y;
      draw();
    });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 2: interval rules stack up to a staircase
  function stairsWidget(root) {
    const small = narrow(root);
    const W = small ? 340 : 560, M = { l: 40, r: 14 };
    const chart = svg(W, 10);
    chart.style.cursor = "ew-resize";
    chart.style.touchAction = "none";
    root.appendChild(chart);
    const ctrl = control(root, `<label>dose <input type="range" name="x" min="0.05" max="9.95" step="0.1" value="3.65"> <span class="val"></span> mg</label>`);
    const out = readout(root);
    hint(root, `The ${CDEF.m} rules of a regression TM trained on 300 noisy points (Step 5's default settings: ${CDEF.m} rules, T = ${CDEF.T}, s = ${CDEF.s}, ${CDEF.epochs} epochs). Each bar is one rule's dose interval; "×2" means two rules learned the same interval. Drag on the chart or use the slider.`);
    const xIn = ctrl.querySelector("[name=x]");
    const x = linear(0, 10, M.l, W - M.r);
    let tm = null, groups = null;

    function draw() {
      const c = colors(), xv = +xIn.value;
      ctrl.querySelector(".val").textContent = xv.toFixed(2);
      chart.innerHTML = "";
      if (!tm) { chart.setAttribute("viewBox", `0 0 ${W} 60`); el("text", { x: W / 2, y: 34, "text-anchor": "middle", style: textStyle }, chart).textContent = "training the model…"; return; }
      const rowH = small ? 7 : 6.5, top = 24, n = groups.list.length;
      const barsH = n * rowH;
      const y0 = top + barsH + 30, plotH = small ? 190 : 200, H = y0 + plotH + 36;
      chart.setAttribute("viewBox", `0 0 ${W} ${H}`);
      el("text", { x: M.l, y: 14, style: textStyle + "; font-size: 12px" }, chart).textContent = "rules (dose intervals where each one fires)";
      let v = 0;
      groups.list.forEach((I, i) => {
        const on = xv >= I.lo && xv < I.hi;
        if (on) v += I.n;
        const yy = top + i * rowH;
        el("rect", { x: x(I.lo), y: yy + 0.8, width: x(I.hi) - x(I.lo), height: rowH - 1.6, rx: 1.5, fill: on ? c.model : c.muted, "fill-opacity": on ? 0.9 : 0.35 }, chart);
        if (I.n > 1) el("text", { x: x(I.hi) + 3, y: yy + rowH - 0.5, style: "font-size: 8px" }, chart).textContent = "×" + I.n;
      });
      // lower panel: data, true curve, staircase
      const y = linear(-1, 9, y0 + plotH, y0);
      const g = el("g", {}, chart);
      axes(g, x, y, { xDomain: [0, 10], yDomain: [-1, 9], xTicks: ticks(0, 10, small ? 2 : 1), yTicks: [0, 2, 4, 6, 8], xLabel: "dose (mg)" });
      el("text", { x: M.l, y: y0 - 8, style: textStyle + "; font-size: 12px" }, chart).textContent = small ? "data, true curve (dashed), TM output" : "benefit score: data, true curve (dashed), TM output (orange)";
      drawCurvePanel(g, c, x, y, tm, { points: CURVE.train });
      el("line", { x1: x(xv), x2: x(xv), y1: top - 4, y2: y0 + plotH, stroke: c.text, "stroke-width": 1 }, chart);
      v = Math.min(v, tm.T);
      const yhat = tm.fromVotes(v);
      el("circle", { cx: x(xv), cy: y(yhat), r: 5, fill: c.model, stroke: c.bg, "stroke-width": 1.5 }, chart);
      const silent = groups.empty + groups.never;
      out.textContent =
        `dose ${xv.toFixed(2)} mg: ${v} of the ${tm.m} rules fire → v = ${v} (clipped to at most T = ${tm.T})\n` +
        `output = ${f2(tm.yMin)} + ${v} / ${tm.T} × (${f2(tm.yMax)} − (${f2(tm.yMin)})) = ${f2(yhat)} · true curve ${f2(fTrue(xv))}` +
        (silent ? `\n${silent} rule(s) are empty or can never fire, so they never vote.` : "");
    }
    function dragTo(e) {
      const b = chart.getBoundingClientRect();
      const xv = x.invert(((e.clientX - b.left) / b.width) * W);
      xIn.value = Math.max(0.05, Math.min(9.95, Math.round(xv * 10) / 10 + 0.05)).toFixed(2);
      draw();
    }
    let dragging = false;
    chart.addEventListener("pointerdown", (e) => { dragging = true; chart.setPointerCapture(e.pointerId); dragTo(e); });
    chart.addEventListener("pointermove", (e) => { if (dragging) dragTo(e); });
    chart.addEventListener("pointerup", () => { dragging = false; });
    xIn.addEventListener("input", draw);
    onThemeChange(draw);
    draw();
    whenVisible(root, () => setTimeout(() => { tm = getCurveModel(); groups = intervals(tm); draw(); }, 0));
  }

  // ---------------------------------------------------------------- Widget 3: the feedback rule
  function feedbackWidget(root) {
    const small = narrow(root);
    const W = small ? 340 : 560, H = 290, M = { l: 44, r: 14, t: 30, b: 72 };
    const chart = svg(W, H);
    chart.style.cursor = "ew-resize";
    chart.style.touchAction = "none";
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>resolution <b>T</b> <input type="range" name="T" min="2" max="20" step="1" value="10"> <span class="val"></span></label>
      <label>target in votes <b>τ</b> <input type="range" name="tau" min="0" max="10" step="0.5" value="6.5"> <span class="val"></span></label>
      <label>vote sum <b>v</b> <input type="range" name="v" min="0" max="10" step="1" value="3"> <span class="val"></span></label>`);
    const out = readout(root);
    hint(root, "Drag on the chart or use the sliders. Each rule is picked for feedback independently with the probability shown. The dashed line is the squared version used by the TMU library.");
    const tIn = ctrl.querySelector("[name=T]"), tauIn = ctrl.querySelector("[name=tau]"), vIn = ctrl.querySelector("[name=v]");

    function draw() {
      const c = colors(), T = +tIn.value;
      tauIn.max = String(T); vIn.max = String(T);
      if (+tauIn.value > T) tauIn.value = String(T);
      if (+vIn.value > T) vIn.value = String(T);
      const tau = +tauIn.value, v = +vIn.value;
      ctrl.querySelectorAll(".val").forEach((s) => { s.textContent = s.parentElement.querySelector("input").value; });
      const x = linear(0, T, M.l, W - M.r), y = linear(0, 1, H - M.b, M.t);
      chart.innerHTML = "";
      el("rect", { x: x(0), y: M.t, width: x(tau) - x(0), height: y(0) - M.t, fill: c.opt, "fill-opacity": 0.08 }, chart);
      el("rect", { x: x(tau), y: M.t, width: x(T) - x(tau), height: y(0) - M.t, fill: c.grad, "fill-opacity": 0.08 }, chart);
      axes(chart, x, y, { xDomain: [0, T], yDomain: [0, 1], xTicks: ticks(0, T, T > 10 ? (small ? 4 : 2) : (small && T > 6 ? 2 : 1)), yTicks: [0, 0.25, 0.5, 0.75, 1], xLabel: "vote sum v (number of firing rules)" });
      el("text", { x: M.l, y: 16, style: textStyle + "; font-size: 12px" }, chart).textContent = "probability that a rule gets feedback";
      el("line", { x1: x(tau), x2: x(tau), y1: M.t, y2: y(0), stroke: c.loss, "stroke-width": 2, "stroke-dasharray": "4 3" }, chart);
      el("text", { x: x(tau), y: M.t - 5, "text-anchor": "middle", style: "fill: var(--viz-loss); font-weight: 700" }, chart).textContent = "τ";
      if (x(tau) - x(0) > 70) el("text", { x: (x(0) + x(tau)) / 2, y: M.t + 16, "text-anchor": "middle", style: "fill: var(--viz-opt); font-weight: 700" }, chart).textContent = "too low";
      if (x(T) - x(tau) > 70) el("text", { x: (x(tau) + x(T)) / 2, y: M.t + 16, "text-anchor": "middle", style: "fill: var(--viz-grad); font-weight: 700" }, chart).textContent = "too high";
      const seg = (a, b, f, col, dash) => {
        let d = "";
        for (let i = 0; i <= 40; i++) { const t = a + (b - a) * i / 40; d += (i ? "L" : "M") + x(t).toFixed(1) + "," + y(f(t)).toFixed(1); }
        el("path", { d, fill: "none", stroke: col, "stroke-width": dash ? 1.5 : 2.5, "stroke-dasharray": dash || "none" }, chart);
      };
      const lin = (t) => Math.min(1, Math.abs(t - tau) / T), sq = (t) => lin(t) * lin(t);
      seg(0, tau, lin, c.opt); seg(tau, T, lin, c.grad);
      seg(0, T, sq, c.muted, "6 4");
      el("line", { x1: x(v), x2: x(v), y1: M.t, y2: y(0), stroke: c.text, "stroke-width": 1 }, chart);
      el("circle", { cx: x(v), cy: y(lin(v)), r: 6, fill: v < tau ? c.opt : v > tau ? c.grad : c.muted, stroke: c.bg, "stroke-width": 1.5 }, chart);
      const lx = M.l, ly = H - 16;
      el("line", { x1: lx, x2: lx + 22, y1: ly, y2: ly, stroke: c.opt, "stroke-width": 2.5 }, chart);
      el("text", { x: lx + 26, y: ly + 4, style: textStyle }, chart).textContent = small ? "Recognize/Erase" : "too low: Recognize / Erase";
      const lx2 = small ? lx + 118 : lx + 180;
      el("line", { x1: lx2, x2: lx2 + 22, y1: ly, y2: ly, stroke: c.grad, "stroke-width": 2.5 }, chart);
      el("text", { x: lx2 + 26, y: ly + 4, style: textStyle }, chart).textContent = small ? "Reject" : "too high: Reject";
      const lx3 = small ? lx + 190 : lx + 330;
      el("line", { x1: lx3, x2: lx3 + 22, y1: ly, y2: ly, stroke: c.muted, "stroke-width": 1.5, "stroke-dasharray": "6 4" }, chart);
      el("text", { x: lx3 + 26, y: ly + 4, style: textStyle }, chart).textContent = small ? "squared" : "squared (TMU)";
      const e = v - tau, p = lin(v);
      let text;
      if (e === 0) text = `v = ${v} equals the target τ = ${tau}: no error, so no rule gets feedback.`;
      else {
        text = `v = ${v}, target τ = ${tau}: prediction too ${e < 0 ? "LOW" : "HIGH"} by ${Math.abs(e)} vote${Math.abs(e) === 1 ? "" : "s"} → each rule is picked with probability |v − τ| / T = ${Math.abs(e)} / ${T} = ${fmt(p, 3)} (squared: ${fmt(p * p, 3)})\n`;
        text += e < 0 ? "Goal: MORE rules should fire. Picked rules that fire get Recognize Feedback; picked rules that are silent get Erase Feedback."
          : "Goal: FEWER rules should fire. Picked rules that fire get Reject Feedback; silent rules are left alone.";
      }
      out.textContent = text;
    }
    function dragTo(e) {
      const b = chart.getBoundingClientRect(), T = +tIn.value;
      const x = linear(0, T, M.l, W - M.r);
      vIn.value = String(Math.max(0, Math.min(T, Math.round(x.invert(((e.clientX - b.left) / b.width) * W)))));
      draw();
    }
    let dragging = false;
    chart.addEventListener("pointerdown", (e) => { dragging = true; chart.setPointerCapture(e.pointerId); dragTo(e); });
    chart.addEventListener("pointermove", (e) => { if (dragging) dragTo(e); });
    chart.addEventListener("pointerup", () => { dragging = false; });
    ctrl.querySelectorAll("input").forEach((i) => i.addEventListener("input", draw));
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 4: train on the dose curve
  function trainWidget(root) {
    const small = narrow(root);
    const W = small ? 340 : 560, M = { l: 40, r: 14 };
    const H1 = 250, H2 = 150;
    const plots = document.createElement("div");
    root.appendChild(plots);
    const fitSvg = svg(W, H1), errSvg = svg(W, H2);
    errSvg.style.marginTop = "8px";
    plots.appendChild(fitSvg); plots.appendChild(errSvg);
    const ctrl = control(root, `
      <label>rules <input type="range" name="m" min="5" max="80" step="5" value="${CDEF.m}"> <span class="val"></span></label>
      <label>resolution <b>T</b> <input type="range" name="T" min="2" max="40" step="1" value="${CDEF.T}"> <span class="val"></span></label>
      <label>specificity <b>s</b> <input type="range" name="s" min="1.5" max="6" step="0.5" value="${CDEF.s}"> <span class="val"></span></label>
      <label>seed <input class="rtm-sel" type="number" name="seed" value="${CDEF.seed}" min="1" max="999" style="width: 4.5em"></label>
      <button data-a="train">Train again</button><button data-a="defaults">Defaults</button>`);
    const out = readout(root);
    hint(root, `300 training and 300 test points from the same invented curve, with noise. Dose is booleanized with 19 thresholds (dose ≥ 0.5, 1.0, …, 9.5 mg). ${CDEF.epochs} epochs, memory depth ${CDEF.depth}, Memorize Value 1, Forget Value 1/s. The seed sets the machine's random choices; the data stay the same.`);
    const val = (name) => +ctrl.querySelector(`[name=${name}]`).value;
    const x = linear(0, 10, M.l, W - M.r);
    let hist = [], tm = null, run = null, status = "", done = false;

    function start() {
      if (run) run.cancel = true;
      const cfg = { m: val("m"), T: val("T"), s: val("s"), seed: Math.max(1, Math.round(val("seed")) || 1) };
      tm = new RTM(cfg.m, cfg.T, cfg.s, CDEF.depth, cfg.seed, 2 * K, CURVE.yMin, CURVE.yMax);
      hist = []; done = false;
      const me = { cancel: false };
      run = me;
      let e = 0;
      (function next() {
        if (me.cancel) return;
        tm.epoch(CURVE.train);
        e++;
        hist.push([tm.mae(CURVE.train), tm.mae(CURVE.test)]);
        status = e < CDEF.epochs ? `training… epoch ${e} / ${CDEF.epochs}` : "";
        done = e === CDEF.epochs;
        draw();
        if (e < CDEF.epochs) setTimeout(next, 0);
      })();
    }

    function draw() {
      const c = colors();
      ctrl.querySelectorAll(".val").forEach((s) => { s.textContent = s.parentElement.querySelector("input").value; });
      fitSvg.innerHTML = ""; errSvg.innerHTML = "";
      const y = linear(-1, 9, H1 - 36, 24);
      axes(fitSvg, x, y, { xDomain: [0, 10], yDomain: [-1, 9], xTicks: ticks(0, 10, small ? 2 : 1), yTicks: [0, 2, 4, 6, 8], xLabel: "dose (mg)" });
      el("text", { x: M.l, y: 14, style: textStyle + "; font-size: 12px" }, fitSvg).textContent = hist.length ? (small ? `TM output after epoch ${hist.length}` : `TM output after epoch ${hist.length} (orange) vs true curve (dashed)`) : (small ? "training data and true curve" : "training data (blue) and true curve (dashed)");
      drawCurvePanel(fitSvg, c, x, y, hist.length ? tm : null, { points: CURVE.train });
      // learning curve
      const ex = linear(1, CDEF.epochs, M.l, W - M.r), ey = linear(0, 3, H2 - 34, 22);
      axes(errSvg, ex, ey, { xDomain: [1, CDEF.epochs], yDomain: [0, 3], xTicks: [1, 10, 20, 30, 40, 50, 60], yTicks: [0, 1, 2, 3], xLabel: "epoch" });
      el("text", { x: M.l, y: 13, style: textStyle + "; font-size: 12px" }, errSvg).textContent = "mean absolute error";
      el("line", { x1: ex(1), x2: ex(CDEF.epochs), y1: ey(CURVE.floor), y2: ey(CURVE.floor), stroke: c.opt, "stroke-dasharray": "5 4", "stroke-width": 1.5 }, errSvg);
      [[0, c.model], [1, c.loss]].forEach(([i, col]) => {
        if (!hist.length) return;
        const d = hist.map((h, e) => (e ? "L" : "M") + ex(e + 1).toFixed(1) + "," + ey(Math.min(3, h[i])).toFixed(1)).join("");
        el("path", { d, fill: "none", stroke: col, "stroke-width": 2, "stroke-linejoin": "round" }, errSvg);
      });
      const lx = small ? M.l + 6 : W - M.r - 300, ly = 13;
      if (!small) {
        el("line", { x1: lx, x2: lx + 16, y1: ly - 4, y2: ly - 4, stroke: c.model, "stroke-width": 2.5 }, errSvg);
        el("text", { x: lx + 20, y: ly }, errSvg).textContent = "train";
        el("line", { x1: lx + 62, x2: lx + 78, y1: ly - 4, y2: ly - 4, stroke: c.loss, "stroke-width": 2.5 }, errSvg);
        el("text", { x: lx + 82, y: ly }, errSvg).textContent = "test";
        el("line", { x1: lx + 118, x2: lx + 134, y1: ly - 4, y2: ly - 4, stroke: c.opt, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, errSvg);
        el("text", { x: lx + 138, y: ly }, errSvg).textContent = "true curve (noise floor)";
      }
      if (status) el("text", { x: (M.l + W - M.r) / 2, y: H1 / 2, "text-anchor": "middle", style: textStyle }, fitSvg).textContent = status;
      if (!done) { out.textContent = status || " "; return; }
      const h = hist[hist.length - 1], tail = hist.slice(10).map((q) => q[1]);
      const g = intervals(tm);
      out.textContent =
        `after epoch ${CDEF.epochs}: mean absolute error train ${fmt(h[0], 3)} · test ${fmt(h[1], 3)} · true curve on test ${fmt(CURVE.floor, 3)}` +
        (small ? "\n(orange = train, purple = test, green dashed = true curve)" : "") +
        `\ntest error over epochs 11–${CDEF.epochs}: ${fmt(Math.min(...tail), 3)} to ${fmt(Math.max(...tail), 3)} · output step (y range / T): ${fmt((CURVE.yMax - CURVE.yMin) / tm.T, 3)}` +
        `\nrules: ${g.list.length} distinct dose intervals, ${g.never} that can never fire, ${g.empty} empty` +
        (g.list.length ? ` · widest: ${doseText(g.list.reduce((a, b) => (b.hi - b.lo > a.hi - a.lo ? b : a)))}` : "");
    }

    ctrl.querySelectorAll("input[type=range]").forEach((i) => {
      i.addEventListener("input", () => { i.parentElement.querySelector(".val").textContent = i.value; });
      i.addEventListener("change", start);
    });
    ctrl.querySelector("[name=seed]").addEventListener("change", start);
    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "train") start();
      if (a === "defaults") {
        for (const k of ["m", "T", "s", "seed"]) ctrl.querySelector(`[name=${k}]`).value = CDEF[k];
        start();
      }
    });
    onThemeChange(draw);
    status = "scroll here to train";
    draw();
    whenVisible(root, start);
  }

  const mounts = { "viz-score": scoreWidget, "viz-stairs": stairsWidget, "viz-feedback": feedbackWidget, "viz-train": trainWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
