// Interactive widgets for the "Weighted clauses and clause sharing" lesson.
// Widgets, each mounted on a <div> with a matching id:
//   #viz-weight  – one rule's integer weight as it learns: it settles near T(2q − 1), q = the rule's precision
//   #viz-budget  – the same budget of rules, three machines: plain, weighted and coalesced (shared pool)
//   #viz-pool    – explore a trained coalesced machine: one pool of rules, a signed weight for every digit
// All randomness is seeded (mulberry32) and uses only + − × ÷ and comparisons, so the numbers match the
// Python check script exactly. Runs after DOMContentLoaded so window.Viz (assets/js/viz-common.js) exists.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

  const style = document.createElement("style");
  style.textContent = `
    .wtm-cards { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .wtm-card { flex: 1 1 150px; min-width: 0; background: var(--viz-bg); border: 1px solid var(--viz-border); border-radius: 6px; padding: 6px 8px; font-size: 0.7rem; color: var(--viz-muted); }
    .wtm-card b { color: var(--viz-text); }
    .wtm-rules { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 4px; }
    .wtm-rule { display: inline-flex; align-items: center; gap: 3px; }
    .wtm-rule svg { width: 26px; height: 42px; }
    .wtm-w { font-weight: 700; font-variant-numeric: tabular-nums; }
    .viz select.wtm-sel, .viz input.wtm-sel { font: inherit; padding: 2px 6px; border-radius: 6px; border: 1px solid var(--viz-border); background: var(--viz-bg); color: var(--viz-text); }
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

  // ---------------------------------------------------------------- Noisy 7-segment digits
  const SEGS = "abcdefg";
  const DIGITS = ["abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg"];
  const PAT = DIGITS.map((d) => SEGS.split("").map((s) => (d.includes(s) ? 1 : 0)));
  const NOISE = 0.05;
  const toLits = (x) => x.concat(x.map((b) => 1 - b));                   // a … g, then NOT a … NOT g
  function makeData(n, eps, seed) {
    const rand = mulberry32(seed), out = [];
    for (let i = 0; i < n; i++) {
      const y = Math.floor(rand() * 10);
      const x = PAT[y].map((b) => b ^ (rand() < eps ? 1 : 0));
      out.push({ lits: toLits(x), y });
    }
    return out;
  }
  let dataCache = null;
  const data = () => dataCache || (dataCache = { train: makeData(500, NOISE, 1), test: makeData(1000, NOISE, 2) });
  const NEAREST = 0.873;                    // nearest-clean-pattern classifier on this test set (Python check script)

  // ---------------------------------------------------------------- The three machines
  // kind "plain" / "weighted": one team per class, rules 0 … n/2 − 1 vote for, the rest against; W holds signed weights.
  // kind "coalesced": one pool of rules shared by all classes; W[i][j] = signed weight of rule j for class i.
  class Machine {
    constructor(kind, nClasses, nLits, rules, T, s, depth, seed) {
      this.kind = kind; this.m = nClasses; this.L = nLits; this.n = rules; this.T = T;
      this.depth = depth; this.half = Math.floor(depth / 2);
      this.memV = (s - 1) / s; this.fgV = 1 / s;
      this.rand = mulberry32(seed);
      const fresh = () => new Array(nLits).fill(this.half);
      if (kind === "coalesced") {
        this.mem = Array.from({ length: rules }, fresh);
        this.W = [];
        for (let i = 0; i < nClasses; i++) {
          const row = [];
          for (let j = 0; j < rules; j++) row.push(this.rand() < 0.5 ? 1 : -1);
          this.W.push(row);
        }
      } else {
        this.mem = Array.from({ length: nClasses }, () => Array.from({ length: rules }, fresh));
        this.W = Array.from({ length: nClasses }, () => Array.from({ length: rules }, (_, j) => (j < Math.floor(rules / 2) ? 1 : -1)));
      }
    }
    fires(rule, lits, training) {
      let empty = true;
      for (let k = 0; k < this.L; k++) {
        if (rule[k] > this.half) { empty = false; if (!lits[k]) return 0; }
      }
      return training || !empty ? 1 : 0;
    }
    outputs(lits, i, training) {
      const rules = this.kind === "coalesced" ? this.mem : this.mem[i];
      return rules.map((r) => this.fires(r, lits, training));
    }
    classSum(i, c) { let v = 0; for (let j = 0; j < c.length; j++) v += this.W[i][j] * c[j]; return v; }
    sums(lits) {
      if (this.kind === "coalesced") { const c = this.outputs(lits, 0, false); return this.W.map((_, i) => this.classSum(i, c)); }
      return this.W.map((_, i) => this.classSum(i, this.outputs(lits, i, false)));
    }
    predict(lits) {
      const s = this.sums(lits);
      let best = 0;
      for (let i = 1; i < this.m; i++) if (s[i] > s[best]) best = i;
      return best;
    }
    feedback(rule, lits, kind) {
      if (kind === "recognize") {
        for (let k = 0; k < this.L; k++) {
          if (lits[k]) { if (this.rand() <= this.memV && rule[k] < this.depth) rule[k]++; }
          else if (this.rand() <= this.fgV && rule[k] > 1) rule[k]--;
        }
      } else if (kind === "erase") {
        for (let k = 0; k < this.L; k++) if (this.rand() <= this.fgV && rule[k] > 1) rule[k]--;
      } else {
        for (let k = 0; k < this.L; k++) if (!lits[k] && rule[k] <= this.half) rule[k]++;
      }
    }
    updateClass(lits, i, y, c) {
      const T = this.T;
      let v = this.classSum(i, c);
      v = Math.max(-T, Math.min(T, v));
      const p = y === 1 ? (T - v) / (2 * T) : (T + v) / (2 * T);
      const rules = this.kind === "coalesced" ? this.mem : this.mem[i];
      const Wi = this.W[i];
      for (let j = 0; j < this.n; j++) {
        if (this.rand() > p) continue;
        const w = Wi[j];
        const positive = this.kind === "coalesced" ? w >= 0 : j < Math.floor(this.n / 2);
        const own = positive === (y === 1);
        if (own) this.feedback(rules[j], lits, c[j] ? "recognize" : "erase");
        else if (c[j]) this.feedback(rules[j], lits, "reject");
        if (!c[j]) continue;
        if (this.kind === "weighted") {
          if (own) Wi[j] = positive ? w + 1 : w - 1;            // Recognize: one more vote
          else if (w !== 0) Wi[j] = positive ? w - 1 : w + 1;   // Reject: one vote less, never below 0
        } else if (this.kind === "coalesced") {
          Wi[j] = y === 1 ? w + 1 : w - 1;                      // towards the example's side; may cross 0
        }
      }
    }
    update(lits, y) {
      let other = Math.floor(this.rand() * (this.m - 1));
      if (other >= y) other++;
      for (const [i, yy] of [[y, 1], [other, 0]]) this.updateClass(lits, i, yy, this.outputs(lits, i, true));
    }
    epoch(set) {
      const idx = set.map((_, i) => i);
      for (let a = idx.length - 1; a > 0; a--) {
        const b = Math.floor(this.rand() * (a + 1));
        [idx[a], idx[b]] = [idx[b], idx[a]];
      }
      for (const a of idx) this.update(set[a].lits, set[a].y);
    }
    accuracy(set) { return set.reduce((n, d) => n + (this.predict(d.lits) === d.y ? 1 : 0), 0) / set.length; }
    allRules() { return this.kind === "coalesced" ? this.mem : [].concat(...this.mem); }
    literalCount() { return this.allRules().reduce((n, r) => n + r.filter((m) => m > this.half).length, 0); }
  }

  // The settings used in the lesson (T picked on a validation set by the Python script)
  const CONFIG = {
    20: { plain: { rules: 2, T: 32 }, weighted: { rules: 2, T: 8 }, coalesced: { rules: 20, T: 128 } },
    60: { plain: { rules: 6, T: 2 }, weighted: { rules: 6, T: 64 }, coalesced: { rules: 60, T: 128 } },
  };
  const S = 3, DEPTH = 100, EPOCHS = 40;
  const KINDS = [
    { k: "plain", name: "plain (weight 1 each)", short: "plain", role: "muted" },
    { k: "weighted", name: "weighted (integer weights)", short: "weighted", role: "model" },
    { k: "coalesced", name: "coalesced (shared pool)", short: "coalesced", role: "loss" },
  ];

  // ---------------------------------------------------------------- Drawing a 7-segment glyph
  // state per segment: 1 = on / must be on, -1 = must be off, 0 = off / don't care, 2 = impossible (both)
  function glyph(parent, x, y, w, h, states, opts) {
    const c = colors(), t = opts.t || Math.max(2.5, w * 0.16), gap = t * 0.55;
    const g = el("g", {}, parent);
    const geo = {
      a: [x + gap, y - t / 2, w - 2 * gap, t], g: [x + gap, y + h / 2 - t / 2, w - 2 * gap, t], d: [x + gap, y + h - t / 2, w - 2 * gap, t],
      f: [x - t / 2, y + gap, t, h / 2 - 2 * gap], b: [x + w - t / 2, y + gap, t, h / 2 - 2 * gap],
      e: [x - t / 2, y + h / 2 + gap, t, h / 2 - 2 * gap], c: [x + w - t / 2, y + h / 2 + gap, t, h / 2 - 2 * gap],
    };
    SEGS.split("").forEach((s, k) => {
      const [rx, ry, rw, rh] = geo[s], st = states[k];
      const a = { x: rx, y: ry, width: rw, height: rh, rx: Math.min(rw, rh) / 2 };
      if (opts.input) Object.assign(a, { fill: st ? c.data : c.grid, stroke: "none" });
      else if (st === 1) Object.assign(a, { fill: c.opt, stroke: "none" });
      else if (st === -1) Object.assign(a, { fill: "none", stroke: c.opt, "stroke-width": Math.max(1, t * 0.28), "stroke-dasharray": `${Math.max(1.5, t * 0.5)} ${Math.max(1, t * 0.35)}` });
      else if (st === 2) Object.assign(a, { fill: c.grad, stroke: "none" });
      else Object.assign(a, { fill: c.grid, stroke: "none" });
      const r = el("rect", a, g);
      if (opts.onClick) { r.style.cursor = "pointer"; r.addEventListener("click", () => opts.onClick(k)); }
      if (opts.onClick) {   // a wider invisible hit area for fingers
        const pad = 6;
        const hit = el("rect", { x: rx - pad, y: ry - pad, width: rw + 2 * pad, height: rh + 2 * pad, fill: "transparent" }, g);
        hit.style.cursor = "pointer"; hit.addEventListener("click", () => opts.onClick(k));
      }
    });
    return g;
  }
  // A rule's memory → segment states
  function ruleStates(rule, half) {
    return SEGS.split("").map((_, k) => {
      const on = rule[k] > half, off = rule[k + 7] > half;
      return on && off ? 2 : on ? 1 : off ? -1 : 0;
    });
  }
  function ruleText(rule, half) {
    const parts = [];
    SEGS.split("").forEach((s, k) => { if (rule[k] > half) parts.push(s); });
    SEGS.split("").forEach((s, k) => { if (rule[k + 7] > half) parts.push("¬" + s); });
    return parts.length ? parts.join(" ∧ ") : "(empty)";
  }
  function glyphSVG(states) {
    const s = svg(20, 32);
    glyph(s, 4, 3, 12, 26, states, { t: 2.6 });
    return s;
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
  const signed = (v) => (v > 0 ? "+" + v : v < 0 ? "−" + -v : "0");
  function whenVisible(root, fn) {
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
      }, { rootMargin: "200px" });
      io.observe(root);
    } else fn();
  }

  // ---------------------------------------------------------------- Widget 1: one rule's weight
  // The rule fires on a stream of examples; a share q of them belong to its own class.
  // Own class: v = +w (clipped to T), feedback probability (T − v)/2T, picked → weight + 1 (Recognize).
  // Other class: v = −w from that example's side, probability (T + w)/2T, picked → weight − 1 (Reject), never below 0.
  function walk(q, T, steps, seed) {
    const rand = mulberry32(seed), hist = [];
    let w = 1;
    for (let t = 0; t < steps; t++) {
      const own = rand() < q;
      const vv = Math.min(w, T);
      const p = own ? (T - vv) / (2 * T) : (T + vv) / (2 * T);
      if (rand() <= p) { if (own) w++; else if (w > 0) w--; }
      hist.push(w);
    }
    return hist;
  }

  function weightWidget(root) {
    const small = narrow(root);
    const W = small ? 340 : 560, H = 250, M = { l: 40, r: 14, t: 28, b: 40 };
    const STEPS = 2000, SEED = 11;
    const chart = svg(W, H);
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>precision <b>q</b> <input type="range" name="q" min="0.3" max="1" step="0.05" value="0.9"> <span class="val"></span></label>
      <label>Voting Margin <b>T</b> <input type="range" name="T" min="2" max="20" step="1" value="10"> <span class="val"></span></label>
      <button data-a="reset">Defaults</button>`);
    const out = readout(root);
    hint(root, `The rule fires on ${STEPS.toLocaleString("en")} examples in a row (seed ${SEED}); a share q of them belong to the class it votes for. It starts at weight 1. Picked on an own-class example: weight + 1 (Recognize). Picked on an other-class example: weight − 1 (Reject), never below 0. Picking probabilities use the rule's own votes as v, as if no other rule fired.`);
    const qIn = ctrl.querySelector("[name=q]"), tIn = ctrl.querySelector("[name=T]");

    function draw() {
      const c = colors(), q = parseFloat(qIn.value), T = +tIn.value;
      ctrl.querySelectorAll(".val").forEach((s) => { s.textContent = s.parentElement.querySelector("input").value; });
      const h = walk(q, T, STEPS, SEED);
      const yMax = 20;
      const x = linear(0, STEPS, M.l, W - M.r), y = linear(0, yMax, H - M.b, M.t);
      chart.innerHTML = "";
      axes(chart, x, y, { xDomain: [0, STEPS], yDomain: [0, yMax], xTicks: ticks(0, STEPS, small ? 1000 : 500), yTicks: ticks(0, yMax, 5), xLabel: "examples the rule fired on" });
      el("text", { x: M.l, y: 16, style: textStyle + "; font-size: 12px" }, chart).textContent = "the rule's weight w";
      // T and the predicted resting point T(2q − 1)
      el("line", { x1: x(0), x2: x(STEPS), y1: y(T), y2: y(T), stroke: c.muted, "stroke-dasharray": "3 3" }, chart);
      el("text", { x: x(STEPS) - 4, y: y(T) - 4, "text-anchor": "end" }, chart).textContent = "T";
      const pred = Math.max(0, T * (2 * q - 1));
      el("line", { x1: x(0), x2: x(STEPS), y1: y(pred), y2: y(pred), stroke: c.opt, "stroke-width": 2, "stroke-dasharray": "7 4" }, chart);
      let d = "";
      h.forEach((w, t) => { d += (t ? "L" : "M") + x(t + 1).toFixed(1) + "," + y(w).toFixed(1); });
      el("path", { d, fill: "none", stroke: c.loss, "stroke-width": 1.4, "stroke-linejoin": "round" }, chart);
      const lx = M.l + 8, ly = M.t + 8;
      el("line", { x1: lx, x2: lx + 18, y1: ly, y2: ly, stroke: c.loss, "stroke-width": 2.5 }, chart);
      el("text", { x: lx + 22, y: ly + 4, style: textStyle }, chart).textContent = "weight";
      el("line", { x1: lx + 84, x2: lx + 102, y1: ly, y2: ly, stroke: c.opt, "stroke-width": 2, "stroke-dasharray": "7 4" }, chart);
      el("text", { x: lx + 106, y: ly + 4, style: textStyle }, chart).textContent = "T(2q − 1)";
      const tail = h.slice(STEPS / 2);
      const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
      const zero = tail.filter((w) => w === 0).length / tail.length;
      out.textContent =
        `precision q = ${fmt(q, 2)}, T = ${T}: the drift settles where q(T − w) = (1 − q)(T + w), i.e. w = T(2q − 1) = ${fmt(T * (2 * q - 1), 1).replace("-", "−")}${T * (2 * q - 1) < 0 ? " → below 0, so the weight sits at the floor" : ""}\n` +
        `mean weight over examples ${STEPS / 2 + 1}–${STEPS}: ${fmt(mean, 2)} · share of those examples at weight 0 (rule switched off): ${Math.round(100 * zero)}% · highest weight reached: ${Math.max(...h)}`;
    }
    ctrl.querySelectorAll("input").forEach((i) => i.addEventListener("input", draw));
    ctrl.querySelector("[data-a=reset]").addEventListener("click", () => { qIn.value = "0.9"; tIn.value = "10"; draw(); });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 2: same budget, three machines
  function budgetWidget(root) {
    const small = narrow(root);
    const W = small ? 340 : 560, H = 250, M = { l: 44, r: 14, t: 44, b: 38 };
    const chart = svg(W, H);
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>rule budget <select class="wtm-sel" name="budget"><option value="20" selected>20 rules</option><option value="60">60 rules</option></select></label>
      <label>seed <input class="wtm-sel" type="number" name="seed" value="1" min="1" max="999" style="width: 4.5em"></label>
      <button data-a="train">Train again</button>
      <label>show rules of <select class="wtm-sel" name="show"><option value="weighted" selected>weighted</option><option value="plain">plain</option></select></label>`);
    const out = readout(root);
    const cards = document.createElement("div");
    cards.className = "wtm-cards";
    root.appendChild(cards);
    hint(root, `Noisy 7-segment digits (5% of segments flipped): 500 training and 1000 test examples, 10 classes. Each machine gets the same number of rules in total. Plain and weighted: per digit, half the rules vote for it and half against. Coalesced: one pool shared by all digits. s = ${S}, ${EPOCHS} epochs, T per machine as in the lesson's table. Rule pictures: <b>solid green</b> segment = must be on, <b>dashed green</b> = must be off, grey = don't care.`);
    const x = linear(1, EPOCHS, M.l, W - M.r), y = linear(0.5, 0.9, H - M.b, M.t);
    let run = null, hist = null, machines = null, status = "";

    function start() {
      if (run) run.cancel = true;
      const budget = +ctrl.querySelector("[name=budget]").value;
      const seed = Math.max(1, Math.round(+ctrl.querySelector("[name=seed]").value) || 1);
      const { train, test } = data();
      machines = KINDS.map(({ k }) => { const cf = CONFIG[budget][k]; return new Machine(k, 10, 14, cf.rules, cf.T, S, DEPTH, seed); });
      hist = KINDS.map(() => []);
      const me = { cancel: false, budget, seed, done: false };
      run = me;
      let e = 0;
      (function next() {
        if (me.cancel) return;
        machines.forEach((m, i) => { m.epoch(train); hist[i].push(m.accuracy(test)); });
        e++;
        status = e < EPOCHS ? `training… epoch ${e} / ${EPOCHS}` : "";
        if (e === EPOCHS) me.done = true;
        draw();
        if (e < EPOCHS) setTimeout(next, 0);
      })();
    }

    function draw() {
      const c = colors();
      chart.innerHTML = "";
      axes(chart, x, y, { xDomain: [1, EPOCHS], yDomain: [0.5, 0.9], xTicks: [1, 10, 20, 30, 40], yTicks: [0.5, 0.6, 0.7, 0.8, 0.9], xLabel: "epoch" });
      el("text", { x: M.l, y: 14, style: textStyle + "; font-size: 12px" }, chart).textContent = "test accuracy";
      el("line", { x1: x(1), x2: x(EPOCHS), y1: y(NEAREST), y2: y(NEAREST), stroke: c.opt, "stroke-dasharray": "5 4", "stroke-width": 1.5 }, chart);
      el("text", { x: x(EPOCHS) - 4, y: y(NEAREST) + 14, "text-anchor": "end" }, chart).textContent = small ? "nearest pattern" : "nearest clean pattern (best possible ≈ 87%)";
      if (hist) KINDS.forEach((kd, i) => {
        const hh = hist[i];
        if (!hh.length) return;
        const col = c[kd.role];
        const d = hh.map((a, e) => (e ? "L" : "M") + x(e + 1).toFixed(1) + "," + y(Math.max(0.5, a)).toFixed(1)).join("");
        el("path", { d, fill: "none", stroke: col, "stroke-width": 2, "stroke-linejoin": "round" }, chart);
        el("circle", { cx: x(hh.length), cy: y(Math.max(0.5, hh[hh.length - 1])), r: 3.5, fill: col }, chart);
      });
      // legend in one row above the plot
      let lx = M.l;
      KINDS.forEach((kd) => {
        el("line", { x1: lx, x2: lx + 16, y1: 30, y2: 30, stroke: c[kd.role], "stroke-width": 2.5 }, chart);
        el("text", { x: lx + 20, y: 34, style: textStyle }, chart).textContent = small ? kd.short : kd.name;
        lx += small ? 98 : 172;
      });
      if (status) el("text", { x: (M.l + W - M.r) / 2, y: (M.t + H - M.b) / 2 + 30, "text-anchor": "middle", style: textStyle }, chart).textContent = status;

      if (!run || !run.done) { out.textContent = status || " "; cards.innerHTML = ""; return; }
      out.textContent = `budget ${run.budget} rules · seed ${run.seed} · nearest clean pattern scores ${fmt(NEAREST, 3)}\n` + KINDS.map((kd, i) => {
        const hh = hist[i], tail = hh.slice(30), m = machines[i];
        const per = kd.k === "coalesced" ? `pool of ${m.n}` : `${m.n} per digit`;
        return `${kd.short} (${per}, T = ${m.T}): final ${fmt(hh[hh.length - 1], 3)} · mean of epochs 31–40 ${fmt(tail.reduce((a, b) => a + b, 0) / tail.length, 3)} (${fmt(Math.min(...tail), 3)}–${fmt(Math.max(...tail), 3)}) · literals memorized ${m.literalCount()}`;
      }).join("\n");
      drawCards();
    }

    function drawCards() {
      const which = ctrl.querySelector("[name=show]").value;
      const m = machines[which === "plain" ? 0 : 1];
      cards.innerHTML = "";
      for (let d = 0; d < 10; d++) {
        const card = document.createElement("div");
        card.className = "wtm-card";
        card.innerHTML = `<b>digit ${d}</b>`;
        const row = document.createElement("div");
        row.className = "wtm-rules";
        for (let j = 0; j < m.n; j++) {
          const r = m.mem[d][j], w = m.W[d][j];
          const span = document.createElement("span");
          span.className = "wtm-rule";
          span.title = `${j < m.n / 2 ? "votes for" : "votes against"} digit ${d}: IF ${ruleText(r, m.half)} · weight ${Math.abs(w)}`;
          span.appendChild(glyphSVG(ruleStates(r, m.half)));
          const lab = document.createElement("span");
          lab.className = "wtm-w";
          lab.style.color = `var(--viz-${w > 0 ? "loss" : w < 0 ? "grad" : "muted"})`;
          lab.textContent = (j < m.n / 2 ? "+" : "−") + Math.abs(w);
          span.appendChild(lab);
          row.appendChild(span);
        }
        card.appendChild(row);
        cards.appendChild(card);
      }
    }

    ctrl.querySelector("[name=budget]").addEventListener("change", start);
    ctrl.querySelector("[name=seed]").addEventListener("change", start);
    ctrl.querySelector("[name=show]").addEventListener("change", () => { if (run && run.done) drawCards(); });
    ctrl.querySelector("[data-a=train]").addEventListener("click", start);
    onThemeChange(draw);
    status = "scroll here to train";
    draw();
    whenVisible(root, start);
  }

  // ---------------------------------------------------------------- Widget 3: explore the shared pool
  function poolWidget(root) {
    const small = narrow(root);
    const cf = CONFIG[20].coalesced, SEED = 1;
    const quick = control(root, `<span>clean digit:</span>${[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => `<button data-d="${d}">${d}</button>`).join("")}<button data-d="test">next test example ▶</button>`);
    const W = small ? 360 : 640;
    const L = small
      ? { inX: 22, inY: 26, inW: 46, inH: 78, rowsY: 150, labW: 52, cellW: 30.5, rowH: 22, sumH: 110 }
      : { inX: 36, inY: 64, inW: 56, inH: 100, rowsY: 46, labW: 248, cellW: 39, rowH: 22, sumH: 120 };
    const N = cf.rules;
    const H = L.rowsY + 22 + N * L.rowH + 16 + L.sumH;
    const chart = svg(W, H);
    root.appendChild(chart);
    const out = readout(root);
    hint(root, `A coalesced machine with a pool of ${N} rules, trained on the noisy 7-segment digits (T = ${cf.T}, s = ${S}, ${EPOCHS} epochs, seed ${SEED}; the same run as the coalesced line in the widget above). Click the segments of the input to switch them on or off. Each row is one rule: its picture (solid green = must be on, dashed green = must be off) and its weight for every digit (<b style="color: var(--viz-loss)">purple</b> = votes for, <b style="color: var(--viz-grad)">red</b> = votes against). Rows of rules that fire are highlighted; a digit's class sum is the sum of its column over those rows.`);
    let tm = null, input = PAT[5].slice(), label = 5, testPos = -1;

    function train() {
      const { train } = data();
      tm = new Machine("coalesced", 10, 14, cf.rules, cf.T, S, DEPTH, SEED);
      for (let e = 0; e < EPOCHS; e++) tm.epoch(train);
    }

    function draw() {
      const c = colors();
      chart.innerHTML = "";
      if (!tm) { el("text", { x: W / 2, y: 80, "text-anchor": "middle", style: textStyle }, chart).textContent = "scroll here to train the pool"; return; }
      const lits = toLits(input);
      const fire = tm.outputs(lits, 0, false);
      const sums = tm.sums(lits), pred = tm.predict(lits);
      const maxW = Math.max(...tm.W.map((r) => Math.max(...r.map(Math.abs))));
      const x0 = L.labW;
      const colX = (i) => x0 + i * L.cellW;
      // input display
      el("text", { x: small ? 10 : 20, y: small ? 14 : 30, style: textStyle + "; font-weight: 700" }, chart).textContent = "input";
      glyph(chart, L.inX, L.inY, L.inW, L.inH, input, { input: true, t: small ? 8 : 11, onClick: (k) => { input[k] = 1 - input[k]; label = null; draw(); } });
      if (small) {
        el("text", { x: 96, y: 40, style: textStyle }, chart).textContent = `fires: ${fire.reduce((a, b) => a + b, 0)} of ${N} rules`;
        el("text", { x: 96, y: 62, style: "fill: var(--viz-loss); font-weight: 700" }, chart).textContent = `prediction: ${pred}`;
        el("text", { x: 96, y: 84 }, chart).textContent = label === null ? "(edited input)" : `true digit: ${label}`;
        el("text", { x: 96, y: 106 }, chart).textContent = "tap segments to edit";
      } else {
        el("text", { x: 14, y: L.inY + L.inH + 36 }, chart).textContent = "click segments";
        el("text", { x: 14, y: L.inY + L.inH + 52 }, chart).textContent = "to edit them";
        el("text", { x: 14, y: L.inY + L.inH + 84, style: textStyle }, chart).textContent = `fires: ${fire.reduce((a, b) => a + b, 0)} of ${N}`;
        el("text", { x: 14, y: L.inY + L.inH + 104, style: "fill: var(--viz-loss); font-weight: 700" }, chart).textContent = `prediction: ${pred}`;
        if (label !== null) el("text", { x: 14, y: L.inY + L.inH + 124 }, chart).textContent = `true digit: ${label}`;
      }
      // header: digit labels
      const top = L.rowsY;
      for (let i = 0; i < 10; i++) el("text", { x: colX(i) + L.cellW / 2, y: top + 12, "text-anchor": "middle", style: textStyle + "; font-weight: 700" }, chart).textContent = i;
      el("text", { x: small ? 8 : 126, y: top + 12 }, chart).textContent = small ? "rule" : "rule";
      // rows
      for (let j = 0; j < N; j++) {
        const yy = top + 22 + j * L.rowH;
        const on = fire[j] === 1;
        if (on) el("rect", { x: small ? 2 : 122, y: yy, width: colX(10) - (small ? 2 : 122), height: L.rowH, fill: c.model, "fill-opacity": 0.16, rx: 3 }, chart);
        const g = el("g", { opacity: on ? 1 : 0.38 }, chart);
        const lx = small ? 6 : 126;
        el("text", { x: lx, y: yy + L.rowH / 2 + 4, style: on ? textStyle : "" }, g).textContent = `R${j + 1}`;
        glyph(g, lx + (small ? 27 : 28), yy + 4, 8, L.rowH - 8, ruleStates(tm.mem[j], tm.half), { t: 2.2 });
        if (!small) el("text", { x: lx + 44, y: yy + L.rowH / 2 + 4, style: "font-size: 10px" + (on ? "; fill: var(--viz-text)" : "") }, g).textContent = ruleText(tm.mem[j], tm.half);
        for (let i = 0; i < 10; i++) {
          const w = tm.W[i][j];
          el("rect", { x: colX(i) + 1, y: yy + 1, width: L.cellW - 2, height: L.rowH - 2, rx: 2,
            fill: w > 0 ? c.loss : w < 0 ? c.grad : c.grid, "fill-opacity": w === 0 ? 1 : 0.12 + 0.75 * Math.abs(w) / maxW }, g);
          el("text", { x: colX(i) + L.cellW / 2, y: yy + L.rowH / 2 + 3.5, "text-anchor": "middle", style: `font-size: ${small ? 9 : 10}px; fill: var(--viz-text)` }, g).textContent = w;
        }
      }
      // class sums as bars, aligned with the columns
      const sTop = top + 22 + N * L.rowH + 16;
      const maxS = Math.max(50, ...sums.map(Math.abs));
      const sy = linear(-maxS, maxS, sTop + L.sumH - 16, sTop + 10);
      el("text", { x: small ? 6 : 126, y: sTop + 14, style: textStyle }, chart).textContent = small ? "sum" : "class sum v";
      if (!small) el("text", { x: 126, y: sTop + 30 }, chart).textContent = "(firing rows)";
      el("line", { x1: colX(0), x2: colX(10), y1: sy(0), y2: sy(0), stroke: c.muted }, chart);
      for (let i = 0; i < 10; i++) {
        const v = sums[i];
        el("rect", { x: colX(i) + 4, y: Math.min(sy(0), sy(v)), width: L.cellW - 8, height: Math.abs(sy(v) - sy(0)), fill: v >= 0 ? c.loss : c.grad, "fill-opacity": i === pred ? 1 : 0.55, rx: 2 }, chart);
        el("text", { x: colX(i) + L.cellW / 2, y: v >= 0 ? sy(v) - 4 : sy(v) + 12, "text-anchor": "middle", style: `font-size: ${small ? 8.5 : 10}px; ${i === pred ? "fill: var(--viz-loss); font-weight: 700" : ""}` }, chart).textContent = v;
        if (i === pred) el("rect", { x: colX(i) + 1, y: top + 1, width: L.cellW - 2, height: sTop + L.sumH - top - 2, fill: "none", stroke: c.loss, "stroke-width": 1.5, rx: 3 }, chart);
      }
      const firing = fire.map((f, j) => (f ? `R${j + 1} (${ruleText(tm.mem[j], tm.half)})` : null)).filter(Boolean);
      const order = sums.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
      out.textContent =
        `input segments on: ${SEGS.split("").filter((_, k) => input[k]).join(" ") || "none"}${label === null ? "" : ` · true digit ${label}`}\n` +
        `firing rules: ${firing.join(", ") || "none"}\n` +
        `prediction: ${pred} (class sum ${signed(sums[pred])}) · runner-up: ${order[1][1]} (${signed(order[1][0])})${label === null ? "" : pred === label ? " · ✓" : " · ✗"}`;
    }

    quick.addEventListener("click", (e) => {
      const b = e.target.dataset.d;
      if (b === undefined) return;
      if (b === "test") {
        const { test } = data();
        testPos = (testPos + 1) % test.length;
        input = test[testPos].lits.slice(0, 7); label = test[testPos].y;
      } else { input = PAT[+b].slice(); label = +b; }
      draw();
    });
    onThemeChange(draw);
    draw();
    whenVisible(root, () => { train(); draw(); });
  }

  const mounts = { "viz-weight": weightWidget, "viz-budget": budgetWidget, "viz-pool": poolWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
