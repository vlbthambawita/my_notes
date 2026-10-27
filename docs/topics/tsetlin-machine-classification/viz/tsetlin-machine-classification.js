// Interactive widgets for the "Many rules voting: classification with a Tsetlin machine" lesson.
// Widgets, each mounted on a <div> with a matching id:
//   #viz-vote    – rules for and against Recurrence vote on a patient you choose; see the vote sum v
//   #viz-margin  – how the Voting Margin T turns the vote sum into a feedback probability
//   #viz-orgame  – two rules learning "x1 OR x2" with and without a Voting Margin (book Ch. 2, §2.3)
//   #viz-train   – train a complete Tsetlin machine on the UCI breast-cancer data in your browser
// All randomness is seeded (mulberry32) and uses only + − × ÷ and comparisons, so the numbers match the
// Python check script exactly. Runs after DOMContentLoaded so window.Viz (assets/js/viz-common.js) exists.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

  // ---------------------------------------------------------------- Small styles for the rule lists
  const style = document.createElement("style");
  style.textContent = `
    .tmv-cols { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
    .tmv-col { flex: 1 1 260px; min-width: 0; background: var(--viz-bg); border: 1px solid var(--viz-border); border-radius: 6px; padding: 8px 10px; }
    .tmv-col h4 { margin: 0 0 6px; font-size: 0.75rem; }
    .tmv-rule { display: flex; gap: 8px; align-items: flex-start; font-size: 0.72rem; line-height: 1.35; padding: 3px 4px; border-radius: 4px; color: var(--viz-muted); }
    .tmv-rule.on { color: var(--viz-text); background: color-mix(in srgb, var(--role) 14%, transparent); }
    .tmv-dot { flex: 0 0 auto; width: 10px; height: 10px; margin-top: 3px; border-radius: 50%; border: 2px solid var(--role); }
    .tmv-rule.on .tmv-dot { background: var(--role); }
    .tmv-n { flex: 0 0 auto; font-weight: 700; color: var(--role); min-width: 1.8em; }
    .tmv-txt { flex: 1 1 auto; min-width: 0; overflow-wrap: anywhere; }
    .tmv-sub { color: var(--viz-muted); font-size: 0.68rem; }
    .viz select.tmv-sel, .viz input.tmv-sel { font: inherit; padding: 2px 6px; border-radius: 6px; border: 1px solid var(--viz-border); background: var(--viz-bg); color: var(--viz-text); }
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

  // ---------------------------------------------------------------- The breast-cancer data (UCI, Ljubljana)
  // One patient per group of 4 digits, in the file's order: Menopause, Inv-nodes, Deg-malig (value indices), Recurrence.
  const PROPS = [
    { name: "Menopause", vals: ["lt40", "ge40", "premeno"] },
    { name: "Inv-nodes", vals: ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "24-26"] },
    { name: "Deg-malig", vals: ["1", "2", "3"] },
  ];
  const RAW = "2020 2010 2010 1010 2010 1010 2010 1000 2010 2010 2020 1010 0000 1020 2020 0000 2010 2020 1020 1000 1010 1010 2010 2010 2010 2010 2000 1020 2010 1010 1020 2010 2020 1000 1000 2010 1010 1000 2010 2000 1000 1010 1000 2010 1000 1000 2010 2000 2000 2000 0010 2010 2000 1020 1000 1000 1000 1010 1000 1000 2010 2000 1000 2010 2000 2000 2000 1020 1020 1010 1000 2010 1010 2010 2010 1000 1010 2000 2010 2010 1010 1000 1010 1010 2000 1020 1010 1000 1020 2000 2010 1020 1010 1010 2000 2010 1010 1010 2000 2010 1020 1020 2010 2010 2010 2000 2010 2010 2020 1000 1000 1000 2010 2000 2010 1010 2000 1020 2000 1000 1000 1010 1020 1000 1000 1010 2210 2210 2010 2310 2310 2120 2210 1020 1010 2110 2120 2000 2010 2010 1110 2110 1000 2010 1220 2010 1010 2110 2020 0020 1520 1120 1520 1020 0020 1110 1120 2000 1000 2000 2020 2110 1010 1100 1100 2110 2110 1510 2010 1010 2110 2000 1210 1110 2110 2010 1010 1010 2110 2020 2020 2020 2000 1320 1320 2210 1000 1010 2010 1010 2420 2010 1210 2010 2010 1010 2100 2010 1010 2000 1020 2011 2001 1011 2011 2011 1021 2011 2021 2001 2011 1011 2021 2001 1011 1011 1011 2011 1021 2001 2021 2001 1021 1021 2121 1121 2521 2021 1121 1001 2311 2111 2121 1301 1021 2121 2011 2011 2021 2111 1111 2521 1221 1121 2421 2311 2221 1321 1221 2111 2021 2011 2001 1021 2021 2221 2001 2111 1211 1121 1111 1421 1021 0001 0001 2321 2111 1621 2021 2011 1221 2021 2021 1011 1111 2321 2221 1221 2521 1221 1221 2011 2021 1001 1121 1121";
  const PATIENTS = RAW.split(" ").map((s) => ({ p: [+s[0], +s[1], +s[2]], y: +s[3] }));
  const NF = PROPS.reduce((n, p) => n + p.vals.length, 0);               // 13 Boolean features
  const FEAT = [];                                                       // feature k → [property, value]
  PROPS.forEach((p, i) => p.vals.forEach((_, v) => FEAT.push([i, v])));
  const features = (p) => FEAT.map(([i, v]) => (p[i] === v ? 1 : 0));
  const literals = (x) => x.concat(x.map((b) => 1 - b));                 // x_1..x_K, then NOT x_1..NOT x_K
  PATIENTS.forEach((pt) => { pt.x = features(pt.p); pt.lits = literals(pt.x); });

  // Shuffle the patients with a seed; the first n_train are for training, the rest for testing
  function split(seed, nTrain) {
    const rand = mulberry32(seed), idx = PATIENTS.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return { train: idx.slice(0, nTrain).map((i) => PATIENTS[i]), test: idx.slice(nTrain).map((i) => PATIENTS[i]) };
  }

  // ---------------------------------------------------------------- The two-class Tsetlin machine (book Ch. 1)
  // memory[c][j][k] = position (1 … depth) of literal k in rule j of class c; memorized if > depth / 2.
  // Class 1 = Recurrence (rules vote for it), class 0 = No recurrence (rules vote against it).
  class TM {
    constructor(nRules, T, s, depth, seed) {
      this.n = nRules; this.T = T; this.depth = depth; this.half = depth / 2; this.L = 2 * NF;
      this.memV = (s - 1) / s; this.forgetV = 1 / s;
      this.rand = mulberry32(seed);
      this.memory = [0, 1].map(() => Array.from({ length: nRules }, () => new Int16Array(this.L).fill(depth / 2)));
    }
    fires(rule, lits, training) {
      let empty = true;
      for (let k = 0; k < this.L; k++) {
        if (rule[k] > this.half) { empty = false; if (!lits[k]) return 0; }
      }
      return training || !empty ? 1 : 0;          // an empty rule is True while learning, silent when classifying
    }
    votes(lits, training) {
      const f = [0, 1].map((c) => this.memory[c].map((r) => this.fires(r, lits, training)));
      return { v: f[1].reduce((a, b) => a + b, 0) - f[0].reduce((a, b) => a + b, 0), f };
    }
    predict(lits) { return this.votes(lits, false).v > 0 ? 1 : 0; }   // a tie goes to No recurrence
    update(lits, y) {
      const { v: vRaw, f } = this.votes(lits, true);
      let v = y === 1 ? vRaw : -vRaw;               // votes for the patient's own class minus votes against it
      v = Math.max(-this.T, Math.min(this.T, v));
      const p = (this.T - v) / (2 * this.T);        // the Voting Margin turns v into a feedback probability
      const top = this.depth;
      for (const c of [1, 0]) {
        for (let j = 0; j < this.n; j++) {
          if (this.rand() > p) continue;
          const rule = this.memory[c][j];
          if (c === y) {
            if (f[c][j]) {                          // Recognize Feedback
              for (let k = 0; k < this.L; k++) {
                if (lits[k]) { if (this.rand() <= this.memV && rule[k] < top) rule[k]++; }
                else if (this.rand() <= this.forgetV && rule[k] > 1) rule[k]--;
              }
            } else {                                // Erase Feedback
              for (let k = 0; k < this.L; k++) if (this.rand() <= this.forgetV && rule[k] > 1) rule[k]--;
            }
          } else if (f[c][j]) {                     // Reject Feedback
            for (let k = 0; k < this.L; k++) if (!lits[k] && rule[k] <= this.half) rule[k]++;
          }
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
    accuracy(data) { return data.reduce((a, d) => a + (this.predict(d.lits) === d.y ? 1 : 0), 0) / data.length; }
    // The rule as a list of memorized literal indices
    rule(c, j) {
      const out = [];
      for (let k = 0; k < this.L; k++) if (this.memory[c][j][k] > this.half) out.push(k);
      return out;
    }
  }

  // ---------------------------------------------------------------- Reading a rule as plain English
  // lits: memorized literal indices (k < NF: feature k; k >= NF: NOT feature k − NF)
  function english(lits) {
    if (!lits.length) return { text: "(empty rule: no literals memorized)", never: true };
    const parts = [];
    let never = false;
    PROPS.forEach((p, i) => {
      const pos = [], neg = [];
      for (const k of lits) {
        const [pi, v] = FEAT[k % NF];
        if (pi === i) (k < NF ? pos : neg).push(v);
      }
      if (!pos.length && !neg.length) return;
      if (pos.length > 1 || pos.some((v) => neg.includes(v))) { never = true; parts.push(`${p.name} is ${pos.map((v) => p.vals[v]).join(" and ")}${neg.length ? " (and not)" : ""}`); return; }
      if (pos.length === 1) { parts.push(`${p.name} is ${p.vals[pos[0]]}`); return; }
      const allowed = p.vals.map((_, v) => v).filter((v) => !neg.includes(v));
      if (!allowed.length) { never = true; parts.push(`${p.name} is none of its values`); return; }
      const list = (vs) => vs.map((v) => p.vals[v]).join(", ").replace(/, ([^,]*)$/, " or $1");
      if (allowed.length === 1) parts.push(`${p.name} is ${p.vals[allowed[0]]}`);
      else if (allowed.length < neg.length) parts.push(`${p.name} is ${list(allowed)}`);
      else parts.push(`${p.name} is not ${list(neg)}`);
    });
    return { text: parts.join(" AND "), never };
  }
  const litName = (k) => (k < NF ? "" : "NOT ") + `${PROPS[FEAT[k % NF][0]].name} ${PROPS[FEAT[k % NF][0]].vals[FEAT[k % NF][1]]}`;

  // Group rules that say the same thing (same plain-English condition): [{lits, count}]
  function groupRules(list) {
    const map = new Map();
    for (const lits of list) {
      const key = english(lits).text;
      if (map.has(key)) map.get(key).count++; else map.set(key, { lits, count: 1 });
    }
    return [...map.values()];
  }
  const firesOn = (lits, ptLits, training) => (lits.length === 0 ? training : lits.every((k) => ptLits[k] === 1));

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

  // The model trained with the lesson's default settings (also used by the voting panel)
  const DEFAULTS = { n: 10, T: 10, s: 3, depth: 200, epochs: 60, seed: 1, nTrain: 200 };
  let defaultModel = null;
  function getDefaultModel() {
    if (defaultModel) return defaultModel;
    const { train, test } = split(DEFAULTS.seed, DEFAULTS.nTrain);
    const tm = new TM(DEFAULTS.n, DEFAULTS.T, DEFAULTS.s, DEFAULTS.depth, DEFAULTS.seed);
    for (let e = 0; e < DEFAULTS.epochs; e++) tm.epoch(train);
    defaultModel = { tm, train, test };
    return defaultModel;
  }

  // ---------------------------------------------------------------- Widget 1: rules vote on a patient
  // Book rules use literal indices: feature index of a value = offset of its property + value index
  const fi = (prop, val) => FEAT.findIndex(([i, v]) => i === prop && PROPS[i].vals[v] === val);
  const DEG3 = fi(2, "3"), LT40 = fi(0, "lt40"), N02 = fi(1, "0-2");
  const BOOK_PATIENTS = [["ge40", "3-5", "3", 1], ["lt40", "0-2", "3", 0], ["ge40", "6-8", "3", 1], ["ge40", "0-2", "2", 0], ["premeno", "0-2", "3", 1], ["premeno", "0-2", "1", 0]];

  function voteWidget(root) {
    const ctrl = control(root, `
      <label>rules <select class="tmv-sel" name="set">
        <option value="t19">book example: R1 = R2, R3 (Table 1.9)</option>
        <option value="t110">book example: R1, R2, R3 (Table 1.10)</option>
        <option value="learned">learned from 200 patients (Step 6)</option>
      </select></label>`);
    const pick = control(root, PROPS.map((p, i) => `<label>${p.name} <select class="tmv-sel" name="p${i}">${p.vals.map((v, j) => `<option value="${j}">${v}</option>`).join("")}</select></label>`).join(""));
    const quick = control(root, `<span>book patients:</span>${BOOK_PATIENTS.map((_, i) => `<button data-b="${i}">#${i + 1}</button>`).join("")}<button data-b="test">next test patient ▶</button>`);
    const cols = document.createElement("div");
    cols.className = "tmv-cols";
    root.appendChild(cols);
    const BW = narrow(root) ? 340 : 560;
    const bar = svg(BW, 92);
    bar.style.marginTop = "10px";
    root.appendChild(bar);
    const out = readout(root);
    hint(root, "Filled dot = the rule's condition is True for this patient, so it casts its votes (+2 or −2 means two rules with the same condition). v = votes for Recurrence − votes against. v &gt; 0 predicts Recurrence; v ≤ 0 predicts No recurrence.");

    const setSel = ctrl.querySelector("[name=set]");
    const sels = PROPS.map((_, i) => pick.querySelector(`[name=p${i}]`));
    let label = null, testPos = -1;

    function ruleSet() {
      if (setSel.value === "t19") return { for: [{ lits: [DEG3, NF + LT40], count: 2 }], against: [{ lits: [N02], count: 1 }] };
      if (setSel.value === "t110") return { for: [{ lits: [DEG3, NF + LT40], count: 1 }, { lits: [DEG3], count: 1 }], against: [{ lits: [N02], count: 1 }] };
      const { tm } = getDefaultModel();
      const g = (c) => groupRules(Array.from({ length: tm.n }, (_, j) => tm.rule(c, j))).sort((a, b) => b.count - a.count);
      return { for: g(1), against: g(0) };
    }

    function setPatient(p, lab) {
      p.forEach((v, i) => { sels[i].value = String(v); });
      label = lab;
      draw();
    }

    function draw() {
      const c = colors();
      const p = sels.map((s) => +s.value);
      const lits = literals(features(p));
      const rs = ruleSet();
      let vFor = 0, vAg = 0;
      const col = (title, list, role, sign) => {
        let html = `<h4 style="color: var(--viz-${role})">${title}</h4>`;
        for (const r of list) {
          const on = firesOn(r.lits, lits, false);
          if (on) { if (sign > 0) vFor += r.count; else vAg += r.count; }
          const e = english(r.lits);
          const raw = r.lits.map(litName).join(" AND ");
          html += `<div class="tmv-rule${on ? " on" : ""}" style="--role: var(--viz-${role})" title="${raw}"><span class="tmv-dot"></span><span class="tmv-n">${sign > 0 ? "+" : "−"}${r.count}</span><span class="tmv-txt">IF ${e.text}${e.never ? ' <span class="tmv-sub">(can never be True)</span>' : ""}</span></div>`;
        }
        return `<div class="tmv-col">${html}</div>`;
      };
      cols.innerHTML = col("Rules voting FOR Recurrence", rs.for, "grad", 1) + col("Rules voting AGAINST (for No recurrence)", rs.against, "opt", -1);
      const v = vFor - vAg;
      // vote bar
      bar.innerHTML = "";
      const maxV = Math.max(2, rs.for.reduce((a, r) => a + r.count, 0), rs.against.reduce((a, r) => a + r.count, 0));
      const x = linear(-maxV, maxV, 30, BW - 30);
      el("rect", { x: x(-maxV), y: 26, width: x(maxV) - x(-maxV), height: 18, fill: c.grid, rx: 4 }, bar);
      el("rect", { x: Math.min(x(0), x(v)), y: 26, width: Math.abs(x(v) - x(0)), height: 18, fill: c.loss, "fill-opacity": 0.85, rx: 3 }, bar);
      el("line", { x1: x(0), x2: x(0), y1: 20, y2: 50, stroke: c.text, "stroke-width": 1.5 }, bar);
      for (const t of ticks(-maxV, maxV, maxV > 10 ? 5 : maxV > 4 ? (BW < 400 ? 4 : 2) : 1)) el("text", { x: x(t), y: 62, "text-anchor": "middle" }, bar).textContent = t > 0 ? "+" + t : t;
      el("text", { x: 30, y: 16 }, bar).textContent = "← No recurrence";
      el("text", { x: BW - 30, y: 16, "text-anchor": "end" }, bar).textContent = "Recurrence →";
      el("text", { x: x(v), y: 84, "text-anchor": "middle", style: "fill: var(--viz-loss); font-weight: 700; font-size: 13px" }, bar).textContent = `v = ${v > 0 ? "+" : ""}${v}`;
      const pred = v > 0 ? "Recurrence" : "No recurrence";
      const same = PATIENTS.filter((pt) => pt.p.every((q, i) => q === p[i]));
      const rec = same.filter((pt) => pt.y === 1).length;
      let text = `votes for Recurrence ${vFor} · votes against ${vAg} · v = ${v > 0 ? "+" : ""}${v} → predicts ${pred}`;
      if (label !== null) text += ` · true label: ${label ? "Recurrence" : "No recurrence"} ${(v > 0 ? 1 : 0) === label ? "✓" : "✗"}`;
      text += same.length ? `\nIn all 286 patients, ${same.length} have exactly this profile and ${rec} of them (${Math.round((100 * rec) / same.length)}%) had a recurrence.` : "\nNo patient in the dataset has exactly this profile.";
      out.textContent = text;
    }

    setSel.addEventListener("change", draw);
    sels.forEach((s) => s.addEventListener("change", () => { label = null; draw(); }));
    quick.addEventListener("click", (e) => {
      const b = e.target.dataset.b;
      if (b === undefined) return;
      if (b === "test") {
        const { test } = getDefaultModel();
        testPos = (testPos + 1) % test.length;
        setPatient(test[testPos].p, test[testPos].y);
        out.textContent = `test patient ${testPos + 1} of ${test.length}\n` + out.textContent;
        return;
      }
      const [m, n, d, y] = BOOK_PATIENTS[+b];
      setPatient([PROPS[0].vals.indexOf(m), PROPS[1].vals.indexOf(n), PROPS[2].vals.indexOf(d)], y);
    });
    onThemeChange(draw);
    setPatient([1, 1, 2], 1);                          // book patient #1
  }

  // ---------------------------------------------------------------- Widget 2: Voting Margin explorer
  function marginWidget(root) {
    const small = narrow(root);
    const W = small ? 340 : 560, H = small ? 290 : 290, M = { l: 44, r: 14, t: 30, b: 72 };
    const VMAX = 12;
    const chart = svg(W, H);
    chart.style.cursor = "ew-resize";
    chart.style.touchAction = "none";
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>Voting Margin <b>T</b> <input type="range" name="T" min="1" max="10" step="1" value="2"> <span class="val"></span></label>
      <label>vote sum <b>v</b> <input type="range" name="v" min="-${VMAX}" max="${VMAX}" step="1" value="1"> <span class="val"></span></label>
      <label>rules per class <input type="range" name="n" min="1" max="20" step="1" value="10"> <span class="val"></span></label>`);
    const out = readout(root);
    hint(root, "Drag on the chart or use the sliders. Each rule is picked for feedback independently with the probability shown, so on average p × (number of rules) rules learn from this patient.");
    const tIn = ctrl.querySelector("[name=T]"), vIn = ctrl.querySelector("[name=v]"), nIn = ctrl.querySelector("[name=n]");
    const x = linear(-VMAX, VMAX, M.l, W - M.r), y = linear(0, 1, H - M.b, M.t);
    const pRec = (v, T) => (T - Math.max(-T, Math.min(T, v))) / (2 * T);
    const pNon = (v, T) => (T + Math.max(-T, Math.min(T, v))) / (2 * T);

    function draw() {
      const c = colors(), T = +tIn.value, v = +vIn.value, n = +nIn.value;
      ctrl.querySelectorAll(".val").forEach((s) => { const i = s.parentElement.querySelector("input"); s.textContent = (i.name === "v" && +i.value > 0 ? "+" : "") + i.value; });
      chart.innerHTML = "";
      el("rect", { x: x(-T), y: M.t, width: x(T) - x(-T), height: y(0) - M.t, fill: c.loss, "fill-opacity": 0.08 }, chart);
      axes(chart, x, y, { xDomain: [-VMAX, VMAX], yDomain: [0, 1], xTicks: ticks(-VMAX, VMAX, small ? 6 : 3), yTicks: [0, 0.25, 0.5, 0.75, 1], xLabel: "vote sum v (votes for Recurrence − votes against)" });
      el("text", { x: M.l, y: 16, style: textStyle + "; font-size: 12px" }, chart).textContent = "probability that a rule gets feedback";
      for (const s of [-1, 1]) {
        el("line", { x1: x(s * T), x2: x(s * T), y1: M.t, y2: y(0), stroke: c.loss, "stroke-dasharray": "4 3", "stroke-width": 1.5 }, chart);
        el("text", { x: x(s * T), y: M.t - 4, "text-anchor": "middle", style: "fill: var(--viz-loss); font-weight: 700" }, chart).textContent = s > 0 ? "+T" : "−T";
      }
      const line = (f, color, dash) => {
        let d = "";
        for (let t = -VMAX; t <= VMAX; t++) d += (t === -VMAX ? "M" : "L") + x(t).toFixed(1) + "," + y(f(t, T)).toFixed(1);
        el("path", { d, fill: "none", stroke: color, "stroke-width": 2.5, "stroke-dasharray": dash || "none" }, chart);
      };
      line(pRec, c.grad);
      line(pNon, c.opt, "7 4");
      el("line", { x1: x(v), x2: x(v), y1: M.t, y2: y(0), stroke: c.text, "stroke-width": 1 }, chart);
      el("circle", { cx: x(v), cy: y(pRec(v, T)), r: 6, fill: c.grad, stroke: c.bg, "stroke-width": 1.5 }, chart);
      el("circle", { cx: x(v), cy: y(pNon(v, T)), r: 6, fill: c.opt, stroke: c.bg, "stroke-width": 1.5 }, chart);
      // legend
      const lx = M.l, ly = H - 16, lx2 = small ? M.l + 150 : M.l + 200;
      el("line", { x1: lx, x2: lx + 22, y1: ly, y2: ly, stroke: c.grad, "stroke-width": 2.5 }, chart);
      el("text", { x: lx + 28, y: ly + 4, style: textStyle }, chart).textContent = small ? "WITH recurrence" : "patient WITH recurrence";
      el("line", { x1: lx2, x2: lx2 + 22, y1: ly, y2: ly, stroke: c.opt, "stroke-width": 2.5, "stroke-dasharray": "7 4" }, chart);
      el("text", { x: lx2 + 28, y: ly + 4, style: textStyle }, chart).textContent = small ? "WITHOUT recurrence" : "patient WITHOUT recurrence";

      const a = pRec(v, T), b = pNon(v, T), vs = (v > 0 ? "+" : "") + v;
      const own = (u) => (u >= T ? `already at the margin (${u > T ? "past it" : "exactly"})` : `${T - Math.max(-T, u)} short of the margin`);
      out.textContent =
        `WITH recurrence, v = ${vs}: votes for its own class minus against = ${vs}, ${own(v)} → each rule gets feedback with probability ${fmt(a, 3)} (≈ ${fmt(a * 2 * n, 1)} of ${2 * n} rules)\n` +
        `   Recurrence rules: Recognize (if they fired) or Erase (if not) · No-recurrence rules that fired: Reject\n` +
        `WITHOUT recurrence, v = ${vs}: own class minus against = ${-v > 0 ? "+" : ""}${-v}, ${own(-v)} → probability ${fmt(b, 3)} (≈ ${fmt(b * 2 * n, 1)} of ${2 * n} rules)\n` +
        `   No-recurrence rules: Recognize or Erase · Recurrence rules that fired: Reject`;
    }

    function dragTo(e) {
      const b = chart.getBoundingClientRect();
      const v = Math.round(x.invert(((e.clientX - b.left) / b.width) * W));
      vIn.value = String(Math.max(-VMAX, Math.min(VMAX, v)));
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

  // ---------------------------------------------------------------- Widget 3: two rules, x1 OR x2 (book Ch. 2, §2.3)
  // Two rules, both voting for Y. Literals x1 and x2 only. Memory positions 1–4 (3–4 = Memorized).
  // Memorize Value 1.0, Forget Value 1/s. Label noise 10%. Feedback probability with Voting Margin T:
  //   Y observed: (T − v)/T,  Y-bar observed: v/T,  where v = number of rules that fire (capped at T).
  function orGame(seed, s, T) {
    const rand = mulberry32(seed), half = 2;
    const mem = [[half, half], [half, half]];
    const st = { t: 0, mem, last: null, occ: new Map() };
    st.step = function () {
      const x = [rand() < 0.5 ? 1 : 0, rand() < 0.5 ? 1 : 0];
      let y = x[0] || x[1] ? 1 : 0;
      if (rand() < 0.1) y = 1 - y;
      const fire = mem.map((r) => (r[0] <= half || x[0]) && (r[1] <= half || x[1]) ? 1 : 0);
      let p = 1, v = fire[0] + fire[1];
      if (T > 0) { v = Math.min(T, v); p = y === 1 ? (T - v) / T : v / T; }
      const fb = [0, 0];
      for (let j = 0; j < 2; j++) {
        if (rand() >= p) continue;
        fb[j] = 1;
        const r = mem[j];
        if (y === 1) {
          for (let k = 0; k < 2; k++) {
            if (fire[j] && x[k]) { if (r[k] < 2 * half) r[k]++; }
            else if (rand() < 1 / s && r[k] > 1) r[k]--;
          }
        } else if (fire[j]) {
          for (let k = 0; k < 2; k++) if (!x[k] && r[k] <= half) r[k]++;
        }
      }
      st.t++;
      st.last = { x, y, fire, p, fb };
      const key = ruleName(mem[0]) + "|" + ruleName(mem[1]);
      st.occ.set(key, (st.occ.get(key) || 0) + 1);
    };
    return st;
  }
  const ruleName = (r) => (r[0] > 2 && r[1] > 2 ? "x1 AND x2" : r[0] > 2 ? "x1" : r[1] > 2 ? "x2" : "∅");
  function shares(st) {
    let solved = 0, dup = 0;
    for (const [k, n] of st.occ) {
      if (k === "x1|x2" || k === "x2|x1") solved += n;
      if (k === "x1 AND x2|x1 AND x2") dup += n;
    }
    return { solved: solved / st.t, dup: dup / st.t, other: 1 - (solved + dup) / st.t };
  }

  function orWidget(root) {
    const S = 4, SEED = 42, TOTAL = 200000;
    const small = narrow(root);
    const W = small ? 340 : 560, H = small ? 390 : 205;
    const chart = svg(W, H);
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>Voting Margin <select class="tmv-sel" name="T"><option value="0" selected>off (feedback every time)</option><option value="1">T = 1</option><option value="2">T = 2</option><option value="3">T = 3</option><option value="4">T = 4</option></select></label>
      <button data-a="step">Step</button><button data-a="play">▶ Play</button><button data-a="ff">⏩ to ${TOTAL.toLocaleString("en")} steps</button><button data-a="reset">Reset</button>`);
    const out = readout(root);
    hint(root, `Class Y = x1 OR x2, with 10% of labels flipped. Both rules vote for Y. Memory depth 4: positions 1–4; 3 and 4 (green band) are Memorized. Memorize Value 1.0, Forget Value 1/s = 1/${S}. Seed ${SEED}. The bars show the share of all steps so far spent in each kind of rule pair.`);
    const tSel = ctrl.querySelector("[name=T]"), playBtn = ctrl.querySelector("[data-a=play]");
    let st, timer = null, ff = null;

    function reset() { stop(); st = orGame(SEED, S, +tSel.value); draw(); }
    function stop() {
      clearInterval(timer); timer = null; playBtn.textContent = "▶ Play";
      if (ff) { clearTimeout(ff); ff = null; }
    }

    function drawRule(g0x, g0y, j) {
      const c = colors(), r = st.mem[j];
      const cw = small ? 52 : 46, ch = 26;
      el("text", { x: g0x, y: g0y - 8, style: textStyle + "; font-weight: 700" }, chart).textContent = `Rule ${j === 0 ? "One" : "Two"}`;
      const y0 = g0y + 6;
      el("rect", { x: g0x, y: y0, width: cw * 2 + 50, height: ch * 2, fill: c.opt, "fill-opacity": 0.1 }, chart);
      for (let pos = 4; pos >= 1; pos--) {
        const yy = y0 + (4 - pos) * ch;
        el("text", { x: g0x + 44, y: yy + ch / 2 + 4, "text-anchor": "end" }, chart).textContent = pos;
        for (let k = 0; k < 2; k++) {
          const xx = g0x + 50 + k * cw;
          const here = r[k] === pos;
          el("rect", { x: xx + 3, y: yy + 3, width: cw - 6, height: ch - 6, rx: 4, fill: here ? (pos > 2 ? c.opt : c.muted) : c.bg, stroke: c.border }, chart);
        }
      }
      el("line", { x1: g0x, x2: g0x + 50 + 2 * cw, y1: y0 + 2 * ch, y2: y0 + 2 * ch, stroke: c.text, "stroke-dasharray": "3 3" }, chart);
      for (let k = 0; k < 2; k++) el("text", { x: g0x + 50 + k * cw + cw / 2, y: y0 + 4 * ch + 14, "text-anchor": "middle", style: textStyle }, chart).textContent = `x${k + 1}`;
      el("text", { x: g0x, y: y0 + 4 * ch + 32, style: "fill: var(--viz-model); font-weight: 700" }, chart).textContent = `IF ${ruleName(r) === "∅" ? "(empty)" : ruleName(r)} THEN Y`;
    }

    function draw() {
      const c = colors();
      chart.innerHTML = "";
      if (small) { drawRule(14, 24, 0); drawRule(180, 24, 1); }
      else { drawRule(14, 24, 0); drawRule(170, 24, 1); }
      // share bars
      const bx = small ? 14 : 340, by = small ? 220 : 30, bw = small ? 312 : 200;
      el("text", { x: bx, y: by + 2, style: textStyle + "; font-weight: 700" }, chart).textContent = "share of steps so far";
      const sh = st.t ? shares(st) : { solved: 0, dup: 0, other: 0 };
      const rows = [["solved: one rule x1, the other x2", sh.solved, c.opt], ["both rules x1 AND x2", sh.dup, c.grad], ["anything else", sh.other, c.muted]];
      rows.forEach(([lab, val, col], i) => {
        const yy = by + 16 + i * 46;
        el("text", { x: bx, y: yy + 10 }, chart).textContent = lab;
        el("rect", { x: bx, y: yy + 16, width: bw, height: 14, fill: c.grid, rx: 3 }, chart);
        el("rect", { x: bx, y: yy + 16, width: bw * val, height: 14, fill: col, rx: 3 }, chart);
        el("text", { x: bx + bw + 6 > W - 40 ? bx + bw - 4 : bx + bw + 6, y: yy + 27, "text-anchor": bx + bw + 6 > W - 40 ? "end" : "start", style: textStyle }, chart).textContent = st.t ? (100 * val).toFixed(1) + "%" : "–";
      });
      let text = `step ${st.t.toLocaleString("en")} · Voting Margin ${+tSel.value ? "T = " + tSel.value : "off"}`;
      if (st.last) {
        const L = st.last;
        text += `\nlast example: x1 = ${L.x[0]}, x2 = ${L.x[1]}, label ${L.y ? "Y" : "not Y"} · rules firing: ${L.fire[0] + L.fire[1]} · feedback probability ${fmt(L.p, 2)} · rules updated: ${L.fb[0] + L.fb[1]}`;
      }
      out.textContent = text;
    }

    function fastForward() {
      stop();
      const chunk = () => {
        const end = Math.min(TOTAL, st.t + 20000);
        while (st.t < end) st.step();
        draw();
        if (st.t < TOTAL) ff = setTimeout(chunk, 0); else ff = null;
      };
      chunk();
    }

    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "step") { stop(); st.step(); draw(); }
      if (a === "reset") reset();
      if (a === "ff") fastForward();
      if (a === "play") {
        if (timer) { stop(); return; }
        playBtn.textContent = "⏸ Pause";
        timer = setInterval(() => { for (let i = 0; i < 5; i++) st.step(); draw(); }, 60);
      }
    });
    tSel.addEventListener("change", reset);
    onThemeChange(draw);
    reset();
  }

  // ---------------------------------------------------------------- Widget 4: train on the breast-cancer data
  function trainWidget(root) {
    const small = narrow(root);
    const W = small ? 340 : 560, H = 240, M = { l: 44, r: 14, t: 26, b: 38 };
    const chart = svg(W, H);
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>rules per class <input type="range" name="n" min="1" max="20" step="1" value="${DEFAULTS.n}"> <span class="val"></span></label>
      <label>Voting Margin <b>T</b> <input type="range" name="T" min="1" max="30" step="1" value="${DEFAULTS.T}"> <span class="val"></span></label>
      <label>specificity <b>s</b> <input type="range" name="s" min="1.5" max="8" step="0.5" value="${DEFAULTS.s}"> <span class="val"></span></label>
      <label>seed <input class="tmv-sel" type="number" name="seed" value="${DEFAULTS.seed}" min="1" max="999" style="width: 4.5em"></label>
      <button data-a="train">Train again</button><button data-a="defaults">Defaults</button>`);
    const out = readout(root);
    const cols = document.createElement("div");
    cols.className = "tmv-cols";
    root.appendChild(cols);
    hint(root, `Each run shuffles the 286 patients with the seed, trains on 200 and tests on the other 86, for ${DEFAULTS.epochs} epochs (memory depth ${DEFAULTS.depth}: positions 1–${DEFAULTS.depth}, Memorized above ${DEFAULTS.depth / 2}). Forget Value = 1/s, Memorize Value = 1 − 1/s. The numbers after each rule count the training patients it matches: with recurrence / without.`);
    const val = (name) => +ctrl.querySelector(`[name=${name}]`).value;
    const x = linear(1, DEFAULTS.epochs, M.l, W - M.r), y = linear(0.4, 0.9, H - M.b, M.t);
    let hist = [], run = null, status = "", base = 0, info = null;

    function start() {
      if (run) run.cancel = true;
      const cfg = { n: val("n"), T: val("T"), s: val("s"), seed: Math.max(1, Math.round(val("seed")) || 1) };
      const { train, test } = split(cfg.seed, DEFAULTS.nTrain);
      const tm = new TM(cfg.n, cfg.T, cfg.s, DEFAULTS.depth, cfg.seed);
      base = test.filter((d) => d.y === 0).length / test.length;
      hist = []; info = null;
      const me = { cancel: false };
      run = me;
      let e = 0;
      (function next() {
        if (me.cancel) return;
        tm.epoch(train);
        e++;
        hist.push([tm.accuracy(train), tm.accuracy(test)]);
        status = e < DEFAULTS.epochs ? `training… epoch ${e} / ${DEFAULTS.epochs}` : "";
        if (e === DEFAULTS.epochs) info = { tm, train, test, cfg };
        draw();
        if (e < DEFAULTS.epochs) setTimeout(next, 0);
      })();
    }

    function draw() {
      const c = colors();
      ctrl.querySelectorAll(".val").forEach((s) => { s.textContent = s.parentElement.querySelector("input").value; });
      chart.innerHTML = "";
      axes(chart, x, y, { xDomain: [1, DEFAULTS.epochs], yDomain: [0.4, 0.9], xTicks: [1, 10, 20, 30, 40, 50, 60], yTicks: ticks(0.4, 0.9, 0.1).map((t) => +t.toFixed(1)), xLabel: "epoch" });
      el("text", { x: M.l, y: 15, style: textStyle + "; font-size: 12px" }, chart).textContent = "accuracy";
      el("line", { x1: x(1), x2: x(DEFAULTS.epochs), y1: y(base), y2: y(base), stroke: c.muted, "stroke-dasharray": "5 4", "stroke-width": 1.5 }, chart);
      [[0, c.model, "train"], [1, c.data, "test"]].forEach(([i, col, lab]) => {
        if (!hist.length) return;
        const d = hist.map((h, e) => (e ? "L" : "M") + x(e + 1).toFixed(1) + "," + y(Math.max(0.4, h[i])).toFixed(1)).join("");
        el("path", { d, fill: "none", stroke: col, "stroke-width": 2, "stroke-linejoin": "round" }, chart);
        const last = hist[hist.length - 1][i];
        el("circle", { cx: x(hist.length), cy: y(Math.max(0.4, last)), r: 3.5, fill: col }, chart);
      });
      const lx = M.l + 8, ly = M.t + 6;
      el("line", { x1: lx, x2: lx + 18, y1: ly, y2: ly, stroke: c.model, "stroke-width": 2.5 }, chart);
      el("text", { x: lx + 22, y: ly + 4, style: textStyle }, chart).textContent = "train (200)";
      el("line", { x1: lx + 100, x2: lx + 118, y1: ly, y2: ly, stroke: c.data, "stroke-width": 2.5 }, chart);
      el("text", { x: lx + 122, y: ly + 4, style: textStyle }, chart).textContent = "test (86)";
      el("line", { x1: lx + (small ? 170 : 190), x2: lx + (small ? 188 : 208), y1: ly, y2: ly, stroke: c.muted, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, chart);
      el("text", { x: lx + (small ? 192 : 212), y: ly + 4, style: textStyle }, chart).textContent = small ? "always 'no'" : "always 'no recurrence' (test)";
      if (status) el("text", { x: (M.l + W - M.r) / 2, y: (M.t + H - M.b) / 2, "text-anchor": "middle", style: textStyle }, chart).textContent = status;

      if (!info) { out.textContent = status || " "; cols.innerHTML = ""; return; }
      const { tm, train, test } = info;
      const h = hist[hist.length - 1];
      let tp = 0, fn = 0, fp = 0, tn = 0;
      for (const d of test) { const p = tm.predict(d.lits); if (d.y) p ? tp++ : fn++; else p ? fp++ : tn++; }
      const tail = hist.slice(10).map((q) => q[1]);
      out.textContent =
        `after epoch ${DEFAULTS.epochs}: train ${fmt(h[0], 3)} · test ${fmt(h[1], 3)} (${tp + tn} of ${test.length} right) · always 'no recurrence' would score ${fmt(base, 3)}\n` +
        `test patients with recurrence caught: ${tp} of ${tp + fn} · false alarms: ${fp} of ${fp + tn} · test accuracy over epochs 11–${DEFAULTS.epochs}: ${fmt(Math.min(...tail), 3)} to ${fmt(Math.max(...tail), 3)}`;
      const col = (c, title, role, sign) => {
        const g = groupRules(Array.from({ length: tm.n }, (_, j) => tm.rule(c, j))).sort((a, b) => b.count - a.count);
        let html = `<h4 style="color: var(--viz-${role})">${title}</h4>`;
        for (const r of g) {
          const own = train.filter((d) => d.y === 1 && firesOn(r.lits, d.lits, false)).length;
          const oth = train.filter((d) => d.y === 0 && firesOn(r.lits, d.lits, false)).length;
          const e = english(r.lits);
          html += `<div class="tmv-rule on" style="--role: var(--viz-${role})" title="${r.lits.map(litName).join(" AND ")}"><span class="tmv-n">${sign}${r.count}</span><span class="tmv-txt">IF ${e.text}${e.never ? ' <span class="tmv-sub">(can never be True)</span>' : ""} <span class="tmv-sub">· ${own} / ${oth}</span></span></div>`;
        }
        return `<div class="tmv-col">${html}</div>`;
      };
      cols.innerHTML = col(1, "Learned rules FOR Recurrence", "grad", "+") + col(0, "Learned rules AGAINST (No recurrence)", "opt", "−");
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
        for (const k of ["n", "T", "s", "seed"]) ctrl.querySelector(`[name=${k}]`).value = DEFAULTS[k];
        start();
      }
    });
    onThemeChange(draw);
    status = "scroll here to train";
    draw();
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) { io.disconnect(); start(); }
      }, { rootMargin: "200px" });
      io.observe(root);
    } else start();
  }

  const mounts = { "viz-vote": voteWidget, "viz-margin": marginWidget, "viz-orgame": orWidget, "viz-train": trainWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
