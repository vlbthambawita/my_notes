// Interactive widgets for the "Hierarchical and Graph Tsetlin Machines" lesson.
// The learning code (GTM below) is a line-by-line port of the lesson's Python check script, with the same
// seeded random numbers, so the widget reproduces the numbers quoted in the lesson exactly.

const GTM = (function () {
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), a | 1);
      t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const ELEMENTS = ["C", "N", "O"];

  // A random tree-shaped "molecule" of 6–8 atoms: all carbon except one N and one O, each atom at most 3 bonds.
  function makeMolecule(rand) {
    const n = 6 + Math.floor(rand() * 3);
    const deg = new Array(n).fill(0), edges = [];
    for (let k = 1; k < n; k++) {
      let p = Math.floor(rand() * k);
      while (deg[p] >= 3) p = Math.floor(rand() * k);
      deg[p]++; deg[k]++;
      edges.push([p, k]);
    }
    const el = new Array(n).fill(0);
    const iN = Math.floor(rand() * n);
    let iO = Math.floor(rand() * (n - 1));
    if (iO >= iN) iO++;
    el[iN] = 1; el[iO] = 2;
    return withNbr({ el, edges });
  }
  function withNbr(g) {
    g.nbr = g.el.map(() => []);
    for (const [a, b] of g.edges) { g.nbr[a].push(b); g.nbr[b].push(a); }
    return g;
  }
  function distances(g, s) {
    const d = g.el.map(() => -1);
    d[s] = 0;
    const q = [s];
    for (let i = 0; i < q.length; i++) for (const w of g.nbr[q[i]]) if (d[w] < 0) { d[w] = d[q[i]] + 1; q.push(w); }
    return d;
  }
  // 1 if the N and the O are at most TASKS[task] bonds apart ("bond": they are bonded), else 0
  const TASKS = { bond: 1, two: 2, three: 3 };
  function label(g, task) {
    const d = distances(g, g.el.indexOf(1));
    return d[g.el.indexOf(2)] <= TASKS[task] ? 1 : 0;
  }
  function makeDataset(n, rand, task) {
    const G = [], y = [];
    for (let i = 0; i < n; i++) {
      const cls = rand() < 0.5 ? 0 : 1;
      let g;
      do { g = makeMolecule(rand); } while (label(g, task) !== cls);
      G.push(g); y.push(cls);
    }
    return { G, y };
  }
  // What a standard TM sees: which elements occur and how often (thermometer: at least 1, 2, 3)
  function flatFeatures(g) {
    const f = [];
    for (let e = 0; e < 3; e++) { const c = g.el.filter((x) => x === e).length; for (const t of [1, 2, 3]) f.push(c >= t ? 1 : 0); }
    return f;
  }
  const asFlatGraph = (g) => ({ props: [flatFeatures(g)], nbr: [[]] });
  const asGraph = (g) => ({ props: g.el.map((e) => [0, 1, 2].map((k) => (e === k ? 1 : 0))), nbr: g.nbr });

  class Machine {
    // n rules per class (rules 0…n−1 vote for class 0), depth D. Memory positions 1 … 2·half; Memorized = upper half.
    constructor(P, n, D, T, forget, half, rand) {
      Object.assign(this, { P, n, D, T, F: forget, M: 1 - forget, N: half, rand });
      this.m = 2 * n;
      this.L0 = 2 * P;
      this.Lm = this.m;                 // message literals: "a neighbour sent R_k" (no negations, see the lesson)
      this.L = this.L0 + (D - 1) * this.Lm;
      this.state = Array.from({ length: this.m }, () => new Int16Array(this.L).fill(half));
      this.cls = Array.from({ length: this.m }, (_, j) => (j < n ? 0 : 1));
    }
    layerRange(i) { if (i === 0) return [0, this.L0]; const a = this.L0 + (i - 1) * this.Lm; return [a, a + this.Lm]; }
    partMatches(j, lit, i) {
      const s = this.state[j], N = this.N, [a, b] = this.layerRange(i);
      for (let k = a; k < b; k++) if (s[k] > N && !lit[k]) return false;
      return true;
    }
    // lits[v]: literal vector of node v over all layers; layers[i][j][v]: rule j matches parts 0…i at node v
    evaluate(G) {
      const nn = G.props.length, m = this.m;
      const lits = G.props.map((p) => p.concat(p.map((x) => 1 - x)));
      let M = [];
      for (let j = 0; j < m; j++) M.push(lits.map((l) => this.partMatches(j, l, 0)));
      const layers = [M];
      for (let i = 1; i < this.D; i++) {
        for (let v = 0; v < nn; v++) {
          const inbox = [];
          for (let k = 0; k < m; k++) inbox.push(G.nbr[v].some((u) => M[k][u]) ? 1 : 0);
          lits[v] = lits[v].concat(inbox);
        }
        const newM = [];
        for (let j = 0; j < m; j++) newM.push(lits.map((l, v) => M[j][v] && this.partMatches(j, l, i)));
        M = newM;
        layers.push(M);
      }
      return { lits, M, layers };
    }
    outputs(G) { return this.evaluate(G).M.map((row) => row.some(Boolean)); }
    vote(G) { let v = 0; this.outputs(G).forEach((o, j) => { if (o) v += this.cls[j] === 0 ? 1 : -1; }); return v; }
    predict(G) { return this.vote(G) >= 0 ? 0 : 1; }
    update(G, y, log) {
      const { rand, T, L, N } = this;
      const { lits, M } = this.evaluate(G), out = M.map((row) => row.some(Boolean));
      let v = 0;
      out.forEach((o, j) => { if (o) v += this.cls[j] === y ? 1 : -1; });
      const raw = v;
      v = Math.max(-T, Math.min(T, v));
      const prob = (T - v) / (2 * T);
      if (log) { log.v = raw; log.prob = prob; log.fb = []; }
      for (let j = 0; j < this.m; j++) {
        if (rand() > prob) { if (log) log.fb.push({ type: "none" }); continue; }
        const s = this.state[j];
        if (this.cls[j] === y) {
          if (out[j]) {                                   // Recognize Feedback at one random matching node
            const idx = []; M[j].forEach((b, q) => { if (b) idx.push(q); });
            const pick = idx[Math.floor(rand() * idx.length)], p = lits[pick];
            for (let k = 0; k < L; k++) {
              const u = rand();
              if (p[k]) { if (u <= this.M && s[k] < 2 * N) s[k]++; }
              else if (u <= this.F && s[k] > 1) s[k]--;
            }
            if (log) log.fb.push({ type: "recognize", node: pick, of: idx.length });
          } else {                                          // Erase Feedback: no node needed
            for (let k = 0; k < L; k++) if (rand() <= this.F && s[k] > 1) s[k]--;
            if (log) log.fb.push({ type: "erase" });
          }
        } else if (out[j]) {                                // Reject Feedback at one random matching node
          const idx = []; M[j].forEach((b, q) => { if (b) idx.push(q); });
          const pick = idx[Math.floor(rand() * idx.length)], p = lits[pick];
          for (let k = 0; k < L; k++) if (!p[k] && s[k] <= N) s[k]++;
          if (log) log.fb.push({ type: "reject", node: pick, of: idx.length });
        } else if (log) log.fb.push({ type: "none" });
      }
    }
    // memorized literals of rule j, per layer: layer 0 → [{name, neg}], layer i → [{rule k, neg}]
    parts(j) {
      const s = this.state[j], N = this.N, out = [];
      for (let i = 0; i < this.D; i++) {
        const [a, b] = this.layerRange(i), lits = [];
        if (i === 0) {
          const half = (b - a) / 2;
          for (let k = 0; k < half; k++) {
            if (s[a + k] > N) lits.push({ k, neg: false });
            if (s[a + half + k] > N) lits.push({ k, neg: true });
          }
        } else for (let k = 0; k < b - a; k++) if (s[a + k] > N) lits.push({ k, neg: false });
        out.push(lits);
      }
      return out;
    }
  }

  function accuracy(tm, Gs, y) { let ok = 0; Gs.forEach((g, i) => { if (tm.predict(g) === y[i]) ok++; }); return ok / Gs.length; }
  function shuffledOrder(n, rand) {
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const k = Math.floor(rand() * (i + 1)); [order[i], order[k]] = [order[k], order[i]]; }
    return order;
  }
  function experiment({ seed = 1, task = "bond", model = "graph", D = 2, n = 8, T = 4, forget = 0.8, half = 5, ntrain = 300, ntest = 300 } = {}) {
    const rand = mulberry32(seed);
    const tr = makeDataset(ntrain, rand, task), te = makeDataset(ntest, rand, task);
    const conv = model === "flat" ? asFlatGraph : asGraph;
    const Xtr = tr.G.map(conv), Xte = te.G.map(conv);
    const tm = new Machine(Xtr[0].props[0].length, n, model === "flat" ? 1 : D, T, forget, half, rand);
    return { rand, tr, te, Xtr, Xte, tm, model, history: [] };
  }
  function trainEpoch(ex) {
    for (const i of shuffledOrder(ex.Xtr.length, ex.rand)) ex.tm.update(ex.Xtr[i], ex.tr.y[i]);
    ex.history.push([accuracy(ex.tm, ex.Xtr, ex.tr.y), accuracy(ex.tm, ex.Xte, ex.te.y)]);
  }

  return { mulberry32, ELEMENTS, makeMolecule, withNbr, distances, label, makeDataset, flatFeatures, asFlatGraph, asGraph, Machine, accuracy, shuffledOrder, experiment, trainEpoch };
})();

if (typeof module !== "undefined") module.exports = GTM;

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";
  const pct = (a) => (100 * a).toFixed(1) + "%";
  const EL = GTM.ELEMENTS;

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
  const label = (g, x, y, txt, opts = {}) => {
    const t = el("text", Object.assign({ x, y, "text-anchor": "middle" }, opts), g);
    t.textContent = txt;
    return t;
  };
  const isNarrow = (root) => root.clientWidth < 480;
  function mark(ctrl, attr, value) {
    ctrl.querySelectorAll(`button[data-${attr}]`).forEach((b) => (b.style.borderColor = b.dataset[attr] === String(value) ? "var(--viz-model)" : ""));
  }

  // A small, deterministic force layout for a tree-shaped molecule. Returns [x, y] in [0, 1]².
  function layout(g) {
    if (g.pos) return g.pos;
    const n = g.el.length;
    // start from a BFS tree layout (depth → x, order → y), then relax with springs + repulsion
    const d = GTM.distances(g, 0), byDepth = {};
    const P = g.el.map((_, v) => { (byDepth[d[v]] = byDepth[d[v]] || []).push(v); return [0, 0]; });
    Object.keys(byDepth).forEach((k) => byDepth[k].forEach((v, i, arr) => { P[v] = [+k, i - (arr.length - 1) / 2]; }));
    for (let it = 0; it < 400; it++) {
      const F = P.map(() => [0, 0]);
      for (let a = 0; a < n; a++) for (let b = a + 1; b < n; b++) {
        const dx = P[a][0] - P[b][0], dy = P[a][1] - P[b][1], r2 = Math.max(0.01, dx * dx + dy * dy), f = 0.6 / r2;
        F[a][0] += f * dx; F[a][1] += f * dy; F[b][0] -= f * dx; F[b][1] -= f * dy;
      }
      for (const [a, b] of g.edges) {
        const dx = P[b][0] - P[a][0], dy = P[b][1] - P[a][1], r = Math.sqrt(dx * dx + dy * dy) || 1, f = 0.5 * (r - 1);
        F[a][0] += (f * dx) / r; F[a][1] += (f * dy) / r; F[b][0] -= (f * dx) / r; F[b][1] -= (f * dy) / r;
      }
      for (let v = 0; v < n; v++) { P[v][0] += 0.1 * F[v][0]; P[v][1] += 0.1 * F[v][1]; }
    }
    // centre, then rotate so that the long axis is horizontal (the boxes are wider than tall)
    const mx = P.reduce((a, p) => a + p[0], 0) / n, my = P.reduce((a, p) => a + p[1], 0) / n;
    let sxx = 0, syy = 0, sxy = 0;
    P.forEach(([x, y]) => { sxx += (x - mx) ** 2; syy += (y - my) ** 2; sxy += (x - mx) * (y - my); });
    const th = 0.5 * Math.atan2(2 * sxy, sxx - syy), cs = Math.cos(th), sn = Math.sin(th);
    g.pos = P.map(([x, y]) => [(x - mx) * cs + (y - my) * sn, -(x - mx) * sn + (y - my) * cs]);
    return g.pos;
  }

  // Draw a molecule into box (x, y, w, h). opts: {r, hl: Set of atoms to ring green, dim: Set to fade,
  // badges: atom → [text], onClick(v), arrows: [[from, to]]}
  function drawMolecule(g, mol, box, opts = {}) {
    const c = colors(), pos = layout(mol), r = opts.r || 16;
    // uniform scale (bonds keep their length in every direction), at most 4.2 atom radii per bond
    const xs = pos.map((p) => p[0]), ys = pos.map((p) => p[1]);
    const xr = Math.max(...xs) - Math.min(...xs), yr = Math.max(...ys) - Math.min(...ys);
    const pad = r + 22;                                      // room for the badges under the atoms
    const k = Math.min((box.w - 2 * pad) / Math.max(xr, 1e-6), (box.h - 2 * pad) / Math.max(yr, 1e-6), 4.2 * r);
    const cx = box.x + box.w / 2 - k * (Math.max(...xs) + Math.min(...xs)) / 2, cy = box.y + box.h / 2 - k * (Math.max(...ys) + Math.min(...ys)) / 2;
    const px = (v) => [cx + k * pos[v][0], cy + k * pos[v][1]];
    for (const [a, b] of mol.edges) {
      const [x1, y1] = px(a), [x2, y2] = px(b);
      el("line", { x1, y1, x2, y2, stroke: c.muted, "stroke-width": 3, "stroke-opacity": 0.6 }, g);
    }
    (opts.arrows || []).forEach(([a, b]) => {
      const [x1, y1] = px(a), [x2, y2] = px(b), dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
      const ux = dx / L, uy = dy / L, ox = -uy * 5, oy = ux * 5;            // offset so both directions are visible
      const sx = x1 + ux * (r + 2) + ox, sy = y1 + uy * (r + 2) + oy, ex = x2 - ux * (r + 4) + ox, ey = y2 - uy * (r + 4) + oy;
      el("line", { x1: sx, y1: sy, x2: ex, y2: ey, stroke: c.model, "stroke-width": 2.5 }, g);
      el("path", { d: `M${ex},${ey} L${ex - ux * 8 - uy * 4.5},${ey - uy * 8 + ux * 4.5} L${ex - ux * 8 + uy * 4.5},${ey - uy * 8 - ux * 4.5} Z`, fill: c.model }, g);
    });
    mol.el.forEach((e, v) => {
      const [x, y] = px(v), dim = opts.dim && opts.dim.has(v);
      el("circle", { cx: x, cy: y, r: r + 1, fill: c.bg }, g);           // hides the bond lines under faded atoms
      const grp = el("g", { opacity: dim ? 0.3 : 1 }, g);
      if (opts.hl && opts.hl.has(v)) el("circle", { cx: x, cy: y, r: r + 5, fill: c.opt, "fill-opacity": 0.25, stroke: c.opt, "stroke-width": 2.5 }, grp);
      const circ = el("circle", { cx: x, cy: y, r, fill: c.bg, stroke: e === 0 ? c.muted : c.data, "stroke-width": e === 0 ? 1.5 : 3 }, grp);
      label(grp, x, y + 5, EL[e], { style: `fill: ${e === 0 ? c.text : c.data}; font-weight: 700; font-size: ${Math.round(r * 0.9)}px; pointer-events: none` });
      if (opts.sel === v) el("circle", { cx: x, cy: y, r: r + 2, fill: "none", stroke: c.model, "stroke-width": 2, "stroke-dasharray": "4 3" }, grp);
      const b = opts.badges && opts.badges[v];
      if (b && b.length) {
        const t = b.join(" ");
        const bw = 7 * t.length + 8;
        el("rect", { x: x - bw / 2, y: y + r + 3, width: bw, height: 15, rx: 4, fill: c.model, "fill-opacity": 0.18, stroke: c.model }, grp);
        label(grp, x, y + r + 14, t, { style: `fill: ${c.text}; font-size: 10.5px; font-weight: 600; pointer-events: none` });
      }
      if (opts.onClick) { circ.style.cursor = "pointer"; circ.addEventListener("click", () => opts.onClick(v)); }
    });
    return px;
  }
  function mol(el, edges) { return GTM.withNbr({ el: el.slice(), edges: edges.map((e) => e.slice()) }); }
  function nOBonded(m) { return m.edges.some(([a, b]) => (m.el[a] === 1 && m.el[b] === 2) || (m.el[a] === 2 && m.el[b] === 1)); }
  function counts(m) { return [0, 1, 2].map((e) => m.el.filter((x) => x === e).length); }

  // ---------------------------------------------------------------- Widget 1: same atoms, different bonds
  function flatWidget(root) {
    const EDGES = [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5]];
    const PRESET = () => [mol([0, 0, 1, 2, 0, 0], EDGES), mol([0, 0, 1, 0, 0, 2], EDGES)];
    const ctrl = control(root, `<button data-a="book">Two molecules, same atoms</button><button data-a="random">Random pair</button>`);
    hint(root, "Click an atom to change it: C → N → O → C. Each molecule is described twice: as a <b>flat</b> list of features (what a standard Tsetlin machine sees) and as a <b>graph</b> of atoms and bonds.");
    const out = readout(root);
    let mols = PRESET(), seed = 1, s = null;

    function randomPair() {
      const rand = GTM.mulberry32(seed++);
      for (;;) {
        const a = GTM.makeMolecule(rand), b = mol(a.el, a.edges);
        // move the O to another carbon so that N–O bonded-ness differs
        const iO = b.el.indexOf(2), cs = b.el.map((e, v) => (e === 0 ? v : -1)).filter((v) => v >= 0);
        const pick = cs[Math.floor(rand() * cs.length)];
        b.el[iO] = 0; b.el[pick] = 2;
        if (nOBonded(a) !== nOBonded(b)) return nOBonded(a) ? [a, b] : [b, a];
      }
    }
    function draw() {
      if (s) s.remove();
      const c = colors(), narrow = isNarrow(root);
      const W = narrow ? 340 : 620, colW = narrow ? W : W / 2, molH = narrow ? 210 : 180, tabH = 82, block = molH + tabH + 18;
      const H = narrow ? 2 * block : molH + tabH + 6;
      s = svg(W, H);
      s.style.maxWidth = W + "px"; s.style.margin = "0 auto"; s.style.display = "block";
      root.insertBefore(s, ctrl);
      mols.forEach((m, i) => {
        const x0 = narrow ? 0 : i * colW, y0 = narrow ? i * block : 0;
        label(s, x0 + colW / 2, y0 + 16, `Molecule ${"AB"[i]}`, { style: textStyle + "; font-weight: 600" });
        const bonded = nOBonded(m);
        const hl = new Set();
        if (bonded) m.el.forEach((e, v) => { if (e === 1 && m.nbr[v].some((u) => m.el[u] === 2)) hl.add(v); });
        drawMolecule(s, m, { x: x0 + 30, y: y0 + 26, w: colW - 60, h: molH - 30 }, { r: narrow ? 13 : 15, hl, onClick: (v) => { m.el[v] = (m.el[v] + 1) % 3; draw(); } });
        // flat features table
        const ty = y0 + molH + 8, cnt = counts(m), cw = (colW - 40) / 3;
        label(s, x0 + colW / 2, ty + 2, "flat features (counts)", { style: "font-size: 11px" });
        cnt.forEach((k, e) => {
          const x = x0 + 20 + e * cw;
          el("rect", { x: x + 4, y: ty + 10, width: cw - 8, height: 26, rx: 4, fill: c.data, "fill-opacity": 0.1, stroke: c.data }, s);
          label(s, x + cw / 2, ty + 28, `${EL[e]} × ${k}`, { style: textStyle + "; font-weight: 600" });
        });
        el("rect", { x: x0 + 24, y: ty + 46, width: colW - 48, height: 30, rx: 6, fill: bonded ? c.opt : c.bg, "fill-opacity": bonded ? 0.15 : 1, stroke: c.model, "stroke-width": 2 }, s);
        label(s, x0 + colW / 2, ty + 66, `graph rule "N bonded to O": ${bonded ? "True" : "False"}`, { style: `fill: ${bonded ? c.opt : c.text}; font-weight: 600; font-size: 12px` });
      });
      const same = counts(mols[0]).join() === counts(mols[1]).join();
      const b = mols.map(nOBonded);
      out.textContent = (same
        ? "Flat features: identical (" + counts(mols[0]).map((k, e) => `${k} ${EL[e]}`).join(", ") + "). Any rule built from them gives A and B the same answer."
        : "Flat features: different now, so a flat rule could tell these two apart, but only by counting atoms, not by looking at bonds.") +
        `\nGraph rule "an N with an O neighbour": A ${b[0] ? "True" : "False"} · B ${b[1] ? "True" : "False"}` +
        (same && b[0] !== b[1] ? " · the graph sees a difference the flat features cannot." : "");
    }
    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "book") mols = PRESET();
      if (a === "random") mols = randomPair();
      if (a) draw();
    });
    onThemeChange(draw);
    window.addEventListener("resize", draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 2: message passing, step by step
  // Hand-written rules. parts[i] = literals of the layer-i part: {el: e} / {notEl: e} on layer 0, {msg: k} on layers ≥ 1.
  const HAND_RULES = [
    { name: "R1", parts: [[{ el: 2 }]], text: "O" },
    { name: "R2", parts: [[{ el: 1 }], [{ msg: 0 }]], text: "N, next to a neighbour that sent R1" },
    { name: "R3", parts: [[{ el: 0 }], [{ msg: 0 }]], text: "C, next to a neighbour that sent R1" },
    { name: "R4", parts: [[{ el: 1 }], [], [{ msg: 2 }]], text: "N, next to a neighbour that sent R3 in round 2" },
  ];
  const STAGES = [
    "Start: every atom has one property (its element). Nothing has been sent yet.",
    "Layer 0: every rule checks its first part against each atom's own properties. Orange badges show which rules match where.",
    "Message round 1: wherever a rule's part 0 matches, the atom sends that rule's name to all its neighbours (orange arrows). Each atom collects what it receives in its inbox.",
    "Layer 1: rules with a second part now check each atom's inbox. A rule matches at an atom only if parts 0 and 1 both match there.",
    "Message round 2: wherever a rule matched parts 0 and 1, the atom sends the rule's name again. These messages carry information from two bonds away.",
    "Layer 2: R4's third part checks the round-2 inbox: \"a neighbour sent R3\", i.e. a neighbour that is a C next to an O.",
    "OR over atoms: a rule is True for the molecule if it matched (all its parts) at one atom or more. These truth values vote, exactly as in a standard Tsetlin machine.",
  ];
  function runHand(m) {
    // returns per layer i: match[j][v] (parts 0..i), and inbox[i][v] = Set of rule indices received in round i
    const n = m.el.length, R = HAND_RULES.length, match = [], inbox = [null];
    const part0 = (r, v) => r.parts[0].every((l) => (l.el !== undefined ? m.el[v] === l.el : m.el[v] !== l.notEl));
    match.push(HAND_RULES.map((r) => m.el.map((_, v) => part0(r, v))));
    for (let i = 1; i <= 2; i++) {
      const box = m.el.map((_, v) => new Set());
      for (let k = 0; k < R; k++) for (let u = 0; u < n; u++) if (match[i - 1][k][u]) m.nbr[u].forEach((v) => box[v].add(k));
      inbox.push(box);
      match.push(HAND_RULES.map((r, j) => m.el.map((_, v) => match[i - 1][j][v] && (r.parts[i] || []).every((l) => box[v].has(l.msg)))));
    }
    return { match, inbox };
  }
  function messageWidget(root) {
    const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [1, 7]];
    const PRESET = () => mol([0, 0, 1, 0, 2, 0, 0, 0], EDGES);
    const ctrl1 = control(root, `<button data-a="prev">◀ Back</button><button data-a="next">Next ▶</button><button data-a="play">▶ Play</button><button data-a="reset">Reset molecule</button>`);
    hint(root, "Rules: <b>R1</b> = O · <b>R2</b> = N ∧ (a neighbour sent R1) · <b>R3</b> = C ∧ (a neighbour sent R1) · <b>R4</b> = N ∧ (round 2: a neighbour sent R3). Click an atom to change its element; click it again with the dashed ring to see its inbox.");
    const out = readout(root);
    let m = PRESET(), stage = 0, sel = 2, timer = null, s = null;
    const stop = () => { clearInterval(timer); timer = null; };

    function draw() {
      if (s) s.remove();
      const c = colors(), narrow = isNarrow(root);
      const W = narrow ? 340 : 620, H = narrow ? 220 : 290;
      s = svg(W, H);
      s.style.maxWidth = W + "px"; s.style.margin = "0 auto"; s.style.display = "block";
      root.insertBefore(s, ctrl1);
      const res = runHand(m), layer = stage >= 5 ? 2 : stage >= 3 ? 1 : 0;
      const badges = {}, arrows = [];
      // badge = rule matched all its parts up to this layer (a rule with fewer parts keeps its last result)
      if (stage >= 1) m.el.forEach((_, v) => {
        badges[v] = HAND_RULES.map((r, j) => {
          const L = Math.min(layer, r.parts.length - 1);
          return res.match[L][j][v] ? r.name : null;
        }).filter(Boolean);
      });
      if (stage === 2 || stage === 4) {
        const round = stage === 2 ? 0 : 1;
        for (let u = 0; u < m.el.length; u++) if (HAND_RULES.some((r, j) => res.match[round][j][u]))
          m.nbr[u].forEach((v) => arrows.push([u, v]));
      }
      const dim = new Set();
      if (stage >= 1 && sel !== null) {
        const d = GTM.distances(m, sel);
        m.el.forEach((_, v) => { if (d[v] > layer) dim.add(v); });
      }
      label(s, W / 2, 16, stage === 0 ? "input graph" : `after: ${["", "layer 0", "message round 1", "layer 1", "message round 2", "layer 2", "OR over atoms"][stage]}`, { style: textStyle + "; font-weight: 600" });
      drawMolecule(s, m, { x: 20, y: 28, w: W - 40, h: H - 50 }, {
        r: narrow ? 15 : 17, badges, arrows, dim: stage >= 1 && stage < 6 ? dim : null, sel,
        onClick: (v) => { stop(); if (sel === v) m.el[v] = (m.el[v] + 1) % 3; sel = v; draw(); },
      });
      // readout
      let txt = `Step ${stage + 1} of ${STAGES.length}: ${STAGES[stage]}`;
      if (sel !== null) {
        const ib = (i) => { const b = res.inbox[i][sel]; return b.size ? [...b].map((k) => HAND_RULES[k].name).join(", ") : "empty"; };
        txt += `\nSelected atom (${EL[m.el[sel]]}, dashed ring): inbox round 1 = {${ib(1)}} · inbox round 2 = {${ib(2)}}`;
        if (stage >= 1 && stage < 6) txt += `\nFaded atoms are more than ${layer} bond${layer === 1 ? "" : "s"} away: nothing they have can reach the selected atom by this layer.`;
      }
      if (stage === 6) {
        txt += "\n" + HAND_RULES.map((r, j) => {
          const Lr = r.parts.length - 1, at = m.el.map((_, v) => v).filter((v) => res.match[Lr][j][v]);
          return `${r.name} (${r.text}): ${at.length ? "True" : "False"}`;
        }).join(" · ");
      }
      out.textContent = txt;
      mark(ctrl1, "a", "");
    }
    ctrl1.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (!a) return;
      if (a === "prev") { stop(); stage = Math.max(0, stage - 1); }
      if (a === "next") { stop(); stage = Math.min(STAGES.length - 1, stage + 1); }
      if (a === "reset") { stop(); m = PRESET(); stage = 0; sel = 2; }
      if (a === "play") {
        if (timer) { stop(); return; }
        stage = 0;
        timer = setInterval(() => { stage++; if (stage >= STAGES.length - 1) { stage = STAGES.length - 1; stop(); } draw(); }, 1600);
      }
      draw();
    });
    onThemeChange(draw);
    window.addEventListener("resize", draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 3: train a graph Tsetlin machine
  // Readable description of what rule j has matched at a node after layer i (its parts 0..i), traced back
  // through the messages to atoms, as the paper does with its Table 1.
  function describe(tm, j, i, model, depthLeft = 3) {
    const parts = tm.parts(j);
    if (model === "flat") {
      const names = ["C", "N", "O"].flatMap((e) => [1, 2, 3].map((t) => `${e}≥${t}`));
      const lits = parts[0].map((l) => (l.neg ? "¬" : "") + names[l.k]);
      return lits.length ? lits.join(" ∧ ") : "(empty: always True)";
    }
    // layer 0: one-hot elements, so reduce the literals to the set of elements still allowed
    let allowed = [0, 1, 2];
    parts[0].forEach((l) => { allowed = allowed.filter((e) => (l.neg ? e !== l.k : e === l.k)); });
    const atom = allowed.length === 0 ? "nothing" : allowed.length === 3 ? "any atom" : allowed.length === 1 ? EL[allowed[0]] : allowed.map((e) => EL[e]).join(" or ");
    const bits = [atom];
    for (let q = 1; q <= i; q++) {
      const seen = new Set();
      parts[q].forEach((l) => {
        const inner = depthLeft > 0 ? describe(tm, l.k, q - 1, model, depthLeft - 1) : `R${l.k + 1}`;
        const t = q === 1 ? `next to [${inner}]` : `next to [${inner}] (round ${q})`;
        if (!seen.has(t)) { seen.add(t); bits.push(t); }
      });
    }
    return bits.join(", ");
  }
  function drawChart(s, W, H, history, epochs) {
    const c = colors(), M = { l: 40, r: 10, t: 14, b: 32 };
    const x = linear(0, epochs, M.l, W - M.r), y = linear(0.4, 1, H - M.b, M.t);
    axes(s, x, y, { xDomain: [0, epochs], yDomain: [0.4, 1], xTicks: ticks(0, epochs, 5), yTicks: [0.4, 0.6, 0.8, 1], xLabel: "epoch" });
    const line = (k, col, dash) => {
      if (!history.length) return;
      const d = history.map((h, i) => (i ? "L" : "M") + x(i + 1) + "," + y(Math.max(0.4, h[k]))).join("");
      el("path", { d, fill: "none", stroke: col, "stroke-width": 2.5, "stroke-dasharray": dash || "" }, s);
      history.forEach((h, i) => el("circle", { cx: x(i + 1), cy: y(Math.max(0.4, h[k])), r: 2.5, fill: col }, s));
    };
    line(0, c.muted, "5 4");
    line(1, c.opt);
    label(s, W - M.r - 4, H - M.b - 26, "test accuracy", { "text-anchor": "end", style: `fill: ${c.opt}; font-weight: 600` });
    label(s, W - M.r - 4, H - M.b - 10, "training accuracy (dashed)", { "text-anchor": "end" });
  }

  function trainWidget(root) {
    const ctrl1 = control(root, `Model: <button data-m="flat">Standard TM (atom counts)</button><button data-m="1">Graph TM, depth 1</button><button data-m="2">Graph TM, depth 2</button><button data-m="3">Graph TM, depth 3</button>`);
    const ctrl2 = control(root, `Task: <button data-t="bond">N bonded to O?</button><button data-t="two">N within 2 bonds of O?</button>
      <label>seed <input type="number" min="1" max="999" value="1" style="width:4em"></label>`);
    const ctrl3 = control(root, `<button data-a="step">Step: one molecule</button><button data-a="epoch">Train 1 epoch</button><button data-a="all">Train to 15 epochs</button><button data-a="test">Show a test molecule</button><button data-a="reset">Reset</button>`);
    const seedIn = ctrl2.querySelector("input");
    hint(root, "Click a rule in the list to see where it matches in the molecule (green rings). After a step, the dashed ring marks the atom that rule was given feedback at, picked at random among the atoms where it matched.");
    const out = readout(root);
    const list = document.createElement("div");
    list.className = "viz-readout";
    list.style.whiteSpace = "pre-line";
    root.appendChild(list);
    const EPOCHS = 15;
    let model = "2", task = "bond", ex, order, ptr, last, sel, view, s = null, chart = null, busy = null;

    function reset() {
      clearTimeout(busy); busy = null;
      ex = GTM.experiment({ seed: +seedIn.value || 1, task, model: model === "flat" ? "flat" : "graph", D: model === "flat" ? 1 : +model });
      order = null; ptr = 0; last = null; sel = ex.tm.n;
      view = { k: 0, test: false };
      draw();
    }
    function step() {
      if (ex.history.length >= EPOCHS) return;
      if (!order) { order = GTM.shuffledOrder(ex.Xtr.length, ex.rand); ptr = 0; }
      const i = order[ptr++], log = {};
      ex.tm.update(ex.Xtr[i], ex.tr.y[i], log);
      last = log; view = { k: i, test: false };
      const fbRule = log.fb.findIndex((f) => f.type === "recognize" || f.type === "reject");
      if (fbRule >= 0) sel = fbRule;
      if (ptr >= order.length) { ex.history.push([GTM.accuracy(ex.tm, ex.Xtr, ex.tr.y), GTM.accuracy(ex.tm, ex.Xte, ex.te.y)]); order = null; }
    }
    function epoch() {
      if (ex.history.length >= EPOCHS) return;
      const before = ex.history.length;
      while (ex.history.length === before) step();
      last = null;
    }
    function draw() {
      if (s) s.remove();
      if (chart) chart.remove();
      const c = colors(), narrow = isNarrow(root), tm = ex.tm;
      mark(ctrl1, "m", model); mark(ctrl2, "t", task);
      const g = view.test ? ex.te.G[view.k] : ex.tr.G[view.k], y = view.test ? ex.te.y[view.k] : ex.tr.y[view.k];
      const X = view.test ? ex.Xte[view.k] : ex.Xtr[view.k];
      const W = narrow ? 340 : 620, H = 210;
      s = svg(W, H);
      s.style.maxWidth = W + "px"; s.style.margin = "0 auto"; s.style.display = "block";
      root.insertBefore(s, ctrl1);
      const taskName = task === "bond" ? "N bonded to O" : "N within 2 bonds of O";
      label(s, W / 2, 16, `${view.test ? "test" : "training"} molecule #${view.k + 1}: ${taskName}? ${y ? "yes" : "no"}`, { style: textStyle + "; font-weight: 600" });
      const hl = new Set();
      let fbAtom = null;
      if (model !== "flat") {
        const { M } = tm.evaluate(X);
        M[sel].forEach((b, v) => { if (b) hl.add(v); });
        if (last && last.fb[sel] && last.fb[sel].node !== undefined) fbAtom = last.fb[sel].node;
      }
      drawMolecule(s, g, { x: 20, y: 26, w: W - 40, h: H - 34 }, { r: narrow ? 14 : 16, hl, sel: fbAtom });
      // chart
      const CW = narrow ? 340 : 620, CH = 180;
      chart = svg(CW, CH);
      chart.style.maxWidth = CW + "px"; chart.style.margin = "8px auto 0"; chart.style.display = "block";
      root.insertBefore(chart, ctrl1);
      drawChart(chart, CW, CH, ex.history, EPOCHS);
      // readout
      const ep = ex.history.length, h = ex.history[ep - 1];
      let txt = `epochs trained: ${ep}/${EPOCHS}` + (h ? ` · training accuracy ${pct(h[0])} · test accuracy ${pct(h[1])}` : "");
      if (order && ptr) txt += `\nepoch ${ep + 1}: molecule ${ptr} of ${order.length}`;
      if (last) {
        const nf = last.fb.filter((f) => f.type !== "none").length;
        txt += `\nvote sum for the true class v = ${last.v}, so each rule gets feedback with probability (T − v)/2T = ${fmt(last.prob, 3)} → ${nf} of ${tm.m} rules did.`;
        const f = last.fb[sel];
        if (f && f.node !== undefined && model !== "flat") txt += `\nselected rule R${sel + 1}: ${f.type === "recognize" ? "Recognize" : "Reject"} Feedback at 1 atom (dashed ring), picked at random from the ${f.of} atom${f.of > 1 ? "s" : ""} where it matched.`;
      } else {
        const v = tm.vote(X);
        txt += `\nOn this molecule the vote sum for "no" is ${v}, so the machine says ${v >= 0 ? "no" : "yes"} (${(v >= 0 ? 0 : 1) === y ? "correct" : "wrong"}).`;
      }
      out.textContent = txt;
      // rule list
      const outs = ex.Xte.map((x) => tm.outputs(x));
      const rows = [];
      for (let j = 0; j < tm.m; j++) {
        let fy = 0, fn = 0;
        outs.forEach((o, q) => { if (o[j]) { if (ex.te.y[q]) fy++; else fn++; } });
        rows.push({ j, txt: describe(tm, j, tm.D - 1, ex.model), fy, fn });
      }
      list.innerHTML = "";
      const head = document.createElement("div");
      head.innerHTML = `<b>The ${tm.m} rules</b>, read back to atoms (fires on: test molecules with answer yes / no). Click a rule to select it.`;
      list.appendChild(head);
      [1, 0].forEach((cls) => {
        const hd = document.createElement("div");
        hd.style.marginTop = "6px";
        hd.innerHTML = `<b>voting for "${cls ? "yes" : "no"}"</b>`;
        list.appendChild(hd);
        rows.filter((r) => tm.cls[r.j] === (cls ? 1 : 0)).forEach((r) => {
          const d = document.createElement("div");
          d.style.cursor = "pointer";
          d.style.paddingLeft = "8px";
          if (r.j === sel) { d.style.borderLeft = "3px solid var(--viz-model)"; d.style.paddingLeft = "5px"; }
          const dead = r.fy + r.fn === 0;
          d.style.opacity = dead ? 0.55 : 1;
          d.textContent = `R${r.j + 1}: ${r.txt}  (${r.fy} / ${r.fn})${dead ? " never fires" : ""}`;
          d.addEventListener("click", () => { sel = r.j; draw(); });
          list.appendChild(d);
        });
      });
    }
    function run() {
      if (busy || ex.history.length >= EPOCHS) return;
      const tick = () => { epoch(); draw(); busy = ex.history.length < EPOCHS ? setTimeout(tick, 30) : null; };
      busy = setTimeout(tick, 0);
    }
    ctrl1.addEventListener("click", (e) => { if (e.target.dataset.m) { model = e.target.dataset.m; reset(); } });
    ctrl2.addEventListener("click", (e) => { if (e.target.dataset.t) { task = e.target.dataset.t; reset(); } });
    ctrl2.addEventListener("change", reset);
    ctrl3.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (!a || (busy && a !== "reset")) return;
      if (a === "step") { step(); draw(); }
      if (a === "epoch") { epoch(); draw(); }
      if (a === "all") run();
      if (a === "reset") reset();
      if (a === "test") {
        view = { k: Math.floor(Math.random() * ex.te.G.length), test: true };
        last = null;
        const X = ex.Xte[view.k], o = ex.tm.outputs(X), firing = o.findIndex((b, j) => b && ex.tm.cls[j] === ex.te.y[view.k]);
        if (firing >= 0) sel = firing;
        draw();
      }
    });
    onThemeChange(draw);
    window.addEventListener("resize", draw);
    reset();
  }

  const mounts = { "viz-flat": flatWidget, "viz-messages": messageWidget, "viz-train": trainWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
