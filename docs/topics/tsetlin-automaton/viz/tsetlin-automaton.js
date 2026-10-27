// Interactive widgets for "The Tsetlin Automaton: A Memory That Learns".
// Widgets, each mounted on a <div> with a matching id:
//   #viz-ta-bandit     – Tsetlin's automaton (one integer) plays a two-armed bandit, next to ε-greedy
//   #viz-literal       – one literal's memory position as a random walk (only examples of its own class)
//   #viz-sharpen       – P(memorized) against how often the literal is True, for several memory depths
//   #viz-literal-full  – the book's complete single-literal chain (both classes), with the book's four scenarios
// All simulations use a seeded mulberry32 generator and only + − × ÷ and comparisons, so the numbers are the
// same in every browser and match the Python check script exactly.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), a | 1);
      t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
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
    div.textContent = text;
    parent.appendChild(div);
    return div;
  }
  const pct = (v, d = 1) => fmt(100 * v, d) + "%";
  // Show the value of every range slider next to it
  function bindSliders(ctrl, format) {
    ctrl.querySelectorAll("input[type=range]").forEach((inp) => {
      const show = () => { const s = inp.parentElement.querySelector(".val"); if (s) s.textContent = format(inp.name, +inp.value); };
      inp.addEventListener("input", show);
      show();
    });
  }

  // Exact stationary distribution of a birth–death chain (states 0..D−1) from up[i] = P(i → i+1), down = P(i → i−1)
  function birthDeath(up, down) {
    const w = [1];
    for (let i = 0; i < up.length - 1; i++) w.push(w[i] * up[i] / down[i + 1]);
    let sum = 0;
    for (const v of w) sum += v;
    return w.map((v) => v / sum);
  }

  // ---------------------------------------------------------------- Widget 1: a Tsetlin automaton plays a two-armed bandit
  // States 0..N−1 choose arm A (state 0 is deepest), states N..2N−1 choose arm B (state 2N−1 is deepest).
  // Reward: one step deeper into the current half. Penalty: one step toward the centre (crossing it = switching arm).
  function taLongRun(r, N) {
    const up = [], down = [];
    for (let i = 0; i < 2 * N; i++) {
      if (i < N) { up.push(1 - r[0]); down.push(r[0]); } else { up.push(r[1]); down.push(1 - r[1]); }
    }
    const pi = birthDeath(up, down);
    let a = 0;
    for (let i = 0; i < N; i++) a += pi[i];
    return a;                                   // long-run share of pulls on arm A
  }

  function banditWidget(root) {
    const narrow = root.clientWidth < 480;
    const W = narrow ? 340 : 600, H = 360;
    const chart = svg(W, H);
    root.appendChild(chart);
    const ctrl = control(root, `
      <label>arm A pays <input type="range" name="rA" min="0.05" max="0.95" step="0.05" value="0.4"> <span class="val"></span></label>
      <label>arm B <input type="range" name="rB" min="0.05" max="0.95" step="0.05" value="0.7"> <span class="val"></span></label>
      <label>states per arm <b>N</b> <input type="range" name="N" min="1" max="10" step="1" value="3"> <span class="val"></span></label>`);
    const btns = control(root, `<button data-a="step">Step</button><button data-a="play">▶ Play</button>
      <button data-a="k">+1000 pulls</button><button data-a="swap">Swap the arms</button><button data-a="new">New run</button>`);
    const playBtn = btns.querySelector("[data-a=play]");
    const out = readout(root);
    hint(root, "Top: the automaton's 2N states. The orange dot is its whole memory; its half decides the arm. Bottom: share of the last 100 pulls that went to the better arm. Swap the arms to see who notices.");
    bindSliders(ctrl, (name, v) => (name === "N" ? v : v.toFixed(2)));
    const inp = (n) => ctrl.querySelector(`[name=${n}]`);

    const EPS = 0.1, MAXT = 20000;
    let seed = 1, sim, timer = null;

    function reset() {
      stop();
      const N = +inp("N").value;
      sim = {
        r: [+inp("rA").value, +inp("rB").value], N, k: N - 1, prev: null, last: null,
        ta: mulberry32(seed), eg: mulberry32(seed + 1000), Q: [0, 0], n: [0, 0],
        t: 0, taHist: [], egHist: [], taBest: 0, egBest: 0, switches: 0, swaps: [],
      };
      draw();
    }

    const isBest = (r, arm) => r[arm] >= r[1 - arm];

    function step() {
      const s = sim, N = s.N;
      if (s.t >= MAXT) return;
      // Tsetlin automaton: one integer s.k
      const arm = s.k < N ? 0 : 1;
      const reward = s.ta() < s.r[arm];
      s.prev = s.k;
      if (arm === 0) s.k = reward ? Math.max(0, s.k - 1) : s.k + 1;
      else s.k = reward ? Math.min(2 * N - 1, s.k + 1) : s.k - 1;
      const switched = (s.k < N ? 0 : 1) !== arm;
      if (switched) s.switches++;
      s.last = { arm, reward, switched };
      const b1 = isBest(s.r, arm) ? 1 : 0;
      s.taBest += b1; s.taHist.push(b1);
      // ε-greedy with sample averages (from the bandits lesson)
      const u1 = s.eg(), u2 = s.eg(), u3 = s.eg(), u4 = s.eg();
      let a;
      if (u1 < EPS) a = u2 < 0.5 ? 0 : 1;
      else a = s.Q[0] > s.Q[1] ? 0 : s.Q[1] > s.Q[0] ? 1 : (u3 < 0.5 ? 0 : 1);
      const rw = u4 < s.r[a] ? 1 : 0;
      s.n[a] += 1; s.Q[a] += (rw - s.Q[a]) / s.n[a];
      const b2 = isBest(s.r, a) ? 1 : 0;
      s.egBest += b2; s.egHist.push(b2);
      s.t++;
    }

    function stop() { clearInterval(timer); timer = null; playBtn.textContent = "▶ Play"; }

    const last100 = (h) => { const n = Math.min(100, h.length); let s = 0; for (let i = h.length - n; i < h.length; i++) s += h[i]; return n ? s / n : 0; };

    function draw() {
      const c = colors(), s = sim, N = s.N;
      chart.innerHTML = "";
      // --- automaton strip
      const top = 22, cy = 62, gap = 22;
      const x0 = 18, x1 = W - 18, half = (x1 - x0 - gap) / 2, cell = half / N;
      const rad = Math.min(13, cell * 0.36);
      const cx = (i) => (i < N ? x0 + (i + 0.5) * cell : x0 + half + gap + (i - N + 0.5) * cell);
      const bestArm = s.r[0] >= s.r[1] ? 0 : 1;
      ["A", "B"].forEach((name, arm) => {
        const xa = arm === 0 ? x0 : x0 + half + gap;
        el("rect", { x: xa, y: top + 6, width: half, height: 56, rx: 8, fill: c.data, "fill-opacity": arm === (s.k < N ? 0 : 1) ? 0.12 : 0.04, stroke: c.border }, chart);
        el("text", { x: xa + half / 2, y: top, "text-anchor": "middle", style: textStyle + "; font-weight: 600" }, chart).textContent =
          `arm ${name} · pays ${fmt(s.r[arm], 2)}${arm === bestArm && s.r[0] !== s.r[1] ? " ★" : ""}`;
      });
      el("line", { x1: (x1 + x0) / 2, x2: (x1 + x0) / 2, y1: top + 2, y2: top + 70, stroke: c.muted, "stroke-dasharray": "3 3" }, chart);
      el("text", { x: x0, y: top + 80, "text-anchor": "start" }, chart).textContent = "← deepest A";
      el("text", { x: x1, y: top + 80, "text-anchor": "end" }, chart).textContent = "deepest B →";
      el("text", { x: (x0 + x1) / 2, y: top + 80, "text-anchor": "middle" }, chart).textContent = "centre";
      if (s.last && s.prev !== null && s.prev !== s.k) {
        const col = s.last.reward ? c.opt : c.grad;
        const xa = cx(s.prev), xb = cx(s.k), dir = Math.sign(xb - xa);
        el("path", { d: `M${xa + dir * rad},${cy - rad - 2} Q${(xa + xb) / 2},${cy - rad - 18} ${xb - dir * rad},${cy - rad - 2}`, fill: "none", stroke: col, "stroke-width": 2 }, chart);
        el("circle", { cx: xb - dir * rad, cy: cy - rad - 2, r: 2.5, fill: col }, chart);
      }
      for (let i = 0; i < 2 * N; i++) {
        const on = i === s.k;
        el("circle", { cx: cx(i), cy, r: rad, fill: on ? c.model : c.bg, stroke: on ? c.model : c.muted, "stroke-width": 1.5 }, chart);
      }
      // --- chart: share of the last 100 pulls on the better arm
      const M = { l: 40, r: 12, t: 140, b: 52 };
      const T = Math.max(200, s.t);
      const x = linear(0, T, M.l, W - M.r), y = linear(0, 1, H - M.b, M.t);
      const step = T <= 400 ? 100 : T <= 2000 ? 500 : T <= 6000 ? 1000 : 5000;
      axes(chart, x, y, { xDomain: [0, T], yDomain: [0, 1], xTicks: ticks(0, T, step), yTicks: [0, 0.5, 1], xLabel: "pulls" });
      el("text", { x: M.l, y: M.t - 8, style: textStyle + "; font-size: 12px" }, chart).textContent = "on the better arm (last 100 pulls)";
      for (const sw of s.swaps) el("line", { x1: x(sw), x2: x(sw), y1: y(0), y2: y(1), stroke: c.muted, "stroke-dasharray": "2 3" }, chart);
      const lr = taLongRun(s.r, N), lrBest = s.r[0] >= s.r[1] ? lr : 1 - lr;
      el("line", { x1: x(0), x2: x(T), y1: y(lrBest), y2: y(lrBest), stroke: c.opt, "stroke-width": 1.5, "stroke-dasharray": "6 4" }, chart);
      const line = (h, col, dash) => {
        if (!h.length) return;
        const every = Math.max(1, Math.floor(h.length / 400));
        let d = "", sum = 0;
        const pre = [0];
        for (let i = 0; i < h.length; i++) { sum += h[i]; pre.push(sum); }
        for (let i = 1; i <= h.length; i += every) {
          const n = Math.min(100, i), v = (pre[i] - pre[i - n]) / n;
          d += (d ? "L" : "M") + x(i).toFixed(1) + "," + y(v).toFixed(1);
        }
        el("path", { d, fill: "none", stroke: col, "stroke-width": 1.8, "stroke-dasharray": dash || "none", "stroke-linejoin": "round" }, chart);
      };
      line(s.egHist, c.data, "4 3");
      line(s.taHist, c.model);
      // legend
      const ly = H - 6;
      const legend = [[c.model, "none", narrow ? "automaton" : "Tsetlin automaton"], [c.data, "4 3", "ε-greedy"], [c.opt, "6 4", narrow ? "exact" : "exact long run"]];
      let lx = M.l;
      for (const [col, dash, label] of legend) {
        el("line", { x1: lx, x2: lx + 16, y1: ly - 4, y2: ly - 4, stroke: col, "stroke-width": 2, "stroke-dasharray": dash }, chart);
        el("text", { x: lx + 20, y: ly }, chart).textContent = label;
        lx += narrow ? 96 : 150;
      }
      // --- readout
      const arm = s.k < N ? "A" : "B", depth = s.k < N ? N - s.k : s.k - N + 1;
      let text = `run #${seed} · pulls ${s.t}`;
      if (s.last) text += ` · last pull: arm ${s.last.arm ? "B" : "A"}, ${s.last.reward ? "reward → one step deeper" : "penalty → one step toward the centre"}${s.last.switched ? " (switched arm!)" : ""}`;
      text += `\nTsetlin automaton: state = arm ${arm}, depth ${depth} of ${N} · better arm ${s.t ? pct(s.taBest / s.t) : "–"} of all pulls, ${s.t ? pct(last100(s.taHist), 0) : "–"} of the last 100 · switches ${s.switches}`;
      text += `\nε-greedy (ε = 0.1): Q(A) = ${fmt(s.Q[0], 2)}, Q(B) = ${fmt(s.Q[1], 2)} · better arm ${s.t ? pct(s.egBest / s.t) : "–"} of all pulls, ${s.t ? pct(last100(s.egHist), 0) : "–"} of the last 100`;
      text += `\nexact long-run share of the automaton on the better arm: ${s.r[0] === s.r[1] ? "both arms are equal" : pct(lrBest)}`;
      if (s.t >= MAXT) text += `\n(stopped at ${MAXT} pulls: press New run)`;
      out.textContent = text;
    }

    ctrl.addEventListener("change", reset);
    btns.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (!a) return;
      if (a === "step") { stop(); step(); draw(); }
      if (a === "k") { stop(); for (let i = 0; i < 1000; i++) step(); draw(); }
      if (a === "new") { seed++; reset(); }
      if (a === "swap") {
        sim.r = [sim.r[1], sim.r[0]];
        inp("rA").value = sim.r[0]; inp("rB").value = sim.r[1];
        bindSliders(ctrl, (name, v) => (name === "N" ? v : v.toFixed(2)));
        sim.swaps.push(sim.t);
        draw();
      }
      if (a === "play") {
        if (timer) { stop(); return; }
        playBtn.textContent = "⏸ Pause";
        timer = setInterval(() => { step(); step(); draw(); if (sim.t >= MAXT) stop(); }, 50);
      }
    });
    onThemeChange(draw);
    reset();
  }

  // ---------------------------------------------------------------- Widgets 2 and 4: one literal's memory position
  // Positions 1..D; 1..D/2 Forgotten, D/2+1..D Memorized. Memorize Value 1, Forget Value 1/s (Granmo, Ch. 2, §2.1).
  function literalRates(p) {
    const a = p.pLY * p.pY + p.pnLnY * (1 - p.pY);   // up while forgotten: Recognize + Reject
    const r = p.pLY * p.pY;                          // up while memorized: Recognize only
    const e = (1 - p.pLY) * p.pY / p.s;              // down: Erase
    return { a, r, e };
  }
  function literalExact(p) {
    const { a, r, e } = literalRates(p), D = p.D;
    const up = [], down = [];
    for (let i = 1; i <= D; i++) { up.push(i === D ? 0 : i <= D / 2 ? a : r); down.push(i === 1 ? 0 : e); }
    if (e === 0) {                                   // the literal is never forgotten
      const pi = new Array(D).fill(0);
      pi[r > 0 ? D - 1 : D / 2 - 1] = 1;             // r = 0 and e = 0: it stays at the top of Forgotten after climbing there
      if (r === 0 && a === 0) return null;           // nothing ever happens: depends on the start
      return pi;
    }
    return birthDeath(up, down);
  }

  const SCENARIOS = {
    1: { pY: 0.9, pLY: 0.1, pnLnY: 0.9, s: 1, D: 4 },
    2: { pY: 0.9, pLY: 0.1, pnLnY: 0.9, s: 15, D: 4 },
    3: { pY: 0.82, pLY: 0.99, pnLnY: 0.5, s: 15, D: 4 },
    4: { pY: 0.82, pLY: 0.99, pnLnY: 0.5, s: 15, D: 8 },
  };

  function literalWidget(root, full) {
    const narrow = root.clientWidth < 480;
    const W = narrow ? 340 : 600, H = 270;
    const chart = svg(W, H);
    root.appendChild(chart);
    const sliders = full ? `
      <label><b>P(Y)</b> <input type="range" name="pY" min="0.05" max="1" step="0.01" value="0.9"> <span class="val"></span></label>
      <label><b>P(L | Y)</b> <input type="range" name="pLY" min="0" max="1" step="0.01" value="0.1"> <span class="val"></span></label>
      <label><b>P(¬L | other)</b> <input type="range" name="pnLnY" min="0" max="1" step="0.01" value="0.9"> <span class="val"></span></label>
      <label><b>s</b> <input type="range" name="s" min="1" max="25" step="0.5" value="15"> <span class="val"></span></label>
      <label>depth <input type="range" name="D" min="2" max="20" step="2" value="4"> <span class="val"></span></label>` : `
      <label>literal True in a fraction <b>p</b> of class Y <input type="range" name="pLY" min="0" max="1" step="0.01" value="0.3"> <span class="val"></span></label>
      <label><b>s</b> <input type="range" name="s" min="1" max="25" step="0.5" value="3"> <span class="val"></span></label>
      <label>memory depth <input type="range" name="D" min="2" max="20" step="2" value="10"> <span class="val"></span></label>`;
    const ctrl = control(root, sliders);
    const presets = full ? control(root, "book scenario: " + [1, 2, 3, 4].map((k) => `<button data-p="${k}">${k}</button>`).join("")) : null;
    const btns = control(root, `<button data-a="step">Step</button><button data-a="play">▶ Play</button>
      <button data-a="k">+10,000 examples</button><button data-a="new">New run</button>`);
    const playBtn = btns.querySelector("[data-a=play]");
    const out = readout(root);
    hint(root, "Bars: share of time the literal has spent at each memory position so far (grey = Forgotten, green = Memorized). Orange ticks: the exact long-run distribution. The orange triangle marks where the literal is now.");
    const fmtSlider = (name, v) => (name === "D" ? `${v} positions` : name === "s" ? `${v.toFixed(1)} · Forget Value ${(1 / v).toFixed(3)}` : v.toFixed(2));
    bindSliders(ctrl, fmtSlider);
    const read = () => {
      const v = (n) => +ctrl.querySelector(`[name=${n}]`).value;
      return full ? { pY: v("pY"), pLY: v("pLY"), pnLnY: v("pnLnY"), s: v("s"), D: v("D") }
        : { pY: 1, pLY: v("pLY"), pnLnY: 0, s: v("s"), D: v("D") };
    };

    let seed = 1, sim, timer = null;
    function reset() {
      stop();
      const p = read();
      sim = { p, pos: p.D / 2, rand: mulberry32(seed), counts: new Array(p.D + 1).fill(0), t: 0, last: "", exact: literalExact(p) };
      draw();
    }
    function step() {
      const s = sim, p = s.p, D = p.D;
      const u1 = s.rand(), u2 = s.rand(), u3 = s.rand();
      if (u1 < p.pY) {
        if (u2 < p.pLY) { s.last = "class Y, literal True → Recognize: one step up"; if (s.pos < D) s.pos++; }
        else if (u3 >= 1 / p.s) s.last = `class Y, literal False → Erase, but skipped this time (probability ${fmt(1 - 1 / p.s, 3)})`;
        else if (s.pos > 1) { s.last = "class Y, literal False → Erase: one step down"; s.pos--; }
        else s.last = "class Y, literal False → Erase, but already Maximally Forgotten";
      } else {
        if (u2 < p.pnLnY) {
          if (s.pos <= D / 2) { s.pos++; s.last = "other class, literal False → Reject: one step up (it was Forgotten)"; }
          else s.last = "other class, literal False → nothing (already Memorized)";
        } else s.last = "other class, literal True → no feedback";
      }
      s.counts[s.pos]++; s.t++;
    }
    function stop() { clearInterval(timer); timer = null; playBtn.textContent = "▶ Play"; }

    function draw() {
      const c = colors(), s = sim, D = s.p.D;
      chart.innerHTML = "";
      const M = { l: 40, r: 12, t: 30, b: 50 };
      let ymax = 0.2;
      for (let i = 1; i <= D; i++) ymax = Math.max(ymax, s.t ? s.counts[i] / s.t : 0, s.exact ? s.exact[i - 1] : 0);
      ymax = Math.min(1, Math.ceil(ymax * 5 - 1e-9) / 5);
      const x = linear(0.5, D + 0.5, M.l, W - M.r), y = linear(0, ymax, H - M.b, M.t);
      axes(chart, x, y, { xDomain: [0.5, D + 0.5], yDomain: [0, ymax], xTicks: [], yTicks: ticks(0, ymax, ymax > 0.4 ? 0.2 : 0.1) });
      el("text", { x: M.l, y: 16, style: textStyle + "; font-size: 12px" }, chart).textContent = "share of time at each position";
      const bw = x(1) - x(0);
      for (let i = 1; i <= D; i++) {
        const mem = i > D / 2, v = s.t ? s.counts[i] / s.t : 0;
        el("rect", { x: x(i) - bw * 0.4, y: y(v), width: bw * 0.8, height: y(0) - y(v), fill: mem ? c.opt : c.muted, "fill-opacity": 0.55, stroke: i === s.pos ? c.model : "none", "stroke-width": 2 }, chart);
        if (s.exact) {
          const ey = y(s.exact[i - 1]);
          el("line", { x1: x(i) - bw * 0.45, x2: x(i) + bw * 0.45, y1: ey, y2: ey, stroke: c.model, "stroke-width": 2.5 }, chart);
        }
        if (bw >= 14 || i % 2 === 0 || i === 1) el("text", { x: x(i), y: y(0) + 14, "text-anchor": "middle" }, chart).textContent = i;
      }
      // where the literal is now
      const px = x(s.pos), py = y(0) + 20;
      el("path", { d: `M${px},${py} l-6,9 l12,0 z`, fill: c.model }, chart);
      // Forgotten | Memorized threshold
      const tx = x(D / 2 + 0.5);
      el("line", { x1: tx, x2: tx, y1: M.t - 6, y2: y(0), stroke: c.text, "stroke-dasharray": "4 3" }, chart);
      el("text", { x: tx - 6, y: M.t + 6, "text-anchor": "end", style: textStyle }, chart).textContent = "Forgotten";
      el("text", { x: tx + 6, y: M.t + 6, "text-anchor": "start", style: "fill: var(--viz-opt); font-weight: 600" }, chart).textContent = "Memorized";
      el("text", { x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle" }, chart).textContent = "memory position (1 = Maximally Forgotten)";

      let memT = 0, memE = 0;
      for (let i = D / 2 + 1; i <= D; i++) { memT += s.counts[i]; if (s.exact) memE += s.exact[i - 1]; }
      const { a, r, e } = literalRates(s.p);
      let text = `run #${seed} · examples seen ${s.t} · position now ${s.pos} of ${D} (${s.pos > D / 2 ? "Memorized" : "Forgotten"})`;
      if (s.last) text += `\nlast example: ${s.last}`;
      text += `\ntime Memorized so far: ${s.t ? pct(memT / s.t) : "–"} · exact long-run P(Memorized): ${!s.exact ? "depends on the start" : memE < 1 && 1 - memE < 1e-3 ? `≈ 100% (1 − ${(1 - memE).toExponential(1)})` : pct(memE)}`;
      if (full) text += `\nper example: up while Forgotten a = ${fmt(a, 4)} · up while Memorized r = ${fmt(r, 4)} · down e = ${fmt(e, 4)}`;
      else {
        const rho = e > 0 ? r / e : Infinity;
        text += `\nup ${fmt(r, 3)} · down ${fmt(e, 3)} per example · ratio ρ = up / down = ${e > 0 ? fmt(rho, 3) : "∞ (never forgotten)"} · threshold p* = 1/(s+1) = ${fmt(1 / (s.p.s + 1), 3)}`;
      }
      out.textContent = text;
    }

    ctrl.addEventListener("change", reset);
    if (presets) presets.addEventListener("click", (e) => {
      const k = e.target.dataset.p;
      if (!k) return;
      const sc = SCENARIOS[k];
      for (const n in sc) ctrl.querySelector(`[name=${n}]`).value = sc[n];
      bindSliders(ctrl, fmtSlider);
      reset();
    });
    btns.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (!a) return;
      if (a === "step") { stop(); step(); draw(); }
      if (a === "k") { stop(); for (let i = 0; i < 10000; i++) step(); draw(); }
      if (a === "new") { seed++; reset(); }
      if (a === "play") {
        if (timer) { stop(); return; }
        playBtn.textContent = "⏸ Pause";
        timer = setInterval(() => { for (let i = 0; i < 3; i++) step(); draw(); }, 60);
      }
    });
    onThemeChange(draw);
    reset();
  }

  // ---------------------------------------------------------------- Widget 3: deeper memory, sharper decisions
  function sharpenWidget(root) {
    const narrow = root.clientWidth < 480;
    const W = narrow ? 340 : 600, H = 290, M = { l: 44, r: 14, t: 26, b: 40 };
    const chart = svg(W, H);
    root.appendChild(chart);
    const ctrl = control(root, `
      <label><b>s</b> <input type="range" name="s" min="1" max="25" step="0.5" value="3"> <span class="val"></span></label>
      <label>read off at <b>p</b> = <input type="range" name="p" min="0.01" max="0.99" step="0.01" value="0.3"> <span class="val"></span></label>`);
    const legend = control(root, "");
    const out = readout(root);
    hint(root, "Each curve is the exact probability that the literal is Memorized in the long run, P(L | Y) = p on the x-axis, only examples of class Y. Click a legend entry to hide or show a depth.");
    bindSliders(ctrl, (name, v) => (name === "s" ? `${v.toFixed(1)} · Forget Value ${(1 / v).toFixed(3)}` : v.toFixed(2)));
    const DEPTHS = [{ D: 2, role: "muted", dash: "3 3" }, { D: 4, role: "data" }, { D: 8, role: "loss" }, { D: 20, role: "model" }];
    const hidden = new Set();
    // ρ = s·p / (1 − p); with D = 2n positions, P(Memorized) = ρⁿ / (1 + ρⁿ)
    function pMem(p, s, D) {
      if (p >= 1) return 1;
      const rho = s * p / (1 - p);
      let rn = 1;
      for (let i = 0; i < D / 2; i++) rn *= rho;
      return Number.isFinite(rn) ? rn / (1 + rn) : 1;
    }
    function legendHtml() {
      legend.innerHTML = DEPTHS.map((d) => `<button data-d="${d.D}" style="opacity:${hidden.has(d.D) ? 0.4 : 1}"><span style="display:inline-block;width:18px;height:0;border-top:3px ${d.dash ? "dashed" : "solid"} var(--viz-${d.role});vertical-align:middle;margin-right:6px"></span>depth ${d.D}</button>`).join("");
    }
    function draw() {
      const c = colors();
      const s = +ctrl.querySelector("[name=s]").value, p0 = +ctrl.querySelector("[name=p]").value;
      chart.innerHTML = "";
      const x = linear(0, 1, M.l, W - M.r), y = linear(0, 1, H - M.b, M.t);
      axes(chart, x, y, { xDomain: [0, 1], yDomain: [0, 1], xTicks: ticks(0, 1, narrow ? 0.25 : 0.1), yTicks: ticks(0, 1, 0.25), xLabel: "p = how often the literal is True in class Y" });
      el("text", { x: M.l, y: 15, style: textStyle + "; font-size: 12px" }, chart).textContent = "P(Memorized) in the long run";
      const thr = 1 / (s + 1);
      el("line", { x1: x(thr), x2: x(thr), y1: y(0), y2: y(1), stroke: c.text, "stroke-dasharray": "4 3" }, chart);
      el("text", { x: x(thr) + 5, y: y(0.05), style: textStyle }, chart).textContent = `p* = 1/(s+1) = ${fmt(thr, 3)}`;
      el("line", { x1: x(p0), x2: x(p0), y1: y(0), y2: y(1), stroke: c.muted, "stroke-width": 1 }, chart);
      for (const d of DEPTHS) {
        if (hidden.has(d.D)) continue;
        let path = "";
        for (let i = 0; i <= 400; i++) {
          const p = i / 400;
          path += (i ? "L" : "M") + x(p).toFixed(1) + "," + y(pMem(p, s, d.D)).toFixed(1);
        }
        el("path", { d: path, fill: "none", stroke: c[d.role], "stroke-width": 2, "stroke-dasharray": d.dash || "none" }, chart);
        el("circle", { cx: x(p0), cy: y(pMem(p0, s, d.D)), r: 3.5, fill: c[d.role] }, chart);
      }
      const rho = s * p0 / (1 - p0);
      out.textContent = `at p = ${fmt(p0, 2)}, s = ${fmt(s, 1)}: ρ = s·p/(1−p) = ${fmt(rho, 3)}\n` +
        DEPTHS.map((d) => `depth ${d.D}: ${fmt(pMem(p0, s, d.D), 3)}`).join(" · ");
    }
    ctrl.addEventListener("input", draw);
    legend.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      const D = +b.dataset.d;
      hidden.has(D) ? hidden.delete(D) : hidden.add(D);
      legendHtml(); draw();
    });
    onThemeChange(draw);
    legendHtml();
    draw();
  }

  const mounts = {
    "viz-ta-bandit": banditWidget,
    "viz-literal": (n) => literalWidget(n, false),
    "viz-sharpen": sharpenWidget,
    "viz-literal-full": (n) => literalWidget(n, true),
  };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
