// Interactive widgets for the "RL from Human Feedback (RLHF) and DPO" lesson.
// Three widgets, each mounted on a <div> with a matching id:
//   #viz-bt  – Bradley–Terry: two reward scores → preference probability and loss
//   #viz-kl  – the KL-regularised optimum π* ∝ π_ref · exp(r/β) on five candidate replies (+ reward hacking)
//   #viz-dpo – DPO on the same five replies: gradient steps on real preference counts
// Runs after DOMContentLoaded so the site-wide helpers (assets/js/viz-common.js) are available.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;
  const textStyle = "fill: var(--viz-text)";
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const softplus = (x) => (x > 30 ? x : Math.log1p(Math.exp(x)));   // −log σ(−x)

  // ---------------------------------------------------------------- The toy prompt and its five replies
  // Prompt: "Why is the sky blue? (asked by a 10-year-old)"
  const NAMES = ["A simple", "B technical", "C wrong", "D refuses", "E flattery"];
  const REF = [0.15, 0.20, 0.30, 0.30, 0.05];       // π_ref: the SFT model's probabilities
  const RM = [2.0, 1.0, -1.0, -1.5, 3.0];           // reward-model scores (E is over-rated!)
  const RM_FIXED_E = -0.5;                          // what the score of E should have been
  const TRUE = [2.0, 1.0, -1.5, -1.0, -0.5];        // hidden "true" quality (what humans really want)
  const K = NAMES.length;

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
  function plots(parent, n, W, H) {
    const box = document.createElement("div");
    box.className = "viz-plots";
    parent.appendChild(box);
    return Array.from({ length: n }, () => { const s = svg(W, H); box.appendChild(s); return s; });
  }
  const dot = (v) => v.reduce((s, x, i) => s + x * TRUE[i], 0);
  const klDiv = (p) => p.reduce((s, x, i) => s + (x > 0 ? x * Math.log(x / REF[i]) : 0), 0);

  // ---------------------------------------------------------------- Widget 1: Bradley–Terry
  function btWidget(root) {
    const W = 340, H = 240, M = { l: 40, r: 12, t: 40, b: 38 };
    const [probSvg, lossSvg] = plots(root, 2, W, H);
    const ctrl = control(root, `
      <label>r(y<sub>A</sub>) <input type="range" min="-4" max="4" step="0.1" value="1"> <span class="ra"></span></label>
      <label>r(y<sub>B</sub>) <input type="range" min="-4" max="4" step="0.1" value="0"> <span class="rb"></span></label>`);
    const btns = control(root, `<button data-a="shift">Add +1 to both</button><button data-a="eq">Make them equal</button><button data-a="reset">Reset</button>`);
    const [ra, rb] = ctrl.querySelectorAll("input");
    const out = readout(root);
    hint(root, "Only the <b>difference</b> Δ = r(y<sub>A</sub>) − r(y<sub>B</sub>) matters. The reward model is trained to make Δ large whenever humans preferred A.");

    const x = linear(-8, 8, M.l, W - M.r);
    const yP = linear(0, 1, H - M.b, M.t), yL = linear(0, 5, H - M.b, M.t);
    const DS = Array.from({ length: 161 }, (_, i) => -8 + i * 0.1);
    const path = (f, y) => DS.map((d, i) => (i ? "L" : "M") + x(d) + "," + y(Math.min(f(d), 5))).join("");

    function draw() {
      const c = colors(), a = +ra.value, b = +rb.value, d = a - b, p = sigmoid(d);
      ra.nextElementSibling.textContent = fmt(a, 1);
      rb.nextElementSibling.textContent = fmt(b, 1);

      probSvg.innerHTML = "";
      axes(probSvg, x, yP, { xDomain: [-8, 8], yDomain: [0, 1], xTicks: ticks(-8, 8, 4), yTicks: [0, 0.5, 1], xLabel: "Δ = r(y_A) − r(y_B)" });
      el("path", { d: path(sigmoid, yP), fill: "none", stroke: c.model, "stroke-width": 2.5 }, probSvg);
      el("line", { x1: x(d), x2: x(d), y1: yP(0), y2: yP(p), stroke: c.model, "stroke-dasharray": "4 3" }, probSvg);
      el("circle", { cx: x(d), cy: yP(p), r: 5.5, fill: c.model, stroke: c.bg, "stroke-width": 2 }, probSvg);
      el("text", { x: M.l + 4, y: 16, style: textStyle }, probSvg).textContent = `P(A preferred) = σ(Δ) = ${fmt(p, 3)}`;

      lossSvg.innerHTML = "";
      axes(lossSvg, x, yL, { xDomain: [-8, 8], yDomain: [0, 5], xTicks: ticks(-8, 8, 4), yTicks: ticks(0, 5, 1), xLabel: "Δ = r(y_A) − r(y_B)" });
      el("path", { d: path((t) => softplus(-t), yL), fill: "none", stroke: c.loss, "stroke-width": 2.5 }, lossSvg);
      el("path", { d: path((t) => softplus(t), yL), fill: "none", stroke: c.grad, "stroke-width": 2, "stroke-dasharray": "5 4" }, lossSvg);
      el("circle", { cx: x(d), cy: yL(Math.min(softplus(-d), 5)), r: 5.5, fill: c.loss, stroke: c.bg, "stroke-width": 2 }, lossSvg);
      el("circle", { cx: x(d), cy: yL(Math.min(softplus(d), 5)), r: 4.5, fill: c.grad, stroke: c.bg, "stroke-width": 2 }, lossSvg);
      el("text", { x: M.l + 4, y: 14, style: "fill: var(--viz-loss)" }, lossSvg).textContent = "— loss if humans chose A";
      el("text", { x: M.l + 4, y: 29, style: "fill: var(--viz-grad)" }, lossSvg).textContent = "- - loss if humans chose B";

      out.textContent =
        `Δ = ${fmt(a, 1)} − (${fmt(b, 1)}) = ${fmt(d, 1)} · P(A ≻ B) = σ(Δ) = ${fmt(p, 3)} · P(B ≻ A) = ${fmt(1 - p, 3)}\n` +
        `if humans chose A: loss = −log σ(Δ) = ${fmt(softplus(-d), 3)} · if they chose B: loss = −log σ(−Δ) = ${fmt(softplus(d), 3)}`;
    }

    ctrl.addEventListener("input", draw);
    btns.addEventListener("click", (e) => {
      const act = e.target.dataset.a;
      if (act === "shift") {
        if (+ra.value > 3 || +rb.value > 3) { ra.value = +ra.value - 3; rb.value = +rb.value - 3; }
        else { ra.value = +ra.value + 1; rb.value = +rb.value + 1; }
      }
      if (act === "eq") rb.value = ra.value;
      if (act === "reset") { ra.value = 1; rb.value = 0; }
      if (act) draw();
    });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Bar chart shared by widgets 2 and 3
  // bars: π_ref as an outlined bar, the policy as a filled orange bar, optional dashed target tick.
  const BW = 340, BH = 260, BM = { l: 36, r: 8, t: 38, b: 58 };
  function drawBars(root, policy, opts) {
    const c = colors();
    root.innerHTML = "";
    const xb = linear(0, K, BM.l, BW - BM.r), y = linear(0, 1, BH - BM.b, BM.t);
    axes(root, xb, y, { xDomain: [0, K], yDomain: [0, 1], xTicks: [], yTicks: ticks(0, 1, 0.25) });
    const cw = xb(1) - xb(0);
    for (let i = 0; i < K; i++) {
      const x0 = xb(i) + cw * 0.18, w = cw * 0.64;
      el("rect", { x: x0, y: y(policy[i]), width: w, height: y(0) - y(policy[i]), fill: c.model, "fill-opacity": 0.85 }, root);
      el("rect", { x: x0, y: y(REF[i]), width: w, height: y(0) - y(REF[i]), fill: "none", stroke: c.text, "stroke-width": 1.5, "stroke-dasharray": "4 3" }, root);
      if (opts.target) el("line", { x1: x0 - 3, x2: x0 + w + 3, y1: y(opts.target[i]), y2: y(opts.target[i]), stroke: c.opt, "stroke-width": 3 }, root);
      el("text", { x: x0 + w / 2, y: y(policy[i]) - 5, "text-anchor": "middle", style: textStyle }, root).textContent = fmt(policy[i], 2);
      const [l1, l2] = NAMES[i].split(" ");
      el("text", { x: x0 + w / 2, y: BH - BM.b + 15, "text-anchor": "middle", style: textStyle + "; font-weight: 600" }, root).textContent = l1;
      el("text", { x: x0 + w / 2, y: BH - BM.b + 28, "text-anchor": "middle" }, root).textContent = l2;
      (opts.rows || []).forEach(([fn, style], k) => {
        el("text", { x: x0 + w / 2, y: BH - BM.b + 42 + 13 * k, "text-anchor": "middle", style }, root).textContent = fn(i);
      });
    }
    el("text", { x: BM.l + 2, y: 14, style: textStyle }, root).textContent = opts.title;
  }

  // ---------------------------------------------------------------- Widget 2: KL-regularised optimum
  function klWidget(root) {
    const [barSvg, curveSvg] = plots(root, 2, BW, BH);
    const ctrl = control(root, `
      <label>KL strength <b>β</b> <input type="range" min="0" max="100" step="1" value="60"> <span class="val"></span></label>
      <label><input type="checkbox"> fix the reward model's mistake on E</label>`);
    const presets = control(root, `Try: <button data-b="10">β = 10</button><button data-b="2">2</button><button data-b="1">1</button><button data-b="0.25">0.25</button><button data-b="0.05">0.05</button>`);
    const [slider, fix] = ctrl.querySelectorAll("input");
    const out = readout(root);
    hint(root, "Dashed outline = π<sub>ref</sub> (the starting model). Orange = π*, the best KL-regularised policy. Under each reply: <span style='color: var(--viz-loss)'>reward-model score</span> / <span style='color: var(--viz-opt)'>true quality</span>.");

    const LB0 = Math.log(0.05), LB1 = Math.log(10);
    const toBeta = (v) => Math.exp(LB0 + (v / 100) * (LB1 - LB0));
    const toSlider = (b) => Math.round(((Math.log(b) - LB0) / (LB1 - LB0)) * 100);
    const rmScores = () => (fix.checked ? RM.map((r, i) => (i === 4 ? RM_FIXED_E : r)) : RM);
    function optimum(beta, rm) {
      const w = REF.map((p, i) => p * Math.exp((rm[i] - Math.max(...rm)) / beta));
      const z = w.reduce((s, v) => s + v, 0);
      return w.map((v) => v / z);
    }

    const M = { l: 36, r: 10, t: 38, b: 58 };
    const xc = linear(LB1, LB0, M.l, BW - M.r), yc = linear(-1, 3, BH - M.b, M.t);   // β decreases to the right

    let beta = 1;   // exact value (presets set it exactly; the slider moves on a log scale)
    function draw() {
      const c = colors(), rm = rmScores();
      slider.nextElementSibling.textContent = beta < 1 ? fmt(beta, 2) : fmt(beta, 1);
      const p = optimum(beta, rm);
      const eRm = p.reduce((s, x, i) => s + x * rm[i], 0), eTrue = dot(p), kl = klDiv(p);

      drawBars(barSvg, p, {
        title: "π* ∝ π_ref · exp(r / β)",
        rows: [[(i) => `${fmt(rm[i], 1)} / ${fmt(TRUE[i], 1)}`, "fill: var(--viz-muted)"]],
      });
      // colour the two scores under each bar
      barSvg.querySelectorAll("text").forEach((t) => {
        const m = /^(-?\d+\.\d) \/ (-?\d+\.\d)$/.exec(t.textContent);
        if (!m) return;
        t.textContent = "";
        const a = el("tspan", { style: "fill: var(--viz-loss)" }, t); a.textContent = m[1];
        const s = el("tspan", {}, t); s.textContent = " / ";
        const b = el("tspan", { style: "fill: var(--viz-opt)" }, t); b.textContent = m[2];
      });

      curveSvg.innerHTML = "";
      const tk = [10, 3, 1, 0.3, 0.1];
      axes(curveSvg, xc, yc, { xDomain: [LB1, LB0], yDomain: [-1, 3], xTicks: [], yTicks: ticks(-1, 3, 1), xLabel: "β (weaker KL leash →)" });
      tk.forEach((b) => el("text", { x: xc(Math.log(b)), y: yc(-1) + 14, "text-anchor": "middle" }, curveSvg).textContent = b);
      const N = 120, pts = Array.from({ length: N + 1 }, (_, i) => LB1 + (i / N) * (LB0 - LB1));
      const line = (f) => pts.map((lb, i) => (i ? "L" : "M") + xc(lb) + "," + yc(f(optimum(Math.exp(lb), rm)))).join("");
      el("path", { d: line((q) => q.reduce((s, x, i) => s + x * rm[i], 0)), fill: "none", stroke: c.loss, "stroke-width": 2.5 }, curveSvg);
      el("path", { d: line(dot), fill: "none", stroke: c.opt, "stroke-width": 2.5 }, curveSvg);
      el("line", { x1: xc(Math.log(beta)), x2: xc(Math.log(beta)), y1: yc(-1), y2: yc(3), stroke: c.model, "stroke-width": 1.5, "stroke-dasharray": "4 3" }, curveSvg);
      el("circle", { cx: xc(Math.log(beta)), cy: yc(eRm), r: 5, fill: c.loss, stroke: c.bg, "stroke-width": 2 }, curveSvg);
      el("circle", { cx: xc(Math.log(beta)), cy: yc(eTrue), r: 5, fill: c.opt, stroke: c.bg, "stroke-width": 2 }, curveSvg);
      el("text", { x: M.l + 2, y: 14, style: "fill: var(--viz-loss)" }, curveSvg).textContent = "— reward-model score";
      el("text", { x: M.l + 150, y: 14, style: "fill: var(--viz-opt)" }, curveSvg).textContent = "— true quality";

      out.textContent =
        `β = ${fmt(beta, 2)} · E_π*[reward-model score] = ${fmt(eRm, 3)} · E_π*[true quality] = ${fmt(eTrue, 3)} · KL(π* ‖ π_ref) = ${fmt(kl, 3)}\n` +
        `(starting model π_ref: score ${fmt(REF.reduce((s, x, i) => s + x * rm[i], 0), 3)}, true quality ${fmt(dot(REF), 3)})`;
    }

    slider.addEventListener("input", () => { beta = toBeta(+slider.value); draw(); });
    fix.addEventListener("change", draw);
    presets.addEventListener("click", (e) => {
      if (e.target.dataset.b === undefined) return;
      beta = +e.target.dataset.b;
      slider.value = toSlider(beta);
      draw();
    });
    slider.value = toSlider(beta);
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 3: DPO on the toy replies
  // 200 human comparisons (20 per pair), sampled with Bradley–Terry from TRUE (seed 0, same as the lesson's code).
  // [i, j, times i won, times j won]
  const COUNTS = [[0, 1, 17, 3], [0, 2, 20, 0], [0, 3, 19, 1], [0, 4, 17, 3], [1, 2, 19, 1],
                  [1, 3, 18, 2], [1, 4, 18, 2], [2, 3, 9, 11], [2, 4, 5, 15], [3, 4, 6, 14]];
  const N_PAIRS = COUNTS.reduce((s, [, , a, b]) => s + a + b, 0);
  const LR = 1.0;

  function dpoWidget(root) {
    const [barSvg] = plots(root, 1, BW, BH);
    barSvg.style.maxWidth = "560px";
    barSvg.style.margin = "0 auto";
    const ctrl = control(root, `
      β: <button data-b="0.5">0.5</button><button data-b="1">1</button><button data-b="2">2</button>
      <button data-s="1">1 step</button><button data-s="10">10 steps</button><button data-s="100">100 steps</button><button data-s="play">▶ Play</button><button data-s="reset">Reset</button>`);
    const out = readout(root);
    hint(root, "Dashed outline = π<sub>ref</sub>. Orange = the policy π<sub>θ</sub> being trained. Green tick = where theory says DPO ends up: π<sub>ref</sub> · exp(r̂/β) / Z, with r̂ a reward model fitted to the same comparisons. Under each bar: the implicit reward β·log(π<sub>θ</sub>/π<sub>ref</sub>).");
    const table = hint(root, "");

    let beta = 1, theta, t, timer = null;
    const logRef = REF.map(Math.log);
    // Bradley–Terry maximum-likelihood rewards for the counts (Newton-free: plain gradient ascent, run once)
    const rhat = (() => {
      const r = new Array(K).fill(0);
      for (let it = 0; it < 20000; it++) {
        const g = new Array(K).fill(0);
        for (const [i, j, wi, wj] of COUNTS) {
          const s = sigmoid(r[i] - r[j]);
          g[i] += wi * (1 - s) - wj * s; g[j] -= wi * (1 - s) - wj * s;
        }
        for (let k = 0; k < K; k++) r[k] += (2 / N_PAIRS) * g[k] * 5;
      }
      return r;
    })();
    const softmax = (v) => { const m = Math.max(...v); const e = v.map((x) => Math.exp(x - m)); const z = e.reduce((s, x) => s + x, 0); return e.map((x) => x / z); };

    // DPO loss and gradient. For a categorical policy with logits θ, log π(i) − log π(j) = θ_i − θ_j,
    // so h = β[(θ_w − θ_l) − (log π_ref(w) − log π_ref(l))] and dL/dθ_w = −β σ(−h) / N.
    function lossAndGrad() {
      let loss = 0;
      const g = new Array(K).fill(0);
      for (const [i, j, wi, wj] of COUNTS) {
        const h = beta * ((theta[i] - theta[j]) - (logRef[i] - logRef[j]));   // i beats j
        loss += wi * softplus(-h) + wj * softplus(h);
        const gi = (-wi * beta * sigmoid(-h) + wj * beta * sigmoid(h)) / N_PAIRS;
        g[i] += gi; g[j] -= gi;
      }
      return [loss / N_PAIRS, g];
    }

    function reset() { stop(); theta = logRef.slice(); t = 0; draw(); }
    function step(n) { for (let k = 0; k < n; k++) { const [, g] = lossAndGrad(); theta = theta.map((v, i) => v - LR * g[i]); t++; } draw(); }
    function stop() { clearInterval(timer); timer = null; }

    function draw() {
      const p = softmax(theta), [loss] = lossAndGrad();
      const target = softmax(REF.map((q, i) => Math.log(q) + rhat[i] / beta));
      const logp = p.map(Math.log);
      drawBars(barSvg, p, {
        title: `DPO, β = ${beta}, after ${t} gradient steps`,
        target,
        rows: [[(i) => fmt(beta * (logp[i] - logRef[i]), 2), "fill: var(--viz-loss)"]],
      });
      ctrl.querySelectorAll("[data-b]").forEach((b) => (b.style.borderColor = +b.dataset.b === beta ? "var(--viz-model)" : ""));
      out.textContent =
        `step ${t} · DPO loss = ${fmt(loss, 3)} · E_π[true quality] = ${fmt(dot(p), 3)} · KL(π ‖ π_ref) = ${fmt(klDiv(p), 3)}\n` +
        `(no reward model and no sampling: each step is one gradient step on the 200 labelled pairs)`;
    }

    table.innerHTML = "Training data, 200 human comparisons (wins–losses): " +
      COUNTS.map(([i, j, wi, wj]) => `${NAMES[i][0]} vs ${NAMES[j][0]} ${wi}–${wj}`).join(" · ");

    ctrl.addEventListener("click", (e) => {
      const b = e.target.dataset.b, s = e.target.dataset.s;
      if (b) { beta = +b; reset(); }
      if (s === "reset") reset();
      else if (s === "play") {
        if (timer) { stop(); e.target.textContent = "▶ Play"; return; }
        e.target.textContent = "⏸ Pause";
        timer = setInterval(() => { step(1); if (t >= 300) { stop(); e.target.textContent = "▶ Play"; } }, 60);
      } else if (s) { stop(); ctrl.querySelector("[data-s=play]").textContent = "▶ Play"; step(+s); }
    });
    onThemeChange(draw);
    reset();
  }

  const mounts = { "viz-bt": btWidget, "viz-kl": klWidget, "viz-dpo": dpoWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
