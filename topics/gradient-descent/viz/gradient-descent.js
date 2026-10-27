// Interactive widgets for the Gradient Descent lesson.
// Three widgets, each mounted on a <div> with a matching id:
//   #viz-fit    – drag w, see the line fit and the loss curve with its slope
//   #viz-gd1d   – run gradient descent on one parameter, try different learning rates
//   #viz-gd2d   – run (stochastic) gradient descent on two parameters over a contour map
// Runs after DOMContentLoaded so the site-wide helpers (assets/js/viz-common.js),
// which load at the end of the page, are available.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks, fmt } = window.Viz;

  // Toy dataset: roughly y = 2x, with a little fixed noise
  const X = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
  const Y = [1.3, 1.6, 3.2, 4.5, 4.7, 6.1, 6.8];
  const N = X.length;

  // Model with one parameter: y = w·x, mean squared error loss
  const loss1 = (w) => X.reduce((s, x, i) => s + (w * x - Y[i]) ** 2, 0) / N;
  const grad1 = (w) => X.reduce((s, x, i) => s + 2 * x * (w * x - Y[i]), 0) / N;

  // Model with two parameters: y = w·x + b
  const loss2 = (w, b) => X.reduce((s, x, i) => s + (w * x + b - Y[i]) ** 2, 0) / N;
  function grad2(w, b, idx) {
    // idx = undefined → full-batch gradient, otherwise gradient of a single sample
    const items = idx === undefined ? X.map((_, i) => i) : [idx];
    let gw = 0, gb = 0;
    for (const i of items) {
      const err = w * X[i] + b - Y[i];
      gw += 2 * X[i] * err;
      gb += 2 * err;
    }
    return [gw / items.length, gb / items.length];
  }

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
    parent.appendChild(div);
    return div;
  }

  // ---------------------------------------------------------------- Widget 1
  function fitExplorer(root) {
    const W = 340, H = 260, M = { l: 40, r: 12, t: 12, b: 38 };
    const plots = document.createElement("div");
    plots.className = "viz-plots";
    root.appendChild(plots);
    const left = svg(W, H), right = svg(W, H);
    plots.append(left, right);

    const ctrl = control(root, `
      <label>weight <b>w</b> <input type="range" min="-1" max="5" step="0.01" value="0.5"> <span class="val"></span></label>`);
    const slider = ctrl.querySelector("input"), val = ctrl.querySelector(".val");
    const out = readout(root);

    const xd = linear(0, 4, M.l, W - M.r), yd = linear(-4, 10, H - M.b, M.t);
    const xl = linear(-1, 5, M.l, W - M.r), yl = linear(0, 50, H - M.b, M.t);

    function draw() {
      const c = colors(), w = +slider.value;
      val.textContent = w.toFixed(2);
      left.innerHTML = right.innerHTML = "";

      // Left: data points, the model line, and the errors (residuals)
      axes(left, xd, yd, { xDomain: [0, 4], yDomain: [-4, 10], xTicks: ticks(0, 4, 1), yTicks: ticks(-4, 10, 2), xLabel: "x", yLabel: "y" });
      X.forEach((x, i) => el("line", { x1: xd(x), x2: xd(x), y1: yd(Y[i]), y2: yd(Math.max(-4, Math.min(10, w * x))), stroke: c.grad, "stroke-dasharray": "3 3" }, left));
      const xEnd = 4 * w > 10 ? 10 / w : 4 * w < -4 ? -4 / w : 4;   // keep the line inside the plot
      el("line", { x1: xd(0), y1: yd(0), x2: xd(xEnd), y2: yd(xEnd * w), stroke: c.model, "stroke-width": 3 }, left);
      X.forEach((x, i) => el("circle", { cx: xd(x), cy: yd(Y[i]), r: 5, fill: c.data }, left));
      el("text", { x: M.l + 6, y: M.t + 12 }, left).textContent = "data and the line y = w·x";

      // Right: the loss curve, current position and its tangent (the slope)
      axes(right, xl, yl, { xDomain: [-1, 5], yDomain: [0, 50], xTicks: ticks(-1, 5, 1), yTicks: ticks(0, 50, 10), xLabel: "w", yLabel: "loss" });
      let d = "";
      for (let v = -1; v <= 5.0001; v += 0.05) d += (d ? "L" : "M") + xl(v) + "," + yl(loss1(v));
      el("path", { d, fill: "none", stroke: c.loss, "stroke-width": 2.5 }, right);
      const L = loss1(w), g = grad1(w), span = 0.8;
      el("line", { x1: xl(w - span), y1: yl(L - g * span), x2: xl(w + span), y2: yl(L + g * span), stroke: c.grad, "stroke-width": 2 }, right);
      el("circle", { cx: xl(w), cy: yl(L), r: 6, fill: c.model, stroke: c.bg, "stroke-width": 2 }, right);
      el("text", { x: M.l + 6, y: M.t + 12 }, right).textContent = "loss for every possible w";

      const dir = Math.abs(g) < 0.5 ? "≈ flat: you are near the bottom!" : g > 0 ? "positive → go LEFT (decrease w)" : "negative → go RIGHT (increase w)";
      out.textContent = `loss = ${fmt(L, 2)}   slope dL/dw = ${fmt(g, 2)}   ${dir}`;
    }
    slider.addEventListener("input", draw);
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 2
  function gd1d(root) {
    const W = 340, H = 260, M = { l: 40, r: 12, t: 12, b: 38 };
    const plots = document.createElement("div");
    plots.className = "viz-plots";
    root.appendChild(plots);
    const curve = svg(W, H), hist = svg(W, H);
    plots.append(curve, hist);

    const ctrl = control(root, `
      <label>learning rate <b>η</b> <input type="range" min="0.005" max="0.25" step="0.005" value="0.05"> <span class="val"></span></label>
      <button data-a="step">Step</button><button data-a="run">Run 30 steps</button><button data-a="reset">Reset</button>`);
    const presets = control(root, `Try: <button data-lr="0.01">Too small (0.01)</button><button data-lr="0.05">Good (0.05)</button><button data-lr="0.18">Big (0.18)</button><button data-lr="0.22">Too big (0.22)</button>`);
    const slider = ctrl.querySelector("input"), val = ctrl.querySelector(".val");
    const out = readout(root);

    const START = -0.5, STEPS = 30;
    let path = [START], timer = null;

    const xl = linear(-1, 5, M.l, W - M.r), yl = linear(0, 50, H - M.b, M.t);
    const xh = linear(0, STEPS, M.l, W - M.r), yh = linear(0, 50, H - M.b, M.t);
    const clampY = (v) => Math.min(50, v);

    function step() {
      const w = path[path.length - 1];
      if (Math.abs(w) > 1e3) return;
      path.push(w - +slider.value * grad1(w));
    }

    function draw() {
      const c = colors(), lr = +slider.value;
      val.textContent = lr.toFixed(3);
      curve.innerHTML = hist.innerHTML = "";

      axes(curve, xl, yl, { xDomain: [-1, 5], yDomain: [0, 50], xTicks: ticks(-1, 5, 1), yTicks: ticks(0, 50, 10), xLabel: "w", yLabel: "loss" });
      let d = "";
      for (let v = -1; v <= 5.0001; v += 0.05) d += (d ? "L" : "M") + xl(v) + "," + yl(loss1(v));
      el("path", { d, fill: "none", stroke: c.loss, "stroke-width": 2.5 }, curve);
      const visible = path.filter((w) => w >= -1 && w <= 5);
      for (let i = 1; i < path.length; i++) {
        const a = path[i - 1], b = path[i];
        if (a < -1 || a > 5 || b < -1 || b > 5) continue;
        el("line", { x1: xl(a), y1: yl(loss1(a)), x2: xl(b), y2: yl(loss1(b)), stroke: c.model, "stroke-width": 1.5, opacity: 0.8 }, curve);
      }
      visible.forEach((w, i) => el("circle", { cx: xl(w), cy: yl(loss1(w)), r: i === visible.length - 1 ? 6 : 3.5, fill: c.model }, curve));
      const wStar = X.reduce((s, x, i) => s + x * Y[i], 0) / X.reduce((s, x) => s + x * x, 0);
      el("circle", { cx: xl(wStar), cy: yl(loss1(wStar)), r: 5, fill: "none", stroke: c.opt, "stroke-width": 2 }, curve);
      el("text", { x: M.l + 6, y: M.t + 12 }, curve).textContent = "path of w on the loss curve";

      axes(hist, xh, yh, { xDomain: [0, STEPS], yDomain: [0, 50], xTicks: ticks(0, STEPS, 5), yTicks: ticks(0, 50, 10), xLabel: "step", yLabel: "loss" });
      let h = "";
      path.forEach((w, i) => { if (i <= STEPS) h += (h ? "L" : "M") + xh(i) + "," + yh(clampY(loss1(w))); });
      el("path", { d: h, fill: "none", stroke: c.loss, "stroke-width": 2 }, hist);
      el("text", { x: M.l + 6, y: M.t + 12 }, hist).textContent = "loss after each step";

      const w = path[path.length - 1];
      const status = Math.abs(w) > 5 ? "  ⚠ diverging: steps are too big, w is flying away!" : Math.abs(w - wStar) < 0.01 ? "  ✓ converged to the minimum" : "";
      out.textContent = `step ${path.length - 1}   w = ${fmt(w)}   loss = ${fmt(loss1(w), 3)}${status}`;
    }

    function reset() { clearInterval(timer); timer = null; path = [START]; draw(); }

    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "step") { step(); draw(); }
      if (a === "reset") reset();
      if (a === "run") {
        reset();
        timer = setInterval(() => {
          step(); draw();
          if (path.length > STEPS) { clearInterval(timer); timer = null; }
        }, 120);
      }
    });
    presets.addEventListener("click", (e) => {
      if (!e.target.dataset.lr) return;
      slider.value = e.target.dataset.lr;
      ctrl.querySelector('[data-a="run"]').click();
    });
    slider.addEventListener("input", reset);
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 3
  function gd2d(root) {
    const CW = 600, CH = 420, M = { l: 44, r: 12, t: 12, b: 36 };
    const canvas = document.createElement("canvas");
    canvas.width = CW; canvas.height = CH;
    canvas.style.cursor = "crosshair";
    root.appendChild(canvas);
    const hint = document.createElement("div");
    hint.className = "viz-hint";
    hint.textContent = "Click anywhere on the map to choose a starting point.";
    root.appendChild(hint);

    const ctrl = control(root, `
      <label>learning rate <b>η</b> <input type="range" min="0.005" max="0.2" step="0.005" value="0.1"> <span class="val"></span></label>
      <label>gradient from <select><option value="full">all data (batch GD)</option><option value="sgd">1 random point (SGD)</option></select></label>
      <button data-a="run">Run</button><button data-a="reset">Reset</button>`);
    const slider = ctrl.querySelector("input"), val = ctrl.querySelector(".val"), mode = ctrl.querySelector("select");
    const out = readout(root);
    const ctx = canvas.getContext("2d");

    const WD = [-1, 5], BD = [-4, 6], MAX_STEPS = 200;
    const xw = linear(WD[0], WD[1], M.l, CW - M.r), yb = linear(BD[0], BD[1], CH - M.b, M.t);
    let start = [-0.5, 5], path = [start.slice()], timer = null, background = null;

    // Pre-render the contour map once per theme (it never changes otherwise)
    function renderBackground() {
      const c = colors();
      const off = document.createElement("canvas");
      off.width = CW; off.height = CH;
      const g = off.getContext("2d");
      g.fillStyle = c.bg; g.fillRect(0, 0, CW, CH);
      const img = g.getImageData(0, 0, CW, CH);
      const base = hexToRgb(c.loss), bg = hexToRgb(c.bg);
      const levels = 14, lo = Math.log(0.05), hi = Math.log(80);
      for (let py = M.t; py < CH - M.b; py++) {
        for (let px = M.l; px < CW - M.r; px++) {
          const L = loss2(xw.invert(px), yb.invert(py));
          const t = Math.max(0, Math.min(1, (Math.log(L) - lo) / (hi - lo)));
          const band = Math.floor(t * levels) / levels;     // quantise → visible contour bands
          const a = 0.85 - 0.75 * band;                     // low loss = strong colour
          const k = (py * CW + px) * 4;
          img.data[k] = bg[0] + (base[0] - bg[0]) * a;
          img.data[k + 1] = bg[1] + (base[1] - bg[1]) * a;
          img.data[k + 2] = bg[2] + (base[2] - bg[2]) * a;
          img.data[k + 3] = 255;
        }
      }
      g.putImageData(img, 0, 0);
      // axes labels
      g.fillStyle = c.muted; g.font = "12px sans-serif"; g.textAlign = "center";
      for (const t of ticks(WD[0], WD[1], 1)) g.fillText(t, xw(t), CH - M.b + 16);
      g.fillText("w (slope)", (M.l + CW - M.r) / 2, CH - 4);
      g.textAlign = "right";
      for (const t of ticks(BD[0], BD[1], 2)) g.fillText(t, M.l - 6, yb(t) + 4);
      g.save(); g.translate(12, (M.t + CH - M.b) / 2); g.rotate(-Math.PI / 2); g.textAlign = "center";
      g.fillText("b (intercept)", 0, 0); g.restore();
      background = off;
    }

    function hexToRgb(hex) {
      const h = hex.replace("#", "");
      return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
    }

    // Exact minimum via least squares, used to mark the target
    const mx = X.reduce((a, b) => a + b) / N, my = Y.reduce((a, b) => a + b) / N;
    const wOpt = X.reduce((s, x, i) => s + (x - mx) * (Y[i] - my), 0) / X.reduce((s, x) => s + (x - mx) ** 2, 0);
    const bOpt = my - wOpt * mx;

    function draw() {
      const c = colors();
      val.textContent = (+slider.value).toFixed(3);
      ctx.drawImage(background, 0, 0);

      ctx.save();
      ctx.beginPath(); ctx.rect(M.l, M.t, CW - M.l - M.r, CH - M.t - M.b); ctx.clip();
      ctx.strokeStyle = c.model; ctx.lineWidth = 2; ctx.beginPath();
      path.forEach(([w, b], i) => (i ? ctx.lineTo(xw(w), yb(b)) : ctx.moveTo(xw(w), yb(b))));
      ctx.stroke();
      ctx.fillStyle = c.model;
      for (const [w, b] of path) { ctx.beginPath(); ctx.arc(xw(w), yb(b), 2.5, 0, 7); ctx.fill(); }
      const [cw, cb] = path[path.length - 1];
      ctx.beginPath(); ctx.arc(xw(cw), yb(cb), 6, 0, 7); ctx.fill();
      ctx.strokeStyle = c.opt; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(xw(wOpt), yb(bOpt), 7, 0, 7); ctx.stroke();
      ctx.restore();

      const L = loss2(cw, cb);
      const status = !Number.isFinite(L) || L > 1e4 ? "  ⚠ diverged" : Math.hypot(cw - wOpt, cb - bOpt) < 0.02 ? "  ✓ reached the minimum" : "";
      out.textContent = `step ${path.length - 1}   w = ${fmt(cw, 2)}   b = ${fmt(cb, 2)}   loss = ${fmt(L, 3)}${status}`;
    }

    function step() {
      const [w, b] = path[path.length - 1];
      if (!Number.isFinite(w) || Math.abs(w) > 1e3) return false;
      const idx = mode.value === "sgd" ? Math.floor(Math.random() * N) : undefined;
      const [gw, gb] = grad2(w, b, idx), lr = +slider.value;
      path.push([w - lr * gw, b - lr * gb]);
      return true;
    }

    function reset() { clearInterval(timer); timer = null; path = [start.slice()]; draw(); }

    ctrl.addEventListener("click", (e) => {
      const a = e.target.dataset.a;
      if (a === "reset") reset();
      if (a === "run") {
        reset();
        timer = setInterval(() => {
          if (!step() || path.length > MAX_STEPS) { clearInterval(timer); timer = null; }
          draw();
        }, 30);
      }
    });
    slider.addEventListener("input", reset);
    mode.addEventListener("change", reset);
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * CW, py = ((e.clientY - r.top) / r.height) * CH;
      if (px < M.l || px > CW - M.r || py < M.t || py > CH - M.b) return;
      start = [xw.invert(px), yb.invert(py)];
      ctrl.querySelector('[data-a="run"]').click();
    });
    onThemeChange(() => { renderBackground(); draw(); });
    renderBackground();
    draw();
  }

  const mounts = { "viz-fit": fitExplorer, "viz-gd1d": gd1d, "viz-gd2d": gd2d };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
