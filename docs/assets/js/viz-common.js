// Small helpers shared by all lesson widgets (no external libraries).
window.Viz = (function () {
  const NS = "http://www.w3.org/2000/svg";

  // Read the current colour tokens defined in assets/css/viz.css
  function colors() {
    const s = getComputedStyle(document.body);
    const get = (name) => s.getPropertyValue(name).trim();
    return {
      bg: get("--viz-bg"), text: get("--viz-text"), muted: get("--viz-muted"),
      grid: get("--viz-grid"), border: get("--viz-border"),
      data: get("--viz-data"), model: get("--viz-model"), loss: get("--viz-loss"),
      grad: get("--viz-grad"), opt: get("--viz-opt"),
    };
  }

  // Call fn whenever the user toggles light/dark mode
  function onThemeChange(fn) {
    new MutationObserver(fn).observe(document.body, {
      attributes: true, attributeFilter: ["data-md-color-scheme"],
    });
  }

  // Map a value from domain [d0, d1] to range [r0, r1]
  function linear(d0, d1, r0, r1) {
    const f = (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
    f.invert = (p) => d0 + ((p - r0) / (r1 - r0)) * (d1 - d0);
    return f;
  }

  // Create an SVG element with attributes, optionally appended to parent
  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs || {}) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function svg(width, height) {
    return el("svg", { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "xMidYMid meet" });
  }

  // Draw light grid lines and labelled axes into an SVG
  function axes(root, x, y, opts) {
    const c = colors();
    const g = el("g", {}, root);
    const [x0, x1] = opts.xDomain, [y0, y1] = opts.yDomain;
    for (const t of opts.xTicks) {
      el("line", { x1: x(t), x2: x(t), y1: y(y0), y2: y(y1), stroke: c.grid }, g);
      el("text", { x: x(t), y: y(y0) + 14, "text-anchor": "middle" }, g).textContent = t;
    }
    for (const t of opts.yTicks) {
      el("line", { x1: x(x0), x2: x(x1), y1: y(t), y2: y(t), stroke: c.grid }, g);
      el("text", { x: x(x0) - 6, y: y(t) + 4, "text-anchor": "end" }, g).textContent = t;
    }
    if (opts.xLabel) el("text", { x: (x(x0) + x(x1)) / 2, y: y(y0) + 30, "text-anchor": "middle" }, g).textContent = opts.xLabel;
    if (opts.yLabel) el("text", { x: 12, y: (y(y0) + y(y1)) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(y(y0) + y(y1)) / 2})` }, g).textContent = opts.yLabel;
    return g;
  }

  function ticks(a, b, step) {
    const out = [];
    for (let t = Math.ceil(a / step) * step; t <= b + 1e-9; t += step) out.push(+t.toFixed(6));
    return out;
  }

  function fmt(v, d = 3) {
    return Number.isFinite(v) ? v.toFixed(d) : "∞";
  }

  return { colors, onThemeChange, linear, el, svg, axes, ticks, fmt };
})();
