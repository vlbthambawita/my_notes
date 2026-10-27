// Rich HTML flow diagrams (the site's replacement for Mermaid).
// A diagram is written directly in a lesson's Markdown as a JSON spec:
//
//   <div class="viz diagram">
//   <script type="application/json">
//   { "title": "...", "cols": 4, "rows": 2, "nodes": [...], "edges": [...], "steps": [...] }
//   </script>
//   </div>
//
// Boxes are HTML (so they wrap text, use the site fonts and follow light/dark mode),
// arrows are an SVG layer underneath that is redrawn whenever the diagram resizes.
// The full spec format is documented in .claude/skills/write-lesson/diagrams.md.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const NARROW = 560;           // below this width the optional `narrow` layout is used
  let uid = 0;

  function div(cls, parent, html) {
    const d = document.createElement("div");
    d.className = cls;
    if (html !== undefined) d.innerHTML = html;
    if (parent) parent.appendChild(d);
    return d;
  }

  function svgEl(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  // Point where a ray from the centre of box b in direction (dx, dy) leaves the box (plus a small gap)
  function clip(b, dx, dy, gap = 4) {
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    const t = Math.min(ux ? (b.hw + gap) / Math.abs(ux) : Infinity, uy ? (b.hh + gap) / Math.abs(uy) : Infinity);
    return [b.x + ux * t, b.y + uy * t];
  }

  function build(root, spec) {
    const id = "dg" + uid++;
    if (spec.title) div("viz-title", root, spec.title);
    const stage = div("dg-stage", root);
    const svg = svgEl("svg", { class: "dg-arrows" }, stage);
    const defs = svgEl("defs", {}, svg);
    for (const kind of ["base", "active"]) {
      const m = svgEl("marker", { id: `${id}-${kind}`, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
      svgEl("path", { d: "M0,0 L10,5 L0,10 z", class: `dg-head dg-head-${kind}` }, m);
    }

    const groups = (spec.groups || []).map((g) => {
      const el = div("dg-group", stage);
      div("dg-group-label", el, g.label || "");
      return { spec: g, el };
    });

    const nodes = spec.nodes.map((n) => {
      const el = div("dg-node", stage);
      el.dataset.role = n.role || "neutral";
      if (n.shape) el.dataset.shape = n.shape;
      if (n.icon) div("dg-icon", el, n.icon);
      div("dg-label", el, n.label);
      if (n.sub) div("dg-sub", el, n.sub);
      return { spec: n, el };
    });

    const edges = (spec.edges || []).map((e) => {
      const path = svgEl("path", { class: "dg-edge" + (e.dashed ? " dg-dashed" : ""), "marker-end": `url(#${id}-base)` }, svg);
      const label = e.label ? div("dg-edge-label", stage, e.label) : null;
      return { spec: e, id: e.id || `${e.from}>${e.to}`, path, label };
    });

    const caption = div("dg-caption", root);
    const steps = spec.steps || [];
    let current = -1;

    function layout() {
      const w = stage.clientWidth;
      if (!w) return;
      const narrow = spec.narrow && w < NARROW;
      const L = narrow ? { ...spec, ...spec.narrow } : spec;
      const pos = (narrow && spec.narrow.pos) || {};
      const rowH = L.rowHeight || 100, cellW = w / L.cols;
      stage.style.height = L.rows * rowH + "px";
      stage.classList.toggle("dg-compact", narrow || cellW < 120);
      const boxes = {};

      for (const g of groups) {
        const [c, r, cs, rs] = pos[g.spec.id] || [g.spec.col, g.spec.row, g.spec.cols, g.spec.rows];
        Object.assign(g.el.style, { left: c * cellW + 6 + "px", top: r * rowH + 4 + "px", width: cs * cellW - 12 + "px", height: rs * rowH - 8 + "px" });
        if (g.spec.id) boxes[g.spec.id] = { x: (c + cs / 2) * cellW, y: (r + rs / 2) * rowH, hw: (cs * cellW - 12) / 2, hh: (rs * rowH - 8) / 2 };
      }
      for (const n of nodes) {
        const [c, r] = pos[n.spec.id] || [n.spec.col, n.spec.row];
        const span = narrow ? 1 : n.spec.span || 1;     // spans are for the wide layout only
        const width = Math.min(cellW * (narrow ? 0.94 : 0.88) * span, n.spec.maxWidth || 190);
        Object.assign(n.el.style, { left: (c + 0.5) * cellW + "px", top: (r + 0.5) * rowH + "px", width: width + "px" });
      }
      for (const n of nodes) {
        const [c, r] = pos[n.spec.id] || [n.spec.col, n.spec.row];
        boxes[n.spec.id] = { x: (c + 0.5) * cellW, y: (r + 0.5) * rowH, hw: n.el.offsetWidth / 2, hh: n.el.offsetHeight / 2 };
      }

      svg.setAttribute("viewBox", `0 0 ${w} ${L.rows * rowH}`);
      for (const e of edges) {
        const a = boxes[e.spec.from], b = boxes[e.spec.to];
        const k = narrow && e.spec.narrowCurve !== undefined ? e.spec.narrowCurve : e.spec.curve || 0;
        // Bend: push the midpoint sideways by k × length (k > 0 bends to the left of travel).
        // First aim from the box centres to find where the arrow leaves each box,
        // then rebuild the control point from those clipped ends so short arrows stay straight.
        const bend = (p, q) => [(p[0] + q[0]) / 2 + k * (q[1] - p[1]), (p[1] + q[1]) / 2 - k * (q[0] - p[0])];
        const [ax, ay] = bend([a.x, a.y], [b.x, b.y]);
        const p0 = clip(a, ax - a.x, ay - a.y), p2 = clip(b, ax - b.x, ay - b.y, 6);
        const [cx, cy] = bend(p0, p2);
        e.path.setAttribute("d", `M${p0[0]},${p0[1]} Q${cx},${cy} ${p2[0]},${p2[1]}`);
        if (e.label) {
          const lx = 0.25 * p0[0] + 0.5 * cx + 0.25 * p2[0], ly = 0.25 * p0[1] + 0.5 * cy + 0.25 * p2[1];
          Object.assign(e.label.style, { left: lx + "px", top: ly + "px" });
        }
      }
    }

    // ---- step-through walkthrough and hover notes
    const defaultText = spec.caption || (steps.length ? "Press <b>Next</b> to walk through the diagram, or hover over a box." : "Hover over a box for details.");

    function show(i) {
      current = i;
      const step = steps[i];
      const on = new Set(step ? step.show : []);
      stage.classList.toggle("dg-stepping", !!step);
      for (const n of nodes) n.el.classList.toggle("dg-active", on.has(n.spec.id));
      for (const g of groups) g.el.classList.toggle("dg-active", on.has(g.spec.id));
      for (const e of edges) {
        const act = on.has(e.id);
        e.path.classList.toggle("dg-active", act);
        e.path.setAttribute("marker-end", `url(#${id}-${act ? "active" : "base"})`);
        if (e.label) e.label.classList.toggle("dg-active", act);
      }
      caption.innerHTML = step ? `<span class="dg-step-no">${i + 1}/${steps.length}</span> ${step.text}` : defaultText;
      if (counter) counter.textContent = step ? `step ${i + 1} of ${steps.length}` : "";
    }

    let counter = null, timer = null;
    if (steps.length) {
      const ctrl = div("viz-controls", root, `<button data-a="prev">◀ Prev</button><button data-a="next">Next ▶</button><button data-a="play">▶ Play</button><button data-a="reset">Reset</button><span class="dg-counter"></span>`);
      counter = ctrl.querySelector(".dg-counter");
      const playBtn = ctrl.querySelector('[data-a="play"]');
      const stop = () => { clearInterval(timer); timer = null; playBtn.textContent = "▶ Play"; };
      ctrl.addEventListener("click", (ev) => {
        const a = ev.target.dataset.a;
        if (!a) return;
        if (a !== "play") stop();
        if (a === "next") show(Math.min(current + 1, steps.length - 1));
        if (a === "prev") show(Math.max(current - 1, -1));
        if (a === "reset") show(-1);
        if (a === "play") {
          if (timer) return stop();
          playBtn.textContent = "⏸ Pause";
          show((current + 1) % steps.length);
          timer = setInterval(() => show((current + 1) % steps.length), spec.interval || 2200);
        }
      });
    }

    for (const n of nodes) {
      if (!n.spec.note) continue;
      n.el.classList.add("dg-has-note");
      n.el.addEventListener("mouseenter", () => { caption.innerHTML = `<b>${n.el.querySelector(".dg-label").textContent}:</b> ${n.spec.note}`; });
      n.el.addEventListener("mouseleave", () => show(current));
    }

    new ResizeObserver(layout).observe(stage);
    layout();
    show(-1);
  }

  document.addEventListener("DOMContentLoaded", function () {
    for (const root of document.querySelectorAll(".diagram")) {
      if (root.dataset.ready) continue;
      root.dataset.ready = "1";
      const src = root.querySelector('script[type="application/json"]');
      try {
        build(root, JSON.parse(src.textContent));
      } catch (err) {
        root.textContent = "Diagram error: " + err.message;
        console.error(err);
      }
    }
  });
})();
