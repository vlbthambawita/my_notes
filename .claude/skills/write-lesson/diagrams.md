# HTML diagrams (the replacement for Mermaid)

Diagrams are drawn by `docs/assets/js/diagram.js` (loaded on every page through `mkdocs.yml`) and styled by
the "Diagrams" section of `docs/assets/css/viz.css`. Boxes are real HTML (wrapping text, site fonts,
light/dark themes); arrows are an SVG layer that is redrawn on resize.

## Embedding

Put this directly in the lesson Markdown, with a blank line before and after:

```html
<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 The gradient-descent loop",
  "cols": 5,
  "rows": 2.3,
  "rowHeight": 95,
  "nodes": [
    {"id": "pred", "col": 1, "row": 0.15, "icon": "🔮", "label": "Make predictions", "role": "model", "note": "Shown on hover."}
  ],
  "edges": [
    {"from": "step", "to": "pred", "label": "repeat", "curve": 0.42, "narrowCurve": 0}
  ],
  "steps": [
    {"show": ["init>pred", "pred"], "text": "Caption shown for this step (HTML allowed)."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 95, "pos": {"pred": [0, 1]}}
}
</script>
</div>
```

Keep one node/edge/step per line so the spec stays readable. It must be valid JSON: double quotes and no
trailing commas. Use Unicode and `<sub>`/`<b>` for math in diagrams (`θ ← θ − η∇L`, `s<sub>t+1</sub>`), not `$…$`.

## Spec reference

| Key | Meaning |
|---|---|
| `title` | Panel title (start with an emoji like the widgets do). |
| `cols`, `rows` | Layout grid. Fractions are fine (`rows: 2.6`). Width is split evenly into `cols`. |
| `rowHeight` | Pixels per row (default 100). Stage height = `rows × rowHeight`. |
| `caption` | Text shown under the diagram before any step is selected. |
| `interval` | Milliseconds per step when playing (default 2200). |
| `narrow` | Layout used below 560px width: overrides `cols`, `rows`, `rowHeight`, and `pos: {id: [col, row]}` (groups: `[col, row, cols, rows]`). Boxes switch to a compact style automatically. |

**Nodes**: `{"id", "col", "row", "label", "icon", "sub", "role", "shape", "note", "span", "maxWidth"}`

- `col`/`row` give the **centre** cell: the centre is at `(col + 0.5, row + 0.5)` cells. Fractions place nodes between cells.
- `label` (HTML allowed; `<b>` takes the role colour), `icon` (emoji), `sub` (small muted second line).
- `role`: `data` (blue), `model` (orange), `loss` (purple), `grad` (red), `opt` (green), or omit for neutral grey.
  Use the same meaning as the widgets.
- `shape`: `pill` (start/end), `decision` (dashed pill, for yes/no questions).
- `note`: shown in the caption on hover (adds a help cursor).
- `span`: width multiplier in the wide layout (ignored in `narrow`); `maxWidth` caps the box (default 190px).

**Groups**: `{"id", "label", "col", "row", "cols", "rows"}` draws a dashed container (e.g. "Repeat for many
epochs"). Its position is the cell's **top-left corner** plus a size. Edges can start or end at a group `id`.

**Edges**: `{"from", "to", "label", "curve", "narrowCurve", "dashed", "id"}`

- `curve`: bend as a fraction of the arrow's length; positive bends to the **left of the direction of travel**
  (a rightward arrow bends up, a leftward arrow bends down, a downward arrow bends right). Use 0.3–0.45 for
  loop-back arrows that must clear a row of boxes.
- `narrowCurve` overrides `curve` in the narrow layout (often `0` when the narrow layout makes it a straight line).
- `id` defaults to `"from>to"`; set it when two edges join the same pair of nodes.

**Steps**: `{"show": [ids…], "text": "…"}` turns on the Prev / Next / Play / Reset walkthrough. Listed nodes,
groups and edges (`"from>to"` or the edge `id`) are highlighted, everything else dims, active arrows animate,
and `text` appears in the caption. Add steps to every diagram that shows a process. Steps are what make it
richer than a static flowchart.

## Layout recipes

- **Pipeline with a loop back**: put the nodes in one row near the top (`row: 0.15`, `rows: 2.3`) and give
  the return edge `curve: 0.42` so it arcs under the row. Narrow: a 2×2 ring with `narrowCurve: 0`.
- **Two-party loop** (agent ↔ environment): two nodes in cols 0 and 2 of 3. Forward edge `curve: 0.3`;
  return edges with curves `0.1` and `0.38` so their labels stack.
- **Training loop inside a container**: a `group` around a ring of nodes (top row left→right, then down,
  then back left, then up); outside nodes connect to the group `id`. Leave about 0.2 columns between the
  group and outside nodes so the arrows are visible.
- **Tree / taxonomy**: parents at fractional columns centred over their children (`col: 1.5` over 0 and 3),
  `rowHeight` ≥ 100 so arrows between rows are long enough to read.
- **Phones**: always give a `narrow` layout for anything wider than 3 columns. Two columns, top to bottom,
  with the loop-back running up the left column. Check that long diagonal arrows don't cross boxes; bend
  them with `narrowCurve` when they do.
