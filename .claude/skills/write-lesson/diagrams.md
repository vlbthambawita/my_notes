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
  "cols": 3,
  "rows": 2,
  "rowHeight": 100,
  "nodes": [
    {"id": "pred", "col": 1, "row": 0, "icon": "🔮", "label": "Make predictions", "role": "model", "note": "Shown on hover."}
  ],
  "edges": [
    {"from": "step", "to": "pred", "label": "repeat"}
  ],
  "steps": [
    {"show": ["init>pred", "pred"], "text": "Caption shown for this step (HTML allowed)."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"pred": [0, 1]}}
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
| `rowHeight` | **Minimum** pixels per row (default 100). Rows grow automatically to the tallest box + 48px, so wrapped text never pushes a box into the next row. Stage height = `rows × row height`. |
| `caption` | Text shown under the diagram before any step is selected. |
| `interval` | Milliseconds per step when playing (default 2200). |
| `narrow` | Layout used below 560px width: overrides `cols`, `rows`, `rowHeight`, and `pos: {id: [col, row]}` (groups: `[col, row, cols, rows]`). `"hideSub": true` hides subtitles on phones when boxes need to be smaller. Boxes switch to a compact style automatically. |

Box widths are automatic: cell width × `span` minus a 44px gap (36px on phones), capped at `maxWidth`.
Neighbouring boxes therefore always leave room for a visible arrow. Don't fight this with fractional columns.

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
They attach to the part of the border facing the other box, so an arrow into a group runs straight
(horizontal or vertical) whenever the other box is level with some part of the group.

**Edges**: `{"from", "to", "label", "curve", "narrowCurve", "dashed", "id"}`

- `curve`: bend as a fraction of the arrow's length; positive bends to the **left of the direction of travel**
  (a rightward arrow bends up, a leftward arrow bends down, a downward arrow bends right). Prefer straight
  arrows on a ring layout (rule 3 below); use curves mainly to separate two arrows between the same boxes.
- `narrowCurve` overrides `curve` in the narrow layout (often `0` when the narrow layout makes it a straight line).
- `id` defaults to `"from>to"`; set it when two edges join the same pair of nodes.

**Steps**: `{"show": [ids…], "text": "…"}` turns on the Prev / Next / Play / Reset walkthrough. Listed nodes,
groups and edges (`"from>to"` or the edge `id`) are highlighted, everything else dims, active arrows animate,
and `text` appears in the caption. Add steps to every diagram that shows a process. Steps are what make it
richer than a static flowchart.

## Layout rules (learned from fixing real overlaps)

These come from bugs that actually shipped: overlapping text, slanted arrows, boxes sticking out, words split
mid-word. Follow them and the checker below will usually pass first time.

1. **Design for a 660px-wide stage.** At 1440px, Material shows both sidebars and the content column is only
   about 660px wide. Use **at most 4 columns** in the wide layout (3 is best for boxes with two lines of text)
   and **at most 2 columns** on phones (2.4 when two boxes need room for labelled arrows between them).
   With more columns, boxes shrink to ~90px and words like "parameters" break mid-word.
2. **Stay on the grid.** Boxes in the same row get the *same* `row` value, and boxes in the same column the
   same `col`. "Nearly aligned" (row 0 vs 0.15, col 1.5 vs 1.6) looks like a mistake and gives slightly
   slanted arrows. Use fractional positions only to centre a parent over its children, or to shift a whole
   block (see rule 6).
3. **Loops are rings, not arcs.** Lay a cycle out around a rectangle (e.g. 3×2: top row left→right, down,
   bottom row right→left, up) so every arrow is straight. Avoid long `curve`d arrows that jump back across a
   row of boxes: they graze boxes and their labels land on other arrows.
4. **Diagonals only across an empty cell.** The only acceptable slanted arrow joins two boxes at opposite
   corners of an empty cell (e.g. "no" from *Episode over?* back to *Observe*).
5. **Short edge labels** (≤ 12 characters, like "no", "repeat", "next batch", "state s<sub>t+1</sub>").
   Long explanations belong in step `text` or node `note`, not on arrows.
6. **Containers need breathing room.** A group's border sits closer to the next box than a box would.
   Start a group 0.15 row below a full row step (e.g. a box at `row: 0`, then the group at `row: 1.15`,
   with the boxes inside at `row: 1.6` and `2.75`), and leave ≥ 0.4 row between the group's top and
   its first boxes so the group label stays clear.
7. **Two arrows between the same pair** (agent → environment, environment → agent) need different `curve`
   values (e.g. 0.1 and 0.38) so their labels don't stack on top of each other.
8. **Phones get their own layout** whenever the wide one has 3+ columns: re-flow the same ring into 2
   columns. Use `hideSub` if the boxes are still too wide for the arrows between them.

## Layout recipes

- **Process loop** (gradient descent): 3×2 ring. `init` (0,0) → `pred` (1,0) → `loss` (2,0) ↓ `grad` (2,1) →
  `step` (1,1) ↑ back to `pred`. Phone: `init` (0,0), ring in rows 1–2 of 2 columns.
- **Loop with an exit and a restart** (RL episodes): 3×2 ring `reset` (0,0) → `obs` (1,0) → `pol` (2,0) ↓
  `env` (2,1) → `learn` (1,1) → `end` (0,1); "yes" goes straight up to `reset`, "no" goes diagonally across to
  `obs`. Phone: 2×3 ring with the same idea.
- **Two-party loop** (agent ↔ environment): boxes in cols 0 and 2 of 3 at `row: 0.45` of `rows: 1.9`.
  Forward edge `curve: 0.3`; return edges `curve: 0.1` and `0.38`. Phone: `cols: 2.4`, positions 0 and 1.4,
  `hideSub: true`.
- **Training loop inside a container**: top-to-bottom. Start box centred above (`col: 1, row: 0`), group
  below it (rule 6) holding a 3×2 ring, result box centred below the group. Arrows into and out of the group
  are then perfectly vertical.
- **Tree / taxonomy**: parents at fractional columns centred over their children (`col: 1.5` over 0 and 3).
  On phones, put a parent's single child beside it (same row) rather than squeezing three levels into two columns.

## Checking a diagram

Always run the layout checker after adding or changing a diagram, and fix everything it reports:

```bash
mkdocs build -q -d <scratchpad>/site && node scripts/check_diagrams.mjs <scratchpad>/site [page-filter]
```

It loads each page in headless Chromium at 1440, 1100, 820 and 390px and reports, per diagram, overlapping
or touching boxes, boxes sticking out of the stage, labels on boxes or on other labels, arrows through boxes
or under other labels, arrows too short to see (< 22px), words broken mid-word, text overflowing its box,
and boxes or arrows that are *nearly* aligned. It exits non-zero on any problem. The checker can't judge
looks, so still take a screenshot at 1440px and 390px (light and dark) and look at it: is anything cramped,
lopsided, or leaving big empty bands?
