// Interactive widgets for the "Confidence, trustworthiness and composites" Tsetlin machine lesson.
// Widgets, each mounted on a <div> with a matching id:
//   #viz-students     – the book's six-student example: confidence = |vote sum|, accuracy per confidence level
//   #viz-confidence   – real patients: vote sums, a confidence threshold, and the coverage/accuracy trade-off
//   #viz-reliability  – how often the positive class really occurs at each vote sum (with 95% intervals)
//   #viz-composite    – three specialist TMs, normalised and added: accuracy and a patient-by-patient view
// The vote sums below come from Tsetlin machines trained in Python (numpy, seed 1, 5-fold cross-validation),
// so every vote is from a machine that never saw that patient. The lesson's numbers come from the same run.
document.addEventListener("DOMContentLoaded", function () {
  const { colors, onThemeChange, linear, el, svg, axes, ticks } = window.Viz;
  const textStyle = "fill: var(--viz-text)";
  // Tables inside widgets have their own class, so Material's table styles don't apply; style them here.
  const css = document.createElement("style");
  css.textContent = ".tmc-table{border-collapse:collapse;margin:0 auto;font-size:0.72rem;background:var(--viz-bg);color:var(--viz-text)}" +
    ".tmc-table th,.tmc-table td{border:1px solid var(--viz-border);padding:3px 7px;text-align:center}" +
    ".tmc-table th{background:var(--viz-panel);font-weight:700}.tmc-narrow th,.tmc-narrow td{padding:3px 4px}";
  document.head.appendChild(css);

  // Wisconsin Diagnostic Breast Cancer data (y = 1: malignant). all20: one TM on all 30 measurements.
  // spec: three specialists; nf = normalisation factor (highest − lowest training vote sum) per fold.
  const WDBC = {"y":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0,1,0,1,1,0,0,0,0,1,0,1,1,0,1,0,1,1,0,0,0,1,1,0,1,1,1,0,0,0,1,0,0,1,1,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,1,1,0,1,0,1,1,0,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,0,1,1,1,0,1,0,1,0,0,1,0,1,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,1,0,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0],"fold":[1,4,2,0,4,3,0,3,0,4,4,0,4,4,3,2,3,2,3,3,3,2,2,2,4,1,0,3,2,4,4,2,4,3,1,1,4,1,4,3,1,1,0,1,2,1,1,1,0,0,0,4,3,1,3,2,1,4,2,3,2,0,0,0,3,0,1,4,1,3,1,3,2,2,1,4,0,2,2,0,2,1,2,3,4,0,2,4,0,1,0,3,0,3,1,4,3,2,1,3,1,4,4,3,2,3,1,4,2,4,3,4,0,0,3,4,1,2,0,2,3,2,3,1,4,1,3,0,4,3,4,1,1,2,3,3,2,3,2,4,4,4,1,1,3,2,1,4,2,2,1,0,1,3,3,4,3,4,4,3,4,3,3,2,3,4,2,2,2,1,2,0,4,2,1,4,4,2,2,4,3,3,2,4,4,4,0,0,0,0,1,0,4,1,4,3,2,2,4,0,2,1,1,4,3,3,1,0,4,3,2,0,0,4,0,2,0,0,0,2,0,0,2,0,2,3,3,1,3,0,3,3,4,4,2,3,4,0,3,0,1,1,2,3,3,2,4,1,4,1,1,4,3,3,3,1,1,0,4,0,1,4,0,0,0,1,0,2,1,3,1,0,2,1,0,1,1,4,0,1,4,2,3,1,0,1,3,2,3,3,1,0,4,0,0,3,2,4,1,4,0,1,1,2,4,1,1,1,2,0,3,0,2,3,1,0,2,2,0,2,3,1,4,3,1,1,2,2,3,0,4,0,3,0,0,3,2,2,3,0,0,1,0,1,4,4,3,0,3,4,3,4,0,1,2,0,4,2,1,1,4,2,2,0,1,3,4,0,3,3,0,3,2,0,1,2,3,4,3,1,1,4,4,2,0,2,1,2,2,1,1,2,2,1,0,2,1,3,3,0,0,2,1,3,3,3,3,4,0,3,0,1,0,4,4,3,2,2,2,2,4,2,4,2,3,2,2,4,2,4,0,2,0,2,2,4,1,4,2,0,0,4,2,4,1,2,1,1,2,3,4,1,3,4,3,3,0,1,1,0,2,4,1,3,0,2,1,0,1,2,2,3,4,0,1,1,4,4,0,0,4,3,2,0,4,2,1,2,4,1,3,4,2,2,3,3,2,0,4,4,4,1,0,1,1,1,4,3,3,2,4,2,3,0,0,0,3,3,3,4,0,0,0,4,3,0,0,2,3,0,2,0,4,4,0,4,2,0,3,2,1,4,4,3,4,1,1,0,0,1,0,1,4,0,1,4,3,1,3,1,1,1,4,1,1,0,0,2,3],"all20":[7,3,10,1,3,1,10,4,2,2,0,9,7,4,2,8,1,9,8,-5,-5,-8,3,7,10,9,7,9,9,2,10,1,10,9,8,10,6,-10,0,1,-10,0,8,6,-1,9,-9,3,-8,-3,-5,-8,-8,10,0,-7,9,9,-6,-7,-3,-5,9,-4,2,8,-6,-8,0,-7,6,-1,10,-1,-10,7,-4,5,9,-10,-2,0,9,7,-5,9,8,8,-7,7,-1,5,-1,-7,10,8,-8,-6,-10,5,0,-5,-7,-6,-6,4,-4,-6,9,-3,-5,-2,0,-5,-6,-6,-6,8,9,2,-7,8,9,-1,-3,-10,0,4,4,10,-7,8,9,0,8,-7,-6,-8,5,-1,-8,5,-10,-9,-7,-2,0,1,-1,-7,-6,0,0,-7,-5,-9,9,1,-8,-8,-1,2,10,-5,9,-3,-8,2,7,-4,-8,-1,4,-7,-10,-9,-2,8,-8,-8,10,8,1,-2,0,-7,2,-8,-8,-9,3,-2,-7,0,5,-7,6,3,9,2,-8,9,10,9,-4,0,-9,0,-1,-1,8,-10,4,4,9,1,-5,-7,10,8,-8,-7,-8,7,-8,-1,-8,1,-6,2,7,-5,-7,6,-9,-9,10,4,-2,10,-10,-10,1,-3,9,-5,-6,-1,-7,-10,10,-8,9,6,9,8,10,8,10,5,8,0,7,-1,9,8,-5,-7,-8,-3,-4,-9,10,-8,1,-1,-10,0,-6,-8,10,-2,10,8,-6,-10,-2,-8,-1,-5,1,0,-6,-9,-6,-7,-9,0,-3,-7,10,-8,10,-9,-9,-7,-10,-10,-8,-5,-4,-2,-7,-6,-8,-6,-9,8,0,-7,-5,3,-4,10,-10,-10,-5,-10,9,3,4,-6,-4,-8,-9,9,-4,9,-8,10,1,-3,-9,10,-6,-6,-9,-3,-7,-5,-8,10,10,8,-6,-5,-1,-8,-3,-8,-8,-7,-8,-1,-10,7,9,-9,4,9,9,-3,5,9,-9,-1,0,-7,-5,0,-1,-9,-2,-3,-5,-1,-9,-7,-2,10,-10,-5,9,10,-9,-6,-1,-5,-7,-10,10,-9,-8,-6,-5,-6,-1,-1,10,-4,-4,-10,-7,0,0,-8,-3,9,-6,-8,-5,0,-3,0,-1,-9,-4,-5,-10,-8,8,-7,10,9,-4,4,-9,-3,-6,-6,0,6,-6,-6,3,-1,9,-2,0,8,-3,6,-7,0,-6,-6,-6,-7,-7,-6,9,8,-2,-9,-7,0,-3,-9,8,1,-6,-2,-2,-3,-2,-9,1,-7,-10,8,-7,-3,-4,-6,0,0,-3,10,-7,0,-7,0,8,-7,-8,-1,0,-9,9,9,1,3,-7,9,0,0,-1,-5,-1,8,-3,-5,6,-1,-1,-9,9,9,0,-4,-4,10,-9,-2,-8,-4,-3,-8,-1,-6,-7,-3,-5,7,-7,8,1,-1,-1,-2,-7,0,-1,-5,-8,-7,-10,-9,-9,-6,-7,-8,-7,-8,-7,-2,-7,-7,0,-4,-4,-7,5,10,8,8,5,10,-4],"spec":{"size":{"v":[5,4,4,-1,4,-3,3,1,0,-1,2,2,4,2,-1,1,1,3,2,-3,-3,-3,1,4,2,5,-1,2,2,3,4,0,3,2,0,1,0,-5,-1,-1,-3,-4,4,0,-1,5,-4,-3,-5,-3,-4,-3,-5,5,1,-4,5,2,-3,-4,-3,-4,2,-3,1,1,-4,-3,-4,-5,5,-3,4,0,-4,1,-1,4,4,-3,-4,-3,4,2,-4,4,1,4,-3,1,0,1,0,-3,1,4,-5,-3,-5,1,1,-3,-4,-4,-3,1,-5,-4,4,-3,-3,-2,0,-4,-4,-3,-3,1,2,3,-4,4,2,0,-3,-3,1,4,0,2,-2,1,1,0,2,-3,-4,-4,2,-2,-3,1,-4,-3,-4,-4,-4,2,0,-2,-3,-4,-2,-4,-3,-4,2,2,-4,-4,-2,2,2,-4,2,-1,-3,3,4,0,-4,1,2,-3,-4,-3,-2,0,-3,-1,2,2,1,-2,0,-2,3,-5,-4,-5,-1,-2,-2,-2,0,-5,1,2,4,-1,-1,3,5,1,-1,1,-4,2,-3,1,4,-5,4,3,1,-2,-4,-3,4,4,-3,-3,-3,1,-2,1,-4,0,-5,-2,1,-4,-3,4,-3,-3,4,4,0,3,-3,-5,-3,-1,2,-3,-3,-3,-3,-4,5,-3,2,2,2,1,5,2,1,0,3,1,4,0,3,5,-3,-2,-5,-4,0,-4,4,-4,3,-2,-4,2,-2,-3,4,-3,2,0,-2,-5,-2,-3,-2,-4,2,0,-3,-5,-5,-3,-3,-1,0,-3,4,-5,5,-3,-3,-4,-3,-4,-2,-1,-4,0,-2,-4,-4,-5,-4,3,-3,-4,-4,5,-3,2,-5,-5,-1,-4,1,1,1,-3,-4,-4,-5,2,-3,4,-4,4,1,-4,-4,5,-3,-3,-5,0,-4,-3,-5,2,4,3,-3,-4,-2,-2,-2,-3,-3,-2,-3,1,-3,2,4,-3,2,2,1,0,3,4,-3,0,-4,-3,-3,-3,-4,-3,-4,-1,-3,-1,-5,-2,-3,5,-4,-3,3,5,-4,-2,-3,-3,-4,-4,3,-4,-3,-5,-3,-3,1,-2,3,-5,-4,-4,-4,0,2,-5,-3,3,-4,-3,-4,1,-4,-2,-2,-3,-3,-3,-3,-1,0,-4,3,4,-1,0,-3,-1,-2,-3,-3,3,-2,-3,2,-4,3,0,0,2,-4,5,-5,-1,-3,-3,-4,-4,-4,-4,3,4,0,-5,-3,0,-3,-4,5,-4,-3,-2,0,-3,-4,-3,1,-2,-4,1,-4,0,-2,-3,0,-1,0,4,-3,0,-3,3,4,-4,-3,0,-1,-3,4,4,2,-2,-5,5,-3,-4,-4,-4,1,0,-4,-1,1,1,1,-4,2,2,-1,-2,-4,4,-4,-2,-4,-4,-3,-4,0,-5,-4,-5,-1,4,-4,4,-1,-5,-3,-3,-4,0,0,-3,-2,-2,-4,-4,-4,-3,-4,-4,-2,-4,-5,-3,-4,-3,-1,-4,-2,-4,0,5,5,4,1,4,-3],"nf":[9,10,8,7,8],"rules":10,"T":4,"feats":9},"shape":{"v":[17,1,18,17,15,15,16,5,20,16,-14,15,14,0,14,14,1,20,20,-3,-5,-10,17,14,18,18,20,13,20,8,18,16,18,13,12,18,15,-14,-11,18,-13,15,17,9,1,17,-16,18,-13,-4,-17,-14,-16,5,-2,-15,20,16,-14,-15,-15,-19,17,-12,15,17,-16,-16,1,-13,8,-1,17,-1,-19,5,-6,17,17,-19,-17,15,17,8,-15,15,14,17,-13,9,-18,5,-15,-20,17,19,-12,-14,-12,15,-1,-16,-16,-12,-13,18,-3,-10,17,-4,-18,0,12,-10,-14,-19,-2,20,17,0,-12,17,16,3,-1,-16,1,0,8,16,-11,10,9,0,9,-13,-10,-14,16,-7,-16,1,-16,-6,-17,-7,16,-3,2,-16,-16,-2,13,-16,-3,-15,14,-13,-16,-15,-11,10,19,-14,19,-16,-14,1,15,-13,-4,-13,19,-3,-16,-16,-1,17,-14,-16,19,18,1,-8,-2,-16,2,-14,-17,-18,13,-7,-16,2,14,-18,14,-1,13,15,-14,19,17,19,-12,-2,-16,-4,-1,-7,17,-18,8,0,15,16,-2,-12,20,16,-18,-9,-15,13,-7,-13,-15,0,-7,15,19,-17,-16,11,-15,-18,18,14,-6,20,-15,-16,0,-19,17,-13,-14,1,-16,-13,17,-16,16,18,20,16,19,17,15,20,9,-16,6,-13,16,9,-10,-17,-14,-2,-16,-15,17,-16,-13,-3,-16,0,-17,-15,19,-3,19,16,-1,-16,-3,-14,0,-18,2,-2,-13,-17,-17,-15,-15,-14,-16,-15,20,-6,17,-15,-18,-16,-16,-16,-14,-17,-16,-17,-16,-17,-16,-17,-14,15,0,-15,-3,4,-13,19,-16,-16,-14,-14,17,0,17,-8,-15,-17,-17,19,-15,16,-18,20,-2,0,-16,19,-16,-10,-15,-4,-15,-13,-15,15,17,11,-6,-1,5,-14,-12,-16,-16,-16,-17,-17,-16,6,15,-17,7,19,17,-20,20,19,-16,0,12,-18,-8,17,0,-19,-3,-2,-10,1,-5,-15,-2,16,-16,-14,18,18,-19,-14,16,-14,-15,-18,17,-15,-10,-18,-16,-17,-12,-4,20,-17,-14,-16,-17,-2,-16,-18,-15,18,-16,-14,-15,0,-8,16,-18,-14,-4,-15,-14,-16,17,-6,20,15,-19,16,-16,-17,-15,-17,1,12,-14,-17,8,-2,18,-14,-3,20,-1,13,-18,-6,-11,-16,-14,-16,-16,-17,6,15,-16,-16,-16,-1,0,-18,17,14,-14,-18,-3,-17,0,-5,-2,-18,-19,17,-18,-15,-1,-13,16,13,-14,20,-17,-6,-17,-16,5,-15,-17,-4,2,-17,19,19,-2,14,-16,19,2,3,-5,-13,-3,17,-4,-16,17,-16,-11,-18,18,17,-2,-14,-15,20,-17,-2,-17,-16,-2,-9,-2,-16,-8,-13,-14,18,-13,19,15,-2,-15,0,-6,-1,-12,-17,-20,-20,-16,-17,-17,-16,-17,-14,-17,-14,-10,-5,-18,-16,-1,-3,-6,-16,15,17,17,16,11,17,-15],"nf":[40,40,40,40,40],"rules":40,"T":20,"feats":9},"texture":{"v":[3,-2,7,4,-6,4,6,5,7,8,3,4,2,1,7,8,2,7,3,-5,-5,-6,-2,8,7,3,9,-4,8,-9,9,3,7,7,6,6,8,-7,-1,3,-1,6,1,7,7,0,-6,6,-1,-1,-4,-7,-6,-3,0,-4,3,6,-5,-1,-3,3,5,-4,7,6,0,-1,2,-7,-1,-2,8,-3,-5,1,-4,-1,5,-2,4,3,3,1,-2,4,4,2,0,-5,-2,-4,-5,-5,0,4,-6,-3,-2,-1,2,-2,2,-2,-3,2,2,-2,1,2,-5,0,-3,1,0,4,-7,3,9,0,0,-1,1,-5,-7,-7,4,-1,-5,5,-2,0,6,-5,4,1,-2,-9,-5,-6,-8,-5,-4,-4,-9,-2,2,-4,-5,-5,0,3,-2,-7,-2,-6,0,-1,-7,-6,3,-7,1,-1,5,-2,-4,-1,3,-3,-5,1,0,-5,-6,-9,-2,1,-2,-7,2,6,8,-10,5,-4,-1,-10,-2,-6,8,-2,-7,2,4,-5,5,-2,7,7,-2,-2,1,7,-2,3,-4,1,6,-6,0,-4,-4,-2,3,1,1,-3,4,4,-8,-1,-5,7,-4,-4,-7,-5,0,9,2,4,-1,1,-3,-3,6,-1,-3,3,-4,-8,-1,-1,1,0,-2,-5,5,-3,2,-8,2,-1,3,-2,4,2,7,9,6,2,1,0,6,6,-9,-6,-2,-2,-5,-6,2,-4,1,-5,-7,0,-2,-3,9,-7,3,-1,-9,-9,-2,-5,-1,-2,-8,-3,-2,-6,-9,-6,-9,-8,-4,-3,1,-8,3,-4,-9,0,-7,-7,-5,-5,-2,-5,-8,-3,-4,-5,-5,4,-2,-5,0,-4,-1,7,-7,-8,-5,-8,4,1,-3,-4,3,-7,-5,2,-2,2,-4,6,-7,-4,-1,4,-4,-5,-4,-8,-3,-5,-7,4,5,0,-6,-9,2,-6,-8,-8,-3,-4,-5,-2,-6,0,3,-4,-5,3,5,-7,-3,0,-1,-3,2,0,1,7,0,-9,-3,-2,-4,1,-9,-5,-4,0,-6,-5,1,7,1,-8,1,-9,-7,-4,9,-2,-5,-3,-10,-5,-6,-3,8,1,3,-6,-2,1,3,1,4,4,-5,0,-3,-4,-4,-3,-3,-3,-3,-1,-9,-4,4,-5,2,3,-8,7,-4,-10,-3,-5,0,5,-7,-7,-5,3,5,-3,-4,2,-2,3,-2,-5,-3,-1,-1,2,-1,-1,7,4,-1,-4,-2,-2,-3,-3,1,-1,-4,-2,-4,-2,-3,-6,-3,-4,-5,0,-3,-3,-3,-5,-1,-3,-3,1,-3,1,0,-4,1,-7,-5,-4,0,-3,-4,5,-4,5,-2,-4,-1,-2,4,0,-6,5,-3,-4,7,-5,-3,-3,2,0,-4,-2,2,5,-5,0,-6,0,6,-6,-4,-3,-4,9,-6,2,-2,1,6,5,1,0,-6,5,2,-2,-3,0,-5,-6,-6,-2,-4,1,1,-2,-2,0,-6,-2,-2,2,1,-1,9,3,2,0,0,6,-2],"nf":[20,19,18,18,19],"rules":20,"T":10,"feats":12}},"acc":{"size":0.8805,"shape":0.9069,"texture":0.826,"all20":0.9227,"all70":0.9613}};
  // Ljubljana breast-cancer recurrence data from the book (y = 1: recurrence), one TM with 20 rules.
  const RECUR = {"y":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],"v":[-1,-2,-5,-7,-6,-4,-7,-5,-4,-7,-2,-9,-1,-2,-2,-3,-6,-2,-4,-4,-3,-4,-3,-4,-5,-7,-6,-2,-3,-7,-4,-7,-2,-9,-9,-3,-4,-5,-8,-6,-6,-8,-3,-7,-2,-3,-7,-4,-2,-3,-4,-2,-6,-1,-1,-3,-5,-2,-6,-3,-6,-7,-7,2,-6,-7,-5,-4,-1,-4,-9,-5,-5,-6,-3,-10,-10,-2,-7,-4,-9,-4,-7,-8,-4,-2,-5,-4,-4,-6,-2,-3,-6,-4,-9,-6,-8,-5,-3,-2,-4,-5,-8,-3,-10,-5,-5,-2,-2,-6,-6,-5,-3,-9,-3,-7,-3,-4,-6,-7,-5,-4,-1,-7,-8,-4,1,0,-1,-1,-1,0,-2,0,-3,1,0,-2,-2,-8,3,0,-9,0,3,-4,-2,0,-2,-2,1,4,-2,-1,-2,0,1,-6,-9,-3,-1,-1,-3,2,2,-1,-1,-1,-9,-2,3,-2,0,-2,0,-1,-5,-7,-2,-1,3,0,-1,3,1,1,-6,-2,-7,-2,0,-3,0,-4,-2,-7,-5,-5,-1,-6,-3,-5,-5,-6,-5,-5,-6,-6,-5,-5,-5,-6,-2,-8,-7,-7,-7,-7,-4,-4,0,-3,-2,-4,2,1,2,-1,4,-1,-2,-5,0,0,0,1,-7,-7,-2,3,-3,0,0,4,0,2,3,0,2,-1,-4,-1,-4,-3,0,2,-3,-1,1,-2,-1,3,-5,-2,-2,2,0,2,-1,-3,3,-3,-3,1,1,1,2,3,0,3,4,-5,-1,-5,-2,-1]};

  const DATASETS = {
    wdbc: { label: "Diagnosis (Wisconsin)", v: WDBC.all20, y: WDBC.y, pos: "malignant", neg: "benign", T: 10 },
    recur: { label: "Recurrence (book's data)", v: RECUR.v, y: RECUR.y, pos: "recurrence", neg: "no recurrence", T: 10 },
  };

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
  function plots(parent) {
    const div = document.createElement("div");
    div.className = "viz-plots";
    parent.appendChild(div);
    return div;
  }
  const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
  const signed = (x, d = 0) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
  const narrowOf = (root) => root.clientWidth < 480;
  function datasetButtons(ctrl, current) {
    for (const b of ctrl.querySelectorAll("button[data-ds]")) {
      b.style.borderColor = b.dataset.ds === current ? "var(--viz-model)" : "";
      b.style.fontWeight = b.dataset.ds === current ? "700" : "";
    }
  }

  // Accuracy when only cases with |v| >= th are answered (v >= 0 means the positive class)
  function selective(ds, th) {
    let kept = 0, correct = 0, pos = 0, refErr = 0, allErr = 0;
    for (let i = 0; i < ds.v.length; i++) {
      const right = (ds.v[i] >= 0 ? 1 : 0) === ds.y[i];
      if (!right) allErr++;
      if (Math.abs(ds.v[i]) >= th) { kept++; if (right) correct++; pos += ds.y[i]; }
      else if (!right) refErr++;
    }
    return { kept, correct, cover: kept / ds.v.length, acc: kept ? correct / kept : NaN,
             base: kept ? Math.max(pos, kept - pos) / kept : NaN, refErr, allErr, n: ds.v.length };
  }

  // ---------------------------------------------------------------- Widget 1: six students
  function studentsWidget(root) {
    // Granmo, Ch. 7, Table 7.1: Lectures, Textbook, Assignments, Pass
    const S = [[1, 1, 1, 1], [0, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]];
    const rows = S.map(([l, t, a, p], i) => {
      const r = [l, t, 1 - a], v = l + t - (1 - a);
      return { id: i + 1, l, t, a, p, r, v, right: (v >= 0 ? 1 : 0) === p };
    });
    const wrap = document.createElement("div");
    wrap.style.overflowX = "auto";
    root.appendChild(wrap);
    const table = document.createElement("table");
    table.className = "tmc-table" + (narrowOf(root) ? " tmc-narrow" : "");
    wrap.appendChild(table);
    const narrow = narrowOf(root);
    const W = narrow ? 340 : 560, H = 120, M = { l: 20, r: 20 };
    const chart = svg(W, H);
    chart.style.maxWidth = "620px";
    chart.style.margin = "10px auto 0";
    root.appendChild(chart);
    const ctrl = control(root, [0, 1, 2].map((t) => `<button data-th="${t}">Confidence ≥ ${t}</button>`).join(""));
    const out = readout(root);
    hint(root, "R1: if Lectures then Pass · R2: if Textbook then Pass · R3: if NOT Assignments then Fail. ● = the rule's condition is true, so it votes.");
    let th = 0;

    function draw() {
      const c = colors();
      const yes = (b) => (narrow ? (b ? "Y" : "N") : b ? "yes" : "no");
      const hd = narrow ? ["#", "Lect.", "Book", "Asg.", "Pass"] : ["#", "Lectures", "Textbook", "Assign.", "Pass"];
      const dot = (on, col) => (on ? `<span style="color:${col};font-weight:700">●</span>` : `<span style="color:var(--viz-muted)">·</span>`);
      table.innerHTML = `<thead><tr>${hd.map((h) => `<th>${h}</th>`).join("")}<th>R1</th><th>R2</th><th>R3</th><th>v</th><th>|v|</th></tr></thead><tbody>` +
        rows.map((s) => `<tr style="opacity:${Math.abs(s.v) >= th ? 1 : 0.35}"><td>${s.id}</td><td>${yes(s.l)}</td><td>${yes(s.t)}</td><td>${yes(s.a)}</td><td>${yes(s.p)}</td>` +
          `<td>${dot(s.r[0], c.data)}</td><td>${dot(s.r[1], c.data)}</td><td>${dot(s.r[2], c.grad)}</td>` +
          `<td style="color:${c.loss};font-weight:700">${signed(s.v)}</td><td>${Math.abs(s.v)}</td></tr>`).join("") + "</tbody>";

      chart.innerHTML = "";
      const x = linear(-2.5, 2.5, M.l, W - M.r), yAxis = 70;
      if (th > 0) {
        el("rect", { x: x(-th + 0.5), y: 8, width: x(th - 0.5) - x(-th + 0.5), height: yAxis - 4, fill: c.muted, "fill-opacity": 0.13, rx: 4 }, chart);
        el("text", { x: x(0), y: 20, "text-anchor": "middle" }, chart).textContent = "refer";
      }
      el("line", { x1: x(-2.5), x2: x(2.5), y1: yAxis, y2: yAxis, stroke: c.border, "stroke-width": 1.5 }, chart);
      for (let t = -2; t <= 2; t++) {
        el("line", { x1: x(t), x2: x(t), y1: yAxis - 4, y2: yAxis + 4, stroke: c.muted }, chart);
        el("text", { x: x(t), y: yAxis + 18, "text-anchor": "middle", style: textStyle }, chart).textContent = signed(t);
      }
      el("text", { x: x(-2.5), y: yAxis + 36 }, chart).textContent = "← confident Fail";
      el("text", { x: x(2.5), y: yAxis + 36, "text-anchor": "end" }, chart).textContent = "confident Pass →";
      el("text", { x: x(0), y: yAxis + 36, "text-anchor": "middle" }, chart).textContent = "vote sum v";
      const stack = {};
      for (const s of rows) {
        const k = (stack[s.v] = (stack[s.v] || 0) + 1);
        const cy = yAxis - 4 - 13 * k, on = Math.abs(s.v) >= th;
        el("circle", { cx: x(s.v), cy, r: 6, fill: s.right ? c.opt : c.grad, "fill-opacity": on ? 1 : 0.3 }, chart);
        el("text", { x: x(s.v) + 10, y: cy + 4, style: on ? textStyle : "" }, chart).textContent = "#" + s.id + (s.right ? "" : " ✗");
      }
      const kept = rows.filter((s) => Math.abs(s.v) >= th), ok = kept.filter((s) => s.right).length;
      const ref = rows.filter((s) => Math.abs(s.v) < th).map((s) => "#" + s.id);
      out.textContent = `confidence ≥ ${th} · kept ${kept.length} of 6 students · correct ${ok} of ${kept.length} · accuracy ${Math.round((100 * ok) / kept.length)}%` +
        (ref.length ? ` · not answered: ${ref.join(", ")}` : "");
      for (const b of ctrl.querySelectorAll("button")) b.style.borderColor = +b.dataset.th === th ? "var(--viz-model)" : "";
    }
    ctrl.addEventListener("click", (e) => { if (e.target.dataset.th !== undefined) { th = +e.target.dataset.th; draw(); } });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 2: confidence explorer
  function confidenceWidget(root) {
    const narrow = narrowOf(root);
    const box = plots(root);
    const W1 = 340, H1 = 250, M1 = { l: 40, r: 10, t: 22, b: 40 };
    const W2 = 320, H2 = 250, M2 = { l: 44, r: 14, t: 22, b: 40 };
    const hist = svg(W1, H1), curve = svg(W2, H2);
    box.appendChild(hist); box.appendChild(curve);
    const ctrl = control(root,
      `<button data-ds="wdbc">${DATASETS.wdbc.label}</button><button data-ds="recur">${DATASETS.recur.label}</button>` +
      `<label>answer only if |v| ≥ <input type="range" min="0" max="11" step="1" value="2"> <b class="th">2</b></label>`);
    const slider = ctrl.querySelector("input"), thLabel = ctrl.querySelector(".th");
    const out = readout(root);
    hint(root, "Left: every patient's vote sum, from a Tsetlin machine (20 rules, Voting Margin T = 10) that never saw that patient. The grey band is the 'refer to a human' zone. Right: accuracy on the patients the machine still answers, against the share it answers; dashed = always guessing the more common class within that same group.");
    let dsKey = "wdbc";

    function draw() {
      const c = colors(), ds = DATASETS[dsKey], th = +slider.value;
      thLabel.textContent = th;
      datasetButtons(ctrl, dsKey);
      // histogram of vote sums, stacked by true class
      hist.innerHTML = "";
      const counts = {};
      for (let i = 0; i < ds.v.length; i++) {
        const k = ds.v[i]; counts[k] = counts[k] || [0, 0]; counts[k][ds.y[i]]++;
      }
      const maxC = Math.max(...Object.values(counts).map((a) => a[0] + a[1]));
      const yMax = Math.ceil(maxC / 10) * 10;
      const x = linear(-10.6, 10.6, M1.l, W1 - M1.r), y = linear(0, yMax, H1 - M1.b, M1.t);
      axes(hist, x, y, { xDomain: [-10.6, 10.6], yDomain: [0, yMax], xTicks: [], yTicks: ticks(0, yMax, yMax > 50 ? 20 : 10), xLabel: "vote sum v" });
      for (const t of ticks(-10, 10, narrow ? 5 : 2)) el("text", { x: x(t), y: H1 - M1.b + 14, "text-anchor": "middle" }, hist).textContent = signed(t);
      if (th > 0) {
        el("rect", { x: x(-th + 0.5), y: M1.t, width: x(th - 0.5) - x(-th + 0.5), height: H1 - M1.b - M1.t, fill: c.muted, "fill-opacity": 0.15 }, hist);
        el("text", { x: x(0), y: M1.t + 12, "text-anchor": "middle", style: textStyle }, hist).textContent = "refer";
      }
      const bw = (x(1) - x(0)) * 0.8;
      for (const k in counts) {
        const [n0, n1] = counts[k], on = Math.abs(+k) >= th, op = on ? 0.9 : 0.35;
        el("rect", { x: x(+k) - bw / 2, y: y(n1), width: bw, height: y(0) - y(n1), fill: c.grad, "fill-opacity": op }, hist);
        el("rect", { x: x(+k) - bw / 2, y: y(n0 + n1), width: bw, height: y(n1) - y(n0 + n1), fill: c.data, "fill-opacity": op }, hist);
      }
      el("rect", { x: M1.l + 6, y: 6, width: 10, height: 10, fill: c.grad }, hist);
      el("text", { x: M1.l + 20, y: 15 }, hist).textContent = ds.pos;
      el("rect", { x: M1.l + 100, y: 6, width: 10, height: 10, fill: c.data }, hist);
      el("text", { x: M1.l + 114, y: 15 }, hist).textContent = ds.neg;

      // coverage / accuracy curve
      curve.innerHTML = "";
      const cx = linear(0, 1, M2.l, W2 - M2.r), cy = linear(0.5, 1, H2 - M2.b, M2.t);
      axes(curve, cx, cy, { xDomain: [0, 1], yDomain: [0.5, 1], xTicks: [], yTicks: [], xLabel: "share of patients answered" });
      for (const t of [0, 0.25, 0.5, 0.75, 1]) el("text", { x: cx(t), y: H2 - M2.b + 14, "text-anchor": "middle" }, curve).textContent = Math.round(t * 100) + "%";
      for (const t of [0.5, 0.6, 0.7, 0.8, 0.9, 1]) el("text", { x: M2.l - 6, y: cy(t) + 4, "text-anchor": "end" }, curve).textContent = Math.round(t * 100) + "%";
      el("text", { x: M2.l, y: 14 }, curve).textContent = "accuracy on the answered patients";
      const pts = [];
      for (let t = 0; t <= 11; t++) { const s = selective(ds, t); if (s.kept) pts.push([t, s]); }
      const line = (key, attrs) => el("polyline", Object.assign({ points: pts.map(([, s]) => `${cx(s.cover)},${cy(Math.max(0.5, s[key]))}`).join(" "), fill: "none" }, attrs), curve);
      line("base", { stroke: c.muted, "stroke-width": 1.5, "stroke-dasharray": "5 4" });
      line("acc", { stroke: c.model, "stroke-width": 2.5 });
      for (const [t, s] of pts) el("circle", { cx: cx(s.cover), cy: cy(Math.max(0.5, s.acc)), r: t === th ? 7 : 3, fill: c.model, stroke: t === th ? c.bg : "none", "stroke-width": 2 }, curve);
      const cur = selective(ds, th);
      if (cur.kept) el("circle", { cx: cx(cur.cover), cy: cy(Math.max(0.5, cur.base)), r: 4, fill: c.muted }, curve);

      if (!cur.kept) { out.textContent = `|v| ≥ ${th}: no patient reaches this confidence, so the machine answers nobody.`; return; }
      out.textContent = `|v| ≥ ${th} · answers ${cur.kept} of ${cur.n} patients (${pct(cur.cover)}) · accuracy on them ${pct(cur.acc)} (${cur.kept - cur.correct} error${cur.kept - cur.correct === 1 ? "" : "s"})` +
        `\nrefers ${cur.n - cur.kept} patients to a human · they contain ${cur.refErr} of the ${cur.allErr} errors` +
        `\nalways guessing the more common class in the answered group: ${pct(cur.base)}`;
    }
    ctrl.addEventListener("click", (e) => { if (e.target.dataset.ds) { dsKey = e.target.dataset.ds; draw(); } });
    slider.addEventListener("input", draw);
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 3: reliability per vote sum
  function reliabilityWidget(root) {
    const narrow = narrowOf(root);
    const W = narrow ? 340 : 600, H = 260, M = { l: 44, r: 12, t: 26, b: 42 };
    const chart = svg(W, H);
    chart.style.maxWidth = "680px";
    chart.style.margin = "0 auto";
    root.appendChild(chart);
    const ctrl = control(root, `<button data-ds="wdbc">${DATASETS.wdbc.label}</button><button data-ds="recur">${DATASETS.recur.label}</button>`);
    const out = readout(root);
    hint(root, "Each bar: among the patients who got this vote sum, the share that truly had the positive class. Whiskers: 95% interval (Wilson), wide when few patients share a vote sum. Hover or tap a bar for its counts.");
    let dsKey = "wdbc", focus = null;

    function wilson(k, n) {
      const z = 1.96, p = k / n, d = 1 + (z * z) / n;
      const mid = (p + (z * z) / (2 * n)) / d, half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
      return [Math.max(0, mid - half), Math.min(1, mid + half)];
    }
    function summary(ds) {
      let pos = 0; for (const t of ds.y) pos += t;
      return `${ds.label}: ${ds.y.length} patients, ${pct(pos / ds.y.length)} ${ds.pos} overall (dashed line).`;
    }

    function draw() {
      const c = colors(), ds = DATASETS[dsKey];
      datasetButtons(ctrl, dsKey);
      chart.innerHTML = "";
      const stats = {};
      for (let i = 0; i < ds.v.length; i++) { const k = ds.v[i]; stats[k] = stats[k] || [0, 0]; stats[k][0]++; stats[k][1] += ds.y[i]; }
      let pos = 0; for (const t of ds.y) pos += t;
      const x = linear(-10.6, 10.6, M.l, W - M.r), y = linear(0, 1, H - M.b, M.t);
      axes(chart, x, y, { xDomain: [-10.6, 10.6], yDomain: [0, 1], xTicks: [], yTicks: [], xLabel: "vote sum v" });
      for (const t of [0, 0.25, 0.5, 0.75, 1]) el("text", { x: M.l - 6, y: y(t) + 4, "text-anchor": "end" }, chart).textContent = Math.round(t * 100) + "%";
      for (const t of ticks(-10, 10, narrow ? 5 : 2)) el("text", { x: x(t), y: H - M.b + 14, "text-anchor": "middle" }, chart).textContent = signed(t);
      el("text", { x: M.l, y: 16 }, chart).textContent = `share truly ${ds.pos}`;
      el("line", { x1: x(-10.6), x2: x(10.6), y1: y(pos / ds.y.length), y2: y(pos / ds.y.length), stroke: c.muted, "stroke-dasharray": "5 4" }, chart);
      const bw = (x(1) - x(0)) * 0.7;
      for (const k in stats) {
        const [n, kpos] = stats[k], p = kpos / n, [lo, hi] = wilson(kpos, n);
        const g = el("g", { style: "cursor: pointer" }, chart);
        el("rect", { x: x(+k) - (x(1) - x(0)) / 2, y: M.t, width: x(1) - x(0), height: H - M.b - M.t, fill: "transparent" }, g);
        el("rect", { "data-bar": k, x: x(+k) - bw / 2, y: y(p), width: bw, height: Math.max(1, y(0) - y(p)), fill: c.loss, "fill-opacity": focus === null || +k === focus ? 0.85 : 0.55 }, g);
        el("line", { x1: x(+k), x2: x(+k), y1: y(lo), y2: y(hi), stroke: c.text, "stroke-width": 1.2 }, g);
        for (const b of [lo, hi]) el("line", { x1: x(+k) - 3, x2: x(+k) + 3, y1: y(b), y2: y(b), stroke: c.text, "stroke-width": 1.2 }, g);
        const show = () => {
          focus = +k;
          for (const r of chart.querySelectorAll("rect[data-bar]")) r.setAttribute("fill-opacity", +r.dataset.bar === focus ? 1 : 0.55);
          out.textContent = `v = ${signed(+k)}: ${n} patients, ${kpos} truly ${ds.pos} (${pct(p, 0)}) · 95% interval ${pct(lo, 0)} to ${pct(hi, 0)}`;
        };
        g.addEventListener("mouseenter", show);
        g.addEventListener("click", show);
      }
      out.textContent = summary(ds);
    }
    ctrl.addEventListener("click", (e) => { if (e.target.dataset.ds) { dsKey = e.target.dataset.ds; focus = null; draw(); } });
    onThemeChange(draw);
    draw();
  }

  // ---------------------------------------------------------------- Widget 4: composite builder
  function compositeWidget(root) {
    const names = ["size", "shape", "texture"];
    const LABEL = { size: "Size TM", shape: "Shape TM", texture: "Texture TM" };
    const S = WDBC.spec, y = WDBC.y, n = y.length;
    const narrow = narrowOf(root);
    const box = plots(root);
    const W1 = 330, H1 = 230, M1 = { l: 92, r: 14, t: 26, b: 34 };
    const W2 = 330, H2 = 230, M2 = { l: 78, r: 14, t: 26, b: 34 };
    const accChart = svg(W1, H1), patChart = svg(W2, H2);
    box.appendChild(accChart); box.appendChild(patChart);
    const ctrl = control(root,
      names.map((k) => `<label><input type="checkbox" data-k="${k}" checked> ${LABEL[k]} (${S[k].rules} rules, T = ${S[k].T})</label>`).join("") +
      `<label><input type="checkbox" data-k="norm" checked> normalise (Eq. 7.3)</label>` +
      `<button data-a="prev">◀ patient</button><button data-a="next">patient ▶</button>`);
    const out = readout(root);
    hint(root, "Each specialist sees only one group of measurements. The patient browser steps through the 183 patients on whom the three specialists disagree. Untick specialists or normalisation to see what the composite loses.");
    const on = { size: true, shape: true, texture: true }; let norm = true;

    const value = (k, i) => (norm ? S[k].v[i] / S[k].nf[WDBC.fold[i]] : S[k].v[i]);
    function compositeAt(i, keys) { let s = 0; for (const k of keys) s += value(k, i); return s; }
    function accuracy(keys) {
      let ok = 0;
      for (let i = 0; i < n; i++) if ((compositeAt(i, keys) >= 0 ? 1 : 0) === y[i]) ok++;
      return ok / n;
    }
    const disagree = [];
    for (let i = 0; i < n; i++) { const p = names.map((k) => (S[k].v[i] >= 0 ? 1 : 0)); if (Math.min(...p) !== Math.max(...p)) disagree.push(i); }
    // start on a patient where normalising changes the answer from wrong to right
    let pos = Math.max(0, disagree.findIndex((i) => {
      let a = 0, b = 0; for (const k of names) { a += S[k].v[i] / S[k].nf[WDBC.fold[i]]; b += S[k].v[i]; }
      return (a >= 0 ? 1 : 0) === y[i] && (b >= 0 ? 1 : 0) !== y[i];
    }));

    function draw() {
      const c = colors(), keys = names.filter((k) => on[k]);
      // accuracy bars
      accChart.innerHTML = "";
      const x = linear(0.75, 1, M1.l, W1 - M1.r);
      const rowsA = names.map((k) => ({ label: LABEL[k], acc: WDBC.acc[k], active: on[k], col: c.model }));
      rowsA.push({ label: "Composite", acc: keys.length ? accuracy(keys) : NaN, active: true, col: c.opt });
      const rowH = (H1 - M1.t - M1.b) / rowsA.length;
      for (const t of [0.75, 0.8, 0.85, 0.9, 0.95, 1]) {
        el("line", { x1: x(t), x2: x(t), y1: M1.t, y2: H1 - M1.b, stroke: c.grid }, accChart);
        if (!narrow || t * 100 % 10 === 0) el("text", { x: x(t), y: H1 - M1.b + 14, "text-anchor": "middle" }, accChart).textContent = Math.round(t * 100) + "%";
      }
      el("text", { x: M1.l, y: 16 }, accChart).textContent = "accuracy on all 569 patients";
      rowsA.forEach((r, i) => {
        const yc = M1.t + rowH * (i + 0.5);
        el("text", { x: M1.l - 6, y: yc + 4, "text-anchor": "end", style: r.active ? textStyle : "" }, accChart).textContent = r.label;
        if (!Number.isFinite(r.acc)) return;
        el("rect", { x: x(0.75), y: yc - rowH * 0.3, width: Math.max(0, x(r.acc) - x(0.75)), height: rowH * 0.6, fill: r.col, "fill-opacity": r.active ? 0.85 : 0.25, rx: 3 }, accChart);
        el("text", { x: x(r.acc) - 4, y: yc + 4, "text-anchor": "end", style: "fill: var(--viz-bg); font-weight: 700" }, accChart).textContent = pct(r.acc);
      });
      const ref = WDBC.acc.all70;
      el("line", { x1: x(ref), x2: x(ref), y1: M1.t - 4, y2: H1 - M1.b, stroke: c.loss, "stroke-dasharray": "4 3", "stroke-width": 1.5 }, accChart);
      el("text", { x: x(ref) - 3, y: H1 - 6, "text-anchor": "end", style: "fill: var(--viz-loss)" }, accChart).textContent = "one 70-rule TM on all 30";

      // patient view
      patChart.innerHTML = "";
      const i = disagree[pos], lim = norm ? 1.6 : 36;
      const px = linear(-lim, lim, M2.l, W2 - M2.r);
      const rowsP = names.map((k) => ({ label: LABEL[k], val: value(k, i), active: on[k] }));
      const comp = compositeAt(i, keys);
      rowsP.push({ label: "Composite", val: comp, active: keys.length > 0, comp: true });
      const rh = (H2 - M2.t - M2.b) / rowsP.length;
      el("line", { x1: px(0), x2: px(0), y1: M2.t, y2: H2 - M2.b, stroke: c.muted }, patChart);
      el("text", { x: px(-lim), y: H2 - 8 }, patChart).textContent = "← benign";
      el("text", { x: px(lim), y: H2 - 8, "text-anchor": "end" }, patChart).textContent = "malignant →";
      for (const t of norm ? [-1.5, -1, -0.5, 0, 0.5, 1, 1.5] : [-30, -20, -10, 0, 10, 20, 30]) {
        if (narrow && (norm ? Math.abs(t) === 0.5 || Math.abs(t) === 1.5 : Math.abs(t) === 10 || Math.abs(t) === 30)) continue;
        el("text", { x: px(t), y: H2 - M2.b + 14, "text-anchor": "middle" }, patChart).textContent = signed(t, norm && t % 1 ? 1 : 0);
      }
      el("text", { x: M2.l, y: 16, style: textStyle }, patChart).textContent = `patient ${i + 1}: truly ${y[i] ? "malignant" : "benign"} (${pos + 1} of ${disagree.length})`;
      rowsP.forEach((r, j) => {
        const yc = M2.t + rh * (j + 0.5);
        el("text", { x: M2.l - 6, y: yc + 4, "text-anchor": "end", style: r.active ? textStyle : "" }, patChart).textContent = r.label;
        if (!r.active) return;
        const v = Math.max(-lim, Math.min(lim, r.val));
        const col = r.comp ? ((comp >= 0 ? 1 : 0) === y[i] ? c.opt : c.grad) : c.loss;
        el("rect", { x: Math.min(px(0), px(v)), y: yc - rh * 0.28, width: Math.max(1.5, Math.abs(px(v) - px(0))), height: rh * 0.56, fill: col, "fill-opacity": 0.85, rx: 2 }, patChart);
      });
      const parts = names.map((k) => {
        if (!on[k]) return null;
        const raw = S[k].v[i], f = S[k].nf[WDBC.fold[i]];
        return norm ? `${k} ${signed(raw)}/${f} = ${signed(raw / f, 2)}` : `${k} ${signed(raw)}`;
      }).filter(Boolean);
      const verdict = keys.length ? `composite ${signed(comp, norm ? 2 : 0)} → ${comp >= 0 ? "malignant" : "benign"} ${(comp >= 0 ? 1 : 0) === y[i] ? "✓" : "✗"}` : "pick at least one specialist";
      out.textContent = parts.join(" · ") + "\n" + verdict;
    }
    ctrl.addEventListener("change", (e) => {
      const k = e.target.dataset.k;
      if (k === "norm") norm = e.target.checked; else if (k) on[k] = e.target.checked;
      draw();
    });
    ctrl.addEventListener("click", (e) => {
      if (e.target.dataset.a === "prev") { pos = (pos - 1 + disagree.length) % disagree.length; draw(); }
      if (e.target.dataset.a === "next") { pos = (pos + 1) % disagree.length; draw(); }
    });
    onThemeChange(draw);
    draw();
  }

  const mounts = { "viz-students": studentsWidget, "viz-confidence": confidenceWidget, "viz-reliability": reliabilityWidget, "viz-composite": compositeWidget };
  for (const id in mounts) {
    const node = document.getElementById(id);
    if (node && !node.dataset.ready) { node.dataset.ready = "1"; mounts[id](node); }
  }
});
