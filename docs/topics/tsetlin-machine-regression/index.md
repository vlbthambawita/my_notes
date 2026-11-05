# Regression with Tsetlin Machines

!!! abstract "TL;DR"
    A **regression Tsetlin machine** predicts a *number* instead of a class. The change is small. All rules vote the same way (there are no rules "against"), the **vote sum** $v$ is simply the number of rules that fire, and it is scaled to the target's range: $\hat y = y_{\min} + \tfrac{v}{T}(y_{\max} - y_{\min})$. Here $T$ is the **resolution**: the number of votes that covers the whole range. Training compares $v$ with the target written in votes, $\tau$. The **sign** of the error picks the kind of feedback (too low: Recognize or Erase, so that more rules fire; too high: Reject, so that fewer fire). Its **size** sets the probability, $|v - \tau|/T$. With thermometer-coded inputs every rule becomes an interval such as *"2 ≤ dose < 5"*, and the prediction is a staircase built from these intervals. We train our own small version on an invented dose–response curve and on a simulated points score. For the score it recovers the published points table exactly, as nine readable rules.

**Level:** intermediate · **Prerequisites:** [Classification by voting](../tsetlin-machine-classification/index.md) (the vote sum $v$, the Voting Margin $T$ and how feedback is handed out) and [thermometer encoding](../tsetlin-machine-intro/index.md) from lesson 1. The math is fractions and absolute values.

!!! note "Beyond the book"
    The book's chapter on this topic (*An Introduction to Tsetlin Machines*, Ch. 3, "Regression") has not been published yet. This lesson is based on the research papers listed under [Further reading](#further-reading), mainly Abeyrathna et al. (2020) in *Phil. Trans. R. Soc. A*, its arXiv preprint and the integer-weighted follow-up. We use the book's vocabulary from the earlier lessons and give the papers' terms in brackets once. All numbers not attributed to a paper come from our own small numpy/JavaScript implementation, described in Step 4.

---

## Why should I care?

Many clinical questions want a number, not a yes/no:

- **How much?** A drug response at a given dose, a length of stay, a lab value a few hours ahead, a risk score.
- **Ranked outcomes.** A classifier treats "score 3" and "score 4" as unrelated classes. A regression model knows that 4 is closer to 3 than to 9, and is penalised less for being off by one.
- **Readable formulas.** Clinicians already use additive points scores, such as CHA₂DS₂-VASc for stroke risk in atrial fibrillation: one point per risk factor, add them up. A regression Tsetlin machine produces exactly this kind of model, a sum of AND-rules, and learns it from data.

The papers report that this simple machine is competitive with regression trees, random forests and support vector regression on several small real datasets (Step 5 quotes the numbers). Our job here is to understand *how* it works and what its rules look like.

---

## Step 1 — From a vote count to a number

In the [classification lesson](../tsetlin-machine-classification/index.md), rules voted **for** or **against** a class and the sign of the vote sum $v$ was the prediction. For a number, two things change (Abeyrathna et al., 2020, §4):

1. **No more "against".** Every rule votes the same way: when its condition is True it adds **one vote**. (In the papers: the clause *polarity* is removed.) So $v$ is just the number of rules that fire, between 0 and the number of rules.
2. **Scale the votes to the target's range.** Choose a whole number $T$, the **resolution** (the papers call it the *threshold* or *resolution* $T$). Then $T$ votes means "the largest value", 0 votes means "the smallest":

$$
\hat y \;=\; y_{\min} + \frac{v}{T}\,\bigl(y_{\max} - y_{\min}\bigr), \qquad v \text{ clipped to } [0, T],
$$

where $y_{\min}$ and $y_{\max}$ are the smallest and largest targets in the training data. The output can only take $T + 1$ different values, spaced $(y_{\max} - y_{\min})/T$ apart. A larger $T$ gives a finer output but needs more rules to fire.

The clearest example is a points score. In our simulated version of CHA₂DS₂-VASc, a patient gets 1 point each for heart failure, hypertension, diabetes, vascular disease, female sex and age 65–74, and 2 points for a prior stroke or TIA and for age 75 or older. The maximum is 9. Write each point as a rule ("IF prior stroke/TIA THEN +1", twice; "IF age ≥ 65 THEN +1" plus "IF age ≥ 75 THEN +1" for the two age points), set $T = 9$ and the range to 0–9, and each vote is worth exactly one point.

<div id="viz-score" class="viz"><div class="viz-title">🧮 Interactive: a points score as rules that add up</div></div>

!!! tip "What to notice"
    - **The default patient** (78 years old, hypertension, female) fires four rules: hypertension, female, age ≥ 65 and age ≥ 75. So $v = 4$ and the output is $0 + \tfrac{4}{9} \times 9 = 4$ points.
    - **A 2-point item is two rules with the same condition.** Duplicated rules act as a weight, exactly like the duplicated R1 and R2 in the classification lesson.
    - **Age is thermometer-coded.** A 78-year-old has "age ≥ 65" *and* "age ≥ 75" True, so the two age rules together give 2 points; a 70-year-old fires only the first.
    - Switch the rules to **"learned by a regression TM"**. These nine rules were *learned* from 600 simulated patients (Step 6 shows how). Click **next simulated patient** a few times: the learned rules give the same score as the table for every one of the 600 test patients.

---

## Step 2 — Numbers in, intervals out: the staircase

For a numeric input such as a dose, use **thermometer encoding** from [lesson 1](../tsetlin-machine-intro/index.md): one Boolean feature per threshold, "dose ≥ 0.5 mg", "dose ≥ 1.0 mg", …, "dose ≥ 9.5 mg" (19 features, 38 literals). A rule is an AND of literals, and with thermometer literals that AND is always an **interval**:

> dose ≥ 2.0 **AND** dose ≥ 1.5 **AND NOT** dose ≥ 5.0 **AND NOT** dose ≥ 6.5 → fires for **2.0 ≤ dose < 5.0**

The largest memorized "≥" threshold is the lower end, the smallest negated one is the upper end. (If the lower end is not below the upper end, the rule can never fire.) The regression output is therefore a sum of intervals, each adding one vote over its own range of doses: a **staircase**.

Our example is an **invented** dose–response curve: a benefit score that rises quickly with the dose, levels off around 3–5 mg and falls at high doses (as if side effects took over). We drew 300 training doses at random between 0 and 10 mg and added noise to the benefit. The widget shows the 40 rules that a regression TM learned from these points (the run of Step 5).

<div id="viz-stairs" class="viz"><div class="viz-title">🪜 Interactive: interval rules stack up to a staircase</div></div>

!!! tip "What to notice"
    - **Each rule is one bar.** At dose 3.65 mg, 17 of the 40 rules fire, so $v = 17$ and the output is $-0.66 + \tfrac{17}{20} \times 8.87 = 6.88$ (the true curve is at 7.31). The training targets ran from $-0.66$ to $8.21$, so each vote is worth $8.87/20 = 0.44$.
    - **Overlapping intervals make a smooth-ish shape.** The 40 rules form 27 different intervals, between 2 and 5.5 mg wide, that overlap like roof tiles. Where many overlap (around 3–5 mg) the output is high; at the edges few do. At dose 0.25 mg, 5 rules fire (output 1.56, true 1.71).
    - **The edges are hardest.** At 9.75 mg only 3 rules fire (output 0.67, true curve 1.84). Few training points lie there, and every interval has to end somewhere.
    - **No rule is "the" answer.** Unlike the points score, the curve has no true list of rules. The machine finds *a* set of intervals whose sum fits the data. The bars are the whole model.

The original paper booleanizes a little differently: it uses **every distinct value** of a feature in the training data as a threshold, with bits "$x \le v_i$" (Abeyrathna et al., 2020, §3). That keeps all the ordering information, at the cost of many more literals. Fixed, evenly spaced thresholds are a simpler choice that works well for a toy.

---

## Step 3 — Feedback: the error's sign picks the kind, its size the probability

Training has to answer the same two questions as in classification: **which kind** of feedback does each rule get, and **how often**?

First, write the target in votes. A target $y$ corresponds to

$$
\tau \;=\; \frac{y - y_{\min}}{y_{\max} - y_{\min}}\, T
$$

votes. During training, compute $v$ (with empty rules counting as True, as before, and clipped to $[0, T]$) and the error $v - \tau$.

**The sign picks the kind** (Abeyrathna et al., 2020, Eq. 4.2):

- **Too low** ($v < \tau$): more rules should fire. A picked rule that fired gets **Recognize Feedback** (it memorizes the True literals, so it becomes a better description of this example). A picked rule that stayed silent gets **Erase Feedback** (it forgets literals, so it becomes broader and starts to fire on more examples). Together these are the papers' *Type I* feedback.
- **Too high** ($v > \tau$): fewer rules should fire. A picked rule that fired gets **Reject Feedback** (*Type II*): it memorizes a Forgotten literal that is False for this example, so it stops firing on it. Silent rules are left alone.

**The size sets the probability.** Each rule is picked independently with probability

$$
p \;=\; \frac{|v - \tau|}{T}.
$$

A prediction that is off by a quarter of the range gives each rule a 25% chance of feedback; a perfect prediction gives none. Compare the classification rule, $p = (T - v)/2T$: there, examples already safely won by $T$ votes stopped asking for feedback. Here, examples already **predicted correctly** stop asking. In both cases the feedback goes where the machine is still wrong, which is the **resource allocation** from the classification lesson: rules that are not needed for well-predicted examples are free to fix the others.

<div id="viz-feedback" class="viz"><div class="viz-title">🎚️ Interactive: how the error hands out feedback</div></div>

!!! tip "What to notice"
    - **A V shape.** The probability is zero at the target and rises linearly on both sides. The green side (too low) leads to Recognize and Erase; the red side (too high) to Reject.
    - **With the defaults** ($T = 10$, target 6.5 votes, $v = 3$) the prediction is 3.5 votes too low, so each rule is picked with probability $3.5/10 = 0.35$.
    - **Raising $T$ lowers every probability** for the same error in votes. A finer resolution also makes learning more cautious.
    - **The dashed line** is the version in the `cair/tmu` library, which squares the probability, $(|v - \tau|/T)^2$: small errors get much less feedback, so a nearly right prediction is disturbed less often. The papers use the linear version (with an extra scaling constant $K$ that we set to 1).

!!! note "The versions in the literature"
    The first preprint (arXiv 1905.04206) scales the votes by $y_{\max}/T$ only and uses one rule per vote ($T$ = number of rules). The journal version adds $y_{\min}$ (our formula above) and the probability $K\,|y - \hat y|/(y_{\max} - y_{\min})$, which in votes is $K\,|v - \tau|/T$. The integer-weighted follow-up writes it as $|y - \hat y|/T$. The `TMRegressor` in `cair/tmu` rounds the target down to a whole number of votes, clips $v$ to $[0, T]$, keeps the number of rules separate from $T$, and squares the probability. We use the journal version with $K = 1$, a separate number of rules, clipping, and a target that is not rounded.

---

## Step 4 — The complete learning step

Put the pieces together. Every rule starts **empty**, with every literal at the top of the Forgotten half of its memory. Then, for each training example in turn:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 One learning step of a regression Tsetlin machine",
  "cols": 3,
  "rows": 2,
  "rowHeight": 120,
  "nodes": [
    {"id": "obs", "col": 0, "row": 0, "icon": "🧾", "label": "Observe an example", "sub": "literals and a target y", "role": "data", "note": "The example's thermometer-coded literals and its numeric target."},
    {"id": "eval", "col": 1, "row": 0, "icon": "🧩", "label": "Evaluate every rule", "sub": "empty rules count as True", "role": "model", "note": "A rule fires when all its memorized literals are True. While training, an empty rule counts as True."},
    {"id": "vote", "col": 2, "row": 0, "icon": "🗳️", "label": "Vote sum v, clipped to [0, T]", "sub": "every firing rule adds 1", "role": "loss", "note": "No rules vote against: v is the number of firing rules."},
    {"id": "err", "col": 2, "row": 1, "icon": "📏", "label": "Error v − τ", "sub": "τ = target in votes", "role": "loss", "note": "τ = (y − y_min) / (y_max − y_min) × T. Zero error: nothing happens."},
    {"id": "prob", "col": 1, "row": 1, "icon": "🎲", "label": "Pick each rule with probability |v−τ|/T", "role": "loss", "note": "An independent coin flip per rule. A bigger error means more rules learn from this example."},
    {"id": "fb", "col": 0, "row": 1, "icon": "🛠️", "label": "Too low: Recognize / Erase. Too high: Reject", "role": "grad", "note": "Too low: picked rules that fired get Recognize, picked silent rules get Erase. Too high: picked rules that fired get Reject."}
  ],
  "edges": [
    {"from": "obs", "to": "eval"},
    {"from": "eval", "to": "vote"},
    {"from": "vote", "to": "err"},
    {"from": "err", "to": "prob"},
    {"from": "prob", "to": "fb"},
    {"from": "fb", "to": "obs", "label": "next"}
  ],
  "steps": [
    {"show": ["obs"], "text": "Take the next training example: its literals (each threshold feature and its negation) and its numeric target y."},
    {"show": ["obs>eval", "eval"], "text": "Every rule checks its condition. While training, an empty rule counts as True, so fresh rules have something to learn from."},
    {"show": ["eval>vote", "vote"], "text": "Count the firing rules. That is the vote sum v; clip it to at most T. The prediction is y<sub>min</sub> + v / T × (y<sub>max</sub> − y<sub>min</sub>)."},
    {"show": ["vote>err", "err"], "text": "Write the target in votes, τ = (y − y<sub>min</sub>) / (y<sub>max</sub> − y<sub>min</sub>) × T, and compute the error v − τ. If it is exactly zero, nothing happens."},
    {"show": ["err>prob", "prob"], "text": "Each rule independently gets feedback with probability <b>|v − τ| / T</b>: the bigger the error, the more rules learn from this example."},
    {"show": ["prob>fb", "fb"], "text": "<b>Too low</b>: picked rules that fired get <b>Recognize Feedback</b>, picked silent rules get <b>Erase Feedback</b>, so more rules fire next time. <b>Too high</b>: picked rules that fired get <b>Reject Feedback</b>, so fewer fire."},
    {"show": ["fb>obs", "obs"], "text": "On to the next example. One pass through the training set is an <b>epoch</b>; repeat for many epochs."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 120, "pos": {"obs": [0, 0], "eval": [1, 0], "vote": [1, 1], "err": [1, 2], "prob": [0, 2], "fb": [0, 1]}}
}
</script>
</div>

Our implementation (in the widgets and in the code at the end) follows this diagram. Its details:

- **Memorize Value 1, Forget Value $1/s$**, the convention the book uses from Ch. 2 on and the default in `cair/tmu`. (The journal paper's feedback tables use $(s-1)/s$ for memorizing, like our classification lesson.)
- **Memory depth 100** (positions 1–100, Memorized above 50).
- **Empty rules** fire while training and stay silent when predicting, as in the earlier lessons and in `cair/tmu`.
- A seeded random number generator, so the widgets reproduce the numbers in this lesson exactly.

!!! tip "What to notice"
    - **Only one number is shared:** the error $v - \tau$. As in classification, no rule knows about any other; the vote sum coordinates them.
    - **The two directions balance each other.** Recognize and Erase push the vote sum up, Reject pushes it down. A rule that fires on examples whose targets are too low for it gets rejected there, so it narrows to the range of inputs where its vote is wanted.

---

## Step 5 — Training on a noisy curve

Now the full run on the invented dose–response data: **300 training and 300 test points**, the same curve with independent noise. The settings, fixed in advance:

- 19 thresholds (dose ≥ 0.5, 1.0, …, 9.5 mg), so 38 literals;
- **40 rules**, resolution $T = 20$ (so one vote is worth 0.44 benefit points), Forget Value $1/s$ with $s = 2.5$;
- 60 epochs, seed 1 for the machine's random choices.

We measure the error with the **mean absolute error** (MAE): the average distance between prediction and target, in benefit points. That is the measure the papers report. The widget trains this machine in your browser; change the knobs and it retrains in a second or two.

<div id="viz-train" class="viz"><div class="viz-title">🏋️ Interactive: train a regression Tsetlin machine on a dose curve</div></div>

!!! tip "What to notice"
    - **With the defaults** the machine ends at a test MAE of **0.660** (train 0.606). The green dashed line is the error of the *true* curve itself, **0.472**: that much is pure noise, which no model can predict. The machine is about 0.19 points above that floor.
    - **The test curve jumps around.** Between epochs 11 and 60 the test MAE moves between 0.568 and 1.216. Each Reject Feedback switches a rule off for a range of doses at once, so a single unlucky example can move the staircase by a step for a while. Don't read much into one epoch.
    - **Rules and resolution must fit each other.** With 20 rules and $T = 20$ the machine never gets near the peak (test MAE 3.307): every one of the rules would have to fire at the top of the curve. With 80 rules and $T = 20$ there are too many: 22 rules end up unable to fire, 8 stay empty, and the MAE is 3.206. Twenty rules with $T = 10$ (0.703) or 80 rules with $T = 40$ (0.645) work as well as the default. In our runs about twice as many rules as $T$ worked best on this curve.
    - **Specificity matters too.** $s = 1.5$ gives 1.976 (rules too broad), $s = 4$ gives 0.787, $s = 6$ gives 3.000 (rules too narrow).

### How good is that, honestly?

On the same 300 test points, same data:

| Method (fitted on the same 300 training points) | test MAE |
|---|---|
| Always predict the training mean | 1.625 |
| Straight line (least squares) | 1.557 |
| Regression tree, depth 2 / 3 / 4 / 5 | 0.777 / 0.660 / 0.534 / 0.544 |
| **Regression TM**, defaults, seed 1 | **0.660** |
| **Regression TM**, defaults, seeds 1–10 | **0.595 to 0.874**, mean 0.693 |
| The true curve (noise floor) | 0.472 |

So our toy machine is clearly better than a straight line, about as good as a depth-3 regression tree, and a bit worse than a depth-4 tree on this one-input problem. It is a small, untuned implementation. The papers tune $T$, the number of rules and $s$ by grid search for each dataset.

What the papers report on real data (not rerun by us): Abeyrathna et al. (2020, Table 3) compare the regression TM with a regression tree, a random forest and support vector regression on five public datasets (six target series), averaging 20 runs. The regression TM had the lowest MAE on four of the six, for example **5.160** against 7.769, 9.122 and 5.305 on dengue incidence in the Philippines, and **0.487** against 0.598, 0.563 and 1.155 on the heating load of buildings. On aerofoil noise the random forest was best (1.921 against the TM's 2.206), and on annual stock return the support vector regression was (0.073 against 0.078).

---

## Step 6 — Reading the rules: rediscovering a points score

A model you can read is most convincing when you know the right answer. So we generated 1,200 **simulated** patients (600 to train, 600 to test): each condition is present at random with a fixed, invented prevalence (heart failure 20%, hypertension 60%, diabetes 25%, prior stroke/TIA 15%, vascular disease 20%, female 45%), and age is uniform between 45 and 90. The target is the patient's points score from Step 1, so there is **no noise**: the exact answer is a sum of nine rules.

Age is thermometer-coded with 8 thresholds (age ≥ 50, 55, …, 85), so the machine has to find out *which* thresholds matter. Settings: **9 rules, $T = 9$** (so one vote = one point), range 0–9, $s = 2$, 40 epochs. After **16 epochs** (seed 1) the test MAE is **0.000**: every one of the 600 test patients gets exactly the right score. The learned rules (the "learned" option in the Step 1 widget):

| Learned rule (each +1 point) | how many |
|---|---|
| IF prior stroke/TIA | 2 |
| IF age ≥ 60 AND age ≥ 75 (that is: age ≥ 75) | 1 |
| IF age ≥ 65 | 1 |
| IF heart failure / hypertension / diabetes / vascular disease / female | 1 each |

That **is** the points table: two rules for a stroke, two age steps at exactly 65 and 75, and one rule for each other risk factor. One rule carries a redundant literal (*age ≥ 60* is implied by *age ≥ 75*); merging literals of the same property, as in the classification lesson, removes it.

How robust is this? Over 10 seeds of the machine (same data), with $s = 2$:

| Rules (resolution $T = 9$) | exact on all 600 test patients | mean test MAE |
|---|---|---|
| 6 | 0 of 10 | 0.971 |
| **9** | **10 of 10** (after 3 to 25 epochs) | **0.000** |
| 12 | 1 of 10 | 1.025 |
| 18 | 0 of 10 | 1.345 |
| *always predict the training mean* | | *1.191* |

With $s = 1.5$, 8 of 10 seeds were exact; with $s = 3$, only 4 of 10. Raising the resolution to $T = 18$ with 18 rules (two votes per point) gave MAE 0.654 on average, with no exact run.

!!! tip "What to notice"
    - **Nine rules for nine points is a perfect fit**, the same observation as the paper's artificial datasets: with $T = 7$ rules, an output of $100 \times$ (a 3-bit number) is learned exactly by four copies of "bit 1", two of "bit 2" and one of "bit 3" (Abeyrathna et al., 2020, §5).
    - **Too many rules hurt more than too few here.** Extra rules have no place to go: every vote they cast is a point too many. With 18 rules the machine was worse than predicting the mean. Integer weights (next step) are one fix.
    - **The rules are checkable.** A clinician can compare each rule with the published score and see at once that it is right. On a real outcome (say, actual strokes) the data would be noisy, and the rules would be a hypothesis to check, not a rediscovered truth.

!!! warning "Simulated, not clinical"
    The patients, prevalences and the independence between risk factors are invented for this exercise, and the target is the score itself, not an outcome. The point is to show that the machine can find thresholds and point values that are really there, not to build a risk model.

---

## Step 7 — Weighted rules: one rule instead of many copies

In Step 6, a 2-point item needed **two identical rules**, and in Step 2 several intervals appeared twice. The resolution is tied to the number of rules: to output 7 the machine needs 7 firing rules. Abeyrathna, Granmo and Goodwin (2020, arXiv 2002.01245) therefore gave each rule a whole-number **weight** $w_j \ge 0$:

$$
\hat y \;=\; y_{\min} + \frac{1}{T}\sum_j w_j\, C_j\,\bigl(y_{\max} - y_{\min}\bigr).
$$

The weight is learned with the same feedback, by counting: when a rule is picked, fires and the prediction was too low, its weight goes up by 1; when it is picked and the prediction was too high, its weight goes down by 1 (never below 0). A weight of 0 switches a useless rule off, which also deals with the "too many rules" problem above.

The paper reports, on its 3-bit dataset with $T = 7$: the plain machine needs **7 rules** (4 × "bit 1", 2 × "bit 2", 1 × "bit 3"), the integer-weighted one needs **3 rules with weights 4, 2 and 1**, and both reach MAE 0. Real-valued weights, learned with a learning rate, did not reach zero error (MAE 1.857 on training and 1.799 on test) (Table 1). The authors argue that integer weights are also easier to read, because a weight of 4 simply means "four copies of this rule".

Weights are the subject of the next part of the book; see [Weighted clauses and clause sharing](../weighted-tsetlin-machine/index.md). We did not include them in our implementation. A quick test of the update above on the points score was much less reliable than the plain machine: with 9 weighted rules, none of 10 runs was exact (test MAE 0.318 to 1.368); with 18 weighted rules, 2 of 5 runs were. The paper's results come from a tuned implementation; tuning ours is beyond this lesson.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧠 A regression Tsetlin machine, from record to number",
  "cols": 3,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "raw", "col": 0, "row": 0, "icon": "🧾", "label": "Patient record", "sub": "age 78 · hypertension · female", "role": "data"},
    {"id": "bool", "col": 1, "row": 0, "icon": "🌡️", "label": "Booleanize", "sub": "thermometer bits + negations", "role": "data", "note": "One feature per threshold (age ≥ 50, …, age ≥ 85), one per yes/no fact, plus the negation of each."},
    {"id": "rules", "col": 2, "row": 0, "icon": "🧩", "label": "Rules check their conditions", "sub": "all vote the same way", "role": "model", "note": "Each rule is an AND of the literals in the Memorized half of its memory. With thresholds, a rule is an interval."},
    {"id": "vote", "col": 2, "row": 1, "icon": "🗳️", "label": "Vote sum v = firing rules", "sub": "clipped to [0, T]", "role": "loss"},
    {"id": "out", "col": 1, "row": 1, "icon": "🔢", "label": "ŷ = y<sub>min</sub> + v / T × range", "shape": "pill", "role": "opt"},
    {"id": "learn", "col": 0, "row": 1, "icon": "🎲", "label": "Training: feedback", "sub": "probability |v − τ| / T", "role": "grad", "note": "Too low: Recognize / Erase. Too high: Reject."}
  ],
  "edges": [
    {"from": "raw", "to": "bool"},
    {"from": "bool", "to": "rules"},
    {"from": "rules", "to": "vote"},
    {"from": "vote", "to": "out"},
    {"from": "out", "to": "learn"},
    {"from": "learn", "to": "raw", "label": "next"}
  ],
  "steps": [
    {"show": ["raw"], "text": "Start from a record with numbers and yes/no facts."},
    {"show": ["raw>bool", "bool"], "text": "Booleanize: thermometer bits for each number (age ≥ 65, age ≥ 75, …), one bit per yes/no fact, and the negation of every bit. These are the literals."},
    {"show": ["bool>rules", "rules"], "text": "Each rule fires if all its memorized literals are True. On thermometer bits, a rule is an interval such as 'age 65–74'."},
    {"show": ["rules>vote", "vote"], "text": "Every firing rule adds one vote. There are no rules against. Clip the count to at most T."},
    {"show": ["vote>out", "out"], "text": "Scale the votes to the target's range. That is the whole prediction: a sum of readable rules."},
    {"show": ["out>learn", "learn"], "text": "During training, compare with the target in votes τ. Too low: Recognize or Erase; too high: Reject; each rule picked with probability |v − τ| / T."},
    {"show": ["learn>raw", "raw"], "text": "Move on to the next example, for many epochs, until the staircase fits."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 115, "pos": {"raw": [0, 0], "bool": [1, 0], "rules": [1, 1], "vote": [1, 2], "out": [0, 2], "learn": [0, 1]}}
}
</script>
</div>

Here is the complete regression machine in numpy, about 60 lines, trained on the simulated points score of Step 6. It follows Step 4 line by line, vectorised over rules and literals. It draws its own patients and uses numpy's random numbers, so its numbers differ slightly from the widgets'.

```python
from collections import Counter

import numpy as np

# ---- 1. Simulated patients and a points score modelled on CHA2DS2-VASc (invented prevalences, not real data)
rng = np.random.default_rng(0)
n = 1200
yes_no = rng.random((n, 6)) < [0.2, 0.6, 0.25, 0.15, 0.2, 0.45]   # CHF, hypertension, diabetes, stroke/TIA, vascular, female
age = rng.uniform(45, 90, n)
score = yes_no @ [1, 1, 1, 2, 1, 1] + (age >= 65) + (age >= 75)   # 0 … 9 points

AGE_CUTS = [50, 55, 60, 65, 70, 75, 80, 85]                       # thermometer: "age ≥ 50", "age ≥ 55", …
X = np.hstack([yes_no, age[:, None] >= AGE_CUTS]).astype(np.int8)
names = ["heart failure", "hypertension", "diabetes", "stroke/TIA", "vascular", "female"] + [f"age ≥ {a}" for a in AGE_CUTS]
train, test = np.arange(600), np.arange(600, n)


class RegressionTM:
    """memory[j, k] = position (1 … depth) of literal k in rule j; memorized if > depth / 2."""

    def __init__(self, n_features, n_rules=9, T=9, s=2.0, depth=100, y_min=0.0, y_max=9.0, seed=1):
        self.T, self.s, self.depth, self.half = T, s, depth, depth // 2
        self.y_min, self.y_max = y_min, y_max
        self.rng = np.random.default_rng(seed)
        self.memory = np.full((n_rules, 2 * n_features), depth // 2)  # top of Forgotten

    def fires(self, lits, training=False):
        memorized = self.memory > self.half
        ok = ~np.any(memorized & (lits == 0), axis=1)
        return ok if training else ok & memorized.any(axis=1)     # empty rules stay silent when predicting

    def predict(self, x):
        v = min(self.fires(np.concatenate([x, 1 - x])).sum(), self.T)  # vote sum, clipped to [0, T]
        return self.y_min + v / self.T * (self.y_max - self.y_min)

    def update(self, x, y):
        lits = np.concatenate([x, 1 - x])
        f = self.fires(lits, training=True)
        v = min(f.sum(), self.T)
        target = (y - self.y_min) / (self.y_max - self.y_min) * self.T  # the target, in votes
        error = v - target
        if error == 0:
            return
        chosen = self.rng.random(len(f)) <= min(abs(error) / self.T, 1.0)  # error size -> feedback probability
        u = self.rng.random(self.memory.shape)
        m = self.memory                                            # changes go straight into self.memory
        if error < 0:                                              # too low: Recognize / Erase (Type I)
            rec = (chosen & f)[:, None]
            m += rec & (lits == 1) & (m < self.depth)              # Memorize Value 1
            m -= rec & (lits == 0) & (u <= 1 / self.s) & (m > 1)   # Forget Value 1/s
            era = (chosen & ~f)[:, None]
            m -= era & (u <= 1 / self.s) & (m > 1)
        else:                                                      # too high: Reject (Type II) for firing rules
            rej = (chosen & f)[:, None]
            m += rej & (lits == 0) & (m <= self.half)

    def fit(self, X, y, epochs):
        for _ in range(epochs):
            for i in self.rng.permutation(len(X)):
                self.update(X[i], y[i])
        return self

    def rules(self, names):
        lits = names + ["NOT " + n for n in names]
        return [" AND ".join(l for l, m in zip(lits, rule) if m > self.half) or "(empty)" for rule in self.memory]


tm = RegressionTM(X.shape[1], n_rules=9, T=9, s=2.0).fit(X[train], score[train], epochs=40)
pred = np.array([tm.predict(x) for x in X[test]])
print(f"test MAE {np.abs(pred - score[test]).mean():.3f} · always predicting the mean {np.abs(score[train].mean() - score[test]).mean():.3f}")
for rule, count in Counter(tm.rules(names)).most_common():
    print(f"{count} × IF {rule} THEN +1 point")
```

```text
test MAE 0.000 · always predicting the mean 1.147
2 × IF stroke/TIA THEN +1 point
1 × IF hypertension THEN +1 point
1 × IF vascular THEN +1 point
1 × IF diabetes THEN +1 point
1 × IF age ≥ 60 AND age ≥ 65 THEN +1 point
1 × IF heart failure THEN +1 point
1 × IF female THEN +1 point
1 × IF age ≥ 65 AND age ≥ 75 THEN +1 point
```

It runs in about a second. Again the nine rules are the points table, with redundant age literals (*age ≥ 60 AND age ≥ 65* is just *age ≥ 65*). We also tried machine seeds 2–6: seeds 2–5 were exact too, and seed 6 ended at a test MAE of 0.638.

| Piece | What it does | Book term (research term) |
|---|---|---|
| Rule | AND of memorized literals; adds one vote when True | rule (clause $C_j$, no polarity) |
| Vote sum $v$ | number of firing rules, clipped to $[0, T]$ | vote sum ($\sum_j C_j$) |
| Resolution $T$ | number of votes for the full target range | (threshold / resolution $T$) |
| Output | $y_{\min} + \tfrac{v}{T}(y_{\max} - y_{\min})$ | (normalised vote sum) |
| Target in votes $\tau$ | $\tfrac{y - y_{\min}}{y_{\max} - y_{\min}}T$ | — |
| Feedback probability | $\lvert v - \tau\rvert / T$ | (activation probability $P_{act}$) |
| Too low / too high | Recognize or Erase / Reject | (Type I / Type II feedback) |
| Rule weight $w_j$ | whole-number number of votes per rule | (integer-weighted clauses) |

---

## Common misconceptions

!!! warning "\"A regression Tsetlin machine is just a classifier with many classes.\""
    No. A multi-class machine would treat each output value as an unrelated class, with its own team of rules. The regression machine has **one** team whose votes *add up*, so one rule can contribute to every output value in its interval, and being off by one vote is a small error, not a wrong class.

!!! warning "\"T is the Voting Margin, so it only matters during training.\""
    In classification, $T$ was used only during training. In regression, $T$ is also part of the **prediction**: $\hat y$ uses $v/T$. It fixes the size of one output step, $(y_{\max} - y_{\min})/T$, and with it how many rules must fire to reach the top of the range.

!!! warning "\"More rules can only help.\""
    Not without weights. Every firing rule adds a vote, so surplus rules must learn to *never* fire. On the points score, 18 rules for $T = 9$ did worse (MAE 1.345) than predicting the mean (1.191). On the dose curve, 80 rules for $T = 20$ gave 3.206 against 0.660 for 40 rules.

!!! warning "\"The output is a smooth curve.\""
    It is a staircase with at most $T + 1$ levels, changing only at the thresholds you chose for the inputs. Finer thresholds and a larger $T$ make the steps smaller, at the cost of more rules and more training.

!!! warning "\"Zero training error means the machine found the truth.\""
    On the points score it did, because the data were generated from that score with no noise. On real data, a perfect training fit usually means memorized noise. Always compare the test error with simple baselines and, when you can, with the noise floor.

---

## Check yourself

??? question "1. A regression TM has T = 10 and was trained on targets from 20 to 70. Six rules fire on a patient. What does it predict?"
    $\hat y = 20 + \tfrac{6}{10} \times (70 - 20) = 20 + 30 = $ **50**. One vote is worth 5 units, so the machine can only output 20, 25, 30, …, 70.

??? question "2. Same machine, the target for this training example is 38. What feedback happens?"
    In votes, $\tau = (38 - 20)/50 \times 10 = 3.6$. The prediction is $v = 6$, too **high** by 2.4 votes, so each rule is picked with probability $2.4/10 = 0.24$. Picked rules that fired get **Reject Feedback**: each memorizes a Forgotten literal that is False for this patient, so it stops firing on it. Silent rules are left alone.

??? question "3. A rule has memorized 'dose ≥ 2.0', 'dose ≥ 3.5', 'NOT dose ≥ 6.0' and 'NOT dose ≥ 8.0'. When does it fire?"
    For $3.5 \le$ dose $< 6.0$. The largest "≥" threshold sets the lower end, the smallest negated threshold the upper end; the other two literals are implied and can be dropped.

??? question "4. Why does the probability of feedback shrink as the prediction gets closer to the target?"
    So that the machine settles instead of oscillating. If every rule got feedback on every example, the vote sum would overshoot back and forth around the target (the papers introduce the probability for exactly this reason). With $p = |v - \tau|/T$, well-predicted examples leave the rules alone, and the feedback goes to the examples that are still wrong. It is the regression version of the Voting Margin's resource allocation.

??? question "5. A points score has items worth 1, 2 and 3 points (maximum 6). How would a plain regression TM represent it, and what changes with integer weights?"
    With $T = 6$ and the range 0–6, each vote is one point, so the plain machine needs **6 rules**: one copy of the 1-point rule, two of the 2-point rule and three of the 3-point rule. With integer weights it needs **3 rules**, with weights 1, 2 and 3. Same model, fewer rules, and each weight can be read directly as points.

---

## Further reading

- K. Darshana Abeyrathna, Ole-Christoffer Granmo, Xuan Zhang, Lei Jiao and Morten Goodwin, [The regression Tsetlin machine: a novel approach to interpretable nonlinear regression](https://doi.org/10.1098/rsta.2019.0165), *Philosophical Transactions of the Royal Society A* 378 (2164): 20190165, 2020. The main source for this lesson: the output formula, the feedback rule and its activation probability, the threshold booleanization, and the comparison on real datasets (we read the accepted version in the University of Agder's open archive).
- K. Darshana Abeyrathna, Ole-Christoffer Granmo, Lei Jiao and Morten Goodwin, [The Regression Tsetlin Machine: A Tsetlin Machine for Continuous Output Problems](https://arxiv.org/abs/1905.04206), arXiv 1905.04206, 2019. The earlier preprint, with output scaled by $y_{\max}/T$ and experiments on six artificial bit datasets.
- K. Darshana Abeyrathna, Ole-Christoffer Granmo and Morten Goodwin, [A Regression Tsetlin Machine with Integer Weighted Clauses for Compact Pattern Representation](https://arxiv.org/abs/2002.01245), arXiv 2002.01245, 2020. Integer weights for the regression machine (Step 7).
- K. Darshana Abeyrathna, Ole-Christoffer Granmo, Xuan Zhang and Morten Goodwin, [A Scheme for Continuous Input to the Tsetlin Machine with Applications to Forecasting Disease Outbreaks](https://arxiv.org/abs/1905.04199), arXiv 1905.04199, 2019. The threshold booleanization for continuous inputs, applied to dengue outbreak forecasting.
- K. Darshana Abeyrathna, Ole-Christoffer Granmo and Morten Goodwin, [Convolutional Regression Tsetlin Machine: An Interpretable Approach to Convolutional Regression](https://doi.org/10.1145/3468891.3468901), ICMLT 2021. Combines this lesson with the [convolutional Tsetlin machine](../convolutional-tsetlin-machine/index.md) for numeric outputs from images.
- The [`cair/tmu`](https://github.com/cair/tmu) library, `TMRegressor` (`tmu/models/regression/vanilla_regressor.py`): the maintained implementation, with the squared feedback probability and optional weights.
- Ole-Christoffer Granmo, [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/). Chapter 3, "Regression", is announced but not yet published; the vocabulary of this lesson comes from the published chapters.
- [Classification by voting](../tsetlin-machine-classification/index.md) and [Recognize, Erase & Reject feedback](../tsetlin-machine-feedback/index.md): the machinery this lesson reuses.

**Next lessons:** [Convolutional Tsetlin machines](../convolutional-tsetlin-machine/index.md)

<script src="viz/tsetlin-machine-regression.js"></script>
