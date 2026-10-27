# Confidence, Trustworthiness and Composites

!!! abstract "TL;DR"
    A Tsetlin machine (TM) can tell you **how sure it is**, for free: the **confidence** of a prediction is the size of the vote sum, $|v|$. Many rules agreeing gives a large $|v|$; few rules, or rules that disagree, give a $|v|$ near zero. A TM is **trustworthy** when its confidence matches its performance: the cases it is sure about should be the ones it gets right. You check this on held-out data by answering only the cases above a confidence threshold and measuring accuracy on them, which gives a **coverage/accuracy trade-off** (answer fewer patients, make fewer mistakes, refer the rest to a human). Vote sums are a *ranking*, not probabilities, so check how they line up with real outcomes before reading too much into them. Finally, several specialised TMs can form a **TM composite**: each one **normalises** its vote sum by its own range, and the normalised sums are added. On a breast-cancer dataset, three specialists that each see one group of measurements reach 88.0%, 90.7% and 82.6% accuracy alone, and **96.1%** together.

**Level:** intermediate · **Prerequisites:** [Classification by voting](../tsetlin-machine-classification/index.md) (rules voting for and against a class, the vote sum $v$, the Voting Margin $T$). [Convolutional Tsetlin machines](../convolutional-tsetlin-machine/index.md) helps for the image examples, but isn't needed.

This lesson follows Chapter 7 of Ole-Christoffer Granmo's free book [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/), redrawn and retold in this site's style. The patient experiments are our own runs, made for this lesson.

---

## Why should I care?

Think about your family doctor (the chapter opens with this example). Nobody expects them to know every corner of medicine. You trust them because of one skill: **they know when they don't know**, and then they refer you to a specialist.

A model that helps with clinical decisions needs the same skill, and medical AI is full of situations where it matters:

- **Triage.** A screening model could handle the clear-cut cases and send the doubtful ones to a radiologist or pathologist. That only works if "doubtful" is something the model can measure.
- **Safety.** A confident wrong answer is the most dangerous kind. You want to know how often the model is confident *and* wrong before anyone relies on it.
- **Combining sources.** Hospitals hold different patient populations; one patient has blood tests, images and clinical notes. Models trained on each source could work as a team, if their opinions can be put on the same scale.

Tsetlin machines are a good fit here because the same vote sum that decides the class also measures how sure the machine is, and the rules behind every vote can be read. This lesson shows how to measure confidence, how to check whether it deserves trust, and how to build TM composites.

---

## Step 1 — Confidence is the size of the vote sum

Recall from [Classification by voting](../tsetlin-machine-classification/index.md) how a two-class TM decides: every rule whose condition is true casts a vote, votes for the class count $+1$, votes against it count $-1$, and the **vote sum** $v$ decides (class if $v \ge 0$). The chapter's key observation is that $v$ carries more information than its sign:

- **Many rules agree on "yes"** → a large positive $v$. The machine is **confident in the class**.
- **Many rules agree on "no"** → a large negative $v$. The machine is **confident in the other class**.
- **Few rules vote, or they disagree** → $v$ near zero. There is no clear consensus: the machine is **uncertain**.

So the chapter defines confidence as the size of the vote sum, ignoring its sign:

$$
\text{confidence} = |v|
$$

The chapter illustrates this with a small, invented example: predicting which of six students pass a course from three habits (did they follow the lectures, read the textbook, do the assignments?), using three rules. Click the buttons to answer only the students whose confidence reaches a threshold.

<div id="viz-students" class="viz"><div class="viz-title">🎓 Interactive: confidence for six students (the book's Table 7.1)</div></div>

!!! tip "What to notice"
    - **Students #1 and #3** match R1 and R2 and nothing votes against them: $v = +2$, the most confident Pass.
    - **Student #2** is the interesting one. R2 (Textbook) votes Pass, R3 (no Assignments) votes Fail, so $v = 0$: a tie. The machine still has to pick something ($v \ge 0$ gives Pass), and it is wrong. It is also the **least confident** student.
    - **Ranking by confidence:** #1 and #3 ($|v| = 2$), then #4, #5, #6 ($|v| = 1$), then #2 ($|v| = 0$). Answer everyone and accuracy is 5 of 6 (83%). Answer only $|v| \ge 1$ and it is 5 of 5 (100%). The one mistake was the case the machine was least sure about.

**How big is "big"?** The Voting Margin $T$ gives the natural scale. During learning, a TM clips $v$ at $\pm T$ and stops giving feedback for an example once its vote sum reaches $T$ in the right direction: $T$ is how sure training *asks* the machine to be. So $|v| \ge T$ means "as sure as training ever pushed it", and $|v|/T$ is a handy 0-to-1 reading of confidence. (Vote sums can go beyond $T$ when more rules happen to agree; they are simply not pushed further.)

!!! note "More than two classes"
    A multi-class TM keeps one vote sum per class and predicts the class with the **largest** sum. The chapter takes that largest sum as the confidence:

    $$
    \text{multi-class confidence} = \max_{\text{classes}} v_{\text{class}}
    $$

    A common alternative in other models, not used in the chapter, is the gap between the largest and second-largest sums, which also catches the case where two classes are both strongly supported.

---

## Step 2 — Trustworthiness: confidence must match performance

Confidence on its own proves nothing. A person can be very confident and still wrong, and someone too timid to act also does badly. Confidence is only useful when it **matches ability**. The chapter sums this up in a 2 × 2 table:

| | **Low performance** | **High performance** |
|---|---|---|
| **High confidence** | 🚩 Overconfident | ✅ Justifiably confident |
| **Low confidence** | ✅ Justifiably uncertain | 🐢 Underconfident |

For a classifier, you test this by comparing predictions with the true labels at each confidence level, exactly as in the student widget: if accuracy rises as you keep only the more confident cases, then confidence is **faithful to performance**, and the machine is trustworthy in this sense. If accuracy is flat (or even falls), its confidence tells you nothing.

The chapter also reports this test for a much larger problem: a multi-class TM that sorts colour images into 100 classes, using Boolean features made by thresholding each pixel's colour against its neighbours. Ranking 10,000 test images by multi-class confidence (from about −2,500 to 10,000), **accuracy climbs from about 34% for the least confident images to 100% for the most confident ones** (Granmo, Ch. 7, Fig. 7.3; we did not rerun this). The machine was unsure about images made mostly of colour texture, such as a pink flower, and sure about large, distinct objects such as a tank or the Eiffel Tower.

!!! tip "What to notice"
    - Trustworthiness here is a property you **measure** on data the machine has not seen, not something you assume.
    - The test does not need the confidence to be a probability. It only asks: *are the more confident cases more often right?*

---

## Step 3 — Confidence on real patients

Now the real test, on two breast-cancer datasets. For each, we trained our own TM (numpy, written for this lesson following the book's algorithm; 20 rules, Voting Margin $T = 10$, Memorize Value $2/3$, Forget Value $1/3$, 40 passes over the data). To make the test honest, we used **5-fold cross-validation**: the patients are split into five groups, and each group's vote sums come from a machine trained on the other four. Every vote sum below is for a patient the machine had **never seen**.

1. **Diagnosis (Wisconsin).** The public [Wisconsin Diagnostic Breast Cancer data](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (also bundled with scikit-learn): 569 patients, 212 malignant and 357 benign. Each has 30 measurements of the cell nuclei in a digitised image of a fine-needle aspirate (10 properties such as radius, texture and concavity, each as a mean, a standard error and a "worst" value). We Booleanize each measurement with 4 thresholds ("mean radius > 11.39?", and so on, at the 20th, 40th, 60th and 80th percentiles of the training patients), giving 120 Boolean features.
2. **Recurrence (the book's data).** The public [UCI Breast Cancer data](https://archive.ics.uci.edu/dataset/14/breast+cancer) from the Institute of Oncology in Ljubljana, which Chapter 1 of the book uses: 286 patients, 85 of whom had a recurrence, with 9 properties (age, menopause, tumour size, involved nodes, malignancy grade and so on), one Boolean feature per value: 41 features.

Slide the threshold to answer only patients with $|v|$ at or above it. The rest are referred.

<div id="viz-confidence" class="viz"><div class="viz-title">🎚️ Interactive: answer only the confident cases</div></div>

!!! tip "What to notice"
    - **Diagnosis: confidence is very informative.** Answering everyone, the machine is right 92.3% of the time (44 errors in 569). At $|v| \ge 2$ it answers 473 patients (83.1%) with **99.2%** accuracy (4 errors). The 96 referred patients hold **40 of the 44 errors**.
    - **The silent cases.** 42 patients (7.4%) got $v = 0$: either no rule fired, or the votes for and against cancelled out. They are classed malignant by the $v \ge 0$ rule, but 25 of them are benign: a machine with nothing to say should not be trusted to say it.
    - **Confident is not the same as infallible.** At $|v| \ge 10$ (the full Voting Margin) it still makes 1 error in 64 patients: a malignant case that got the most confident possible *benign* vote, $v = -10$.
    - **Recurrence: switch datasets.** Here the TM's overall accuracy is 72.4%, barely above the 70.3% you get by always saying "no recurrence". Accuracy does rise with confidence (84.1% at $|v| \ge 6$), but look at the dashed line: all 69 of those patients are predicted "no recurrence", and always guessing "no recurrence" in that same group *also* gives 84.1%. The rising curve here only shows the machine finding the patients whose recurrence was least likely to begin with; within that group its confidence adds nothing beyond the class balance.

That last point is a general lesson, not a quirk of Tsetlin machines. **Always compare against the simplest baseline, in the same subgroup.** An accuracy-vs-confidence curve that rises can still describe a model that has learned little.

---

## Step 4 — Knowing when to refer: a triage rule

A confidence threshold turns a classifier into a **triage tool**: answer when sure, refer when not. Because a TM is built from readable rules, it can hand the human more than "I don't know". It can show *which* rules fired on each side, so the clinician sees what the conflict was.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🩺 A Tsetlin machine that knows when to refer",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "pat", "col": 0, "row": 0, "icon": "🧾", "label": "New patient", "sub": "Boolean features", "role": "data", "note": "Measurements turned into true/false features, the same way as during training."},
    {"id": "tm", "col": 1, "row": 0, "icon": "🗳️", "label": "Rules vote", "sub": "vote sum v", "role": "model", "note": "Rules whose condition is true vote for or against the class. v = votes for − votes against."},
    {"id": "gate", "col": 1, "row": 1, "label": "|v| ≥ threshold?", "shape": "decision", "role": "loss", "note": "The threshold is chosen on validation data, from the coverage/accuracy trade-off."},
    {"id": "ans", "col": 2, "row": 1, "icon": "✅", "label": "Answer", "sub": "class + rules that fired", "role": "opt", "note": "The prediction comes with its reasons: the rules that voted."},
    {"id": "ref", "col": 0, "row": 1, "icon": "👩‍⚕️", "label": "Refer to a clinician", "sub": "show the conflicting rules", "role": "grad", "note": "Low confidence: few rules fired, or rules on both sides fired. Show both sides to the human."}
  ],
  "edges": [
    {"from": "pat", "to": "tm"},
    {"from": "tm", "to": "gate"},
    {"from": "gate", "to": "ans", "label": "yes"},
    {"from": "gate", "to": "ref", "label": "no"}
  ],
  "steps": [
    {"show": ["pat"], "text": "A new patient arrives, described by the same Boolean features the machine was trained on."},
    {"show": ["pat>tm", "tm"], "text": "Every rule whose condition holds casts its vote. The vote sum <b>v</b> gives both the class (its sign) and the confidence (its size |v|)."},
    {"show": ["tm>gate", "gate"], "text": "Compare |v| with a threshold chosen in advance on validation data. On our diagnosis run, |v| ≥ 2 answered 83.1% of patients at 99.2% accuracy."},
    {"show": ["gate>ans", "ans"], "text": "Confident: report the class, together with the rules that voted for it. That is the explanation."},
    {"show": ["gate>ref", "ref"], "text": "Uncertain: refer the case. Show the rules that fired on each side, or say that none fired, so the clinician knows what the machine saw."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 100, "pos": {"pat": [0, 0], "tm": [1, 0], "gate": [1, 1], "ans": [1, 2], "ref": [0, 1]}}
}
</script>
</div>

What does "the rules that fired" look like? Here are some rules our size specialist (Step 8) learned, from one of the five training runs. Each is an AND of thresholded measurements:

- **if** worst radius > 20.35 **then** malignant
- **if** mean radius > 17.16 **then** malignant
- **if** area SE > 57.94 **then** malignant (SE = standard error)
- **if** worst perimeter ≤ 91.31 **then** benign
- **if** worst perimeter ≤ 91.31 **and** mean perimeter ≤ 91.35 **then** benign

(A few of its other rules are long chains of thresholds that a person would want to simplify, and one rule appears twice, just like the duplicated rules in the book's Chapter 1 example.) This is what the chapter means by trustworthiness in a wider sense: an answer you can check (*why*) together with an honest signal of doubt (*how sure*).

!!! warning "Before anything like this touches patients"
    These are teaching experiments, not a validated clinical tool. A real triage threshold has to be chosen on a separate validation set and then checked on fresh data from the setting where it will be used. The Wisconsin data come from one centre, decades ago; a new scanner, lab or patient population can shift every number above. Errors also have different costs: missing a cancer is not the same as a false alarm, so a real system might demand more confidence before saying "benign" than before saying "malignant" (our own suggestion, not from the book).

---

## Step 5 — An honest look at calibration

It is tempting to read $|v|/T$ as a probability: "$v = 8$ with $T = 10$ means 80% sure". That is **not** what the vote sum is. The TM never learns probabilities; it learns rules and pushes vote sums towards $\pm T$. Whether a vote sum corresponds to 60% or 99% of patients actually being positive is an empirical question, and the answer differs between machines and datasets.

The standard way to look is a **reliability diagram** (as in Guo et al., 2017). Group the held-out patients by their vote sum and, for each group, plot the share that truly had the positive class. Here are both of our machines.

<div id="viz-reliability" class="viz"><div class="viz-title">📊 Interactive: what does each vote sum really mean?</div></div>

!!! tip "What to notice"
    - **Diagnosis: the scale saturates fast.** Every one of the 11 patients with $v = +2$ was malignant, as were all 115 with $v \ge 8$. A vote sum of +2 is "only" 20% of the margin, yet it already behaves like near-certainty. Reading $|v|/T$ as a probability would badly *understate* this machine's confidence.
    - **The middle is where the doubt lives.** $v = 0$: 17 of 42 malignant (40%). $v = +1$: 9 of 18 (50%), even though the machine says "malignant" for all of them. $v = -1$: 6 of 36 (17%).
    - **Small groups, wide error bars.** 11 out of 11 sounds conclusive, but the 95% interval for $v = +2$ runs from 74% to 100%. With a few hundred patients, a per-vote-sum table is a rough sketch.
    - **Recurrence: nothing is certain.** The positive vote sums never pass +4, and their recurrence shares bounce around (8 of 11 at $v = +2$, 6 of 11 at $v = +3$). Even $v = -5$ still had 11 recurrences among 29 patients (38%). This machine's ranking is weak, and the picture shows it immediately.

If you need actual probabilities (for a risk score, or to combine with costs), you can **learn a mapping** from vote sum to probability on validation data, for example by binning as above or by fitting a logistic curve to $v$ (Platt scaling). That mapping is only as good as the data it was fitted on, and it has to be re-checked when the patient population changes.

---

## Step 6 — Different machines, different skills

The second half of the chapter is about teamwork. Just as a house needs carpenters, plumbers and electricians, a hard problem may need several Tsetlin machines with different skills. The chapter names two ways TMs end up with different skills:

- **Different populations.** Two hospitals train their own recurrence TMs. Different demographics, treatments and equipment give each machine its own perspective.
- **Different modalities.** One TM reads blood tests, another mammograms, a third the doctor's notes. Each sees something the others miss.

A **TM composite** lets such machines work together. Each machine runs on its own, completely unchanged, but instead of announcing a class it reports its **vote sum**. Because the only thing exchanged is a vote sum, any kind of TM (plain, convolutional, trained on any data) can join a composite: the chapter calls this *plug-and-play*. The research paper behind the chapter calls them [TMComposites](https://arxiv.org/abs/2309.04801).

The chapter's toy example reuses the recurrence rules from Chapter 1, split into two machines:

- **Recurrence TM**, trained only on recurrence patients: R1 and R2, both "if Deg-malig 3 and NOT Menopause lt40 then Recurrence".
- **Non-Recurrence TM**, trained only on non-recurrence patients: R3, "if Inv-nodes 0-2 then Non-Recurrence".

Step through the composite for patient #1 of the book's Table 1.9 (Menopause ge40, Inv-nodes 3-5, Deg-malig 3; she did have a recurrence).

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧩 A two-machine composite (after the book's Fig. 7.5)",
  "cols": 4,
  "rows": 2,
  "rowHeight": 120,
  "nodes": [
    {"id": "pat", "col": 0, "row": 0, "icon": "🧾", "label": "Patient #1", "sub": "ge40 · nodes 3-5 · grade 3", "role": "data", "note": "Menopause ge40, Inv-nodes 3-5, Deg-malig 3."},
    {"id": "rec", "col": 1, "row": 0, "icon": "🗳️", "label": "Recurrence TM", "sub": "R1, R2 fire: v = +2", "role": "model", "note": "R1 = R2 = if Deg-malig 3 and NOT Menopause lt40 then Recurrence. Both conditions hold."},
    {"id": "non", "col": 1, "row": 1, "icon": "🗳️", "label": "Non-Recurrence TM", "sub": "R3 silent: v = 0", "role": "model", "note": "R3 = if Inv-nodes 0-2 then Non-Recurrence. This patient has 3-5 nodes, so R3 stays silent."},
    {"id": "nrec", "col": 2, "row": 0, "icon": "➗", "label": "÷ 2", "sub": "+2 → +1", "role": "loss", "note": "On the six patients, this machine's vote sums range from 0 to +2, so its normalisation factor is 2."},
    {"id": "nnon", "col": 2, "row": 1, "icon": "➗", "label": "÷ 1", "sub": "0 → 0", "role": "loss", "note": "This machine's vote sums range from 0 to 1, so its factor is 1."},
    {"id": "sum", "col": 3, "row": 0, "icon": "➕", "label": "+1 − 0 = +1", "sub": "≥ 0 → Recurrence", "role": "opt", "note": "Non-Recurrence votes count against Recurrence. The composite sum +1 is ≥ 0, so the composite predicts Recurrence: correct."}
  ],
  "edges": [
    {"from": "pat", "to": "rec"},
    {"from": "pat", "to": "non"},
    {"from": "rec", "to": "nrec"},
    {"from": "non", "to": "nnon"},
    {"from": "nrec", "to": "sum"},
    {"from": "nnon", "to": "sum"}
  ],
  "steps": [
    {"show": ["pat"], "text": "Patient #1: Menopause ge40, 3-5 involved nodes, malignancy grade 3. Both machines see the same patient."},
    {"show": ["pat>rec", "rec", "pat>non", "non"], "text": "Each machine votes on its own. The Recurrence TM's two rules both fire (<b>v = +2</b>); the Non-Recurrence TM's rule does not (<b>v = 0</b>)."},
    {"show": ["rec>nrec", "nrec", "non>nnon", "nnon"], "text": "Each machine divides by its own range (highest − lowest vote sum on the data): 2 for the Recurrence TM, 1 for the other. Now both speak on the same scale."},
    {"show": ["nrec>sum", "nnon>sum", "sum"], "text": "Add them, counting Non-Recurrence votes as negative: <b>+1 − 0 = +1 ≥ 0</b>, so the composite says Recurrence. The confident machine decided; the silent one stayed out of the way."}
  ],
  "narrow": {"cols": 2, "rows": 4, "rowHeight": 100, "hideSub": false, "pos": {"pat": [0.5, 0], "rec": [0, 1], "non": [1, 1], "nrec": [0, 2], "nnon": [1, 2], "sum": [0.5, 3]}}
}
</script>
</div>

Doing the same for all six patients of the book's Table 1.9 (our calculation):

| Patient | Recurrence TM $v$ → ÷ 2 | Non-Recurrence TM $v$ → ÷ 1 | Composite | Prediction | Truth |
|---|---|---|---|---|---|
| #1 | +2 → +1 | 0 → 0 | +1 | Recurrence | Recurrence |
| #2 | 0 → 0 | 1 → 1 | −1 | No recurrence | No recurrence |
| #3 | +2 → +1 | 0 → 0 | +1 | Recurrence | Recurrence |
| #4 | 0 → 0 | 1 → 1 | −1 | No recurrence | No recurrence |
| #5 | +2 → +1 | 1 → 1 | 0 | Recurrence | Recurrence |
| #6 | 0 → 0 | 1 → 1 | −1 | No recurrence | No recurrence |

!!! note "A note on signs"
    The chapter writes the composite as the *sum* of the two normalised vote sums and predicts Recurrence when it is $\ge 0$. For that to work, the Non-Recurrence machine's votes must count as **negative** (they are votes against Recurrence), just like rules voting against a class in a single TM. With more than two classes this question disappears: each machine reports one sum per class, the sums are added class by class, and the largest wins.

---

## Step 7 — Normalisation: one scale for every machine

Why divide at all? Because members of a composite can be built very differently: different data, different numbers of rules, different Voting Margins. A machine with 40 rules can produce vote sums of ±20; one with 10 rules rarely goes past ±5. Added raw, the big machine would drown out the small one, whatever their actual skill.

The chapter's fix is simple. Each machine finds the **highest** and **lowest** vote sum it produces on the (training) data, and divides by the difference:

$$
\text{normalised vote sum} = \frac{v}{\;v_{\text{highest}} - v_{\text{lowest}}\;}
$$

The composite then adds the normalised sums and decides as usual:

$$
v_{\text{composite}} = \sum_{\text{machines } m} \frac{v_m}{\;v_{m,\text{highest}} - v_{m,\text{lowest}}\;}, \qquad \text{class if } v_{\text{composite}} \ge 0
$$

After normalisation, every machine's opinion lives on a range of width 1. A machine that is confident relative to its *own* range counts for more than one that is lukewarm, regardless of size. That is how the composite "leans on the confident ones".

!!! tip "What to notice"
    - The factor is a property of the **machine**, computed once from data, not of the patient.
    - Use the training data (or a validation set) for the range, never the patients you test on.
    - Normalising by the range is the chapter's choice. It is simple and needs no labels, but it is sensitive to a single extreme vote sum; other scalings are possible.

---

## Step 8 — Build a composite

Back to the Wisconsin data. Instead of one machine that sees all 30 measurements, we trained three **specialists**, each seeing only one group of measurements, and gave them deliberately different sizes:

| Specialist | Sees (mean, SE and worst of …) | Boolean features | Rules | $T$ | Normalisation factor (per fold) | Accuracy alone |
|---|---|---|---|---|---|---|
| Size TM | radius, perimeter, area | 36 | 10 | 4 | 7 to 10 | 88.0% |
| Shape TM | compactness, concavity, concave points | 36 | 40 | 20 | 40 | 90.7% |
| Texture TM | texture, smoothness, symmetry, fractal dimension | 48 | 20 | 10 | 18 to 20 | 82.6% |

All other settings are as in Step 3, with the same 5-fold cross-validation and seed. Tick specialists on and off, switch normalisation off, and browse the patients on whom the specialists disagree.

<div id="viz-composite" class="viz"><div class="viz-title">🧩 Interactive: three specialists, one composite</div></div>

!!! tip "What to notice"
    - **The team beats every member.** Alone, the specialists reach 88.0%, 90.7% and 82.6%. The normalised composite of all three reaches **96.1%** (22 errors in 569). The specialists disagree on 183 patients; the composite gets 164 of those right.
    - **Normalisation matters here.** Switch it off: the raw sum drops to 94.2%, because the 40-rule shape machine shouts loudest. The first patient shown (patient 11, malignant) is a case in point: size +2/8 = +0.25, shape −14/40 = −0.35, texture +3/19 = +0.16. Normalised, the sum is +0.06 → malignant ✓. Raw, it is 2 − 14 + 3 = −9 → benign ✗.
    - **But it is not magic.** Among the 183 disagreement cases, normalising fixed 14 and broke 3. For the shape + texture pair it even does worse than the raw sum (90.0% vs 91.7%). Normalising makes the scales comparable; it cannot know which machine is right.
    - **Fair comparison.** One TM on all 30 measurements with 20 rules reaches 92.3%; with 70 rules (as many as the three specialists together) it reaches 96.1%, the same as the composite. The composite's advantage is not raw accuracy: it is that **separately trained machines can be combined without retraining**, which is exactly the situation of different hospitals or different modalities.
    - **The composite has a confidence too**: $|v_{\text{composite}}|$. Answering only $|v_{\text{composite}}| \ge 0.25$ covers 90.7% of patients at 98.6% accuracy, and $\ge 0.5$ covers 76.1% at 99.3%.

We repeated the whole experiment with two other random splits (seeds 2 and 3). The normalised composite reached 94.9% and 96.1%, the raw sum 93.5% and 94.2%, and no single specialist went above 90.7%. So the ranking held, though the exact numbers move by a percentage point or so.

!!! note "The chapter's image composite"
    The chapter's larger example (Fig. 7.6) combines four multi-class TMs for colour images, each with a different view of the same picture: one on a **histogram of gradients**, one on **colour thresholding** with 10 × 10 convolution filters, and two on **colour thermometers** with 3 × 3 and 4 × 4 convolution filters (see [Convolutional Tsetlin machines](../convolutional-tsetlin-machine/index.md)). Each normalises its class sums, the sums are added class by class, and the composite predicts the class with the largest total (argmax). The book develops image understanding in a later chapter.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔗 A trustworthy TM composite, end to end",
  "cols": 3.3,
  "rows": 3.1,
  "rowHeight": 95,
  "groups": [
    {"id": "team", "label": "Specialist TMs, trained separately", "col": 0, "row": 0, "cols": 1, "rows": 3.1}
  ],
  "nodes": [
    {"id": "size", "col": 0, "row": 0.45, "icon": "📏", "label": "Size TM", "sub": "v ÷ its range", "role": "model", "note": "Sees only radius, perimeter and area. 10 rules, T = 4."},
    {"id": "shape", "col": 0, "row": 1.3, "icon": "🔷", "label": "Shape TM", "sub": "v ÷ its range", "role": "model", "note": "Sees only compactness and concavity. 40 rules, T = 20."},
    {"id": "tex", "col": 0, "row": 2.15, "icon": "🧶", "label": "Texture TM", "sub": "v ÷ its range", "role": "model", "note": "Sees only texture, smoothness, symmetry and fractal dimension. 20 rules, T = 10."},
    {"id": "sum", "col": 1.25, "row": 1.3, "icon": "➕", "label": "Add normalised votes", "sub": "composite v", "role": "loss", "note": "Two classes: one number, class if ≥ 0. Many classes: add per class, then argmax."},
    {"id": "gate", "col": 2.3, "row": 1.3, "label": "|v| ≥ threshold?", "shape": "decision", "role": "loss", "note": "The composite's own confidence. Pick the threshold on validation data."},
    {"id": "ans", "col": 2.3, "row": 0.45, "icon": "✅", "label": "Answer + rules", "role": "opt"},
    {"id": "ref", "col": 2.3, "row": 2.15, "icon": "👩‍⚕️", "label": "Refer", "role": "grad"}
  ],
  "edges": [
    {"from": "team", "to": "sum"},
    {"from": "sum", "to": "gate"},
    {"from": "gate", "to": "ans", "label": "yes"},
    {"from": "gate", "to": "ref", "label": "no"}
  ],
  "steps": [
    {"show": ["team", "size", "shape", "tex"], "text": "Each specialist is an ordinary TM trained on its own view of the data (its own features, rules and Voting Margin). It reports a vote sum, divided by its own range of vote sums."},
    {"show": ["team>sum", "sum"], "text": "Add the normalised votes. On our run: 96.1% accuracy, against 88.0%, 90.7% and 82.6% for the specialists alone."},
    {"show": ["sum>gate", "gate"], "text": "The size of the composite vote sum is its confidence. |v| ≥ 0.25 kept 90.7% of patients at 98.6% accuracy."},
    {"show": ["gate>ans", "ans"], "text": "Confident: answer, with the rules that fired in each specialist as the explanation."},
    {"show": ["gate>ref", "ref"], "text": "Not confident: refer to a clinician. Check the whole pipeline on held-out data, against simple baselines, before trusting it."}
  ],
  "narrow": {"cols": 2, "rows": 5.8, "rowHeight": 90, "hideSub": true, "pos": {"team": [0, 0, 2, 2.5], "size": [0, 0.6], "shape": [1, 0.6], "tex": [0.5, 1.5], "sum": [0, 2.85], "gate": [0, 3.85], "ref": [1, 3.85], "ans": [0, 4.85]}}
}
</script>
</div>

Here is the whole composite experiment in numpy, the Tsetlin machine included. It reproduces the numbers in Step 8 and runs in about half a minute.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer


class TsetlinMachine:
    """Two-class TM: the first half of the rules vote for class 1, the rest for class 0."""

    def __init__(self, n_features, n_rules, T, s=3.0, depth=200, seed=0):
        self.T, self.s, self.N = T, s, depth // 2                       # positions 1..depth; memorized above N
        self.rng = np.random.default_rng(seed)
        self.memory = np.full((n_rules, 2 * n_features), self.N)     # every literal starts barely forgotten
        self.sign = np.where(np.arange(n_rules) < n_rules // 2, 1, -1)

    def vote(self, X):                                                  # vote sum v for every row of X
        lit, mem = np.hstack([X, 1 - X]), self.memory > self.N
        fired = ((1 - lit) @ mem.T == 0) & mem.any(axis=1)              # all memorized literals true, rule not empty
        return fired.astype(int) @ self.sign

    def fit(self, X, y, epochs=40):
        for _ in range(epochs):
            for i in self.rng.permutation(len(y)):
                lit = np.concatenate([X[i], 1 - X[i]])
                out = ~np.any((self.memory > self.N) & (lit == 0), axis=1)
                v = int(out @ self.sign) * (1 if y[i] == 1 else -1)       # votes for the object's own class
                p = (self.T - np.clip(v, -self.T, self.T)) / (2 * self.T)   # far from the margin: more feedback
                chosen = self.rng.random(len(out)) < p
                own = self.sign == (1 if y[i] == 1 else -1)
                for j in np.where(chosen & own)[0]:                     # Recognize / Erase feedback
                    r = self.rng.random(lit.size)
                    if out[j]:
                        self.memory[j] += (lit == 1) & (r < (self.s - 1) / self.s)
                        self.memory[j] -= (lit == 0) & (r < 1 / self.s)
                    else:
                        self.memory[j] -= r < 1 / self.s
                    np.clip(self.memory[j], 1, 2 * self.N, out=self.memory[j])
                for j in np.where(chosen & ~own & out)[0]:              # Reject feedback
                    self.memory[j] += (lit == 0) & (self.memory[j] <= self.N)
        return self


def thermometer(train, test, q=4):                                      # "x > threshold" bits, 4 per measurement
    cuts = np.quantile(train, np.arange(1, q + 1) / (q + 1), axis=0)
    bits = lambda A: np.concatenate([(A > c).astype(np.int8) for c in cuts], axis=1)
    return bits(train), bits(test)


data = load_breast_cancer()
X, y, names = data.data, (data.target == 0).astype(int), list(data.feature_names)   # y = 1: malignant
cols = lambda *kinds: [i for k in kinds for i, n in enumerate(names) if k in n]
views = {"size": (cols("radius", "perimeter", "area"), 10, 4),                        # (columns, rules, T)
         "shape": (cols("compactness", "concavity", "concave points"), 40, 20),
         "texture": (cols("texture", "smoothness", "symmetry", "fractal"), 20, 10)}

rng = np.random.default_rng(1)                                          # stratified 5-fold split
fold = np.zeros(len(y), int)
for c in (0, 1):
    idx = rng.permutation(np.where(y == c)[0])
    fold[idx] = np.arange(len(idx)) % 5

normalised, raw = {}, {}
for name, (columns, rules, T) in views.items():
    raw[name], normalised[name] = np.zeros(len(y)), np.zeros(len(y))
    for k in range(5):                                                  # every vote comes from a TM that never saw the patient
        tr, te = thermometer(X[fold != k][:, columns], X[fold == k][:, columns])
        tm = TsetlinMachine(tr.shape[1], rules, T, seed=100 + k).fit(tr, y[fold != k])
        seen = tm.vote(tr)
        raw[name][fold == k] = tm.vote(te)
        normalised[name][fold == k] = tm.vote(te) / (seen.max() - seen.min())   # Eq. 7.3
    print(f"{name:8s} alone: accuracy {np.mean((raw[name] >= 0) == y):.1%}")

composite = normalised["size"] + normalised["shape"] + normalised["texture"]
print(f"composite, raw sums:        {np.mean((sum(raw.values()) >= 0) == y):.1%}")
print(f"composite, normalised sums: {np.mean((composite >= 0) == y):.1%}")
for threshold in (0.25, 0.5):                                           # confidence = |composite vote sum|
    keep = np.abs(composite) >= threshold
    print(f"|composite| >= {threshold}: answer {keep.mean():.1%} of patients, "
          f"accuracy {np.mean((composite[keep] >= 0) == y[keep]):.1%}, refer {np.sum(~keep)}")
```

```text
size     alone: accuracy 88.0%
shape    alone: accuracy 90.7%
texture  alone: accuracy 82.6%
composite, raw sums:        94.2%
composite, normalised sums: 96.1%
|composite| >= 0.25: answer 90.7% of patients, accuracy 98.6%, refer 53
|composite| >= 0.5: answer 76.1% of patients, accuracy 99.3%, refer 136
```

Each literal's memory here has a depth of 200 positions (1 to 200, memorized above 100), rules start with every literal at position 100, just below the memorized half, and `s = 3` gives Memorize Value 2/3 and Forget Value 1/3, as in [Recognize, Erase & Reject feedback](../tsetlin-machine-feedback/index.md). The feedback probability $(T - v)/(2T)$ is the book's $(2 - v)/4$ with a general Voting Margin.

---

## Common misconceptions

!!! warning "\"A confident Tsetlin machine is a correct one.\""
    Confidence makes errors *rarer*, not impossible. Our diagnosis TM gave one malignant patient the most confident possible benign vote ($v = -10$). Always report how many confident errors remain, not just the accuracy.

!!! warning "\"If accuracy rises with confidence, the model is useful.\""
    Check the baseline in the same subgroup. On the recurrence data, accuracy rose to 84.1% at $|v| \ge 6$, but every one of those 69 patients was predicted "no recurrence", and simply always saying "no recurrence" scores 84.1% there too.

!!! warning "\"$v = 8$ with $T = 10$ means an 80% chance.\""
    Vote sums are a ranking, not probabilities. On the diagnosis data, all 11 patients with $v = +2$ (only 20% of the margin) were malignant, while $v = +1$ was a coin flip (9 of 18). To get probabilities, fit a mapping on validation data, and re-check it whenever the setting changes.

!!! warning "\"A composite wins because it has more rules.\""
    A single TM with the same total of 70 rules matched our composite (96.1% each). The point of a composite is that machines trained **separately**, on different data, sizes and margins, can be combined plug-and-play, and each stays small and readable.

!!! warning "\"Normalisation always improves the composite.\""
    It makes the machines' scales comparable, which helped the full trio (96.1% vs 94.2% raw). But for the shape + texture pair, the raw sum did better (91.7% vs 90.0%). Normalisation decides *how loud* each machine is, not *which one is right*.

---

## Check yourself

??? question "1. One TM gives patient A the vote sum $v = -3$ and patient B $v = +1$. Which prediction is more confident, and what does each predict?"
    Confidence is $|v|$: patient A has confidence 3, patient B has 1, so **A's prediction is more confident**. A is predicted as the *other* class (negative sum); B is predicted as the class ($v \ge 0$), but only weakly: few rules or disagreeing rules.

??? question "2. A multi-class TM gives vote sums Car = 4, Plane = 9, Boat = −2. What does it predict, and with what confidence (the chapter's definition)?"
    It predicts the class with the largest sum, **Plane**, and the chapter's multi-class confidence is that largest sum, **9**. (The gap to the runner-up, 9 − 4 = 5, is another useful measure, though not the chapter's.)

??? question "3. Machine A's training vote sums range from −5 to +4; machine B's from −20 to +20. For a patient, A votes +3 and B votes −12. What does the composite predict, with and without normalisation?"
    Normalised: A gives $3 / (4 - (-5)) = 3/9 \approx +0.33$, B gives $-12 / 40 = -0.30$. The sum is about **+0.03 ≥ 0**: the positive class, although only just (very low composite confidence). Raw: $3 - 12 = -9$, the **negative class**. Without normalisation, B wins simply because it has more rules.

??? question "4. On our diagnosis run, answering only $|v| \ge 2$ kept 473 of 569 patients at 99.2% accuracy. What is the price, and how would you choose the threshold for real use?"
    The price is **96 referred patients** (16.9%) that a human must handle; in exchange, 40 of the 44 errors land in that referred group instead of being reported confidently. A real threshold is chosen on a **separate validation set**, weighing the cost of a missed cancer, a false alarm and a referral, and then checked on fresh data from the clinic where it will be used, not on the test data quoted here.

??? question "5. Why must the confidence check use patients the machine has not seen?"
    On its training patients a TM has been pushed to reach the Voting Margin, so its vote sums there are large and mostly right: confidence would look perfectly trustworthy. Only held-out patients (here via 5-fold cross-validation) show how confidence behaves on new cases, which is the only situation that matters in practice.

---

## Further reading

- Ole-Christoffer Granmo, [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/), **Chapter 7** (confidence, trustworthiness and composites; the source of this lesson, including the student example, Tables 7.1 to 7.3 and Figs. 7.3 to 7.6), and Chapter 1, §1.5 to §1.6 (the vote sum and the Voting Margin; Table 1.9).
- Ole-Christoffer Granmo, [*TMComposites: Plug-and-Play Collaboration Between Specialized Tsetlin Machines*](https://arxiv.org/abs/2309.04801) (2023). The research paper behind the composites.
- Chuan Guo, Geoff Pleiss, Yu Sun & Kilian Weinberger, [*On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599) (2017). Reliability diagrams and why confident models are often miscalibrated.
- Yonatan Geifman & Ran El-Yaniv, [*Selective Classification for Deep Neural Networks*](https://arxiv.org/abs/1705.08500) (2017). The coverage/accuracy trade-off ("selective prediction") in general.
- Data: W. N. Street, W. H. Wolberg & O. L. Mangasarian (1993), the [Wisconsin Diagnostic Breast Cancer data](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic); M. Zwitter & M. Soklic, the [UCI Breast Cancer (recurrence) data](https://archive.ics.uci.edu/dataset/14/breast+cancer), University Medical Centre, Institute of Oncology, Ljubljana.

**Next lessons:** This is the last published chapter of the book so far. See the [Tsetlin machines section of the roadmap](../../roadmap.md#tsetlin-machines) for what comes next; later chapters (for example on image understanding) will be added when they are published.

<script src="viz/tsetlin-machine-confidence.js"></script>
