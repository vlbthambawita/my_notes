# Many Rules Voting: Classification with a Tsetlin Machine

!!! abstract "TL;DR"
    One AND-rule can describe only one kind of patient. A **Tsetlin machine** therefore uses a whole team of rules: some vote **for** a class, some vote **against** it. Each rule whose condition is True casts one vote, and the **vote sum** $v$ (votes for minus votes against) decides the class. During training, a single number, the **Voting Margin** $T$, decides how much feedback each example hands out: examples the team already gets right by a margin of $T$ votes give no feedback at all, and examples it gets wrong give a lot. That simple rule makes the rules spread out over different sub-patterns instead of all learning the same one. This lesson builds the complete machine, extends it to many classes, and runs it end to end on the breast-cancer recurrence data, with an honest look at the accuracy.

**Level:** beginner–intermediate · **Prerequisites:** [Recognize, Erase & Reject feedback](../tsetlin-machine-feedback/index.md) (how a single rule learns). Fractions and a sum are all the math you need.

---

## Why should I care?

In the last lesson a single rule learned a single pattern, such as *"if Deg-malig 3 and not Menopause lt40 then Recurrence"*. Real classes are rarely that tidy:

- **Patients recur for different reasons.** In the breast-cancer data, some recurrences come with a highly malignant tumour, others with many affected lymph nodes, and a rule demanding *both* misses everyone who has only one of them.
- **Evidence should add up.** A clinician weighs several findings, some pointing towards recurrence and some away from it. A sum of votes does the same thing, and you can still read every vote as a plain rule.
- **This is the whole machine.** Everything from the earlier lessons (literals, the graded memory, the three kinds of feedback) comes together here. After this lesson you can write a working Tsetlin machine in about 70 lines of numpy and read what it learned.

The ideas follow Granmo, *An Introduction to Tsetlin Machines*, Ch. 1, §1.5–1.7, and Ch. 2, §2.3. The visuals and code are redrawn and rewritten for this site.

---

## Step 1 — One rule is not enough

A rule in a Tsetlin machine is an **AND** of literals: *every* memorized literal must be True for the rule to fire. That makes each rule a description of **one kind** of object. But a class often contains several kinds.

Look at the 85 patients with a recurrence in the UCI breast-cancer data (286 patients in total; we count "3 or more involved lymph nodes" as *not Inv-nodes 0-2*):

| Recurrence patients (85) | how many |
|---|---|
| Deg-malig 3 **and** 3+ involved nodes | 27 |
| Deg-malig 3 only | 18 |
| 3+ involved nodes only | 12 |
| neither | 28 |

And how often patients with each profile recur, out of everyone with that profile:

| Profile | recurred |
|---|---|
| Deg-malig 3 and 3+ nodes | 27 of 37 (73%) |
| Deg-malig 3 only | 18 of 48 (38%) |
| 3+ nodes only | 12 of 36 (33%) |
| neither | 28 of 165 (17%) |

There are (at least) **two recurrence profiles**: a highly malignant tumour, and spread to the lymph nodes. The single rule *"Deg-malig 3 AND not Inv-nodes 0-2"* only catches the 27 patients who have both. To also catch the 18 + 12 who have just one of them, you need an **OR**: *"Deg-malig 3"* **or** *"not Inv-nodes 0-2"*. An AND-rule cannot express an OR, but **two rules** can: one for each profile.

This is the general recipe. Any yes/no pattern can be written as an OR of AND-rules (logicians call it *disjunctive normal form*), so a team of AND-rules can describe any class, as long as the team is big enough and its members **learn different things**. Making them learn different things is the main topic of this lesson.

!!! tip "What to notice"
    - One AND-rule = one sub-pattern (one "profile"). Several sub-patterns need several rules.
    - None of the profiles is certain: even the strongest one recurs in only 73% of cases, and 28 of the 85 recurrences fit neither profile. Keep this in mind when we measure accuracy in Step 6.

---

## Step 2 — Rules vote: the vote sum v

A Tsetlin machine for two classes has two groups of rules:

- rules that vote **for** the class (here: *Recurrence*),
- rules that vote **against** it (they vote for *No recurrence*).

To classify a patient, every rule checks its condition on the patient's literals. Each rule whose condition is True casts **one vote**. Then

$$
v \;=\; \underbrace{\#\{\text{rules for Recurrence that fire}\}}_{\text{votes for}} \;-\; \underbrace{\#\{\text{rules against that fire}\}}_{\text{votes against}}
$$

and the machine predicts *Recurrence* if $v > 0$ and *No recurrence* otherwise. A tie ($v = 0$) has to go one way or the other; here it goes to the more common class, *No recurrence* (the book's notebook sends ties to the first class instead, and the [convolutional lesson](../convolutional-tsetlin-machine/index.md) follows the notebook). (In the research papers the two groups are the *positive* and *negative polarity clauses* $C^+_j$ and $C^-_j$, and $v = \sum_j C^+_j - \sum_j C^-_j$.) An empty rule, one with no literals memorized, stays silent when classifying.

The book's first example uses the six patients from the earlier lessons and three rules: R1 and R2 are the *same* rule, *"Deg-malig 3 and not Menopause lt40"*, voting for Recurrence, and R3, *"Inv-nodes 0-2"*, votes against. Pick the book's patients in the widget below:

<div id="viz-vote" class="viz"><div class="viz-title">🗳️ Interactive: rules vote on a patient</div></div>

!!! tip "What to notice"
    - **Patient #1** (ge40, 3-5 nodes, Deg-malig 3): R1 and R2 fire, R3 does not, so $v = +2$ → Recurrence. Correct.
    - **Patient #5** (premeno, 0-2 nodes, Deg-malig 3) fires **all three** rules: R3 votes against because of the 0-2 nodes. The duplicate R1 + R2 still win, $v = 2 - 1 = +1$. A lone rule would have been out-voted to a tie. Duplicated rules act as a **weight**.
    - The vote sums for the six patients are $+2, -1, +2, -1, +1, -1$: all six correct.
    - Switch to **"learned from 200 patients"** to see the 20 rules the machine learns in Step 6, written in plain English. Hover a rule to see its raw literals.

This is what the book means by unifying **rule-based** and **summation-based** decisions: each vote is a readable rule, like a branch of a decision tree, and the decision is a sum, like logistic regression with all weights equal to 1.

---

## Step 3 — The Voting Margin T: how much feedback does an example get?

Training has to decide, for every example, **which rules learn from it**. A naive answer is "all of them, every time". Then every rule for Recurrence sees the same stream of examples and the same feedback, and they all drift towards the same, most frequent pattern. You get ten copies of one rule.

The Tsetlin machine instead asks: *how far is this example from being safely classified?* You choose a **Voting Margin** $T$, a whole number of votes. For a training example:

1. Compute the vote sum **from the example's own point of view**: votes for its true class minus votes for the other class. (Empty rules count as True during training, so a fresh machine has something to work on.)
2. Clip it to the range $[-T, T]$.
3. Give **each rule**, independently, feedback with probability

$$
p \;=\; \frac{T - v}{2T}.
$$

The book's example uses $T = 2$, so $p = (2 - v)/4$:

| vote sum $v$ (own class − other) | $-2$ or less | $-1$ | $0$ | $+1$ | $+2$ or more |
|---|---|---|---|---|---|
| feedback probability $p$ | 1 | 0.75 | 0.5 | 0.25 | 0 |

If the team already wins by $T$ votes, nobody learns from this example: the job is done, and those rules are free to work on other examples. If the team is losing badly, every rule gets feedback.

The **kind** of feedback is the same as in the previous lesson. A rule of the example's own class gets **Recognize Feedback** if it fired and **Erase Feedback** if it did not. A rule of the other class gets **Reject Feedback** if it fired (it voted wrongly) and nothing otherwise.

<div id="viz-margin" class="viz"><div class="viz-title">🎚️ Interactive: how the Voting Margin hands out feedback</div></div>

!!! tip "What to notice"
    - **Two mirror-image lines.** The chart's $v$ counts votes *for Recurrence*. For a patient with recurrence that is "own minus other"; for a patient without, it is the reverse. So the two kinds of patients want $v$ pushed in opposite directions.
    - **Outside the purple band nothing changes.** Once $v \ge T$ for a recurrence patient (or $v \le -T$ for a non-recurrence patient), the probability is 0. Raising $T$ widens the band: the team must win by more votes before it stops learning.
    - **The book's prioritisation example** ($T = 2$, rules *"Deg-malig 3 and not lt40"* and *"Deg-malig 3"* for, *"Inv-nodes 0-2"* against): patients #1 and #3 reach $v = +2$ and give no feedback; #4, #5 and #6 are one vote short and give feedback a quarter of the time; patient #2 (no recurrence, but both rules for Recurrence fire) ends in a tie, $v = 0$, and gives feedback half of the time. The examples that need it most get the most attention.

---

## Step 4 — The complete learning algorithm

Put the pieces together and you have the whole training procedure. Every rule starts **empty**, with all its literals at the top of the Forgotten half of its memory. Then, for each training example in turn:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 One learning step of a Tsetlin machine",
  "cols": 3,
  "rows": 2,
  "rowHeight": 120,
  "nodes": [
    {"id": "obs", "col": 0, "row": 0, "icon": "🧾", "label": "Observe an example and its class", "sub": "literals x<sub>k</sub> and ¬x<sub>k</sub>", "role": "data", "note": "The example's Boolean features and their negations, plus its true class."},
    {"id": "eval", "col": 1, "row": 0, "icon": "🧩", "label": "Evaluate every rule", "sub": "True if all its memorized literals are True", "role": "model", "note": "While training, an empty rule counts as True."},
    {"id": "vote", "col": 2, "row": 0, "icon": "🗳️", "label": "Vote sum v, clipped to [−T, T]", "sub": "own class − other class", "role": "loss", "note": "Votes from firing rules of the example's own class, minus votes from firing rules of the other class."},
    {"id": "prob", "col": 2, "row": 1, "icon": "🎲", "label": "Pick each rule with probability (T − v) / 2T", "role": "loss", "note": "Independent coin flip per rule. v = T: nobody is picked. v = −T: everybody is."},
    {"id": "own", "col": 1, "row": 1, "icon": "✅", "label": "Own-class rules: Recognize or Erase", "sub": "Recognize if it fired, Erase if not", "role": "opt", "note": "Recognize: memorize True literals (Memorize Value), forget False ones (Forget Value). Erase: forget all literals (Forget Value)."},
    {"id": "rej", "col": 0, "row": 1, "icon": "⛔", "label": "Other-class rules that fired: Reject", "sub": "memorize Forgotten False literals", "role": "grad", "note": "Pushes a literal that is False for this example into the rule, so the rule stops firing on it."}
  ],
  "edges": [
    {"from": "obs", "to": "eval"},
    {"from": "eval", "to": "vote"},
    {"from": "vote", "to": "prob"},
    {"from": "prob", "to": "own"},
    {"from": "own", "to": "rej", "label": "and"},
    {"from": "rej", "to": "obs", "label": "next"}
  ],
  "steps": [
    {"show": ["obs"], "text": "Take the next training example: its literals (each feature and its negation) and its true class."},
    {"show": ["obs>eval", "eval"], "text": "Every rule checks its condition. While training, an empty rule counts as True, so fresh rules have something to learn from."},
    {"show": ["eval>vote", "vote"], "text": "Add the votes of firing rules of the example's own class, subtract those of the other class, and clip the result to [−T, T]."},
    {"show": ["vote>prob", "prob"], "text": "Each rule independently gets feedback with probability <b>(T − v) / 2T</b>: zero when the example already wins by T votes, one when it loses by T."},
    {"show": ["prob>own", "own"], "text": "Picked rules of the example's own class get <b>Recognize Feedback</b> if they fired, <b>Erase Feedback</b> if they did not."},
    {"show": ["own>rej", "rej"], "text": "Picked rules of the other class that fired voted wrongly: they get <b>Reject Feedback</b>. Picked rules of the other class that did not fire are left alone."},
    {"show": ["rej>obs", "obs"], "text": "On to the next example. One pass through the training set is an <b>epoch</b>; repeat for many epochs."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 120, "pos": {"obs": [0, 0], "eval": [1, 0], "vote": [1, 1], "prob": [1, 2], "own": [0, 2], "rej": [0, 1]}}
}
</script>
</div>

That is the entire algorithm. There is no gradient, no loss function and no global optimiser. Each rule only ever sees its own condition, the example, and one coin flip. The coordination between rules comes **entirely** from the shared vote sum inside $p$.

!!! note "Why it coordinates the rules (the book's argument)"
    - **Reject Feedback is targeted.** Only rules that fired *and* voted for the wrong class receive it. Rules that voted correctly are shielded.
    - **Useful rules learn fastest.** A rule that matches many examples of its class and never votes wrongly gets the most Recognize Feedback and no Reject Feedback, so its literals sink deepest into memory.
    - **Satisfied examples stop pulling.** Once a sub-pattern is covered by enough votes, its examples hand out no more feedback. The remaining feedback comes from examples that are *not* yet covered, so idle or weak rules drift towards those. The book calls this a **resource-allocation** effect.

---

## Step 5 — Why the margin matters: two rules, one OR pattern

Chapter 2 of the book makes this precise with the smallest interesting case. The class is $Y = x_1 \text{ OR } x_2$ (with 10% of the labels flipped as noise). There are **two rules**, both voting for $Y$, and each rule can memorize $x_1$, $x_2$, both, or neither. Of the 16 possible pairs of rules, only two solve the problem: one rule *"if $x_1$ then Y"* and the other *"if $x_2$ then Y"*.

A tempting wrong answer is *"if $x_1$ AND $x_2$ then Y"*: it is very precise when it fires, but it misses every example where only one of them is True. Two copies of it are a **sub-optimal configuration** (a *local optimum*): no single rule can improve things by changing alone.

In this game there are no rules voting against $Y$, so $v$ is just the number of firing rules. With margin $T$, a $Y$ example gives each rule feedback with probability $(T - v)/T$, and a not-$Y$ example with probability $v/T$ (with $v$ capped at $T$). With $T = 2$: a $Y$ example with one rule firing gives feedback half of the time; with both firing, never.

<div id="viz-orgame" class="viz"><div class="viz-title">🤝 Interactive: two rules learning x<sub>1</sub> OR x<sub>2</sub></div></div>

!!! tip "What to notice"
    - **Without a margin** (every rule gets feedback every time), fast-forward to 200,000 steps: the pair spends **60.7%** of the time as two copies of *"x1 AND x2"*, and only **7.1%** in the correct OR solution. Both rules chase the same, most precise pattern.
    - **With $T = 2$** the picture flips: **81.4%** of the time in the OR solution, and only **0.3%** as two AND-copies. As soon as one rule covers $x_1$, examples it handles no longer push the other rule, which is left to pick up $x_2$.
    - **$T$ must fit the team.** With $T = 1$ (one vote is "enough") the second rule rarely gets feedback and only **28.6%** of the time is spent in the solution. With $T = 3$ or $T = 4$ the team can never reach the margin (two rules give at most two votes), so feedback never fully stops: **26.4%** and **18.1%**, and with $T = 4$ the two AND-copies are back at **25.4%**.
    - These are time shares from one seeded run (seed 42, Forget Value $1/s$ with $s = 4$, Memorize Value 1.0, memory depth 4), computed in Python and reproduced exactly by the widget. The book computes the exact long-run probabilities with a Markov chain instead and finds the same story: without the margin the AND-copies are the most likely outcome; with $T = 2$ any configuration involving them has essentially zero probability.

The lesson generalises: **the Voting Margin should be roughly the number of votes you expect a well-covered example to collect**. Too small, and rules stop learning before every sub-pattern has a rule. Too large, and feedback never switches off, so rules keep crowding onto the most common pattern.

---

## Step 6 — The full run: breast-cancer recurrence

Now the real thing. The data is the public **UCI Breast Cancer** dataset (University Medical Centre, Ljubljana, donated by M. Zwitter and M. Soklic): 286 patients, 85 of whom (29.7%) had a recurrence. Like the book's chapter, we use three properties, **Menopause**, **Inv-nodes** and **Deg-malig**, and Booleanize them with one feature per value: 3 + 7 + 3 = **13 Boolean features**, so **26 literals**.

The setup, fixed in advance:

- shuffle the patients with seed 1; **200 for training, 86 for testing** (23 of the test patients had a recurrence);
- **10 rules for** and **10 rules against** Recurrence, Voting Margin $T = 10$;
- Forget Value $1/s$ with $s = 3$ (so Memorize Value $2/3$);
- memory depth 200 (positions 1–200; Memorized above 100). The book's toy examples use depth 10; a deeper memory makes the rules steadier on noisy data;
- 60 epochs; a tie ($v = 0$) is predicted as *No recurrence*.

The widget trains exactly this machine in your browser. Change the knobs and it retrains in a second or two.

<div id="viz-train" class="viz"><div class="viz-title">🏋️ Interactive: train a Tsetlin machine on breast-cancer data</div></div>

!!! tip "What to notice"
    - **With the defaults** the machine ends at **75.0%** training and **74.4%** test accuracy (64 of 86 test patients right). That sounds fine until you see the dashed line: always answering *"no recurrence"* scores **73.3%** on the same test set. The machine is one patient better than the trivial answer.
    - **It catches some recurrences.** It flags 15 test patients, 8 of them correctly, so it finds 8 of the 23 recurrences, with 7 false alarms among the 63 others. The trivial answer finds none.
    - **The test curve jumps around.** Between epochs 11 and 60 the test accuracy moves between 68.6% and 76.7%. With 86 test patients, one patient is 1.2 percentage points, so a single lucky or unlucky epoch says little.
    - **The rules for Recurrence spread out.** Look at the learned rules: one broad rule, *"Inv-nodes is not 0-2"*, matches 29 recurrence and 26 non-recurrence training patients; the rest specialise on small groups (6-8, 12-14, 15-17 or 24-26 nodes, or Menopause lt40). That is the resource allocation of Step 4 at work: once the common cases are covered, idle rules move to patients nobody has covered yet. On noisy data that also means **chasing rare individuals**: four identical rules memorized *"Inv-nodes 24-26"*, which matches a single training patient.
    - **Try $T = 1$ or $T = 2$, or 1 rule per class.** With one rule per class and $T = 1$ the machine never predicts Recurrence on this split. Change the seed to get a different train/test split and watch the accuracy move by several points.

### How good is that, honestly?

One split is not enough to judge a model on 86 test patients. So we repeated the run on **10 different random splits** (seeds 1–10, same settings) and compared with simple baselines on the same splits and the same 13 features:

| Method (10 random 200/86 splits) | test accuracy, mean ± sd |
|---|---|
| Always "no recurrence" | 70.7% ± 3.1 |
| The book's single rule *"Deg-malig 3 and not Menopause lt40"* | 71.2% ± 3.8 |
| **Tsetlin machine** (10 + 10 rules, $T = 10$, $s = 3$) | **71.3% ± 3.7** (range 64.0–75.6%) |
| Logistic regression (scikit-learn defaults) | 72.7% ± 3.5 |
| Decision tree, depth 3 | 72.9% ± 3.7 |

All of these are within about one standard deviation of each other and of the trivial baseline. Using all nine properties of the dataset instead of three gave the same Tsetlin-machine accuracy (71.3% ± 2.9). Sweeping the margin over the same 10 splits: $T = 1$ and $T = 2$ were worse (69.1% each, with larger spread), and $T = 5$, 10, 20, 50 and 10000 all landed between 70.8% and 72.4%.

!!! warning "What this tells you about the data, not just the model"
    These three properties (and even all nine) simply do not contain enough information to predict recurrence well: remember from Step 1 that 28 of the 85 recurrences fit neither risk profile. No method in the table escapes that. What the Tsetlin machine adds is not accuracy but **transparency**: every vote it casts is a rule you can read and check against clinical knowledge. With a test set of 86 patients, a difference of a few points between methods, or between one split and another, is noise. Lesson 1 of this series reports 76.7% against a 70.9% baseline on a different split (seed 42) with a different set of 11 features; our 10-split range of 64.0–75.6% shows how much a single split can move.

---

## Step 7 — Reading the model as plain English

A learned rule is a list of memorized literals, for example (from the default run above):

> NOT Inv-nodes 6-8 AND NOT Inv-nodes 24-26 AND NOT Deg-malig 3 → *against* Recurrence

Literals that mention the same property can be merged, the way the book simplifies *"Deg-malig 3 and not Deg-malig 1 and not Deg-malig 2"* to *"Deg-malig 3"*:

> **IF** Inv-nodes is not 6-8 or 24-26 **AND** Deg-malig is not 3 **then** vote against Recurrence.

This rule matches 108 of the 138 non-recurrence training patients and 25 of the 62 recurrence patients. The widgets in Step 2 and Step 6 do this merging automatically. A few habits make a learned model easy to read:

- **Count duplicates.** Rules that say the same thing are one rule with a bigger weight: +4 means four votes. (Two rules can list slightly different literals and still be equivalent, such as *"Deg-malig 2"* and *"Deg-malig 2 AND NOT Deg-malig 1"*; the widgets merge them.)
- **Check each rule's coverage**, the number of training patients of each class it matches. A rule matching 1 patient is an anecdote, not a pattern.
- **Spot rules that can never fire.** Reject Feedback sometimes leaves a rule demanding two values of the same property at once (*"Inv-nodes 9-11 AND Inv-nodes 12-14"*). It is harmless dead weight.
- **Read the vote for one patient.** In the Step 2 widget, pick book patient #5 with the learned rules: no rule fires on either side, $v = 0$, so it is predicted as *No recurrence* even though it recurred. You can see exactly *why* the model got it wrong: no learned rule describes "premeno, 0-2 nodes, Deg-malig 3".

!!! note "Interpretable is not the same as correct"
    Readable rules let you **audit** a model: you can see that four rules memorized a single patient with 24-26 nodes, and decide that this is overfitting. They do not make the model right. The breast-cancer machine above is honest about its uncertainty only because you can look inside it.

---

## Step 8 — More than two classes

So far one team of rules decided between two classes. For $m$ classes (say Car, Plane and Boat), the standard Tsetlin machine (Granmo, 2018) gives **every class its own team**, each with rules voting for and against *that* class. Each team computes its own vote sum $v_i$, and the class with the **highest vote sum wins**.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🏁 A multi-class Tsetlin machine: one team per class",
  "cols": 3,
  "rows": 3,
  "rowHeight": 100,
  "nodes": [
    {"id": "in", "col": 1, "row": 0, "icon": "🧾", "label": "Object's literals", "role": "data", "note": "The same literals go to every team."},
    {"id": "car", "col": 0, "row": 1, "icon": "🚗", "label": "Car team", "sub": "v<sub>Car</sub> = for − against", "role": "model", "note": "Rules voting for Car and rules voting against Car."},
    {"id": "plane", "col": 1, "row": 1, "icon": "✈️", "label": "Plane team", "sub": "v<sub>Plane</sub>", "role": "model", "note": "Rules voting for Plane and rules voting against Plane."},
    {"id": "boat", "col": 2, "row": 1, "icon": "🚢", "label": "Boat team", "sub": "v<sub>Boat</sub>", "role": "model", "note": "Rules voting for Boat and rules voting against Boat."},
    {"id": "max", "col": 1, "row": 2, "icon": "🏆", "label": "Highest v wins", "shape": "pill", "role": "loss", "note": "The predicted class is the one whose team has the largest vote sum."}
  ],
  "edges": [
    {"from": "in", "to": "car"},
    {"from": "in", "to": "plane"},
    {"from": "in", "to": "boat"},
    {"from": "car", "to": "max"},
    {"from": "plane", "to": "max"},
    {"from": "boat", "to": "max"}
  ],
  "steps": [
    {"show": ["in"], "text": "Booleanize the object as before: one literal per feature and one per negated feature."},
    {"show": ["in>car", "in>plane", "in>boat", "car", "plane", "boat"], "text": "Every team evaluates its own rules on the same literals and computes its own vote sum: votes for its class minus votes against it."},
    {"show": ["car>max", "plane>max", "boat>max", "max"], "text": "The class whose team has the <b>highest vote sum</b> is the prediction."},
    {"show": ["car", "plane"], "text": "Training on a Car: the <b>Car team</b> learns it as an example of its class (aim: v<sub>Car</sub> ≥ T). <b>One randomly chosen other team</b>, here Plane, learns it as an example of 'not my class' (aim: v<sub>Plane</sub> ≤ −T). Both use the same feedback probability rule as before."}
  ],
  "narrow": {"cols": 3, "rows": 3, "rowHeight": 90, "hideSub": true}
}
</script>
</div>

Training each example updates two teams: its **own team**, exactly as in Step 4 with the example counted as "own class", and **one other team chosen at random**, with the example counted as "other class" for that team. Updating just one other team, rather than all of them, keeps the cost the same however many classes there are. The two-class machine of this lesson is the special case with a single team: rules for Recurrence and rules against it.

!!! tip "What to notice"
    - Each team is a two-class Tsetlin machine of its own: "this class" against "everything else".
    - The output is still readable: for any prediction you can list the firing rules in the winning team and in its closest rival.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧠 The complete Tsetlin machine, from record to prediction",
  "cols": 3,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "raw", "col": 0, "row": 0, "icon": "🧾", "label": "Patient record", "sub": "ge40 · 3-5 nodes · Deg-malig 3", "role": "data"},
    {"id": "bool", "col": 1, "row": 0, "icon": "🔢", "label": "Booleanize", "sub": "13 features → 26 literals", "role": "data", "note": "One feature per value, plus the negation of every feature."},
    {"id": "rules", "col": 2, "row": 0, "icon": "🧩", "label": "20 rules check their conditions", "sub": "10 for, 10 against", "role": "model", "note": "Each rule is an AND of the literals in the Memorized half of its memory."},
    {"id": "vote", "col": 2, "row": 1, "icon": "🗳️", "label": "Vote sum v = for − against", "role": "loss"},
    {"id": "out", "col": 1, "row": 1, "icon": "🏷️", "label": "Recurrence if v > 0", "shape": "pill", "role": "opt"},
    {"id": "learn", "col": 0, "row": 1, "icon": "🎲", "label": "Training: feedback", "sub": "probability (T − v) / 2T", "role": "grad", "note": "Recognize / Erase for rules of the true class, Reject for wrongly firing rules of the other class."}
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
    {"show": ["raw"], "text": "Start from a patient record with multi-valued properties."},
    {"show": ["raw>bool", "bool"], "text": "Booleanize: one True/False feature per value, and add the negation of each feature. These are the literals."},
    {"show": ["bool>rules", "rules"], "text": "Each rule fires if all its memorized literals are True (Lesson 1 and Lesson 2: the graded memory)."},
    {"show": ["rules>vote", "vote"], "text": "Firing rules vote: +1 for Recurrence or −1 against. Their sum is the vote sum v."},
    {"show": ["vote>out", "out"], "text": "Classification is just the sign of v. That is all a trained machine does."},
    {"show": ["out>learn", "learn"], "text": "During training, compare with the true label and pick rules for feedback with probability (T − v)/2T: Recognize, Erase or Reject (Lesson 3)."},
    {"show": ["learn>raw", "raw"], "text": "Move on to the next patient, for many epochs. The Voting Margin makes the rules divide the patients between them."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 115, "pos": {"raw": [0, 0], "bool": [1, 0], "rules": [1, 1], "vote": [1, 2], "out": [0, 2], "learn": [0, 1]}}
}
</script>
</div>

Here is the complete two-class machine in numpy, about 70 lines, trained on the same three properties. It follows the algorithm of Step 4 line by line, but vectorised over rules and literals. It uses numpy's random numbers and its own train/test shuffle, so its numbers differ a little from the widget's.

```python
from collections import Counter

import numpy as np

# ---- 1. Booleanize: one feature per value of Menopause, Inv-nodes and Deg-malig (UCI Breast Cancer data)
VALUES = {"menopause": (1, ["lt40", "ge40", "premeno"]),             # (column after the label, values)
          "inv-nodes": (3, ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "24-26"]),
          "deg-malig": (5, ["1", "2", "3"])}
rows = [line.strip().split(",") for line in open("data/breast-cancer.data") if line.strip()]
X = np.array([[r[c + 1] == v for c, vals in VALUES.values() for v in vals] for r in rows], dtype=np.int8)
y = np.array([r[0] == "recurrence-events" for r in rows], dtype=np.int8)
names = [f"{p} {v}" for p, (c, vals) in VALUES.items() for v in vals]


class TsetlinMachine:
    """Two classes. memory[c, j, k] = position (1 … depth) of literal k in rule j of class c."""

    def __init__(self, n_features, rules_per_class=10, T=10, s=3.0, depth=200, seed=1):
        self.T, self.s, self.depth, self.half = T, s, depth, depth // 2
        self.rng = np.random.default_rng(seed)
        self.memory = np.full((2, rules_per_class, 2 * n_features), depth // 2)  # top of Forgotten

    def fires(self, lits, training=False):
        """Which rules fire? A rule fires when every memorized literal is True."""
        memorized = self.memory > self.half                        # (2, rules, literals)
        ok = ~np.any(memorized & (lits == 0), axis=2)
        if not training:
            ok &= memorized.any(axis=2)                            # empty rules stay silent when classifying
        return ok

    def predict(self, x):
        f = self.fires(np.concatenate([x, 1 - x]))
        return int(f[1].sum() - f[0].sum() > 0)                    # vote sum v > 0 → class 1

    def update(self, x, y):
        lits = np.concatenate([x, 1 - x])                          # literals: x_k, then NOT x_k
        f = self.fires(lits, training=True)
        v = f[y].sum() - f[1 - y].sum()                            # votes for own class − votes against
        v = np.clip(v, -self.T, self.T)
        p = (self.T - v) / (2 * self.T)                            # Voting Margin: feedback probability
        _, n_rules, n_lits = self.memory.shape
        chosen = self.rng.random((2, n_rules)) <= p                # which rules get feedback this time
        u = self.rng.random((n_rules, n_lits))
        own, other = self.memory[y], self.memory[1 - y]            # views: changes go into self.memory

        # Recognize Feedback (own-class rules that fire): memorize True literals, forget False ones
        rec = (chosen[y] & f[y])[:, None]
        own += rec & (lits == 1) & (u <= (self.s - 1) / self.s) & (own < self.depth)
        own -= rec & (lits == 0) & (u <= 1 / self.s) & (own > 1)
        # Erase Feedback (own-class rules that don't fire): forget every literal
        era = (chosen[y] & ~f[y])[:, None]
        own -= era & (u <= 1 / self.s) & (own > 1)
        # Reject Feedback (other-class rules that fire): memorize Forgotten False literals
        rej = (chosen[1 - y] & f[1 - y])[:, None]
        other += rej & (lits == 0) & (other <= self.half)

    def fit(self, X, y, epochs):
        for _ in range(epochs):
            for i in self.rng.permutation(len(X)):
                self.update(X[i], y[i])
        return self

    def rules(self, c, names):
        lits = names + ["NOT " + n for n in names]
        return [" AND ".join(l for l, m in zip(lits, rule) if m > self.half) or "(empty)" for rule in self.memory[c]]


order = np.random.default_rng(0).permutation(len(X))              # 200 patients to train on, 86 to test
train, test = order[:200], order[200:]
tm = TsetlinMachine(X.shape[1], rules_per_class=10, T=10, s=3.0, depth=200, seed=1).fit(X[train], y[train], epochs=60)
acc = lambda idx: np.mean([tm.predict(X[i]) == y[i] for i in idx])
print(f"train accuracy {acc(train):.3f} · test accuracy {acc(test):.3f} · always 'no recurrence' on test {1 - y[test].mean():.3f}")
for c, label in ((1, "Recurrence"), (0, "No recurrence")):
    for rule, count in Counter(tm.rules(c, names)).most_common(2):
        print(f"{count} × IF {rule} THEN {label}")
```

```text
train accuracy 0.775 · test accuracy 0.744 · always 'no recurrence' on test 0.698
5 × IF inv-nodes 24-26 THEN Recurrence
2 × IF inv-nodes 9-11 THEN Recurrence
3 × IF inv-nodes 0-2 AND NOT menopause lt40 AND NOT inv-nodes 3-5 AND NOT inv-nodes 6-8 AND NOT inv-nodes 9-11 AND NOT inv-nodes 12-14 AND NOT inv-nodes 15-17 AND NOT inv-nodes 24-26 AND NOT deg-malig 3 THEN No recurrence
1 × IF inv-nodes 0-2 AND NOT menopause lt40 AND NOT inv-nodes 6-8 AND NOT inv-nodes 9-11 AND NOT inv-nodes 12-14 AND NOT deg-malig 1 AND NOT deg-malig 2 THEN No recurrence
```

Download `breast-cancer.data` from the [UCI repository](https://archive.ics.uci.edu/dataset/14/breast+cancer) first. On this split the machine beats the trivial answer by more (74.4% vs 69.8%), and again five rules spent themselves on the only patient in the whole dataset with 24-26 nodes. The first No-recurrence rule simplifies to *"Inv-nodes 0-2 and Menopause is not lt40 and Deg-malig is 1 or 2"*.

| Piece | What it does | Book term (research term) |
|---|---|---|
| Rule | AND of memorized literals; votes when True | rule (clause $C_j$) |
| Rules for / against | +1 / −1 votes for the class | (positive / negative polarity clauses) |
| Vote sum $v$ | votes for − votes against; the sign is the prediction | vote sum ($\sum C^+_j - \sum C^-_j$) |
| Voting Margin $T$ | feedback probability $(T - v)/2T$ during training | Voting Margin (threshold / target $T$) |
| Recognize, Erase, Reject | how a picked rule changes its memory | (Type Ia, Type Ib, Type II feedback) |
| One team per class | multi-class: the highest vote sum wins | (multi-class Tsetlin machine) |

---

## Common misconceptions

!!! warning "\"The Voting Margin is a decision threshold.\""
    No. Classification only looks at the **sign** of $v$. The Voting Margin $T$ is used **only during training**, to decide how likely each rule is to get feedback. A trained machine never looks at $T$.

!!! warning "\"More feedback is better, so always give every rule feedback.\""
    In the OR game, feedback on every step left the two rules as copies of *"x1 AND x2"* for 60.7% of the time, and in the correct OR solution only 7.1% of the time. With $T = 2$ the solution held 81.4% of the time. Switching feedback **off** for examples that are already safe is what lets rules specialise.

!!! warning "\"Rules know about each other.\""
    They don't. Each rule only sees its own condition, the example and one random number. The only shared quantity is the vote sum $v$, which sets the feedback probability for everyone. Coordination emerges from that single number.

!!! warning "\"74% test accuracy means the model works.\""
    On this dataset, answering "no recurrence" for everyone already scores 70.7% on average over 10 splits, and the Tsetlin machine averaged 71.3%. Always compare with the trivial baseline, use more than one split, and look at what the model gets right (here: 8 of 23 recurrences on the default split).

!!! warning "\"Duplicate rules are wasted.\""
    Partly true and partly not. Duplicates act as **weights**: in the book's example, the duplicated R1 + R2 out-vote R3 for patient #5. But four copies of a rule that matches a single patient is wasted capacity, and a sign of overfitting.

---

## Check yourself

??? question "1. With T = 2, three rules for Recurrence fire and one rule against fires on a patient with recurrence. What is the feedback probability?"
    Own class minus other: $v = 3 - 1 = 2$, which is already at the margin. $p = (2 - 2)/4 = $ **0**: no rule learns from this patient.

??? question "2. Same rules, but the patient did NOT have a recurrence. What happens?"
    From the patient's point of view $v = 1 - 3 = -2$, so $p = (2 - (-2))/4 = $ **1**: every rule is picked. The rules against Recurrence get Recognize (the one that fired) or Erase (the others). The three rules for Recurrence that fired voted wrongly and get **Reject Feedback**, which pushes a False literal of this patient into each of them.

??? question "3. Why does a Tsetlin machine predict with the sign of v, but train with (T − v)/2T?"
    Prediction only needs to know which side wins. Training also wants a **safety margin**: an example won by a single vote is fragile, so it keeps asking for feedback until it wins by $T$ votes. And examples won by $T$ votes or more stop asking, which frees the rules to cover other sub-patterns.

??? question "4. You have 4 rules for a class whose examples come in 4 clearly different sub-patterns. Is T = 1 or T = 20 the better choice?"
    Neither. With $T = 1$ one firing rule is enough, so a second sub-pattern covered by the "wrong" rule stops the learning early, and there is little pressure to fill every niche. With $T = 20$ the margin can never be reached (4 rules give at most 4 votes), so feedback never switches off and the rules crowd onto the most frequent sub-pattern. A margin around the number of votes a well-covered example can collect (here a few) works best, as the OR game showed with $T = 2$ for two rules.

??? question "5. The learned model predicts 'No recurrence' for book patient #5 (premeno, 0-2 nodes, Deg-malig 3), who did recur. How would you find out why?"
    Read the votes. In the Step 2 widget with the learned rules, **no rule fires on either side**, so $v = 0$ and the tie goes to No recurrence. No learned rule describes this profile, and the data explains why: Deg-malig 3 without lymph-node spread recurred in only 38% of patients. A black-box model would give you the same wrong answer without the explanation.

---

## Further reading

- Ole-Christoffer Granmo, [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/) (free online), Ch. 1, §1.5–1.7 (voting, the Voting Margin and the complete algorithm) and Ch. 2, §2.3 (rule interaction and the OR example with its Markov-chain analysis). This lesson follows those sections.
- Ole-Christoffer Granmo, [*The Tsetlin Machine – A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic*](https://arxiv.org/abs/1804.01508) (2018). The original paper, including the multi-class machine and the clause/threshold terminology.
- M. Zwitter and M. Soklic, [Breast Cancer dataset](https://archive.ics.uci.edu/dataset/14/breast+cancer), UCI Machine Learning Repository. The data used in Step 6.
- [The Tsetlin automaton](../tsetlin-automaton/index.md) and [Recognize, Erase & Reject feedback](../tsetlin-machine-feedback/index.md): the memory and the feedback this lesson builds on.

**Next lessons:** [Convolutional Tsetlin machines](../convolutional-tsetlin-machine/index.md)

<script src="viz/tsetlin-machine-classification.js"></script>
