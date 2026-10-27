# What Is a Tsetlin Machine?

!!! abstract "TL;DR"
    A **Tsetlin machine** is a machine-learning model built from plain **if-then rules**. Its input is a list of yes/no facts (**Boolean features**). Each rule joins a few of those facts, or their opposites, with AND: "**if** Four Wheels **and not** Wings **then** Car". Many such rules vote, and the class with the most votes wins. Because the rules are ordinary logic, you can read every one of them, and because they only use bits, AND and counting, they can run very cheaply in hardware. This lesson shows what the rules look like and how data is turned into bits. How the rules are *learned* comes in the next lessons.

**Level:** beginner · **Prerequisites:** none. If you have read [What is reinforcement learning?](../reinforcement-learning-intro/index.md), you will spot a familiar idea in the next lesson, but it is not needed here.

This series follows Ole-Christoffer Granmo's free book [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/). This first lesson covers the start of Chapter 1 (the overview, §1.1 and §1.2).

---

## Why should I care?

Most of today's strong models, deep neural networks above all, are **black boxes**: millions of numbers multiplied together, with no human-readable reason behind each answer. In medicine that is a real problem. If a model says "high risk of recurrence", a clinician wants to know *why*, and a regulator wants to know whether the reason is sound.

A Tsetlin machine takes a different road. Its whole "knowledge" is a set of rules such as

> **if** malignancy grade 3 **and** tumour through node capsule **and** premenopausal **then** vote *Recurrence*

which a person can read, check and argue with. Granmo's book (Ch. 1, box "Tsetlin Machines vs Black Boxes") sums up what the approach aims for:

| The Tsetlin machine is… | …like | What that means |
|---|---|---|
| a universal function approximator | neural networks | with enough rules it can represent any pattern in Boolean data |
| rule-based | decision trees | its knowledge is a list of readable AND-rules |
| summation-based | logistic regression, Naive Bayes | the final answer adds up votes from many rules |
| hardware-near | (digital logic) | inputs are bits, rules are ANDs, votes are integer counts: a small energy and memory footprint |

That last row is worth spelling out. Once a Tsetlin machine is trained, making a prediction needs **no floating-point arithmetic at all**: test some bits, AND them together, and count. That is why the book points to edge devices and low-energy chips as a natural home for it.

Be careful not to read more into this than it says. Being readable does not make a model accurate, and on hard problems a Tsetlin machine is not automatically better than other methods. You will see an honest example of that at the end of this lesson.

!!! note "Where the name comes from"
    The machine is named after **Michael Lvovitch Tsetlin** (1924–1966), a Soviet mathematician and physicist. He invented a tiny learning device, now called a *Tsetlin automaton*, that improves its behaviour in a random environment while remembering its entire past experience as **a single integer**, and that can change its mind when the environment changes. He also studied how collectives of such devices can solve problems together. The Tsetlin machine (Granmo, 2018) builds on that idea: each rule's memory is a team of Tsetlin automata. (Granmo, Ch. 1, box "Tsetlin Machine Origins".) The [next lesson](../tsetlin-automaton/index.md) is all about that one-integer memory.

---

## Step 1 — Objects described by yes/no facts

A Tsetlin machine looks at **objects** (a vehicle, a patient, an image) and describes each one with **Boolean features**: properties that are either **True** or **False**. A vehicle either has wings or it doesn't; it is either yellow or it isn't.

Here is the toy dataset that runs through the whole of Granmo's Chapter 1 (Table 1.1): three cars and three planes, each described by five Boolean features. ● means True and · means False.

| # | Four Wheels | Transports People | Wings | Yellow | Blue | Type |
|---|:-:|:-:|:-:|:-:|:-:|---|
| 1 | ● | ● | · | · | ● | Car |
| 2 | ● | ● | · | ● | · | Car |
| 3 | ● | ● | · | ● | · | Car |
| 4 | ● | ● | ● | · | ● | Plane |
| 5 | ● | · | ● | ● | · | Plane |
| 6 | · | ● | ● | · | ● | Plane |

We write the features of one object as $x_1, x_2, \dots, x_o$, where $o$ is the number of features (here $o = 5$) and each $x_k$ is 1 (True) or 0 (False). Vehicle 1 is the list $(1, 1, 0, 0, 1)$.

Some features matter for telling cars from planes (Wings, Four Wheels). Others are noise (the colour). Part of what a Tsetlin machine learns is which is which.

!!! tip "What to notice"
    - Plane 4 is awkward: it has four wheels *and* transports people, just like the cars. Any rule that only looks at those two features will mistake it for a car.
    - Plane 6 is the only vehicle without four wheels. A plane on a runway would be a different story, so don't expect real data to be this tidy.

---

## Step 2 — Booleanising real data

Real data is rarely yes/no. A tumour has a size in millimetres; menopausal status has three possible values. Before a Tsetlin machine can use such data you **booleanise** it: turn each property into one or more True/False features. Two recipes cover most cases.

**Categories → one feature per value.** Menopausal status takes the values *lt40*, *ge40* and *premeno*, so it becomes three Boolean features, exactly one of which is True for each patient. This is how Granmo booleanises the breast-cancer data in Ch. 1 (Table 1.4). People in deep learning call it *one-hot encoding*.

**Numbers → thresholds.** For a number, pick some cut points $t_1 < t_2 < \dots$ and make one feature per cut point: "size ≥ $t_1$", "size ≥ $t_2$", and so on. A bigger tumour switches on more of them, the way mercury fills a thermometer, so this is called **thermometer encoding**. The alternative, one feature per interval ("20–29 mm"), is just the category recipe again.

The playground below uses the public **UCI Breast Cancer dataset** (286 patients from the Institute of Oncology in Ljubljana, the same data as in the book). Each bar counts the patients whose tumour size was recorded in that 5 mm range; the red part had a recurrence of their cancer, the blue part did not.

<div id="viz-bool" class="viz"><div class="viz-title">🌡️ Interactive: turn tumour size into Boolean features</div></div>

!!! tip "What to notice"
    - With the default thresholds, **size ≥ 20 mm** is True for 216 patients. Their recurrence rate is 35.2%, against 12.9% for the 70 patients below 20 mm. One bit already carries a useful signal.
    - **size ≥ 40 mm** is nearly useless here: 30.3% recurrence if True and 29.6% if False. Where you put a threshold decides whether a feature is informative. Booleanising is a modelling choice, not a formality.
    - Move the patient slider. In thermometer mode the bits fill up from the left and never switch back off. In interval mode exactly one bit is on. Both describe the same patient; they just make different rules easy to write.

Why go to the trouble? The book (Ch. 1, §1.1) gives three reasons:

1. **Bits suit computers.** Each feature is stored as a single bit, and whole groups of them can be tested at once with bitwise operations: fast, small and low-energy.
2. **Boolean algebra is the language of computers.** If you have written `if a and not b:` in a program, you already know the operations a Tsetlin machine uses.
3. **Logic is readable.** Boolean algebra is the same thing as propositional logic, the "True/False statements" kind of reasoning that people can follow step by step.

---

## Step 3 — Literals: a feature or its opposite

Rules often need to say what an object **does not** have. A car is something *without* wings. So for every feature $x_k$ we also allow its **negation** $\lnot x_k$ ("NOT $x_k$"), which is True exactly when $x_k$ is False:

$$
\lnot x_k = 1 - x_k .
$$

Together, the features and their negations are called **literals**, the usual name in Boolean algebra. With $o$ features there are $2o$ literals. For vehicle 1:

| Literal | Four Wheels | Transports People | Wings | Yellow | Blue |
|---|:-:|:-:|:-:|:-:|:-:|
| $x_k$ | ● | ● | · | · | ● |
| $\lnot x_k$ | · | · | ● | ● | · |

Exactly half of the ten literals are True for any object: one from each column. A Tsetlin machine treats a feature and its negation in exactly the same way. Each is simply one of the $2o$ building blocks a rule can choose from.

---

## Step 4 — Rules: AND-ing literals together

A Tsetlin machine expresses patterns as **rules** of the form

$$
\textbf{if}\ \text{condition}\ \textbf{then}\ \text{class},
$$

where the condition is an **AND of literals**. In the research papers a rule's condition is called a **clause** (a *conjunctive clause*), written $C_j$. For example

> **if** Four Wheels **and** Transports People **and not** Wings **then** Car.

A rule **matches** an object only if *every* literal in its condition is True for that object. One False literal is enough to stop it. In symbols, if $L_j$ is the set of literals in rule $j$:

$$
C_j(\mathbf{x}) = \bigwedge_{l \in L_j} l \;=\; \prod_{l \in L_j} l ,
$$

the AND of the chosen literals, which for 0/1 values is the same as their product.

**Try it:** build rules for Car or Plane and see which vehicles they catch.

<div id="viz-rule" class="viz"><div class="viz-title">🧩 Interactive: build a rule and test it on the vehicles</div></div>

Two numbers describe how good a rule is for its class:

- **coverage**: of all the objects of that class, how many does the rule match? (Statisticians call this *recall* or *sensitivity*.)
- **precision**: of all the objects the rule matches, how many really belong to its class?

!!! tip "What to notice"
    1. **Adding a literal can only shrink what a rule matches.** *Four Wheels* matches 5 vehicles (precision 3/5). Adding *Transports People* leaves 4 (precision 3/4): plane 4 still slips through. Adding *NOT Wings* removes it: 3 matches, all cars, so coverage 3/3 and precision 3/3. This is the book's car rule.
    2. **Too specific is also bad.** Add *Blue* instead and the rule matches only car 1 and plane 4: coverage 1/3, precision 1/2. Colour is a detail of individual cars, not a property of cars in general.
    3. **Shorter can be just as good.** *NOT Wings* on its own also gets coverage 3/3 and precision 3/3 on this table. With only six vehicles many rules fit perfectly; with more data the extra literals start to earn their place.
    4. **The empty rule matches everything** (no literal can fail), and the **contradiction** *Four Wheels AND NOT Four Wheels* matches nothing, because one of the two is always False. The book points out that learning naturally avoids such contradictions (Ch. 1, §1.3).
    5. *Wings → Plane* is a one-literal rule that matches all three planes and nothing else.

!!! note "Why only AND?"
    A Tsetlin machine never puts OR or nested logic *inside* a rule. Each rule is a plain AND-list, which keeps it easy to read (Granmo, Ch. 1, §1.2). "OR" appears one level up: when several rules vote for the same class, the class gets support if *any* of them matches. With enough AND-rules you can describe any pattern in Boolean data, so nothing is lost.

---

## Step 5 — From one rule to a machine: rules vote

One rule rarely captures a whole class. A Tsetlin machine therefore uses **many rules**, each assigned to a class. When an object arrives, every rule that matches casts **one vote** for its class, and the class with the most votes wins. Here is the book's example (Ch. 1, §1.5) with plane 4, the tricky one:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🗳️ Two rules vote on vehicle 4",
  "cols": 2,
  "rows": 3,
  "rowHeight": 100,
  "nodes": [
    {"id": "obs", "col": 0.5, "row": 0, "icon": "✈️", "label": "<b>Vehicle 4</b>", "sub": "Four Wheels, Transports People, Wings, Blue", "role": "data", "maxWidth": 230, "note": "The literals that are True: Four Wheels, Transports People, Wings, NOT Yellow, Blue."},
    {"id": "r1", "col": 0, "row": 1, "icon": "🚗", "label": "<b>Rule 1</b>: Four Wheels AND Transports People AND NOT Wings → Car", "role": "model", "maxWidth": 230, "note": "NOT Wings is False for this vehicle, so the whole AND is False."},
    {"id": "r2", "col": 1, "row": 1, "icon": "✈️", "label": "<b>Rule 2</b>: Wings → Plane", "role": "model", "maxWidth": 230, "note": "Wings is True, so the rule matches."},
    {"id": "out", "col": 0.5, "row": 2, "icon": "🏁", "label": "<b>Plane</b> wins, 1 vote to 0", "role": "loss", "shape": "pill", "maxWidth": 230}
  ],
  "edges": [
    {"from": "obs", "to": "r1", "label": "literals"},
    {"from": "obs", "to": "r2", "label": "literals"},
    {"from": "r1", "to": "out", "label": "silent", "dashed": true},
    {"from": "r2", "to": "out", "label": "+1 Plane"}
  ],
  "steps": [
    {"show": ["obs"], "text": "Vehicle 4 arrives as a list of bits: it has four wheels, transports people, has wings and is blue."},
    {"show": ["obs>r1", "r1"], "text": "Rule 1 checks its three literals. <b>NOT Wings</b> is False, so the AND fails: the rule stays silent and casts no vote."},
    {"show": ["obs>r2", "r2"], "text": "Rule 2 checks its single literal. <b>Wings</b> is True, so it matches."},
    {"show": ["r1>out", "r2>out", "out"], "text": "Count the votes: Car 0, Plane 1. The machine answers <b>Plane</b>."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 110, "hideSub": true, "pos": {"obs": [0.5, 0], "r1": [0, 1], "r2": [1, 1], "out": [0.5, 2]}}
}
</script>
</div>

For two classes it is convenient to give each rule a **sign**: rules **for** a class vote $+1$, rules **against** it vote $-1$. Adding them up gives the **vote sum** $v$, and its sign is the answer. (In the research papers these are rules of *positive* and *negative polarity*.) [Lesson 4](../tsetlin-machine-classification/index.md) works this out in full.

Where do the rules come from? Each rule has a **memory**: every one of the $2o$ literals sits at a position on a scale from *Maximally Forgotten* to *Maximally Memorized*, and only literals in the upper half take part in the rule. Training nudges those positions up and down, one example at a time. That memory is the subject of the [next lesson](../tsetlin-automaton/index.md), and the nudges are [lesson 3](../tsetlin-machine-feedback/index.md).

---

## Step 6 — A first look at a learned rule set

Here is what a *trained* Tsetlin machine looks like. We booleanised the UCI breast-cancer data into 11 features (tumour-size thresholds at 20, 30 and 40 mm, involved lymph nodes ≥ 3 and ≥ 6, malignancy grade 3 or 1, and a few yes/no facts), held back 86 patients for testing, and trained a small machine with **3 rules for Recurrence and 3 for No recurrence** on the other 200. The code is a few dozen lines of numpy that follow the book's algorithm; lesson 4 builds it step by step.

The six rules below are the *entire* model. Step through the test patients, or click a feature to see which rules react.

<div id="viz-rules" class="viz"><div class="viz-title">📜 Interactive: read a learned rule set (breast-cancer recurrence)</div></div>

!!! tip "What to notice"
    - **You can read every rule.** Rule 2 says: grade 3, through the node capsule, premenopausal and no radiotherapy → Recurrence. It matches 8 of the 286 patients, and all 8 had a recurrence. Rule 6 (grade 1, fewer than 6 involved nodes, tumour under 20 mm → No recurrence) matches 27 patients, only 1 with a recurrence.
    - **Rules can be broad and weak.** Rule 4 says only "not premenopausal → No recurrence". It matches 136 patients, and 37 of them *did* have a recurrence. On its own it is a poor rule; it is useful only as one voice among several.
    - **Readable is not the same as accurate.** This rule set gets 66 of the 86 test patients right (76.7%), but always answering "No recurrence" gets 61 right (70.9%), and it catches only 6 of the 25 test patients who had a recurrence. Over five training runs with different random seeds, test accuracy ranged from 69.8% to 76.7% (mean 72.6%). This dataset is known to be hard. The point is that you can *see* why the model is weak: 255 of the 286 patients match none of the three Recurrence rules, so for them the model can never answer "Recurrence".
    - The negated literals ("not 20 mm or larger") are how the machine says "small". Thermometer features plus NOT give you intervals for free.

!!! warning "Not a clinical tool"
    These rules come from a tiny model trained on a small, decades-old dataset. They illustrate how a Tsetlin machine represents knowledge; they are not medical advice.

---

## Putting it all together

The whole pipeline, from an object to an answer. Step through it; the last step shows where learning plugs in.

<div class="viz diagram">
<script type="application/json">
{
  "title": "⚙️ Inside a Tsetlin machine",
  "cols": 3,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "obs", "col": 0, "row": 0, "icon": "🔭", "label": "<b>Object</b>", "sub": "a vehicle, a patient, an image", "role": "data", "note": "Anything you want to classify."},
    {"id": "feat", "col": 1, "row": 0, "icon": "🔢", "label": "<b>Boolean features</b>", "sub": "x<sub>1</sub> … x<sub>o</sub>, each 0 or 1", "role": "data", "note": "Made by booleanising the raw data: one feature per category, thresholds for numbers."},
    {"id": "lits", "col": 2, "row": 0, "icon": "🔀", "label": "<b>Literals</b>", "sub": "every x<sub>k</sub> and ¬x<sub>k</sub>", "role": "data", "note": "2o building blocks; exactly half of them are True for any object."},
    {"id": "rules", "col": 2, "row": 1, "icon": "🧩", "label": "<b>Rules</b> (AND of literals)", "sub": "each with its own memory", "role": "model", "note": "A rule matches only if all its memorized literals are True. Its memory decides which literals are in it."},
    {"id": "votes", "col": 1, "row": 1, "icon": "🗳️", "label": "<b>Votes</b>", "sub": "for minus against = v", "role": "loss", "note": "Every matching rule adds +1 (for) or −1 (against)."},
    {"id": "out", "col": 0, "row": 1, "icon": "🏁", "label": "<b>Class</b>", "sub": "the side with more votes", "shape": "pill"}
  ],
  "edges": [
    {"from": "obs", "to": "feat"},
    {"from": "feat", "to": "lits"},
    {"from": "lits", "to": "rules"},
    {"from": "rules", "to": "votes"},
    {"from": "votes", "to": "out"}
  ],
  "steps": [
    {"show": ["obs", "obs>feat", "feat"], "text": "Turn the object into bits: <b>booleanise</b> it (Step 2)."},
    {"show": ["feat>lits", "lits"], "text": "Add the negation of every feature to get the <b>literals</b> (Step 3)."},
    {"show": ["lits>rules", "rules"], "text": "Each <b>rule</b> ANDs the literals it has memorized and checks whether they are all True (Step 4). How a literal gets memorized is the topic of <a href=\"../tsetlin-automaton/\">lesson 2</a>."},
    {"show": ["rules>votes", "votes"], "text": "Matching rules <b>vote</b> for or against the class, and the votes are added up (Step 5 and <a href=\"../tsetlin-machine-classification/\">lesson 4</a>)."},
    {"show": ["votes>out", "out"], "text": "The side with more votes is the answer. No floating-point arithmetic was needed: only bits, AND and counting."},
    {"show": ["rules", "votes"], "text": "During <b>training</b> the vote is compared with the true class, and feedback goes back to every rule's memory: memorize this literal, forget that one (<a href=\"../tsetlin-machine-feedback/\">lesson 3</a>)."}
  ],
  "narrow": {"cols": 2, "rows": 3, "hideSub": true, "pos": {"obs": [0, 0], "feat": [1, 0], "lits": [1, 1], "rules": [0, 1], "votes": [0, 2], "out": [1, 2]}}
}
</script>
</div>

And the same thing as code: the vehicle table, the book's two rules, and voting. A rule is just a list of literal names.

```python
FEATURES = ["Four Wheels", "Transports People", "Wings", "Yellow", "Blue"]

# Granmo's vehicle table (Ch. 1, Table 1.1): 1 = True, 0 = False
vehicles = [
    ([1, 1, 0, 0, 1], "Car"),
    ([1, 1, 0, 1, 0], "Car"),
    ([1, 1, 0, 1, 0], "Car"),
    ([1, 1, 1, 0, 1], "Plane"),
    ([1, 0, 1, 1, 0], "Plane"),
    ([0, 1, 1, 0, 1], "Plane"),
]


def literals(x):
    """Every feature, then every negated feature: 2 x 5 = 10 literals."""
    lits = {name: v for name, v in zip(FEATURES, x)}
    lits.update({"NOT " + name: 1 - v for name, v in zip(FEATURES, x)})
    return lits


def rule_matches(rule, x):
    """A rule is an AND of literals: it matches only if every one of them is True."""
    lits = literals(x)
    return all(lits[l] == 1 for l in rule)


# Two hand-written rules (the book's examples). A real Tsetlin machine learns these.
rules = [
    (["Four Wheels", "Transports People", "NOT Wings"], "Car"),
    (["Wings"], "Plane"),
]


def classify(x):
    votes = {"Car": 0, "Plane": 0}
    for condition, cls in rules:
        if rule_matches(condition, x):
            votes[cls] += 1                       # each matching rule casts one vote
    return max(votes, key=votes.get), votes


for i, (x, label) in enumerate(vehicles, start=1):
    pred, votes = classify(x)
    print(f"vehicle {i}: votes {votes} -> {pred:5s} (true: {label})")


# Booleanising a number with thresholds ("thermometer" encoding)
def thermometer(value, thresholds):
    return [int(value >= t) for t in thresholds]

print(thermometer(27, [20, 30, 40]))   # a 27 mm tumour
```

```text
vehicle 1: votes {'Car': 1, 'Plane': 0} -> Car   (true: Car)
vehicle 2: votes {'Car': 1, 'Plane': 0} -> Car   (true: Car)
vehicle 3: votes {'Car': 1, 'Plane': 0} -> Car   (true: Car)
vehicle 4: votes {'Car': 0, 'Plane': 1} -> Plane (true: Plane)
vehicle 5: votes {'Car': 0, 'Plane': 1} -> Plane (true: Plane)
vehicle 6: votes {'Car': 0, 'Plane': 1} -> Plane (true: Plane)
[1, 0, 0]
```

Everything here was written by hand. The interesting part, how a machine **finds** good rules by itself from examples, starts in the next lesson.

---

## Common misconceptions

!!! warning "\"Interpretable means correct.\""
    Being able to read a rule tells you *what* the model does, not whether it is right. The breast-cancer rule set above is perfectly readable and only a little better than always guessing "No recurrence". The benefit is that you can spot such weaknesses, and argue with them.

!!! warning "\"A Tsetlin machine is just a decision tree.\""
    Both use readable conditions, but a decision tree follows **one** path to **one** leaf, so one rule decides. A Tsetlin machine lets **many** rules fire at once and **adds up** their votes. In the book's words it combines rule-based and summation-based decision making (Ch. 1, §1.5).

!!! warning "\"Booleanising is a harmless preprocessing step.\""
    It decides what the machine can see. A threshold at 40 mm for tumour size produces a feature that carries almost no signal in this dataset; one at 20 mm produces a useful one. Too few thresholds hide detail, too many give the machine lots of near-duplicate features to sort through.

!!! warning "\"The rules are written by an expert.\""
    In this lesson we wrote them by hand to see what they look like. In a real Tsetlin machine every rule is **learned** from labelled examples, and nobody tells it which literals to use.

---

## Check yourself

??? question "1. A dataset has 8 Boolean features. How many literals does a Tsetlin machine work with, and how many of them are True for any one object?"
    $2 \times 8 = 16$ literals (each feature and its negation). Exactly **8** are True for any object: for every feature, either $x_k$ or $\lnot x_k$ is True, never both.

??? question "2. Does the rule \"if Yellow then Car\" work on the vehicle table? Give its coverage and precision."
    It matches vehicles 2, 3 and 5. Two of those are cars, so **coverage 2/3** (it misses blue car 1) and **precision 2/3** (yellow plane 5 is a false positive). Colour is a poor basis for recognising cars.

??? question "3. Why can't adding a literal to a rule ever make it match more objects?"
    A rule is an AND: an object must make *every* literal True. Adding a literal adds one more requirement, so every object that matched before must still pass the new test, and some may fail it. The set of matches can only stay the same or shrink.

??? question "4. Booleanise a tumour of 33 mm with thermometer thresholds 20, 30 and 40 mm. Then write a single rule that means \"between 20 and 29 mm\"."
    Thermometer bits: size ≥ 20 → 1, size ≥ 30 → 1, size ≥ 40 → 0, so $(1, 1, 0)$. "Between 20 and 29 mm" is **size ≥ 20 AND NOT size ≥ 30**. Negated literals turn thresholds into intervals.

??? question "5. Name two reasons the book gives for using Boolean features."
    Any two of: they are stored as single bits and processed with fast, low-energy bitwise operations (good for edge devices); they let you build patterns with Boolean algebra, the language of computers; and Boolean algebra is propositional logic, which people can follow, so the model stays transparent.

---

## Further reading

- Ole-Christoffer Granmo, [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/) (free book), Ch. 1 introduction, §1.1 "Boolean Features" and §1.2 "Creating Patterns with AND and NOT", plus the boxes "Tsetlin Machine Origins" and "Tsetlin Machines vs Black Boxes". The vehicle table and the overview in this lesson are based on Ch. 1, Table 1.1 and Figure 1.1.
- Ole-Christoffer Granmo, [*The Tsetlin Machine: A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic*](https://arxiv.org/abs/1804.01508) (2018), the original paper, using the research terms (clauses, Type I/II feedback).
- M. Zwitter and M. Soklic, [Breast Cancer dataset](https://archive.ics.uci.edu/dataset/14/breast+cancer), UCI Machine Learning Repository (University Medical Centre, Institute of Oncology, Ljubljana), the data used in Steps 2 and 6.
- Cynthia Rudin, [*Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead*](https://www.nature.com/articles/s42256-019-0048-x), Nature Machine Intelligence (2019), the wider argument for readable models in medicine and other high-stakes fields.

---

## The series

This lesson is the first of six that follow the published chapters of Granmo's book:

1. **What is a Tsetlin machine?** (this lesson): Boolean features, literals, AND-rules and voting at a glance.
2. [The Tsetlin automaton: a memory that learns](../tsetlin-automaton/index.md): the graded memory behind every literal, and its roots in the two-armed bandit.
3. [Learning rules: Recognize, Erase and Reject feedback](../tsetlin-machine-feedback/index.md): how a rule learns frequent patterns, and how it learns to tell classes apart.
4. [Many rules voting: classification with a Tsetlin machine](../tsetlin-machine-classification/index.md): the vote sum, the Voting Margin and the complete learning algorithm, on the breast-cancer data.
5. [Convolutional Tsetlin machines](../convolutional-tsetlin-machine/index.md): finding patterns anywhere in an image or a sequence.
6. [Confidence, trustworthiness and composites](../tsetlin-machine-confidence/index.md): when to trust a Tsetlin machine, and how to combine several.

**Next lesson:** [The Tsetlin automaton: a memory that learns](../tsetlin-automaton/index.md)

<script src="viz/tsetlin-machine-intro.js"></script>
