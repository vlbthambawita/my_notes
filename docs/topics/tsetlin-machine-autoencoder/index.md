# Tsetlin Machine Autoencoders and Word Embeddings

!!! abstract "TL;DR"
    Most text and clinical records come without labels. A **Tsetlin machine autoencoder** still learns from them: it hides one word of a document and tries to predict it from the words that are left. Every word is an output, and all outputs share **one pool of rules**. Each rule has a **signed integer weight per output word**: positive means "when this rule fires, the hidden word is probably there", negative means "probably not". Training uses the same Recognize, Erase and Reject Feedback as classification. The hidden word is **masked**: it counts as True and its memory is frozen. Training examples are **balanced** (the hidden word is present in half of them), and each one is the OR of a few documents. After training, a word's row of positive weights is its **embedding**: a short list of readable rules such as *"fever AND sputum"* or *"breathless AND wheeze"* for *cough*. Words with similar rows are used in similar contexts. On 600 generated clinical notes, the nearest neighbour of every one of 18 findings came from the same condition (17.5 of 18 on average over 10 seeds).

**Level:** intermediate · **Prerequisites:** [Classification by voting](../tsetlin-machine-classification/index.md) (rules voting for and against, the vote sum $v$, the Voting Margin $T$) and [Weighted clauses and clause sharing](../weighted-tsetlin-machine/index.md) (rules with integer weights, a pool of rules shared by several outputs). You need nothing beyond AND, OR, counting, and a cosine.

!!! note "Beyond the book"
    The book chapter on this topic, *An Introduction to Tsetlin Machines*, Ch. 6 "Auto-Encoding", is not published yet. This lesson is based on the research papers listed in [Further reading](#further-reading), mainly Bhattarai et al. (2024), and on the `TMAutoEncoder` in the CAIR group's TMU library. It uses the book's vocabulary. The clinical notes, the code and every number not attributed to a paper come from our own runs, made for this lesson.

---

## Why should I care?

Labels are expensive. Someone has to read each note or image and write down the answer. Unlabelled text is everywhere: discharge letters, referral notes, radiology reports, the scientific literature. A model that can learn something useful from the text alone is very valuable.

- **Words get their meaning from their company.** "Words that occur in the same contexts tend to have similar meanings" (the *distributional hypothesis*, Harris 1954). *Cough* keeps company with *fever* and *sputum* in some notes and with *wheeze* in others. A model that learns this company has learned something about what *cough* means, without anyone defining it.
- **Embeddings power modern NLP, but you can't read them.** Word2Vec and GloVe turn each word into about 300 decimal numbers. Similar words get similar vectors, but no single number means anything to a person.
- **A Tsetlin machine embedding is made of rules.** Bhattarai et al. (2024) represent each word by the logical rules that predict it, for example *coffee* by rules over *hot*, *cup* and *black*. You can read why two words are similar: they are predicted by the same rules.

This lesson builds that autoencoder step by step and trains it on a small set of generated clinical notes, where you know the truth and can check what it learns.

---

## Step 1 — Learning without labels: hide a word, predict it

Take a short clinical note and treat it as a **set of words**, ignoring order and repeats:

> *fever, cough, sputum, chest pain*

Each word of the vocabulary becomes a Boolean feature $x_k$: True if the note mentions it. This is the usual bag-of-words Booleanisation, as in [the first lesson](../tsetlin-machine-intro/index.md).

Now hide one word, say *cough*, and ask: **given the rest of the note, was *cough* there?** The answer is in the note itself, so you can make as many training examples as you have notes, without asking anyone for labels. This is **self-supervised learning**: the data supplies its own targets. Hiding part of the input and predicting it is the same trick as the "masked word" task used to pre-train BERT, here with logic instead of a neural network.

To get the hidden word right, the model has to learn what *cough* usually comes with. If *fever* and *sputum* are there, *cough* is likely. If *sneezing* and *runny nose* are there, it is less likely. These co-occurrence patterns are exactly what an AND-rule can express.

An **autoencoder** does this for every word at once: its output has one slot $\hat x_k$ per input word, and $\hat x_k$ is predicted from all the other inputs. A neural autoencoder squeezes the input through a small hidden layer. The Tsetlin machine version squeezes it through a small **pool of rules** instead.

!!! tip "What to notice"
    - No labels are needed: the target is a word the note already contains (or doesn't).
    - The task is a means to an end. What we want is not a perfect predictor of hidden words, but the **rules** the machine has to learn in order to predict them.

---

## Step 2 — One pool of rules, one weight per word

A separate Tsetlin machine for each of the 18 words would learn 18 separate rule sets, and *fever AND sputum* would have to be discovered again for *cough*, for *chest pain* and for *fatigue*. The autoencoder of Bhattarai et al. (2024) instead builds on the **coalesced Tsetlin machine** (Glimsdal & Granmo, 2021; the [weighted lesson](../weighted-tsetlin-machine/index.md) covers it): **one pool of $n$ rules shared by all outputs**, and a signed integer weight $w_{kj}$ that connects rule $j$ to output word $k$.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧩 The Tsetlin machine autoencoder: predicting one hidden word",
  "cols": 3,
  "rows": 2,
  "rowHeight": 120,
  "nodes": [
    {"id": "note", "col": 0, "row": 0, "icon": "📝", "label": "A note as a set of words", "sub": "x<sub>1</sub> … x<sub>m</sub>", "role": "data", "note": "One Boolean feature per vocabulary word: True if the note mentions it."},
    {"id": "mask", "col": 1, "row": 0, "icon": "🙈", "label": "Hide word k", "sub": "x<sub>k</sub> counts as True", "role": "data", "note": "The rules can't see the hidden word. Setting it to True means a rule that contains it simply ignores it."},
    {"id": "pool", "col": 2, "row": 0, "icon": "🧩", "label": "Shared pool of n rules", "sub": "each an AND of words", "role": "model", "note": "The same rules serve every output word. A rule that fires is True for this note."},
    {"id": "wts", "col": 2, "row": 1, "icon": "⚖️", "label": "Weights of word k", "sub": "w<sub>k1</sub> … w<sub>kn</sub>, signed integers", "role": "loss", "note": "Each output word has its own row of weights over the same rules. Positive: the rule votes 'the word is there'. Negative: 'it is not'."},
    {"id": "sum", "col": 1, "row": 1, "icon": "🗳️", "label": "Vote sum v<sub>k</sub> = Σ w<sub>kj</sub> C<sub>j</sub>", "role": "loss", "note": "Add the weights of the rules that fire."},
    {"id": "out", "col": 0, "row": 1, "icon": "🏷️", "label": "x̂<sub>k</sub> = 1 if v<sub>k</sub> ≥ 0", "shape": "pill", "role": "opt", "note": "The predicted value of the hidden word."}
  ],
  "edges": [
    {"from": "note", "to": "mask"},
    {"from": "mask", "to": "pool"},
    {"from": "pool", "to": "wts", "label": "C<sub>j</sub>"},
    {"from": "wts", "to": "sum"},
    {"from": "sum", "to": "out"}
  ],
  "steps": [
    {"show": ["note"], "text": "Booleanise the note: one feature per vocabulary word. The autoencoder of Bhattarai et al. uses only the words themselves, not their negations."},
    {"show": ["note>mask", "mask"], "text": "Pick the output word k to predict and <b>mask</b> it: its feature is set to True, so any rule that contains it just ignores it."},
    {"show": ["mask>pool", "pool"], "text": "Every rule in the shared pool checks its condition on the masked note: C<sub>j</sub> = 1 if all its memorized words are present."},
    {"show": ["pool>wts", "wts"], "text": "Output word k has its own row of signed integer weights over the same rules. Another word uses the same rules with a different row."},
    {"show": ["wts>sum", "sum"], "text": "The vote sum is the weighted sum of the rules that fire: v<sub>k</sub> = Σ<sub>j</sub> w<sub>kj</sub> C<sub>j</sub>."},
    {"show": ["sum>out", "out"], "text": "Predict the hidden word as present if v<sub>k</sub> ≥ 0. Repeat with every other word as the hidden one, using the same rules."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 115, "pos": {"note": [0, 0], "mask": [1, 0], "pool": [1, 1], "wts": [1, 2], "sum": [0, 2], "out": [0, 1]}}
}
</script>
</div>

In symbols, with rules $C_1, \dots, C_n$ (in the papers: *clauses*), each an AND of words, the prediction for word $k$ is

$$
v_k = \sum_{j=1}^{n} w_{kj}\, C_j(x_1, \dots, x_{k-1}, 1, x_{k+1}, \dots, x_m), \qquad \hat x_k = \begin{cases} 1 & v_k \ge 0 \\ 0 & v_k < 0. \end{cases}
$$

The $1$ in position $k$ is the **mask**. It replaces the real value of $x_k$, so a rule like *"fever AND cough AND sputum"* acts as *"fever AND sputum"* when it predicts *cough*. That is how the same rule can serve as context for *cough* and, unmasked, for *fatigue*.

Two details differ from the classification lessons:

- **No rules "for" and "against".** The sign of the weight says which way a rule votes, and it can change during training. The same rule can vote *for* one word and *against* another.
- **No negated literals** in Bhattarai et al.'s version: a rule only says which words must be **present**. TMU's implementation can add the negations ($\lnot x_k$, "the note does not mention …"), and Kadhim et al. (2024) study that variant. We stick to the paper.

!!! tip "What to notice"
    - **One rule, many words.** With $n = 24$ rules and 18 words there are $18 \times 24 = 432$ weights but only 24 rules to learn. A useful context such as *"fever AND sputum"* is learned once and reused by every word it predicts.
    - A tie ($v_k = 0$, for example when no rule fires) counts as "present", following the paper's $\hat x_k = [0 \le v_k]$. The classification lesson sent ties the other way; this choice doesn't matter for the embeddings, which come from the weights.

---

## Step 3 — Building a training example: a coin flip and a handful of notes

Bhattarai et al. make each training example in a particular way (their Algorithm 1). For a target word $k$:

1. **Flip a fair coin** for the target value $q$. With $q = 1$ the example will contain word $k$; with $q = 0$ it won't. Rare words are therefore present in **half** of their examples, however rare they are in the data. (TMU calls this `output_balancing = 0.5`.)
2. **Pick $u$ notes at random** that mention word $k$ (if $q = 1$) or don't (if $q = 0$). The paper calls $u$ the **accumulation**.
3. **OR them together**: the example contains every word that appears in any of the $u$ notes. Combining several notes makes rare context words show up more often, which helps the rules pick them up.
4. **Mask word $k$** and train output $k$ to answer $q$.

One **round** of training makes one such example for every word of the vocabulary, in random order.

Our toy data are **600 generated clinical notes** (plus 300 held out for testing), using 18 findings. Each note comes from one condition (sometimes two) chosen at random, and each condition mentions its typical findings with fixed probabilities: pneumonia (*fever, cough, sputum, chest pain, breathless, fatigue*), asthma (*wheeze, breathless, cough*), hay fever (*sneezing, runny nose, itchy eyes*), gastroenteritis (*nausea, vomiting, diarrhoea, stomach ache*, sometimes *fever*), diabetes (*thirst, frequent urination, weight loss, fatigue*) and migraine (*headache*, often *nausea*). Every other finding turns up by chance 3% of the time. 125 of the 600 notes come from two conditions, and a note mentions 3.71 findings on average. Some findings belong to two conditions on purpose (*cough*, *fever*, *fatigue*, *nausea*, *breathless*), like a word with two meanings. The conditions are never shown to the machine; we use them only to check its answers.

<div id="viz-sample" class="viz"><div class="viz-title">🎲 Interactive: build one training example</div></div>

!!! tip "What to notice"
    - **The coin flip balances the classes.** *Chest pain* is in 76 of the 600 notes (12.7%), yet it is present in half of its training examples. Without balancing, "never there" would be right 87% of the time and the rules would have little reason to learn.
    - **Accumulation adds context, and noise.** Averaged over all 18 target words, an example built from $u = 1$ note has 3.63 context words when the target is present; with $u = 2$ it has 5.73, with $u = 3$ 7.12, and with $u = 5$ 9.36 (more when the target is absent: 3.53, 6.20, 8.21 and 11.14). With only 18 words, a few notes already switch on half the vocabulary. In a corpus with tens of thousands of words, the paper's $u = 25$ is what makes rare context words visible at all.
    - Try *headache* with $u = 1$. Of its 114 training notes, 38 mention nothing else or only *nausea*; its other companions depend on whatever second condition a note happens to have. A word with little steady company has little context to be predicted from.

---

## Step 4 — Feedback: the weight's sign decides the role

Training an output word $k$ on an example $(x, q)$ is the familiar recipe from [the feedback lesson](../tsetlin-machine-feedback/index.md) and [classification](../tsetlin-machine-classification/index.md), with the weight's sign deciding which way each rule is trained.

**1. How much feedback?** Compute the vote sum $v_k$ on the masked example (empty rules count as True while training) and clip it to $[-T, T]$. Each rule is picked for feedback independently with probability

$$
p = \frac{T - v_k}{2T} \;\text{ if } q = 1, \qquad p = \frac{T + v_k}{2T} \;\text{ if } q = 0.
$$

This is the **Voting Margin** $T$ again: the target is $v_k \ge T$ when the word is there and $v_k \le -T$ when it is not.

**2. Which feedback?** A rule with $w_{kj} \ge 0$ is trying to fire when the word is present; a rule with $w_{kj} < 0$ is trying to fire when it is absent. So, for a picked rule:

| | rule fires ($C_j = 1$) | rule does not fire |
|---|---|---|
| its weight's sign **matches** the target ($w \ge 0$ and $q = 1$, or $w < 0$ and $q = 0$) | **Recognize Feedback**, then the weight moves towards $q$ | **Erase Feedback** |
| its weight's sign **does not match** | **Reject Feedback**, then the weight moves towards $q$ | nothing |

"The weight moves towards $q$" means $w_{kj} \mathrel{+}= 1$ if $q = 1$ and $w_{kj} \mathrel{-}= 1$ if $q = 0$. So a rule that fires on notes with the word is rewarded with a larger weight, and a rule that fires on notes without it loses weight. A rule that keeps firing on the wrong side can **cross zero** and change its role: the paper notes that it then trains to recognise the word's absence instead.

**3. The masked word is frozen.** Recognize, Erase and Reject Feedback skip word $k$ itself: its memory position in every rule stays where it is. Otherwise a rule would learn *"cough"* as context for predicting *cough*, which is useless (and the mask makes it always True anyway).

As in the paper, Recognize Feedback memorizes present words with Memorize Value 1 and forgets absent ones with Forget Value $1/s$. In the research papers: Recognize = Type Ia, Erase = Type Ib, Reject = Type II.

!!! note "One extra detail from the TMU code"
    TMU's `TMAutoEncoder` adds a brake that the paper does not describe. At the start of each round, a rule takes part (votes and learns) only with probability $(T - \bar w_j)/T$, where $\bar w_j$ is the rule's average absolute weight over all output words, capped at $T$. A rule that has collected large weights has found its role and is rarely disturbed. Without the brake, our rules sometimes drifted to new words while keeping weights earned by their old words. We use the brake. TMU's demo also caps every rule at 3 words (a "literal budget"); we don't.

!!! tip "What to notice"
    - Nothing here is new machinery. An autoencoder is a multi-output, weighted Tsetlin machine with two twists: the input it sees is masked, and the training examples are sampled in a balanced way.
    - The weights are integers that only ever move by ±1, so a weight of +12 means "this rule fired on examples with the word, 12 more times than on examples without it, when it was picked for feedback".

---

## Step 5 — What the rules learn

We trained the autoencoder on the 600 notes with **24 rules**, Voting Margin $T = 24$, Forget Value $1/s$ with $s = 4$, memory depth 100 (positions 1–100, Memorized above 50), $u = 2$ notes per example, and **300 rounds** (each round one example per word, so 5,400 examples). It takes about two seconds in plain Python, and the widget below runs the same seeded code (seed 7), so its numbers match ours exactly.

Hide a word, build a note, and see which rules vote:

<div id="viz-mask" class="viz"><div class="viz-title">🙈 Interactive: hide a word, let the rules vote</div></div>

!!! tip "What to notice"
    - ***Cough* has two contexts.** Its strongest rules are R16 *"fever AND sputum"* (+12), the pneumonia context, and R13 *"breathless AND wheeze"* (+7), the asthma context. A single average vector can't show this; a list of rules can. Bhattarai et al. see the same with real words: *heart* and *surgery* share the rules *went ∧ hospital* and *old ∧ disease ∧ patient*, while *heart* also has *woman ∧ love*.
    - **Rules can contain the hidden word.** R7 is *"fever AND <u>cough</u> AND sputum"* (+5). For *cough* the mask makes it *"fever AND sputum"*; the rule is shared, and for *fatigue* (+11) it really does need *cough*.
    - **Negative weights are the "unlikely" contexts.** For *cough*: −11 for *"thirst AND frequent urination"* and −10 for *"sneezing AND runny nose AND itchy eyes"*.
    - **Try the presets.** *fever + sputum + chest pain*: R7 (+5) and R16 (+12) fire, $v = +17$, *cough* predicted present. *breathless + wheeze*: only R13 fires, $v = +7$. *sneezing + runny nose*: no rule fires (the hay-fever rules also want *itchy eyes*), so $v = 0$, a tie, which the paper's rule counts as "present". Hide *fatigue* and use *thirst* + *frequent urination*: three diabetes rules fire, $v = 9 + 8 + 10 = +27$.
    - **Not every rule is clean.** *Fatigue* has weight +8 on R1 *"wheeze AND frequent urination"*, a rule that matches only 16 of the 600 training notes (9 with *fatigue*, 7 without). Compare R16 for *cough*: 74 notes, 65 of them with *cough*. Check how often a rule fires before you read much into it.
    - ***Headache* has no strong rules for it.** In migraine notes it comes with nothing but (sometimes) *nausea*, and *nausea* is more often a gastroenteritis word (77 of its 162 notes also mention *vomiting*, 58 mention *headache*). Little steady context, no strong rule.

---

## Step 6 — From rules to embeddings

After training, the weight matrix $W$ (18 words × 24 rules) holds everything the machine learned about each word's company. Bhattarai et al. turn it into embeddings in two ways:

- the **weighted logical embedding** $E = \operatorname{clip}(W, 0, T)$: keep the positive weights, set negative ones to zero, cap at $T$;
- the **purely logical embedding** $B = [W > 0]$: just which rules vote for the word.

Row $k$ of $E$ is the embedding of word $k$: a vector with one entry per rule. Two words are compared with the **cosine similarity** of their rows,

$$
\cos(\mathbf e_k, \mathbf e_l) = \frac{\mathbf e_k \cdot \mathbf e_l}{\lVert \mathbf e_k \rVert \, \lVert \mathbf e_l \rVert},
$$

which is 1 when two words put their weight on the same rules in the same proportions, and 0 when they share no rule. Unlike a Word2Vec vector, every coordinate has a name: it is a rule you can read.

<div id="viz-embed" class="viz"><div class="viz-title">🗺️ Interactive: train the autoencoder and explore the embedding</div></div>

!!! tip "What to notice"
    - **The blocks on the diagonal are the conditions.** With the defaults, the nearest neighbour of **every one of the 18 words** comes from a condition it belongs to: *sneezing* → *itchy eyes* (0.93), *nausea* → *stomach ache*, *thirst* → *frequent urination*.
    - **Two-context words sit between their groups.** *Cough* is close to *sputum* (0.55) and less close to *wheeze* (0.32), and far from *sneezing* (0.07). *Fatigue* is closest to *thirst* (0.72) but also related to *fever* (0.40), and unrelated to *itchy eyes* (0.01).
    - **The embedding is sparse.** Each word puts positive weight on only 3 to 9 of the 24 rules (6.7 on average). Bhattarai et al. report that with 600 rules, a word links to at most 10% of them.
    - **Compare with plain counting.** Switch "similarity from" to *co-occurrence counts (PPMI)*: cosine similarity of positive pointwise mutual information, a classic count-based embedding computed from the same 600 notes. It gets 16 of 18 nearest neighbours right. On this toy data counting works almost as well; what the rules add is that you can **read why** two words are similar.
    - **Change $u$.** With $u = 5$, almost every example contains half the vocabulary and the rules get long and vague. Change the seed to see how much a single run varies.

---

## Step 7 — How good is it, honestly?

Our nearest-neighbour check is crude, and the data are generated. Still, it shows the trade-offs. We repeated the training with **10 seeds** (1–10) for several values of $u$, and also scored each model as a hidden-word predictor on the 300 held-out notes. Because most words are absent from most notes, we use **balanced accuracy**: the average of the hit rate on notes with the word and on notes without it (a coin scores 50%).

| notes per example $u$ | nearest neighbour from the same condition (of 18) | hidden-word balanced accuracy, mean ± sd |
|---|---|---|
| 1 | 16.8 (16–17) | 76.2% ± 1.3 |
| **2** (our default) | **17.5** (17–18) | 72.8% ± 2.4 |
| 3 | 16.6 (16–17) | 64.3% ± 2.5 |
| 5 | 16.2 (15–17) | 53.8% ± 1.6 |
| *baselines on the same notes* | raw co-occurrence counts: 13; PPMI: 16 | logistic regression per word: 84.3% |

- **It is not the best predictor of hidden words.** A logistic regression for each word (scikit-learn, balanced class weights) predicts the hidden word better: 84.3% against 72.8%. The autoencoder's 24 shared rules, trained on ORs of several notes, are tuned for describing contexts, not for predicting single notes. The bigger $u$, the further the training examples are from a single note, and the worse the prediction.
- **The embedding is good on this toy, but so is counting.** PPMI gets 16 of 18. The point of the Tsetlin machine embedding is that each similarity can be explained by rules.
- **Single runs vary.** The default run (seed 7) is one of the better ones (18 of 18) and scores 69.9% balanced accuracy.

**What the papers report** (on real text, not our toy). Bhattarai et al. (2024) trained on the One Billion Word benchmark with 600 rules, $T = 1200$, $s = 5$, $u = 25$ and 2,000 examples per word. On six word-similarity benchmarks, the average Spearman correlation with human similarity scores was 0.52 for the TM embedding, against 0.58 for Word2Vec, 0.57 for GloVe and 0.54 for FastText (trained on the same data), so slightly behind. Used as input features to a BiLSTM classifier, their purely logical embedding was close to pre-trained GloVe on six text-classification tasks, e.g. 96.40% vs 95.20% accuracy on TREC and 75.61% vs 76.38% on SST-2. Kadhim et al. (2024) add negated words to the rules ("reasoning by elimination") and report 90.62% accuracy on IMDB sentiment. Kadhim et al. (2025, preprint) build the embedding from the **whole memory** of every rule, including forgotten words, so that one training run gives reusable embeddings for all words.

!!! warning "What our toy data can and can't show"
    The notes were generated from six conditions with fixed probabilities, so the "right answer" is known and clean. Real clinical notes have negation (*"no fever"*), synonyms, abbreviations, and thousands of words. Don't read our numbers as evidence about real records; read them as a demonstration of the mechanism.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 One training example of a Tsetlin machine autoencoder",
  "cols": 3,
  "rows": 2,
  "rowHeight": 120,
  "nodes": [
    {"id": "pick", "col": 0, "row": 0, "icon": "🎯", "label": "Next target word k", "sub": "every word, once per round", "role": "data", "note": "One round = one example per vocabulary word, in random order."},
    {"id": "coin", "col": 1, "row": 0, "icon": "🪙", "label": "Coin flip q", "sub": "present or absent, 50/50", "role": "data", "note": "Balanced targets: rare words are present in half of their examples."},
    {"id": "or", "col": 2, "row": 0, "icon": "📚", "label": "OR of u notes", "sub": "with word k if q = 1, without if q = 0", "role": "data", "note": "Accumulation: combining notes makes rare context words appear more often."},
    {"id": "eval", "col": 2, "row": 1, "icon": "🧩", "label": "Rules fire, x<sub>k</sub> masked", "sub": "v<sub>k</sub> = Σ w<sub>kj</sub> C<sub>j</sub>", "role": "model", "note": "The hidden word counts as True. While training, empty rules count as True too."},
    {"id": "prob", "col": 1, "row": 1, "icon": "🎲", "label": "Pick rules with probability p", "sub": "(T ∓ v<sub>k</sub>) / 2T", "role": "loss", "note": "The Voting Margin T: no feedback once the vote sum is T on the right side."},
    {"id": "fb", "col": 0, "row": 1, "icon": "✍️", "label": "Feedback by weight sign", "sub": "Recognize, Erase or Reject; w ± 1", "role": "grad", "note": "Matching sign: Recognize (fires) or Erase (doesn't). Other sign and fires: Reject. Firing rules move their weight towards q. The masked word is frozen."}
  ],
  "edges": [
    {"from": "pick", "to": "coin"},
    {"from": "coin", "to": "or"},
    {"from": "or", "to": "eval"},
    {"from": "eval", "to": "prob"},
    {"from": "prob", "to": "fb"},
    {"from": "fb", "to": "pick", "label": "next"}
  ],
  "steps": [
    {"show": ["pick"], "text": "Take the next target word k. A round visits every word of the vocabulary once, in random order."},
    {"show": ["pick>coin", "coin"], "text": "Flip a coin: should this example contain word k (q = 1) or not (q = 0)? This balances rare and common words."},
    {"show": ["coin>or", "or"], "text": "Pick u notes that do (or don't) mention word k and OR them together. The paper uses u = 25 on a large corpus; our toy uses u = 2."},
    {"show": ["or>eval", "eval"], "text": "Mask word k (it counts as True), evaluate every rule in the shared pool, and add the weights of word k for the rules that fire."},
    {"show": ["eval>prob", "prob"], "text": "Clip v<sub>k</sub> to [−T, T] and pick each rule for feedback with probability (T − v<sub>k</sub>)/2T if q = 1, or (T + v<sub>k</sub>)/2T if q = 0."},
    {"show": ["prob>fb", "fb"], "text": "Picked rules get Recognize, Erase or Reject Feedback depending on whether their weight's sign matches q. Firing rules move their weight one step towards q. Word k's own memory is frozen."},
    {"show": ["fb>pick", "pick"], "text": "On to the next word. After training, the embedding of word k is its row of weights, clipped to [0, T]: the rules that predict it."}
  ],
  "narrow": {"cols": 2, "rows": 3, "rowHeight": 115, "pos": {"pick": [0, 0], "coin": [1, 0], "or": [1, 1], "eval": [1, 2], "prob": [0, 2], "fb": [0, 1]}}
}
</script>
</div>

Here is a complete autoencoder in numpy, about 80 lines including the data generator. It follows Steps 2–6 line by line, vectorised over rules and words, and includes TMU's brake from Step 4. It uses numpy's random numbers and its own copy of the notes, so its rules and numbers differ a little from the widgets'.

```python
import numpy as np

# ---- 1. Toy clinical notes: each note is the SET of findings it mentions (no labels are used)
WORDS = ["fever", "cough", "sputum", "chest pain", "breathless", "wheeze", "sneezing", "runny nose",
         "itchy eyes", "nausea", "vomiting", "diarrhoea", "stomach ache", "thirst", "frequent urination",
         "weight loss", "fatigue", "headache"]
CONDITIONS = [{"fever": .9, "cough": .9, "sputum": .7, "chest pain": .5, "breathless": .5, "fatigue": .5},
              {"wheeze": .9, "breathless": .8, "cough": .6},
              {"sneezing": .9, "runny nose": .9, "itchy eyes": .7},
              {"nausea": .8, "vomiting": .7, "diarrhoea": .8, "stomach ache": .7, "fever": .3},
              {"thirst": .8, "frequent urination": .8, "weight loss": .5, "fatigue": .7},
              {"headache": .95, "nausea": .5}]
rng = np.random.default_rng(1)
P = np.array([[c.get(w, 0.03) for w in WORDS] for c in CONDITIONS])     # 3% noise for every other word
notes = []
while len(notes) < 600:
    cs = rng.choice(6, size=1 + (rng.random() < 0.2), replace=False)    # one condition, sometimes two
    x = rng.random(len(WORDS)) < P[cs].max(axis=0)
    if x.any():
        notes.append(x)
X = np.array(notes)
m = len(WORDS)


class TMAutoEncoder:
    """Shared pool of rules; W[k, j] = signed integer weight of rule j for output word k."""

    def __init__(self, n_rules=24, T=24, s=4.0, depth=100, u=2, seed=7):
        self.T, self.s, self.depth, self.half, self.u = T, s, depth, depth // 2, u
        self.rng = np.random.default_rng(seed)
        self.memory = np.full((n_rules, m), depth // 2)                     # top of Forgotten
        self.W = self.rng.choice([-1, 1], size=(m, n_rules))

    def fires(self, x, k, training=False):
        x = x.copy(); x[k] = True                                           # mask: the hidden word counts as True
        memorized = self.memory > self.half
        ok = ~np.any(memorized & ~x, axis=1)
        return ok if training else ok & memorized.any(axis=1)               # empty rules: silent when predicting

    def predict(self, x, k):
        return int(self.W[k] @ self.fires(x, k) >= 0)                       # x̂_k = 1 if v ≥ 0

    def update(self, k, x, q, active):
        C = self.fires(x, k, training=True) & active                        # inactive rules sit this round out
        v = np.clip(self.W[k] @ C, -self.T, self.T)
        p = (self.T - v) / (2 * self.T) if q else (self.T + v) / (2 * self.T)
        picked = (self.rng.random(len(C)) <= p) & active
        recog = (self.W[k] >= 0) == bool(q)                                 # rules that want to fire for this q
        u = self.rng.random(self.memory.shape) <= 1 / self.s                # Forget Value 1/s
        not_k = np.arange(m) != k                                           # the masked word is frozen
        mem = self.memory
        rec = (picked & recog & C)[:, None] & not_k                         # Recognize: memorize present, forget absent
        mem += rec & x & (mem < self.depth)
        mem -= rec & ~x & u & (mem > 1)
        era = (picked & recog & ~C)[:, None] & not_k                        # Erase: forget everything
        mem -= era & u & (mem > 1)
        rej = (picked & ~recog & C)[:, None] & not_k                        # Reject: memorize absent forgotten words
        mem += rej & ~x & (mem <= self.half)
        self.W[k] += np.where(picked & C, 1 if q else -1, 0)                # weights follow the target value

    def fit(self, X, rounds=300):
        for _ in range(rounds):
            # TMU detail: a rule whose weights are already large (on average) is rarely updated
            avg = np.abs(self.W).mean(axis=0)
            active = self.rng.random(len(avg)) <= (self.T - np.minimum(avg, self.T)) / self.T
            for k in self.rng.permutation(m):
                q = self.rng.random() < 0.5                                 # balanced: present or absent
                pool = np.flatnonzero(X[:, k] == q)
                x = X[self.rng.choice(pool, size=self.u)].any(axis=0)       # OR of u notes
                self.update(k, x, q, active)
        return self

    def rule(self, j, k):
        lits = [f"[{w}]" if i == k else w for i, (w, mm) in enumerate(zip(WORDS, self.memory[j])) if mm > self.half]
        return " AND ".join(lits) or "(empty)"


tm = TMAutoEncoder().fit(X)
E = np.clip(tm.W, 0, tm.T)                                                 # the embedding: one row per word
unit = E / np.linalg.norm(E, axis=1, keepdims=True)
S = unit @ unit.T
for w in ["cough", "fatigue", "sneezing"]:
    k = WORDS.index(w)
    best = np.argsort(-tm.W[k])[:3]
    print(f"{w}: " + " | ".join(f"{tm.W[k, j]:+d} IF {tm.rule(j, k)}" for j in best))
    print("   nearest:", ", ".join(f"{WORDS[b]} {S[k, b]:.2f}" for b in np.argsort(-S[k])[1:4]))
```

```text
cough: +12 IF fever AND [cough] AND sputum AND breathless | +12 IF breathless AND wheeze | +10 IF sputum AND fatigue
   nearest: sputum 0.87, fever 0.84, breathless 0.82
fatigue: +10 IF thirst | +9 IF thirst AND frequent urination AND [fatigue] | +6 IF sputum AND [fatigue]
   nearest: frequent urination 0.72, thirst 0.68, weight loss 0.63
sneezing: +16 IF [sneezing] AND runny nose AND itchy eyes | +14 IF runny nose AND itchy eyes | +10 IF [sneezing] AND runny nose AND itchy eyes
   nearest: runny nose 0.86, itchy eyes 0.80, fatigue 0.06
```

The square brackets mark the masked word. The same story appears with different random numbers: *cough* is described by a pneumonia rule and an asthma rule, *fatigue* by diabetes rules and a pneumonia rule. *Sneezing* has two copies of the same rule, which act like one rule with a bigger weight. It runs in under two seconds.

| Piece | What it does | Book term (research term) |
|---|---|---|
| Hidden word $x_k$ | the output to predict; counts as True and is frozen | (masked input variable) |
| Shared pool of rules | every output word uses the same rules | rules (clauses of a coalesced TM) |
| Weight $w_{kj}$ | signed integer; its sign says which way rule $j$ votes for word $k$ | weight (clause weight) |
| Vote sum $v_k$ | $\sum_j w_{kj} C_j$; word present if $v_k \ge 0$ | vote sum (weighted clause sum) |
| Voting Margin $T$ | feedback probability $(T \mp v_k)/2T$ | Voting Margin (target $T$) |
| Coin flip $q$, accumulation $u$ | balanced examples built from $u$ notes ORed together | (output balancing, accumulation) |
| Embedding $\operatorname{clip}(W, 0, T)$ | a word = its positive weights on readable rules | (weighted logical embedding $E$) |

---

## Common misconceptions

!!! warning "\"An autoencoder needs labels, like any Tsetlin machine.\""
    No. The targets come from the data: each example asks whether a word that the notes do or don't contain was there. The only "label" is the hidden word itself. That is why it can learn from unlabelled text.

!!! warning "\"Each word gets its own rules.\""
    All words share **one** pool of rules. What a word owns is a row of weights. In our run, R7 *"fever AND cough AND sputum"* has weight +5 for *cough* (where *cough* is masked) and +11 for *fatigue*, where all three words are real context.

!!! warning "\"The embedding is just co-occurrence counts in disguise.\""
    It is built from co-occurrence, like every embedding, and on our toy data PPMI counts give similar neighbours (16 of 18, against 17.5 on average for the autoencoder). But each coordinate of the TM embedding is an AND-rule over several words, such as *"breathless AND wheeze"*. That separates the asthma context of *cough* from its pneumonia context, and you can read it.

!!! warning "\"A bigger accumulation u is always better.\""
    Not on a small vocabulary. With 18 words, $u = 5$ notes already switch on about 9 to 11 context words per example, so rules get long and vague: our hidden-word balanced accuracy fell from 76.2% ($u = 1$) to 53.8% ($u = 5$). The paper's $u = 25$ suits a corpus with tens of thousands of words, where most words are rare.

!!! warning "\"Logical embeddings beat neural embeddings.\""
    The paper's results are mixed: slightly behind Word2Vec and GloVe on word-similarity correlations, close to GloVe as input to a text classifier. The selling points are sparsity and readability, not accuracy.

---

## Check yourself

??? question "1. The note is {fever, sputum, chest pain} and the hidden word is cough. Rule R7 is \"fever AND cough AND sputum\". Does it fire?"
    **Yes.** The hidden word counts as True, so for predicting *cough* the rule only needs *fever* and *sputum*, which are both there. In our trained model R7 has weight +5 for *cough* and R16 *"fever AND sputum"* has +12, so $v = +17$ and *cough* is predicted present.

??? question "2. T = 24, the hidden word is present (q = 1) and v = −8. What is the feedback probability? What happens to a picked rule with weight −3 that fires?"
    $p = (24 - (-8))/48 = 32/48 \approx 0.67$. The rule's weight is negative (it votes "absent"), but the word is present, so its sign doesn't match: it gets **Reject Feedback** (an absent, Forgotten word is pushed towards its memory, to stop it firing on such notes), and because it fired, its weight moves towards $q = 1$: from −3 to −2.

??? question "3. Why flip a coin for the target value instead of using notes as they come?"
    Most words are absent from most notes (*chest pain* is in only 12.7% of ours). Unbalanced, the easiest answer would be "absent" every time, the vote sums would sit near $-T$, and there would be little feedback to learn the contexts in which the word *does* appear. The coin flip makes present and absent equally common for every word, rare or not.

??? question "4. Two words have cosine similarity 0 between their embeddings. What does that mean in terms of rules?"
    No rule has a positive weight for both words: they share no context rule. In our run *fatigue* and *itchy eyes* score 0.01, almost no overlap. Since every coordinate is a rule, you can list the rules each word uses and see exactly what they don't share.

??? question "5. Why is the hidden word frozen during feedback, and not just masked?"
    The mask makes it True, so it would look like a present word in every example. Recognize Feedback would then keep memorizing it into the rules for this output, and the rule would end up "predicting" the word from itself. Freezing its memory position keeps the rule's content about the **other** words. (The word can still enter a rule through feedback for other outputs, where it is real context.)

---

## Further reading

- Bimal Bhattarai, Ole-Christoffer Granmo, Lei Jiao, Rohan Yadav & Jivitesh Sharma, [*Tsetlin Machine Embedding: Representing Words Using Logical Expressions*](https://aclanthology.org/2024.findings-eacl.103/), Findings of EACL 2024, pp. 1512–1522 (preprint [arXiv:2301.00709](https://arxiv.org/abs/2301.00709)). The Tsetlin machine autoencoder, masking and freezing, the balanced sampling with accumulation (Algorithm 1), and the logical embeddings $E$ and $B$. This lesson follows it.
- Sondre Glimsdal & Ole-Christoffer Granmo, [*Coalesced Multi-Output Tsetlin Machines with Clause Sharing*](https://arxiv.org/abs/2108.07594) (2021). The shared pool of rules with signed weights per output that the autoencoder is built on.
- Ahmed K. Kadhim, Ole-Christoffer Granmo, Lei Jiao & Rishad Shafik, [*Exploring State Space and Reasoning by Elimination in Tsetlin Machines*](https://arxiv.org/abs/2407.09162) (2024). The autoencoder with negated words in the rules, and how $T$ and $s$ shape them.
- Ahmed K. Kadhim, Lei Jiao, Rishad Shafik & Ole-Christoffer Granmo, [*Omni TM-AE: A Scalable and Interpretable Embedding Model Using the Full Tsetlin Machine State Space*](https://arxiv.org/abs/2505.16386) (2025, preprint). Embeddings from the whole memory of each rule, in one training run.
- The [TMU library](https://github.com/cair/tmu) (CAIR, University of Agder): `TMAutoEncoder` in `tmu/models/autoencoder/autoencoder.py`, and the IMDb word-embedding demo in `examples/autoencoder/`, which learns which words keep company with targets such as *awful*, *brilliant* and *scary*.
- Ole-Christoffer Granmo, [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/) (free online). Chapter 6, "Auto-Encoding", is announced but not yet published.
- Zellig S. Harris, *Distributional Structure*, Word 10 (1954). The distributional hypothesis behind every word embedding.

**Next lessons:** [Confidence & composites](../tsetlin-machine-confidence/index.md): does a Tsetlin machine know when it is wrong, and how to combine several specialised machines.

<script src="viz/tsetlin-machine-autoencoder.js"></script>
