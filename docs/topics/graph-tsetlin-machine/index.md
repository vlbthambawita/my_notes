# Hierarchical and Graph Tsetlin Machines

!!! abstract "TL;DR"
    A standard Tsetlin machine reads a flat list of Boolean features, so it cannot say "an atom **with property A, bonded to** an atom with property B": the list says *what* is there, not *what is connected to what*. A **graph Tsetlin machine** takes a **graph** as input: **nodes** with **properties** (symbols such as "N" or "O"), joined by **edges** that may have **types** (such as "left" and "right"). Each rule is split into **parts, one per layer**. Part 0 checks a node's own properties, exactly like a convolutional rule checks a patch. Wherever a rule's part matches, the node **sends a message** (the rule's name) to its neighbours, and part 1 of every rule can then ask "did a neighbour send me R1?". More layers see further: after $D-1$ rounds of messages, a rule can describe a node's surroundings up to $D-1$ steps away, built from **rules over rules**. That is the **hierarchy**: small patterns (an O) feed bigger ones (a C next to an O), which feed bigger ones again. A rule is True for the graph if it matches at **some node** (OR over nodes), and learning is the same Recognize, Erase and Reject Feedback as before, applied at **one matching node, picked at random**. In our toy molecules, flat features and depth-1 rules stay at chance level (about 51%), while depth-2 rules reach **100%** and read like chemistry: "an N bonded to an O".

**Level:** intermediate–advanced · **Prerequisites:** [Convolutional Tsetlin machines](../convolutional-tsetlin-machine/index.md) (patches, the OR over positions, feedback on one random matching patch). It builds on [Classification by voting](../tsetlin-machine-classification/index.md) (the vote sum $v$ and the Voting Margin $T$). The math needs nothing beyond AND, OR and counting steps along a path.

!!! note "Beyond the book"
    Granmo's book [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/) announces Ch. 8, "Hierarchical Representation", and Ch. 9, "Graphs", but they are not published yet. This lesson is therefore based on the research papers listed in [Further reading](#further-reading), mainly Granmo et al. (2025), *The Tsetlin Machine Goes Deep*, and the `GraphTsetlinMachine` code it describes. We use the book's vocabulary (rules, Memorize/Forget, Recognize/Erase/Reject Feedback) and give the research term once, in brackets. All experiments are our own toy runs; numbers that come from the papers are marked as quotes.

---

## Why should I care?

Much of the data in medicine is about **relations**, not just lists of facts:

- **Molecules.** Whether a drug is toxic often depends on which atoms are *bonded* to which, not on how many atoms of each element it contains. Two molecules with exactly the same atoms can behave very differently.
- **Sequences.** In DNA or in an ECG, *order* matters: "three A's in a row" is not the same as "three A's somewhere".
- **Patient records.** A hospital record links a patient to tests, diagnoses, drugs and notes. The README of the `GraphTsetlinMachine` code sketches exactly this use case: nodes for ECGs and clinical notes, and edges of different types, such as *Measurement* edges from tests to a patient and *Condition* edges from diseases to patients.

A flat feature list loses the links. You could add one feature per possible link ("atom 3 is bonded to atom 5"), but then a rule must name exact slots, and you need one rule per slot combination, the same problem the [convolutional lesson](../convolutional-tsetlin-machine/index.md) had with positions. A graph Tsetlin machine solves it the same way convolution did: rules look at **one node at a time**, and the node's **neighbours tell it what they are** through messages.

---

## Step 1 — What flat features cannot say

Take two small toy "molecules" (tree-shaped, made up for this lesson; not real chemistry). Both have **four carbons, one nitrogen and one oxygen**. In molecule A, the N is bonded to the O. In molecule B, the N and the O sit three bonds apart.

A standard Tsetlin machine needs a fixed list of Boolean features. The natural ones are "contains N", "contains at least 2 C", and so on (thermometer-coded counts, as in the [classification lesson](../tsetlin-machine-classification/index.md)). For A and B these features are **identical**, so every rule, and therefore the whole machine, must give A and B the **same** answer. If the question is "is the N bonded to the O?", it is bound to be wrong on one of them.

**Try it:** click atoms to change their element, or press *Random pair*.

<div id="viz-flat" class="viz"><div class="viz-title">🧪 Interactive: same atoms, different bonds</div></div>

!!! tip "What to notice"
    - In the starting pair, the counts are C × 4, N × 1, O × 1 for **both** molecules, yet the graph rule "an N with an O neighbour" is True for A and False for B.
    - Change an atom and the counts may differ, but then a flat rule can only tell the molecules apart by **counting**, never by looking at the bonds.
    - You *could* give a standard machine one feature per pair of atom slots ("slot 3 is N, slot 5 is O, and they are bonded"). With up to 8 atoms, "N bonded to O" alone then needs a rule for each ordered pair of slots: $8 \times 7 = 56$ rules, and the slot numbers mean nothing chemically. The graph rule says it **once**.

---

## Step 2 — Inputs as graphs: nodes, properties and typed edges

A graph Tsetlin machine takes as input a graph (Granmo et al. 2025, §2):

- **Nodes**, each with a set of **properties**: symbols such as "C", "A", "pixel (3, 5) is white" or "the word *pain*". A property is a Boolean feature of the node, so the node's literals are its properties and their negations, exactly as before.
- **Edges** between nodes, each with a **type**: "bond", "left", "right", "Measurement", … Edges are directed; an undirected bond is two edges, one each way.

The same input format covers many kinds of data:

| Data | Nodes | Properties | Edges (types) |
|---|---|---|---|
| Our toy molecules | atoms | the element: C, N or O | bonds (one type) |
| A sequence such as `BAAAE` | positions | the letter | to the neighbour on the left / right (two types) |
| An image, split into patches | patches | the pixels and the patch position | **none** |
| A tabular patient record | one node | all the features | none |
| A hospital record | patient, tests, notes, diagnoses | values, words, codes | *Measurement*, *Condition*, … |

The last two rows show that the earlier machines are special cases. A graph with **one node** is a standard Tsetlin machine. A graph of **patches without edges** is a convolutional Tsetlin machine: the paper checks this and reports test accuracies close to those of a convolutional machine on MNIST (98.42% vs 98.93%) and Fashion-MNIST (89.49% vs 91.05%) (Granmo et al. 2025, Table 3, quoted).

!!! note "Hypervectors: how the real code stores symbols"
    With thousands of possible symbols (all the words of a language, say), one Boolean feature per symbol becomes large. The `GraphTsetlinMachine` code therefore stores each node's properties in a **hypervector**: a fixed-length bit vector in which every symbol is given a few random bit positions, and a node's bits are the OR of its symbols' bits. Its README example uses vectors of 32 bits with 2 bits per symbol, which gives $32 \cdot 31 / 2 = 496$ different bit pairs (quoted from the README). Rules then memorize *bits*, and a learned rule is read back by checking which symbols' bits it contains. Two symbols can share a bit by accident, which adds a little noise. This lesson skips hypervectors and gives **every symbol its own feature** (the code also has such a one-hot option). Nothing else changes.

!!! tip "What to notice"
    - Edge **types** matter. In the sequence `BAAAE`, a node that only knows "a neighbour is an A" can't tell left from right; with *left* and *right* edges it can (the README makes this remark for its sequence demo).
    - The graph can have **any size**. A rule never names a node number, so molecules of 6 atoms and of 8 atoms are handled by the same rules.

---

## Step 3 — Rules that look at one node: part 0

Start with the simplest graph rule: an AND of **property literals**, checked at **every node**, with an OR over the nodes. "O" matches every oxygen atom; "¬C" matches every atom that isn't carbon. The rule is True for the molecule if it matches at **one node or more**.

This is word for word the convolutional machine, with **nodes in place of patches**:

$$
C_j(\text{graph}) \;=\; \bigvee_{v \in \text{nodes}} \; \bigwedge_{k \in I_j} l_k^{(v)} ,
$$

where $I_j$ is the set of literals rule $C_j$ has memorized and $l_k^{(v)}$ is the truth value of literal $l_k$ at node $v$. A rule like this can say "there is an N" and "there is an O", but not "the N is next to the O": it sees one node at a time, and the OR throws away *which* nodes matched. That is why, in Step 7, a graph machine with only this one part (depth 1) is stuck at chance level on the bonding task, just like the flat machine.

In the paper's terms, this is the **layer-0 component** $C^0_j$ of a clause (Granmo et al. 2025, §2). To see more than one node, the rule needs more parts, and the parts need a way to talk.

!!! tip "What to notice"
    - Part 0 of a rule is a **pattern detector for single nodes**: "an O", "an N", "an atom that is N or O".
    - On its own it is a convolutional machine whose "patches" are nodes. Everything new in this lesson is about what happens **between** nodes.

---

## Step 4 — Messages: rules tell the neighbours what they found

Here is the key idea of the graph Tsetlin machine (Granmo et al. 2025, §2, "Messages" and "Message Submission"):

1. Evaluate **part 0** of every rule at every node.
2. Wherever part 0 of rule $R_k$ matches, the node **sends the message "$R_k$"** along all its outgoing edges (bound to the edge's type, when there are several types).
3. Every node collects what it receives in its **inbox**. The inbox gives the node new Boolean features: "**a neighbour sent $R_k$**", one per rule (and per edge type).
4. **Part 1** of each rule is an AND of these **message literals**. A rule matches at a node in layer 1 only if part 0 **and** part 1 both match there.
5. Finally, OR over the nodes, and vote, exactly as before.

<div class="viz diagram">
<script type="application/json">
{
  "title": "📨 One round of message passing",
  "cols": 3,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "props", "col": 0, "row": 0, "icon": "⚛️", "label": "Node properties", "sub": "C, N, O … per node", "role": "data", "note": "Each node's own Boolean features and their negations."},
    {"id": "p0", "col": 1, "row": 0, "icon": "🔍", "label": "Part 0 of each rule", "sub": "match at every node", "role": "model", "note": "An AND of property literals, checked node by node."},
    {"id": "send", "col": 2, "row": 0, "icon": "📨", "label": "Send messages", "sub": "R<sub>k</sub> along every edge", "role": "model", "note": "Where part 0 of rule R_k matches, the node sends 'R_k' to its neighbours (bound to the edge type)."},
    {"id": "inbox", "col": 2, "row": 1, "icon": "📥", "label": "Inbox", "sub": "\"a neighbour sent R<sub>k</sub>\"", "role": "data", "note": "One new Boolean feature per rule (and edge type): the message literals."},
    {"id": "p1", "col": 1, "row": 1, "icon": "🔍", "label": "Part 1 of each rule", "sub": "parts 0 AND 1 must match", "role": "model", "note": "An AND of message literals, checked at each node together with part 0."},
    {"id": "or", "col": 0, "row": 1, "icon": "∨", "label": "OR over nodes", "sub": "then vote as before", "role": "loss", "note": "A rule is True for the graph if all its parts match at one node or more."}
  ],
  "edges": [
    {"from": "props", "to": "p0"},
    {"from": "p0", "to": "send"},
    {"from": "send", "to": "inbox", "label": "along edges"},
    {"from": "inbox", "to": "p1"},
    {"from": "p1", "to": "or"}
  ],
  "steps": [
    {"show": ["props"], "text": "Each node starts with its own <b>properties</b>: here, the element of an atom."},
    {"show": ["props>p0", "p0"], "text": "<b>Part 0</b> of every rule is checked at every node, using only that node's properties."},
    {"show": ["p0>send", "send"], "text": "Wherever part 0 of rule R<sub>k</sub> matches, the node <b>sends the message R<sub>k</sub></b> to its neighbours."},
    {"show": ["send>inbox", "inbox"], "text": "Each node's <b>inbox</b> collects the messages. They become new literals: \"a neighbour sent R<sub>k</sub>\"."},
    {"show": ["inbox>p1", "p1"], "text": "<b>Part 1</b> of every rule checks the inbox. A rule matches at a node only if parts 0 and 1 both match there. With more layers, the rules that matched send again, and part 2 reads the next inbox."},
    {"show": ["p1>or", "or"], "text": "A rule is True for the graph if it matches at <b>some node</b>. From here on, voting and classification are the same as in a standard Tsetlin machine."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"props": [0, 0], "p0": [1, 0], "send": [1, 1], "inbox": [1, 2], "p1": [0, 2], "or": [0, 1]}}
}
</script>
</div>

So the "bonded" rule from Step 1 is two rules working together:

$$
\begin{aligned}
R_1 &: \quad \text{O} \\
R_2 &: \quad \underbrace{\text{N}}_{\text{part 0}} \;\wedge\; \underbrace{\text{a neighbour sent } R_1}_{\text{part 1}} .
\end{aligned}
$$

$R_2$ reads as "an N with an O neighbour". Its part 1 does not mention O at all; it mentions **another rule**. That is what the paper calls a **deep** or **nested clause**: rules built from rules. To read it, you trace each message back to the rule that sent it (the paper does exactly this in its Table 1).

**Try it:** step through the evaluation. The widget starts on the N atom (dashed ring); its inbox is shown in the readout.

<div id="viz-messages" class="viz"><div class="viz-title">📨 Interactive: rules send messages to their neighbours</div></div>

!!! tip "What to notice"
    - **Layer 0:** R1 matches at the O; R2 and R4 match at the N (their part 0 is "N"); R3 matches at every C.
    - **Round 1:** only the O's neighbour (the C in the middle) receives R1. The N's inbox gets R3 (from its C neighbours) and nothing else, so **R2 fails**: this N is not bonded to an O.
    - **Layer 1:** R3 ("a C next to an O") now matches only at the middle C. **Round 2** carries that fact one bond further, so in **layer 2**, R4 ("an N next to a C that is next to an O") matches at the N.
    - **Faded atoms** are the ones that cannot have influenced the selected atom yet. After layer 0 the N sees only itself, after layer 1 its neighbours, after layer 2 everything within two bonds.
    - Click the O's neighbour twice to turn it into an N: now R2 fires in layer 1, and R4 no longer does.

---

## Step 5 — Depth is hierarchy: seeing further, one round at a time

With $D$ layers (the paper's **depth** $D$), a rule has parts $0, 1, \dots, D-1$, and there are $D - 1$ rounds of messages. In round $i$, a node sends $R_k$ if rule $R_k$ matched **all its parts up to layer $i-1$** at that node (Granmo et al. 2025, Algorithm in §2 and Appendix A.3). So messages get richer every round:

| Round | A message "$R_k$" from a neighbour means … | What a rule can see from a node |
|---|---|---|
| — (part 0) | — | the node itself |
| 1 | the neighbour matches part 0 of $R_k$ | its neighbours |
| 2 | the neighbour matches parts 0–1 of $R_k$, which already looked at **its** neighbours | everything up to 2 edges away |
| $D-1$ | … | everything up to $D-1$ edges away |

This growing **receptive field** is the same idea as stacking convolution layers in a convolutional neural network, and it is where the hierarchy comes from: R1 (an O) is used by R3 (a C next to an O), which is used by R4 (an N next to a C next to an O). Each level is a readable rule, and it can be traced back to atoms.

Two details make the rules shorter than you might expect:

- **The receptive field is a radius, not a path length.** "N and O two bonds apart" (N–X–O) needs only depth 2, if the rule is centred on the middle atom: "any atom, with a neighbour that sent R(N) and a neighbour that sent R(O)".
- **Reasoning by elimination.** A rule may describe a node by what it is *not*. In the paper's Figure 1, a part that memorized ¬1 ∧ ¬3 ∧ ¬5 ∧ ¬7 ∧ ¬9 matches every even number. In our molecules (one N and one O each), "an N-or-O atom next to an N-or-O atom" can only be the N–O bond. The paper argues that this kind of sharing, rules reusing other rules, lets the machine recognise sub-graph patterns "with exponentially fewer clauses" (quoted from its abstract).

!!! note "The paper's sequence example"
    Granmo et al. (2025, §3.1) classify 5-letter sequences by whether they contain three A's in a row. Each letter is a node, with *left* and *right* edges. A depth-2 machine with only **four** rules reaches 100% test accuracy after four epochs (quoted), and its rules read as, for example, "an A whose left and right neighbours match the A-rules": three A's centred on the middle one.

!!! tip "What to notice"
    - **More layers = a wider view**, built from rules about neighbours of neighbours. The paper reports that depth matters on real data: on IMDB sentiment, 86.43% test accuracy at depth 1 vs 88.15% at depth 2, and on 5-class viral genome sequences, 59.81% at depth 1 vs 95.14% at depth 2 (Granmo et al. 2025, Tables 5 and 8, quoted).
    - Depth also makes learning **harder**: a deep rule only works once the rules it relies on have learned something useful. Step 7 shows this in our toy runs.

---

## Step 6 — Learning: same feedback, at one matching node

Nothing in the learning rule is new; it is the convolutional machine's procedure with **nodes in place of patches** (Granmo et al. 2025, §2, step 5; the code picks the node at random among the nodes where the clause is True):

- Compute every rule's output for the graph (all layers, then OR over nodes), and the vote sum $v$ for the true class, clipped to $[-T, T]$.
- Select each rule for feedback with probability $(T - v)/2T$, as in the [classification lesson](../tsetlin-machine-classification/index.md).
- A selected rule of the true class that is **True** gets **Recognize Feedback** at **one node where it matched, picked at random**. A selected rule of the other class that is True gets **Reject Feedback** at one random matching node. A selected rule of the true class that is **False** gets **Erase Feedback** (no node needed).
- At the chosen node, the rule's literal vector is **everything the node had**: its properties (for part 0) and its inboxes (for parts 1, 2, …). Message literals are simply more literals, learned with the same memory, Memorize Value and Forget Value.

One thing *is* new in spirit: a message literal means "a neighbour sent $R_k$", and what $R_k$ matches **changes while $R_k$ learns**. Rules are learning on top of each other's moving results. It works because a rule that is useful to others keeps being reinforced, so its meaning settles, and the rules built on it settle with it.

!!! warning "Our simplifications (what this lesson's code does differently from the paper)"
    - **One feature per symbol**, no hypervectors (Step 2), and one edge type (bonds).
    - **Votes for and against** with equal weight, as in the [classification lesson](../tsetlin-machine-classification/index.md). The paper's machine is built on the Coalesced Tsetlin machine (Glimsdal & Granmo 2021): one shared pool of rules, each with a learned integer weight per class.
    - **Only positive message literals** ("a neighbour sent $R_k$"). The paper's rules can also use negated ones ("no neighbour sent $R_k$"; its Table 1 shows several). In our small runs with negated message literals, Reject Feedback quickly taught rules "no neighbour sent $R_k$" for some $R_k$ that matches everywhere. That literal is False at every node, so the rule never fired again, and the machine never got above 51.0% test accuracy (seed 1, the Step 7 task, 15 epochs). With thousands of rules and the full update scheme this is presumably not a problem in the paper, but we could not check it at toy scale, so our toy drops the negations.
    - Small and slow: 8 rules per class, plain Python and JavaScript, no GPU. The paper's code runs on GPUs with thousands of rules.

!!! tip "What to notice"
    - The only new ingredient compared with the convolutional machine is **where the literals come from**: a node's own properties *and* its inbox. "Pick one matching node at random" is the convolutional machine's "pick one matching patch at random".
    - Erase Feedback lowers message literals too, so a rule that fails everywhere loosens both what it asks of the node and what it asks of the neighbours.

---

## Step 7 — Training a graph Tsetlin machine on toy molecules

The task: random tree-shaped molecules of 6 to 8 atoms, all carbon except **exactly one N and one O**, and the question "**is the N bonded to the O?**". Every molecule has the same element counts (apart from the number of carbons), so flat features carry almost no information. There are 300 training and 300 test molecules, half of each class. The machine has 8 rules per class, Voting Margin $T = 4$, Forget Value 0.8 (Memorize Value 0.2), and a memory depth of 10 positions per literal, as in the convolutional lesson. Everything is seeded; the widget runs a line-by-line port of our Python check script with the same random numbers, so it shows exactly the numbers below.

**Try it:** press *Train to 15 epochs* for each model. Then click rules in the list to see where they match.

<div id="viz-train" class="viz"><div class="viz-title">🧠 Interactive: train a graph Tsetlin machine on "is the N bonded to the O?"</div></div>

!!! tip "What to notice"
    - **Flat features fail.** The standard machine sees thermometer-coded element counts. It ends at **51.0%** test accuracy (seed 1), and **50.9%** on average over seeds 1 to 10: a coin flip, because the counts say nothing about bonds.
    - **Depth 1 fails too.** Rules that see one node at a time end at **51.0%** (seed 1; **51.8%** on average). This is the convolutional machine of Step 3.
    - **Depth 2 solves it.** With one round of messages, the machine reaches **100%** test accuracy after the first epoch (seed 1), and every seed from 1 to 10 is at 100% after 15 epochs (reached within 1 to 4 epochs). Depth 3 also ends at 100% for all ten seeds.
    - **The rules read like chemistry.** With seed 1 and depth 2, all eight "yes" rules fire on all 147 test molecules where the N is bonded to the O, and on none of the other 153. Their descriptions say it in different words, for example "N, next to [O], next to [N or O]", "O, next to [N], next to [N or O]", and, by elimination, "N or O, next to [N or O]". One "no" rule found a pattern that rules out a bond: "C, next to [N], next to [O], next to [N or O]", i.e. N–C–O, which fires on 78 of the 153 "no" molecules and on none of the "yes" ones.
    - **A harder question exposes the learning problem.** Switch the task to "N within 2 bonds of O?". Depth 2 can express the answer (Step 5), and with seed 1 it does reach **100%** test accuracy at epoch 5, but then loses it and ends at **50.7%**. Over seeds 1 to 10 the final accuracy averages **56.2%** (49.3% to 69.3%); three seeds touched 100% at some epoch. Depth 3 does no better (**49.6%** on average). The "no" class is the hard part: "the N has no O within 2 bonds" is a statement about **every** path, and a rule that is an OR over nodes can only find something that **exists**. The "yes" rules have to carry the whole decision, and in our small runs they don't settle. Bigger machines, weighted rules and more data are what the paper uses.

---

## Step 8 — Another kind of depth: stacking Tsetlin machines

Message passing builds a hierarchy **over the graph**: rules about neighbours of neighbours. There is a second, older idea for hierarchy that works on ordinary flat inputs: **stack whole Tsetlin machines**, so that the rule outputs of one machine become the Boolean features of the next.

- Granmo's original Tsetlin machine paper (2018, §6.2) sketches a "Fully Connected Deep Tsetlin Machine": the clause outputs of one machine are the input of the next, and each layer is trained on its own, directly from the class label, similar to how deep belief networks were trained layer by layer. If one layer has learned $P \wedge \lnot Q$ and $\lnot P \wedge Q$, the next layer can combine them into larger formulas.
- Tarasyuk, Gorbenko, Yakovlev and Shafik (2024) built and tested such a **multi-layer Tsetlin machine**. They train the layers one after another: train layer 1 until its accuracy levels off, **freeze** its rules, then use their outputs as inputs for layer 2, and so on. On Digits-MNIST with 100 rules per class, they report validation accuracy 96.58% with one layer, 97.42% with two and 96.90% with five, while training accuracy keeps rising (97.72% to 99.93%). They conclude that two layers were best and that more layers overfit (quoted from their Tables II and III).

| | Stacked (multi-layer) Tsetlin machine | Graph Tsetlin machine |
|---|---|---|
| Input | a flat feature list | a graph of nodes, properties and typed edges |
| What a higher layer reads | the outputs of **all** rules of the layer below, for the whole input | messages from **neighbouring nodes**: which rules matched next door |
| Hierarchy means | formulas made of formulas | patterns made of neighbouring patterns, wider each layer |
| Training | layer by layer, earlier layers frozen (Tarasyuk et al.) | all parts of all rules at once, with the usual feedback |

!!! tip "What to notice"
    - Both kinds of depth build **rules over rules**, and both keep every level readable: a higher-level rule is an AND of lower-level rules, each of which is an AND of literals.
    - Both papers find that depth helps up to a point, and both show that it is not free: the multi-layer machine overfits beyond two layers, and our toy graph machine struggles to learn deeper patterns.

---

## Putting it all together

The complete learning loop is the [convolutional](../convolutional-tsetlin-machine/index.md) loop with layers of messages in the middle.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 How a graph Tsetlin machine learns",
  "cols": 3,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "obs", "col": 0, "row": 0, "icon": "🕸️", "label": "Observe a graph and its class", "role": "data", "note": "Nodes with properties, edges with types."},
    {"id": "layers", "col": 1, "row": 0, "icon": "📨", "label": "Evaluate layer by layer", "sub": "parts, then messages", "role": "model", "note": "Part 0 on properties; send messages; part 1 on the inbox; and so on for D layers."},
    {"id": "or", "col": 2, "row": 0, "icon": "∨", "label": "OR over nodes", "sub": "one output per rule", "role": "model", "note": "A rule is True for the graph if all its parts match at some node."},
    {"id": "vote", "col": 2, "row": 1, "icon": "🗳️", "label": "Vote sum v", "sub": "clipped to [−T, T]", "role": "loss", "note": "Rules of the true class that fire add 1; rules of the other class that fire subtract 1."},
    {"id": "sel", "col": 1, "row": 1, "icon": "🎲", "label": "Select rules", "sub": "probability (T − v) / 2T", "role": "loss", "note": "Far from the Voting Margin: many rules learn. At or beyond it: none do."},
    {"id": "fb", "col": 0, "row": 1, "icon": "✏️", "label": "Feedback", "sub": "at 1 random matching node", "role": "opt", "note": "Recognize / Reject use the chosen node's properties and inboxes. Erase needs no node."}
  ],
  "edges": [
    {"from": "obs", "to": "layers"},
    {"from": "layers", "to": "or"},
    {"from": "or", "to": "vote"},
    {"from": "vote", "to": "sel"},
    {"from": "sel", "to": "fb"},
    {"from": "fb", "to": "obs", "label": "next graph"}
  ],
  "steps": [
    {"show": ["obs"], "text": "Take a training graph and its class: here, a molecule and \"bonded or not\"."},
    {"show": ["obs>layers", "layers"], "text": "<b>New:</b> evaluate the rules layer by layer. Part 0 checks each node's properties; matching parts send messages to the neighbours; the next part checks the inbox."},
    {"show": ["layers>or", "or"], "text": "As in convolution: a rule is True for the graph if it matches at <b>some node</b>."},
    {"show": ["or>vote", "vote"], "text": "As before: votes for the true class minus votes against, clipped to the Voting Margin T."},
    {"show": ["vote>sel", "sel"], "text": "As before: each rule is selected for feedback with probability (T − v) / 2T."},
    {"show": ["sel>fb", "fb"], "text": "<b>New detail:</b> Recognize and Reject Feedback use <b>one matching node, picked at random</b>, and update the property literals and the message literals from that node's inboxes."},
    {"show": ["fb>obs", "obs"], "text": "Repeat for the next graph, epoch after epoch."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"obs": [0, 0], "layers": [1, 0], "or": [1, 1], "vote": [1, 2], "sel": [0, 2], "fb": [0, 1]}}
}
</script>
</div>

Here is a complete (simplified) graph Tsetlin machine in numpy, on the Step 7 task. It uses numpy's random numbers, so its exact numbers differ from the widget's.

```python
import numpy as np

rng = np.random.default_rng(0)
ELEMENTS = "CNO"


def make_molecule():
    """A random tree of 6-8 atoms: all C except one N and one O; at most 3 bonds per atom."""
    n = rng.integers(6, 9)
    nbr = [[] for _ in range(n)]
    for k in range(1, n):
        p = rng.integers(0, k)
        while len(nbr[p]) >= 3:
            p = rng.integers(0, k)
        nbr[p].append(k)
        nbr[k].append(p)
    el = np.zeros(n, int)
    el[rng.choice(n, 2, replace=False)] = [1, 2]                     # one N, one O
    return el, nbr


def make_data(size):
    """Balanced data: label 1 if the N is bonded to the O."""
    X, y = [], []
    for i in range(size):
        while True:
            el, nbr = make_molecule()
            bonded = any(el[u] == 2 for u in nbr[int(np.flatnonzero(el == 1)[0])])
            if bonded == i % 2:
                break
        props = np.eye(3, dtype=bool)[el]                              # one-hot element per atom
        X.append((np.hstack([props, ~props]), nbr))
        y.append(i % 2)
    return X, np.array(y)


class GraphTM:
    def __init__(self, rules_per_class=8, depth=2, T=4, forget=0.8, mem=10):
        self.m, self.D, self.T, self.F, self.mem = 2 * rules_per_class, depth, T, forget, mem
        self.cls = np.repeat([0, 1], rules_per_class)
        self.L = [6] + [self.m] * (depth - 1)                         # literals per layer: properties, then messages
        self.state = [np.full((self.m, L), mem // 2) for L in self.L]  # start just below Memorized

    def evaluate(self, graph):
        """Literals of every node, layer by layer, and M[j, v] = rule j matches (all layers) at node v."""
        lits, nbr = [graph[0]], graph[1]
        M = ~((self.state[0] > self.mem // 2)[:, None, :] & ~lits[0][None]).any(2)
        for i in range(1, self.D):
            inbox = np.array([M[:, u].any(1) if u else np.zeros(self.m, bool) for u in nbr])  # a neighbour sent R_k
            lits.append(inbox)
            M = M & ~((self.state[i] > self.mem // 2)[:, None, :] & ~inbox[None]).any(2)
        return lits, M

    def predict(self, graph):
        fires = self.evaluate(graph)[1].any(1)                             # OR over nodes
        return 0 if fires[self.cls == 0].sum() >= fires[self.cls == 1].sum() else 1

    def update(self, graph, y):
        lits, M = self.evaluate(graph)
        fires = M.any(1)
        v = np.clip(fires[self.cls == y].sum() - fires[self.cls != y].sum(), -self.T, self.T)
        for j in np.flatnonzero(rng.random(self.m) <= (self.T - v) / (2 * self.T)):
            node = rng.choice(np.flatnonzero(M[j])) if fires[j] else None  # ONE matching node, at random
            for i, s in enumerate(self.state):
                if node is None:
                    if self.cls[j] == y:                                   # Erase Feedback
                        s[j] -= (rng.random(self.L[i]) <= self.F) & (s[j] > 1)
                    continue
                p = lits[i][node]
                if self.cls[j] == y:                                       # Recognize Feedback
                    s[j] += p & (rng.random(self.L[i]) <= 1 - self.F) & (s[j] < self.mem)
                    s[j] -= ~p & (rng.random(self.L[i]) <= self.F) & (s[j] > 1)
                else:                                                      # Reject Feedback
                    s[j] += ~p & (s[j] <= self.mem // 2)


Xtr, ytr = make_data(300)
Xte, yte = make_data(300)
for depth in (1, 2):
    tm = GraphTM(depth=depth)
    for epoch in range(15):
        for i in rng.permutation(len(Xtr)):
            tm.update(Xtr[i], ytr[i])
    acc = np.mean([tm.predict(g) == y for g, y in zip(Xte, yte)])
    print(f"depth {depth}: test accuracy {acc:.3f}")


def atom(j):
    """Which elements rule j's layer-0 part allows (x_e memorized: must be e; NOT x_e memorized: must not be e)."""
    mem = tm.state[0][j] > tm.mem // 2
    ok = [e for e in range(3) if not mem[3 + e] and not any(mem[q] for q in range(3) if q != e)]
    return "any" if len(ok) == 3 else "/".join(ELEMENTS[e] for e in ok) if ok else "nothing"


for j in range(tm.m):                                                      # read the rules back to atoms
    sent = sorted({atom(k) for k in np.flatnonzero(tm.state[1][j] > tm.mem // 2)})
    vote = "bonded" if tm.cls[j] else "not bonded"
    print(f"R{j + 1:<2} votes {vote:10}: atom {atom(j):5} next to " + (", ".join(f"[{t}]" for t in sent) or "anything"))
```

```text
depth 1: test accuracy 0.500
depth 2: test accuracy 1.000
R1  votes not bonded: atom C     next to anything
R2  votes not bonded: atom C     next to [N], [O]
R3  votes not bonded: atom any   next to [N]
R4  votes not bonded: atom nothing next to [any]
R5  votes not bonded: atom N     next to [N]
R6  votes not bonded: atom N     next to anything
R7  votes not bonded: atom O     next to anything
R8  votes not bonded: atom any   next to [N], [O]
R9  votes bonded    : atom N     next to [N/O]
R10 votes bonded    : atom N/O   next to [N/O], [O], [any]
R11 votes bonded    : atom O     next to [N], [N/O], [any]
R12 votes bonded    : atom any   next to [O]
R13 votes bonded    : atom any   next to anything
R14 votes bonded    : atom N/O   next to [N/O], [any]
R15 votes bonded    : atom any   next to [N/O], [O], [any]
R16 votes bonded    : atom N/O   next to [N/O], [O], [any]
```

Each message literal is printed as the part 0 of the rule that sends it ("next to [O]" = "a neighbour sent a rule whose part 0 is O"). Depth 1 is at chance level and depth 2 is perfect, as in the widget. The "bonded" rules again say "an N, or an N/O atom, next to an O or N/O atom". R2 is the N–C–O pattern that rules out a bond. Some rules are vague or empty ("any atom next to anything" fires on every molecule) and some can never fire (R4: "nothing"); the clean rules outvote them. To go deeper, give a rule a third state array and read the second inbox, which is built from `M` after layer 1, exactly as the loop in `evaluate` does.

---

## Common misconceptions

!!! warning "\"A graph Tsetlin machine is a graph neural network with Boolean weights.\""
    Both pass messages along edges, but what is passed is different. A graph neural network passes vectors of real numbers and mixes them with learned weights. A graph Tsetlin machine passes **rule names**: "rule $R_k$ matched here". Every message literal can be traced back to a readable rule about the neighbour.

!!! warning "\"More layers always help.\""
    More layers let a rule see further, but each layer adds rules that depend on other rules' moving results. In our toy runs, depth 2 solved "N bonded to O", while neither depth 2 nor depth 3 settled on "N within 2 bonds of O". The multi-layer Tsetlin machine paper also found that validation accuracy peaks at two layers and then drops (Tarasyuk et al. 2024).

!!! warning "\"To find a pattern two bonds away you need three layers.\""
    The receptive field is a **radius** around the node where the rule matches. A depth-2 rule centred on the middle atom sees both ends of N–X–O. You need more layers only for patterns that are wide in every direction from their centre.

!!! warning "\"Messages are extra input features you have to design.\""
    You only design the graph: the nodes, their properties and the edges. The message literals are created by the machine itself, one per rule (and edge type), and their meaning is learned along with the rules.

!!! warning "\"The convolutional Tsetlin machine is something different.\""
    It is the special case of a graph with no edges: one node per patch, with the patch's pixels and position as properties. The paper reports similar accuracy for the two on MNIST and Fashion-MNIST (Step 2).

---

## Check yourself

??? question "1. A molecule has 7 atoms. Molecule A has its N bonded to its O; molecule B has the same atoms but the N and O are 3 bonds apart. Why can no standard Tsetlin machine on element counts tell them apart?"
    The counts (and any thermometer features made from them) are the same for A and B, so every rule has the same truth value on both, every vote is the same, and so is the prediction. Bonds are not in the input at all. In Step 7 this showed up as 51.0% test accuracy (seed 1).

??? question "2. Write the rule \"a C bonded to an O\" as a two-part graph rule. Which rule must exist alongside it?"
    You need a rule whose part 0 matches O, say $R_1$: O. Then $R_3$: part 0 = C, part 1 = "a neighbour sent $R_1$". The widget in Step 4 uses exactly this $R_3$.

??? question "3. A graph Tsetlin machine has depth 3. Which atoms can influence whether a rule matches at atom v?"
    All atoms at most 2 bonds from v: part 0 sees v itself, part 1 sees messages from v's neighbours (about themselves), and part 2 sees messages that the neighbours sent after reading **their** neighbours. Depth $D$ gives a radius of $D - 1$.

??? question "4. During training, a \"bonded\" rule is True on a bonded molecule at two atoms (the N and the O), and is selected for feedback. What happens?"
    It belongs to the true class and is True, so it gets **Recognize Feedback**. One of the two matching atoms is picked at random. The rule's literals are then compared with that atom's properties (part 0) and its inbox (part 1): True literals move towards Memorized with the Memorize Value, False ones towards Forgotten with the Forget Value.

??? question "5. Why is \"the N has no O within 2 bonds\" harder for a Tsetlin machine than \"the N has an O within 2 bonds\"?"
    A rule is an OR over nodes of an AND, so it says "there **exists** a node where this local pattern holds". The positive statement is exactly that kind of claim. The negative one says something about **all** paths from the N, which a single existential rule can't express; the machine has to rely on the "yes" rules not firing, and in our runs that was hard to learn (56.2% average at depth 2).

---

## Further reading

- Ole-Christoffer Granmo, Youmna Abdelwahab, Per-Arne Andersen, Paul F. A. Clarke, Kunal Dumbre, Ylva Grønningsæter, Vojtech Halenka, Runar Helin, Lei Jiao, Ahmed Khalid, Rebekka Omslandseter, Rupsa Saha, Mayur Shende and Xuan Zhang, [*The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With Graphs*](https://arxiv.org/abs/2507.14874) (arXiv 2507.14874, 2025; v2 2026). The graph Tsetlin machine: graph input, deep clauses, messages and depth (§2), the three-A sequence example (§3.1, Appendix A.3), results on images, text, action coreference, recommendation and viral genomes. Our main source for Steps 2 to 6.
- The [`cair/GraphTsetlinMachine`](https://github.com/cair/GraphTsetlinMachine) repository and its README: the tutorial (nodes, edges, properties, hypervector sizes), the sequence and hospital examples, and the CUDA code in which feedback goes to one random node where the clause is True.
- Vojtech Halenka, Ahmed K. Kadhim, Paul F. A. Clarke, Bimal Bhattarai, Rupsa Saha, Ole-Christoffer Granmo, Lei Jiao and Per-Arne Andersen, [*Exploring Effects of Hyperdimensional Vectors for Tsetlin Machines*](https://arxiv.org/abs/2406.02648) (ISTM 2024; arXiv 2406.02648). Hypervector inputs for Tsetlin machines, the representation the graph machine uses for its symbols.
- Sondre Glimsdal and Ole-Christoffer Granmo, [*Coalesced Multi-Output Tsetlin Machines with Clause Sharing*](https://arxiv.org/abs/2108.07594) (arXiv 2108.07594, 2021). The shared, weighted pool of rules that the graph machine is built on.
- Olga Tarasyuk, Anatoliy Gorbenko, Alex Yakovlev and Rishad Shafik, [*Multi-Layer Tsetlin Machine: Architecture and Performance Evaluation*](https://doi.org/10.1109/ISTM62799.2024.10931518) (ISTM 2024; open accepted version at [Leeds Beckett](https://eprints.leedsbeckett.ac.uk/id/eprint/11082/)). Stacked Tsetlin machines trained layer by layer (Step 8).
- Ole-Christoffer Granmo, [*The Tsetlin Machine – A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic*](https://arxiv.org/abs/1804.01508) (arXiv 1804.01508, 2018). §6.2 sketches the Fully Connected Deep Tsetlin Machine.
- Ole-Christoffer Granmo, Sondre Glimsdal, Lei Jiao, Morten Goodwin, Christian W. Omlin and Geir Thore Berge, [*The Convolutional Tsetlin Machine*](https://arxiv.org/abs/1905.09688) (2019): the special case of a graph without edges.
- Ole-Christoffer Granmo, [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/): Ch. 8 "Hierarchical Representation" and Ch. 9 "Graphs" are announced; this lesson will be revisited when they are published.

**Next lessons:** [Language & images](../tsetlin-machine-language-images/index.md): Tsetlin machines for text and images, from bag-of-words rules to image booleanization.

<script src="viz/graph-tsetlin-machine.js"></script>
