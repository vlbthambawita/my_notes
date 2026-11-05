# Tsetlin Machines for Language and Images

!!! abstract "TL;DR"
    A Tsetlin machine only reads True/False features, so text and images must first be **Booleanized**. For text, the classic recipe is the **set of words**: pick a vocabulary, and give every sentence one feature per word, True if the word occurs in it. Rules are then ANDs of words and of **negated words** ("the note does NOT mention *rash*"). Because a sentence holds only a handful of the vocabulary's words, almost every negated literal is True, and learned text rules end up with many of them. For images, each pixel (or each colour channel, or each edge or gradient direction) becomes one or more features through a **threshold**, a **thermometer** or an **adaptive threshold**, and rules become little pictures of must-be-ink, must-be-blank and don't-care pixels. On a small restaurant-review task our numpy Tsetlin machine reaches 75.0% ± 1.9 test accuracy, close to logistic regression (76.7%) on the same words; on 8×8 handwritten digits it reaches 96.9% with a thermometer. The published results tell the same story at scale: competitive with classic methods and small neural networks, behind large modern ones, and readable all the way down.

**Level:** intermediate · **Prerequisites:** [Classification by voting](../tsetlin-machine-classification/index.md) (rules voting for and against a class, the vote sum $v$, the Voting Margin $T$) and [Convolutional Tsetlin machines](../convolutional-tsetlin-machine/index.md) (windows, patches, thermometer encoding). [Hierarchical & graph TMs](../graph-tsetlin-machine/index.md) is useful background but not needed.

!!! note "Beyond the book"
    The chapter of Granmo's book on this topic (Ch. 10, *Natural Language and Image Processing*) has not been published yet. This lesson is therefore based on the research papers listed in [Further reading](#further-reading), retold with the book's vocabulary (rules, memory, Recognize/Erase/Reject Feedback, Voting Margin). The text and digit experiments are our own small numpy runs, made for this lesson; they are toy-scale on purpose and are labelled as ours throughout.

---

## Why should I care?

Most medical information is not a neat table. It is **free text** (admission notes, discharge letters, radiology reports) and **images** (X-rays, slides, photographs of the skin). The models that read them best today are large neural networks, and nobody can read their reasoning.

- **Clinical text.** Berge et al. (2019) used a Tsetlin machine to find notes about drug allergies in real electronic health records from a Norwegian hospital, written in a mixture of Latin, English and the local language, full of jargon and misspellings. Their example rule is *if "rash" and "reaction" and "penicillin" then Allergy*: a clinician can check that in a second.
- **Images.** A convolutional Tsetlin machine learns small pixel patterns you can draw and inspect, and colour images are now handled by teams of machines that each look at the image in a different way (Grønningsæter et al., 2024).
- **An honest trade.** You give up some accuracy compared with the best neural networks, and in exchange every decision is a sum of rules you can list. This lesson shows both sides with numbers.

The learning algorithm doesn't change at all. Everything new in this lesson is about **turning words and pixels into literals**, and **reading the rules** that come out.

---

## Step 1 — Text as a set of words

The simplest way to Booleanize text (Berge et al., 2019) is:

1. Build a **vocabulary**: a list of words, $w_1, \dots, w_K$.
2. Give every document one Boolean feature per vocabulary word: $x_k$ is True if word $w_k$ occurs in the document at least once, False otherwise.

That's it. Word order and word counts are thrown away; only "which words are present" survives. The literature calls this a **set of words** (or binary **bag of words**). Each feature brings two literals as always: $x_{\text{good}}$ ("the text contains *good*") and $\lnot x_{\text{good}}$ ("the text does NOT contain *good*").

Our running example is real data: the **Yelp part of the UCI "Sentiment Labelled Sentences" dataset** (Kotzias et al., 2015): 1,000 one-sentence restaurant reviews, 500 labelled Positive and 500 Negative. We split them once, 700 for training and 300 for testing. The training sentences contain 1,661 different words, far too many for a toy machine, so we keep a **vocabulary of 150 words**. To pick them, we drop words that appear in fewer than 2 training sentences and rank the rest by how unevenly they are spread over the two classes (a $\chi^2$ score, the same feature selection the TMU library's IMDb example uses). The top of the list is *not, great, good, amazing, delicious, friendly, bad, at, minutes, don't, …*

**Try it:** type a sentence, or pick one of the buttons.

<div id="viz-sow" class="viz"><div class="viz-title">🔤 Interactive: a sentence becomes 150 True/False features</div></div>

!!! tip "What to notice"
    - **Text is sparse.** A training sentence contains on average only **3.05** of the 150 vocabulary words (at most 12). So a typical sentence has about 3 True features and 147 False ones, and **147 of its negated literals are True**. Hold on to that; it explains Step 2.
    - **Words outside the vocabulary are invisible.** In *"Never had anything to complain about here."* only *never* and *to* are in the vocabulary. 15 of the 300 test sentences contain no vocabulary word at all.
    - **The vote is the familiar one.** *"Crust is not good."* fires 9 rules for Positive (all "good") and 10 for Negative (8 copies of "not" plus two refined "not" rules), so $v = 9 - 10 = -1$: Negative, which is right. The machine never learned that *not* belongs to *good*; the "not" rules simply outvote the "good" rules here.

---

## Step 2 — What a text does NOT say: negated word literals

In a table of patient data, $x$ and $\lnot x$ are usually about equally common. In text they are not: any given word is **absent** from almost every sentence. That has two consequences.

**Negation is a real, useful signal.** "The note does not mention *penicillin*" is evidence too. Berge et al. stress that "even the absence of terms (negated features) can be used for categorization". Their worked example (Figure 4 of the paper) shows how a rule for **Allergy** grows document by document. Step through it:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🩺 How a text rule grows (Berge et al., 2019, Fig. 4)",
  "cols": 3,
  "rows": 3,
  "rowHeight": 95,
  "nodes": [
    {"id": "d1", "col": 0, "row": 0, "icon": "📄", "label": "Note 1: Allergy", "sub": "…reacts to Voltaren…", "role": "data", "note": "A note labelled Allergy. Rule C1 is currently just “reacts”, so it fires."},
    {"id": "f1", "col": 1, "row": 0, "icon": "✅", "label": "Recognize", "sub": "memorize a True literal", "role": "opt", "note": "Recognize Feedback (Type I): the literal “Voltaren” is True here and moves into the Memorized half."},
    {"id": "r1", "col": 2, "row": 0, "icon": "🧩", "label": "C1 = reacts ∧ Voltaren", "role": "model", "note": "More precise: both words must be present."},
    {"id": "d3", "col": 0, "row": 1, "icon": "📄", "label": "Note 3: Allergy", "sub": "reacts, Apocillin, no Voltaren", "role": "data", "note": "C1 does not fire (Voltaren is missing), but another rule, C2 = “reacts”, does."},
    {"id": "f3", "col": 1, "row": 1, "icon": "✅", "label": "Recognize", "sub": "for rule C2", "role": "opt", "note": "Recognize Feedback on C2: the literal “Apocillin” crosses into Memorized."},
    {"id": "r3", "col": 2, "row": 1, "icon": "🧩", "label": "C2 = reacts ∧ Apocillin", "role": "model", "note": "A second rule now covers a second drug: the team of rules is an OR of ANDs."},
    {"id": "d7", "col": 0, "row": 2, "icon": "📄", "label": "Note 7: not Allergy", "sub": "has reacts, Voltaren and “not”", "role": "data", "note": "C1 fires on a note that is not about an allergy: a false positive."},
    {"id": "f7", "col": 1, "row": 2, "icon": "⛔", "label": "Reject", "sub": "memorize a False literal", "role": "grad", "note": "Reject Feedback (Type II): “NOT not” is False in this note, so it moves towards Memorized; it was the closest to the boundary and crossed first."},
    {"id": "r7", "col": 2, "row": 2, "icon": "🧩", "label": "C1 = reacts ∧ Voltaren ∧ ¬not", "role": "model", "note": "The rule now refuses notes that contain the word “not”."}
  ],
  "edges": [
    {"from": "d1", "to": "f1"},
    {"from": "f1", "to": "r1"},
    {"from": "d3", "to": "f3"},
    {"from": "f3", "to": "r3"},
    {"from": "d7", "to": "f7"},
    {"from": "f7", "to": "r7"}
  ],
  "steps": [
    {"show": ["d1"], "text": "A note labelled <b>Allergy</b> contains “reacts to Voltaren”. Rule C1 has memorized only <b>reacts</b>, so it fires."},
    {"show": ["d1>f1", "f1", "f1>r1", "r1"], "text": "<b>Recognize Feedback</b> pushes the True literal “Voltaren” into the Memorized half: C1 = <b>reacts ∧ Voltaren</b>."},
    {"show": ["d3", "d3>f3", "f3", "f3>r3", "r3"], "text": "A later Allergy note mentions Apocillin but not Voltaren. C1 stays silent; another rule, C2 = reacts, fires and learns <b>reacts ∧ Apocillin</b>. Two rules, two facets of the class."},
    {"show": ["d7", "d7>f7", "f7"], "text": "A note that is <b>not</b> about an allergy still contains “reacts” and “Voltaren”, so C1 fires wrongly. <b>Reject Feedback</b> pushes False literals towards Memorized; the negated word “NOT not” crosses first."},
    {"show": ["f7>r7", "r7"], "text": "C1 = <b>reacts ∧ Voltaren ∧ ¬not</b>: a negated word literal fixed the false positive. In general, many such updates over several epochs are needed."}
  ],
  "narrow": {"cols": 2, "rows": 6, "rowHeight": 80, "hideSub": true, "pos": {"d1": [0, 0], "f1": [1, 0], "r1": [1, 1], "d3": [0, 2], "f3": [1, 2], "r3": [1, 3], "d7": [0, 4], "f7": [1, 4], "r7": [1, 5]}}
}
</script>
</div>

**Negated literals pile up.** Recall [Recognize Feedback](../tsetlin-machine-feedback/index.md): it memorizes the literals that are **True** in the example. In a sentence with 3 vocabulary words, 147 of the True literals are "NOT *word*". So every Recognize step nudges a rule towards *"NOT this AND NOT that AND NOT …"*, and only Erase Feedback and the Voting Margin hold it back. Sharma et al. (2021) report exactly this on movie reviews: "for NLP, the majority of the features appear in negated form", and a well-trained machine explains a negative review with literals like *NOT witty*, *NOT graceful*, *NOT terrific*, *NOT honest*: the review lacks every sign of praise.

!!! tip "What to notice"
    - **In our run** (Step 3's machine, seed 1), the 200 rules hold **182** memorized plain-word literals and **562** "NOT word" literals: 76% are negated, but they are very unevenly spread: 175 rules are a single plain word like "great", while 6 rules carry 10 or more NOTs each, and three Positive rules carry 140 to 144.
    - **What such a rule means.** One Positive rule is *"NOT any of 144 words"*: every vocabulary word except *great, good, i, is, another, being*. In plain English: *the sentence uses none of the warning words*. It matches 61 Positive and 21 Negative training sentences. It is a description of what Positive reviews **lack**, the same thing Sharma et al. saw.
    - **Medical notes have the same shape.** "No rash", "denies chest pain", "no known drug allergies": clinical text is full of negation, and a set of words cannot tell which word a "no" belongs to. Berge et al.'s rule handles it with $\lnot$*not*; Step 4 shows where that breaks.

---

## Step 3 — Train a text Tsetlin machine

Nothing about the learning changes. It is the two-class machine from [Classification by voting](../tsetlin-machine-classification/index.md): rules voting **for** Positive and rules voting **against** it (for Negative), feedback probability $(T - v)/2T$, Recognize, Erase and Reject Feedback. The settings, chosen after a small search on this same split (so our test accuracy is slightly optimistic; the baselines below were not tuned at all):

- 150 features → 300 literals; **100 rules per class**, Voting Margin **$T = 15$**;
- specificity $s = 8$: Forget Value $1/8$, Memorize Value $7/8$ (the research papers' version; the book's later chapters use Memorize Value 1);
- memory depth 100 (positions 1–100, Memorized above 50); 40 epochs; a tie ($v = 0$) is predicted as Negative.

Text needs **many more rules** than the breast-cancer data did. Each sentence only touches 3 of the 150 words, so one rule built around one word only ever votes on the few sentences containing that word. You need rules for many words before most sentences get any votes at all.

**Try it:** press *Train*. The widget runs exactly the code of our Python check, with the same seeded random numbers, so with the defaults you will see the numbers below (it takes about 10 seconds).

<div id="viz-texttrain" class="viz"><div class="viz-title">🏋️ Interactive: train a Tsetlin machine on restaurant reviews</div></div>

!!! tip "What to notice"
    - **With the defaults (seed 1)** the test accuracy climbs from 59.7% after one epoch to **78.3%** after 40 (training accuracy 83.7%). Seed 1 happens to be the best of ten: over seeds 1–10 the final test accuracy is **75.0% ± 1.9** (range 72.0–78.3%).
    - **The baselines on the same 150 Boolean features**: logistic regression **76.7%**, Bernoulli naive Bayes **77.0%**, a decision tree 75.0%, always answering Positive 51.0%. With 300 test sentences, one sentence is 0.33 percentage points, so the Tsetlin machine is roughly level with the classic methods, not ahead of them. That matches the literature: on the private clinical notes, Berge et al. report an F-measure of 69.3 for the Tsetlin machine, 69.9 for naive Bayes and 69.6 for logistic regression.
    - **Too few rules hurt.** With 20 rules per class and $T = 5$ the test accuracy ends at 68.7%; with 50 rules per class and $T = 10$, 72.7% (seed 1). Many sentences get no votes at all.
    - **The rules read like a sentiment dictionary.** The most common Positive rules are *good* (×9, matching 50 Positive and 16 Negative training sentences), *great* (×7, 40 and 0) and *amazing* (×7, 15 and 0). The most common Negative rules are *not* (×8, 9 and 73) and *bad* (×5, 0 and 13). A refined rule says *"not" AND NOT "great" AND NOT "over"*: it matches 7 Positive and 73 Negative sentences, two fewer Positive ones than plain *not*.
    - **Some rules are leftovers.** With seed 1 a few Positive rules memorized words such as *waiting* or *barely*, which only occur in Negative training sentences. Reject Feedback memorizes a word that is *absent* from the sentence it fired on wrongly, and sometimes the word it picks happens to be a negative one. They cost a few wrong votes; the majority outvotes them. Reading the rules is how you find them.

---

## Step 4 — Reading the rules, and what a set of words cannot see

Because every vote is a rule, you can explain any single prediction by listing the rules that fired, as the first widget does. Try these in it:

| Sentence | Vocabulary words | Firing rules | $v$ | Prediction |
|---|---|---|---|---|
| *The food was good but the service was slow.* | good, slow, but, was | 9 × *good* for; 2 × *slow*, 1 × *but* against | $9 - 3 = +6$ | Positive |
| *Not bad at all, the staff was friendly.* | not, friendly, bad, at, staff, was | 5 × *friendly*, 2 × *staff* for; 8 × *not*, 5 × *bad*, … against | $7 - 17 = -10$ | Negative ✗ |
| *Never had anything to complain about here.* | never, to | 1 × *never* against | $0 - 1 = -1$ | Negative ✗ |

(The first and second sentences are our own; the third is a Positive sentence from the test set. All vote sums use the seed-1 rules.)

The two mistakes have the same cause. "Not bad" and "never … complain" are praise built from negative words, and a set of words keeps the words but **loses their order and their scope**. A rule can say "*not* AND *bad*", but not "*not* directly before *bad*". The research has three main answers:

- **Add structure to the features.** Yadav et al. (2021) classify the sentiment of one **aspect** of a review ("the *price* is reasonable although the *service* is poor"). They split each sentence at the aspect word and add features saying where sentiment words occur (before or after the aspect), plus a few features from the SentiWordNet lexicon. They also replace every word from an opinion lexicon by a common token *positive* or *negative*, which shrinks the vocabulary. They report 78.02% accuracy on the SemEval-2014 Restaurant data and 73.51% on Laptop, on par with the neural network models they compare against. Word pairs (bigrams such as *not_bad*) are another simple option; the TMU library's IMDb example uses words and word pairs.
- **Convolution over text.** ConvTextTM (Bhattarai et al., 2022) cuts a text into a few consecutive **windows** and runs a [convolutional Tsetlin machine](../convolutional-tsetlin-machine/index.md) over them, with thermometer-coded window positions, exactly as over image patches. Each window is a set of words over BERT's fixed vocabulary of word pieces, so new words and misspellings still map to known pieces. A rule can then say *where* in the text its words must occur.
- **Better words.** Combining the vocabulary with word similarities (GloVe synonyms, or Tsetlin machine word embeddings; see [Autoencoders & embeddings](../tsetlin-machine-autoencoder/index.md)) lets a rule about *excellent* also cover *superb*.

!!! tip "What to notice"
    - **An explanation is not a justification.** The second sentence is classified Negative for a perfectly readable reason: 17 rules about *not*, *bad* and *at* outvote 7 about *friendly* and *staff*. The model is wrong, but you can see exactly why, which is the point of interpretability.
    - **The vocabulary decides what can be learned.** *complain* was not among the 150 words, so no rule could use it. In a clinical setting, building the vocabulary (spelling variants, abbreviations, negation cues like *denies*) is half the work.

---

## Step 5 — Images as Boolean features

An image is a grid of numbers: grey levels, or three colour channels per pixel. Like text, it needs Booleanizing. The papers use a small toolbox of methods; Grønningsæter et al. (2024) collect seven of them for colour photographs:

| Method | One pixel (or region) becomes… | Keeps |
|---|---|---|
| **Fixed threshold** | 1 feature: grey level ≥ $t$ | shape of dark strokes |
| **Thermometer** (as for [positions](../convolutional-tsetlin-machine/index.md)) | $t$ features: level ≥ $t_1$, ≥ $t_2$, … | the order of grey levels; a range is an AND |
| **Adaptive threshold** (mean or Gaussian) | 1 feature: brighter than the local average minus $C$ | local contrast, robust to lighting |
| **Otsu's threshold** | 1 feature per channel, threshold chosen per image | object vs background |
| **Colour thermometers** | $t$ features per colour channel | colour |
| **Canny edges** | 1 feature: is this pixel on an edge? | outlines |
| **Histogram of oriented gradients (HOG)** | features from thresholded gradient-direction histograms of small cells | local edge directions |

The convolutional Tsetlin machine paper (Granmo et al., 2019) Booleanized MNIST-style digits with **adaptive Gaussian thresholding** (window 11, threshold value 2), leaving just 1 bit per pixel. The toolbox paper's best single machine on CIFAR-10 used a **colour thermometer** on 5×5 convolution windows.

Our image example is the scikit-learn **digits** dataset: 1,797 handwritten digits of 8×8 pixels with grey levels 0–16 (a small cousin of MNIST). We split it 1,347 / 450 and compare three Booleanizations:

- **threshold**: ink ≥ 8 → 64 features (20.6 True per image on average);
- **thermometer**: ink ≥ 4, ≥ 8, ≥ 12 → 192 features;
- **adaptive**: brighter than the Gaussian-weighted mean of the 3×3 neighbourhood, minus 2 → 64 features. (On 8×8 images this marks smooth background and the light inside of strokes as True, and the darkest ink as False, so the True pixels are now mostly background.)

For each, we trained a multi-class Tsetlin machine (one team per class, as in [Classification, Step 8](../tsetlin-machine-classification/index.md)) with 50 rules for and 50 against each digit, $T = 15$, $s = 4$, memory depth 100, 30 epochs, seed 1. The widget shows the Booleanized image, every team's vote sum, and the rules that fired for the winner. Its accuracy readout is computed in your browser from the learned rules, which we exported from the Python run.

<div id="viz-digits" class="viz"><div class="viz-title">🖼️ Interactive: Booleanize a digit and read the rules that recognise it</div></div>

!!! tip "What to notice"
    - **The Booleanization matters as much as the machine.** Test accuracy: **95.1%** with the threshold, **96.9%** with the thermometer, **89.8%** with the adaptive threshold (seed 1; the means over seeds 1–3 are 95.3%, 96.9% and 90.9%). The thermometer keeps the most information about grey levels. Adaptive thresholding was designed for bigger, unevenly lit images and wastes the tiny 8×8 ones.
    - **The rules are pictures.** A thermometer rule shows three stencils, one per level. A memorized "≥ 4" pixel with a ring on its "≥ 12" plane means *some ink, but not dark*: a grey-level range written as one AND, as in the convolutional lesson. Rules are short: on average 11.9 memorized literals per thermometer rule and 7.4 per threshold rule, out of 384 and 128.
    - **Blank space is evidence.** With the threshold, 71% of the memorized literals are "must be blank" (5,304 of 7,424): a 0 is recognised as much by its empty centre as by its ring. With the adaptive threshold the roles flip: True mostly means background, and 82% of its memorized literals are plain ones. Whether a rule talks in literals or negated literals depends on which value is rare, just as with words.
    - **Press "next mistake".** The thermometer machine gets 14 of the 450 test digits wrong. On those, the winning team fires on average only 6.9 of its 50 rules for the digit and wins by 4.9 votes over the runner-up; on the 436 correct ones it fires 17.7 rules and wins by 22.1 votes. The vote sums double as a [confidence](../tsetlin-machine-confidence/index.md) score.

---

## Step 6 — From small digits to real photographs

The digits above fit a standard Tsetlin machine because they are tiny and centred. Real images need the [convolution window](../convolutional-tsetlin-machine/index.md): rules become patch patterns that may fire anywhere, optionally restricted to a region by thermometer-coded positions. The papers then add two ideas.

- **Look at the image in several ways at once.** No single Booleanization keeps everything: thresholds keep shape, colour thermometers keep colour, HOG keeps edge directions. A **TM composite** (see [Confidence & composites](../tsetlin-machine-confidence/index.md)) trains one machine per view, a *TM specialist*, normalises each machine's class sums by its own range, and adds them. Grønningsæter et al. (2024) built 22 specialists from 7 Booleanizations for CIFAR-10 (60,000 colour photos of 10 kinds of objects). With 64,000 rules each, the best specialist (5×5 colour thermometers) reached **75.4%**, HOG 67.5%, adaptive Gaussian thresholding 70.2% and Canny edges 60.6%; added together they reached **82.8%**.
- **Train with fewer rules at a time.** *Drop clause* (Sharma et al., 2021) ignores a random share of the rules in each epoch, like dropout in neural networks. On CIFAR-10 it took a convolutional Tsetlin machine from 69.3% to 75.1%, and the authors show its pixel-level explanations become more object-like.

What the learned patterns look like is the same as in our widget, only bigger. The convolutional Tsetlin machine paper prints 10×10 rule patterns for MNIST in which `1` is must-be-ink, `0` must-be-blank and `*` don't-care, each with the window positions where it may fire; patterns labelled upper-left, upper-right, lower-left and lower-right of the digit 0 jointly make up its shape.

!!! tip "What to notice"
    - The **history of CIFAR-10 accuracy** for Tsetlin machines, as told by the toolbox paper: about 61% in 2020, 75.1% in 2023 with drop clause, 82.8% in 2024 with composites. Progress came mostly from better Booleanization and teams of machines, not from changing the learning rule.
    - On **medical images** we did not find a published Tsetlin machine study when writing this lesson. The toolbox paper names high-stakes medical applications as the motivation; the methods above are what such work would start from.

---

## Step 7 — An honest comparison with neural networks

First our own numbers, all on the same data splits:

| Task (ours) | Tsetlin machine | Classic baselines | Neural network |
|---|---|---|---|
| Yelp sentences, 150-word set of words | 75.0% ± 1.9 (10 seeds) | logistic regression 76.7%, naive Bayes 77.0% | — |
| 8×8 digits, thermometer Booleans | 96.9% (seed 1) | logistic regression on the same Booleans 96.7% | — |
| 8×8 digits, raw grey levels | (needs Booleans) | logistic regression 97.3%, SVM 99.3%, 3-nearest-neighbours 99.1% | one-hidden-layer MLP 98.4% |

And results **reported by the papers** (accuracy unless noted; each from its own setup, so compare within a row only):

| Benchmark | Tsetlin machine | Neural networks in the same paper | Source |
|---|---|---|---|
| IMDb movie reviews (F-measure) | 89.1 | LSTM 85.6, LSTM-CNN 88.1, Bi-LSTM-CNN 87.9 | Berge et al., 2019 |
| Hospital notes, drug allergy (F-measure) | 69.3 | MLP 67.2; naive Bayes 69.9, logistic regression 69.6 | Berge et al., 2019 |
| R8 news topics | ConvTextTM 96.32 | BERT 97.8, XLNet 98 | Bhattarai et al., 2022 |
| BBC Sports topics | ConvTextTM 96.78 | BERT 99, XLNet 98 | Bhattarai et al., 2022 |
| MNIST digits | CTM 99.33 (mean), 99.4 (peak) | simple CNN 99.06, PreActResNet-18 99.56 | Granmo et al., 2019 |
| Fashion-MNIST | CTM 91.18 (mean), 91.5 (peak) | simple CNN 90.7, PreActResNet-18 92.00 | Granmo et al., 2019 |
| CIFAR-10 | CTM with drop clause 75.1 | Maxout network 86.8 | Sharma et al., 2021 |
| CIFAR-10 | composite of 22 specialists 82.8 | (Vision Transformer 99.9, cited by the authors) | Grønningsæter et al., 2024 |

!!! warning "How to read these tables"
    - **Our runs are tiny**: 300 test sentences and 450 test digits, one fixed split each. Differences of a point or two are within the noise.
    - **The papers' numbers are their own**, with their own preprocessing, tuning effort and ways of averaging (mean over late epochs, peak, or best run). Berge et al. note that their IMDb numbers are means over 100 runs, which is lower than reporting the best run.
    - **The pattern** across all of them: Tsetlin machines are competitive with classic methods and with small neural networks, especially on binary-friendly inputs (digits, short texts), and fall clearly behind large pre-trained models (BERT, XLNet, Vision Transformers) on harder tasks. The papers' authors argue that the price buys interpretability and very cheap inference; you have to decide whether that trade is right for your application.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔗 From words and pixels to readable votes",
  "cols": 3.3,
  "rows": 2.6,
  "rowHeight": 100,
  "nodes": [
    {"id": "text", "col": 0, "row": 0.3, "icon": "📝", "label": "Text", "sub": "a note, a review", "role": "data"},
    {"id": "sow", "col": 1, "row": 0.3, "icon": "🔤", "label": "Set of words", "sub": "x<sub>k</sub>: word k present?", "role": "data", "note": "Choose a vocabulary (e.g. by χ²); optionally add word pairs, lexicon tokens, or windows for a convolutional TM."},
    {"id": "img", "col": 0, "row": 1.7, "icon": "🖼️", "label": "Image", "sub": "grey levels or RGB", "role": "data"},
    {"id": "ib", "col": 1, "row": 1.7, "icon": "🌡️", "label": "Booleanize pixels", "sub": "threshold · thermometer · adaptive", "role": "data", "note": "Or colour thermometers, Canny edges, HOG; with a convolution window over patches for real images."},
    {"id": "rules", "col": 2.275, "row": 0.45, "icon": "🧩", "label": "Rules: AND of literals", "sub": "words and NOT-words; pixel stencils", "role": "model", "note": "Learned with Recognize, Erase and Reject Feedback, exactly as before."},
    {"id": "vote", "col": 2.275, "row": 1.55, "icon": "🗳️", "label": "Vote sum per class", "sub": "every vote is a readable rule", "role": "loss", "note": "The largest vote sum wins; listing the firing rules explains the prediction."}
  ],
  "groups": [
    {"id": "tm", "label": "Tsetlin machine (unchanged)", "col": 2.25, "row": 0, "cols": 1.05, "rows": 2.6}
  ],
  "edges": [
    {"from": "text", "to": "sow"},
    {"from": "img", "to": "ib"},
    {"from": "sow", "to": "tm"},
    {"from": "ib", "to": "tm"},
    {"from": "rules", "to": "vote"}
  ],
  "steps": [
    {"show": ["text", "text>sow", "sow"], "text": "<b>Text</b> becomes a set of words: one True/False feature per vocabulary word. Each feature gives a literal and a negated literal."},
    {"show": ["img", "img>ib", "ib"], "text": "<b>Images</b> become Boolean planes: a threshold, a thermometer of levels, or an adaptive threshold (and for colour photos, colour thermometers, edges or HOG)."},
    {"show": ["sow>tm", "ib>tm", "tm", "rules"], "text": "From here on it is the <b>same Tsetlin machine</b>: rules are ANDs of literals, learned by Recognize, Erase and Reject Feedback. For text they mix words and NOT-words; for images they are pixel stencils."},
    {"show": ["rules>vote", "vote"], "text": "Each class's rules vote; the largest <b>vote sum</b> wins. To explain a prediction, list the rules that fired."}
  ],
  "narrow": {"cols": 2, "rows": 3.95, "rowHeight": 95, "pos": {"text": [0, 0], "img": [1, 0], "sow": [0, 1], "ib": [1, 1], "rules": [0, 2.6], "vote": [1, 2.6], "tm": [0, 2.15, 2, 1.75]}}
}
</script>
</div>

Here is the whole text pipeline in numpy: Booleanize the Yelp sentences, train the machine of Step 3, and print the most common short rules. It uses numpy's random numbers and its own split (`default_rng(0)`), so its numbers differ a little from the widget's. It needs `yelp_labelled.txt` from the [UCI dataset](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) in a `data/` folder, and runs in about 10 seconds.

```python
import re

import numpy as np
from sklearn.feature_selection import chi2

# ---- 1. Booleanize: a set of words. One feature per vocabulary word: is it in the sentence?
rows = [line.rstrip("\n").rsplit("\t", 1) for line in open("data/yelp_labelled.txt", encoding="utf-8") if line.strip()]
words = [set(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())) for text, _ in rows]
y = np.array([int(label) for _, label in rows])
order = np.random.default_rng(0).permutation(len(rows))
train, test = order[:700], order[700:]

candidates = sorted({w for i in train for w in words[i]})
X_all = np.array([[w in words[i] for w in candidates] for i in range(len(rows))], dtype=np.int8)
keep = X_all[train].sum(0) >= 2                                  # words seen in 2+ training sentences
score = np.where(keep, np.nan_to_num(chi2(X_all[train], y[train])[0]), -1)
vocab_idx = np.argsort(-score, kind="stable")[:150]                # the 150 most class-related words
vocab = [candidates[k] for k in vocab_idx]
X = X_all[:, vocab_idx]


class TsetlinMachine:
    """Two classes, as in the classification lesson. memory[c, j, k]: c = 0 votes Negative, 1 votes Positive."""

    def __init__(self, n_features, rules_per_class=100, T=15, s=8.0, depth=100, seed=1):
        self.T, self.s, self.depth, self.half = T, s, depth, depth // 2
        self.rng = np.random.default_rng(seed)
        self.memory = np.full((2, rules_per_class, 2 * n_features), depth // 2)

    def fires(self, lits, training=False):
        memorized = self.memory > self.half
        ok = ~np.any(memorized & (lits == 0), axis=2)
        return ok if training else ok & memorized.any(axis=2)      # empty rules stay silent when classifying

    def predict(self, x):
        f = self.fires(np.concatenate([x, 1 - x]))
        return int(f[1].sum() - f[0].sum() > 0)                     # tie → Negative

    def update(self, x, y):
        lits = np.concatenate([x, 1 - x])                           # words, then NOT words
        f = self.fires(lits, training=True)
        v = np.clip(f[y].sum() - f[1 - y].sum(), -self.T, self.T)
        chosen = self.rng.random(f.shape) <= (self.T - v) / (2 * self.T)
        u = self.rng.random(self.memory.shape[1:])
        own, other = self.memory[y], self.memory[1 - y]
        rec = (chosen[y] & f[y])[:, None]                           # Recognize Feedback
        own += rec & (lits == 1) & (u <= (self.s - 1) / self.s) & (own < self.depth)
        own -= rec & (lits == 0) & (u <= 1 / self.s) & (own > 1)
        own -= (chosen[y] & ~f[y])[:, None] & (u <= 1 / self.s) & (own > 1)   # Erase Feedback
        other += (chosen[1 - y] & f[1 - y])[:, None] & (lits == 0) & (other <= self.half)  # Reject

    def rule(self, c, j):
        m = self.memory[c, j] > self.half
        n = len(vocab)
        return [vocab[k] for k in range(n) if m[k]] + ["NOT " + vocab[k] for k in range(n) if m[n + k]]


tm = TsetlinMachine(len(vocab))
for epoch in range(1, 41):
    for i in tm.rng.permutation(train):
        tm.update(X[i], y[i])
    if epoch in (1, 10, 40):
        acc = np.mean([tm.predict(X[i]) == y[i] for i in test])
        print(f"epoch {epoch:2d}: test accuracy {acc:.3f}")

memorized = tm.memory > tm.half
print("memorized literals: plain words", memorized[..., :150].sum(), "· NOT words", memorized[..., 150:].sum())
for c, name in ((1, "Positive"), (0, "Negative")):
    rules = [tuple(tm.rule(c, j)) for j in range(tm.memory.shape[1])]
    common = sorted({r: rules.count(r) for r in rules if 0 < len(r) <= 3}.items(), key=lambda kv: -kv[1])[:5]
    print(name, "  ".join(f"{' AND '.join(r)} (x{n})" for r, n in common))
```

```text
epoch  1: test accuracy 0.563
epoch 10: test accuracy 0.757
epoch 40: test accuracy 0.767
memorized literals: plain words 191 · NOT words 526
Positive great (x8)  delicious (x6)  nice (x4)  amazing (x4)  awesome (x4)
Negative don't (x6)  disappointed (x5)  not (x4)  no (x3)  bad (x3)
```

The same picture as in the widget: about three quarters of the memorized literals are NOT-words, the short rules form a small sentiment dictionary, and the accuracy is close to logistic regression's. For images, replace the first block by a Booleanization (`(images >= 8)`, or a thermometer `np.concatenate([images >= t for t in (4, 8, 12)])`) and use one team of rules per class.

---

## Common misconceptions

!!! warning "\"A Tsetlin machine needs special learning rules for text or images.\""
    It doesn't. Every result in this lesson uses the same Recognize, Erase and Reject Feedback and the same Voting Margin as the classification lesson. What changes is the **Booleanization** in front of it (and, for images, the convolution window). Most of the progress reported on CIFAR-10 came from better Booleanizations and teams of machines.

!!! warning "\"Rules full of NOT-words are a bug.\""
    They are a consequence of sparse data: a sentence contains a handful of the vocabulary's words, so almost every "NOT word" literal is True, and Recognize Feedback memorizes True literals. A rule like "*NOT any of these 144 words*" means "*none of the warning signs appear*", which can be a sensible description of a class (Sharma et al. found the same in movie reviews). Very long NOT-lists can also signal a rule that has memorized a few training sentences, so check its coverage.

!!! warning "\"A set of words understands negation.\""
    It keeps the word *not* but not what it applies to. "Not bad at all" contains *not* and *bad*, and our machine called it Negative with $v = -10$. Word pairs, position features (Yadav et al.) or windows over the text (ConvTextTM) give the rules some word order back.

!!! warning "\"Readable means right.\""
    Readable rules let you **audit** a model: you can spot a Positive rule built on *waiting*, or a vocabulary that misses *complain*. They do not make the model correct, and on hard tasks Tsetlin machines are still clearly behind large neural networks.

!!! warning "\"A threshold at mid-grey is all an image needs.\""
    On our digits the choice of Booleanization moved accuracy from 89.8% (adaptive threshold) to 96.9% (thermometer), with the same machine. For colour photos the toolbox paper's single specialists ranged from 60.6% (Canny edges) to 75.4% (colour thermometers). Try several, or combine them in a composite.

---

## Check yourself

??? question "1. A vocabulary has 2,000 words. How many literals does each rule choose from, and roughly how many of them are True for a 40-word clinical note with 30 distinct vocabulary words?"
    2 × 2,000 = **4,000 literals**. 30 plain-word literals are True (the words present), and 2,000 − 30 = **1,970 negated literals** are True (the words absent): 2,000 True literals in total, as always one of each pair. Almost all of the True ones are negated, which is why text rules collect so many NOT-words.

??? question "2. A rule for Allergy is *\"penicillin\" AND \"rash\"*. It fires on a note that says \"no rash after penicillin\". Which feedback does it get when this note is labelled not-Allergy, and which literal might fix it?"
    It fired on the wrong class, so if it is selected it gets **Reject Feedback**: every forgotten literal that is False in this note moves one step towards Memorized. "NOT no" is False here (the note contains *no*), so it may cross first, giving *penicillin AND rash AND NOT no*, just like Berge et al.'s *¬not*. It is a blunt fix: the rule now also rejects "rash after penicillin, no fever".

??? question "3. Why did our text machine need 100 rules per class when the breast-cancer machine managed with 10?"
    A typical sentence contains only 3 of the 150 vocabulary words, and most learned rules are built around one word. A rule can only vote on sentences that contain its words, so many rules (about different words) are needed before most sentences receive any votes. With 20 rules per class the test accuracy dropped to 68.7%.

??? question "4. Write the thermometer literals that say \"this pixel has some ink but is not dark\" for our digits (planes ≥ 4, ≥ 8, ≥ 12)."
    $x_{\ge 4} \wedge \lnot x_{\ge 12}$: at least level 4, below level 12. A single AND expresses the range, which a single threshold feature cannot.

??? question "5. The toolbox paper's composite reaches 82.8% on CIFAR-10, and its best single machine 75.4%. Where does the gain come from, given that the learning rule is the same?"
    From **different views of the same image**: each specialist Booleanizes the image differently (colour thermometers, adaptive thresholds, Canny edges, HOG, …) and so makes different mistakes. The composite normalises each machine's class sums and adds them, so a confident specialist can outvote an unsure one ([Confidence & composites](../tsetlin-machine-confidence/index.md)).

---

## Further reading

- Ole-Christoffer Granmo, [*An Introduction to Tsetlin Machines*](https://tsetlinmachine.org/) (free online). Ch. 10, *Natural Language and Image Processing*, is announced but not yet published; this lesson uses the vocabulary of the published chapters.
- Geir Thore Berge, Ole-Christoffer Granmo, Tor Oddbjørn Tveit, Morten Goodwin, Lei Jiao and Bernt Viggo Matheussen, [*Using the Tsetlin Machine to Learn Human-Interpretable Rules for High-Accuracy Text Categorization With Medical Applications*](https://arxiv.org/abs/1809.04547), IEEE Access 7 (2019), 115134–115146. The set-of-words representation, negated words, the Allergy walk-through (our Step 2 diagram), and results on 20 Newsgroups, IMDb and hospital notes.
- Rohan Kumar Yadav, Lei Jiao, Ole-Christoffer Granmo and Morten Goodwin, [*Human-Level Interpretable Learning for Aspect-Based Sentiment Analysis*](https://doi.org/10.1609/aaai.v35i16.17671), AAAI 2021. Booleanizing aspect, position and lexicon information for sentiment towards one aspect of a review.
- Bimal Bhattarai, Ole-Christoffer Granmo and Lei Jiao, [*ConvTextTM: An Explainable Convolutional Tsetlin Machine Framework for Text Classification*](https://aclanthology.org/2022.lrec-1.401), LREC 2022. Convolution over text windows with BERT's vocabulary.
- Jivitesh Sharma, Rohan Yadav, Ole-Christoffer Granmo and Lei Jiao, [*Drop Clause: Enhancing Performance, Interpretability and Robustness of the Tsetlin Machine*](https://arxiv.org/abs/2105.14506) (arXiv, 2021). Dropping rules during training; the observation that text rules are mostly negated words; CIFAR-10 at 75.1%.
- Ole-Christoffer Granmo, Sondre Glimsdal, Lei Jiao, Morten Goodwin, Christian W. Omlin and Geir Thore Berge, [*The Convolutional Tsetlin Machine*](https://arxiv.org/abs/1905.09688) (arXiv, 2019). Adaptive Gaussian thresholding, MNIST / Fashion-MNIST / Kuzushiji-MNIST results, printed rule patterns.
- Ylva Grønningsæter, Halvor S. Smørvik and Ole-Christoffer Granmo, [*An Optimized Toolbox for Advanced Image Processing with Tsetlin Machine Composites*](https://arxiv.org/abs/2406.00704) (arXiv, 2024). Seven image Booleanizations, 22 specialists, CIFAR-10 at 82.8%.
- The [`cair/tmu`](https://github.com/cair/tmu) library: its `examples/classification` folder has the IMDb text example (words and word pairs, χ² feature selection) and MNIST convolution examples.
- Data: Dimitrios Kotzias, Misha Denil, Nando de Freitas and Padhraic Smyth, *From Group to Individual Labels using Deep Features*, KDD 2015: the [Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) dataset (CC BY 4.0). The 8×8 digits are scikit-learn's `load_digits` (UCI Optical Recognition of Handwritten Digits).

**Next lessons:** you have reached the end of the Tsetlin machine series. Head back to the [roadmap's Tsetlin machine section](../../roadmap.md#tsetlin-machines) to revisit any lesson.

<script src="viz/tsetlin-machine-language-images.js"></script>
