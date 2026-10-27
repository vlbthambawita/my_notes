# RL from Human Feedback (RLHF) and DPO

!!! abstract "TL;DR"
    A pre-trained language model predicts text; it does not try to be helpful. **RLHF** fixes that in three stages: fine-tune on good examples (**SFT**), train a **reward model** from human comparisons ("reply A is better than reply B"), then use RL (usually **PPO**) to make the model produce replies the reward model likes, while a **KL penalty** keeps it close to where it started. **DPO** reaches the same goal with a single supervised-style loss on the comparisons: no reward model and no RL loop.

**Level:** advanced · **Prerequisites:** [Policy gradients](../policy-gradients/index.md), [Actor-critic](../actor-critic/index.md) (PPO). Knowing what a language model does ("predict the next token") helps.

---

## Why should I care?

This is where the whole RL series pays off. In the [first lesson](../reinforcement-learning-intro/index.md) the chatbot row of the table said: state = the conversation so far, action = the next word, reward = a score for the finished reply. This lesson fills in that row.

- **It is how chat assistants are aligned.** OpenAI's InstructGPT paper (Ouyang et al., 2022) showed that labellers preferred replies from a 1.3-billion-parameter model fine-tuned with RLHF over replies from the 175-billion-parameter GPT-3 it started from. The same recipe, or a close relative, is used to fine-tune most chat assistants.
- **It turns "be helpful" into something you can optimise.** Nobody can write a formula for a good answer. But people can easily say which of two answers is better. RLHF learns a reward from those judgements.
- **It shows RL's classic failure modes in a modern setting.** Reward hacking, distribution shift and the need for a "leash" on the policy all show up, and the fixes reuse ideas from PPO and from the value-function lessons.

---

## Step 1 — The problem: a text predictor is not an assistant

A **pre-trained** language model has read a huge amount of text and learned one thing: given some text, predict the next token (a word or a piece of a word). Ask it a question and it continues the text in whatever way is most typical of the internet. That might be an answer. It might also be three more questions, a forum argument, or a confident wrong answer.

| Prompt: "Why is the sky blue? Explain for a 10-year-old." | |
|---|---|
| What a pre-trained model might write | "Why is grass green? Why is the sea salty? 50 science questions kids ask…" (it continues the *list* it thinks it is reading) |
| What we want | "Sunlight is made of all colours. Air bounces blue light around much more than red, so blue comes at you from every part of the sky." |

**Supervised fine-tuning (SFT)** is the obvious first fix: collect prompts with good human-written answers and train the model to imitate them (ordinary next-token cross-entropy). It helps a lot. But it has limits:

1. **Writing ideal answers is slow and expensive**, and the model can only be as good as the demonstrations.
2. **Imitation never says what is *bad*.** The model sees good answers, not "this answer is wrong", "this one is rude", "this one is long-winded".
3. **"Helpful, honest, harmless" is hard to write as a reward function.** You can't code `def reward(answer): ...` for helpfulness.

The key observation behind RLHF (Christiano et al., 2017) is that **comparing is much easier than writing or scoring**. Show a person two replies and ask "which is better?". They can answer in seconds, and they don't need to agree on an absolute scale. Those comparisons are the training signal.

---

## Step 2 — A language model is a policy

To use RL, we need the familiar pieces from lesson 1: states, actions, a policy and a reward. A language model already is a policy. Generating a reply token by token is an episode:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔤 Writing a reply, one token at a time, is an RL episode",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "prompt", "col": 0, "row": 0, "icon": "💬", "label": "Prompt x", "sub": "“Why is the sky blue?”", "shape": "pill", "role": "data", "note": "Each episode starts from a prompt drawn from a dataset of user requests."},
    {"id": "state", "col": 1, "row": 0, "icon": "📜", "label": "State s<sub>t</sub>", "sub": "prompt + tokens so far", "role": "data", "note": "The state is all the text so far. It grows by one token per step, and nothing about it is hidden: the transition is deterministic."},
    {"id": "policy", "col": 2, "row": 0, "icon": "🤖", "label": "Policy π<sub>θ</sub>", "sub": "the language model", "role": "model", "note": "π_θ(a | s) is exactly the model's next-token distribution: a softmax over the whole vocabulary."},
    {"id": "token", "col": 2, "row": 1, "icon": "🎲", "label": "Action a<sub>t</sub>", "sub": "sample the next token", "role": "model", "note": "The action space is the vocabulary: tens of thousands of possible tokens at every step."},
    {"id": "done", "col": 1, "row": 1, "label": "End of reply?", "shape": "decision", "role": "grad"},
    {"id": "reward", "col": 0, "row": 1, "icon": "⚖️", "label": "Reward r(x, y)", "sub": "one score for the whole reply", "role": "loss", "note": "In RLHF the reward comes from a learned reward model and arrives only once, at the end: a sparse, delayed reward, just like winning a game of chess."}
  ],
  "edges": [
    {"from": "prompt", "to": "state"},
    {"from": "state", "to": "policy"},
    {"from": "policy", "to": "token"},
    {"from": "token", "to": "done"},
    {"from": "done", "to": "state", "label": "no"},
    {"from": "done", "to": "reward", "label": "yes"}
  ],
  "steps": [
    {"show": ["prompt"], "text": "An episode starts with a <b>prompt</b> x, for example a user's question."},
    {"show": ["prompt>state", "state"], "text": "The <b>state</b> is the prompt plus every token written so far."},
    {"show": ["state>policy", "policy"], "text": "The language model is the <b>policy</b>: it outputs a probability for every possible next token."},
    {"show": ["policy>token", "token"], "text": "The <b>action</b> is the token we sample. It is appended to the text."},
    {"show": ["token>done", "done", "done>state", "state"], "text": "If the reply isn't finished, the new text is the next state and the loop repeats. Intermediate rewards are zero."},
    {"show": ["done>reward", "reward"], "text": "When the model emits its end-of-reply token, the episode ends and the whole reply y gets <b>one reward</b>."}
  ],
  "narrow": {"cols": 2, "rows": 4, "pos": {"prompt": [0, 0], "state": [0, 1], "policy": [1, 1], "token": [1, 2], "done": [0, 2], "reward": [0, 3]}}
}
</script>
</div>

The probability of a whole reply $y = (y_1, \dots, y_T)$ is the product of the token probabilities, so its log-probability is a sum:

$$
\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})
$$

That is exactly the $\log \pi_\theta(a_t \mid s_t)$ term from the [policy gradients](../policy-gradients/index.md) lesson, summed over the episode. So REINFORCE, actor-critic and PPO all apply without change. The hard part is the reward.

!!! tip "What to notice"
    - The "environment" is trivial: appending a token is deterministic, and nothing pushes back. All the difficulty sits in the **reward**, which is given only once per reply.
    - Because the reward is per reply, it is common to treat the *whole reply* as one big action. That simpler view (a "contextual bandit", like the [bandits](../multi-armed-bandits/index.md) lesson with a prompt as context) is the one the rest of this lesson uses, and it is the view DPO is built on.

---

## Step 3 — The RLHF recipe in three stages

The pipeline popularised by InstructGPT (Ouyang et al., 2022), building on Christiano et al. (2017) and Stiennon et al. (2020), has three stages. Click through it:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🏭 The classic RLHF pipeline",
  "cols": 3,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "pre", "col": 0, "row": 0, "icon": "📚", "label": "Pre-trained LM", "sub": "predicts the next token", "role": "model", "note": "Trained on a huge text corpus. Knows a lot, but doesn't follow instructions."},
    {"id": "sft", "col": 1, "row": 0, "icon": "✍️", "label": "<b>1. SFT model</b>", "sub": "imitates good answers", "role": "model", "note": "Supervised fine-tuning on prompts with human-written demonstrations. Ordinary cross-entropy training."},
    {"id": "pairs", "col": 2, "row": 0, "icon": "👥", "label": "Humans compare replies", "sub": "y<sub>w</sub> ≻ y<sub>l</sub>", "role": "data", "note": "For each prompt, sample several replies from the SFT model and ask labellers to rank them. Every ranking gives pairs (winner, loser)."},
    {"id": "rm", "col": 2, "row": 1, "icon": "⚖️", "label": "<b>2. Reward model</b> r<sub>φ</sub>", "sub": "Bradley–Terry loss", "role": "loss", "note": "Usually the SFT model with its output layer replaced by a single number. Trained so that winners score higher than losers."},
    {"id": "ppo", "col": 1, "row": 1, "icon": "🔁", "label": "<b>3. RL with PPO</b>", "sub": "max r − β·KL", "role": "model", "note": "The policy writes replies, the reward model scores them, PPO updates the policy. A KL penalty keeps it close to the SFT model."},
    {"id": "aligned", "col": 0, "row": 1, "icon": "✅", "label": "Aligned assistant", "shape": "pill", "role": "opt"}
  ],
  "edges": [
    {"from": "pre", "to": "sft"},
    {"from": "sft", "to": "pairs"},
    {"from": "pairs", "to": "rm"},
    {"from": "rm", "to": "ppo"},
    {"from": "sft", "to": "ppo", "label": "π<sub>ref</sub>, init"},
    {"from": "ppo", "to": "aligned"}
  ],
  "steps": [
    {"show": ["pre"], "text": "Start from a <b>pre-trained</b> language model."},
    {"show": ["pre>sft", "sft"], "text": "<b>Stage 1, SFT:</b> fine-tune it on human-written demonstrations, so it at least answers in the right style."},
    {"show": ["sft>pairs", "pairs"], "text": "Sample several replies per prompt from the SFT model. Humans say which they prefer. (InstructGPT asked labellers to rank 4 to 9 replies per prompt.)"},
    {"show": ["pairs>rm", "rm"], "text": "<b>Stage 2, reward model:</b> train a network to give higher scores to the replies humans preferred (Step 4)."},
    {"show": ["rm>ppo", "sft>ppo", "ppo"], "text": "<b>Stage 3, RL:</b> starting from the SFT model, PPO maximises the reward model's score minus a KL penalty that keeps the policy near the SFT model π<sub>ref</sub> (Step 5)."},
    {"show": ["ppo>aligned", "aligned"], "text": "The result is the assistant you chat with. In practice stages 2 and 3 are often repeated with fresh comparisons of the new model's replies."}
  ],
  "narrow": {"cols": 2.4, "rows": 4, "pos": {"pre": [0, 0], "sft": [0, 1], "pairs": [1.4, 1], "rm": [1.4, 2], "ppo": [0, 2], "aligned": [0, 3]}}
}
</script>
</div>

!!! tip "What to notice"
    - Humans appear only once, as **labellers of comparisons**. Everything after that is automatic: the reward model stands in for the humans during RL, so the policy can be scored millions of times.
    - The SFT model plays two roles in stage 3: it is the **starting point** of the policy and the **reference** $\pi_{\text{ref}}$ that the KL penalty measures distance from.

---

## Step 4 — Learning a reward from comparisons (Bradley–Terry)

The labellers give us triples $(x, y_w, y_l)$: a prompt, the **winning** (preferred) reply and the **losing** one. We want a reward model $r_\phi(x, y)$ that outputs one number per reply. How do we turn "A beat B" into a training signal for numbers?

The **Bradley–Terry model** (from 1952, originally for ranking chess players and sports teams) says: the probability that A beats B depends only on the **difference** of their scores, squashed through the sigmoid $\sigma(z) = 1/(1+e^{-z})$:

$$
P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)
$$

Training the reward model is then ordinary maximum likelihood, i.e. logistic regression on the score difference:

$$
\mathcal{L}_{\text{RM}}(\phi) = -\,\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]
$$

**Try it:** set the reward model's two scores and see what it predicts and how much loss it pays for each possible human choice.

<div id="viz-bt" class="viz"><div class="viz-title">⚖️ Interactive: Bradley–Terry preference probability and loss</div></div>

!!! tip "What to notice"
    - **Equal scores mean a coin flip:** $\Delta = 0$ gives $P = 0.5$ and loss $\log 2 \approx 0.693$ either way. That is where training starts.
    - **Getting it right pays little, getting it wrong costs a lot.** At $\Delta = 2$ the model says $P = 0.881$; if humans chose A the loss is only 0.127, but if they chose B it is 2.127. At $\Delta = 4$ the loss for a correct prediction is down to 0.018, so confident, correct pairs stop contributing much.
    - **Add +1 to both:** nothing changes. Rewards are only defined **up to a constant**. A reward model's absolute numbers mean nothing; only differences between replies to the same prompt do. (This shift-freedom is exactly what makes DPO possible in Step 6.)
    - Humans disagree and are noisy, so a perfect reward model would *not* hit 100% accuracy. In the code below, the true hidden preference itself only predicts 84.8% of held-out human choices.

In practice the reward model is a copy of the SFT model with the next-token output layer swapped for a single scalar output. In the toy version below, each "reply" is just three features (helpful, correct, length) and the reward model is linear. The hidden human taste ignores length:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
X = torch.randn(500, 3)                     # 500 responses, features: helpful, correct, length
w_true = torch.tensor([1.5, 2.0, 0.0])      # the hidden "human taste": length doesn't matter
r_true = X @ w_true

# 5000 comparisons: the labeller prefers a over b with probability sigmoid(r_a - r_b)
a, b = torch.randint(0, 500, (5000,)), torch.randint(0, 500, (5000,))
a_wins = torch.rand(5000) < torch.sigmoid(r_true[a] - r_true[b])
chosen, rejected = torch.where(a_wins, a, b), torch.where(a_wins, b, a)
train, test = slice(0, 4000), slice(4000, 5000)


def reward_model_loss(r_chosen, r_rejected):
    """Bradley-Terry: -log P(chosen beats rejected) = -log sigmoid(r_w - r_l)."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()


reward_model = torch.nn.Linear(3, 1, bias=False)   # in real RLHF: a transformer with a scalar head
opt = torch.optim.Adam(reward_model.parameters(), lr=0.05)
for step in range(301):
    loss = reward_model_loss(reward_model(X[chosen[train]]).squeeze(-1),
                             reward_model(X[rejected[train]]).squeeze(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.3f}")

with torch.no_grad():
    r = reward_model(X).squeeze(-1)
    acc = (r[chosen[test]] > r[rejected[test]]).float().mean().item()
    acc_true = (r_true[chosen[test]] > r_true[rejected[test]]).float().mean().item()
print("learned weights:", [round(v, 2) for v in reward_model.weight[0].tolist()])
print(f"held-out accuracy: reward model {acc:.3f}, true reward {acc_true:.3f}")
```

```text
step   0  loss 0.774
step 100  loss 0.334
step 200  loss 0.334
step 300  loss 0.334
learned weights: [1.41, 1.86, 0.05]
held-out accuracy: reward model 0.850, true reward 0.848
```

From nothing but "A beat B" labels, the reward model recovered the hidden taste ($[1.41, 1.86, 0.05]$ vs the true $[1.5, 2.0, 0.0]$) and predicts held-out human choices as well as the true reward does. With a real network and real text, it is never this clean: the reward model sees a limited set of comparisons and has to generalise to replies unlike any it was trained on. That is the weak spot Step 5 deals with.

---

## Step 5 — RL fine-tuning with a KL leash

Now the RL stage. The policy $\pi_\theta$ (initialised from the SFT model) writes replies, the reward model scores them, and we maximise

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot \mid x)}\big[\, r_\phi(x, y) \,\big] \;-\; \beta\, \mathrm{KL}\big(\pi_\theta(\cdot \mid x)\,\big\|\,\pi_{\text{ref}}(\cdot \mid x)\big)
$$

The first term is "get a high score". The second is a **leash**: the KL divergence $\mathrm{KL}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi}\big[\log \tfrac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\big]$ measures how far the policy has drifted from the SFT model, and $\beta$ sets how strongly drifting is punished.

The optimiser is usually **PPO** from the [actor-critic](../actor-critic/index.md) lesson. In InstructGPT-style implementations the KL term is folded into the reward, $r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$, often spread over tokens as a per-token penalty, and a value head (the critic) estimates advantages. So during training four networks are in play: the policy, the frozen reference, the frozen reward model and the critic.

**Why the leash?** The reward model is only a proxy for what people want. It was trained on replies from the SFT model, so it is only trustworthy *near* the SFT model's replies. A policy that is pushed hard to maximise the proxy will find the places where the proxy is wrong: replies the reward model over-rates. This is **reward hacking** or **reward over-optimisation**, an instance of Goodhart's law ("when a measure becomes a target, it ceases to be a good measure"). Gao et al. (2022) measured it directly: as RL pushes the policy further from its start, the learned reward keeps rising while the true quality first rises and then falls.

The KL-regularised objective has a beautiful exact solution. For one prompt, the best possible policy is

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y \mid x)\,\exp\!\Big(\frac{r(x, y)}{\beta}\Big), \qquad Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x)\exp\!\Big(\frac{r(x, y)}{\beta}\Big)
$$

**Reweight the reference model by the exponentiated reward.** (Why: a few lines of algebra show $\mathbb{E}_\pi[r] - \beta\,\mathrm{KL}(\pi \| \pi_{\text{ref}}) = \beta \log Z(x) - \beta\,\mathrm{KL}(\pi \| \pi^*)$. The first term doesn't depend on $\pi$, and a KL is zero only when the two distributions are equal, so $\pi = \pi^*$ is the maximum.) We can't compute $Z(x)$ for a real model, since it sums over every possible reply. That is why RLHF needs RL to *approximate* $\pi^*$. But in a toy with five candidate replies we can compute it exactly.

**Try it:** the prompt is "Why is the sky blue?", asked by a 10-year-old. The SFT model $\pi_{\text{ref}}$ spreads its probability over five replies: **A** a simple correct answer, **B** a correct but technical one, **C** a wrong one ("it reflects the ocean"), **D** a refusal, and **E** gushing flattery ("What a WONDERFUL question!!!") with no real answer. The reward model got E badly wrong: it scores it 3.0, the highest of all, while its true quality is −0.5. Lower $\beta$ (loosen the leash) and watch.

<div id="viz-kl" class="viz"><div class="viz-title">🪢 Interactive: the KL-regularised optimum and reward hacking</div></div>

!!! tip "What to notice"
    - **Large $\beta$ = tight leash.** At $\beta = 10$, $\pi^*$ is almost $\pi_{\text{ref}}$ (KL 0.012) and the true quality barely moves from the starting model's −0.275 to −0.112.
    - **Moderate $\beta$ helps for real.** At $\beta = 1$ the policy shifts mass from C and D to A (0.391), and the true quality rises to 0.715. Scanning $\beta$, true quality peaks at 0.717 around $\beta \approx 1.05$.
    - **Small $\beta$ = reward hacking.** At $\beta = 0.25$ the policy puts 0.947 of its mass on the flattery reply E. The reward-model score climbs to 2.945, but true quality falls to −0.368, *worse* than the model we started with. As $\beta \to 0$ the policy collapses onto E entirely (score 3.0, quality −0.5). The purple and green curves on the right are Goodhart's law in one picture.
    - **Tick "fix the reward model's mistake on E"**: now the two curves move together, and loosening the leash keeps helping (true quality 1.976 at $\beta = 0.25$, nearly all mass on A). The KL penalty is insurance against *the reward model's errors*, which in real life you can't see.
    - Notice $\pi^*$ can only move mass between replies that $\pi_{\text{ref}}$ already gives some probability to: a reply with $\pi_{\text{ref}} = 0$ stays at 0. RLHF sharpens and re-ranks what the SFT model can already say. It doesn't teach it things it could never produce.

---

## Step 6 — DPO: skip the reward model and the RL loop

RLHF with PPO works, but it is heavy: two training stages, four networks in memory, sampling replies inside the training loop, and PPO's many hyperparameters. **Direct Preference Optimization** (DPO; Rafailov et al., 2023) asks: since we know the exact form of the optimal policy, can we skip straight to it?

The trick is to read the closed form from Step 5 **backwards**. Take logs and solve for the reward:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Any policy *implies* a reward: $\beta$ times its log-ratio to the reference, plus a term that depends only on the prompt. Now plug this into the Bradley–Terry model from Step 4. Bradley–Terry only uses **differences** of rewards for the same prompt, so the troublesome $\beta \log Z(x)$ **cancels**:

$$
P(y_w \succ y_l \mid x) = \sigma\Big(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)
$$

Replace $\pi^*$ by our trainable policy $\pi_\theta$ and fit it to the human comparisons by maximum likelihood, exactly as we fitted the reward model:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\Big(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)\Big]
$$

That's it: **the reward model loss with the reward replaced by $\beta \log \pi_\theta / \pi_{\text{ref}}$**. The policy *is* the reward model ("your language model is secretly a reward model" is the paper's subtitle). Training needs only the log-probabilities of the two fixed replies under $\pi_\theta$ and $\pi_{\text{ref}}$: no sampling, no critic, no reward model. Its gradient pushes $\log \pi_\theta(y_w)$ up and $\log \pi_\theta(y_l)$ down, weighted by $\sigma(\hat r_l - \hat r_w)$, i.e. by how badly the implicit rewards $\hat r = \beta \log \pi_\theta / \pi_{\text{ref}}$ currently rank the pair.

**Try it:** the same five replies and the same $\pi_{\text{ref}}$ as above. The training data are 200 human comparisons (20 for each pair of replies), drawn from the hidden true quality. Each click runs plain gradient descent (learning rate 1) on the DPO loss.

<div id="viz-dpo" class="viz"><div class="viz-title">🎯 Interactive: DPO on five candidate replies</div></div>

!!! tip "What to notice"
    - **Step 0:** the policy equals $\pi_{\text{ref}}$, every implicit reward is 0 and the loss is $\log 2 = 0.693$, just like an untrained reward model.
    - **With $\beta = 1$:** after 10 steps A has grown from 0.15 to 0.377 and C has shrunk from 0.30 to 0.126. After 200 steps the policy is $[0.65, 0.268, 0.028, 0.039, 0.015]$ and the loss has levelled off at 0.388.
    - **It lands on the green ticks.** Fit a Bradley–Terry reward model $\hat r$ to the same 200 comparisons and compute $\pi_{\text{ref}} \exp(\hat r/\beta)/Z$: you get $[0.651, 0.267, 0.028, 0.039, 0.015]$, the same policy. DPO and "reward model + exact KL-regularised RL" reach the same answer on this toy, which is the claim of the DPO paper.
    - **$\beta$ is still the leash.** $\beta = 0.5$ ends further from $\pi_{\text{ref}}$ (A at 0.884, KL 1.485, true quality 1.877); $\beta = 2$ stays closer (A at 0.405, KL 0.295, true quality 0.773).
    - The flattery reply E never gets a chance here: humans compared it directly and it lost most of its comparisons, and there is no separate reward model to fool. That doesn't make DPO immune to over-optimisation, though; with few or one-sided comparisons it can also drift far from $\pi_{\text{ref}}$ in ways the data don't justify.

---

## Step 7 — RLHF with PPO vs DPO, and what came next

| | **RLHF with PPO** | **DPO** |
|---|---|---|
| What is trained | a reward model, then the policy (plus a critic) | the policy only |
| Networks in memory during training | policy, reference, reward model, critic | policy, reference |
| Data used while training the policy | fresh replies **sampled from the current policy** (on-policy) | a **fixed** dataset of (prompt, winner, loser) |
| Loss | PPO's clipped policy-gradient objective on $r - \beta\,\mathrm{KL}$ | a logistic loss on log-probability ratios |
| Strengths | can keep exploring and score new replies; a reward model can be reused | simple, stable, cheap; ordinary supervised training code |
| Weak spots | complex and sensitive to hyperparameters; reward hacking of the reward model | can't score replies outside its dataset; depends heavily on the quality and coverage of the comparisons |

Neither one is "the winner": both are widely used, and they solve the same objective in theory. Other well-known variations change one ingredient each:

- **Who labels?** In **RLAIF**, an AI model produces the preference labels instead of people. **Constitutional AI** (Bai et al., 2022) does this guided by a short written list of principles.
- **Which reward?** For maths or code, the reward can come from a **checker** (is the final answer right? do the tests pass?) instead of a learned model, which removes one source of reward hacking.
- **Which RL algorithm?** **GRPO** (Shao et al., 2024, in the DeepSeekMath paper) drops PPO's critic: it samples a group of replies per prompt and uses the group's average reward as the baseline, the same baseline idea as in REINFORCE.
- **Which preference loss?** Many DPO variants (IPO, KTO, ...) change the loss, for example to cope with noisy labels or with single "thumbs up / thumbs down" ratings instead of pairs.

---

## Putting it all together

Both routes start from the same human comparisons and the same SFT model, and aim for the same KL-regularised optimum $\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔀 Two routes from human preferences to an aligned policy",
  "cols": 3,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "prefs", "col": 0, "row": 0, "icon": "👥", "label": "Preference pairs", "sub": "(x, y<sub>w</sub>, y<sub>l</sub>) + π<sub>ref</sub>", "role": "data", "note": "Prompts, a preferred and a rejected reply for each, and the SFT model as the reference."},
    {"id": "rm", "col": 1, "row": 0, "icon": "⚖️", "label": "Reward model r<sub>φ</sub>", "sub": "−log σ(r<sub>w</sub> − r<sub>l</sub>)", "role": "loss"},
    {"id": "ppo", "col": 2, "row": 0, "icon": "🔁", "label": "PPO loop", "sub": "sample, score, update", "role": "model", "note": "Maximise r_φ(x, y) − β·KL(π_θ ‖ π_ref) with on-policy samples."},
    {"id": "dpo", "col": 0, "row": 1, "icon": "🎯", "label": "DPO loss", "sub": "−log σ(β·Δ log-ratio)", "role": "loss", "note": "The reward model's loss with r replaced by β log π_θ/π_ref. Plain supervised training."},
    {"id": "policy", "col": 2, "row": 1, "icon": "✅", "label": "Aligned policy", "sub": "≈ π<sub>ref</sub>·exp(r/β)/Z", "shape": "pill", "role": "opt"}
  ],
  "edges": [
    {"from": "prefs", "to": "rm"},
    {"from": "rm", "to": "ppo"},
    {"from": "ppo", "to": "policy"},
    {"from": "prefs", "to": "dpo"},
    {"from": "dpo", "to": "policy", "label": "one loss"}
  ],
  "steps": [
    {"show": ["prefs"], "text": "Both routes start from human comparisons and the SFT model π<sub>ref</sub>."},
    {"show": ["prefs>rm", "rm"], "text": "<b>RLHF:</b> fit a reward model with the Bradley–Terry loss."},
    {"show": ["rm>ppo", "ppo"], "text": "Then run PPO: the policy writes replies, the reward model scores them, a KL penalty keeps the policy near π<sub>ref</sub>."},
    {"show": ["ppo>policy", "policy"], "text": "The result approximates the optimum π<sub>ref</sub>·exp(r/β)/Z."},
    {"show": ["prefs>dpo", "dpo"], "text": "<b>DPO:</b> use the closed form backwards, so the policy's own log-ratios act as the reward. Same Bradley–Terry loss, no reward model."},
    {"show": ["dpo>policy", "policy"], "text": "A single supervised loss reaches the same optimum, without sampling or a critic."}
  ],
  "narrow": {"cols": 2.4, "rows": 3, "hideSub": true, "pos": {"prefs": [0, 0], "rm": [1.4, 0], "ppo": [1.4, 1], "policy": [1.4, 2], "dpo": [0, 2]}}
}
</script>
</div>

Here is DPO on the five-reply toy from the widget, with a compact `dpo_loss`. In a real model, `logp_w` would be the sum of token log-probabilities of the winning reply (Step 2), computed by the policy, and `ref_logp_w` the same under the frozen SFT model:

```python
import itertools
import torch
import torch.nn.functional as F

torch.manual_seed(0)
names = ["A simple", "B technical", "C wrong", "D refuses", "E flattery"]
ref = torch.tensor([0.15, 0.20, 0.30, 0.30, 0.05])     # π_ref: the SFT model's probabilities
quality = torch.tensor([2.0, 1.0, -1.5, -1.0, -0.5])   # hidden human taste (never shown to DPO)

# 20 human comparisons for every pair of responses
winners, losers = [], []
for i, j in itertools.combinations(range(5), 2):
    for _ in range(20):
        i_wins = torch.rand(()) < torch.sigmoid(quality[i] - quality[j])
        winners.append(i if i_wins else j); losers.append(j if i_wins else i)
W, L = torch.tensor(winners), torch.tensor(losers)


def dpo_loss(logp_w, logp_l, ref_logp_w, ref_logp_l, beta):
    """-log sigmoid( beta * [log π/π_ref (y_w) - log π/π_ref (y_l)] )"""
    h = beta * ((logp_w - ref_logp_w) - (logp_l - ref_logp_l))
    return -F.logsigmoid(h).mean()


beta = 1.0
theta = ref.log().clone().requires_grad_(True)          # the policy's logits; start at π_ref
opt = torch.optim.SGD([theta], lr=1.0)
for step in range(201):
    logp = torch.log_softmax(theta, dim=0)
    loss = dpo_loss(logp[W], logp[L], ref.log()[W], ref.log()[L], beta)
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (0, 10, 50, 200):
        print(f"step {step:3d}  loss {loss.item():.3f}  π = {[round(p, 3) for p in logp.exp().tolist()]}")
```

```text
step   0  loss 0.693  π = [0.15, 0.2, 0.3, 0.3, 0.05]
step  10  loss 0.448  π = [0.377, 0.313, 0.126, 0.147, 0.038]
step  50  loss 0.390  π = [0.592, 0.293, 0.041, 0.054, 0.019]
step 200  loss 0.388  π = [0.65, 0.268, 0.028, 0.039, 0.015]
```

For real models, Hugging Face's [TRL](https://huggingface.co/docs/trl) library implements reward-model training, PPO, DPO and GRPO trainers on top of `transformers`.

---

## Common misconceptions

!!! warning "\"RLHF teaches the model new knowledge.\""
    Mostly it doesn't. The optimal policy $\pi_{\text{ref}} \exp(r/\beta)/Z$ only **reweights** replies the starting model could already produce. RLHF changes *which* of its possible behaviours the model prefers (helpful, honest, on-topic, polite), not what it knows. Knowledge comes from pre-training.

!!! warning "\"The reward model knows what humans want.\""
    It is a learned approximation from a limited set of comparisons, and it can be confidently wrong on replies unlike its training data, such as the flattery reply in the widget. Optimise it too hard and you get reward hacking. That's why there is a KL penalty, and why teams keep collecting fresh comparisons.

!!! warning "\"DPO isn't really RL, so it has nothing to do with this course.\""
    DPO is trained like supervised learning, but it solves the **same KL-regularised RL objective** as RLHF; its loss is derived from that objective's exact solution. On the toy above it reaches the same policy as "reward model + optimal RL". It is RL's answer, computed in closed form.

!!! warning "\"A smaller β is always better because the model improves more.\""
    A smaller $\beta$ lets the policy move further from $\pi_{\text{ref}}$, and the **proxy** reward always goes up. Whether the *true* quality goes up depends on whether the reward (or the preference data) is right in the region the policy moves into. In the KL widget, true quality peaks at a moderate $\beta$ and then falls.

!!! warning "\"Rewards from the reward model are on a meaningful absolute scale.\""
    Bradley–Terry only constrains **differences**: adding a constant to every score changes nothing. A reward of 3.2 means nothing alone; it only means something compared with other replies to the same prompt.

---

## Check yourself

??? question "1. In the RL view of a language model, what are the state, the action, the episode and the reward?"
    **State:** the prompt plus all tokens generated so far. **Action:** the next token (from the whole vocabulary). **Episode:** generating one complete reply. **Reward:** in RLHF, a single score from the reward model for the finished reply (plus a KL penalty), so rewards are sparse and delayed.

??? question "2. The reward model gives reply A a score of 3 and reply B a score of 1. What probability does Bradley–Terry give to humans preferring A, and what is the loss if they actually preferred B?"
    $\Delta = 3 - 1 = 2$, so $P(A \succ B) = \sigma(2) = 0.881$. If humans chose B, the loss is $-\log \sigma(-2) = 2.127$: a confident wrong prediction is punished heavily.

??? question "3. What would happen in RLHF if you set β = 0?"
    Nothing would stop the policy from drifting as far as it likes to maximise the reward model's score. It would find replies the reward model over-rates (reward hacking) and could collapse onto a few of them, losing fluency and diversity. In the KL widget, very small $\beta$ puts essentially all mass on the flattery reply E, with true quality −0.5, worse than the starting model.

??? question "4. Why does the partition function Z(x) disappear in the DPO loss, and why does that matter?"
    DPO writes the reward as $\beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$. Bradley–Terry uses only the **difference** of two rewards for the **same prompt**, so the $\beta \log Z(x)$ terms cancel. That matters because $Z(x)$ is a sum over every possible reply and can't be computed; without the cancellation the DPO loss would be intractable.

??? question "5. Give one advantage of RLHF with PPO over DPO, and one advantage of DPO over RLHF with PPO."
    **PPO:** it is on-policy, so it scores fresh replies from the current model and can improve beyond the replies in a fixed dataset; the reward model can also be reused. **DPO:** it is much simpler and cheaper, with no reward model, no critic and no sampling during training, and it trains stably with ordinary supervised code.

---

## Further reading

- Paul Christiano et al., [*Deep Reinforcement Learning from Human Preferences*](https://arxiv.org/abs/1706.03741) (2017). The original idea: learn a reward from comparisons of short clips, then run RL on it.
- Nisan Stiennon et al., [*Learning to Summarize from Human Feedback*](https://arxiv.org/abs/2009.01325) (2020). The reward model + PPO + KL recipe applied to language.
- Long Ouyang et al., [*Training Language Models to Follow Instructions with Human Feedback*](https://arxiv.org/abs/2203.02155) (2022). InstructGPT: SFT → reward model → PPO.
- Rafael Rafailov et al., [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290) (2023). The DPO derivation in full.
- Leo Gao, John Schulman & Jacob Hilton, [*Scaling Laws for Reward Model Overoptimization*](https://arxiv.org/abs/2210.10760) (2022). Goodhart's law measured.
- Yuntao Bai et al., [*Constitutional AI: Harmlessness from AI Feedback*](https://arxiv.org/abs/2212.08073) (2022). AI-generated preference labels guided by written principles.
- Nathan Lambert, [*Reinforcement Learning from Human Feedback*](https://rlhfbook.com/) (free online book). A thorough, practical overview.

**Next lessons:** this is the last lesson of the reinforcement-learning series. Continue with the [Generative AI](../../roadmap.md#generative-ai) section of the roadmap, starting with *Large language models* (coming soon).

<script src="viz/rlhf-dpo.js"></script>
