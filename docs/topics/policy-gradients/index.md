# Policy Gradients and REINFORCE

!!! abstract "TL;DR"
    Instead of learning how good each action is and then picking the best one, a **policy-gradient** method learns the **policy itself**: a network with weights $\theta$ that outputs action probabilities $\pi_\theta(a \mid s)$. It plays episodes and then nudges $\theta$ by **gradient ascent** so that actions followed by a high return become **more likely**, and actions followed by a low return become less likely. **REINFORCE** is the simplest version. Subtracting a **baseline** (the return you expected anyway) keeps it correct and makes it far less noisy.

**Level:** intermediate · **Prerequisites:** [Deep Q-Networks](../deep-q-networks/index.md), [Gradient Descent](../gradient-descent/index.md)

---

## Why should I care?

Value-based methods like Q-learning and DQN answer "how good is each action?" and then act greedily. That works brilliantly for Atari, where there are a handful of buttons. But:

- **A robot arm has continuous actions** (a torque for every joint). You can't take an $\arg\max$ over infinitely many actions at every step.
- **Sometimes the best policy is random.** In rock–paper–scissors any predictable strategy loses. Greedy action selection can only ever produce deterministic policies.
- **RLHF fine-tunes language models with policy gradients.** The "policy" is the chatbot, the "action" is the next word, and PPO (the next lesson) is a policy-gradient method. Everything in that pipeline starts with the idea in this lesson.

Policy gradients are the foundation of almost every modern deep-RL algorithm for control and for language models: A2C, PPO, SAC, and the RL step of RLHF.

---

## Step 1 — Why learn the policy directly?

Recall from [lesson 1](../reinforcement-learning-intro/index.md#step-4-the-agents-strategy-the-policy) that a policy can be **deterministic** (one action per state) or **stochastic** (a probability for each action). A value-based agent like DQN learns $Q(s, a)$ and then picks $\arg\max_a Q(s, a)$, plus some $\varepsilon$-greedy randomness for exploration. Its policy is a by-product of the values.

A **policy-based** agent skips the middleman. It has a policy with adjustable knobs and tunes those knobs directly to get more reward.

| | Value-based (Q-learning, DQN) | **Policy-based** (REINFORCE) |
|---|---|---|
| What is learned | $Q(s,a)$, then act greedily | $\pi_\theta(a \mid s)$ directly |
| Kind of policy | deterministic (+ ε for exploring) | naturally stochastic |
| Continuous actions | hard: needs $\arg\max$ over all actions | easy: output a distribution (e.g. a Gaussian) |
| Small change in parameters | can flip the $\arg\max$ and change behaviour abruptly | changes probabilities smoothly |
| Data | can reuse old experience (replay buffer) | needs fresh episodes from the current policy |

Here is a tiny world where the value-based approach gets stuck. In Sutton & Barto's **short corridor**, every step costs −1 and the episode ends at the goal on the right. The catch: in the second cell the actions are **reversed** ("right" moves left), and the agent **can't tell the cells apart** (it sees the same observation in all three). So its only choice is one number: $p$, the probability of pressing "right".

**Try it:** drag $p$ and watch the value $J(p)$ of the policy (the average total reward from the start). Play a few episodes at different $p$.

<div id="viz-corridor" class="viz"><div class="viz-title">🚪 Interactive: the short corridor, where the best policy is random</div></div>

!!! tip "What to notice"
    - **Both deterministic policies fail forever.** Always "right" bounces between the first two cells; always "left" never leaves the start. $J = -\infty$.
    - **ε-greedy is not much better.** With $\varepsilon = 0.1$ and two actions, the greedy action gets probability 0.95. That gives $J = -82.1$ (greedy = left) or $J = -44.2$ (greedy = right).
    - **The best policy flips a biased coin**: $p = 2 - \sqrt{2} \approx 0.59$, worth $J \approx -11.66$. Only a method that can learn *probabilities* can find it.
    - The purple curve is smooth. If we could compute its slope, we could simply **climb it**. That is exactly what a policy gradient does.

??? note "Where does the curve come from?"
    Write $V_0, V_1, V_2$ for the values of the three cells under the policy. Each is "−1 for this step, plus the value of where you land":
    $V_0 = -1 + pV_1 + (1-p)V_0$, $\;V_1 = -1 + pV_0 + (1-p)V_2$, $\;V_2 = -1 + (1-p)V_1$ (the goal is worth 0).
    Solving these three equations gives

    $$J(p) = V_0 = -\frac{2(2-p)}{p(1-p)},$$

    which is largest at $p = 2-\sqrt{2}$, where $J = -(6+4\sqrt{2}) \approx -11.66$. We checked this by solving the equations numerically on a fine grid of $p$ values: the best grid point is $p = 0.58579$ with $J = -11.6569$.

---

## Step 2 — A policy with knobs

A **parameterised policy** is a function with weights $\theta$ that takes a state and returns a probability for every action. For discrete actions the standard recipe is:

1. A network (any network) turns the state into one score per action, called an **action preference** $h_\theta(s, a)$.
2. A **softmax** turns the scores into probabilities:

$$
\pi_\theta(a \mid s) = \frac{e^{h_\theta(s,a)}}{\sum_{b} e^{h_\theta(s,b)}}
$$

Higher preference → higher probability, all probabilities are positive, and they add up to 1. The agent then **samples** its action from these probabilities, so it explores automatically.

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧭 A softmax policy network",
  "cols": 3,
  "rows": 2,
  "rowHeight": 105,
  "nodes": [
    {"id": "s", "col": 0, "row": 0, "icon": "👀", "label": "State s", "sub": "e.g. pole angle", "role": "data", "note": "Whatever the agent observes: positions, pixels, the conversation so far."},
    {"id": "net", "col": 1, "row": 0, "icon": "🧠", "label": "Network with weights θ", "role": "model", "note": "Any network: a small MLP for CartPole, a CNN for pixels, a transformer for text."},
    {"id": "h", "col": 2, "row": 0, "icon": "📊", "label": "Preferences h(s, a)", "sub": "one score per action", "role": "model", "note": "Raw scores (logits). They can be any real number."},
    {"id": "soft", "col": 2, "row": 1, "icon": "🔀", "label": "Softmax", "sub": "e<sup>h</sup> / Σ e<sup>h</sup>", "note": "Exponentiate and normalise, so the scores become positive and add up to 1."},
    {"id": "pi", "col": 1, "row": 1, "icon": "🎲", "label": "Probabilities π<sub>θ</sub>(a | s)", "role": "model", "note": "For example 0.2 / 0.5 / 0.3 over three actions."},
    {"id": "a", "col": 0, "row": 1, "icon": "🎯", "label": "Sample action a", "shape": "pill", "role": "model", "note": "Random, not argmax: that is what makes the policy explore by itself."}
  ],
  "edges": [
    {"from": "s", "to": "net"},
    {"from": "net", "to": "h"},
    {"from": "h", "to": "soft"},
    {"from": "soft", "to": "pi"},
    {"from": "pi", "to": "a"}
  ],
  "steps": [
    {"show": ["s"], "text": "The agent observes the state s."},
    {"show": ["s>net", "net"], "text": "A network with weights θ processes it. θ are the knobs that learning will turn."},
    {"show": ["net>h", "h"], "text": "It outputs one <b>preference</b> per action: a raw score that can be any number."},
    {"show": ["h>soft", "soft"], "text": "The softmax exponentiates and normalises the scores."},
    {"show": ["soft>pi", "pi"], "text": "Now we have a proper probability distribution π<sub>θ</sub>(a | s) over the actions."},
    {"show": ["pi>a", "a"], "text": "The agent <b>samples</b> its action. Changing θ a little changes these probabilities a little: no sudden jumps."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"s": [0, 0], "net": [1, 0], "h": [1, 1], "soft": [0, 1], "pi": [0, 2], "a": [1, 2]}}
}
</script>
</div>

For **continuous actions** (a torque, a steering angle) the network instead outputs the mean $\mu_\theta(s)$ (and often the spread $\sigma$) of a **Gaussian**, and the action is sampled from it: $a \sim \mathcal{N}(\mu_\theta(s), \sigma^2)$. Everything below works the same way; only the formula for $\ln \pi$ changes.

!!! tip "What to notice"
    - The policy is **stochastic by design**: even with fixed $\theta$, the same state can lead to different actions. Exploration is built in, no $\varepsilon$ needed.
    - The probabilities are a **smooth** function of $\theta$. That is what will let us use gradients.
    - In the short corridor, the "network" is just two numbers, $h(\text{right})$ and $h(\text{left})$, and $p = \pi(\text{right})$ is their softmax.

---

## Step 3 — The goal: climb the expected return

We want the policy whose episodes collect the most reward on average. So the **objective** is the expected return from the start:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\left[\, G_0 \,\right]
$$

This is the value of the start state, $V^{\pi_\theta}(s_0)$, seen as a function of the weights. In the short corridor it was the purple curve in Step 1.

To make $J$ **bigger**, use the update from the [Gradient Descent](../gradient-descent/index.md#step-4-taking-the-step-the-update-rule) lesson with the sign flipped. This is **gradient ascent**: walk *uphill*.

$$
\theta \leftarrow \theta + \alpha\, \nabla_\theta J(\theta)
$$

The problem: how do we compute $\nabla_\theta J$? $J$ depends on $\theta$ through the actions the policy picks, and on the environment through what happens next. We don't know the environment's rules (they might be a physics engine or the real world), so we can't differentiate through them. We can only **play episodes and look at the returns**. The next step shows that this is enough.

---

## Step 4 — The policy gradient: make good actions more likely

Here is the key result, the **policy gradient theorem**:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\, \sum_{t} G_t \, \nabla_\theta \ln \pi_\theta(a_t \mid s_t) \right]
$$

Read it in words, piece by piece:

- $\nabla_\theta \ln \pi_\theta(a_t \mid s_t)$ is the direction in weight space that makes the action we actually took, $a_t$, **more likely** in state $s_t$.
- $G_t$ is the return that followed that action: how well things went from then on.
- So the gradient says: **push up the log-probability of each action in proportion to the return that followed it.** Good outcome: take a big step towards repeating it. Bad outcome: a smaller step, or (if $G_t$ is negative) a step away from it.
- The expectation means we can **estimate it by sampling**: play an episode with the current policy and compute the sum. No model of the environment needed.

Why the log? The **log-derivative trick**: $\nabla_\theta \pi = \pi\, \nabla_\theta \ln \pi$. Dividing by $\pi$ means that actions the policy already picks often (which therefore show up often in our samples) don't get an unfair head start; rare actions that turn out well get a bigger push per sample.

For a softmax policy the gradient with respect to the preferences is very simple:

$$
\frac{\partial \ln \pi(a \mid s)}{\partial h(s, b)} = \mathbb{1}[a = b] - \pi(b \mid s)
$$

The sampled action's preference goes **up** by $1 - \pi(a)$; every other action's goes **down** by its probability. Example: $\pi = (0.2, 0.5, 0.3)$, we sampled the middle action and got return $G = +2$, with $\alpha = 0.1$. The update is $\Delta h = 0.1 \cdot 2 \cdot (-0.2,\, 0.5,\, -0.3) = (-0.04,\, +0.10,\, -0.06)$. (PyTorch's autograd gives the same gradient, $(-0.2, 0.5, -0.3)$.)

??? note "Where does the theorem come from? (a short derivation)"
    Write $\tau = (s_0, a_0, r_1, s_1, a_1, \dots)$ for a whole episode and $R(\tau)$ for its return. Then $J(\theta) = \sum_\tau P_\theta(\tau)\, R(\tau)$, and

    $$
    \nabla_\theta J = \sum_\tau \nabla_\theta P_\theta(\tau)\, R(\tau) = \sum_\tau P_\theta(\tau)\, \nabla_\theta \ln P_\theta(\tau)\, R(\tau) = \mathbb{E}\left[ R(\tau)\, \nabla_\theta \ln P_\theta(\tau) \right].
    $$

    The probability of an episode is a product: start state × policy choices × environment transitions, so its log is a sum:

    $$
    \ln P_\theta(\tau) = \ln p(s_0) + \sum_t \Big[ \ln \pi_\theta(a_t \mid s_t) + \ln p(s_{t+1} \mid s_t, a_t) \Big].
    $$

    Only the policy terms depend on $\theta$, so the unknown environment **drops out** of the gradient. Finally, an action can't affect rewards that came *before* it, so $R(\tau)$ can be replaced by the reward-to-go $G_t$ for each term (this removes noise without changing the expectation). With discounting, the exact theorem has an extra factor $\gamma^t$; practical code almost always drops it.

**Try it:** here is a softmax policy over three actions (a one-state problem, like a bandit). Each action pays its mean reward plus a little noise, so **every reward is positive** (around 4 to 6). Press *Sample an action & update* a few times and read the update in the readout. Then run 100 updates.

<div id="viz-softmax" class="viz"><div class="viz-title">🎰 Interactive: a REINFORCE update, one sample at a time</div></div>

!!! tip "What to notice"
    - **One update, by hand.** On run 1 the first sample is B with reward 6.60. All three probabilities start at $\tfrac13$, so the update is $\Delta h = 0.1 \cdot 6.60 \cdot (-\tfrac13, \tfrac23, -\tfrac13) = (-0.220, +0.440, -0.220)$: B goes up, the others go down.
    - **Without a baseline, every sampled action is reinforced**, because every reward is positive. Learning still works *on average* (better actions get pushed harder), but it is a lottery. Click **New run** (run 2) and then **Run 100 updates**: the policy settles on C, not the best action B ($\pi(C) = 0.89$).
    - Now tick **subtract a baseline** (it replays the same run). The weight becomes $r - b$: positive for better-than-average rewards, negative for worse ones. Run 100 updates again: run 2 now ends with $\pi(B) = 0.92$.
    - **Test 1000 runs** at $\alpha = 0.1$: $\pi(B)$ ends above 0.5 in 68.9% of runs without a baseline and 99.8% with one. Step 6 explains why.

---

## Step 5 — REINFORCE: the algorithm

**REINFORCE** (Williams, 1992) turns the theorem into an algorithm. It is a **Monte Carlo** method, like the Monte Carlo prediction from the earlier lessons: it waits until the episode is over, so that every return $G_t$ is known exactly, and then updates.

!!! note "REINFORCE"
    Repeat for many episodes:

    1. Play one episode with the current policy $\pi_\theta$: $s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T$.
    2. For every time step, compute the return that followed it: $G_t = r_{t+1} + \gamma\, r_{t+2} + \dots$
    3. Update the weights with one gradient-ascent step (the formula below).
    4. Throw the episode away and start again.

    $$
    \theta \leftarrow \theta + \alpha \sum_{t=0}^{T-1} G_t\, \nabla_\theta \ln \pi_\theta(a_t \mid s_t)
    $$

That's all. No value table, no replay buffer, no target network. In PyTorch you write the sum as a "loss" $-\sum_t G_t \ln \pi_\theta(a_t \mid s_t)$ and let the optimiser *minimise* it, which is the same as gradient ascent on $J$.

**Try it:** run REINFORCE on the short corridor, starting from the bad policy $p = 0.05$. Focus on the **red** curve for now (the green one is Step 6). Each curve is the average of 50 seeded runs. Change the step size $\alpha$.

<div id="viz-learn" class="viz"><div class="viz-title">📈 Interactive: REINFORCE learns the short corridor</div></div>

!!! tip "What to notice"
    - **It finds the random policy.** At $\alpha = 2^{-10}$ the red curve climbs from about $-80$ to the dashed line, the best possible $-11.66$. The runs end with an average return of $-12.2$ over their last 100 episodes. A value-based method with ε-greedy could never do better than $-44.2$ here.
    - **Small steps are slow.** At $2^{-13}$ it still gets there, but only towards the end of the 1000 episodes.
    - **Big steps are dangerous.** At $2^{-8}$, 2 of the 50 runs get **stuck** off the chart, and at $2^{-7}$, 32 of 50 do. One unlucky, very long episode produces a huge update that pushes $p$ almost to 0 or 1, and from there episodes become endless. The strip plot on the right shows every run's final score, so you can see the stragglers.
    - This sensitivity is a symptom of **noise**: each update is based on a single episode, and the returns vary wildly from episode to episode.

---

## Step 6 — Taming the noise: baselines

REINFORCE is **unbiased**: on average, its update points exactly along the true gradient. But each single estimate is very **noisy** (high **variance**). Look at the update again: every action is weighted by $G_t$. In the short corridor every return is negative, so *every* action gets pushed down, and only the *difference* between big and small pushes carries information. In the bandit above every reward was positive, so every action got pushed up. Most of each update is noise that has to cancel out over many episodes.

The fix is to compare each return with what you **expected** to get, and only react to the surprise. Subtract a **baseline** $b(s_t)$:

$$
\theta \leftarrow \theta + \alpha \sum_{t} \big(G_t - b(s_t)\big)\, \nabla_\theta \ln \pi_\theta(a_t \mid s_t)
$$

Better than expected → positive weight → make the action more likely. Worse than expected → negative weight → make it less likely.

**Why it doesn't break anything:** the baseline does not depend on the action, so its contribution averages to zero:

$$
\mathbb{E}_{a \sim \pi}\big[\, b(s)\, \nabla_\theta \ln \pi_\theta(a \mid s) \big] = b(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) = b(s)\, \nabla_\theta \underbrace{\sum_a \pi_\theta(a \mid s)}_{=\,1} = 0
$$

So the update is **still unbiased**, whatever baseline you choose. A good baseline just makes it much less noisy. The natural choice is the state value, $b(s) = V^\pi(s)$, learned by a second network (that's what the code below does).

**Try it:** each histogram shows 2000 single-episode estimates of the gradient for a fixed corridor policy $p$. Left: weight $G_t$. Right: weight $G_t - b$, with $b$ the average return $J(p)$. The purple line is the true gradient.

<div id="viz-variance" class="viz"><div class="viz-title">📊 Interactive: the same gradient, with less noise</div></div>

!!! tip "What to notice"
    - **Same centre, narrower spread.** At $p = 0.3$ the true gradient is $8.48$. The estimates average $7.16$ without a baseline and $7.73$ with one (both are within sampling noise of the truth), but the spread drops from $36.0$ to $21.0$, **42% less**.
    - Positive gradient means "increase $p$" (at $p = 0.3$ we are left of the optimum). Move the slider past 0.59 and the true gradient turns negative.
    - Now scroll back to the **learning curves** in Step 5 and look at the **green** curve. At $\alpha = 2^{-8}$ the baseline version has no stuck runs (average $-12.3$ over the last 100 episodes, against $-53.0$ for plain REINFORCE), and at $2^{-7}$ only 1 of 50 runs gets stuck instead of 32. Less noise lets you take bigger steps safely.
    - The baseline is not magic: at very small step sizes it doesn't help here (at $2^{-13}$ it even starts a little slower, because our learned baseline begins at 0 and has to catch up).

!!! note "From baselines to advantages"
    With $b(s) = V^\pi(s)$, the weight $G_t - V^\pi(s_t)$ is a sample estimate of the **advantage**

    $$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s),$$

    how much better action $a$ was than the policy's average behaviour in state $s$. If we also use the learned $V$ to estimate the return itself (instead of waiting for the episode to end), we get **actor-critic** methods, the topic of the next lesson.

---

## Step 7 — The price: on-policy and sample-hungry

The policy gradient is an expectation over episodes played **by the current policy**. As soon as we update $\theta$, the old episodes come from a different policy and no longer give a correct estimate. So REINFORCE is **on-policy**: it uses each episode for one update and throws it away. Compare DQN, which stores experience in a replay buffer and learns from each transition many times.

Together with the high variance, that makes vanilla REINFORCE **sample-hungry**: it often needs many thousands of episodes. Most of what comes next is about fixing this:

- **Actor-critic** (A2C) learns a value function to cut the variance further and updates during the episode.
- **PPO** reuses each batch of episodes for several careful updates, keeping the new policy close to the old one.
- **SAC** brings back a replay buffer for continuous control.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧩 REINFORCE with a baseline",
  "cols": 3,
  "rows": 5.15,
  "rowHeight": 100,
  "groups": [
    {"id": "loop", "label": "Repeat for many episodes", "col": 0, "row": 1.15, "cols": 3, "rows": 2.6}
  ],
  "nodes": [
    {"id": "init", "col": 1, "row": 0, "icon": "🎲", "label": "Random policy weights θ, baseline weights w", "shape": "pill", "maxWidth": 230},
    {"id": "play", "col": 0, "row": 1.6, "icon": "🎮", "label": "Play an episode with π<sub>θ</sub>", "role": "model", "note": "Sample every action from the current policy until the episode ends."},
    {"id": "ret", "col": 1, "row": 1.6, "icon": "🧮", "label": "Returns G<sub>t</sub>", "sub": "for every step t", "role": "data", "note": "G_t = r_{t+1} + γ r_{t+2} + …, computed backwards from the end of the episode."},
    {"id": "adv", "col": 2, "row": 1.6, "icon": "⚖️", "label": "G<sub>t</sub> − b(s<sub>t</sub>)", "sub": "better than expected?", "role": "loss", "note": "Positive: the action did better than the baseline expected. Negative: worse."},
    {"id": "pol", "col": 2, "row": 2.75, "icon": "⬆️", "label": "Policy step", "sub": "θ += α Σ weight · ∇ln π", "role": "opt", "note": "Gradient ascent: make better-than-expected actions more likely, worse ones less likely."},
    {"id": "base", "col": 0, "row": 2.75, "icon": "📏", "label": "Fit baseline b(s) to G<sub>t</sub>", "role": "loss", "note": "Regression: move the value network's prediction V_w(s_t) towards the observed G_t."},
    {"id": "done", "col": 1, "row": 4.15, "icon": "🏁", "label": "Trained policy", "shape": "pill", "role": "opt"}
  ],
  "edges": [
    {"from": "init", "to": "loop"},
    {"from": "play", "to": "ret"},
    {"from": "ret", "to": "adv"},
    {"from": "adv", "to": "pol"},
    {"from": "pol", "to": "base"},
    {"from": "base", "to": "play", "label": "next episode"},
    {"from": "loop", "to": "done", "label": "return is high"}
  ],
  "steps": [
    {"show": ["init"], "text": "Start with a random policy network θ (and, for the baseline, a value network w)."},
    {"show": ["init>loop", "loop", "play"], "text": "Play one whole episode, <b>sampling</b> every action from the current policy."},
    {"show": ["play>ret", "ret"], "text": "Go backwards through the episode and compute the return G<sub>t</sub> that followed each action."},
    {"show": ["ret>adv", "adv"], "text": "Subtract the baseline: was each return better or worse than expected from that state?"},
    {"show": ["adv>pol", "pol"], "text": "One gradient-ascent step on Σ<sub>t</sub> (G<sub>t</sub> − b(s<sub>t</sub>)) ln π<sub>θ</sub>(a<sub>t</sub> | s<sub>t</sub>). Without a baseline, the weight is just G<sub>t</sub>."},
    {"show": ["pol>base", "base"], "text": "Improve the baseline: regress V<sub>w</sub>(s<sub>t</sub>) towards the observed returns."},
    {"show": ["base>play", "play"], "text": "Throw the episode away (it came from the old policy) and play a fresh one."},
    {"show": ["loop>done", "done"], "text": "After enough episodes the average return stops improving: that's the trained policy."}
  ],
  "narrow": {"cols": 2, "rows": 6.35, "pos": {"init": [0.5, 0], "loop": [0, 1.15, 2, 3.8], "play": [0, 1.6], "ret": [1, 1.6], "adv": [1, 2.75], "pol": [1, 3.9], "base": [0, 3.9], "done": [0.5, 5.35]}}
}
</script>
</div>

Here is REINFORCE, with and without a learned baseline, on **CartPole**: push a cart left or right to keep a pole upright, +1 for every step it stays up, at most 500 steps. The environment is written out in plain Python (it's the standard CartPole physics), so you only need PyTorch.

```python
import math, random
import numpy as np
import torch
import torch.nn as nn


class CartPole:
    """The classic pole-balancing task: +1 per step, at most 500 steps."""

    def reset(self):
        self.s = np.random.uniform(-0.05, 0.05, size=4)   # cart pos, cart vel, pole angle, pole ang. vel
        self.t = 0
        return self.s.astype(np.float32)

    def step(self, action):                               # action 0 = push left, 1 = push right
        x, x_dot, th, th_dot = self.s
        force = 10.0 if action == 1 else -10.0
        cos, sin = math.cos(th), math.sin(th)
        temp = (force + 0.05 * th_dot**2 * sin) / 1.1
        th_acc = (9.8 * sin - cos * temp) / (0.5 * (4 / 3 - 0.1 * cos**2 / 1.1))
        x_acc = temp - 0.05 * th_acc * cos / 1.1
        self.s = np.array([x + 0.02 * x_dot, x_dot + 0.02 * x_acc,
                           th + 0.02 * th_dot, th_dot + 0.02 * th_acc])
        self.t += 1
        fell = abs(self.s[0]) > 2.4 or abs(self.s[2]) > 0.2095
        return self.s.astype(np.float32), 1.0, fell or self.t >= 500


def returns_to_go(rewards, gamma):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return torch.tensor(out[::-1])


def train(use_baseline, seed, episodes=600, gamma=0.99):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = CartPole()
    policy = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))   # state -> action preferences
    value = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))    # baseline b(s) = V(s)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=1e-3)
    opt_v = torch.optim.Adam(value.parameters(), lr=1e-3)
    history = []
    for ep in range(episodes):
        # 1. play one episode with the current policy
        states, actions, rewards = [], [], []
        s, done = env.reset(), False
        while not done:
            with torch.no_grad():
                probs = torch.softmax(policy(torch.from_numpy(s)), dim=-1)
            a = torch.multinomial(probs, 1).item()
            states.append(s); actions.append(a)
            s, r, done = env.step(a)
            rewards.append(r)
        history.append(sum(rewards))

        # 2. the return G_t from every time step
        S = torch.from_numpy(np.array(states))
        A = torch.tensor(actions)
        G = returns_to_go(rewards, gamma)

        # 3. weight for each log-probability: G_t, or G_t - b(s_t)
        if use_baseline:
            V = value(S).squeeze(-1)
            weight = G - V.detach()
            opt_v.zero_grad(); ((G - V) ** 2).mean().backward(); opt_v.step()
        else:
            weight = G

        # 4. gradient ASCENT on  sum_t weight_t * log pi(a_t | s_t)
        log_probs = torch.log_softmax(policy(S), dim=-1)[torch.arange(len(A)), A]
        loss = -(weight * log_probs).sum()              # minus sign: optimisers minimise
        opt_pi.zero_grad(); loss.backward(); opt_pi.step()
    return history


torch.set_num_threads(1)
for use_baseline in [False, True]:
    history = train(use_baseline, seed=0)
    print(f"baseline={use_baseline!s:5}  average return: first 100 episodes {np.mean(history[:100]):5.1f}, "
          f"last 100 episodes {np.mean(history[-100:]):5.1f}")
```

```text
baseline=False  average return: first 100 episodes  38.4, last 100 episodes 348.1
baseline=True   average return: first 100 episodes  35.2, last 100 episodes 483.4
```

Both versions learn to balance the pole (a random policy lasts only 22 steps on average), and both runs together take about 35 seconds on one CPU core. One seed can be lucky, so we also trained each version with 10 different seeds. Averaged over the last 100 episodes, plain REINFORCE reached **376.8** (worst seed 163.1) and REINFORCE with a baseline **439.4** (worst seed 347.0). The baseline won on 8 of the 10 seeds.

!!! note "Notes on the code"
    - `V.detach()`: the baseline only *scales* the policy update. Without `detach()`, the policy loss would also try to change the value network.
    - The "loss" is not a measure of how good the policy is: its value can go up while the policy improves. Only its **gradient** means something.
    - Real implementations add a few tricks: normalising the weights in each batch, batching several episodes per update, and an entropy bonus that keeps the policy from becoming deterministic too early.

---

## Common misconceptions

!!! warning "\"The PyTorch loss in REINFORCE is a loss like in supervised learning.\""
    It is a trick to get the right gradient out of autograd. Its value depends on the sampled episode and the size of the returns, and it doesn't go down steadily as the agent improves. Track the **average return** instead.

!!! warning "\"Subtracting a baseline changes what the algorithm learns.\""
    It doesn't: the baseline term averages to exactly zero (Step 6), so the expected update is the same. It only reduces the noise. That is why any reasonable baseline is allowed, even a bad one.

!!! warning "\"If every reward is positive, every action gets reinforced, so the agent can't learn.\""
    Every *sampled* action is pushed up, but better actions are pushed up *more*, so on average the policy still improves. It is just very noisy and can lock onto a worse action, as in run 2 of the bandit widget. A baseline fixes that.

!!! warning "\"Stochastic policies are only for exploration; the best policy is always deterministic.\""
    That is true in a fully observed MDP, but not when the agent can't see everything (the short corridor) or in games against opponents (rock–paper–scissors). Policy-gradient methods can represent and find the best random policy.

!!! warning "\"Policy gradient finds the best policy.\""
    It is gradient ascent, so it finds a **local** optimum of $J(\theta)$. With neural networks, $J$ has many hills, and an unlucky run can settle on a mediocre one.

---

## Check yourself

??? question "1. A softmax policy has π = (0.2, 0.5, 0.3). You sample the second action and the return is G = +2. With α = 0.1, how do the three preferences change?"
    $\Delta h = \alpha\, G\, (\mathbb{1}[a=b] - \pi(b)) = 0.1 \cdot 2 \cdot (-0.2,\, 0.5,\, -0.3) = $ **(−0.04, +0.10, −0.06)**. The sampled action goes up, the other two go down.

??? question "2. Why can't you just take the gradient of the return through the environment, as you would through a loss function?"
    The return depends on what the environment does after each action, and we don't have a differentiable formula for that (it may be a physics engine, a game or the real world). The policy gradient theorem avoids this: after taking the log, the environment's transition probabilities drop out, and only $\nabla_\theta \ln \pi_\theta(a_t \mid s_t)$ remains, which we *can* differentiate.

??? question "3. Show that subtracting a baseline b(s) doesn't change the expected update."
    $\mathbb{E}_{a\sim\pi}[b(s)\nabla_\theta \ln \pi_\theta(a\mid s)] = b(s)\sum_a \pi_\theta(a\mid s)\frac{\nabla_\theta \pi_\theta(a\mid s)}{\pi_\theta(a\mid s)} = b(s)\,\nabla_\theta \sum_a \pi_\theta(a \mid s) = b(s)\,\nabla_\theta 1 = 0$. It works because $b$ doesn't depend on the action. A "baseline" that depends on the action taken would bias the update.

??? question "4. Why can't REINFORCE learn from a replay buffer of old episodes, like DQN does?"
    The policy gradient is an average over episodes played **by the current policy** $\pi_\theta$. Old episodes were played by older policies, so they over-represent actions the current policy no longer likes, and the estimate is wrong. REINFORCE is on-policy. (PPO and other methods correct for this with importance weights, within limits.)

??? question "5. A Gaussian policy for a continuous action has mean μ = 0.5 and σ = 0.2. It samples a = 0.8, and the outcome is better than expected. Which way does the update move μ?"
    For a Gaussian, $\frac{\partial \ln \pi(a)}{\partial \mu} = \frac{a - \mu}{\sigma^2} = \frac{0.3}{0.04} = 7.5$ (autograd agrees). The weight $G_t - b$ is positive, so μ moves **up, towards 0.8**: the policy shifts towards the action that worked well.

---

## Further reading

- Richard Sutton & Andrew Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed.), Chapter 13: Policy Gradient Methods. The short corridor is Example 13.1.
- Ronald Williams, [*Simple statistical gradient-following algorithms for connectionist reinforcement learning*](https://link.springer.com/article/10.1007/BF00992696), Machine Learning (1992). The original REINFORCE paper.
- Richard Sutton et al., [*Policy Gradient Methods for Reinforcement Learning with Function Approximation*](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) (NeurIPS 1999). The policy gradient theorem.
- OpenAI, [*Spinning Up: Intro to Policy Optimization*](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html), a clear derivation with a minimal PyTorch implementation.
- Andrej Karpathy, [*Deep Reinforcement Learning: Pong from Pixels*](https://karpathy.github.io/2016/05/31/rl/), policy gradients explained through a game.
- Lilian Weng, [*Policy Gradient Algorithms*](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/), a tour from REINFORCE to modern methods.

**Next lessons:** [Actor-critic methods: A2C, PPO and SAC](../actor-critic/index.md)

<script src="viz/policy-gradients.js"></script>
