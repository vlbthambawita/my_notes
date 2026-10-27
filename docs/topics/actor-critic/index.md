# Actor-Critic Methods: A2C, PPO and SAC

!!! abstract "TL;DR"
    An **actor-critic** agent has two parts. The **actor** is a policy that picks actions. The **critic** is a learned value function that judges them. After every step the critic computes the **TD error** $\delta$: "was that better or worse than I expected?". The actor then makes that action more likely if $\delta > 0$ and less likely if $\delta < 0$. This one idea, with a few safety rails added, gives **A2C**, **PPO** (the algorithm behind RLHF for chatbots) and **SAC** (a favourite for robot control).

**Level:** intermediate · **Prerequisites:** [Policy gradients](../policy-gradients/index.md), [Monte Carlo & TD learning](../monte-carlo-td-learning/index.md)

---

## Why should I care?

Most deep RL you will meet in practice is actor-critic:

- **PPO** (Proximal Policy Optimization) trained OpenAI Five to play Dota 2, trains many simulated robots, and is the RL step in the original **RLHF** recipe used to fine-tune ChatGPT-style assistants.
- **SAC** (Soft Actor-Critic) is a go-to algorithm for continuous control: robot arms, legged robots, anything where the action is a vector of motor torques.
- **A2C** is the simplest member of the family and the easiest place to see how all the parts fit together.

In the [policy gradients](../policy-gradients/index.md) lesson, REINFORCE learned a policy directly, but slowly and noisily. In the [Monte Carlo & TD learning](../monte-carlo-td-learning/index.md) lesson, TD learning estimated values quickly from single steps. Actor-critic methods glue those two lessons together.

---

## Step 1 — REINFORCE's problem: one noisy number per episode

Recall the REINFORCE update. For every step of an episode, it nudges the policy parameters $\theta$ in the direction that makes the taken action more likely, scaled by the **return** $G_t$ that followed:

$$
\theta \leftarrow \theta + \alpha\, G_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

The trouble is $G_t$. It is the sum of **every** reward until the end of the episode, so it soaks up the randomness of every later action and every later slip of the environment. A good action followed by bad luck gets punished. A bad action followed by good luck gets rewarded. Averaged over many, many episodes this washes out, but each single update is very noisy. In statistics language, the gradient estimate is **unbiased but high-variance**, so learning needs many samples and small steps.

Here is the size of the effect in the cliff world from [lesson 1](../reinforcement-learning-intro/index.md). Take a policy that follows the safe detour 70% of the time and moves at random 30% of the time ($\gamma = 0.95$). From the top-left corner:

| Signal used to judge the first action | Average | Standard deviation |
|---|---|---|
| The whole return $G_t$ (REINFORCE) | −3.42 | **3.77** |
| One-step TD error $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ | ≈ 0 | **0.68** |

The TD error is more than five times less noisy, because it contains the randomness of **one** step only; the critic's value $V(s_{t+1})$ stands in for everything after that. (This table uses the *true* $V$. A learned critic is only an estimate, which brings in some bias; more on that trade-off in Step 4.)

!!! tip "What to notice"
    The fix for noise is not a better policy, it is a better **yardstick**. Instead of asking "what was the total return?", ask "how much better than expected did this step turn out?". Answering "what did I expect?" needs a value function: the critic.

---

## Step 2 — The critic and the advantage

The question the actor really wants answered is: **was this action better or worse than what I usually do here?** That quantity has a name, the **advantage**:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

- $A > 0$: action $a$ is better than the policy's average in state $s$. Do it more.
- $A < 0$: worse than average. Do it less.
- Subtracting $V(s)$ is the **baseline** trick from the policy-gradients lesson. It doesn't change the average gradient, it just centres the signal around zero.

We don't know $Q$ or $V$, so we train a second model, the **critic** $V_w(s)$ with parameters $w$, using exactly the TD learning from [lesson 5](../monte-carlo-td-learning/index.md). Then we use the TD error as our advantage estimate:

$$
\delta_t = r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t) \;\approx\; A(s_t, a_t)
$$

Why does this work? On average, $r_{t+1} + \gamma V(s_{t+1})$ after taking $a_t$ is exactly $Q(s_t, a_t)$. Subtract $V(s_t)$ and you get the advantage. So **one number, $\delta_t$, does two jobs**: it is the critic's learning signal (make $V(s_t)$ closer to the target) and the actor's learning signal (how good was $a_t$?).

!!! note "Actor and critic, side by side"
    | | Actor | Critic |
    |---|---|---|
    | What it is | the policy $\pi_\theta(a \mid s)$ | a value function $V_w(s)$ |
    | What it outputs | action probabilities (or a Gaussian for continuous actions) | one number: expected return from $s$ |
    | How it learns | policy gradient, weighted by $\delta$ | TD learning: move $V_w(s)$ toward $r + \gamma V_w(s')$ |
    | Lesson it comes from | [Policy gradients](../policy-gradients/index.md) | [Monte Carlo & TD learning](../monte-carlo-td-learning/index.md) |

---

## Step 3 — The actor-critic loop

Put the two together and you get the basic **one-step actor-critic** algorithm. Every single step:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🎭 One step of actor-critic",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "obs", "col": 0, "row": 0, "icon": "👀", "label": "Observe state s", "role": "data"},
    {"id": "act", "col": 1, "row": 0, "icon": "🎭", "label": "<b>Actor</b> samples a ~ π<sub>θ</sub>(·|s)", "role": "model", "note": "The policy is stochastic, so it explores by itself: no ε-greedy needed."},
    {"id": "env", "col": 2, "row": 0, "icon": "🌍", "label": "Environment returns r and s′", "role": "data"},
    {"id": "td", "col": 2, "row": 1, "icon": "⚖️", "label": "<b>Critic</b> computes δ", "sub": "δ = r + γV(s′) − V(s)", "role": "loss", "note": "Better or worse than expected? If s′ is terminal, V(s′) = 0."},
    {"id": "cupd", "col": 1, "row": 1, "icon": "📏", "label": "Critic update", "sub": "w ← w + α<sub>w</sub> δ ∇V(s)", "role": "loss", "note": "Plain TD(0): move V(s) toward the target r + γV(s′)."},
    {"id": "aupd", "col": 0, "row": 1, "icon": "🎯", "label": "Actor update", "sub": "θ ← θ + α<sub>θ</sub> δ ∇log π(a|s)", "role": "model", "note": "δ > 0: make a more likely in s. δ < 0: make it less likely."}
  ],
  "edges": [
    {"from": "obs", "to": "act"},
    {"from": "act", "to": "env"},
    {"from": "env", "to": "td"},
    {"from": "td", "to": "cupd", "label": "δ"},
    {"from": "cupd", "to": "aupd"},
    {"from": "aupd", "to": "obs", "label": "s ← s′"}
  ],
  "steps": [
    {"show": ["obs"], "text": "The agent is in state s."},
    {"show": ["obs>act", "act"], "text": "<b>Act:</b> the actor samples an action from its policy π<sub>θ</sub>(a|s)."},
    {"show": ["act>env", "env"], "text": "The environment answers with a reward r and the next state s′."},
    {"show": ["env>td", "td"], "text": "<b>Critic computes δ:</b> the TD error compares what happened (r + γV(s′)) with what the critic expected (V(s))."},
    {"show": ["td>cupd", "cupd"], "text": "<b>Critic update:</b> nudge V(s) toward r + γV(s′), so next time its expectation is a bit more accurate."},
    {"show": ["cupd>aupd", "aupd"], "text": "<b>Actor update:</b> the same δ scales the policy gradient. Positive δ: action a becomes more likely in s. Negative: less likely."},
    {"show": ["aupd>obs", "obs"], "text": "Move to s′ and repeat. No need to wait for the end of the episode: the agent learns after every step."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"obs": [0, 0], "act": [1, 0], "env": [1, 1], "td": [1, 2], "cupd": [0, 2], "aupd": [0, 1]}}
}
</script>
</div>

$$
\begin{aligned}
\delta_t &= r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t) && \text{(critic: surprise)}\\
w &\leftarrow w + \alpha_w\, \delta_t\, \nabla_w V_w(s_t) && \text{(critic update)}\\
\theta &\leftarrow \theta + \alpha_\theta\, \delta_t\, \nabla_\theta \log \pi_\theta(a_t \mid s_t) && \text{(actor update)}
\end{aligned}
$$

A worked first step in the cliff world. The critic starts at $V = 0$ everywhere, the actor picks each direction with probability 0.25. From the start cell the actor samples **→** and falls straight into the cliff ($r = -20$, episode over, so $V(s') = 0$):

- $\delta = -20 + 0 - 0 = -20$
- Critic ($\alpha_w = 0.1$): $V(\text{start})$ becomes $0 + 0.1 \times (-20) = -2.0$
- Actor ($\alpha_\theta = 0.1$, a softmax over four action preferences): π(→ | start) drops from 0.25 to **0.043**, and the other three rise to 0.319 each.

Had it sampled **↑** instead ($r = -1$), $\delta$ would be only $-1$, $V(\text{start})$ would become $-0.1$, and π(↑) would dip just slightly, from 0.25 to 0.232. A big surprise means a big update.

**Try it:** watch a tabular actor-critic learn the cliff world, one transition at a time. Press **1 step** a few times and read the numbers, then press **20 episodes** a few times.

<div id="viz-ac-trace" class="viz"><div class="viz-title">🎭 Interactive: actor and critic learning the cliff world</div></div>

!!! tip "What to notice"
    - **Every step updates both parts.** The readout shows $\delta$ being computed, then the critic's $V(s)$ moving, then the actor's $\pi(a \mid s)$ moving in the direction of $\delta$'s sign.
    - **Early on, the critic is wrong, and that's fine.** The first few $\delta$'s are mostly the raw reward. As values spread backwards from the goal, $\delta$ becomes a much more informative "better or worse than expected" signal.
    - **It learns fast.** We ran this exact algorithm in Python (20 seeds, $\alpha_w = \alpha_\theta = 0.1$, no slip). In episodes 1–10 only 21% of episodes reached the goal and 69% fell off the cliff. In episodes 11–50 it was 92% goal, and from episode 51 on 99%. After 300 episodes, 16 of the 20 runs had learned the 7-step path along the cliff edge.
    - The widget uses `Math.random()`, so your run will differ from ours. Press **Reset** and try again; also try big step sizes (0.5) and see learning become jumpier.

---

## Step 4 — A2C: batches, parallel workers and GAE

Updating after every single step with a neural network is unstable: consecutive states are almost identical, so the gradients all point the same way. **A2C** (Advantage Actor-Critic) fixes this with batching:

1. Run $N$ copies of the environment **in parallel** (say 8), each for $n$ steps (say 5). That gives a batch of $N \times n$ transitions from many different situations.
2. Compute an advantage $\hat{A}_t$ for every transition.
3. Take **one** gradient step on a combined loss, then throw the batch away and collect a fresh one.

The combined loss has three parts:

$$
\begin{aligned}
L(\theta, w) = &-\frac{1}{B}\sum_t \hat{A}_t \log \pi_\theta(a_t \mid s_t) && \text{(actor: policy gradient)}\\
&+ c_v\,\frac{1}{B}\sum_t \big(V_w(s_t) - \hat{G}_t\big)^2 && \text{(critic: value regression)}\\
&- c_e\,\frac{1}{B}\sum_t \mathcal{H}\big(\pi_\theta(\cdot \mid s_t)\big) && \text{(entropy bonus)}
\end{aligned}
$$

where $B$ is the batch size, $\hat{G}_t = \hat{A}_t + V_w(s_t)$ is the critic's target, $c_v$ and $c_e$ are small weights (0.5 and 0.01 in the code below), and $\mathcal{H}$ is the policy's entropy (how spread out its probabilities are; Step 6 has a widget). The **entropy bonus** rewards the actor for not becoming certain too early, which keeps it exploring. Actor and critic often share the first layers of one network.

**How far to look ahead? GAE.** The one-step $\delta_t$ is low-variance but leans heavily on the critic, which may be wrong (bias). The full return $G_t$ has no critic bias but is noisy. **Generalized Advantage Estimation** (GAE) blends all the in-between options with one knob $\lambda \in [0, 1]$:

$$
\hat{A}_t = \delta_t + (\gamma\lambda)\,\delta_{t+1} + (\gamma\lambda)^2\,\delta_{t+2} + \dots
$$

- $\lambda = 0$: just $\delta_t$, the one-step TD error. Low variance, most bias.
- $\lambda = 1$: the sum telescopes to $G_t - V(s_t)$, the Monte Carlo advantage. No critic bias, high variance.
- $\lambda \approx 0.95$ is the usual compromise. It is the same bias–variance dial as TD(λ) in [lesson 5](../monte-carlo-td-learning/index.md).

!!! tip "What to notice"
    "A2C" names three ideas you have already met: the **advantage** (Step 2) weights the policy gradient, a **critic** supplies it, and many parallel environments make the batch diverse. Its asynchronous older sibling, **A3C** (2016), ran the workers on separate CPU threads that each updated a shared network; the synchronous A2C version turned out to work just as well and is simpler.

---

## Step 5 — PPO: don't move the policy too far

A2C has a fragile spot: the step size. A policy-gradient step that is a little too big can wreck the policy. If the actor suddenly makes a bad action very likely, the next batch is collected with that bad policy, the data gets worse, and learning can collapse and never recover. This is different from supervised learning, where a bad step is just a bad step: in RL, **the policy decides what data you see next**.

Also, collecting data is expensive, yet A2C uses each batch for a single gradient step. It would be nice to take several steps on the same batch, but after the first step the batch was collected by an *older* policy.

**PPO** (Proximal Policy Optimization, Schulman et al. 2017) solves both problems with one trick. Call the policy that collected the batch $\pi_{\text{old}}$. For each sample, compute the **probability ratio**

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

$r_t = 1$ means the new policy is just as likely to take $a_t$ as the old one; $r_t = 1.5$ means 50% more likely. Weighting the advantage by this ratio corrects for the policy having changed (an *importance sampling* correction). Then PPO **clips** it:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\,\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\,\hat{A}_t\big)\Big]
$$

with a typical $\varepsilon = 0.2$. In words: **improve the objective, but get no extra credit for moving the ratio outside $[1-\varepsilon, 1+\varepsilon]$.** Once an action has been made 20% more (or less) likely, that sample stops pushing. This lets PPO safely run several epochs of minibatch updates on every batch.

**Try it:** drag the ratio point and change $\varepsilon$.

<div id="viz-ppo-clip" class="viz"><div class="viz-title">✂️ Interactive: the PPO clipped objective</div></div>

!!! tip "What to notice"
    - **Good action ($A = +1$, $\varepsilon = 0.2$):** the objective rises with $r$ until $r = 1.2$, then goes flat. At $r = 1.5$ it is $\min(1.5, 1.2) = 1.2$ and its gradient is **0**: the action is already 20% more likely, so this sample stops pushing.
    - **Bad action ($A = -1$):** the flat part is on the **left**. At $r = 0.5$ the objective is $\min(-0.5, -0.8) = -0.8$, gradient 0: the action has already been made less likely enough.
    - **The clip is one-sided.** For $A = -1$ at $r = 1.5$, the objective is $-1.5$ and the gradient is still active (slope −1). If an update accidentally made a *bad* action *more* likely, PPO keeps pulling it back. The $\min$ makes the objective a pessimistic bound: it ignores improvements beyond the trust region but never ignores mistakes.
    - A smaller $\varepsilon$ means a narrower band around $r = 1$: smaller, safer policy steps.

The full PPO loss is the A2C loss with the actor term swapped for $L^{\text{CLIP}}$: clipped policy objective, plus a critic loss, plus an **entropy bonus** to keep exploring. That is all of PPO. Its popularity comes from being simple, robust to hyperparameters, and good enough almost everywhere. In **RLHF**, the "environment" is a prompt, the action is the model's reply (one token at a time), the reward comes from a learned reward model, and PPO is what nudges the language model toward higher-reward replies (with an extra penalty for drifting far from the original model). The [RLHF & DPO](../rlhf-dpo/index.md) lesson covers that in detail.

---

## Step 6 — SAC: off-policy, continuous actions, maximum entropy

A2C and PPO are **on-policy**: each batch must come from the current policy, is used briefly, and is then thrown away. For a real robot, where every sample costs time and wear, that is wasteful. **SAC** (Soft Actor-Critic, Haarnoja et al. 2018) is **off-policy**: like [DQN](../deep-q-networks/index.md), it stores every transition in a **replay buffer** and learns from old data again and again.

SAC is built around three ideas:

**1. Maximum-entropy objective.** SAC's agent maximises reward **plus** the entropy of its own policy, weighted by a **temperature** $\alpha$:

$$
J(\pi) = \mathbb{E}\Big[\sum_t \gamma^t \big(r_{t+1} + \alpha\, \mathcal{H}(\pi(\cdot \mid s_t))\big)\Big]
$$

The agent is paid to stay random wherever being random costs little. That gives built-in exploration, robustness (it learns many ways to succeed, not one brittle way), and smoother learning. The best policy under this objective is a **softmax over Q-values** with temperature $\alpha$: $\pi(a \mid s) \propto \exp\big(Q(s,a)/\alpha\big)$.

**Try it:** four actions with Q-values 3, 2.5, 1 and 0. Slide the temperature and watch the policy and its entropy.

<div id="viz-entropy" class="viz"><div class="viz-title">🌡️ Interactive: temperature, softmax policy and entropy</div></div>

!!! tip "What to notice"
    - **$\alpha = 0.1$:** almost greedy. The best action gets probability 0.993 and the entropy is only 0.040 nats.
    - **$\alpha = 1$:** the policy is [0.558, 0.339, 0.076, 0.028], entropy 0.987. The close runner-up ($Q = 2.5$) still gets a third of the choices.
    - **$\alpha = 10$:** nearly uniform ([0.285, 0.271, 0.233, 0.211]), entropy 1.379, close to the maximum $\ln 4 = 1.386$ for four actions.
    - Actions with similar Q-values get similar probabilities; clearly bad actions fade first. Modern SAC **tunes $\alpha$ automatically** to hit a target entropy, so you rarely set it by hand.
    - PPO's entropy bonus ($c_e$ in Step 4) is the same idea in a milder form: a small reward for keeping the policy spread out.

**2. Continuous actions.** The actor outputs the mean and standard deviation of a Gaussian, samples an action, and squashes it with $\tanh$ into the allowed range (e.g. torques in $[-1, 1]$). Because the sample is written as $a = \tanh(\mu_\theta(s) + \sigma_\theta(s)\,\xi)$ with random noise $\xi$ (the *reparameterisation trick*), gradients flow straight from the critic's $Q$ into the actor's parameters. The actor minimises $\mathbb{E}\big[\alpha \log \pi_\theta(a \mid s) - Q(s, a)\big]$: pick high-Q actions, but stay random.

**3. Twin Q critics.** SAC's critic learns $Q(s, a)$ rather than $V(s)$. Like Q-learning, a learned Q tends to **overestimate**: the actor seeks out actions whose Q happens to be too high, and errors snowball. SAC trains **two** Q-networks and uses the smaller of the two in its target (an idea from the TD3 algorithm), with slowly-updated target networks $\theta^-$ like DQN:

$$
y = r + \gamma \Big(\min_{j=1,2} Q_{\theta_j^-}(s', a') - \alpha \log \pi(a' \mid s')\Big), \qquad a' \sim \pi(\cdot \mid s')
$$

---

## Step 7 — When to use which

| | A2C | PPO | SAC |
|---|---|---|---|
| On- or off-policy | on-policy | on-policy | **off-policy** (replay buffer) |
| Actions | discrete or continuous | discrete or continuous | **continuous** (discrete variants exist) |
| Critic | $V(s)$ | $V(s)$ | two $Q(s,a)$ + target networks |
| Keeps updates safe by | small learning rate | **clipped ratio** | soft (slow) target updates |
| Exploration from | stochastic policy + entropy bonus | stochastic policy + entropy bonus | **maximum entropy**, auto-tuned $\alpha$ |
| Sample efficiency | low | medium | **high** |
| Wall-clock with a fast simulator | fast | fast, very robust | slower per sample |
| Typical use | teaching, quick baselines | games, simulated robots, **RLHF** | real or simulated robot control |

A practical rule of thumb: **start with PPO.** It works for almost anything and forgives imperfect hyperparameters. If your actions are continuous and samples are expensive (a real robot, a slow simulator), try **SAC**.

---

## Putting it all together

All three algorithms share the same skeleton: an actor that acts, a critic that judges, and an update that uses the critic's verdict. Here is PPO's version, the one you are most likely to use:

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 The PPO training loop",
  "cols": 3,
  "rows": 2,
  "rowHeight": 110,
  "nodes": [
    {"id": "init", "col": 0, "row": 0, "icon": "🎲", "label": "Initialise actor π<sub>θ</sub> and critic V<sub>w</sub>", "shape": "pill"},
    {"id": "roll", "col": 1, "row": 0, "icon": "🌍", "label": "Collect a batch with π<sub>old</sub>", "sub": "N envs × T steps", "role": "data", "note": "Store s, a, r, log π_old(a|s) and V(s) for every step."},
    {"id": "adv", "col": 2, "row": 0, "icon": "⚖️", "label": "Advantages with GAE", "sub": "Â = δ + γλÂ′", "role": "loss", "note": "Run backwards through the batch. Critic targets: Ĝ = Â + V(s)."},
    {"id": "upd", "col": 2, "row": 1, "icon": "✂️", "label": "K epochs of minibatch updates", "sub": "clipped loss + value loss − entropy", "role": "model", "note": "Each minibatch: ratio r = π_θ/π_old, clipped objective, critic regression, entropy bonus."},
    {"id": "sync", "col": 1, "row": 1, "icon": "🔄", "label": "π<sub>old</sub> ← π<sub>θ</sub>", "sub": "discard the batch", "role": "opt", "note": "On-policy: the old batch no longer matches the policy, so it is thrown away."}
  ],
  "edges": [
    {"from": "init", "to": "roll"},
    {"from": "roll", "to": "adv"},
    {"from": "adv", "to": "upd"},
    {"from": "upd", "to": "sync"},
    {"from": "sync", "to": "roll", "label": "repeat"}
  ],
  "steps": [
    {"show": ["init"], "text": "Start with a random actor network and a random critic network (often two heads on one body)."},
    {"show": ["init>roll", "roll"], "text": "Run the current policy (now called π<sub>old</sub>) in several environments for T steps each. Remember the log-probability of every action taken."},
    {"show": ["roll>adv", "adv"], "text": "Compute TD errors δ with the critic, and blend them into advantages with GAE(λ)."},
    {"show": ["adv>upd", "upd"], "text": "Several passes (epochs) over the batch in minibatches: maximise the clipped objective, regress the critic toward its targets, add an entropy bonus. The clip stops the policy from moving too far."},
    {"show": ["upd>sync", "sync"], "text": "The improved policy becomes the new π<sub>old</sub>. The batch is thrown away."},
    {"show": ["sync>roll", "roll"], "text": "Collect a fresh batch and repeat."}
  ],
  "narrow": {"cols": 2, "rows": 3, "pos": {"init": [0, 0], "roll": [0, 1], "adv": [1, 1], "upd": [1, 2], "sync": [0, 2]}}
}
</script>
</div>

Here is a complete PPO in PyTorch, on a home-made CartPole (balance a pole on a cart by pushing left or right; +1 reward per step, up to 500 steps). The three numbered blocks match the diagram. It runs in about a minute on one CPU core:

```python
import math, random, time
import numpy as np
import torch
import torch.nn as nn


class CartPole:
    """The classic cart-pole: push the cart left or right to keep the pole upright. +1 per step, max 500."""

    def reset(self):
        self.s = np.random.uniform(-0.05, 0.05, size=4)   # x, x_dot, theta, theta_dot
        self.t = 0
        return self.s.copy()

    def step(self, action):
        x, x_dot, th, th_dot = self.s
        force = 10.0 if action == 1 else -10.0
        cos, sin = math.cos(th), math.sin(th)
        tmp = (force + 0.05 * th_dot**2 * sin) / 1.1      # 0.05 = pole mass × half-length, 1.1 = total mass
        th_acc = (9.8 * sin - cos * tmp) / (0.5 * (4 / 3 - 0.1 * cos**2 / 1.1))
        x_acc = tmp - 0.05 * th_acc * cos / 1.1
        x, x_dot = x + 0.02 * x_dot, x_dot + 0.02 * x_acc  # Euler step, 20 ms
        th, th_dot = th + 0.02 * th_dot, th_dot + 0.02 * th_acc
        self.s = np.array([x, x_dot, th, th_dot])
        self.t += 1
        fell = abs(x) > 2.4 or abs(th) > 12 * math.pi / 180
        return self.s.copy(), 1.0, fell, self.t >= 500     # state, reward, terminated, truncated


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))   # action logits
        self.critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))  # V(s)

    def dist(self, s):
        return torch.distributions.Categorical(logits=self.actor(s))


random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.set_num_threads(1)                               # tiny networks: one CPU thread is fastest
env, net = CartPole(), ActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
GAMMA, LAM, CLIP, T, EPOCHS, BATCH = 0.99, 0.95, 0.2, 2048, 10, 64

s, ep_len, lengths = env.reset(), 0, []
start = time.time()
for iteration in range(40):
    # 1. Collect T steps with the current ("old") policy
    S, A, R, D, LOGP, V = [], [], [], [], [], []
    for _ in range(T):
        with torch.no_grad():
            st = torch.as_tensor(s, dtype=torch.float32)
            d = net.dist(st)
            a = d.sample()
            S.append(st); A.append(a); LOGP.append(d.log_prob(a)); V.append(net.critic(st).squeeze())
        s, r, term, trunc = env.step(a.item())
        R.append(r); D.append(term or trunc)     # (treating the 500-step time-out as an ending, for simplicity)
        ep_len += 1
        if term or trunc:
            lengths.append(ep_len)
            s, ep_len = env.reset(), 0
    with torch.no_grad():
        last_v = net.critic(torch.as_tensor(s, dtype=torch.float32)).item()

    # 2. Advantages with GAE(λ): A_t = δ_t + γλ A_{t+1}
    adv, gae = torch.zeros(T), 0.0
    for t in reversed(range(T)):
        next_v = last_v if t == T - 1 else V[t + 1].item()
        if D[t]:
            next_v, gae = 0.0, 0.0
        delta = R[t] + GAMMA * next_v - V[t].item()
        gae = delta + GAMMA * LAM * gae
        adv[t] = gae
    S, A, LOGP, V = torch.stack(S), torch.stack(A), torch.stack(LOGP), torch.stack(V)
    returns = adv + V                                   # targets for the critic

    # 3. Several epochs of minibatch updates on the clipped objective
    for _ in range(EPOCHS):
        for idx in torch.randperm(T).split(BATCH):
            d = net.dist(S[idx])
            ratio = torch.exp(d.log_prob(A[idx]) - LOGP[idx])          # r_t(θ) = π_θ / π_old
            a_n = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)
            actor_loss = -torch.min(ratio * a_n, ratio.clamp(1 - CLIP, 1 + CLIP) * a_n).mean()
            critic_loss = (net.critic(S[idx]).squeeze(-1) - returns[idx]).pow(2).mean()
            entropy = d.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            opt.zero_grad(); loss.backward(); opt.step()

    if iteration == 0 or iteration % 5 == 4:
        print(f"iteration {iteration:2d}  steps {T * (iteration + 1):6d}  "
              f"avg episode length (last 20) = {np.mean(lengths[-20:]):6.1f}")
print(f"time: {time.time() - start:.0f}s;  last 20 episodes that reached 500 steps: {sum(l == 500 for l in lengths[-20:])}")
```

```text
iteration  0  steps   2048  avg episode length (last 20) =   20.4
iteration  4  steps  10240  avg episode length (last 20) =   71.9
iteration  9  steps  20480  avg episode length (last 20) =  207.3
iteration 14  steps  30720  avg episode length (last 20) =  316.4
iteration 19  steps  40960  avg episode length (last 20) =  404.2
iteration 24  steps  51200  avg episode length (last 20) =  404.8
iteration 29  steps  61440  avg episode length (last 20) =  473.8
iteration 34  steps  71680  avg episode length (last 20) =  450.5
iteration 39  steps  81920  avg episode length (last 20) =  468.1
time: 57s;  last 20 episodes that reached 500 steps: 17
```

The random policy keeps the pole up for about 20 steps. After about 82,000 steps of experience the average over the last 20 episodes is 468 steps, and 17 of those 20 episodes hit the 500-step maximum. Notice that the curve is not monotonic (404.8 → 473.8 → 450.5): even PPO's learning is noisy, which is why RL papers report averages over several seeds.

A few details worth pointing out in the code:

- **Advantage normalisation** (`a_n`): advantages are rescaled to mean 0, standard deviation 1 within each minibatch. It is a standard PPO trick that makes one learning rate work across problems with very different reward scales.
- **To turn this into A2C**, set `EPOCHS = 1`, use the whole batch at once, and replace the clipped term with `-(d.log_prob(A) * a_n).mean()`.
- **Separate actor and critic networks** here; sharing a body is also common.

For real projects, use a tested library such as [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) or [CleanRL](https://docs.cleanrl.dev/) (single-file implementations, great for reading). With Gymnasium installed, the whole thing is a few lines:

```python
import gymnasium as gym
from stable_baselines3 import PPO   # also: A2C, SAC (SAC needs continuous actions, e.g. "Pendulum-v1")

model = PPO("MlpPolicy", gym.make("CartPole-v1"), verbose=1)
model.learn(total_timesteps=100_000)
```

---

## Common misconceptions

!!! warning "\"The critic picks the actions.\""
    Only the actor acts. In A2C and PPO the critic outputs $V(s)$, which has no actions in it at all; it only supplies the yardstick $\delta$ that tells the actor how to change. (In SAC the critic is a $Q(s, a)$, but the actor still does the choosing; the critic only grades its choices.)

!!! warning "\"PPO's clip limits the size of the parameter update.\""
    It limits how much the policy's **action probabilities** change on the samples in the batch, by zeroing the gradient of samples whose ratio has left $[1-\varepsilon, 1+\varepsilon]$. The weights can still move a lot, and policies can still drift beyond the band over many epochs. That is why implementations also watch the KL divergence and use modest learning rates and epoch counts.

!!! warning "\"Adding a critic makes the gradient unbiased and better in every way.\""
    Subtracting a baseline $V(s)$ never adds bias. But **bootstrapping**, using $r + \gamma V(s')$ in place of the true return, does: if the critic is wrong, the advantage is wrong. Actor-critic methods trade a little bias for a lot less variance, and GAE's $\lambda$ is the dial for that trade.

!!! warning "\"Maximum entropy just means acting randomly.\""
    SAC's policy is random only where randomness is cheap. Where one action is clearly better, the softmax over Q-values puts almost all the probability on it (try $\alpha = 0.1$ in the widget). The temperature sets the exchange rate between reward and randomness.

---

## Check yourself

??? question "1. The critic predicts V(s) = 4. The agent takes action a, gets reward r = 2 and lands in s′ with V(s′) = 3. With γ = 0.9, what is δ, and what does the actor do?"
    $\delta = 2 + 0.9 \times 3 - 4 = 0.7$. It is positive: the outcome was better than the critic expected, so the actor makes $a$ **more likely** in $s$ (and the critic raises $V(s)$ toward $4.7$).

??? question "2. Why does A2C run several environments in parallel instead of updating after every step of one environment?"
    Consecutive steps from one environment are strongly correlated (nearly identical states), so the gradients all point the same way and learning becomes unstable. Parallel environments give a batch of diverse, less correlated transitions, and the batch also averages out noise. It plays the role that the replay buffer plays in DQN, without leaving on-policy learning.

??? question "3. In PPO with ε = 0.2, a sample has A = +1 and ratio r = 1.3. What is its clipped objective, and does it contribute a gradient? What if A = −1?"
    With $A = +1$: $\min(1.3, 1.2) = 1.2$. The clipped term wins and is constant in $\theta$, so the gradient is **zero**. With $A = -1$: $\min(-1.3, -1.2) = -1.3$, the unclipped term wins, so the gradient is **active**: PPO keeps pushing this bad action's probability back down.

??? question "4. What do λ = 0 and λ = 1 mean in GAE?"
    $\lambda = 0$ gives the one-step TD error $\delta_t$ (low variance, but biased if the critic is wrong). $\lambda = 1$ gives $G_t - V(s_t)$, the Monte Carlo return minus a baseline (no bootstrapping bias, but high variance). Values in between blend the two; 0.95 is a common default.

??? question "5. Name two things SAC does that PPO doesn't, and why each helps."
    (1) It is **off-policy** with a replay buffer, so every sample is reused many times: much better sample efficiency. (2) It maximises reward **plus entropy** with a tuned temperature α: built-in exploration and more robust policies. (3) It uses **twin Q critics** and takes the minimum, which counters the overestimation bias of learned Q-values. Any two of these.

---

## Further reading

- Richard Sutton & Andrew Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html), Chapter 13.5 (actor-critic methods)
- Mnih et al., [*Asynchronous Methods for Deep Reinforcement Learning*](https://arxiv.org/abs/1602.01783) (2016), the A3C paper
- Schulman et al., [*High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438) (2015), GAE
- Schulman et al., [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) (2017), PPO
- Haarnoja et al., [*Soft Actor-Critic*](https://arxiv.org/abs/1801.01290) (2018) and [*Soft Actor-Critic Algorithms and Applications*](https://arxiv.org/abs/1812.05905) (2018, automatic temperature)
- Huang et al., [*The 37 Implementation Details of Proximal Policy Optimization*](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) (ICLR blog track, 2022): everything the paper leaves out
- OpenAI, [*Spinning Up: PPO*](https://spinningup.openai.com/en/latest/algorithms/ppo.html) and [*SAC*](https://spinningup.openai.com/en/latest/algorithms/sac.html), clear write-ups with pseudocode

**Next lessons:** [Model-based RL & planning](../model-based-rl/index.md) · [RL from human feedback (RLHF) and DPO](../rlhf-dpo/index.md)

<script src="viz/actor-critic.js"></script>
