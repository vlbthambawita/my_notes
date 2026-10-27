# Deep Q-Networks (DQN)

!!! abstract "TL;DR"
    A **Deep Q-Network** replaces the Q-table of Q-learning with a **neural network** $Q(s, a; \theta)$ that takes a state and outputs one Q-value per action. Training it is ordinary **regression**: push $Q(s,a;\theta)$ towards the target $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ with gradient descent. Two tricks made this stable enough to learn Atari games from pixels (2015): an **experience replay buffer**, so the network learns from random, decorrelated mini-batches of old experience, and a **target network** $\theta^-$, a frozen copy that keeps the target still while the network chases it.

**Level:** intermediate · **Prerequisites:** [Q-learning & SARSA](../q-learning-sarsa/index.md), [Gradient Descent](../gradient-descent/index.md)

---

## Why should I care?

In 2015, a team at DeepMind published *Human-level control through deep reinforcement learning* in *Nature*. One agent, with **the same network, the same algorithm and the same settings**, learned to play 49 different Atari games from nothing but the screen pixels and the score. On many of them it played about as well as a professional human tester. Before that, RL with neural networks was known to be unstable, and most successes used hand-made features.

The algorithm was **DQN**. It is Q-learning from the [previous lesson](../q-learning-sarsa/index.md) plus a neural network plus two stabilising tricks. Those tricks, a replay buffer and a target network, are still inside many modern RL methods, and the overestimation problem DQN ran into explains design choices you will meet again in actor-critic methods like SAC.

---

## Step 1 — Why Q-tables break

Tabular Q-learning keeps one number $Q(s, a)$ for every state–action pair. On the cliff world that is a table of at most $24 \times 4 = 96$ numbers, and each entry is learned separately. That stops working for two reasons.

**1. There are far too many states.** DQN's Atari input is the last 4 screen frames, shrunk to $84 \times 84$ grey pixels with 256 shades each. That is $84 \times 84 \times 4 = 28{,}224$ pixels, so the number of possible inputs is $256^{28224}$, roughly $10^{67970}$. No table can hold that, and the agent will never see the same screen twice anyway.

**2. States can be continuous.** In **CartPole**, the task we will solve at the end, a cart must keep a pole balanced by pushing left or right. The state is four real numbers: cart position, cart speed, pole angle and pole angular speed. There are infinitely many states.

What we really need is **generalisation**: if the agent learned that "pole tilting right, falling fast" is bad, it should assume that a *slightly different* tilt is also bad, without having visited it. A table cannot do this. A function with shared parameters can, because changing its parameters for one state also changes its output for similar states. That is exactly what neural networks are good at.

| | Q-table | Q-network |
|---|---|---|
| Stores | one number per $(s, a)$ | weights $\theta$ shared by all states |
| Unseen state | knows nothing | guesses from similar states |
| Size | grows with the number of states | fixed (4,610 weights for our CartPole network) |
| Update one state | changes only that entry | changes the outputs for many states |

The last row is the blessing *and* the curse of this lesson. Generalisation is why DQN works; it is also why naive training is unstable (Steps 4 and 5).

---

## Step 2 — The Q-network: one output per action

We approximate the optimal action-value function with a network: $Q(s, a; \theta) \approx Q^*(s, a)$, where $\theta$ are the weights.

There are two ways to feed a network. You could give it $(s, a)$ and get back one number, but then choosing an action needs one forward pass per action. DQN instead takes **only the state** as input and has **one output per action**. A single forward pass gives all the Q-values, and acting greedily is just "pick the biggest output".

<div class="viz diagram">
<script type="application/json">
{
  "title": "🧠 A Q-network for CartPole",
  "cols": 4,
  "rows": 2,
  "rowHeight": 105,
  "nodes": [
    {"id": "s", "col": 0, "row": 0.5, "icon": "🛒", "label": "<b>State s</b>", "sub": "4 numbers for CartPole", "role": "data", "note": "Cart position x and speed ẋ, pole angle θ and angular speed θ̇. For Atari it is a stack of 4 screen frames."},
    {"id": "net", "col": 1, "row": 0.5, "icon": "🧠", "label": "<b>Network θ</b>", "sub": "2 hidden layers, 64 ReLU units", "role": "model", "note": "For Atari, DQN used 3 convolutional layers and one fully connected layer of 512 units before the outputs."},
    {"id": "ql", "col": 2, "row": 0, "label": "Q(s, ← ; θ)", "sub": "value of pushing left", "role": "loss"},
    {"id": "qr", "col": 2, "row": 1, "label": "Q(s, → ; θ)", "sub": "value of pushing right", "role": "loss"},
    {"id": "act", "col": 3, "row": 0.5, "icon": "🎯", "label": "<b>Greedy action</b>", "sub": "argmax over the outputs", "role": "opt"}
  ],
  "edges": [
    {"from": "s", "to": "net"},
    {"from": "net", "to": "ql"},
    {"from": "net", "to": "qr"},
    {"from": "ql", "to": "act"},
    {"from": "qr", "to": "act"}
  ],
  "steps": [
    {"show": ["s"], "text": "The input is the <b>state only</b>: here 4 numbers, for Atari 4 stacked frames of 84×84 pixels."},
    {"show": ["s>net", "net"], "text": "The network mixes them through hidden layers. The same weights θ are used for every state, which is what lets it generalise."},
    {"show": ["net>ql", "net>qr", "ql", "qr"], "text": "The output layer has <b>one unit per action</b>. One forward pass gives Q(s, a; θ) for every action a."},
    {"show": ["ql>act", "qr>act", "act"], "text": "Acting greedily is just picking the largest output. With ε-greedy we sometimes pick at random instead."}
  ],
  "narrow": {"cols": 2, "rows": 3.4, "rowHeight": 90, "pos": {"s": [0, 0], "net": [1, 0], "ql": [0, 1.2], "qr": [1, 1.2], "act": [0.5, 2.4]}}
}
</script>
</div>

!!! tip "What to notice"
    - The number of outputs is fixed by the number of actions (2 for CartPole, up to 18 for Atari). This is why DQN only works for **discrete** actions: with a continuous action like "steering angle", there is no finite list of outputs to take the max over. Continuous control needs the actor-critic methods of a later lesson.
    - Nothing about the network is special to RL. It is a plain regression network. All the RL lives in **what we train it towards**, which is the next step.

---

## Step 3 — Turning Q-learning into regression

Recall the tabular Q-learning update from the previous lesson:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[\underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{target}} - Q(s,a)\Big]
$$

Read it as "move the estimate a little towards the target". With a network we cannot overwrite one entry, but we can do the next best thing: treat the target as a **label** and take a gradient step that moves $Q(s,a;\theta)$ towards it. For one transition $(s, a, r, s')$:

$$
y = \begin{cases} r & \text{if } s' \text{ is terminal} \\ r + \gamma \max_{a'} Q(s', a'; \theta^-) & \text{otherwise} \end{cases}
\qquad
L(\theta) = \big(y - Q(s, a; \theta)\big)^2
$$

and then, exactly as in [Gradient Descent](../gradient-descent/index.md), step downhill:

$$
\theta \leftarrow \theta - \eta\, \nabla_\theta L(\theta) = \theta + 2\eta\, \big(y - Q(s,a;\theta)\big)\, \nabla_\theta Q(s,a;\theta)
$$

The bracket $y - Q(s,a;\theta)$ is the TD error $\delta$ from earlier lessons. For now read $\theta^-$ as "the same weights $\theta$"; Step 5 explains why DQN uses a separate copy.

Three details matter:

- **The target is treated as a constant.** We do not backpropagate through $y$, even though it depends on the network too. (In code: compute $y$ under `torch.no_grad()`.) This is called a **semi-gradient** method. Differentiating through $y$ as well is a different, usually worse, algorithm.
- **It is still Q-learning.** If the "network" were a table with one weight per $(s,a)$, this gradient step would be the tabular update above, with step size $2\eta$.
- **It is off-policy.** The target uses $\max_{a'}$, the greedy action, no matter which action the agent actually took next. So the transition can come from *any* behaviour, including old versions of the agent. That is what makes the replay buffer of Step 4 legal.

!!! note "Huber loss"
    DQN on Atari clipped the TD error to $[-1, 1]$ in the gradient, which is the same as using the **Huber loss** (squared for small errors, linear for large ones). It stops a single surprising transition from producing a huge gradient. Our CartPole code uses the plain squared error, to match the formula above.

---

## Step 4 — Problem 1: consecutive samples are correlated → experience replay

The naive way to train is **online**: take one step in the environment, do one gradient step on that transition, repeat. This breaks an assumption that [stochastic gradient descent](../gradient-descent/index.md#step-6-real-data-is-big-stochastic-gradient-descent-sgd) relies on: that each mini-batch is a fair, random sample of the data.

Consecutive transitions are nearly identical (frame 1001 of a game looks almost exactly like frame 1000). Training on them in order is like fitting a regression line using only the points from one corner of the plot, then only from another corner. The network over-fits to whatever it saw last and forgets the rest. Worse, the data depends on the network: if it learns "always go left", it only sees left-hand states, which reinforces the habit.

**Experience replay** fixes this. Every transition $(s, a, r, s', \text{done})$ goes into a big **replay buffer** (DQN on Atari kept the last 1 million). For each gradient step, we sample a **random mini-batch** from the buffer.

**Try it:** the buffer below fills with real CartPole transitions from a random policy. Sample a mini-batch of 12 "in a row" (the last 12 steps, as online learning would see them) and "at random", and compare where they land in the state plot.

<div id="viz-replay" class="viz"><div class="viz-title">🗃️ Interactive: a replay buffer</div></div>

!!! tip "What to notice"
    - The 12 transitions in a row come from one or two episodes and trace **one short, smooth path** in the state plot: a single swing of the pole. Each one is almost a copy of its neighbour. A gradient step on them only teaches the network about that corner of the state space.
    - A random batch comes from several episodes, and its orange line **zigzags across the whole cloud**. It is much closer to an independent, representative sample, which is what SGD expects.
    - Every transition stays in the buffer for a long time and gets sampled many times. Experience is **reused**, which matters when collecting it is slow (a robot) or expensive (an Atari emulator running for days).

Replay only works because Q-learning is off-policy (Step 3). The old transitions were collected by older, worse versions of the policy, but the Q-learning target doesn't care who chose the action.

---

## Step 5 — Problem 2: chasing a moving target → the target network

Look at the target again: $y = r + \gamma \max_{a'} Q(s', a'; \theta)$. It is computed **with the same network we are training**. In a table that is harmless: updating $Q(s, a)$ doesn't change $Q(s', \cdot)$. With a network it is not, because of generalisation. A gradient step that raises $Q(s, a)$ also raises $Q$ for similar states, and $s'$ is usually very similar to $s$ (it is the next frame). So the target **moves every time you step towards it**, like a dog chasing its tail. Errors can feed on themselves and grow.

DQN's fix is a second network, the **target network** with weights $\theta^-$. It is a copy of the online network that is only updated every $C$ steps ($\theta^- \leftarrow \theta$) and **frozen** in between. For $C$ steps, the targets $y$ are fixed numbers, and learning is plain supervised regression towards fixed labels. Then the labels are refreshed. (DQN on Atari used $C = 10{,}000$ updates.)

**Try it:** a tiny example with three states. From states A and B the agent gets reward +1 and moves to state C, which loops back to itself with reward 0 forever. With $\gamma = 0.9$ the true values are $Q(A) = Q(B) = 1$ and $Q(C) = 0$ (green dashed lines). The "network" is linear with only **two weights** shared by all three states ($Q(A) = w_1$, $Q(B) = 2w_1 + w_2$, $Q(C) = 2w_1 + 2w_2$), so it can generalise and it can represent the true values exactly. The buffer holds A and B three times as often as C, and each step is one gradient step on the average squared TD error over the buffer. The slider $C$ sets how often the target is synced; $C = 1$ means the target is live.

<div id="viz-target" class="viz"><div class="viz-title">🎯 Interactive: live target vs frozen target</div></div>

!!! tip "What to notice"
    - **Live target ($C = 1$): it spirals out of control.** With $\alpha = 0.3$, after 300 steps the estimates are around $-10^9$. A smaller step size does not save it: in this example the error grows by a constant factor every step for *any* $\alpha$ (about 1.068 per step at $\alpha = 0.3$, 1.002 at $\alpha = 0.01$), so a smaller $\alpha$ only makes the explosion slower.
    - **Frozen target: it converges.** With $\alpha = 0.3$ and $C = 10$, all three estimates are within 0.01 of the truth after 300 steps. With $C = 3$ the syncs are too frequent and it is still off by about 4.
    - **Look at the staircase.** Between syncs, each line glides towards a fixed label, then jumps when the target is refreshed. Each phase is an ordinary, well-behaved regression problem.
    - **There is a price.** With a small step size ($\alpha = 0.05$), even $C = 100$ is still off by about 0.5 after 300 steps: values travel through the target only once per sync, so a large $C$ learns slowly.

!!! note "No guarantees, just much better odds"
    The divergence here is an instance of what Sutton and Barto call the **deadly triad**: function approximation + bootstrapping (targets built from your own estimates) + off-policy data. Any two are fine; all three together can diverge. DQN has all three. The target network and replay buffer don't remove the triad, and you can build examples where DQN still diverges. In practice they make it stable enough to work across dozens of games.

---

## Step 6 — Exploration: an ε schedule

DQN acts **ε-greedily** with respect to its own Q-network, like tabular Q-learning. The difference is that ε is **annealed**: at the start, the network's outputs are meaningless, so the agent explores a lot; later it mostly exploits what it has learned.

- DQN on Atari: ε falls linearly from 1.0 to 0.1 over the first million frames, then stays at 0.1.
- Our CartPole code: ε falls from 1.0 to 0.05 over the first 10,000 steps.

It also waits until the buffer has some data (1,000 transitions in our code) before the first gradient step, so the first mini-batches are not all from the same episode.

The Atari agent used a few more engineering tricks that are worth knowing by name: it stacked the last **4 frames** as the state (one frame can't show which way the ball is moving), repeated each action for 4 frames (**frame skipping**), and **clipped rewards** to $-1, 0, +1$ so one set of hyperparameters worked for games with very different score scales.

---

## Step 7 — Improvements: Double DQN, dueling networks, prioritised replay

DQN started a long line of improvements. Three are especially common.

### Double DQN: the max is too optimistic

The target uses $\max_{a'} Q(s', a')$. If the estimates are noisy, the max tends to pick an action whose estimate happens to be **too high**. So even when the noise averages out to zero for each action, the max is biased **upwards**. The bias feeds into the targets, then into other estimates through bootstrapping, and DQN's Q-values can end up far higher than the returns the agent really gets.

**Try it:** every action below is truly worth 0. Each draw gives two independent noisy estimates of all actions (blue and purple). DQN's target takes the max of the blue estimates. **Double DQN** uses the blue estimates only to **choose** the action, then reads that action's value from the purple estimate.

<div id="viz-overest" class="viz"><div class="viz-title">📈 Interactive: why the max overestimates</div></div>

!!! tip "What to notice"
    - The max of the blue estimates is almost always positive, although every true value is 0. In a simulation with 200,000 draws and noise $\sigma = 1$, the average of the max was 0.57 with 2 actions, 1.03 with 4 and 1.54 with 10. More actions → more chances to be lucky → more overestimation.
    - The Double estimate averages about 0 (to within 0.003 in the same simulation), whatever the number of actions. Choosing with one estimate and judging with an independent one removes the "winner was lucky" bias.

DQN already has a second network, so Double DQN costs one line. The target becomes

$$
y = r + \gamma\, Q\big(s',\ \underbrace{\arg\max_{a'} Q(s', a'; \theta)}_{\text{online net chooses}};\ \theta^-\big) \quad (\text{target net evaluates})
$$

### Dueling networks

Split the last layer into two streams: one estimates the state value $V(s)$, one estimates how much better each action is than average, the **advantage** $A(s, a)$. They are recombined as $Q(s,a) = V(s) + A(s,a) - \tfrac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')$. In many states the action barely matters (nothing is happening on screen), and the $V$ stream can learn that without learning every action separately.

### Prioritised experience replay

Sample transitions with large TD error $|\delta|$ more often, because that is where the network is most wrong and has the most to learn. Importance-sampling weights correct the bias this introduces.

**Rainbow** (2018) combined these with three more ideas (multi-step targets, distributional values, noisy networks) and beat each of them alone on Atari.

!!! note "What DQN can't do"
    DQN needs a lot of data (the Atari agents trained on 50 million frames per game, about 38 days of play), it only handles discrete actions, and it can be brittle: results vary a lot between random seeds, as you'll see below. Policy-gradient and actor-critic methods (the next lessons) address the continuous-action problem; model-based methods address data efficiency.

---

## Putting it all together

<div class="viz diagram">
<script type="application/json">
{
  "title": "🔁 The DQN training loop",
  "cols": 4,
  "rows": 2,
  "rowHeight": 115,
  "nodes": [
    {"id": "obs", "col": 0, "row": 0, "icon": "👀", "label": "Observe state s", "role": "data"},
    {"id": "act", "col": 1, "row": 0, "icon": "🎲", "label": "ε-greedy action", "sub": "from Q(s, ·; θ)", "role": "model"},
    {"id": "env", "col": 2, "row": 0, "icon": "🌍", "label": "Environment", "sub": "returns r, s′", "role": "data"},
    {"id": "store", "col": 3, "row": 0, "icon": "🗃️", "label": "Replay buffer", "sub": "store (s, a, r, s′)", "role": "data", "note": "Keeps the last N transitions (1 million on Atari, 50,000 in our code). The oldest are overwritten."},
    {"id": "sample", "col": 3, "row": 1, "icon": "🎰", "label": "Random mini-batch", "sub": "e.g. 64 old transitions", "role": "data"},
    {"id": "target", "col": 2, "row": 1, "icon": "🎯", "label": "Targets y", "sub": "from target net θ⁻", "role": "loss", "note": "Computed with the frozen target network θ⁻ and treated as fixed labels (no gradient)."},
    {"id": "update", "col": 1, "row": 1, "icon": "👣", "label": "Gradient step on θ", "sub": "loss (y − Q)²", "role": "grad"},
    {"id": "sync", "col": 0, "row": 1, "icon": "📋", "label": "Every C steps", "sub": "copy θ⁻ ← θ", "role": "opt"}
  ],
  "edges": [
    {"from": "obs", "to": "act"},
    {"from": "act", "to": "env"},
    {"from": "env", "to": "store"},
    {"from": "store", "to": "sample"},
    {"from": "sample", "to": "target"},
    {"from": "target", "to": "update"},
    {"from": "update", "to": "sync"},
    {"from": "sync", "to": "obs", "label": "next step"}
  ],
  "steps": [
    {"show": ["obs"], "text": "The agent looks at the current state s (4 numbers for CartPole, 4 stacked frames for Atari)."},
    {"show": ["obs>act", "act"], "text": "It runs the online network once and acts ε-greedily: usually the action with the largest Q-value, sometimes a random one."},
    {"show": ["act>env", "env"], "text": "The environment returns a reward r and the next state s′."},
    {"show": ["env>store", "store"], "text": "The transition goes into the <b>replay buffer</b>. Nothing is learned from it yet."},
    {"show": ["store>sample", "sample"], "text": "Learning uses a <b>random</b> mini-batch of old transitions, not the one that just happened. Problem 1 solved: the batch is decorrelated."},
    {"show": ["sample>target", "target"], "text": "For each transition the target y = r + γ max<sub>a′</sub> Q(s′, a′; θ⁻) is computed with the frozen <b>target network</b> θ⁻ (y = r if s′ is terminal). Problem 2 solved: the labels hold still."},
    {"show": ["target>update", "update"], "text": "One gradient-descent step makes Q(s, a; θ) closer to y on this batch. This is ordinary regression."},
    {"show": ["update>sync", "sync"], "text": "Every C steps the target network is refreshed with a copy of the online weights."},
    {"show": ["sync>obs", "obs"], "text": "Then on to the next environment step. Acting and learning interleave: one gradient step per environment step."}
  ],
  "narrow": {"cols": 2, "rows": 4, "rowHeight": 100, "pos": {"obs": [0, 0], "act": [1, 0], "env": [1, 1], "store": [1, 2], "sample": [1, 3], "target": [0, 3], "update": [0, 2], "sync": [0, 1]}}
}
</script>
</div>

Here is a complete DQN in PyTorch. CartPole is written out in plain Python with the standard physics (it is Gym's `CartPole-v1`: reward +1 for every step the pole stays up, episodes end when it falls or after 500 steps). The two flags `use_replay` and `use_target` let us switch the two tricks off.

```python
import math, random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class CartPole:
    """The classic cart-pole task (same physics as Gym's CartPole-v1). Reward +1 per step, max 500 steps."""

    def reset(self):
        self.s = np.random.uniform(-0.05, 0.05, size=4)     # cart position, cart speed, pole angle, pole speed
        self.t = 0
        return self.s.astype(np.float32)

    def step(self, action):                                  # 0 = push left, 1 = push right
        x, x_dot, th, th_dot = self.s
        force = 10.0 if action == 1 else -10.0
        cos, sin = math.cos(th), math.sin(th)
        temp = (force + 0.05 * th_dot**2 * sin) / 1.1
        th_acc = (9.8 * sin - cos * temp) / (0.5 * (4 / 3 - 0.1 * cos**2 / 1.1))
        x_acc = temp - 0.05 * th_acc * cos / 1.1
        self.s = np.array([x + 0.02 * x_dot, x_dot + 0.02 * x_acc, th + 0.02 * th_dot, th_dot + 0.02 * th_acc])
        self.t += 1
        fell = abs(self.s[0]) > 2.4 or abs(self.s[2]) > 12 * math.pi / 180
        return self.s.astype(np.float32), 1.0, fell, self.t >= 500   # s', r, terminal?, time limit?


def q_network():                                             # Q(s, ·; θ): 4 numbers in, one Q-value per action out
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))


def greedy_test(q_net, episodes=20):                         # play with ε = 0, return the average episode length
    env, lengths = CartPole(), []
    for _ in range(episodes):
        s, done = env.reset(), False
        while not done:
            with torch.no_grad():
                a = q_net(torch.tensor(s)).argmax().item()
            s, r, term, trunc = env.step(a)
            done = term or trunc
        lengths.append(env.t)
    return np.mean(lengths)


def train_dqn(seed=0, total_steps=50_000, gamma=0.99, batch_size=64, sync_every=500,
              use_replay=True, use_target=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = CartPole()
    q_net, target_net = q_network(), q_network()
    target_net.load_state_dict(q_net.state_dict())            # θ⁻ ← θ
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = deque(maxlen=50_000 if use_replay else 1)        # no replay = learn from the newest transition only
    untrained_score = greedy_test(q_net)
    lengths, step = [], 0

    while step < total_steps:
        s, done = env.reset(), False
        while not done:
            # 1. act ε-greedily; ε falls from 1.0 to 0.05 over the first 10,000 steps
            eps = max(0.05, 1.0 - step / 10_000)
            if random.random() < eps:
                a = random.randrange(2)
            else:
                with torch.no_grad():
                    a = q_net(torch.tensor(s)).argmax().item()
            s2, r, terminal, truncated = env.step(a)

            # 2. store the transition in the replay buffer
            buffer.append((s, a, r, s2, terminal))
            s, done, step = s2, terminal or truncated, step + 1

            # 3. learn from a random mini-batch of old transitions
            if len(buffer) >= (1_000 if use_replay else 1):
                batch = random.sample(buffer, min(batch_size, len(buffer)))
                S, A, R, S2, T = (torch.tensor(np.array(x), dtype=torch.float32) for x in zip(*batch))
                with torch.no_grad():                                         # the target is a fixed label
                    net = target_net if use_target else q_net
                    y = R + gamma * (1 - T) * net(S2).max(dim=1).values      # y = r + γ max_a' Q(s', a'; θ⁻)
                q = q_net(S).gather(1, A.long()[:, None]).squeeze(1)        # Q(s, a; θ)
                loss = ((y - q) ** 2).mean()                                  # (y − Q(s, a; θ))²
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                                              # one gradient step on θ

            # 4. every C steps, copy the online network into the target network
            if use_target and step % sync_every == 0:
                target_net.load_state_dict(q_net.state_dict())
        lengths.append(env.t)

    return untrained_score, lengths, greedy_test(q_net)


if __name__ == "__main__":
    torch.set_num_threads(1)
    before, lengths, after = train_dqn(seed=0)
    print(f"episodes played: {len(lengths)}")
    for i in range(0, len(lengths) - 99, 100):
        print(f"  episodes {i + 1:3d}-{min(i + 100, len(lengths)):3d}: average length {np.mean(lengths[i:i + 100]):6.1f}")
    print(f"average length, first 20 training episodes: {np.mean(lengths[:20]):6.1f}")
    print(f"average length, last 20 training episodes:  {np.mean(lengths[-20:]):6.1f}")
    print(f"greedy test (20 episodes), untrained network: {before:6.1f}")
    print(f"greedy test (20 episodes), trained network:   {after:6.1f}   (500 = never fell)")
```

On one CPU core this takes about 70 seconds:

```text
episodes played: 503
  episodes   1-100: average length   22.2
  episodes 101-200: average length   64.8
  episodes 201-300: average length  128.6
  episodes 301-400: average length  130.8
  episodes 401-500: average length  149.1
average length, first 20 training episodes:   27.1
average length, last 20 training episodes:   145.5
greedy test (20 episodes), untrained network:    9.3
greedy test (20 episodes), trained network:    136.7   (500 = never fell)
```

The untrained network drops the pole after about 9 steps; the trained one keeps it up for 137 steps on average with no exploration, and the training episodes grew from about 22 steps to about 150. It learns, but it doesn't reach the maximum of 500 within 50,000 steps on this seed.

**Ablation.** Same code, three random seeds each, switching off one trick at a time (average length of the last 20 training episodes, and of 20 greedy test episodes, one number per seed). The script below imports `train_dqn` from the code above (saved as `dqn_cartpole.py`) and runs the nine trainings in parallel:

```python
import multiprocessing as mp
import numpy as np, torch
from dqn_cartpole import train_dqn

CONFIGS = {"full DQN": {}, "no target network": {"use_target": False},
           "no replay buffer": {"use_replay": False}}

def job(args):
    torch.set_num_threads(1)
    name, seed = args
    _, lengths, after = train_dqn(seed=seed, **CONFIGS[name])
    return name, seed, np.mean(lengths[-20:]), after

if __name__ == "__main__":
    jobs = [(n, s) for n in CONFIGS for s in range(3)]
    with mp.Pool(len(jobs)) as pool:
        res = pool.map(job, jobs)
    for name in CONFIGS:
        rows = [r for r in res if r[0] == name]
        last = "  ".join(f"{r[2]:5.1f}" for r in rows)
        test = "  ".join(f"{r[3]:5.1f}" for r in rows)
        print(f"{name:18s} last 20 episodes: {last}   greedy test: {test}")
```

```text
full DQN           last 20 episodes: 145.5  412.4  223.3   greedy test: 136.7  215.4  357.5
no target network  last 20 episodes: 213.4  228.3  457.7   greedy test: 234.8  148.8  181.3
no replay buffer   last 20 episodes:  36.1   17.8   69.9   greedy test:   9.7  125.8   69.8
```

What this shows, and what it doesn't:

- **Seeds matter a lot.** The full DQN ended between 145 and 412 depending only on the random seed. DQN is famous for this; always run several seeds before believing a comparison.
- **Replay mattered clearly.** Learning from each transition once, in order, ended far lower on every seed (18 to 70 steps in training).
- **The target network didn't help on CartPole** in these runs: without it, the results were similar or even better. CartPole is tiny and its values are smooth, so the chasing effect of Step 5 didn't hurt here. On Atari it was different: the DQN paper's ablation on five games shows the score dropping on every game without the target network, and dropping much further without replay. Tricks that are essential at scale can be invisible on a toy problem, and the other way round.

The DQN paper's own network for Atari is the same idea with convolutions in front:

```python
atari_q_net = nn.Sequential(                       # input: 4 stacked 84×84 grey frames
    nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
    nn.Flatten(), nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
    nn.Linear(512, n_actions),                     # one Q-value per joystick action
)
```

For real projects, use a tested implementation such as [CleanRL's `dqn.py`](https://docs.cleanrl.dev/rl-algorithms/dqn/) or [Stable-Baselines3's `DQN`](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) with Gymnasium environments.

---

## Common misconceptions

!!! warning "\"DQN is a new learning rule.\""
    It is Q-learning. The update target $r + \gamma \max_{a'} Q(s',a')$ is the same; only the way the Q-function is stored (a network instead of a table) and two stabilisers (replay, target network) are new.

!!! warning "\"The target network is a second network that is trained too.\""
    It is never trained. It is a periodic **copy** of the online network, used only to compute targets. All gradient steps go into $\theta$.

!!! warning "\"Experience replay works with any RL algorithm.\""
    Only with **off-policy** methods. Replayed transitions were chosen by an older policy. Q-learning's target doesn't care (it uses the max), but on-policy methods like SARSA or vanilla policy gradients assume the data comes from the *current* policy, and naive replay breaks them.

!!! warning "\"Higher Q-values mean the agent is doing better.\""
    Not necessarily. Because of the max in the target, DQN's Q-values tend to **overestimate** the real returns. Judge an agent by the returns it actually collects, not by its own value estimates.

!!! warning "\"With a target network, DQN is guaranteed to converge.\""
    No. Function approximation, bootstrapping and off-policy data together (the deadly triad) can still diverge. The tricks make training stable in practice, not in theory.

---

## Check yourself

??? question "1. Why does a DQN have one output per action instead of taking the action as an input?"
    So that one forward pass gives $Q(s, a)$ for **all** actions. Both acting greedily and computing the target $\max_{a'} Q(s', a')$ need every action's value, and with the action as input that would take one forward pass per action.

??? question "2. A transition ends the episode (the pole fell). What is its target y, and why?"
    $y = r$. There is no next state to bootstrap from: the return after a terminal state is 0. In the code this is the `(1 - T)` factor. Forgetting it makes the agent believe falling is followed by more reward.

??? question "3. Which of DQN's two tricks addresses which problem?"
    **Experience replay** addresses correlated, non-stationary training data: random mini-batches from a large buffer are close to independent samples, and each transition is reused many times. The **target network** addresses the moving target: freezing $\theta^-$ for $C$ steps makes the labels fixed, so each phase is ordinary regression.

??? question "4. In the moving-target widget, why does the live target diverge even though the network can represent the true values exactly?"
    The weights are shared. A step that fixes the error at A or B also changes $Q(C)$, which is the target for A and B. With the target live, the update keeps chasing its own effect, and here the errors grow by a fixed factor each step for every step size. Freezing the target for enough steps lets each regression phase finish before the target moves.

??? question "5. What does Double DQN change, and what problem does it fix?"
    The target action is **chosen** by the online network but **evaluated** by the target network: $y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$. Taking a max over noisy estimates is biased upwards (overestimation); using one estimate to choose and an independent one to evaluate removes most of that bias.

---

## Further reading

- Volodymyr Mnih et al., [*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236), Nature 2015. The DQN paper, with the target network and the Atari results. The earlier workshop version is [*Playing Atari with Deep Reinforcement Learning*](https://arxiv.org/abs/1312.5602) (2013).
- Hado van Hasselt, Arthur Guez & David Silver, [*Deep Reinforcement Learning with Double Q-learning*](https://arxiv.org/abs/1509.06461), 2016.
- Ziyu Wang et al., [*Dueling Network Architectures for Deep RL*](https://arxiv.org/abs/1511.06581), 2016, and Tom Schaul et al., [*Prioritized Experience Replay*](https://arxiv.org/abs/1511.05952), 2016.
- Matteo Hessel et al., [*Rainbow: Combining Improvements in Deep RL*](https://arxiv.org/abs/1710.02298), 2018.
- Richard Sutton & Andrew Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html), Chapter 9 (function approximation) and Section 11.3 (the deadly triad).
- PyTorch, [*Reinforcement Learning (DQN) Tutorial*](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), a DQN on Gymnasium's CartPole.

**Next lessons:** [Policy gradients](../policy-gradients/index.md)

<script src="viz/deep-q-networks.js"></script>
