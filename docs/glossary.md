# Glossary

Every term used on this site, in one sentence. Terms link to the lesson where they're explained.

**Action** ($a$): a choice the agent makes at one time step, such as moving left or pressing a button. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**Agent**: in reinforcement learning, the learner and decision-maker that observes states, takes actions and receives rewards. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**Batch gradient descent**: gradient descent where each step uses the gradient computed from *all* training examples. ([Gradient Descent](topics/gradient-descent/index.md#step-6-real-data-is-big-stochastic-gradient-descent-sgd))

**Convex function**: a bowl-shaped function with a single minimum; gradient descent with a suitable learning rate is guaranteed to find it.

**Credit assignment problem**: working out which of many earlier actions deserve the credit (or blame) for a reward that arrives later. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-1-learning-from-consequences-not-from-answers))

**Discount factor** ($\gamma$): a number between 0 and 1 that makes rewards further in the future count for less in the return. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-3-what-does-best-mean-the-return-and-discounting))

**Divergence**: when the learning rate is too large and each update overshoots further, so the loss grows instead of shrinking. ([Gradient Descent](topics/gradient-descent/index.md#step-4-taking-the-step-the-update-rule))

**Environment**: everything outside the agent; it responds to each action with a new state and a reward. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**Episode**: one run of agent–environment interaction, from a starting state to a terminal state (e.g. one game). ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**Epoch**: one full pass over the entire training dataset.

**ε-greedy (epsilon-greedy)**: a policy that usually takes the best-known action but, with small probability ε, takes a random one to explore. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-6-explore-or-exploit))

**Exploration vs. exploitation**: the trade-off between trying new actions to learn more (explore) and choosing the best-known action to collect reward (exploit). ([What is RL?](topics/reinforcement-learning-intro/index.md#step-6-explore-or-exploit))

**Gradient** ($\nabla L$): the list of slopes of the loss with respect to each parameter; it points in the direction of steepest increase. ([Gradient Descent](topics/gradient-descent/index.md#step-3-which-way-is-downhill-the-gradient))

**Gradient descent**: an algorithm that repeatedly moves parameters a small step opposite to the gradient, to reduce the loss. ([Gradient Descent](topics/gradient-descent/index.md))

**Learning rate** ($\eta$): the number that controls how big each gradient-descent step is. ([Gradient Descent](topics/gradient-descent/index.md#step-4-taking-the-step-the-update-rule))

**Loss (loss function)**: a single number that measures how wrong a model's predictions are; training tries to make it small. ([Gradient Descent](topics/gradient-descent/index.md#step-2-what-are-we-trying-to-make-small-the-loss))

**Mean squared error (MSE)**: a loss that averages the squared differences between predictions and true values.

**Mini-batch**: a small random subset of the training data used to estimate the gradient for one step.

**Parameters (weights)**: the adjustable numbers inside a model that training changes.

**Policy** ($\pi$): the agent's strategy, a rule that gives the action (or a probability for each action) in every state. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-4-the-agents-strategy-the-policy))

**Q-value / action-value** ($Q^\pi(s,a)$): the expected return from taking action $a$ in state $s$ and following policy $\pi$ afterwards. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-5-how-good-is-a-state-the-value-function))

**Reinforcement learning (RL)**: learning to act by trial and error, using rewards instead of correct answers, to maximise total future reward. ([What is RL?](topics/reinforcement-learning-intro/index.md))

**Return** ($G_t$): the (discounted) sum of all rewards from time step $t$ until the end of the episode; the quantity an RL agent tries to maximise. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-3-what-does-best-mean-the-return-and-discounting))

**Reward** ($r$): the number the environment sends back after each action, saying how good the immediate outcome was. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**State** ($s$): a description of the situation the agent is in at a given time step. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**Stochastic gradient descent (SGD)**: gradient descent that estimates the gradient from one random example (or a mini-batch) per step. ([Gradient Descent](topics/gradient-descent/index.md#step-6-real-data-is-big-stochastic-gradient-descent-sgd))

**Value function** ($V^\pi(s)$): the expected return when starting in state $s$ and following policy $\pi$. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-5-how-good-is-a-state-the-value-function))
