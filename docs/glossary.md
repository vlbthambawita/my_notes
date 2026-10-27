# Glossary

Every term used on this site, in one sentence. Terms link to the lesson where they're explained.

**10-armed testbed**: the standard benchmark from Sutton & Barto, made of many random 10-armed bandits with $q_*(a) \sim \mathcal{N}(0,1)$ and unit-variance reward noise, used to compare exploration methods. ([Multi-armed bandits](topics/multi-armed-bandits/index.md#step-4-greedy-vs-greedy))

**1-D convolution**: convolution over a sequence such as an ECG, where the window slides only along time. ([Convolutional Tsetlin machines](topics/convolutional-tsetlin-machine/index.md#step-2-looking-through-a-window-patches-and-patch-features))

**A2C (advantage actor-critic)**: an actor-critic algorithm that collects short rollouts from several parallel environments and takes one gradient step per batch on an advantage-weighted policy loss plus a value loss and an entropy bonus. ([Actor-critic](topics/actor-critic/index.md#step-4-a2c-batches-parallel-workers-and-gae))

**Action** ($a$): a choice the agent makes at one time step, such as moving left or pressing a button. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**Action preference** (h(s, a)): the raw score a policy network gives each action; a softmax turns these scores into probabilities. ([Policy gradients](topics/policy-gradients/index.md#step-2-a-policy-with-knobs))

**Actor-critic**: a family of RL methods that pairs a policy (the actor) with a learned value function (the critic), whose TD error tells the actor whether an action was better or worse than expected. ([Actor-critic](topics/actor-critic/index.md#step-3-the-actor-critic-loop))

**Advantage** (A^π(s,a) = Q^π(s,a) − V^π(s)): how much better an action is than the policy's average behaviour in that state; G_t − V(s_t) is a sample estimate of it. ([Policy gradients](topics/policy-gradients/index.md#step-6-taming-the-noise-baselines))

**Agent**: in reinforcement learning, the learner and decision-maker that observes states, takes actions and receives rewards. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**AlphaZero**: A game-playing agent that guides Monte Carlo Tree Search with a policy-and-value network and trains that network by self-play on the search's results. ([Model-based RL & planning](topics/model-based-rl/index.md#step-6-alphago-alphazero-and-muzero-search-deep-learning))

**Backup** (backup diagram): computing a state's value from the rewards and values one step ahead, drawn as a tree of action branches then outcome branches. ([MDPs & Bellman equations](topics/mdp-bellman-equations/index.md#step-4-the-bellman-expectation-equation-for-vpi))

**Baseline** (b(s)): a value subtracted from the return in a policy-gradient update; it does not depend on the action, so the update stays unbiased while its variance drops. ([Policy gradients](topics/policy-gradients/index.md#step-6-taming-the-noise-baselines))

**Batch gradient descent**: gradient descent where each step uses the gradient computed from *all* training examples. ([Gradient Descent](topics/gradient-descent/index.md#step-6-real-data-is-big-stochastic-gradient-descent-sgd))

**Behaviour cloning** (BC): imitation learning done as plain supervised learning, mapping each recorded state to the expert's action. ([Offline RL & imitation](topics/offline-rl-imitation/index.md#step-2-imitation-learning-behaviour-cloning-is-supervised-learning))

**Behaviour policy**: the policy that actually chooses the agent's actions and so generates its experience, e.g. ε-greedy. ([Q-learning & SARSA](topics/q-learning-sarsa/index.md#step-6-on-policy-versus-off-policy))

**Bellman expectation equation**: $V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s',r} p(s', r \mid s, a)\,[r + \gamma V^\pi(s')]$; the value of a state equals the expected one-step reward plus the discounted value of the next state. ([MDPs & Bellman equations](topics/mdp-bellman-equations/index.md#step-4-the-bellman-expectation-equation-for-vpi))

**Bellman optimality equation**: the Bellman equation with the average over actions replaced by a max, $V^*(s) = \max_a \sum_{s',r} p(s', r \mid s, a)\,[r + \gamma V^*(s')]$, satisfied only by the optimal values. ([MDPs & Bellman equations](topics/mdp-bellman-equations/index.md#step-6-the-bellman-optimality-equations))

**Bellman residual**: the gap between a value estimate and the right-hand side of its Bellman equation; it is zero for the true values. ([MDPs & Bellman equations](topics/mdp-bellman-equations/index.md#step-4-the-bellman-expectation-equation-for-vpi))

**Boolean feature** ($x_k$): a property of an object that is either True (1) or False (0), such as "has wings" or "tumour ≥ 20 mm"; the only kind of input a Tsetlin machine reads. ([What is a Tsetlin machine?](topics/tsetlin-machine-intro/index.md#step-1-objects-described-by-yesno-facts))

**Booleanisation**: turning raw data into Boolean features, for example one feature per category value or one per threshold on a number. ([What is a Tsetlin machine?](topics/tsetlin-machine-intro/index.md#step-2-booleanising-real-data))

**Bootstrapping**: updating an estimate toward a target that contains another estimate, such as $V(s')$: "learning a guess from a guess". ([Monte Carlo & TD learning](topics/monte-carlo-td-learning/index.md#step-3-td0-learn-a-guess-from-a-guess))

**Bradley–Terry model**: A model of pairwise choices where the probability that A beats B is σ(r_A − r_B), so only score differences matter. ([RLHF & DPO](topics/rlhf-dpo/index.md#step-4-learning-a-reward-from-comparisons-bradleyterry))

**Clipped objective** ($L^{\text{CLIP}}$): PPO's policy objective $\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t)$, which gives no extra credit for moving an action's probability ratio outside $[1-\varepsilon, 1+\varepsilon]$. ([Actor-critic](topics/actor-critic/index.md#step-5-ppo-dont-move-the-policy-too-far))

**Composite vote sum**: the sum of the normalised vote sums of all Tsetlin machines in a composite; its sign decides the class (argmax per class for multi-class) and its size is the composite's confidence. ([Confidence & composites](topics/tsetlin-machine-confidence/index.md#step-7-normalisation-one-scale-for-every-machine))

**Compounding errors**: in imitation learning, small mistakes lead the learner to unfamiliar states where it makes more mistakes, so the error can grow like $\varepsilon T^2$ instead of $\varepsilon T$. ([Offline RL & imitation](topics/offline-rl-imitation/index.md#step-3-why-copying-drifts-compounding-errors))

**Confidence** (|v|): how sure a Tsetlin machine is about a prediction, measured by the size of its vote sum (for multi-class, the largest class vote sum). ([Confidence & composites](topics/tsetlin-machine-confidence/index.md#step-1-confidence-is-the-size-of-the-vote-sum))

**Conservative Q-Learning** (CQL): an offline RL method that adds a loss term pushing Q-values down on all actions and up on the dataset's actions, so unseen actions are never overestimated. ([Offline RL & imitation](topics/offline-rl-imitation/index.md#step-6-the-fix-be-pessimistic-about-what-you-have-not-seen))

**Contextual bandit**: a bandit where the agent sees a context (such as a user profile) before choosing, and the best arm depends on that context. ([Multi-armed bandits](topics/multi-armed-bandits/index.md#putting-it-all-together))

**Convex function**: a bowl-shaped function with a single minimum; gradient descent with a suitable learning rate is guaranteed to find it.

**Convolution window**: a small rectangle, such as 3×3 pixels, placed at every position of an image so that a rule sees one piece of the image at a time. ([Convolutional Tsetlin machines](topics/convolutional-tsetlin-machine/index.md#step-2-looking-through-a-window-patches-and-patch-features))

**Convolutional Tsetlin machine**: a Tsetlin machine whose rules are matched against every window patch of the input and fire if they match anywhere, so one rule recognises a pattern wherever it appears. ([Convolutional Tsetlin machines](topics/convolutional-tsetlin-machine/index.md#step-3-evaluating-a-rule-match-every-patch-then-or))

**Coverage**: the fraction of a class's examples that a rule matches. ([Recognize, Erase & Reject feedback](topics/tsetlin-machine-feedback/index.md#step-8-train-one-rule-yourself))

**Coverage/accuracy trade-off**: answering only cases whose confidence reaches a threshold raises accuracy on the answered cases but leaves more cases to refer to a human. ([Confidence & composites](topics/tsetlin-machine-confidence/index.md#step-3-confidence-on-real-patients))

**Credit assignment problem**: working out which of many earlier actions deserve the credit (or blame) for a reward that arrives later. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-1-learning-from-consequences-not-from-answers))

**Curse of dimensionality**: the number of states grows exponentially with the number of variables describing a state, so sweeping over all of them quickly becomes impossible. ([Dynamic programming](topics/dynamic-programming/index.md#step-6-the-catch-you-need-the-model-and-you-need-every-state))

**DAgger** (Dataset Aggregation): imitation learning that runs the learner, has the expert label the states the learner visits, adds them to the dataset and retrains. ([Offline RL & imitation](topics/offline-rl-imitation/index.md#step-4-dagger-ask-the-expert-about-your-own-mistakes))

**Deadly triad**: Function approximation, bootstrapping and off-policy data together, a combination that can make value learning diverge. ([Deep Q-Networks](topics/deep-q-networks/index.md#step-5-problem-2-chasing-a-moving-target-the-target-network))

**Decision-time planning**: planning focused on the current state at the moment an action must be chosen, for example by growing a search tree, rather than improving values for all states in the background. ([Model-based RL & planning](topics/model-based-rl/index.md#step-5-planning-at-decision-time-monte-carlo-tree-search))

**Deep Q-Network** (DQN): Q-learning with a neural network as the Q-function, made stable with an experience replay buffer and a target network. ([Deep Q-Networks](topics/deep-q-networks/index.md#step-2-the-q-network-one-output-per-action))

**Direct Preference Optimization (DPO)**: trains a policy directly on preference pairs with the loss −log σ(β log π_θ(y_w)/π_ref(y_w) − β log π_θ(y_l)/π_ref(y_l)), with no reward model or RL loop. ([RLHF & DPO](topics/rlhf-dpo/index.md#step-6-dpo-skip-the-reward-model-and-the-rl-loop))

**Discount factor** ($\gamma$): a number between 0 and 1 that makes rewards further in the future count for less in the return. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-3-what-does-best-mean-the-return-and-discounting))

**Distribution shift**: when the states a policy meets at test time differ from the states it was trained on, for example because its own actions lead it elsewhere. ([Offline RL & imitation](topics/offline-rl-imitation/index.md#step-3-why-copying-drifts-compounding-errors))

**Divergence**: when the learning rate is too large and each update overshoots further, so the loss grows instead of shrinking. ([Gradient Descent](topics/gradient-descent/index.md#step-4-taking-the-step-the-update-rule))

**Double DQN**: A DQN variant whose target picks the next action with the online network but scores it with the target network, removing most of the overestimation bias. ([Deep Q-Networks](topics/deep-q-networks/index.md#step-7-improvements-double-dqn-dueling-networks-prioritised-replay))

**Dueling network**: A Q-network whose last layer splits into a state-value stream V(s) and an advantage stream A(s, a) that are recombined into Q(s, a). ([Deep Q-Networks](topics/deep-q-networks/index.md#step-7-improvements-double-dqn-dueling-networks-prioritised-replay))

**Dynamic programming (DP)**: computing optimal policies from a known model p(s′, r | s, a) by turning the Bellman equations into update rules applied to every state. ([Dynamic programming](topics/dynamic-programming/index.md#step-1-turn-the-bellman-equation-into-an-update-rule-policy-evaluation))

**Dynamics function** ($p(s', r \mid s, a)$): the probability of landing in state $s'$ with reward $r$ after taking action $a$ in state $s$; the "rules of the game" of an MDP. ([MDPs & Bellman equations](topics/mdp-bellman-equations/index.md#step-2-the-markov-decision-process))

**Dyna-Q**: Q-learning plus a learned model: after every real step it makes n extra Q-learning updates on transitions imagined by the model. ([Model-based RL & planning](topics/model-based-rl/index.md#step-3-dyna-q-learn-model-plan))

**Entropy bonus**: a term added to the loss that rewards the policy for keeping its action probabilities spread out, so it keeps exploring. ([Actor-critic](topics/actor-critic/index.md#step-4-a2c-batches-parallel-workers-and-gae))

**Environment**: everything outside the agent; it responds to each action with a new state and a reward. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**Episode**: one run of agent–environment interaction, from a starting state to a terminal state (e.g. one game). ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**Epoch**: one full pass over the entire training dataset.

**ε-greedy (epsilon-greedy)**: a policy that usually takes the best-known action but, with small probability ε, takes a random one to explore. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-6-explore-or-exploit))

**Erase Feedback** (Type Ib): when a rule fails to match an example of its own class, every literal is forgotten one step with probability equal to the Forget Value. ([Recognize, Erase & Reject feedback](topics/tsetlin-machine-feedback/index.md#step-4-erase-feedback-loosen-a-rule-that-is-too-picky))

**Expected SARSA**: A variant of SARSA whose TD target averages Q(s′, a′) over the policy's action probabilities instead of using one sampled next action, which gives less noisy updates. ([Q-learning & SARSA](topics/q-learning-sarsa/index.md#step-6-on-policy-versus-off-policy))

**Experience replay** (replay buffer): storing past transitions (s, a, r, s′) in a buffer and training on random mini-batches from it, which decorrelates the data and reuses experience. ([Deep Q-Networks](topics/deep-q-networks/index.md#step-4-problem-1-consecutive-samples-are-correlated-experience-replay))

**Exploration vs. exploitation**: the trade-off between trying new actions to learn more (explore) and choosing the best-known action to collect reward (exploit). ([What is RL?](topics/reinforcement-learning-intro/index.md#step-6-explore-or-exploit))

**Extrapolation error**: in offline RL, the error from bootstrapping on Q-values of actions that never appear in the data; the max picks them when they are too high, causing overestimation. ([Offline RL & imitation](topics/offline-rl-imitation/index.md#step-5-offline-rl-and-why-plain-q-learning-breaks))

**First-visit Monte Carlo**: A Monte Carlo method that uses only the return after the first visit to a state in each episode. ([Monte Carlo & TD learning](topics/monte-carlo-td-learning/index.md#step-2-monte-carlo-average-the-returns-you-actually-got))

**Forget Value** (1/s): the probability that a forgetting step (a decrement) actually happens; it also acts as the frequency threshold a literal must beat to stay in a rule. ([Recognize, Erase & Reject feedback](topics/tsetlin-machine-feedback/index.md#step-6-recognize-erase-find-frequent-patterns))

**Frequent pattern**: a combination of literals that is True for most examples of a class, found by Recognize and Erase Feedback. ([Recognize, Erase & Reject feedback](topics/tsetlin-machine-feedback/index.md#step-6-recognize-erase-find-frequent-patterns))

**Function approximation**: representing a value function or policy with a parameterised function, such as a neural network, instead of a table, so it can generalise to unseen states. ([Deep Q-Networks](topics/deep-q-networks/index.md#step-1-why-q-tables-break))

**Gaussian policy** (a ~ N(μ_θ(s), σ²)): a policy for continuous actions whose network outputs the mean (and often the spread) of a normal distribution that actions are sampled from. ([Policy gradients](topics/policy-gradients/index.md#step-2-a-policy-with-knobs))

**Generalized Advantage Estimation (GAE)** ($\lambda$): an advantage estimate that blends multi-step TD errors, $\hat{A}_t = \sum_k (\gamma\lambda)^k \delta_{t+k}$, trading bias ($\lambda=0$) against variance ($\lambda=1$). ([Actor-critic](topics/actor-critic/index.md#step-4-a2c-batches-parallel-workers-and-gae))

**Generalized policy iteration (GPI)**: the general pattern of interleaving policy evaluation and policy improvement, which almost every RL method follows. ([Dynamic programming](topics/dynamic-programming/index.md#step-5-the-big-picture-generalized-policy-iteration))

**Gradient** ($\nabla L$): the list of slopes of the loss with respect to each parameter; it points in the direction of steepest increase. ([Gradient Descent](topics/gradient-descent/index.md#step-3-which-way-is-downhill-the-gradient))

**Gradient descent**: an algorithm that repeatedly moves parameters a small step opposite to the gradient, to reduce the loss. ([Gradient Descent](topics/gradient-descent/index.md))

**Greedy action**: the action with the highest current value estimate, $\arg\max_a Q_t(a)$; choosing it is exploiting. ([Multi-armed bandits](topics/multi-armed-bandits/index.md#step-2-action-values-what-is-each-arm-worth))

**Greedy policy** ($\pi^*(s) = \arg\max_a Q^*(s,a)$): a policy that always picks the action with the highest action-value; greedy with respect to $Q^*$ is optimal. ([MDPs & Bellman equations](topics/mdp-bellman-equations/index.md#step-6-the-bellman-optimality-equations))

**Image patch** (patch features $x_{i,j}$): the pixels inside the convolution window at one position; their truth values are the patch features, which mean the same thing at every position. ([Convolutional Tsetlin machines](topics/convolutional-tsetlin-machine/index.md#step-2-looking-through-a-window-patches-and-patch-features))

**Imitation learning**: learning a policy from an expert's demonstrations (states and actions) instead of from rewards. ([Offline RL & imitation](topics/offline-rl-imitation/index.md#step-1-three-ways-to-get-experience-online-off-policy-offline))

**Implicit Q-Learning** (IQL): an offline RL method that never queries unseen actions; it learns values only on dataset actions and extracts a policy by advantage-weighted behaviour cloning. ([Offline RL & imitation](topics/offline-rl-imitation/index.md#step-6-the-fix-be-pessimistic-about-what-you-have-not-seen))

**Incremental update**: updating an estimate one sample at a time as new estimate = old estimate + step size × (target − old estimate). ([Multi-armed bandits](topics/multi-armed-bandits/index.md#step-3-learning-incrementally-new-estimate-old-estimate-step-error))

**Interpretability**: the property of a model whose reasoning a person can read and check directly, as with a Tsetlin machine's AND-rules; it does not by itself make the model accurate. ([What is a Tsetlin machine?](topics/tsetlin-machine-intro/index.md#why-should-i-care))

**Iterative policy evaluation**: computing V^π by repeatedly applying the Bellman expectation equation as an assignment until the values stop changing. ([Dynamic programming](topics/dynamic-programming/index.md#step-1-turn-the-bellman-equation-into-an-update-rule-policy-evaluation))

**KL penalty** (β): A term −β·KL(π‖π_ref) added to the RLHF objective that keeps the fine-tuned policy close to the reference model. ([RLHF & DPO](topics/rlhf-dpo/index.md#step-5-rl-fine-tuning-with-a-kl-leash))

**Learning rate** ($\eta$): the number that controls how big each gradient-descent step is. ([Gradient Descent](topics/gradient-descent/index.md#step-4-taking-the-step-the-update-rule))

**Literal** ($x_k$, $\lnot x_k$): a Boolean feature or its negation; with $o$ features there are $2o$ literals, and exactly half are True for any object. ([What is a Tsetlin machine?](topics/tsetlin-machine-intro/index.md#step-3-literals-a-feature-or-its-opposite))

**Literal game**: the two literals of a rule seen as players that each choose Memorized or Forgotten without communicating; each one's feedback depends on the other's choice. ([Recognize, Erase & Reject feedback](topics/tsetlin-machine-feedback/index.md#step-9-two-literals-one-rule-a-game-without-talking))

**Log-derivative trick** (∇π = π ∇ln π): the identity that turns the gradient of an expectation into an expectation you can estimate by sampling, so the environment's unknown dynamics drop out. ([Policy gradients](topics/policy-gradients/index.md#step-4-the-policy-gradient-make-good-actions-more-likely))

**Loss (loss function)**: a single number that measures how wrong a model's predictions are; training tries to make it small. ([Gradient Descent](topics/gradient-descent/index.md#step-2-what-are-we-trying-to-make-small-the-loss))

**Markov chain**: a random process whose next state depends only on the current state, like a literal's memory position moving one step per training example. ([The Tsetlin automaton](topics/tsetlin-automaton/index.md#step-4-one-literal-one-class-a-random-walk))

**Markov decision process** (MDP): the formal model of an RL problem: states, actions, dynamics $p(s', r \mid s, a)$ and a discount $\gamma$. ([MDPs & Bellman equations](topics/mdp-bellman-equations/index.md#step-2-the-markov-decision-process))

**Markov property**: the next state and reward depend only on the current state and action, not on the earlier history. ([MDPs & Bellman equations](topics/mdp-bellman-equations/index.md#step-1-the-markov-property-the-state-is-all-you-need))

**Match map**: the grid of True/False results of one rule at every window position, the Boolean version of a CNN feature map, which the OR collapses to one bit. ([Convolutional Tsetlin machines](topics/convolutional-tsetlin-machine/index.md#step-3-evaluating-a-rule-match-every-patch-then-or))

**Maximum-entropy RL** ($\alpha$): an objective that maximises reward plus the policy's entropy weighted by a temperature $\alpha$, as used by SAC. ([Actor-critic](topics/actor-critic/index.md#step-6-sac-off-policy-continuous-actions-maximum-entropy))

**Mean squared error (MSE)**: a loss that averages the squared differences between predictions and true values.

**Memorize Value** ((s−1)/s): the probability that a memorizing step (an increment) under Recognize Feedback actually happens; usually Memorize Value + Forget Value = 1. ([Recognize, Erase & Reject feedback](topics/tsetlin-machine-feedback/index.md#step-3-recognize-feedback-become-more-like-this-example))

**Memorized / Forgotten**: the upper and lower halves of a literal's memory; only Memorized literals take part in the rule's AND. ([The Tsetlin automaton](topics/tsetlin-automaton/index.md#step-3-from-actions-to-memory-one-integer-per-literal))

**Memory depth** (D): the number of memory positions per literal (the book's Chapter 1 uses 10), equal to the 2N states of a Tsetlin automaton; deeper memory gives sharper, more decisive but slower-changing outcomes. ([The Tsetlin automaton](topics/tsetlin-automaton/index.md#step-6-deeper-memory-sharper-decisions))

**Mini-batch**: a small random subset of the training data used to estimate the gradient for one step.

**Model (of the environment)** ($\hat p(s', r \mid s, a)$): the agent's own prediction of what happens after taking action $a$ in state $s$: the next state and the reward, or their probabilities. ([Model-based RL & planning](topics/model-based-rl/index.md#step-1-what-is-a-model))

**Model-based RL**: reinforcement learning that uses a model of the environment, given or learned, to plan with imagined experience instead of relying only on real experience. ([Model-based RL & planning](topics/model-based-rl/index.md#step-1-what-is-a-model))

**Model-free**: describes methods that learn values or policies straight from experience, without knowing or learning the transition probabilities $p(s', r \mid s, a)$. ([Monte Carlo & TD learning](topics/monte-carlo-td-learning/index.md#step-1-learning-without-a-model))

**Monte Carlo prediction**: estimating $V^\pi(s)$ by averaging the complete returns actually observed after visiting $s$. ([Monte Carlo & TD learning](topics/monte-carlo-td-learning/index.md#step-2-monte-carlo-average-the-returns-you-actually-got))

**Monte Carlo Tree Search (MCTS)**: A planning method that grows a search tree from the current state, one simulation at a time (selection, expansion, simulation, backup), and then plays the most-visited move. ([Model-based RL & planning](topics/model-based-rl/index.md#step-5-planning-at-decision-time-monte-carlo-tree-search))

**Multi-armed bandit** (k-armed bandit): the simplest RL problem, with one unchanging state where the agent repeatedly picks one of k actions ("arms"), each paying a random reward from its own hidden distribution. ([Multi-armed bandits](topics/multi-armed-bandits/index.md#step-1-the-k-armed-bandit-one-state-k-actions))

**Multi-class Tsetlin machine**: one team of rules (for and against) per class; the class whose team has the highest vote sum wins. ([Classification by voting](topics/tsetlin-machine-classification/index.md#step-8-more-than-two-classes))

**MuZero**: an AlphaZero-style agent that is not given the rules; it learns a model in a hidden state space and runs MCTS inside it. ([Model-based RL & planning](topics/model-based-rl/index.md#step-6-alphago-alphazero-and-muzero-search-deep-learning))

**Normalised vote sum**: a machine's vote sum divided by the difference between the highest and lowest vote sums it produces on the data, so that machines of different sizes speak on one scale. ([Confidence & composites](topics/tsetlin-machine-confidence/index.md#step-7-normalisation-one-scale-for-every-machine))

**n-step TD**: A TD method that adds up $n$ real rewards and then bootstraps from $V(s_{t+n})$; it sits between TD(0) ($n=1$) and Monte Carlo ($n$ = to the end). ([Monte Carlo & TD learning](topics/monte-carlo-td-learning/index.md#step-7-between-the-two-n-step-td-and-backup-diagrams))

**Offline RL** (batch RL): reinforcement learning from a fixed, previously logged dataset of $(s, a, r, s')$ with no interaction with the environment during training. ([Offline RL & imitation](topics/offline-rl-imitation/index.md#step-1-three-ways-to-get-experience-online-off-policy-offline))

**Off-policy learning**: learning the values of one policy (the target policy, e.g. greedy) from experience generated by a different behaviour policy (e.g. ε-greedy). ([Q-learning & SARSA](topics/q-learning-sarsa/index.md#step-6-on-policy-versus-off-policy))

**On-policy learning**: learning the values of the same policy that is generating the experience, exploration included. ([Q-learning & SARSA](topics/q-learning-sarsa/index.md#step-6-on-policy-versus-off-policy))

**Optimal value functions** ($V^*$, $Q^*$): the highest value any policy can reach from each state (or state-action pair). ([MDPs & Bellman equations](topics/mdp-bellman-equations/index.md#step-6-the-bellman-optimality-equations))

**Optimistic initial values**: starting every value estimate far too high, so that even a greedy agent tries every action until reality lowers its estimates. ([Multi-armed bandits](topics/multi-armed-bandits/index.md#step-5-optimistic-initial-values))

**Overestimation bias**: the upward bias that comes from taking a max over noisy value estimates, as in the Q-learning target. ([Deep Q-Networks](topics/deep-q-networks/index.md#step-7-improvements-double-dqn-dueling-networks-prioritised-replay))

**Parameterised policy** (π_θ(a | s)): a policy given by a function with adjustable weights θ (usually a neural network) that outputs action probabilities. ([Policy gradients](topics/policy-gradients/index.md#step-2-a-policy-with-knobs))

**Parameters (weights)**: the adjustable numbers inside a model that training changes.

**Partially observable MDP** (POMDP): a problem where the agent's observation misses information needed to predict the future, so it is not Markov. ([MDPs & Bellman equations](topics/mdp-bellman-equations/index.md#step-1-the-markov-property-the-state-is-all-you-need))

**Pessimism** (offline RL): the principle of distrusting values for actions the data doesn't support, by constraining or penalising them (BCQ, CQL, IQL). ([Offline RL & imitation](topics/offline-rl-imitation/index.md#step-6-the-fix-be-pessimistic-about-what-you-have-not-seen))

**Planning**: computing or improving a policy from experience imagined with a model, without acting in the real environment. ([Model-based RL & planning](topics/model-based-rl/index.md#step-1-what-is-a-model))

**Policy** ($\pi$): the agent's strategy, a rule that gives the action (or a probability for each action) in every state. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-4-the-agents-strategy-the-policy))

**Policy gradient theorem** (∇_θ J = E[Σ_t G_t ∇_θ ln π_θ(a_t | s_t)]): the result that the gradient of expected return can be estimated from sampled episodes, without a model of the environment. ([Policy gradients](topics/policy-gradients/index.md#step-4-the-policy-gradient-make-good-actions-more-likely))

**Policy improvement theorem**: acting greedily with respect to V^π gives a policy at least as good as π in every state. ([Dynamic programming](topics/dynamic-programming/index.md#step-2-policy-improvement-act-greedily-on-the-values))

**Policy iteration**: alternating full policy evaluation and greedy improvement until the policy stops changing, at which point it is optimal. ([Dynamic programming](topics/dynamic-programming/index.md#step-3-policy-iteration-evaluate-improve-repeat))

**Policy objective** (J(θ)): the expected return of the policy with weights θ, which policy-gradient methods maximise by gradient ascent. ([Policy gradients](topics/policy-gradients/index.md#step-3-the-goal-climb-the-expected-return))

**Policy-based method**: an RL method that learns the policy directly instead of deriving it from learned values. ([Policy gradients](topics/policy-gradients/index.md#step-1-why-learn-the-policy-directly))

**Position encoding**: adding each patch's window coordinates as extra Boolean features, so a rule can restrict its pattern to a rectangular region. ([Convolutional Tsetlin machines](topics/convolutional-tsetlin-machine/index.md#step-6-position-encoding-letting-a-rule-say-where))

**PPO (Proximal Policy Optimization)**: an on-policy actor-critic algorithm that runs several epochs of minibatch updates per batch while a clipped objective keeps each policy change small; it is the RL step in classic RLHF. ([Actor-critic](topics/actor-critic/index.md#step-5-ppo-dont-move-the-policy-too-far))

**Precision** (of a rule): among the examples a rule matches, the fraction that really belong to its class. ([Recognize, Erase & Reject feedback](topics/tsetlin-machine-feedback/index.md#step-8-train-one-rule-yourself))

**Prioritised experience replay**: sampling replay transitions in proportion to their TD error, with importance weights to correct the bias this introduces. ([Deep Q-Networks](topics/deep-q-networks/index.md#step-7-improvements-double-dqn-dueling-networks-prioritised-replay))

**Probability ratio** ($r_t(\theta)$): how much more or less likely the new policy is to take the logged action than the policy that collected the data, $\pi_\theta(a_t \mid s_t)/\pi_{\text{old}}(a_t \mid s_t)$. ([Actor-critic](topics/actor-critic/index.md#step-5-ppo-dont-move-the-policy-too-far))

**PUCT**: alphaZero's selection rule $Q(s,a) + c\,P(s,a)\sqrt{N(s)}/(1+N(s,a))$, which adds the network's move prior to a UCB-style exploration bonus. ([Model-based RL & planning](topics/model-based-rl/index.md#step-6-alphago-alphazero-and-muzero-search-deep-learning))

**Q-learning**: A model-free TD control method that updates Q(s, a) towards r + γ max_a′ Q(s′, a′), so it learns the optimal greedy values Q* while behaving ε-greedily. ([Q-learning & SARSA](topics/q-learning-sarsa/index.md#step-3-q-learning-learn-about-the-best-move-whatever-you-do-next))

**Q-network** (Q(s, a; θ)): A neural network that takes a state and outputs one Q-value per action. ([Deep Q-Networks](topics/deep-q-networks/index.md#step-2-the-q-network-one-output-per-action))

**Q-value / action-value** ($Q^\pi(s,a)$): the expected return from taking action $a$ in state $s$ and following policy $\pi$ afterwards. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-5-how-good-is-a-state-the-value-function))

**Random matching patch**: in convolutional learning, Recognize and Reject Feedback use the literals of one patch picked at random among those the rule matched. ([Convolutional Tsetlin machines](topics/convolutional-tsetlin-machine/index.md#step-4-learning-which-patch-does-the-rule-learn-from))

**Random walk**: a position that moves up or down one step at a time at random; the memory position of a literal under training is one. ([The Tsetlin automaton](topics/tsetlin-automaton/index.md#step-4-one-literal-one-class-a-random-walk))

**Recognize Feedback** (Type Ia): when a rule matches an example of its own class, True literals are memorized (Memorize Value) and False literals forgotten (Forget Value). ([Recognize, Erase & Reject feedback](topics/tsetlin-machine-feedback/index.md#step-3-recognize-feedback-become-more-like-this-example))

**Reference policy** (π_ref): the frozen starting model (usually the SFT model) that the KL penalty and DPO's log-ratios measure the policy against. ([RLHF & DPO](topics/rlhf-dpo/index.md#step-3-the-rlhf-recipe-in-three-stages))

**Regret**: the total expected reward lost by not always choosing the best action, $\sum_t \big(q_*(a^*) - q_*(a_t)\big)$. ([Multi-armed bandits](topics/multi-armed-bandits/index.md#step-7-regret-the-yardstick-for-exploration))

**REINFORCE**: the basic Monte Carlo policy-gradient algorithm: play an episode, then raise the log-probability of each action in proportion to the return that followed it. ([Policy gradients](topics/policy-gradients/index.md#step-5-reinforce-the-algorithm))

**Reinforcement learning from human feedback (RLHF)**: aligning a model by learning a reward from human comparisons and then optimising the model against it with RL (typically PPO). ([RLHF & DPO](topics/rlhf-dpo/index.md#step-3-the-rlhf-recipe-in-three-stages))

**Reinforcement learning (RL)**: learning to act by trial and error, using rewards instead of correct answers, to maximise total future reward. ([What is RL?](topics/reinforcement-learning-intro/index.md))

**Reject Feedback** (Type II): when a rule matches an example of another class, every Forgotten literal that is False for that example moves one step towards memorized, always, so the rule stops matching it. ([Recognize, Erase & Reject feedback](topics/tsetlin-machine-feedback/index.md#step-7-frequent-is-not-enough-reject-feedback))

**Reliability diagram**: a plot of how often the positive class really occurs at each confidence level (here, each vote sum), used to check whether confidence can be read as a probability. ([Confidence & composites](topics/tsetlin-machine-confidence/index.md#step-5-an-honest-look-at-calibration))

**Resource allocation**: the effect of the Voting Margin that sends feedback mostly to examples not yet won by T votes, so idle rules move to sub-patterns no rule covers yet. ([Classification by voting](topics/tsetlin-machine-classification/index.md#step-4-the-complete-learning-algorithm))

**Return** ($G_t$): the (discounted) sum of all rewards from time step $t$ until the end of the episode; the quantity an RL agent tries to maximise. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-3-what-does-best-mean-the-return-and-discounting))

**Reward** ($r$): the number the environment sends back after each action, saying how good the immediate outcome was. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**Reward hacking (reward over-optimisation)**: when a policy exploits errors in a learned or proxy reward, so its score rises while true quality falls (Goodhart's law). ([RLHF & DPO](topics/rlhf-dpo/index.md#step-5-rl-fine-tuning-with-a-kl-leash))

**Reward model** (r_φ): A network trained on human preference pairs to output a scalar score for a reply, used as the reward during RL fine-tuning. ([RLHF & DPO](topics/rlhf-dpo/index.md#step-4-learning-a-reward-from-comparisons-bradleyterry))

**Rule (clause)** ($C_j$): "if condition then class", where the condition is an AND of literals; it matches an object only if every literal is True. ([What is a Tsetlin machine?](topics/tsetlin-machine-intro/index.md#step-4-rules-and-ing-literals-together))

**Rules for / against a class** (positive / negative polarity clauses): rules whose True condition adds +1 or −1 to the vote sum. ([Classification by voting](topics/tsetlin-machine-classification/index.md#step-2-rules-vote-the-vote-sum-v))

**SAC (Soft Actor-Critic)**: an off-policy actor-critic for continuous actions that uses a replay buffer, a maximum-entropy objective with an (auto-tuned) temperature, and twin Q critics. ([Actor-critic](topics/actor-critic/index.md#step-6-sac-off-policy-continuous-actions-maximum-entropy))

**Sample average**: estimating an action's value as the mean of the rewards it has given so far. ([Multi-armed bandits](topics/multi-armed-bandits/index.md#step-2-action-values-what-is-each-arm-worth))

**SARSA**: an on-policy TD control method that updates Q(s, a) towards r + γ Q(s′, a′) using the next action it really takes; named after the quintuple (s, a, r, s′, a′). ([Q-learning & SARSA](topics/q-learning-sarsa/index.md#step-2-sarsa-learn-from-the-move-you-actually-make-next))

**Self-play**: training an agent on games it plays against itself, so that it produces its own training data. ([Model-based RL & planning](topics/model-based-rl/index.md#step-6-alphago-alphazero-and-muzero-search-deep-learning))

**Semi-gradient**: an update that treats the bootstrapped target as a constant and does not differentiate through it. ([Deep Q-Networks](topics/deep-q-networks/index.md#step-3-turning-q-learning-into-regression))

**Softmax policy**: a policy that turns action preferences into probabilities with π(a|s) = e^{h(s,a)} / Σ_b e^{h(s,b)}. ([Policy gradients](topics/policy-gradients/index.md#step-2-a-policy-with-knobs))

**Specificity** (s): the papers' single knob, with Forget Value 1/s; a larger s gives longer, more specific rules. ([Recognize, Erase & Reject feedback](topics/tsetlin-machine-feedback/index.md#step-3-recognize-feedback-become-more-like-this-example))

**State** ($s$): a description of the situation the agent is in at a given time step. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-2-the-loop-agent-and-environment))

**Stationary distribution** (π): the long-run share of time a Markov chain spends in each state, unchanged by one more step (π = πM); it gives a literal's exact probability of being Memorized. ([The Tsetlin automaton](topics/tsetlin-automaton/index.md#step-5-the-exact-learning-outcome))

**Step size** (α): how far an incremental update moves the estimate towards the new target; 1/n gives the exact average, and a constant α tracks a changing world. ([Multi-armed bandits](topics/multi-armed-bandits/index.md#step-3-learning-incrementally-new-estimate-old-estimate-step-error))

**Stochastic gradient descent (SGD)**: gradient descent that estimates the gradient from one random example (or a mini-batch) per step. ([Gradient Descent](topics/gradient-descent/index.md#step-6-real-data-is-big-stochastic-gradient-descent-sgd))

**Sub-optimal configuration** (local optimum): a set of rules that no single rule can improve by changing alone, such as two copies of "x1 AND x2" for x1 OR x2. ([Classification by voting](topics/tsetlin-machine-classification/index.md#step-5-why-the-margin-matters-two-rules-one-or-pattern))

**Sub-pattern**: one kind of example inside a class, described by a single AND-rule; a class with several needs several rules. ([Classification by voting](topics/tsetlin-machine-classification/index.md#step-1-one-rule-is-not-enough))

**Supervised fine-tuning (SFT)**: fine-tuning a pre-trained language model on human-written demonstration answers with the ordinary next-token loss. ([RLHF & DPO](topics/rlhf-dpo/index.md#step-1-the-problem-a-text-predictor-is-not-an-assistant))

**Sweep**: one pass of updates over every state of an MDP. ([Dynamic programming](topics/dynamic-programming/index.md#step-1-turn-the-bellman-equation-into-an-update-rule-policy-evaluation))

**Target network** (θ⁻): A periodically synced, frozen copy of the Q-network used only to compute TD targets, so the regression labels hold still. ([Deep Q-Networks](topics/deep-q-networks/index.md#step-5-problem-2-chasing-a-moving-target-the-target-network))

**Target policy**: the policy whose values an agent is learning about, i.e. the one its TD target assumes will be followed from the next state on. ([Q-learning & SARSA](topics/q-learning-sarsa/index.md#step-6-on-policy-versus-off-policy))

**TD control**: using temporal-difference updates of action values Q(s, a) to learn a good policy directly from experience, without a model. ([Q-learning & SARSA](topics/q-learning-sarsa/index.md#step-1-from-v-to-q-without-a-model-learn-action-values))

**TD error** ($\delta_t$): the one-step surprise $r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$, which drives every TD update. ([Monte Carlo & TD learning](topics/monte-carlo-td-learning/index.md#step-3-td0-learn-a-guess-from-a-guess))

**TD target**: the value $r_{t+1} + \gamma V(s_{t+1})$ that TD(0) moves $V(s_t)$ toward. ([Monte Carlo & TD learning](topics/monte-carlo-td-learning/index.md#step-3-td0-learn-a-guess-from-a-guess))

**Temporal-difference (TD) learning** (TD(0)): learning values after every step by moving $V(s)$ toward $r + \gamma V(s')$; it samples like Monte Carlo and bootstraps like DP. ([Monte Carlo & TD learning](topics/monte-carlo-td-learning/index.md#step-3-td0-learn-a-guess-from-a-guess))

**Thermometer encoding**: booleanising a number with features "value ≥ $t_1$", "value ≥ $t_2$", …, so larger values switch on more bits. ([What is a Tsetlin machine?](topics/tsetlin-machine-intro/index.md#step-2-booleanising-real-data))

**Thompson sampling**: exploring by drawing one sample from your belief about each action's value and taking the action with the highest sample. ([Multi-armed bandits](topics/multi-armed-bandits/index.md#step-6-upper-confidence-bounds-ucb-explore-where-you-are-unsure))

**Tipping point** (p* = 1/(s+1)): how often a literal must be True in its class to be Memorized more often than not, when only that class is seen; set by the Forget Value 1/s. ([The Tsetlin automaton](topics/tsetlin-automaton/index.md#step-5-the-exact-learning-outcome))

**TM composite**: several independently trained Tsetlin machines combined plug-and-play by adding their normalised vote sums (research term: TMComposites). ([Confidence & composites](topics/tsetlin-machine-confidence/index.md#step-6-different-machines-different-skills))

**Trustworthiness (of confidence)**: confidence that matches performance: the more confident cases are more often correct, as measured on held-out data. ([Confidence & composites](topics/tsetlin-machine-confidence/index.md#step-2-trustworthiness-confidence-must-match-performance))

**Tsetlin automaton**: M. L. Tsetlin's learning automaton whose whole memory is one integer among 2N states: a reward moves it deeper into its current action, a penalty toward the centre, where it switches action. ([The Tsetlin automaton](topics/tsetlin-automaton/index.md#step-1-tsetlins-idea-a-memory-that-is-a-single-integer))

**Tsetlin machine**: an interpretable machine-learning model in which many AND-rules over Boolean literals vote for or against a class, each rule learning its literals with Tsetlin-automaton memory. ([What is a Tsetlin machine?](topics/tsetlin-machine-intro/index.md#what-is-a-tsetlin-machine))

**Twin Q critics**: two Q-networks whose minimum is used in the target, to counter the overestimation of learned Q-values (from TD3, used in SAC). ([Actor-critic](topics/actor-critic/index.md#step-6-sac-off-policy-continuous-actions-maximum-entropy))

**UCT** (Upper Confidence bounds applied to Trees): MCTS's selection rule $W/N + c\sqrt{\ln N_{\text{parent}}/N}$, the UCB bandit rule applied at every node of the tree. ([Model-based RL & planning](topics/model-based-rl/index.md#step-5-planning-at-decision-time-monte-carlo-tree-search))

**Upper Confidence Bound (UCB)**: choosing $\arg\max_a [Q_t(a) + c\sqrt{\ln t / N_t(a)}]$, i.e. adding a bonus to actions that have been tried rarely ("optimism in the face of uncertainty"). ([Multi-armed bandits](topics/multi-armed-bandits/index.md#step-6-upper-confidence-bounds-ucb-explore-where-you-are-unsure))

**Value function** ($V^\pi(s)$): the expected return when starting in state $s$ and following policy $\pi$. ([What is RL?](topics/reinforcement-learning-intro/index.md#step-5-how-good-is-a-state-the-value-function))

**Value iteration**: repeatedly applying the Bellman optimality equation as an update, V(s) ← max_a Σ p·[r + γV(s′)], then acting greedily on the result. ([Dynamic programming](topics/dynamic-programming/index.md#step-4-value-iteration-one-sweep-then-improve))

**Vote sum** (v): votes from firing rules for a class minus votes from firing rules against it; its sign is the prediction. ([Classification by voting](topics/tsetlin-machine-classification/index.md#step-2-rules-vote-the-vote-sum-v))

**Voting Margin** (T; threshold): during training, each rule gets feedback with probability (T − v)/2T, where v is measured for the example's own class and clipped to [−T, T]; examples already won by T votes give no feedback. ([Classification by voting](topics/tsetlin-machine-classification/index.md#step-3-the-voting-margin-t-how-much-feedback-does-an-example-get))
