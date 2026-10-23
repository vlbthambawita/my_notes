# Glossary

Every term used on this site, in one sentence. Terms link to the lesson where they're explained.

**Batch gradient descent**: gradient descent where each step uses the gradient computed from *all* training examples. ([Gradient Descent](topics/gradient-descent/index.md#step-6-real-data-is-big-stochastic-gradient-descent-sgd))

**Convex function**: a bowl-shaped function with a single minimum; gradient descent with a suitable learning rate is guaranteed to find it.

**Divergence**: when the learning rate is too large and each update overshoots further, so the loss grows instead of shrinking. ([Gradient Descent](topics/gradient-descent/index.md#step-4-taking-the-step-the-update-rule))

**Epoch**: one full pass over the entire training dataset.

**Gradient** ($\nabla L$): the list of slopes of the loss with respect to each parameter; it points in the direction of steepest increase. ([Gradient Descent](topics/gradient-descent/index.md#step-3-which-way-is-downhill-the-gradient))

**Gradient descent**: an algorithm that repeatedly moves parameters a small step opposite to the gradient, to reduce the loss. ([Gradient Descent](topics/gradient-descent/index.md))

**Learning rate** ($\eta$): the number that controls how big each gradient-descent step is. ([Gradient Descent](topics/gradient-descent/index.md#step-4-taking-the-step-the-update-rule))

**Loss (loss function)**: a single number that measures how wrong a model's predictions are; training tries to make it small. ([Gradient Descent](topics/gradient-descent/index.md#step-2-what-are-we-trying-to-make-small-the-loss))

**Mean squared error (MSE)**: a loss that averages the squared differences between predictions and true values.

**Mini-batch**: a small random subset of the training data used to estimate the gradient for one step.

**Parameters (weights)**: the adjustable numbers inside a model that training changes.

**Stochastic gradient descent (SGD)**: gradient descent that estimates the gradient from one random example (or a mini-batch) per step. ([Gradient Descent](topics/gradient-descent/index.md#step-6-real-data-is-big-stochastic-gradient-descent-sgd))
