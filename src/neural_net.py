"""STEP 2 of the neural model: a tiny multi-layer perceptron in pure NumPy.

The network maps a feature vector to one number. Hidden layers use ReLU,
the output layer is linear (a rating offset can be any real number).
"""
from __future__ import annotations

import numpy as np


class MLP:
    """Fully connected network trained with MSE + L2 loss and the Adam optimiser.

    Args:
        layer_sizes: e.g. ``[n_in, 32, 16, 1]``: input, hidden layers, output.
        seed: Seed for the random initial weights.
    """

    def __init__(self, layer_sizes: list[int], seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.weights = [self._he_init(rng, n_in, n_out) for n_in, n_out in zip(layer_sizes, layer_sizes[1:])]
        self.biases = [np.zeros(n_out) for n_out in layer_sizes[1:]]
        # Adam keeps a running mean (m) and running mean of squares (v) of each gradient.
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._steps = 0

    @staticmethod
    def _he_init(rng: np.random.Generator, n_in: int, n_out: int) -> np.ndarray:
        """He initialisation: variance 2 / n_in keeps ReLU activations from shrinking or exploding."""
        return rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))

    def _params(self) -> list[np.ndarray]:
        """All weight matrices followed by all bias vectors (the same array objects, not copies)."""
        return self.weights + self.biases

    # ---- forward pass --------------------------------------------------------

    def _forward_with_cache(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Forward pass that also returns what backprop needs.

        Returns:
            ``(output, inputs, pre_activations)``. ``inputs[l]`` is the input of
            layer ``l`` and ``pre_activations[l]`` its output before ReLU.
        """
        inputs, pre_activations = [X], []
        out = X
        for layer, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = out @ W + b
            pre_activations.append(z)
            is_last = layer == len(self.weights) - 1
            out = z if is_last else np.maximum(z, 0.0)  # ReLU on hidden layers only
            inputs.append(out)
        return out[:, 0], inputs[:-1], pre_activations

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Network output for each row of ``X``, shape ``(n,)``."""
        return self._forward_with_cache(X)[0]

    def loss(self, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> float:
        """Mean squared error plus ``l2 * sum(weights ** 2)`` (biases are not penalised)."""
        error = self.forward(X) - y
        return float(np.mean(error ** 2) + l2 * sum(np.sum(W ** 2) for W in self.weights))

    # ---- backward pass -------------------------------------------------------

    def gradients(self, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> list[np.ndarray]:
        """Backpropagation: gradient of :meth:`loss` for every parameter.

        Returns:
            Gradients in the same order as :meth:`_params` (weights, then biases).
        """
        out, inputs, pre_activations = self._forward_with_cache(X)
        # d(mean squared error) / d(output); shape (n, 1) like a layer output.
        delta = (2.0 / len(y)) * (out - y)[:, None]
        grad_w, grad_b = [None] * len(self.weights), [None] * len(self.biases)
        for layer in reversed(range(len(self.weights))):
            grad_w[layer] = inputs[layer].T @ delta + 2.0 * l2 * self.weights[layer]
            grad_b[layer] = delta.sum(axis=0)
            if layer > 0:
                # Send the error to the previous layer; ReLU lets it through only where z > 0.
                delta = (delta @ self.weights[layer].T) * (pre_activations[layer - 1] > 0)
        return grad_w + grad_b

    # ---- optimiser -----------------------------------------------------------

    def step(
        self, grads: list[np.ndarray], lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
    ) -> None:
        """One Adam update: each parameter moves along its smoothed gradient, scaled by its own recent size."""
        self._steps += 1
        for param, grad, m, v in zip(self._params(), grads, self._m, self._v):
            m *= beta1
            m += (1 - beta1) * grad
            v *= beta2
            v += (1 - beta2) * grad ** 2
            # m and v start at 0, so early averages are too small; correct that bias.
            m_hat = m / (1 - beta1 ** self._steps)
            v_hat = v / (1 - beta2 ** self._steps)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)  # in place: updates the real weights

    # ---- saving the best epoch ------------------------------------------------

    def get_weights(self) -> list[np.ndarray]:
        """A copy of all parameters (used to remember the best epoch)."""
        return [p.copy() for p in self._params()]

    def set_weights(self, saved: list[np.ndarray]) -> None:
        """Restore parameters saved by :meth:`get_weights`."""
        for param, value in zip(self._params(), saved):
            param[...] = value
