import numpy as np

from src.neural_net import MLP

def _numerical_gradient(net, param, X, y, l2, eps=1e-6):
    grad = np.zeros_like(param)
    for index in np.ndindex(param.shape):
        original = param[index]
        param[index] = original + eps
        up = net.loss(X, y, l2)
        param[index] = original - eps
        down = net.loss(X, y, l2)
        param[index] = original
        grad[index] = (up - down) / (2 * eps)
    return grad

def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(12, 5)), rng.normal(size=12)
    net = MLP([5, 7, 4, 1], seed=3)
    analytic = net.gradients(X, y, l2=0.01)
    for param, grad in zip(net._params(), analytic):
        numeric = _numerical_gradient(net, param, X, y, l2=0.01)
        np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-7)

def test_forward_shape_and_he_scale():
    net = MLP([100, 50, 1], seed=0)
    assert net.forward(np.zeros((8, 100))).shape == (8,)
    assert abs(net.weights[0].std() - np.sqrt(2 / 100)) < 0.02

def test_adam_learns_a_simple_function():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] - 2 * X[:, 1]
    net = MLP([3, 8, 1], seed=1)
    before = net.loss(X, y)
    for _ in range(300):
        net.step(net.gradients(X, y), lr=1e-2)
    assert net.loss(X, y) < 0.1 * before

def test_get_and_set_weights_round_trip():
    net = MLP([3, 4, 1], seed=0)
    saved = net.get_weights()
    X = np.ones((2, 3))
    before = net.forward(X).copy()
    net.step(net.gradients(X, np.zeros(2)), lr=0.1)
    assert not np.allclose(net.forward(X), before)
    net.set_weights(saved)
    np.testing.assert_allclose(net.forward(X), before)
