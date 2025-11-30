import numpy as np

from my_torch.linear_layer import Linear

def test_instantiation_no_bias():
    my_layer = Linear(3, 4, False)

    assert my_layer.weights.shape == (4, 3)
    assert my_layer.bias is None

def test_instantiation_bias():
    my_layer = Linear(3, 4, True)
    assert my_layer.weights.shape == (4, 3)
    assert my_layer.bias.shape == (4,)

def test_reset_params_resets():
    my_layer = Linear(3, 4, True)
    old_weights = np.copy(my_layer.weights)
    old_bias = np.copy(my_layer.bias)
    assert np.array_equal(old_weights, my_layer.weights)
    assert np.array_equal(old_bias, my_layer.bias)
    my_layer.reset_params(create_bias=True)
    assert not np.array_equal(old_weights, my_layer.weights)
    assert not np.array_equal(old_bias, my_layer.bias)

