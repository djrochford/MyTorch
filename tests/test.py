from my_torch.linear_layer import Linear

def test_instantiation_no_bias():
    my_layer = Linear(3, 4, False)

    assert my_layer.weights.shape == (4, 3)
    assert my_layer.bias is None

def test_instantiation_bias():
    my_layer = Linear(3, 4, True)
    assert my_layer.weights.shape == (4, 3)
    assert my_layer.bias.shape == (4,)

def test_trivial_initialisation():
    pass
    