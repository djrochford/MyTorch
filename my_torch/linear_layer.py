import math
import numpy as np

from my_torch.constants import LayerOperation, NonLinearityType
from my_torch.initialisation import calculate_gain

class _Layer:
    pass


class Linear(_Layer):


    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):      
        self._in_dim = in_dim
        self._out_dim = out_dim
        
        self.weights = None
        self.bias = None
        self.reset_params(bias)
        
        self._last_input = None
        self._last_operation = None
        
    @property
    def in_dim(self):
        return self._in_dim
    
    @property
    def out_dim(self):
        return self._out_dim

    def reset_params(self, create_bias: bool) -> None:
        """Reset (or initialise) weights, and possibly bias, of layer.
        
        Uses method described in 'Delving deep into rectifiers: Surpassing
        human-level performance on ImageNet classification' - He, K. et al. (2015).

        In the absence of information about the next layer, we (following PyTorch) choose
        a default which works fine in most cases (Leaky ReLU with a = sqrt(5)).

        In any case, normalisation layers make careful fiddling with initialisation otiose.
        """
        gain = calculate_gain(NonLinearityType.LEAKY_RELU, math.sqrt(5))
        std = gain / math.sqrt(self.in_dim)
        bound = math.sqrt(3.0) * std
        self.weights = np.random.uniform(-bound, bound, (self.out_dim, self.in_dim))
        if create_bias:
           self.bias = np.random.uniform(-bound, bound, (self.out_dim,))
        else:
            self.bias = None
    
    def forward(self, input: np.ndarray):
        self._last_input = input
        output = input @ self.weights.T
        if self.bias is not None:
            output += self.bias
        self._last_operation = LayerOperation.FORWARD
        return output