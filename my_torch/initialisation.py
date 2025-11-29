import math

from my_torch.constants import NonLinearityType


def calculate_gain(non_linearity: NonLinearityType, a: float | None) -> float:
    match non_linearity:
        case NonLinearityType.LINEAR:
            return 1.0
        case NonLinearityType.SIGMOID:
            return 1.0
        case NonLinearityType.TANH:
            return 5.0/3.0
        case NonLinearityType.RELU:
            return math.sqrt(2.0)
        case NonLinearityType.LEAKY_RELU:
            assert a is not None, "Expect `a` to be defined when non-linearity is leaky-relu."
            return math.sqrt(2.0 / (1.0 + a**2))