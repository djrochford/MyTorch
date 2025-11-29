from enum import Enum

class LayerOperation(Enum):
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"

class NonLinearityType(Enum):
    LINEAR = "LINEAR"
    SIGMOID = "SIGMOID"
    TANH = "TANH"
    RELU = "RELU"
    LEAKY_RELU = "LEAKY_RELU"
