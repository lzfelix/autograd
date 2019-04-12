import numpy as np

from autograd.tensor import Tensor

class Parameter(Tensor):
    """A Parameter is an automatically initialized Tensor with gradients."""

    def __init__(self, *shape) -> None:
        # TODO: Allow the user to set how to initialize these parameters
        data = np.random.randn(*shape)
        super().__init__(data, requires_grad=True)
