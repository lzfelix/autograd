from typing import List, Callable, NamedTuple, Optional, Union

import numpy as np

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    """Transforms float, list and and ndarrays into np.ndarray"""
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Tensor:
    def __init__(self,
                 data: np.ndarray,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        # Transforms data into a numpy array
        self.data = ensure_array(data)

        # Dependencies for backpropagation
        self.depends_on = depends_on or []

        self.requires_grad = requires_grad
        self.shape = self.data.shape
        self.grad: Optional[Tensor] = None

        # If requires grad. Initially set it to all zeros
        if self.requires_grad:
            self.zero_grad()
    
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad: 'Tensor' = None) -> None:
        """Computes the backward pass. If grad is not specified, it is assumed 1
        for 0-tensors."""

        assert self.requires_grad, "Called backward on tensor that does not requires grad"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError('grad must be specified for non-0-tensor')

        # Forward pass
        # [Node_1] ---[x]---[f(x)]-+--> [Node_2]
        #                           \--> [Node_3]
        # Backward pass
        # [Node_1] <---[f'(z_2 + z_3)] <-+-[z_2]---- [Node_2]
        #                                \-[z_3]---- [Node_3]   

        # Sum all the backwards input to this node
        self.grad.data += grad.data

        for dependency in self.depends_on:
            # Then applies the derivative to the function that establishes the
            # dependency and send the result back to the previous operation
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))

    def sum(self) -> 'Tensor':
        return tensor_sum(self)


def tensor_sum(t: Tensor) -> Tensor:
    """Creates a new 0-tensor that is the sum of all elements in t."""

    data = t.data.sum()
    requires_grad = t.requires_grad
    
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """Each element contributes in 1 (linear) to the output"""
            return grad * np.ones_like(t.data)

        # Creates the dependency, for when computing the backpropagation
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)