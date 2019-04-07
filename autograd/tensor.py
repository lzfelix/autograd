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

        # Dependency between output and input to compute the backprop
        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data,
                  requires_grad,
                  depends_on)

def add(t1: Tensor, t2: Tensor) -> Tensor:
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # d/dx (x+y) = 1
            # This means we just pass the gradient back bc the chain rule

            # Sum out addded dims
            ndims_added = grad.ndim - t1.data.ndim

            # if ndims_added > 0 -> broadcasted by adding dimensions at the
            # beginning of the tensor. Need to sum these gradients, simply
            # because broadcast repeats data along some dimension/s.
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted, but non-added dims
            # (1, 3) + (2, 3) -> (2, 3) and grad (2, 3), but we need (1, 3),
            # since the gradient is wrt x, which is (1, 3)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        depends_on.append(Dependency(t1, grad_fn1))
    
    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # d/dy (x+y) = 1

            # Handling numpy broadcasting properly
            ndims_added = grad.ndim - t2.data.ndim

            # if ndims_added > 0 -> broadcasted by adding dimensions at the
            # beginning of the tensor. Need to sum these gradients
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            # Sum across broadcasted, but non-added dims
            # (2, 3) + (1, 3) -> (2, 3) and grad (2, 3), but we need (1, 3), bc
            # now the grad is wrt y, which is (1, 3).
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad
        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)