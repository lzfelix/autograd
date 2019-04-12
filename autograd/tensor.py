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

Tensorable = Union['Tensor', float, np.ndarray]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    def __init__(self,
                 data: np.ndarray,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None) -> None:
        # Transforms data into a numpy array
        self._data = ensure_array(data)

        # Dependencies for backpropagation
        self.depends_on = depends_on or []

        self.requires_grad = requires_grad
        self.shape = self.data.shape
        self.grad: Optional[Tensor] = None

        # If requires grad. Initially set it to all zeros
        if self.requires_grad:
            self.zero_grad()
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        # Setting the data manually means that grad must be invalidated
        self.grad = None
    
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other) -> 'Tensor':
        """Gets called when t + other"""
        return _add(self, ensure_tensor(other))
    
    def __radd__(self, other) -> 'Tensor': 
        """Gets called when other + t"""
        return _add(ensure_tensor(other), self)
    
    def __iadd__(self, other) -> 'Tensor':
        """Called when we do t += other (inplace) add"""
        # c = f(a) -> dc/da = dc/df * df/da
        self.data = self.data + ensure_tensor(other).data
        
        # Invalidate previous gradients, since now we have a new component that
        # depends on "self", the derivative of this new component must be taken
        # into account when computing the gradient of "self".
        # self.grad = None

        # Now done with properties
        return self
    
    def __mul__(self, other) -> "Tensor":
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> "Tensor":
        return _mul(ensure_tensor(other), self)
    
    def __matmul__(self, other) -> "Tensor":
        return _matmult(self, ensure_tensor(other))

    def __imul__(self, other) -> 'Tensor':
        self.data = self.data * ensure_tensor(other).data
        return self
    
    def __neg__(self) -> 'Tensor':
        return _neg(self)

    def __sub__(self, other) -> 'Tensor':
        return _sub(self, ensure_tensor(other))

    def __rsub__(self, other) -> 'Tensor':
        return _sub(ensure_tensor(other), self)

    def __isub__(self, other) -> 'Tensor':
        self.data = self.data - ensure_tensor(other).data
        return self

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
        self.grad.data = self.grad.data + grad.data # type: ignore

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

def _add(t1: Tensor, t2: Tensor) -> Tensor:
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

def _mul(t1: Tensor, t2: Tensor) -> Tensor:
    """
    y = a * b
    have dL/dy

    DL/da = dL/dy * dy/da = dL/dy * b
    """
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # d/dx (x+y) = y
            # This means we just pass back the gradient scaled by due chain rule
            grad = grad * t2.data

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
            # d/dy (x*y) = y
            grad = grad * t1.data

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

def _neg(t: Tensor) -> Tensor:
    data = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        depends_on = [Dependency(t, lambda x: -x)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)

def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    """Computationally it's a little bit unneficient, since two nodes are added
    to the computational graph: one for neg and another for add, instead of a
    single 'pure' subtraction node that performs all the operations within
    itself."""
    return _add(t1, _neg(t2))

def _matmult(t1: Tensor, t2: Tensor) -> Tensor:
    """
    if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then:
        grad1 = grad3 @ t2.T
        grad2 = t1.T @ grad3
    if t1 is (m1, m2) and t2 in (m2, m3), then t3 is (m1, m3), consequently
    grad3 is (m1, m3)
    """
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # There's no broadcasting happening now! :D
            return grad @ t2.data.T

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data,
                  requires_grad,
                  depends_on)