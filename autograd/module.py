import inspect
from typing import Iterator

from autograd.tensor import Tensor
from autograd.parameter import Parameter

class Module:
    """A colection of parameters that have a forward method (see pyTorch)"""

    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                # now we can have nested models
                yield value.parameters()

    def zero_grad(self) -> None:
        """Calls zero_grad() in all parameters of this module."""
        for parameter in self.parameters():
            parameter.zero_grad()
