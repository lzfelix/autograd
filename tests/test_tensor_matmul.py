import unittest
import numpy as np
from autograd.tensor import Tensor

class TestTensorMatMul(unittest.TestCase):

    def test_simple_matmul(self):
        t1 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True) # (3, 2)
        t2 = Tensor([[10], [20]], requires_grad=True)             # (2, 1)

        t3 = t1 @ t2                                              # (3, 1)
        assert t3.data.tolist() == [[50], [110], [170]]

        grad = Tensor([[1], [2], [3]])
        t3.backward(grad)

        # Just copying the formula from the code in tensor.py ;D
        np.testing.assert_allclose(t1.grad.data, grad.data @ t2.data.T)
        np.testing.assert_allclose(t2.grad.data, t1.data.T @ grad.data)
