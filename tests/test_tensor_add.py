import unittest
import numpy as np
from autograd.tensor import Tensor

class TestTensorAdd(unittest.TestCase):

    def test_simple_add(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = t1 + t2
        t3.backward(Tensor([-1, -2, -3]))

        assert t3.data.tolist() == [5, 7, 9]
        assert t1.grad.data.tolist() == [-1, -2, -3]
        assert t2.grad.data.tolist() == [-1, -2, -3]

    def test_inplace_add(self):
        t = Tensor([1, 2, 3], requires_grad=True)
        t += 0.1

        assert t.grad is None
        np.testing.assert_allclose(t.data, [1.1, 2.1, 3.1])

    def test_broadcast_add(self):
        # If t1.shape == t2.shape, then fine
        # say that t1.shape = (10, 5), t2.shape == (5, )
        #   t1 + t2, t2 gets viewed as (1, 5)
        #   then t2 is copied along the rows

        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True) # shape = (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad=True)              # shape = (3,)

        t3 = t1 + t2                                            # shape (2, 3)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t3.data.tolist() == [[8, 10, 12], [11, 13, 15]]
        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [2, 2, 2]
    
    def test_broadcast_add2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True) # shape = (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)            # shape = (1, 3)

        t3 = t1 + t2                                            # shape (2, 3)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t3.data.tolist() == [[8, 10, 12], [11, 13, 15]]
        assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert t2.grad.data.tolist() == [[2, 2, 2]]