import unittest
from autograd.tensor import Tensor, mul

class TestTensorMul(unittest.TestCase):

    def test_simple_mul(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor([4, 5, 6], requires_grad=True)

        t3 = mul(t1, t2)
        t3.backward(Tensor([-1, -2, -3]))

        assert t1.grad.data.tolist() == [-4, -10, -18]
        assert t2.grad.data.tolist() == [-1, -4, -9]

    def test_broadcast_mul(self):
        # If t1.shape == t2.shape, then fine
        # say that t1.shape = (10, 5), t2.shape == (5, )
        #   t1 + t2, t2 gets viewed as (1, 5)
        #   then t2 is copied along the rows

        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True) # shape = (2, 3)
        t2 = Tensor([7, 8, 9], requires_grad=True)              # shape = (3,)

        t3 = mul(t1, t2)                                        # shape (2, 3)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
        assert t2.grad.data.tolist() == [5, 7, 9]
    
    def test_broadcast_mul2(self):
        t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True) # shape = (2, 3)
        t2 = Tensor([[7, 8, 9]], requires_grad=True)            # shape = (1, 3)

        t3 = mul(t1, t2)                                        # shape (2, 3)
        t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))

        assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
        assert t2.grad.data.tolist() == [[5, 7, 9]]
