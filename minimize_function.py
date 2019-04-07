""" Minimizes f(x) = x ^ 2 """

from autograd.tensor import Tensor, tensor_sum, mul

x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)

# Very primitive SGD to minimize a function
for i in range(100):
    sum_of_squares = tensor_sum(mul(x, x))                      # is a 0-tensor
    sum_of_squares.backward()

    # ugly because we haven't implemented the -= operation (yet?)
    delta_x = mul(Tensor(0.1), x.grad)
    x = Tensor(x.data - delta_x.data, requires_grad=True)

    print(i, sum_of_squares)
