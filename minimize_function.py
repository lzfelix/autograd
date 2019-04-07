""" Minimizes f(x) = sum(x ** 2) """

from autograd.tensor import Tensor, tensor_sum

if __name__ == '__main__':
    x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)

    # Very primitive SGD to minimize a function
    for i in range(100):
        # Since we are not creating a new tensor at each iteration anymore, we
        # need to manually zero the gradients, otherwise they are going to
        # accumulate with gradients from previous operations
        x.zero_grad()

        sum_of_squares = (x * x).sum()                      # is a 0-tensor
        sum_of_squares.backward()

        # Under the hood, the following line performs the operation below:
        #   mul(Tensor(0.1), x.grad)
        delta_x = 0.1 * x.grad

        # To see what really happens here, check the code for __isub__ 
        x -= delta_x

        print(i, sum_of_squares)
