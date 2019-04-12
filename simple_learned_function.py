import numpy as np
from autograd.tensor import Tensor

if __name__ == '__main__':
    x_data = Tensor(np.random.randn(100, 3))
    coef = Tensor(np.array([-1, 3, -2]))

    # The function to be learned y=Ax + b + eps
    # x_data is a tensor, so is y_data
    # y_data = x_data @ coef + 5 + np.random.randint(-2, 2, size=(100,))

    # With a perfect linear regression, we can get 0 error
    y_data = x_data @ coef + 5

    b = Tensor(np.random.randn(1), requires_grad=True)
    w = Tensor(np.random.randn(3), requires_grad=True)

    learning_rate = 1e-3
    batch_size = 32

    for epoch in range(100):
        epoch_loss = 0.0

        for start in range(0, 100, batch_size):
            end = start + batch_size
            
            inputs = x_data[start:end]
            actuals = y_data[start:end]

            w.zero_grad()
            b.zero_grad()

            predicted = inputs @ w + b
            errors = predicted - actuals
            loss = (errors * errors).sum()
            
            loss.backward()
            epoch_loss += loss.data

            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        print(f'{epoch} - {epoch_loss}')
