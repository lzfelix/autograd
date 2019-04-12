import numpy as np
from autograd import Tensor, Parameter, Module

class LinearModel(Module):

    def __init__(self) -> None:
        self.w =  Parameter(3)
        self.b = Parameter()
    
    def predict(self, inputs: Tensor) -> Tensor:
        """Learned function: y = Ax + b"""
        return inputs @ self.w + self.b


if __name__ == '__main__':
    x_data = Tensor(np.random.randn(100, 3))
    coef = Tensor(np.array([-1, 3, -2]))

    # The function to be learned y = Ax + b + eps
    # x_data is a tensor, so is y_data
    # y_data = x_data @ coef + 5 + np.random.randint(-2, 2, size=(100,))

    # With a perfect linear regression, we can get 0 error
    y_data = x_data @ coef + 5

    learning_rate = 1e-3
    batch_size = 32
    model = LinearModel()

    for epoch in range(100):
        epoch_loss = 0.0

        for start in range(0, 100, batch_size):
            end = start + batch_size
            model.zero_grad()
            
            inputs = x_data[start:end]
            actuals = y_data[start:end]

            predicted = model.predict(inputs)
            errors = predicted - actuals
            loss = (errors * errors).sum()
            
            loss.backward()
            epoch_loss += loss.data

            model.w -= learning_rate * model.w.grad
            model.b -= learning_rate * model.b.grad

        print(f'{epoch} - {epoch_loss}')
