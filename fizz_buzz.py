from typing import List

import numpy as np
from autograd import Tensor, Parameter, Module
from autograd.function import tanh
from autograd.optim import SGD

def binary_encode(x: int) -> List[int]:
    """Transforms a decimal into binary number"""
    return [x >> i & 1 for i in range(10)]

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


class FizzBuzzModel(Module):

    def __init__(self, num_hidden: int = 50) -> None:
        # TODO: Introduce linear modules (aka linear layers)
        self.w1 = Parameter(10, num_hidden)
        self.b1 = Parameter(num_hidden)

        self.w2 = Parameter(num_hidden, 4)
        self.b2 = Parameter(4)
    
    def predict(self, inputs: Tensor) -> Tensor:
        # input is (batch_size, 10)
        x1 = inputs @ self.w1 + self.b1     # (batch_size, num_hidden)
        x2 = tanh(x1)                       # (batch_size, num_hidden)
        x3 = x2 @ self.w2 + self.b2         # (batch_size, 4)

        return x3


if __name__ == '__main__':
    x_train = Tensor([binary_encode(x) for x in range(101, 1024)])    # (922, 10) tensor
    y_train = Tensor([fizz_buzz_encode(x) for x in range(101, 1024)]) # (922, 4) tensor

    optimizer = SGD(lr=1e-3)
    batch_size = 32
    model = FizzBuzzModel()

    # Getting a different order of batches at each epoch
    starts = np.arange(0, x_train.shape[0], batch_size)

    for epoch in range(5000):
        epoch_loss = 0.0

        for start in starts:
            end = start + batch_size

            model.zero_grad()
            
            inputs = x_train[start:end]
            actuals = y_train[start:end]

            predicted = model.predict(inputs)
            errors = predicted - actuals
            loss = (errors * errors).sum()
            
            loss.backward()
            epoch_loss += loss.data

            optimizer.step(model)

        print(f'{epoch} - {epoch_loss}')

    num_correct = 0
    for x in range(1, 101):
        inputs = Tensor([binary_encode(x)])         # (1, 10)
        predicted = model.predict(inputs)           # (1, 4)

        predicted_idx = np.argmax(predicted[0].data)
        actual_idx = np.argmax(fizz_buzz_encode(x))

        labels = [str(x), "fizz", "buzz", "fizzbuzz"]
        if predicted_idx == actual_idx:
            num_correct += 1
        
        print(x, labels[predicted_idx], labels[actual_idx])
    print(num_correct, " / 100")