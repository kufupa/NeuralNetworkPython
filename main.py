import numpy as np

# import pandas as pd  # used for reading data
import matplotlib.pyplot as plt
import random

# Coding a neural network library from scratch
# Class for general layers
# Specialised input, hidden, output layers
# Methods for training, backpropogation
# Initial support for relu activation, softmax output, cross entropy cost

# softmax takes output layer and converts to probabilities from 0-1

# learning rate is a hyperparameter, default to 0.05


def relu(x):
    return max(0, x)


def quadratic(x):
    return x ** 4 + 2 * x ** 3 - 2 * x - 10


# Generates around 2000 pieces of data, in interval ~ [-10, 20]
def getRandomQuadraticData():
    data = []
    for i in range(-100000, 100000, 1):
        i = i / 50000
        if random.random() > 0.9:
            data.append(np.array([i, quadratic(i)]))
    print(f"Data size: {len(data)}")
    return np.array(data)


if __name__ == "__main__":
    data = getRandomQuadraticData()
    np.random.shuffle(data)
    print(data)