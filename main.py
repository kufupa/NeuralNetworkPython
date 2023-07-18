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
    return np.maximum(0, x)


def quadratic(x):
    return x ** 4 + 2 * x ** 3 - 2 * x - 10


# Generates around 20,000 pieces of data, in interval ~ [-10, 20]
def getRandomQuadraticData():
    data = []
    for i in range(-100000, 100000, 1):
        i = i / 50000
        if random.random() > 0.9:
            data.append(np.array([i, quadratic(i)]))
    print(f"Data size: {len(data)}")
    return np.array(data)


def initialiseLayers():
    # 10 nodes, 1 input, so 10 rows w 1 column each for matrix mul w input
    W1 = np.random.randn(10, 2)
    # 10 nodes, 1 bias each obvs
    b1 = np.random.randn(10, 1)

    # 1 output, which is number
    W2 = np.random.randn(10, 1)
    b2 = np.random.randn(10, 1)

    return W1, b1, W2, b2


if __name__ == "__main__":
    data = getRandomQuadraticData()
    # For quadratic data, x_val = data[i][0], y_val = data[i][0]
    np.random.shuffle(data)
    print(data[:5].T)
    dataTest = data[0:500].T
    dataTest = data[500:].T
