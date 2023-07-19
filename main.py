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
    # np.maximum can work on a whole numpy array
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
    W1 = np.random.randn(10, 1)
    # 10 nodes, 1 bias each obvs
    b1 = np.random.randn(10, 1)

    # 1 output, which is number
    W2 = np.random.randn(1, 10)
    b2 = np.random.randn(1, 1)

    return W1, b1, W2, b2


def forwardProp(X, W1, b1, W2, b2):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = Z2
    return Z1, A1, Z2, A2

    dA1 = dZ2 * g(dZ1)


def reluG(x):
    return x > 0


def backwardsProp(X, Y, Z1, A1, Z2, A2, W2):
    m = Y.size
    dA2 = A2 - Y  # Since linear output
    dZ2 = dA2
    dW2 = 1 / m * dZ2.dot(A1.T)
    # db2 = 1 / m * dZ2
    db2 = 1 / m * np.sum(dZ2, 2)

    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * reluG(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, 2)

    return dW1, db1, dW2, db2


def updateParams(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def getPredictions(A2):
    return np.argmax(A2, 0)


def getAccuracy(predictions, Y):
    return np.sum(np.abs(predictions - Y) < 0.1) / Y.size


def gradientDescent(x, y, iterations, alpha):
    W1, b1, W2, b2 = initialiseLayers()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(x, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backwardsProp(x, y, Z1, A1, Z2, A2, W2)
        W1, b1, W2, b2 = updateParams(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)
        if i % 50 == 0:
            print(f"Iteration {i}")
            print(f"Accuracy :", getAccuracy(getPredictions(A2), Y))
    return W1, b1, W2, b2


if __name__ == "__main__":
    data = getRandomQuadraticData()
    # For quadratic data, x_val = data[i][0], y_val = data[i][0]
    np.random.shuffle(data)
    print(data[:5].T)
    dataTest = data[0:500]
    dataTrain = data[500:].T
    W1, b1, W2, b2 = gradientDescent(dataTrain[0], dataTrain[1], 200, 0.05)