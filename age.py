import numpy as np

# Generating random ages between 0 and 90
train_in = np.random.randint(0, 90, (100000, 1))
train_out = train_in.flatten()

test_in = np.random.randint(0, 90, (10000, 1))
test_out = test_in.flatten()

# Scaling the age data to improve training
train_in = train_in / 90.0
test_in = test_in / 90.0

# Initialize the model parameters
weights = np.random.randn(1)
bias = np.random.randn()

# Hyperparameters
learning_rate = 0.1
epochs = 1000

# Training the model
for _ in range(epochs):
    predictions = np.dot(train_in, weights) + bias
    error = predictions - train_out  # A2 - Y
    weights -= learning_rate * np.dot(train_in.T, error) / len(train_out)
    bias -= learning_rate * np.sum(error) / len(train_out)

# Testing the model
test_predictions = np.dot(test_in, weights) + bias
test_mae = np.mean(np.abs(test_predictions - test_out))
print("Mean Absolute Error (MAE) on Test Data:", test_mae)
print(10 / 90 * weights + bias)
print(weights, bias)
