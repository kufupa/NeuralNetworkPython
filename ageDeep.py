import numpy as np


def relu(x):
    return np.maximum(0, x)


# Generating random ages between 0 and 90
train_in = np.random.randint(0, 90, (100000, 1))
train_out = train_in.flatten()

# Scaling the age data to improve training
train_in = train_in / 90.0

# Initialize the model parameters
input_size, hidden_size1, hidden_size2, output_size = 1, 10, 10, 1
weights1, bias1 = np.random.randn(input_size, hidden_size1), np.random.randn(
    hidden_size1
)
weights2, bias2 = np.random.randn(hidden_size1, hidden_size2), np.random.randn(
    hidden_size2
)
weights3, bias3 = np.random.randn(hidden_size2, output_size), np.random.randn(
    output_size
)

# Hyperparameters
learning_rate = 0.01
epochs = 100

# Training the model
for epoch in range(epochs):
    # Forward propagation
    hidden_layer1 = np.dot(train_in, weights1) + bias1
    hidden_layer1_activation = relu(hidden_layer1)

    hidden_layer2 = np.dot(hidden_layer1_activation, weights2) + bias2
    hidden_layer2_activation = relu(hidden_layer2)

    output = np.dot(hidden_layer2_activation, weights3) + bias3

    # Calculate the Mean Absolute Error (MAE)
    error = output - train_out
    mae = np.mean(np.abs(error))

    # Backward propagation
    grad_output = np.where(error > 0, 1, -1)  # Derivative of MAE loss w.r.t output

    grad_hidden_layer2_activation = np.dot(grad_output, weights3.T) * np.where(
        hidden_layer2 > 0, 1, 0
    )
    grad_hidden_layer1_activation = np.dot(
        grad_hidden_layer2_activation, weights2.T
    ) * np.where(hidden_layer1 > 0, 1, 0)

    # Update the weights and biases
    weights3 -= learning_rate * np.dot(hidden_layer2_activation.T, grad_output)
    bias3 -= learning_rate * np.sum(grad_output)

    weights2 -= learning_rate * np.dot(
        hidden_layer1_activation.T, grad_hidden_layer2_activation
    )
    bias2 -= learning_rate * np.sum(grad_hidden_layer2_activation)

    weights1 -= learning_rate * np.dot(train_in.T, grad_hidden_layer1_activation)
    bias1 -= learning_rate * np.sum(grad_hidden_layer1_activation)

# Testing the model
# Allow the user to input age
while True:
    try:
        age_input = float(input("Enter an age (between 0 and 90): "))
        if 0 <= age_input <= 90:
            break
        else:
            print("Please enter a valid age between 0 and 90.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Scale the user input
user_input = age_input / 90.0

# Forward propagation on user input
hidden_layer1 = np.dot(user_input, weights1) + bias1
hidden_layer1_activation = relu(hidden_layer1)

hidden_layer2 = np.dot(hidden_layer1_activation, weights2) + bias2
hidden_layer2_activation = relu(hidden_layer2)

user_output = np.dot(hidden_layer2_activation, weights3) + bias3

# Print the result
print(f"Predicted output for age {age_input}: {user_output[0]}")
