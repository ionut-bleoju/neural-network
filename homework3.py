import numpy as np
from torchvision.datasets import MNIST

# Constants
NUM_INPUT_NEURONS = 784
NUM_HIDDEN_NEURONS = 100
NUM_OUTPUT_NEURONS = 10
INITIAL_LEARNING_RATE = 0.01
NUM_GENERATIONS = 20
BATCH_SIZE = 200
RANDOM_SEED = 23
DECAY_FACTOR = 0.75
PATIENCE = 2
EARLY_STOPPING_PATIENCE = 4

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)

    images = []
    labels = []
    for image, label in dataset:
        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def forward_propagation(inputs, weights_input_to_hidden, weights_hidden_to_output):
    hidden_layer_input = np.dot(inputs, weights_input_to_hidden)
    hidden_layer_output = relu(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output)
    output_layer_output = sigmoid(output_layer_input)
    return hidden_layer_output, output_layer_output

def compute_loss(predictions, targets):
    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

def backpropagation(inputs, hidden_layer_output, output_layer_output, targets, weights_input_to_hidden, weights_hidden_to_output, learning_rate):
    output_error = output_layer_output - targets
    
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_error = np.dot(output_delta, weights_hidden_to_output.T)
    
    # Compute the gradient (delta) at the hidden layer
    hidden_delta = hidden_error * relu_derivative(hidden_layer_output)

    # Update the weights between the hidden layer and the output layer
    weights_hidden_to_output -= learning_rate * np.dot(hidden_layer_output.T, output_delta)
    
    # Update the weights between the input layer and the hidden layer
    weights_input_to_hidden -= learning_rate * np.dot(inputs.T, hidden_delta)

    return weights_input_to_hidden, weights_hidden_to_output

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


def get_train_data():
    train_images, train_labels = download_mnist(True)
    train_images = train_images / 255.0
    train_labels = one_hot_encode(train_labels, 10)
    
    return train_images, train_labels

def get_test_data():
    test_images, test_labels = download_mnist(False)
    test_images = test_images / 255.0
    test_labels = one_hot_encode(test_labels, 10)
    
    return test_images, test_labels

def main():
    train_images, train_labels = get_train_data()
    test_images, test_labels = get_test_data()

    np.random.seed(RANDOM_SEED)
    weights_input_to_hidden = np.random.randn(NUM_INPUT_NEURONS, NUM_HIDDEN_NEURONS) * 0.1 
    weights_hidden_to_output = np.random.randn(NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS) * 0.1 

    learning_rate = INITIAL_LEARNING_RATE
    best_loss = 1.0
    patience_counter = 0
    early_stopping_counter = 0
    last_accuracy = 0

    # Training loop
    for generation in range(NUM_GENERATIONS):
        for i in range(0, len(train_images), BATCH_SIZE):
            batch_inputs = train_images[i:i+BATCH_SIZE]
            batch_targets = train_labels[i:i+BATCH_SIZE]

            hidden_layer_output, output_layer_output = forward_propagation(batch_inputs, weights_input_to_hidden, weights_hidden_to_output)
            loss = compute_loss(output_layer_output, batch_targets)
            weights_input_to_hidden, weights_hidden_to_output = backpropagation(batch_inputs, hidden_layer_output, output_layer_output, batch_targets, weights_input_to_hidden, weights_hidden_to_output, learning_rate)

        # Calculate accuracy on training data
        _, train_predictions = forward_propagation(train_images, weights_input_to_hidden, weights_hidden_to_output)
        train_predictions = np.argmax(train_predictions, axis=1)
        train_labels_argmax = np.argmax(train_labels, axis=1)
        train_accuracy = np.mean(train_predictions == train_labels_argmax)
        print(f'Genaration {generation+1}/{NUM_GENERATIONS}, Loss: {loss}, Training Accuracy: {train_accuracy * 100:.2f}%')

        # Check if loss improved
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Reduce learning rate if no improvement for PATIENCE generations
        if patience_counter >= PATIENCE:
            learning_rate *= DECAY_FACTOR
            print(f'Reducing learning rate to {learning_rate}')
            patience_counter = 0

        # Early stopping if accuracy does not improve for EARLY_STOPPING_PATIENCE generations
        if train_accuracy == last_accuracy:
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        last_accuracy = train_accuracy

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping due to no improvement in accuracy.")
            break

    # Evaluation on test data
    _, test_predictions = forward_propagation(test_images, weights_input_to_hidden, weights_hidden_to_output)
    test_predictions = np.argmax(test_predictions, axis=1)
    test_labels = np.argmax(test_labels, axis=1)
    accuracy = np.mean(test_predictions == test_labels)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()