from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sys import stderr
import numpy as np


# --------------------------------------- Classes --------------------------------------- #

class NeuralNetwork:
    def __init__(self, input_size: int=64, hidden1_size: int=32, hidden2_size: int=12, output_size: int=10):
        self.h1_size = hidden1_size
        self.h2_size = hidden2_size

        try:
            self.load_weights_biases()
        except FileNotFoundError:
            # Initializing Weights and Biases:
            self.W1 = np.random.randn(input_size, hidden1_size)  # Input -> Hidden 1...
            self.b1 = np.zeros( (1, hidden1_size) )

            self.W2 = np.random.randn(hidden1_size, hidden2_size) # Hidden 1 -> Hidden 2...
            self.b2 = np.zeros( (1, hidden2_size) )

            self.W3 = np.random.randn(hidden2_size, output_size)  # Hidden 2 -> Output...
            self.b3 = np.zeros( (1, output_size) )

    def load_weights_biases(self) -> None:
        data = np.load('..\\Assets\\Trained_Network.npz')
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']

    def save_weights(self):
        np.savez (
            file = '..\\Assets\\Trained_Network.npz',

            W1 = self.W1,
            b1 = self.b1,

            W2 = self.W2,
            b2 = self.b2,

            W3 = self.W3,
            b3 = self.b3
        )

    def activation(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / ( 1 + np.exp(-x) )   # Sigmoid implementation...

    # Forward propagation:
    def forward(self, X):
        hidden1_activation = self.activation( np.dot(X, self.W1) + self.b1 )
        hidden2_activation = self.activation( np.dot(hidden1_activation, self.W2) + self.b2 )
        output_activation = self.activation( np.dot(hidden2_activation, self.W3) + self.b3 )
        return hidden1_activation, hidden2_activation, output_activation

    # Back propagation:
    def backward(self, X, y, learning_rate, hidden1_result, hidden2_result, output_result):
        m = 1 / X.shape[0]

        # Calculate gradients
        out_error = output_result - y
        out_weight_delta = m * np.dot(hidden2_result.T, out_error)
        out_bias_delta = m * np.sum(out_error, axis=0)

        h2_error = np.dot(out_error, self.W3.T) * self.activation(hidden2_result, derivative=True)
        h2_weight_delta = m * np.dot(hidden1_result.T, h2_error)
        h2_bias_delta = m * np.sum(h2_error, axis=0)

        h1_error = np.dot(h2_error, self.W2.T) * self.activation(hidden1_result, derivative=True)
        h1_weight_delta = m * np.dot(X.T, h1_error)
        h1_bias_delta = m * np.sum(h1_error, axis=0)

        # Update weights and biases
        self.W3 -= learning_rate * out_weight_delta
        self.b3 -= learning_rate * out_bias_delta

        self.W2 -= learning_rate * h2_weight_delta
        self.b2 -= learning_rate * h2_bias_delta

        self.W1 -= learning_rate * h1_weight_delta
        self.b1 -= learning_rate * h1_bias_delta

        return out_error

    def train(self, X, y, iterations: int=1_000, learning_step: float=.5, adaptive_step=False):
        y = digits_to_neurons( y.astype(int) )

        learn_step_val = learning_step  # Stores the learning_step value...
        train_errors = []
        accuracy = []
        weights = []

        for epoch in range(iterations):
            # Get layer outputs:
            hidden1, hidden2, output = self.forward(X)

            if adaptive_step and (not learning_step < 0.01):    # For adaptive weight...
                learn_step_val = learning_step * (1 - epoch / iterations)

            # Loss calculation:
            loss = np.square (  # Squares all values (val**2)...
                self.backward(X, y, learn_step_val, hidden1, hidden2, output)
            )

            if epoch%100 == 0:  # Print loss every 100 epochs:
                print(f'Epoch {epoch}, Loss: {np.mean(loss)}')

                # Store the train-errors and accuracy for plotting:
                train_errors.append( np.mean(loss) )
                accuracy.append( calculate_accuracy(output, y) )
                weights.append(np.abs( np.mean(self.W3, axis=0) ))

        # self.save_weights()
        return train_errors, accuracy, weights

    def predict(self, X):
        _, _, out = self.forward(X)
        return np.argmax(out, axis=1)   # Return the index of the largest value in the columns(axis=1)...


# -------------------------------------- Functions -------------------------------------- #

# Loads the machine-learning data from the files:
# Both (training and testing)
def load_ML_data(filename: str):
    data = np.genfromtxt(filename, delimiter=',')

    features = data[:, :-1]; labels = data[:, -1]
    features /= 16.0  # Convert pixel values to range 0-1...

    return features, labels


# Displays 8x8 images in 2D ASCII art format:
def display_images(images, labels, count=5) -> None:
    for i in range(count):
        print(f'\nLabel: {labels[i]}')
        image_2d = np.reshape( images[i], (8, 8) )
        for row in image_2d:
            line = ''.join( ['# ' if val >= .5 else '. ' for val in row] )
            print(line)


# Get the learning_step:
def get_learning_step() -> float and bool:
    while True:
        try:
            step = input('\nEnter the size of the learning-step(none = adaptive): ')
            if step == '':
                print('\t\x1b[32mLearning-step set to adaptive!\x1b[0m')
                return 1.0, True
            else:
                step = float(step)
                if (step < 0.01) or (step > 1):
                    print(f'\t\x1b[31mInvalid Input. Not in range: {0.01} - {1}\x1b[0m', file=stderr)
                    continue
                break
        except ValueError:
            print('\t\x1b[31mInvalid input. Please enter a number.\x1b[0m', file=stderr)
        except KeyboardInterrupt:
            print('\n\t\x1b[31mPlease enter the appropriate option to quit the program.\x1b[0m', file=stderr)
    return step, False


# Converts labels to one-hot encoded vectors:
def digits_to_neurons(numbers):
    return np.eye(10)[numbers]


# Calculates the accuracy of predictions:
def calculate_accuracy(predictions, labels):
    predictions = np.argmax(predictions, axis=1)
    labels = np.argmax(labels, axis=1)
    return np.mean(predictions == labels)


# Generates and displays subplots of training errors, accuracy, and weight changes:
def generate_graphs(errors, accuracy, weights, topology, learning_step, adaptive) -> None:
    fig, axes = plt.subplots(2)

    axes[0].plot(range( len(errors) ), errors, label='Training Error')
    axes[0].plot(range( len(accuracy) ), accuracy, label='Training Accuracy')
    axes[0].set_title (
        f'Training Error and Accuracy Graph / Topology: {topology} Step: {"adaptive" if adaptive else learning_step}',
        fontsize=16,
        fontweight='bold'
    )
    axes[0].set_ylabel('Error / Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend() # Display lines with their associated colors on the side...

    epochs = range( len(weights) )
    weights = np.transpose(weights)

    # 'enumerate' function : Returns 2 values(index, value):
    for i, neuron_weights in enumerate(weights):
        axes[1].plot(epochs, neuron_weights, label=f'Weight {9-i}')
    axes[1].set_title (
        'Weight changing graph',
        fontsize=16,
        fontweight='bold'
    )
    axes[1].set_ylabel('Weight')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.show()


# Prints a classification report and displays a confusion-matrix of the predictions:
def predictions_report(predictions, labels, topology, learning_step, adaptive) -> None:
    print(f'\n{ classification_report(labels, predictions) }')

    cm = confusion_matrix(labels, predictions)  # Confusion Matrix...
    num_classes = 10    # There are 10 digits total...

    plt.title (
        f'Confusion Matrix / Topology: {topology} Step: {"adaptive" if adaptive else learning_step}',
        fontsize=16,
        fontweight='bold'
    )

    plt.ylabel('True label')
    plt.yticks( ticks=range(num_classes) )

    plt.xlabel('Predicted label')
    plt.xticks( ticks=range(num_classes) )

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()

    for row in range(num_classes):
        for col in range(num_classes):
            plt.text(col, row, cm[row, col], horizontalalignment='center', color='black')

    plt.tight_layout()
    plt.show()


# Performs multiple experiments with random parameters and displays the results:
def perform_experiments(train_data, labels, experiments_num: int=4) -> None:
    fig, axes = plt.subplots(experiments_num, 3)
    plt.subplots_adjust(hspace=0.5) # Adjust vertical spacing between subplots

    axes[0, 1].set_title('Weight Changing')
    axes[0, 2].set_title('Confusion-Matrix')

    for i in range(experiments_num):
        size1 = np.random.randint(5, 50)
        size2 = np.random.randint(5, 50)
        train_epochs = np.random.randint(200, 1500)
        train_step = np.random.uniform(0.01, 1.0)
        adaptive = np.random.choice([True, False])

        print(f'\n\033[35mExperiment {i+1}\033[0m:')

        ann = NeuralNetwork(hidden1_size=size1, hidden2_size=size2)
        errors, accuracy, weights = ann.train (
            X=train_data,
            y=labels,
            iterations=train_epochs,
            learning_step=train_step,
            adaptive_step=adaptive
        )
        predictions = ann.predict(train_data)

        print(f'\n{ classification_report(labels, predictions) }')

        axes[i, 0].plot(range( len(errors) ), errors, label='Training Error')
        axes[i, 0].plot(range( len(accuracy) ), accuracy, label='Training Accuracy')
        axes[i, 0].set_title(f'Training Error and Accuracy / Topology: {(size1, size2)} Step: {"adaptive" if adaptive else round(train_step, 3)}')
        axes[i, 0].set_ylabel('Error / Accuracy')
        axes[i, 0].set_xlabel('Epoch')
        axes[i, 0].legend()

        epochs = range( len(weights) )
        weights = np.transpose(weights)
        for count, neuron_weights in enumerate(weights):
            axes[i, 1].plot(epochs, neuron_weights, label=f'Weight {9-count}')
        axes[i, 1].set_ylabel('Weight')
        axes[i, 1].set_xlabel('Epoch')

        cm = confusion_matrix(labels, predictions)
        axes[i, 2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[i, 2].set_xlabel('Predicted label')
        axes[i, 2].set_ylabel('True label')

    plt.show()

