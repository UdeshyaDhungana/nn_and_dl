from audioop import bias
import numpy as np
import random


class Network:
    def __init__(self, sizes) -> None:
        """Initialize parameters of network.
        sizes: One dimensional np array consisting number of neurons in each layer"""
        self.sizes = sizes
        self.layers = len(sizes)
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(sizes, sizes[1:])]
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]

    def feedForward(self, a):
        """Return output if a is input"""
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Apply a Stochastic Gradient Descent to given training data,
        eta: learning rate
        training_data: 
        test_data: True if you're testing data"""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, batch, eta):
        """
        Updates network using privided batch with learning rate eta
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            del_nabla_w, del_nabla_b = self.backPropagate(x, y)
            nabla_b = [nabla_b + del_nabla_b for nb,
                       dnb in zip(nabla_b, del_nabla_b)]
            nabla_w = [nabla_w + del_nabla_w for nw,
                       dnw in zip(nabla_w, del_nabla_w)]

        # Remember, you're updating np arrays inside a list so use list comprehension
        self.weights = [w - (eta / batch.size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / batch.size) * nb for b, nb in zip(self.biases, nabla_b)]

    def backPropagate(self, x, y):
        """Returns nabla of weights and biases after backpropagation algorithm"""
        nabla_w = [np.zeros(weights.shape) for weights in self.weights]
        nabla_b = [np.zeros(biases.shape) for biases in self.biases]

        # Feedforward (we didn't use feedforward function because it doesn't store intermediate results)
        zs = []
        activations = [x]
        input_values = x
        for weights, biases in zip(self.weights, self.biases):
            z = weights @  + biases
            input_values = sigmoid(z)
            activations.append(input_values)
            zs.append(z)

        # Propagate backwards
        del_L = (activations[-1] - y) * sigmoid_derivative(zs[-1])
        nabla_w[-1] = del_L @ activations[-2].T
        nabla_b[-1] = del_L

        for i in range(2, self.layers):
            del_L = (self.weights[-i + 1].T @ del_L) * sigmoid_derivative(zs[-i])
            nabla_w[-i] = del_L @ activations[-(i+1)].T
            nabla_b[-i] = del_L

        return nabla_w, nabla_b


    def evaluate(self, test_data):
        """Evaluate the model using test_data"""
        test_results = [(np.argmax(self.feedForward(x)), y) for (x,y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


## Miscallaneous functions
def sigmoid(z):
    """Calculate sigmoid of a matrix"""
    return 1 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    """Calculate the derivative of sigmoid of a function"""
    sz = sigmoid(z)
    return sz * (1 - sz)