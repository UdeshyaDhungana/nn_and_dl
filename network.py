import numpy as np
import random


class Network:
    def __init__(self, sizes) -> None:
        """Initialize parameters of network.
        sizes: list consisting number of neurons in each layer"""
        self.sizes = sizes
        self.layers = len(sizes)
        # Converts sizes to list of numpy array of following dimensions
        # [784, 30, 10] -> [(30,784), (10, 30)]
        self.weights = [np.random.randn(y,x)
                        for x,y in zip(sizes, sizes[1:])]
        # similar as above
        # [784, 30, 10] -> [(30,1), (10, 1)]
        self.biases = [np.random.randn(size, 1) for size in sizes[1:]]

    def feedForward(self, a):
        """Return output if a is input"""
        # perform forward pass and return the result
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Apply a Stochastic Gradient Descent to given training data,
        eta: learning rate
        training_data: 
        test_data: True if you're testing data"""
        # training_data, test_data are passed as zip, convert them to list
        train = list(training_data)
        test = list(test_data)
        # if test data is present, capture its length
        if test:
            n_test = len(test)
        # capture length of training data
        n = len(train)
        for j in range(epochs):
            # shuffle training data
            random.shuffle(train)
            # prepare mini batches according to mini batch size
            mini_batches = [
                train[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            # update weights and biases by processing each mini batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # If it's a test round, print epoch accuracy as well
            if test:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test), n_test))
            else:
                # print epoch number only else
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, batch, eta):
        """
        Updates network using privided batch with learning rate eta
        """
        # prepare gradient of weights and biases,
        # these are list of numpy arrays with same dimension as weights and
        # biases of the network
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # length of mini batch
        m = float(len(batch))
        for x, y in batch:
            # using backpropagation calculate required del_nablaW and del_nablaB
            # but this is only a part of what we want because nablaW and nablaB
            # would be computed taking sum of all the gradients of training examples
            # in the mini batch
            del_nabla_w, del_nabla_b = self.backPropagate(x, y)

            # add the nablas from individual training examples
            nabla_b = [nb + dnb for nb,
                       dnb in zip(nabla_b, del_nabla_b)]
            nabla_w = [nw + dnw for nw,
                       dnw in zip(nabla_w, del_nabla_w)]

        # Finally update the weights and biases using the formula
        # Remember, you're updating np arrays inside a list so use list comprehension
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def backPropagate(self, x, y):
        """Returns nabla of weights and biases after backpropagation algorithm"""
        # allocate space for nablaW and nablaB for this training example
        nabla_w = [np.zeros(weights.shape) for weights in self.weights]
        nabla_b = [np.zeros(biases.shape) for biases in self.biases]

        # Feedforward (we didn't use feedforward function because it doesn't store intermediate results)
        # activations in the network
        zs = []
        # weighted activations (after applying sigmoid function) in the network
        activations = [x]
        # this is a variable to which we apply forward pass repeatedly
        activation = x
        for weights, biases in zip(self.weights, self.biases):
            z = weights @ activation  + biases
            activation = sigmoid(z)
            activations.append(activation)
            zs.append(z)

        # Propagate backwards
        # Error in the final layer (using formula)
        del_L = (activations[-1] - y) * sigmoid_derivative(zs[-1])
        # nablaW in final layer
        # nabla B in final layer
        nabla_w[-1] = del_L @ activations[-2].T
        nabla_b[-1] = del_L

        for i in range(2, self.layers):
            # this layer's error in terms of error in the next layer
            del_L = (self.weights[-i + 1].T @ del_L) * sigmoid_derivative(zs[-i])
            # calculate nablaW in this layer according to error in this layer (using formula)
            nabla_w[-i] = del_L @ activations[-(i+1)].T
            # calculate nablaB (using formula)
            nabla_b[-i] = del_L

        return nabla_w, nabla_b


    def evaluate(self, test_data):
        """Evaluate the model using test_data"""
        # return number of correct classififcations in the test dataset
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