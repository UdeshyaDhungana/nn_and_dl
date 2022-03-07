import json
import random
import numpy as np
import sys


# Utils
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def vectorized_result(j):
    result = np.zeros((10, 1))
    result[j] = 1.0
    return result


# Quadratic Cost
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return error when a is evaluation and y is expected output"""
        return 0.5 * np.linalg.norm(a - y)**2

    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)


# CrossEntropy Cost
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return -np.sum(np.nan_to_num(y * np.log(a) + (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return (a - y)


class Network(object):

    # Initialization
    def __init__(self, sizes, cost=CrossEntropyCost) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.num_layers]
        self.weights = [np.random.randn(
            another, now)/np.sqrt(now) for another, now in zip(self.sizes[1:], self.sizes)]

    def larget_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.num_layers]
        self.weights = [np.random.randn(
            another, now) for another, now in zip(self.sizes[1:], self.sizes)]

    def feedForward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False,):
        n = len(training_data)
        if (evaluation_data):
            n_data = len(evaluation_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
            for batch in mini_batches:
                self.update_mini_batch(batch, eta, lmbda, n)
            print("Epoch {} complete", j)
            if (monitor_training_cost):
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost of training data: {}".format(cost))

            if (monitor_training_accuracy):
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy of traiining data: {} / {}".format(accuracy, n))

            if (monitor_evaluation_cost):
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                print("Cost of training data: {}".format(cost))

            if (monitor_evaluation_accuracy):
                accuracy = self.accuracy(evaluation_data, convert=True)
                evaluation_accuracy.append(accuracy)
                print("Accuracy of traiining data: {} / {}".format(accuracy, n_data))
            print()
            return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        m = float(len(mini_batch))

        for x, y in mini_batch:
            del_nabla_w, del_nabla_b = self.backPropagate(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, del_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, del_nabla_w)]

        # Update the params
        self.weights = [(1 - (eta * lmbda) / n) * w - (eta * nw / m)
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta * nb / m)
                       for b, nb in zip(self.biases, nabla_b)]

    def backPropagate(self, x, y):
        zs = []
        activations = [x]
        activation = x
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for w, b in zip(self.weights, self.biases):
            activation = w @ activation + b
            zs.append(activation)
            activation = sigmoid(activation)
            activations.append(activation)

        # Back propagate
        del_L = (self.cost).delta(zs[-1], activation, y)
        nabla_w[-1] = del_L @ activation[-2].T
        nabla_b[-1] = del_L

        for l in range(2, self.num_layers):
            z = zs[-l]
            del_L = (self.weights[-l+1].T @ del_L) * sigmoid_prime(z)
            nabla_w[-l] = del_L @ activation[-l-1].T
            nabla_b[-l] = del_L
        return (nabla_w, nabla_b)

    def accuracy(self, data, convert=True):
        """Convert is False if dataset is validation or test set, True otherwise.
        There is difference in representation in training and other sets"""
        if convert:
            # List of tuples
            results = [( np.argmax(self.feedForward(x)), np.argmax(y) ) for (x, y) in data]
        else:
            # 
            results = [( np.argmax(self.feedForward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x,y in data:
            a = self.feedForward(x)
            if convert:
                y = vectorized_result(y)
                cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save neural network to `filename`"""
        data = {
            'sizes': self.sizes,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'cost': str(self.cost.__name__)
        }
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()


# Load
def load(filename):
    """Load neural network from file `filename`"""
    f = open(filename, 'r')
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data['cost'])
    net = Network(data['sizes'], cost=cost)
    net.weights = [np.array(w) for w in data['weights']]
    net.biases = [np.array(b) for b in data['biases']]
    return net



