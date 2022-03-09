# See the magic
# import network
from cProfile import label
import network2
import network3
import mnist_loader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    sizes = [784, 30, 10]

    # ==========First neural network we built==========
    # network = network.Network(sizes)
    # network.SGD(training_data=training_data, epochs=15, mini_batch_size=50, eta=3.0, test_data=test_data)


    # =====The one with cross entropy cost and l2 regularization=====
    # net = network2.Network(sizes, cost=network2.CrossEntropyCost)
    # evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
    #     training_data,
    #     30,
    #     10,
    #     0.5,
    #     lmbda=5.0,
    #     evaluation_data=validation_data,
    #     monitor_evaluation_accuracy=True,
    #     monitor_evaluation_cost=True,
    #     monitor_training_accuracy=True,
    #     monitor_training_cost=True)
    # plt.plot(evaluation_accuracy, label='Evaluation Accuracy')
    # plt.title('Neural Network Metrics')
    # plt.legend()
    # plt.savefig('network2Metrics.png')
    # net.save('network2.txt')


    # ===== network2.py with velocity term for faster learning =====
    net = network3.Network(sizes, cost=network2.CrossEntropyCost)
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
        training_data,
        30,
        10,
        0.5,
        lmbda=5.0,
        mu=0.5,
        evaluation_data=validation_data,
        monitor_evaluation_accuracy=True,
        monitor_evaluation_cost=True,
        monitor_training_accuracy=True,
        monitor_training_cost=True)
    plt.plot(evaluation_accuracy, label='Evaluation Accuracy')
    plt.title('Neural Network Metrics')
    plt.legend()
    plt.savefig('network3Metrics.png')
    net.save('network3.txt')