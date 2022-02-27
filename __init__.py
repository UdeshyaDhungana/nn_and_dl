# See the magic
import network
import mnist_loader

if __name__=="__main__":
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    network = network.Network([784, 50, 10])
    network.SGD(training_data=training_data, epochs=15, mini_batch_size=50, eta=3.0, test_data=test_data)