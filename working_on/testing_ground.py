import sys
import Network
import Activation
import Datasets
import numpy as np
import time

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)

np.random.seed(1)
p_connect = [0.1, 1.0]
lr = np.array([0.0005, 0.0005])
start = time.time()
network = Network.Neural([img_size, 100, n_labels], Activation.Fn("relu"), lr, True, p_connect)
error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy = network.train_and_test(20, x_train, y_train, x_test, y_test, 10000)
end = time.time()

print(end-start)