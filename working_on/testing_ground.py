import sys
import Network
import Activation
import Datasets
import DataManager
import numpy as np
import time

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)

np.random.seed(1)
p_connect = [1.0, 1.0]
lr = np.array(p_connect)*0.001
start = time.time()
network = Network.Neural([img_size, 100, n_labels], Activation.Fn("relu"), lr, p_connect, True)
error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(2, x_train, y_train, x_test, y_test, 10000)
end = time.time()
print(end-start)

data = DataManager.prepare_experiment_data("first", 
                                    network.layers, 
                                    network.activation_function.fn_name, 
                                    network.lr.tolist(), 
                                    network.p_connect, 
                                    network.bias, 
                                    network.count_synapses(), 
                                    test_error, 
                                    test_accuracy, 
                                    test_energy, 
                                    min_energy, 
                                    samples_seen)

DataManager.save_data("test2", data)