import Network
import Gated
import Activation
import Datasets
import DataManager
import numpy as np
import time
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)
cut_off = 10000
#x_test = x_train[cut_off:]
#y_test = y_train[cut_off:]
#x_train = x_train[:cut_off]
#y_train = y_train[:cut_off]


np.random.seed(2)
p_connect = [1.0, 1.0]
lr = np.array([0.001, 0.001])
start = time.time()
network = Gated.Neural([img_size, 250, n_labels], Activation.Fn("relu"), lr, p_connect, True)
error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(50, x_train, y_train, x_test, y_test, 10000)
end = time.time()
print("Time Taken: ", end-start)
data = DataManager.prepare_simulation_data("gated_250_neurons_lr_50_epochs", 
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
DataManager.save_data("gated_250_neurons_lr_50_epochs", data)

start = time.time()
network = Network.Neural([img_size, 250, n_labels], Activation.Fn("relu"), lr, p_connect, True)
d_error, d_accuracy, d_energy, d_test_error, d_test_accuracy, d_test_energy, min_energy, samples_seen = network.train_and_test(50, x_train, y_train, x_test, y_test, 10000)
end = time.time()
print("Time Taken: ", end-start)
data = DataManager.prepare_simulation_data("not_gated_increasing_threshold_50_epochs", 
                                            network.layers, 
                                            network.activation_function.fn_name, 
                                            network.lr.tolist(), 
                                            network.p_connect, 
                                            network.bias, 
                                            network.count_synapses(), 
                                            d_test_error, 
                                            d_test_accuracy, 
                                            d_test_energy, 
                                            min_energy, 
                                            samples_seen)
DataManager.save_data("gated_increasing_threshold_50_epochs", data)

plt.figure()
plt.title("Accuracy vs Energy", fontsize=24)
plt.xlabel("Accuracy", fontsize=18)
plt.ylabel("Energy", fontsize=18)
plt.plot(test_accuracy, test_energy)
plt.plot(d_test_accuracy, d_test_energy)
plt.yscale("log")
plt.legend(["Gated", "Non-Gated"])
plt.show()

plt.figure()
plt.title("Test Error vs Test Accuracy", fontsize=24)
plt.xlabel("Accuracy", fontsize=18)
plt.ylabel("Error", fontsize=18)
plt.plot(test_accuracy, test_error)
plt.plot(d_test_accuracy, d_test_error)
plt.legend(["Gated", "Non-Gated"])
plt.show()

plt.figure()
plt.title("Training Error vs Testing Error", fontsize=24)
plt.xlabel("Training Error", fontsize=18)
plt.ylabel("Testing Error", fontsize=18)
plt.plot(error, test_error)
plt.plot(d_error, d_test_error)
plt.legend(["Gated", "Non-Gated"])
plt.show()