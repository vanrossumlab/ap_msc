from components import Network, NormNet, Activation, Datasets, DataManager
import numpy as np
import time
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)

p_connect = [0.2, 0.2]
lr = np.array([0.001, 0.001])
start = time.time()
np.random.seed(2)
network = Network.Neural([img_size, 100, n_labels], Activation.Fn("relu"), lr, p_connect, True)
error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(1, x_train, y_train, x_test, y_test, 5000)
end = time.time()
print("Time Taken: ", end-start)
data = DataManager.prepare_simulation_data("ungated_100_neurons_30_epochs", 
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
DataManager.save_data("ungated_100_neurons_30_epochs", data)

start = time.time()
np.random.seed(2)
network = Network.Neural([img_size, 100, n_labels], Activation.Fn("relu"), lr, p_connect, True)
error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(1, x_train, y_train, x_test, y_test, 5000)
end = time.time()
print("Time Taken: ", end-start)
data = DataManager.prepare_simulation_data("normd_100_neurons_30_epochs", 
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
DataManager.save_data("normd_100_neurons_30_epochs", data)

norm_data = DataManager.load_data("normd_100_neurons_30_epochs")
ungated_data = DataManager.load_data("ungated_100_neurons_30_epochs")


plt.figure()
plt.title("Accuracy vs Energy", fontsize=24)
plt.xlabel("Accuracy", fontsize=18)
plt.ylabel("Energy", fontsize=18)
#plt.xlim(90, 95)
plt.plot(norm_data['results']['accuracy'], norm_data['results']['energy'])
plt.plot(ungated_data['results']['accuracy'], ungated_data['results']['energy'])
plt.yscale("log")
plt.legend(["Normalised", "Ungated"])
plt.show()