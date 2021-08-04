import Network, Gated, Activation, Datasets
import numpy as np
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)
cut_off = 10000
#x_test = x_train[cut_off:]
#y_test = y_train[cut_off:]
#x_train = x_train[:cut_off]
#y_train = y_train[:cut_off]


np.random.seed(2)
p_connect = [1.0, 1.0]
p_connect_skip = [0.1, 0.1]
lr = np.array([0.001, 0.001])
network = Network.Neural([img_size, 20, n_labels], Activation.Fn("relu"), lr, p_connect, True)
error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(1, x_train, y_train, x_test, y_test, 10000)

np.random.seed(2)
p_connect = [1.0, 1.0]
p_connect_skip = [0.1, 0.1]
lr = np.array([0.001, 0.001])
network = Network.Neural([img_size, 30, n_labels], Activation.Fn("relu"), lr, p_connect, True)
error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(1, x_train, y_train, x_test, y_test, 10000)

np.random.seed(2)
p_connect = [1.0, 1.0]
p_connect_skip = [0.1, 0.1]
lr = np.array([0.001, 0.001])
network = Network.Neural([img_size, 40, n_labels], Activation.Fn("relu"), lr, p_connect, True)
error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(1, x_train, y_train, x_test, y_test, 10000)




# np.random.seed(2)
# p_connect = [1.0, 1.0]
# network = Gated.Neural([img_size, 100, n_labels], Activation.Fn("relu"), lr, p_connect, True)
# d_error, d_accuracy, d_energy, d_test_error, d_test_accuracy, d_test_energy, d_min_energy, samples_seen = network.train_and_test(10, x_train, y_train, x_test, y_test, 10000)

# plt.figure()
# plt.title("Residue: Accuracy vs Energy (p = " + str(p_connect[0]) + ")", fontsize=24)
# plt.xlabel("Accuracy", fontsize=18)
# plt.ylabel("Energy", fontsize=18)
# plt.plot(test_accuracy, test_energy)
# plt.plot(d_test_accuracy, d_test_energy)
# plt.yscale("log")
# plt.legend(["Residue", "Traditional"])
# plt.show()