import Network, Activation, Datasets
import DataManager as dm
import numpy as np
import time

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)
n_epochs = 20
test_interval = 5000

np.random.seed(1)
p_connect = [0.6, 0.6]
n_hidden_units = [50, 100, 150, 200, 250, 300]
lr = np.array([0.001, 0.001])

experiment_data = []
for h in n_hidden_units:
    print("h = ", h)
    start = time.time()
    np.random.seed(1)
    network = Network.Neural([img_size, h, n_labels], Activation.Fn("relu"), lr, p_connect, True)
    error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(n_epochs, 
                                                                                                                       x_train, 
                                                                                                                       y_train, 
                                                                                                                       x_test,
                                                                                                                       y_test,
                                                                                                                       test_interval)
    name = "units = " + str(h) + " | p_connect = " + str(p_connect)
    sim_data = dm.prepare_simulation_data(name, 
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
    end = time.time()
    print("Time Taken: ", end-start)
    experiment_data.append(sim_data)
data = dm.prepare_experiment_data("variable hidden units and connection probability p3", 
                                  experiment_data,
                                  "[part 3] - seed = 1, epochs = 20, interval = 5000 | single hidden layer, ideally I'd prefer smoother intervals, this is ran effectively multithreaded and therefore has multiple parts")
dm.save_data("data/n_hid_units_vs_p_connect/hidden_units_vs_connection_p3", data)
        