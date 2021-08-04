import Network, Activation, Datasets
import DataManager as dm
import numpy as np
import time

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)
n_epochs = 20
test_interval = 5000

np.random.seed(1)
p_l1 = 0.8
p_l2s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_hidden_units = 150
lr = np.array([0.001, 0.001])

experiment_data = []
for p_l2 in p_l2s:
    print("p_l1 = ", p_l1, " | p_l2 = ", p_l2)
    start = time.time()
    np.random.seed(1)
    p_connect = [p_l1, p_l2]
    network = Network.Neural([img_size, n_hidden_units, n_labels], Activation.Fn("relu"), lr, p_connect, True)
    error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(n_epochs, 
                                                                                                                       x_train, 
                                                                                                                       y_train, 
                                                                                                                       x_test,
                                                                                                                       y_test,
                                                                                                                       test_interval)
    name = "P_connect_1 = " + str(p_l1) + " | p_connect_2 = " + str(p_l2)
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
data = dm.prepare_experiment_data("variable connection probabilities with 200 units p8", 
                                  experiment_data,
                                  "[part 8] - seed = 1, epochs = 20, interval = 5000 | 200 units in the hidden layer, multithreaded therefore multiple parts")
dm.save_data("data/p_connect_1_vs_p_connect_2/p1_vs_p2_200_units_p8", data)
        