import Network, Activation, Datasets
import DataManager as dm
import numpy as np
import time

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)
n_epochs = 25
test_interval = 5000

np.random.seed(1)
p_connects = [[1.0, 1.0], [0.5, 0.5], [0.1, 0.1]]
learning_rates = [np.array([0.1, 0.1]),
                  np.array([0.05, 0.05]), 
                  np.array([0.01, 0.01]),
                  np.array([0.005, 0.005]),
                  np.array([0.001, 0.001]),
                  np.array([0.0005, 0.0005]),
                  np.array([0.0001, 0.0001])]

experiment_data = []
for lr in learning_rates:
    set_data = []
    for p in p_connects:
        start = time.time()
        np.random.seed(1)
        network = Network.Neural([img_size, 100, n_labels], Activation.Fn("relu"), lr, p, True)
        error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(n_epochs, 
                                                                                                                           x_train, 
                                                                                                                           y_train, 
                                                                                                                           x_test,
                                                                                                                           y_test,
                                                                                                                           test_interval)
        name = "lr = " + str(lr) + " | p_connect = " + str(p)
        sim_data = dm.prepare_simulation_data(name, 
                                          network.layers, 
                                          network.activation_function.fn_name, 
                                          network.lr.tolist(), 
                                          network.p_connect, 
                                          network.bias, 
                                          network.count_synapses(), 
                                          error, 
                                          accuracy, 
                                          energy, 
                                          min_energy, 
                                          samples_seen)
        set_data.append(sim_data)
        end = time.time()
        print("Time Taken: ", start-end)
    experiment_data.append(set_data)
data = dm.prepare_experiment_data("variable lr and connection probability", 
                                  experiment_data,
                                  "seed = 1, epochs = 25, interval = 5000")
dm.save_data("varying_lr_and_connection", data)
        