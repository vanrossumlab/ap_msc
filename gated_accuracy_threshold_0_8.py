import Network, Gated, Activation, Datasets
import DataManager as dm
import numpy as np
import time

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)
n_epochs = 30
test_interval = 5000

np.random.seed(1)
p_connect = [1.0, 1.0]
lr = np.array([0.001, 0.001])
threshold = 0.8

experiment_data = []
start = time.time()
np.random.seed(1)
network = Gated.Neural([img_size, 150, n_labels], Activation.Fn("relu"), lr, p_connect, True)
error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(n_epochs, 
                                                                                                                   x_train, 
                                                                                                                   y_train, 
                                                                                                                   x_test,
                                                                                                                   y_test,
                                                                                                                   test_interval,
                                                                                                                   threshold)
sim_data = dm.prepare_simulation_data("thresold = 0.8", 
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
data = dm.prepare_experiment_data("thresholds", 
                                  experiment_data,
                                  "varying thresholds")
dm.save_data("data/gated_accuracy_varying_thresholds/t_0_8", data)
        