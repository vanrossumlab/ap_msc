from components import Experiment
from components import Network, Gated, Activation, Datasets
from components import DataManager as dm
import numpy as np
import matplotlib.pyplot as plt
import time

file_name = 'data/lr_vs_p'

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)

layers = [[img_size, 50, n_labels], [img_size, 100, n_labels],[img_size, 150, n_labels],[img_size, 200, n_labels],[img_size, 250, n_labels],[img_size, 300, n_labels]]
activation_function = Activation.Fn('relu')
learning_rates = np.array([0.001, 0.001])
p_connects = [1.0, 1.0]
bias = True

n_epochs = 1
test_interval = 5000

parameters = [layers,
              activation_function, 
              learning_rates, 
              p_connects,
              bias,
              n_epochs, 
              x_train, 
              y_train, 
              x_test, 
              y_test, 
              test_interval]

independent_parameters_idxs = [0]

# process to make parallel
def f(queue, parameter, name):
    network = Gated.Neural(parameter[0], 
                             parameter[1], 
                             parameter[2], 
                             parameter[3],
                             0.0,
                             parameter[4])
    error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(parameter[5], 
                                                                                                                       parameter[6], 
                                                                                                                       parameter[7], 
                                                                                                                       parameter[8], 
                                                                                                                       parameter[9],   
                                                                                                                       parameter[10],
                                                                                                                       False,)
    # sets up a dictionary such that the data can be stored in an easily addressable format
    # This is stored as a JSON file, which is loaded up later (line 68)
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
    queue.put(sim_data)

# run multiple simulations
# currently sets cpus to number of first independent variables so 3 simulations would instance 3 cpus
# so careful with your iv's
def main():
    Experiment.main(f, parameters, independent_parameters_idxs, "time taken", file_name, 10)
    

if __name__ == '__main__':
    main()
    