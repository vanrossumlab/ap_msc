from components import Experiment
from components import Network, Gated, Activation, Datasets
from components import DataManager as dm
import numpy as np
import matplotlib.pyplot as plt
import time

file_name = '100_units_lr_gated_ungated'

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)

layers = [img_size, 100, n_labels]
activation_function = Activation.Fn('relu')
lr = [np.array([0.0005, 0.0005]), np.array([0.001, 0.001]), np.array([0.005, 0.005]), np.array([0.01, 0.01]), np.array([0.05, 0.05])]
p_connect = [1.0, 1.0]
bias = True

net_type = [0, 1]

n_epochs = 50
test_interval = 5000

parameters = [layers,
              activation_function, 
              lr, 
              p_connect,
              bias,
              n_epochs, 
              x_train, 
              y_train, 
              x_test, 
              y_test, 
              test_interval,
              net_type]

independent_parameters_idxs = [2, 11]

# process to make parallel
def f(queue, parameter, name):
    start = time.time()
    if parameter[11] == 0:
        np.random.seed(2)
        network = Network.Neural(parameter[0], parameter[1], parameter[2], parameter[3], parameter[4])
        error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(parameter[5], 
                                                                                                                                    parameter[6], 
                                                                                                                                    parameter[7], 
                                                                                                                                    parameter[8], 
                                                                                                                                    parameter[9], 
                                                                                                                                    parameter[10],
                                                                                                                                    True)
        data = dm.prepare_simulation_data("ungated_100_neurons", 
                                                    network.layers, 
                                                    network.activation_function.fn_name, 
                                                    network.lr.tolist(), 
                                                    network.p_connect, 
                                                    network.bias, 
                                                    network.count_synapses(), 
                                                    [error.tolist(), test_error.tolist()], 
                                                    [accuracy.tolist(), test_accuracy.tolist()], 
                                                    [energy.tolist(), test_energy.tolist()], 
                                                    min_energy.tolist(),  
                                                    samples_seen.tolist()
                                                    )
        queue.put(data)
    else:
        np.random.seed(2)
        network = Gated.Neural(parameter[0], parameter[1], parameter[2], parameter[3], 0.0, parameter[4])
        error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy,  samples_seen = network.train_and_test(parameter[5], 
                                                                                                                                    parameter[6], 
                                                                                                                                    parameter[7], 
                                                                                                                                    parameter[8], 
                                                                                                                                    parameter[9], 
                                                                                                                                    parameter[10],
                                                                                                                                    True)
        data = dm.prepare_simulation_data("gated_100_neurons", 
                                                    network.layers, 
                                                    network.activation_function.fn_name, 
                                                    network.lr.tolist(), 
                                                    network.p_connect, 
                                                    network.bias, 
                                                    network.count_synapses(), 
                                                    [error.tolist(), test_error.tolist()], 
                                                    [accuracy.tolist(), test_accuracy.tolist()], 
                                                    [energy.tolist(), test_energy.tolist()], 
                                                    min_energy.tolist(), 
                                                    samples_seen.tolist()
                                                    )
        queue.put(data)
    print("Time: ", time.time()-start)

# run multiple simulations
# currently sets cpus to number of first independent variables so 3 simulations would instance 3 cpus
# so careful with your iv's
def main():
    Experiment.main(f, parameters, independent_parameters_idxs, "cumulative accuracy", file_name, 5)
    

if __name__ == '__main__':
    main()
    