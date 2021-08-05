import Experiment
import Network, Gated, Activation, Datasets
import DataManager as dm
import numpy as np
import matplotlib.pyplot as plt

file_name = 'mp_test'

x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = Datasets.load_mnist(True)

layers = [img_size, 20, n_labels]
activation_function = Activation.Fn('relu')
learning_rate = np.array([0.001, 0.001])
p_connects = [[1.0, 1.0], [0.9, 0.9], [0.8, 0.8], [0.7, 0.7], [0.6, 0.6], [0.5, 0.5], [0.4, 0.4], [0.3, 0.3], [0.2, 0.2], [0.1, 0.1]]
bias = True

n_epochs = 1
test_interval = 5000

parameters = [layers,
              activation_function, 
              learning_rate, 
              p_connects,
              bias,
              n_epochs, 
              x_train, 
              y_train, 
              x_test, 
              y_test, 
              test_interval]

independent_parameters_idxs = [3]

# process to make parallel
def f(queue, parameter, name):
    network = Network.Neural(parameter[0], 
                             parameter[1], 
                             parameter[2], 
                             parameter[3],
                             parameter[4])
    error, accuracy, energy, test_error, test_accuracy, test_energy, min_energy, samples_seen = network.train_and_test(parameter[5], 
                                                                                                                       parameter[6], 
                                                                                                                       parameter[7], 
                                                                                                                       parameter[8], 
                                                                                                                       parameter[9],   
                                                                                                                       parameter[10])
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
    Experiment.main(f, parameters, independent_parameters_idxs, "test", file_name, 5)
    d = dm.load_data('mp_test')
    labels = []
    for v in range(0, len(d[0])):
        plt.plot(d[0][v]['results']['accuracy'], d[0][v]['results']['energy'])
        labels.append(str(d[0][v]['network_parameters']['p_connect']))
    plt.legend(labels)
    plt.yscale('log')
    plt.ylabel('Energy /a.u.')
    plt.xlabel('Accuracy /%')
    plt.show()

if __name__ == '__main__':
    main()
    