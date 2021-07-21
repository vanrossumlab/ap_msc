import numpy as np
import Activation

class Neural():
    def __init__(self, layers, activation_function, learning_rate, p_connect=1, bias=True):
        self.layers = layers
        self.n_layers = len(layers)
        self.n_labels = layers[-1]
        self.activation_function = activation_function
        self.lr = learning_rate
        self.weight_mask = self.initialise_weight_mask(p_connect)#[np.random.binomial(1, p_connect, (i, j)) for i, j in zip(self.layers[:-1], self.layers[1:])]
        self.weights = self.initialise_weights(bias)
        self.initial_weights = self.weights
        self.biases = [np.zeros(b) for b in layers[1:]]
        self.dendrite = [np.ones(n) for n in layers[1:]]
        self.axon = [np.ones(n) for n in layers]
        self.d_weights = [np.zeros(w.shape) for w in self.weights]
        self.d_biases = [np.zeros(b.shape) for b in self.biases]
        self.p_connect = p_connect
        self.bias = bias
        self.energy = 0
    
    def initialise_weight_mask(self, p_connect):
        if type(p_connect) is float:
            weight_mask = [np.random.binomial(1, p_connect, (i, j)) for i, j in zip(self.layers[:-1], self.layers[1:])]
            return weight_mask
        else:
            weight_mask = [np.zeros((i, j)) for i, j in zip(self.layers[:-1], self.layers[1:])]
            if len(p_connect) != len(weight_mask):
                raise Exception("Connection probabilities not equal to number of layers")
            else:
                for layer in range(0, self.n_layers-1):
                    weight_mask[layer] = np.random.binomial(1, p_connect[layer], weight_mask[layer].shape)
                return weight_mask
            
    def initialise_weights(self, bias):
        weights = [np.random.randn(i, j)*np.sqrt(1/(i)) for i, j in zip(self.layers[:-1], self.layers[1:])]
        #if bias:
        #    for i in range(0, self.n_layers-1):
        #        weights[i] = np.vstack([weights[i], np.zeros(weights[i][0].shape)])
        for layer in range(0, self.n_layers-1):
            weights[layer] = np.multiply(self.weight_mask[layer], weights[layer])
        return weights
    
    def feedforward(self, x):
        self.axon[0] = x
        for layer in range(0, self.n_layers-1):
            self.dendrite[layer] = np.matmul(np.multiply(self.weights[layer], self.weight_mask[layer]).T, self.axon[layer]) + self.biases[layer]
            if layer == self.n_layers-1:
                self.axon[layer+1] = self.softmax(self.dendrite[layer])
            else:
                self.axon[layer+1] = self.activation_function(self.dendrite[layer])
        return self.dendrite, self.axon
        
    def backpropagate(self, x, y):
        error = self.axon[-1] - y # cross-entropy
        self.d_weights[-1] = np.outer(error, self.axon[-2].T)
        self.d_weights[-1] = np.multiply(self.d_weights[-1], self.weight_mask[-1].T)
        if self.bias:
            self.d_biases[-1] = error
        for layer in reversed(range(0, self.n_layers-2)):
            error = self.activation_function.prime(self.dendrite[layer])*np.matmul(np.multiply(self.weights[layer+1], self.weight_mask[layer+1]), error)
            self.d_weights[layer] = np.outer(error, self.axon[layer].T)
            self.d_weights[layer] = np.multiply(self.d_weights[layer], self.weight_mask[layer].T)
            if self.bias:
                self.d_biases[layer] = error
        return self.d_weights

    def update_weights(self):
        for layer in range(0, self.n_layers-1):
            self.weights[layer] -= self.lr[layer]*self.d_weights[layer].T
            self.biases[layer] -= self.lr[layer]*self.d_biases[layer].T
            
    def compute_network_energy(self):
        energy = 0
        for layer in range(0, self.n_layers-1):
            energy += np.sum(np.fabs(self.lr[layer]*self.d_weights[layer]))+np.sum(np.fabs(self.lr[layer]*self.d_biases[layer]))
        return energy
    
    def compute_layer_energy(self):
        energy = np.zeros(len(self.weights))
        for layer in range(0, self.n_layers-1):
            energy[layer] = np.sum(np.fabs(self.lr[layer]*self.d_weights[layer]))+np.sum(np.fabs(self.lr[layer]*self.d_biases[layer]))
        return energy
    
    # not sure how to define middle layer energy
    def compute_neuron_energy(self):
        energy_of_in = np.sum(np.fabs(self.lr[0]*self.d_weights[0]), axis=1)
        energy_of_out = np.sum(np.fabs(self.lr[-1]*self.d_weights[0]), axis=0) + np.sum(np.fabs(self.lr[-1]*self.d_biases), axis=0)
        return [energy_of_in, energy_of_out]
    
    def compute_synaspe_energy(self):
        energy = [np.zeros(w.shape) for w in self.weights]
        for layer in range(0, self.n_layers-1):
            energy[layer] = np.fabs(self.lr[layer]*self.d_weights[layer])
        return energy
            
    def compute_network_min_energy(self):
        min_energy = 0
        for layer in range(0, self.n_layers-1):
            min_energy += np.sum(np.fabs(self.weights[layer]-self.initial_weights[layer]))+np.sum(np.fabs(self.biases[layer]))
        return min_energy
    
    def count_synapses(self):
        n_synapse = np.zeros(len(self.weights))
        for layer in range(0, self.n_layers-1):
            n_synapse[layer] = np.count_nonzero(self.weights[layer])
        return n_synapse
            
    def train(self, n_epochs, x_train, y_train):
        n_samples = x_train.shape[0]
        errors = np.zeros(n_epochs)
        accuracies = np.zeros(n_epochs)
        energies = np.zeros(n_epochs)
        for epoch in range(0, n_epochs):
            error = 0
            accuracy = 0
            shuffled_idxs = np.random.permutation(n_samples)
            for i in range(0, n_samples):
                sample = shuffled_idxs[i]
                self.feedforward(x_train[sample]) # more explicit 
                self.backpropagate(x_train[sample], y_train[sample])
                self.update_weights()
                error += self.cross_entropy_loss(y_train[sample], self.axon[-1])
                accuracy += self.inference_score(y_train[sample], self.axon[-1])
                self.energy += self.compute_network_energy()
            errors[epoch] = error/n_samples
            accuracies[epoch] = (accuracy/n_samples)*100
            energies[epoch] = self.energy
            print("Epoch ", epoch+1, ": error = ", np.around(errors[epoch], 2), "| accuracy = ", np.around(accuracies[epoch], 2), "| Energy = ", np.around(energies[epoch], 2))
        return errors, accuracies, energies

    def train_and_test(self, n_epochs, x_train, y_train, x_test, y_test, test_interval):
        n_samples = x_train.shape[0]
        train_errors = np.zeros(n_epochs)
        train_accuracies = np.zeros(n_epochs)
        train_energies = np.zeros(n_epochs)
        test_errors = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1) # 
        test_accuracies = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1)
        test_energies = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1)
        min_energies = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1)
        samples_seen = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1)
        for epoch in range(0, n_epochs):
            print("Epoch: ", epoch+1)
            error = 0
            accuracy = 0
            shuffled_idxs = np.random.permutation(n_samples)
            for i in range(0, n_samples):
                sample = shuffled_idxs[i]
                self.feedforward(x_train[sample])
                self.backpropagate(x_train[sample], y_train[sample])
                self.update_weights()
                error += self.cross_entropy_loss(y_train[sample], self.axon[-1])
                accuracy += self.inference_score(y_train[sample], self.axon[-1])
                self.energy += self.compute_network_energy()
                if (epoch*n_samples)+i > 0 and not np.mod(i, test_interval):
                    j = int((n_samples/test_interval)*epoch+((i/test_interval)-1)) # 1D mapping of the test intervals within epochs
                    test_errors[j], test_accuracies[j] = self.evaluate_set(x_test, y_test)
                    test_energies[j] = self.energy
                    min_energies[j] = min_energies[j-1] + self.compute_network_min_energy()
                    samples_seen[j] = (epoch*n_samples)+i
                    print("Samples ", (epoch*n_samples)+i, ": error = ", np.around(test_errors[j], 2), 
                          "| accuracy = ", np.around(test_accuracies[j], 2), 
                          "| Energy = ", np.around(test_energies[j], 2))
            train_errors[epoch] = error/n_samples
            train_accuracies[epoch] = accuracy/n_samples*100
            train_energies[epoch] = self.energy
            #print("Epoch ", epoch+1, ": error = ", np.around(train_errors[epoch], 2), "| accuracy = ", np.around(train_accuracies[epoch], 2), "| Energy = ", np.around(train_energies[epoch], 2))
        return train_errors, train_accuracies, train_energies, test_errors, test_accuracies, test_energies, min_energies, samples_seen
            
    def evaluate_set(self, x_set, y_set):
        n_samples = x_set.shape[0]
        accuracy = 0
        error = 0
        for sample in range(0, n_samples):
            outs, acts = self.feedforward(x_set[sample])
            error += self.cross_entropy_loss(y_set[sample], acts[-1])
            accuracy += self.inference_score(y_set[sample], acts[-1])
        return error/n_samples, accuracy/n_samples*100
            
    
    def inference_score(self, y, y_hat):
        if np.argmax(y) == np.argmax(y_hat):
            return 1
        return 0
            
    def softmax(output):
        return np.exp(output - np.max(output))/np.sum(np.exp(output - np.max(output)))
    
    def cross_entropy_loss(self, y, y_hat):
        return -np.sum(y*np.log(y_hat+1e-10)) #NOTE: prevent log(0)  

def load_neural(data):
    net = Neural(data['network']['layers'], 
           Activation.Fn(data['network']['activation_function']), 
           np.asarray(data['network']['learning_rate']), 
           data['network']['p_connect'], 
           data['network']['bias'])
    net.energy = data['network']['energy']
    for layer in range(0, len(data['network']['weights'])):
        net.biases[layer] = np.asarray(data['network']['biases'][layer])
        for neuron in range(0, len(data['network']['weights'][layer])):
            net.weight_mask[layer][neuron] = np.asarray(data['network']['weight_mask'][layer][neuron])
            net.weights[layer][neuron] = np.asarray(data['network']['weights'][layer][neuron])
            net.initial_weights[layer][neuron] = np.asarray(data['network']['initial_weights'][layer][neuron])
    
    return net
        
    
    
    
    
    
    