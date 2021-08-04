import numpy as np
import time
import Activation

class Neural():
    def __init__(self, layers, activation_function, learning_rate, p_connect=1, p_connect_skip=[1.0, 1.0], bias=True):
        self.layers = layers
        self.skip_collections = np.clip(np.arange(0, len(self.layers)-1), a_min=0, a_max=len(self.layers))
        self.n_layers = len(layers)
        self.n_labels = layers[-1]
        self.activation_function = activation_function
        self.lr = learning_rate
        self.weight_mask = self.initialise_weight_mask(p_connect)
        self.skip_masks = self.initialise_skip_masks(p_connect_skip)
        self.weights = self.initialise_weights(bias)
        self.skips = self.initialise_skips()
        self.biases = [np.zeros(b) for b in layers[1:]]
        self.dendrite = [np.zeros(n) for n in layers[1:]]
        self.skip_dendrites = self.initialise_skip_dendrites()
        self.axon = [np.ones(n) for n in layers]
        self.d_weights = [np.zeros(w) for w in layers[1:]]
        self.d_skips = self.initialise_d_skips()
        self.d_biases = [np.zeros(w) for w in layers[1:]]
        self.p_connect = p_connect
        self.bias = bias
        self.energy = 0
        
    def initialise_weight_mask(self, p_connect):
        weight_mask = [np.zeros((i, j)) for i, j in zip(self.layers[:-1], self.layers[1:])]
        for layer in range(0, self.n_layers-1):
            weight_mask[layer] = np.random.binomial(1, p_connect[layer], weight_mask[layer].shape)
        return weight_mask
    
    def initialise_skip_masks(self, p_connect_skip):
        i = 0
        skip_masks = []
        for layer in range(0, self.n_layers-2):
            skip_mask = []
            for skip in range(2+i, self.n_layers):
                skip_mask.append(np.random.randn(self.layers[layer], self.layers[skip])*np.sqrt(1/(self.layers[layer])))
            i = i + 1
            skip_masks.append(skip_mask)
        
        for layer in range(0, len(skip_masks)):
            for skip in range(0, len(skip_masks[layer])):
                skip_masks[layer][skip] = np.random.binomial(1, p_connect_skip[layer], skip_masks[layer][skip].shape)

        return skip_masks
        
    def initialise_weights(self, bias):
        weights = [np.random.randn(i, j)*np.sqrt(1/(i)) for i, j in zip(self.layers[:-1], self.layers[1:])]
        for layer in range(0, self.n_layers-1):
            weights[layer] = np.multiply(self.weight_mask[layer], weights[layer])
        return weights
    
    def initialise_skips(self):
        i = 0
        skips = []
        for layer in range(0, self.n_layers-2):
            skip_connections = []
            for skip in range(2+i, self.n_layers):
                skip_connections.append( np.random.randn(self.layers[layer], self.layers[skip])*np.sqrt(1/(self.layers[layer])))
            i = i + 1
            skips.append(skip_connections)
            
        for layer in range(0, len(self.skip_masks)):
            for skip in range(0, len(self.skip_masks[layer])):
                skips[layer][skip] = np.multiply(skips[layer][skip], self.skip_masks[layer][skip])
        return skips
    
    def reset_skips(self):
        i = 0 
        skips = []
        for layer in range(0, self.n_layers-2):
            skip_connections = []
            for skip in range(2+i, self.n_layers):
                skip_connections.append(np.zeros((self.layers[layer], self.layers[skip])))
            i = i+1
            skips.append(skip_connections)
        return skips
    
    def initialise_skip_dendrites(self):
        skip_dendrites = []
        for skip_layer in range(0, len(self.skips)):
            dendrites = []
            for skip in range(0, len(self.skips[skip_layer])):
                dendrites.append(np.zeros(np.shape(self.skips[skip_layer][skip])[1]))
            skip_dendrites.append(dendrites)
        return skip_dendrites
    
    def initialise_d_skips(self):
        d_skips = []
        for skip_layer in range(0, len(self.skips)):
            d_skip = []
            for skip in range(0, len(self.skips[skip_layer])):
                d_skip.append(np.zeros(np.shape(self.skips[skip_layer][skip])))
            d_skips.append(d_skip)
        return d_skips
            
    def feedforward(self, x):
        self.axon[0] = x
        for layer in range(0, len(self.weights)):
            for collect in range(0, self.skip_collections[layer]):
                self.skip_dendrites[collect][self.skip_collections[layer]-collect-1] = np.matmul(np.multiply(self.skips[collect][self.skip_collections[layer]-collect-1], self.skip_masks[collect][self.skip_collections[layer]-collect-1]).T, self.axon[collect])
                self.dendrite[layer] = self.skip_dendrites[collect][self.skip_collections[layer]-collect-1]
            self.dendrite[layer] = np.matmul(np.multiply(self.weights[layer], self.weight_mask[layer]).T, self.axon[layer]) + self.biases[layer]
            if layer == self.n_layers-2:
                self.axon[layer+1] = self.softmax(self.dendrite[layer])
            else:
                self.axon[layer+1] = self.activation_function(self.dendrite[layer])
        return self.dendrite, self.skip_dendrites, self.axon
    
    def backpropagate(self, x, y):
        error = self.axon[-1] - y
        self.d_weights[-1] = np.outer(error, self.axon[-2].T)
        self.d_weights[-1] = np.multiply(self.d_weights[-1], self.weight_mask[-1].T)
        for skip in range(0, self.skip_collections[-1]):
            self.d_skips[skip][-1] = np.outer(error, self.axon[skip])
            self.d_skips[skip][-1] = np.multiply(self.d_skips[skip][-1], self.skip_masks[skip][-1].T)
        if self.bias:
            self.d_biases[-1] = error
        for layer in reversed(range(0, len(self.weights)-1)):
            error = self.activation_function.prime(self.dendrite[layer])*np.matmul(np.multiply(self.weights[layer+1], self.weight_mask[layer+1]), error)
            self.d_weights[layer] = np.outer(error, self.axon[layer].T)
            self.d_weights[layer] = np.multiply(self.d_weights[layer], self.weight_mask[layer].T)
            if self.bias:
                self.d_biases[layer] = error
            return self.d_weights, self.d_biases
    
    def update_weights(self):
        for layer in range(0, self.n_layers-1):
            self.weights[layer] -= self.lr[layer]*self.d_weights[layer].T
            self.biases[layer] -= self.lr[layer]*self.d_biases[layer].T
        for layer in range(0, len(self.d_skips)):
            for skip in range(0, len(self.d_skips[layer])):
                self.skips[layer][skip] -= self.lr[layer]*self.d_skips[layer][skip].T
    
    def compute_network_energy(self):
        energy = 0
        for layer in range(0, self.n_layers-1):
            energy += np.sum(np.fabs(self.lr[layer]*self.d_weights[layer]))+np.sum(np.fabs(self.lr[layer]*self.d_biases[layer]))
        for layer in range(0, len(self.d_skips)):
            for skip in range(0, len(self.d_skips[layer])):
                energy += np.sum(np.fabs(self.lr[layer]*self.d_skips[layer][skip]))
        return energy
    
    def compute_network_min_energy(self):
        min_energy = 0
        for layer in range(0, self.n_layers-1):
            min_energy += np.sum(np.fabs(self.weights[layer]))+np.sum(np.fabs(self.biases[layer]))
        for layer in range(0, len(self.d_skips)):
            for skip in range(0, len(self.d_skips[layer])):
                min_energy += np.sum(np.fabs(self.skips[layer][skip]))
        return min_energy
    
    def train_and_test(self, n_epochs, x_train, y_train, x_test, y_test, test_interval):
        n_samples = x_train.shape[0]
        train_errors = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1)
        train_accuracies = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1)
        train_energies = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1)
        test_errors = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1) # 
        test_accuracies = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1)
        test_energies = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1)
        min_energies = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1)
        samples_seen = np.zeros(int(n_epochs*np.floor(n_samples/test_interval))-1)
        for epoch in range(0, n_epochs):
            start = time.time()
            print("Epoch: ", epoch+1)
            error = 0
            accuracy = 0
            shuffled_idxs = np.random.permutation(n_samples)
            for i in range(0, n_samples):
                sample = shuffled_idxs[i]
                self.feedforward(x_train[sample])
                if not self.inference_score(y_train[sample], self.axon[-1]): # or (self.inference_score(y_train[sample], self.axon[-1]) and np.max(self.axon[-1]) < threshold):
                    self.backpropagate(x_train[sample], y_train[sample])
                    self.update_weights()
                else:
                    self.d_weights = [np.zeros(w) for w in self.layers[1:]]
                    self.d_skips = self.reset_skips()
                    self.d_biases = [np.zeros(w) for w in self.layers[1:]]    
                error += self.cross_entropy_loss(y_train[sample], self.axon[-1])
                accuracy += self.inference_score(y_train[sample], self.axon[-1])
                self.energy += self.compute_network_energy()
                self.d_weights = [np.zeros(w) for w in self.layers[1:]]
                self.d_skips = self.reset_skips()
                self.d_biases = [np.zeros(w) for w in self.layers[1:]]
                if (epoch*n_samples)+i > 0 and not np.mod(i, test_interval):
                    j = int((n_samples/test_interval)*epoch+((i/test_interval)-1)) # 1D mapping of the test intervals within epochs
                    test_errors[j], test_accuracies[j] = self.evaluate_set(x_test, y_test)
                    test_energies[j] = self.energy
                    min_energies[j] = min_energies[j-1] + self.compute_network_min_energy()
                    samples_seen[j] = (epoch*n_samples)+i
                    train_errors[j], train_accuracies[j] = self.evaluate_set(x_train, y_train)
                    train_energies[j] = self.energy
                    print("TIme Taken: ", np.around(time.time()-start, 2))
                    print("Train - ", "Samples ", (epoch*n_samples)+i, ": error = ", np.around(train_errors[j], 2), 
                           "| accuracy = ", np.around(train_accuracies[j], 2), 
                           "| Energy = ", np.around(train_energies[j], 2))
                    print("Test -  ", "Samples ", (epoch*n_samples)+i, ": error = ", np.around(test_errors[j], 2), 
                          "| accuracy = ", np.around(test_accuracies[j], 2), 
                          "| Energy = ", np.around(test_energies[j], 2))
            #print("Epoch ", epoch+1, ": error = ", np.around(train_errors[epoch], 2), "| accuracy = ", np.around(train_accuracies[epoch], 2), "| Energy = ", np.around(train_energies[epoch], 2))
        return train_errors, train_accuracies, train_energies, test_errors, test_accuracies, test_energies, min_energies, samples_seen

    def softmax(self, output):
        return np.exp(output - np.max(output))/np.sum(np.exp(output - np.max(output)))
    
    def evaluate_set(self, x_set, y_set):
        n_samples = x_set.shape[0]
        accuracy = 0
        error = 0
        for sample in range(0, n_samples):
            outs, skips, acts = self.feedforward(x_set[sample])
            error += self.cross_entropy_loss(y_set[sample], acts[-1])
            accuracy += self.inference_score(y_set[sample], acts[-1])
        return error/n_samples, accuracy/n_samples*100
            
    
    def inference_score(self, y, y_hat):
        if np.argmax(y) == np.argmax(y_hat):
            return 1
        return 0
    
    def cross_entropy_loss(self, y, y_hat):
        return -np.sum(y*np.log(y_hat+1e-10)) #NOTE: prevent log(0)
        