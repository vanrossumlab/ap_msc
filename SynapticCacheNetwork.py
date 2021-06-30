#Author: Aaron Pache
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

"""
TODO:
    1. Generalisations and clean up (i.e. account for different datasets)
    2. Optional Consolidation
    3. General Neuron Tracker (i.e. track weight changes for each neuron across training)? (Unsure if this is computationally good)
    4. Biological activation functions (leaky-ReLU, GeLU, ELU)
    5. Add CNN kernel layers
"""

"""
    SynapticCacheNetwork:
        Description: Implements feedforward, backproagation and training for
                     arbitrary-sized and biologically-inspired neural networks.
                     Artficial weights are distributed between early and 
                     late Long Term Potentiation weights (e-LTP/l-LTP weights) 
                     after a mini-batch of training, these handle the more 
                     biologically specific parameters of synaptic plasticity,
                     like decay and consolidation.
                     
                     In particular, this network tracks the energy usage of 
                     e-LTP up-keep and l-LTP synaptic changes for various 
                     consolidation schemes and parameters for biological
                     networks, under the assumption that energy usage is 
                     proportional to the change in synaptic strength (l-LTP) 
                     or the weight itself (e-LTP). 
                     
                     It is intended as a general framework for the 
                     investigation of neural network energy usage, which will
                     hopefully expand to include multiple activation functions
                     network architectures and learning rules. Along with 
                     general improvements to learning time and usability.

        Initialisation:
           layers:                      a python list containing the number of 
                                        neurons in each layer (i.e [784 50 10] 
                                        describes a neural network with 3
                                        layers containing 784 input neurons, 
                                        50 hidden and 10 output neurons.)
                                        
           activation_function:         An object that implements the activation
                                        function and its derivitive used in the
                                        neural netowork. See "Activation" for
                                        more info.
                    
           learning_rate:               The learning rate of the network. 
                                        Dicates the step size of weight changes 
                                        in the weight space. Generally, between 
                                        1 and 0.0001 depending on network 
                                        specifications. Larger learning rates 
                                        decrease learning time, but may 
                                        overshoot minimas leading to 
                                        instability. Smaller learning rates 
                                        take longer for training, but are more 
                                        stable in their approach to the minima.
                            
            eLTP_cost:                  The energy penalty for e-LTP weight 
                                        maintanence applied to the weight 
                                        itself. It is generally thought that 
                                        e-LTP or transient synaptic changes 
                                        consume less energy than l-LTP changes.
            
            decay_rate:                 The rate at which eLTP weights decay. 
                                        Unlike computers, biological systems 
                                        are afflicted with neural noise and 
                                        "wear and tear" which leads to 
                                        forgetting. Decay is a means of 
                                        simulating this process.
                            
            consolidation_scheme:       The method of applying e-LTP weights 
                                        to l-LTP weights. When sypantic 
                                        changes pass a specified threshold, 
                                        e-LTP weight changes are consolidated 
                                        into the persistant l-LTP form. This 
                                        can happen under multiple threshold 
                                        conditions: a single synapse surpassing 
                                        the threshold, a neuron, a layer, 
                                        a network or other conditions (maybe 
                                        even dynamically adjusting between 
                                        thresholds or schemes.) See 
                                        "Consolidation()" for more info. There
                                        are currently 8 implementations.
                                  
            consolidation_threshold:    The threshold to consolidate 
                                        the e-LTP weight changes into the 
                                        l-LTP weight. Generally, between 
                                        0.001 and 0.01.                      
"""

class SynapticCacheNetwork():
    def __init__(self, layers, activation_function, learning_rate, connection_chance=1, eLTP_cost=0, decay_rate=0, consolidation_scheme=1, consolidation_threshold=1):
        self.n_layers = len(layers)
        self.layers = layers
        self.n_labels = layers[-1]
        #Xavier initialisation
        self.weights = [np.random.randn(i, j)*np.sqrt(1/(i)) for i, j in zip(layers[:-1], layers[1:])]
        self.weights_eLTP = [np.zeros(w.shape) for w in self.weights]
        self.weights_lLTP = [np.zeros(w.shape) for w in self.weights]
        self.biases = [np.zeros(b) for b in layers[1:]]
        self.weight_mask = [np.random.binomial(1, connection_chance, (w.shape)) for w in self.weights]
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.eLTP_cost = eLTP_cost
        self.decay = 1-decay_rate
        self.consolidation_scheme = consolidation_scheme
        self.consolidation_threshold = consolidation_threshold
        self.energy_control = [np.zeros(w.shape) for w in self.weights_eLTP]
        self.energy_eLTP = [np.zeros(w.shape) for w in self.weights_eLTP] # store on a per synpase basis, we can compute synapse, neuron, layer energies 
        self.energy_lLTP = [np.zeros(w.shape) for w in self.weights_eLTP] 
        self.energy_bias = [np.zeros(b.shape) for b in self.biases] # store bias energies
        self.energy_min = [np.zeros(w.shape) for w in self.weights_eLTP]
        #Track synaptic changes to be made, prevents multiple initialisations, which may be expensive
        self.synapses_to_update_lLTP = [np.zeros(w.shape) for w in self.weights]
        self.neurons_to_update_lLTP = [np.zeros(n) for n in layers[1:]] # consolidation happens to the inputs of neurons, dendritic if you will
        self.layers_to_update_lLTP = np.zeros(np.shape(layers), dtype=bool)
        self.network_update = False
        
    """
        feedforward():
            Description: Gets the network output for a given input.
            
            Arguments:
                output: the input to the neural network, unfortunately named so 
                        so that it can be used in a loop. An array.
            
            Returns:
                activations: The synpatic inputs before passing the activation 
                             function. A python list of numpy arrays.                  
                outputs:     The ouput of the neuron, having passed the activation
                             function. A python list of numpy arrays.
    """
    def feedforward(self, output):
        outputs = [output]
        activations = []
        layer = 1 # avoid from count from 0 error (i think if we start from 0 then it assumes synapses into the input)
        for b, w, m in zip(self.biases, self.weights, self.weight_mask):
            activation = np.matmul(np.multiply(w, m).T, output) + b
            #softmax
            if layer == self.n_layers-1: # again no synapses at the output, final layer of synapses between last hidden and final layer 
                output = self.softmax(activation)
            else:
                output = self.activation_function(activation)
            activations.append(activation)
            outputs.append(output)
            layer = layer + 1
        return activations, outputs
    
    """
        backpropagate():
            Description: The workhorse of modern day neural networks. 
                         Determines the synaptic changes required of a neural
                         network to minimise its objective function, by 
                         tracing the gradient of weight changes with respect 
                         to the objective function.
            
            Parameters: 
                stimulus: The input of the neural network. Backpropagation 
                          requires first a feedforward-pass to determine the 
                          network output and thus error. backpropagate() 
                          performs a feedforward to get the errors associated 
                          with the input or "stimulus". An array.
                          
                target:   The desired output of the network, used to get the 
                          error of the network. An array.
                          
            Returns:
                dWeights: The change in weights to be applied to the network. 
                          A python list of numpy arrays, each array containing
                          the synaptic changes at each layer.
                
                dBiases: The change in biases to be applied to the network.
                         Similarly, to dWeights, a python list of numpy arrays.
                
                outputs: The neuron outputs at each layer. A python list of 
                         numpy arrays.
    """
    
    def backpropagate(self, stimulus, target):
        dWeights = [np.zeros(w.shape) for w in self.weights]
        dBiases = [np.zeros(b.shape) for b in self.biases]
        activations, outputs = self.feedforward(stimulus)
        error = outputs[-1] - target
        dWeights[-1] = np.outer(error, outputs[-2].T) # seems to prefer outer for (n,) vectors, matmul apparently attempts to append (n, 1) which might be failing
        dBiases[-1] = error
        for layer in range(2, self.n_layers):
            error = self.activation_function.prime(activations[-layer])*np.matmul(self.weights[-layer+1], error)
            dWeights[-layer] = np.outer(error, outputs[-layer-1].T)
            dBiases[-layer] = error
        return dWeights, dBiases, outputs
    
    def predict(self, stimulus):
        activations, outputs = self.feedforward(stimulus)
        return np.argmax(outputs[-1])
    
    """
        consolidate():
            Description: Implements the biological side of the neural network. 
                         Specifically, the consolidation scheme used by the 
                         network to update the l-LTP weights, the decay of 
                         e-LTP weights and energy usage of the e-LTP and l-LTP 
                         weights. Consolidation occurs when the weight changes
                         surpass a consolidation threshold. (Note: All 
                         thresholds are normalised to the single synapse level.)
                         
                Scheme:                
                    1: Immediately consolidate, classic learning.
                    
                    2: Synapse theshold/synapse consolidation -
                        Consolidate a single synapse if the synaptic changes 
                        of that synapse passes the threshold.
                        
                    3: Synapse theshold/neuron consolidation -
                        Consolidate the neuron's input synapses if a single
                        input synapse's synpatic changes passes the threshold.
                        
                    4: Neuron theshold/neuron consolidation -
                        Consolidate all of a neuron's synapses if the mean of 
                        synpatic changes passes the threshold. 
                        
                    5: Neuron theshold/layer consolidation -
                        Consolidate the layer of synpases, if the mean of 
                        synaptic changes for a neuron in that layer passes the
                        threshold. 
                        
                    6: Layer theshold/layer consolidation - 
                        Consolidate the layer of synapses, if the mean of 
                        synpatic changes in that layer passes the threshold.
                        
                    7: Layer theshold/network consolidation - 
                        Consolidate the all synapses, if the mean of synaptic
                        changes in a layer passes the threshold. 
                    
                    8: Network theshold/network consolidation.
                        Consolidate all synapses in the mean of synaptic 
                        changes across the network pass the threshold.
                        
                    9: Consolidate based on accuracy (dopamine?) - 
                        An experimental scheme intended to investigate 
                        consolidation based on accuracy. (i.e. consolidate if 
                        the mini-batch of predictions was mostly correct. 
                        Consolidate if correct, in other words.)
                        
            Arguments:
                dWeights:        The weight changes obtained from backpropagation to 
                                 be applied to the network.
                          
                dBiases:         The bias changes to be applied.
                
                score_threshold: Used in an experimental consolidation scheme
                                 that consolidates depending on how many 
                                 predictions were correct.
                          
    """
    
    def consolidate(self, dWeights, dBiases, score_threshold):
        if (self.decay != 1) and (self.consolidation_scheme != 1):
            for layer in range(len(self.weights_eLTP)):
                self.weights_eLTP[layer] = self.decay*self.weights_eLTP[layer]
                
        for layer in range(len(self.weights_eLTP)):
            self.weights_eLTP[layer] -= self.learning_rate*dWeights[layer].T
            self.weights[layer] = (self.weights_eLTP[layer]+self.weights_lLTP[layer])
            self.biases[layer] -= self.learning_rate*dBiases[layer].T
            self.energy_control[layer] += np.fabs(self.learning_rate*dWeights[layer].T)
            self.energy_eLTP[layer] += self.eLTP_cost*np.fabs(self.weights_eLTP[layer])    #dWeights[layer].T) :(
            self.energy_bias[layer] += np.fabs(self.learning_rate*dBiases[layer])
     
        if self.consolidation_scheme == 1: # no caching, classic learning
            for layer in range(len(self.weights_eLTP)):
                self.weights_lLTP[layer] += self.weights_eLTP[layer] # a painful error to catch, negative gradient is already applied to the weights
                self.energy_lLTP[layer] = self.energy_control[layer]
                self.energy_eLTP[layer] = np.zeros(np.shape(self.weights_eLTP[layer]))
                self.weights_eLTP[layer] = np.zeros(np.shape(self.weights_eLTP[layer]))
        
        elif self.consolidation_scheme == 2: # synapse threshold, synapse consolidation  
            for layer in range(len(self.weights_eLTP)):
                self.synapses_to_update_lLTP[layer] = np.fabs(self.weights_eLTP[layer]) > self.consolidation_threshold
                self.weights_lLTP[layer][self.synapses_to_update_lLTP[layer]] += self.weights_eLTP[layer][self.synapses_to_update_lLTP[layer]]
                self.energy_lLTP[layer][self.synapses_to_update_lLTP[layer]] += np.fabs(self.weights_eLTP[layer][self.synapses_to_update_lLTP[layer]])
                self.weights_eLTP[layer] = np.where(np.fabs(self.weights_eLTP[layer]) > self.consolidation_threshold, 0, self.weights_eLTP[layer])
                #print(self.synapses_to_update_lLTP[layer])
                
          # for layer in range(len(self.weights_eLTP)):
          #     for neuron in range(len(self.weights_eLTP[layer])):
          #         for synapse in range(len(self.weights_eLTP[layer][neuron])):
          #              #print("layer: ", layer, " | ", "neuron: ", neuron, " | ", "synapse: ", synapse)
          #              if np.fabs(self.weights_eLTP[layer][neuron][synapse]) > self.consolidation_threshold:
          #                  self.weights_lLTP[layer][neuron][synapse] -= self.weights_eLTP[layer][neuron][synapse]
          #                  self.energy_lLTP[layer][neuron][synapse] += self.weights_eLTP[layer][neuron][synapse]
          #                  self.weights_eLTP[layer][neuron][synapse] = 0
            
        elif self.consolidation_scheme == 3: # synapse threshold, neuron consolidation
            for layer in range(len(self.weights_eLTP)):
                self.synapses_to_update_lLTP[layer] = np.fabs(self.weights_eLTP[layer]) > self.consolidation_threshold
                self.neurons_to_update_lLTP[layer] = np.any(self.synapses_to_update_lLTP[layer], axis=1) #sum of inputs (make sure dims are good)
                self.weights_lLTP[layer][self.neurons_to_update_lLTP[layer].T] += self.weights_eLTP[layer][self.neurons_to_update_lLTP[layer].T]
                self.energy_lLTP[layer][self.neurons_to_update_lLTP[layer].T] += np.fabs(self.weights_eLTP[layer][self.neurons_to_update_lLTP[layer].T])
                self.weights_eLTP[layer][self.neurons_to_update_lLTP[layer].T] = 0
                
        elif self.consolidation_scheme == 4: # neuron threshold, neuron consolidation
            for layer in range(len(self.weights_eLTP)):
                self.neurons_to_update_lLTP[layer] = np.mean(np.fabs(self.weights_eLTP[layer]), axis=1) > self.consolidation_threshold
                self.weights_lLTP[layer][self.neurons_to_update_lLTP[layer].T] += self.weights_eLTP[layer][self.neurons_to_update_lLTP[layer].T]
                self.energy_lLTP[layer][self.neurons_to_update_lLTP[layer].T] += np.fabs(self.weights_eLTP[layer][self.neurons_to_update_lLTP[layer].T])
                self.weights_eLTP[layer][self.neurons_to_update_lLTP[layer].T] = 0
                
        elif self.consolidation_scheme == 5: # neuron threshold, layer consolidation
            for layer in range(len(self.weights_eLTP)):
                self.neurons_to_update_lLTP[layer] = np.mean(np.fabs(self.weights_eLTP[layer]), axis=0) > self.consolidation_threshold
                self.layers_to_update_lLTP[layer] = np.any(self.neurons_to_update_lLTP[layer], axis=0)
                if self.layers_to_update_lLTP[layer]:
                    self.weights_lLTP[layer] += self.weights_eLTP[layer]
                    self.energy_lLTP[layer] += np.fabs(self.weights_eLTP[layer])
                    self.weights_eLTP[layer] = np.zeros(np.shape(self.weights_eLTP[layer]))
                
        elif self.consolidation_scheme == 6: # layer threshold, layer consolidation
            for layer in range(len(self.weights_eLTP)):
                self.layers_to_update_lLTP[layer] = np.mean(np.fabs(self.weights_eLTP[layer])) > self.consolidation_threshold # mean of all synapses 
                if self.layers_to_update_lLTP[layer]:
                    self.weights_lLTP[layer] += self.weights_eLTP[layer]
                    self.energy_lLTP[layer] += np.fabs(self.weights_eLTP[layer])
                    self.weights_eLTP[layer] = np.zeros(np.shape(self.weights_eLTP[layer]))
                
        elif self.consolidation_scheme == 7: # layer threshold, network consolidation
            for layer in range(len(self.weights_eLTP)):
                self.layers_to_update_lLTP[layer] = np.mean(np.fabs(self.weights_eLTP[layer])) > self.consolidation_threshold
                if self.layers_to_update_lLTP[layer]:
                    self.network_update = True
                    break
            if self.network_update:
                for layer in range(len(self.weights_eLTP)):
                    self.weights_lLTP[layer] += self.weights_eLTP[layer]
                    self.energy_lLTP[layer] += np.fabs(self.weights_eLTP[layer])
                    self.weights_eLTP[layer] = np.zeros(np.shape(self.weights_eLTP[layer]))

        elif self.consolidation_scheme == 8: # network threshold, network consolidation
            network_mean = 0
            for layer in range(len(self.weights_eLTP)):
                network_mean += np.mean(np.fabs(self.weights_eLTP[layer]))/(self.n_layers-1) # synapse layers
                if network_mean > self.consolidation_threshold:
                    self.network_update = True
                    break
            if self.network_update:
                for layer in range(len(self.weights_eLTP)):
                    self.weights_lLTP[layer] += self.weights_eLTP[layer]
                    self.energy_lLTP[layer] += np.fabs(self.weights_eLTP[layer])
                    self.weights_eLTP[layer] = np.zeros(np.shape(self.weights_eLTP[layer]))
            
        elif self.consolidation_scheme == 9: #dopamine consolidation? (un-used for assignment, personally interested though)
            if score_threshold:
                for layer in range(len(self.weights_eLTP)):
                    self.weights_lLTP[layer] += self.weights_eLTP[layer]
                    self.energy_lLTP[layer] += np.fabs(self.weights_eLTP[layer])
                    self.weights_eLTP[layer] = np.zeros(np.shape(self.weights_eLTP[layer]))
                    
        else:
            raise Exception("No Such Consolidation Scheme")
        
        self.reset_update_lLTPs()
    
    def reset_update_lLTPs(self): #possibly redundant but also a fail-safe
        self.synapses_to_update_lLTP = [np.zeros(w.shape) for w in self.weights]
        self.neurons_to_update_lLTP = [np.zeros(n) for n in self.layers[1:]]
        self.layers_to_update_lLTP = np.zeros(np.shape(self.layers), dtype=bool)
        self.network_update = False
        
    def compute_network_energy_min(self): 
        energy = 0
        for layer in range(len(self.weights_eLTP)):
            energy += np.sum(self.energy_min[layer])
        return energy
    
    # The total energy of the network
    def compute_network_energy(self):
        energy = 0
        for layer in range(len(self.energy_eLTP)):
            energy += np.sum(self.energy_eLTP[layer]+self.energy_lLTP[layer])
            energy += np.sum(self.energy_bias[layer])
        return energy
    
    # The energy of input and output neurons (used for visualisation purposes.)
    def compute_io_neuron_energy(self): #I'd wanna make this layer-wise energy, but there's difficulty interpreting middle layers
        energy_of_in = np.sum(self.energy_eLTP[0], axis=1) + np.sum(self.energy_lLTP[0], axis=1)
        energy_of_out = np.sum(self.energy_eLTP[-1], axis=0) + np.sum(self.energy_lLTP[-1], axis=0) + np.sum(self.energy_bias[-1])
        return energy_of_in, energy_of_out
    
    def compute_io_neuron_energy_min(self):
        energy_of_in = np.sum(self.energy_min[0], axis=1)
        energy_of_out = np.sum(self.energy_min[-1], axis=0)
        return energy_of_in, energy_of_out
    
    # Visualise the input neuron energy (typically associated with some pixels
    # or features)
    def plot_eLTP_pixel_energy(self, dimensions):
        energy = np.sum(self.energy_eLTP[0], axis=1)
        energy = np.reshape(energy, dimensions)
        plt.imshow(energy)
        plt.show()
        
    def plot_lLTP_pixel_energy(self, dimensions):
        energy = np.sum(self.energy_lLTP[0], axis=1)
        energy = np.reshape(energy, dimensions)
        plt.imshow(energy)
        plt.show()
    
    def plot_pixel_energy(self, dimensions):
        energy = np.sum(self.energy_eLTP[0], axis=1) + np.sum(self.energy_lLTP[0], axis=1)
        energy = np.reshape(energy, dimensions)
        plt.imshow(energy)
        plt.show()
    
    # Plot the e-LTP, l-LTP and bias energy contributions to the output neurons.
    def plot_output_energy_contributions(self):
        lLTP_contribution = np.sum(self.energy_lLTP[-1], axis=0)
        eLTP_contribution = np.sum(self.energy_eLTP[-1], axis=0)
        bias_contribution = np.sum(self.energy_bias[-1])
        energy = [lLTP_contribution, eLTP_contribution, bias_contribution]
        labs = ['lLTP', 'eLTP', 'bias']
        figs, ax = plt.subplots()
        X = np.arange(lLTP_contribution.shape[0])
        for i in range(0, 3):
            plt.bar(X, energy[i], bottom = np.sum(energy[:i], axis=0), label=labs[i])
        ax.legend()
        plt.show()
    
    def train(self, x_train, y_train, n_epochs, desired_batch_size):
        errors = np.zeros((n_epochs,))
        accuracies = np.zeros((n_epochs,))
        energies = np.zeros((n_epochs,))
        n_batches = int(np.ceil(x_train.shape[0]/desired_batch_size))
        for epoch in range(0, n_epochs):
            shuffled_indexes = np.random.permutation(x_train.shape[0])
            for batch in range(0, n_batches):
                if batch == n_batches-1:
                    batch_size = int(x_train.shape[0]-np.floor(x_train.shape[0]/desired_batch_size)*desired_batch_size)
                else:
                    batch_size = desired_batch_size
                accumulated_dWeights = [np.zeros(w.shape).T for w in self.weights_eLTP]
                accumulated_dBiases = [np.zeros(b.shape).T for b in self.biases]
                score = 0
                score_threshold = False
                for index in range(0, batch_size):
                    sample = shuffled_indexes[batch*batch_size + index]
                    x = x_train[sample]
                    y = y_train[sample]
                    dWeights, dBiases, outputs = self.backpropagate(x, y)
                    for layer in range(len(accumulated_dWeights)):
                        accumulated_dWeights[layer] += dWeights[layer]/batch_size
                        accumulated_dBiases[layer] += dBiases[layer]/batch_size
                    errors[epoch] += self.cross_entropy_Loss(outputs[-1], y)/x_train.shape[0]
                    accuracies[epoch] += (self.prediction_score(y, outputs[-1])/y_train.shape[0])*100
                    score += self.prediction_score(y, outputs[-1])
                if batch_size != 0:
                    if score == batch_size/2: # hopefully investigate dopamine-like consolidation
                        score_threshold = True
                self.consolidate(accumulated_dWeights, accumulated_dBiases, score_threshold) # we need to apply accumulated gradients, that was painful
                energies[epoch] = self.compute_network_energy()
            print("Epoch ", epoch+1, ": error = ", np.around(errors[epoch], 4), "| accuracy = ", np.around(accuracies[epoch], 2), "| Energy = ", np.around(energies[epoch], 2))
        for layer in range(len(self.weights_eLTP)):
            self.energy_min[layer] = np.fabs(self.weights_lLTP[layer])
        return accuracies, energies
        # plt.plot(errors) #Mostly for debugging purposes
        # plt.xlabel('Epoch')
        # plt.ylabel('Error')
        # plt.title('Average error per epoch')
        # plt.show()
        # plt.plot(accuracy)
        # plt.show()
        
    def train_until(self, desired_accuracy, x_train, y_train, desired_batch_size):
        accuracy = 0
        epochs = 0
        energies = []
        accuracies = []
        while accuracy < desired_accuracy:
            accuracy, energy = self.train(x_train, y_train, 1, desired_batch_size)
            energies.append(energy[0])
            accuracies.append(accuracy[0])
            epochs = epochs + 1
        return epochs, accuracies, energies
    
    # should probably be renamed train_until_patience_breaks...?
    def train_until_best(self, moving_average_window, x_train, y_train, desired_batch_size):
        accuracy = 0
        epochs = 0
        energies = []
        accuracies = []
        moving_average_accuracy = np.array([0])
        best_found = False
        while not best_found:
            accuracy, energy = self.train(x_train, y_train, 1, desired_batch_size)
            energies.append(energy[0])
            accuracies.append(accuracy[0])           
            if epochs < moving_average_window+moving_average_window/2:
                moving_average_accuracy = np.append(moving_average_accuracy, 0)
                epochs = epochs + 1
                continue
            else:
                moving_average_accuracy = np.append(moving_average_accuracy, self.move_mean(accuracies, moving_average_window)[-1])
            if moving_average_accuracy[-1] - 0.005 <= moving_average_accuracy[int(-np.ceil(moving_average_window/2))]: #intial threshold was 0.25
                best_found = True
            epochs = epochs + 1
        return epochs, accuracies, energies
    
    def train_until_or_best(self, moving_average_window, desired_accuracy, x_train, y_train, desired_batch_size):
        accuracy = 0
        epochs = 0
        energies = []
        accuracies = []
        moving_average_accuracy = np.array([0])
        best_found = False
        while not best_found:
            accuracy, energy = self.train(x_train, y_train, 1, desired_batch_size)
            energies.append(energy[0])
            accuracies.append(accuracy[0])           
            if epochs < moving_average_window+moving_average_window/2:
                moving_average_accuracy = np.append(moving_average_accuracy, 0)
                if accuracy[0] > desired_accuracy:
                    best_found = True
                epochs = epochs + 1
                continue
            else:
                moving_average_accuracy = np.append(moving_average_accuracy, self.move_mean(accuracies, moving_average_window)[-1])
            if moving_average_accuracy[-1] - 0.25 <= moving_average_accuracy[int(-np.ceil(moving_average_window/2))]:
                best_found = True
            if accuracy[0] > desired_accuracy:
                best_found = True
            epochs = epochs + 1
        return epochs, accuracies, energies

    def evaluate(self, x_test, y_test):
        accuracy = 0
        error = 0
        for sample in range(x_test.shape[0]):
            activations, outputs = self.feedforward(x_test[sample])
            accuracy += (self.prediction_score(y_test[sample], outputs[-1])/y_test.shape[0])*100
            error += self.cross_entropy_Loss(outputs[-1], y_test[sample])/x_test.shape[0]
        return accuracy, error
            
    def prediction_score(self, y, prediction):
        if np.argmax(y) == np.argmax(prediction):
            return 1
        return 0
                        
    def cross_entropy_Loss(self, prediction, target):
        return -np.sum(target*np.log(prediction+1e-10)) #NOTE: prevent log(0)  
    
    def softmax(self, output):
        return np.exp(output - np.max(output))/np.sum(np.exp(output - np.max(output)))
    
    def move_mean(self, signal, window):
        signal = np.cumsum(signal)
        signal[window:] = signal[window:] - signal[:-window]
        return signal[window - 1:]/window
"""
    Activation:
        Description: Implements the activation function and its derivitives.
                     Currently, supports ReLU and Sigmoid.        
        Initialisation:
            activation_function: The desired activation function. A string.                     
"""
class Activation(): 
    def __init__(self, activation_function, ratio=0.0):
        self.activation_function = activation_function.lower() #string for desired 
        self.ratio = ratio
    def __call__(self, activation):
        if self.activation_function == 'relu':
            return self.relu(activation)
        elif self.activation_function == 'sigmoid':
            return self.sigmoid(activation)
        elif self.activation_function == 'mixed relu':
            return self.mixed_relu(activation)
        else:
            raise Exception("Invalid Activation Function")
               
    def prime(self, activation):
        if self.activation_function == 'relu':
            return self.relu_prime(activation)
        elif self.activation_function == 'sigmoid':
            return self.sigmoid_prime(activation)
        elif self.activation_function == 'mixed relu':
            return self.mixed_relu_prime(activation)
        else:
            raise Exception("Invalid Activation Function")
                
    def sigmoid(self, activation):
        return 1.0/(1.0+np.exp(-activation))

    def sigmoid_prime(self, activation):
        return self.sigmoid(activation)*(1-self.sigmoid(activation))

    def relu(self, activation):
        return np.maximum(activation, 0)
    
    def relu_prime(self, activation):
        return np.where(activation > 0, 1.0, 0.0)
    
    def relu_negative(self, activation):
        return np.minimum(activation, 0.0)
    
    def relu_negative_prime(self, activation):
        return np.where(activation < 0, 1.0, 0.0)
    
    def mixed_relu(self, activation):
        inhibitory = int(np.shape(activation)[0]*self.ratio)
        inhibit = self.relu_negative(activation[:inhibitory])
        excite = self.relu(activation[inhibitory:])
        return np.concatenate((inhibit, excite))
    
    def mixed_relu_prime(self, activation):
        inhibitory = int(np.shape(activation)[0]*self.ratio)
        inhibit = self.relu_negative_prime(activation[:inhibitory])
        excite = self.relu_prime(activation[inhibitory:])
        return np.concatenate((inhibit, excite))

"""
    load_mnist():
        Description: Used to select the MNIST data loading method. Keras is 
                     much quicker but depends on it being installed. Otherwise
                     it will load text files that can be created from the MNIST
                     data files (http://yann.lecun.com/exdb/mnist/) using the 
                     data converter contained in this repository, 
                     convert_mnist.py.  
"""

def load_mnist(using_keras=True):
    if using_keras: # quick load
        (x_train, y_train_labels), (x_test, y_test_labels) = mnist.load_data()
        x_train = np.reshape(x_train, (60000, 784))/255
        x_test = np.reshape(x_test, (10000, 784))/255
        n_samples, img_size = x_train.shape
        n_labels = 10   
        y_train = np.zeros((y_train_labels.shape[0], n_labels))
        y_test  = np.zeros((y_test_labels.shape[0], n_labels))
        for i in range(0,y_train_labels.shape[0]):   
            y_train[i, y_train_labels[i].astype(int)]=1
            
        for i in range(0,y_test_labels.shape[0]):    
            y_test[i, y_test_labels[i].astype(int)]=1  
    else:   # takes ages to load  
        x_train = np.loadtxt('mnist/train-images.idx3-ubyte.txt')
        x_train = x_train/255 #rescale between 0 and 1
        train_labels = np.loadtxt('mnist/train-labels.idx1-ubyte.txt')
        x_test = np.loadtxt('mnist/t10k-images.idx3-ubyte.txt')
        x_test = x_test/255
        test_labels = np.loadtxt('mnist/t10k-labels.idx1-ubyte.txt')    
        n_samples, img_size = x_train.shape
        n_labels = 10     
        y_train = np.zeros((train_labels.shape[0], n_labels))
        y_test  = np.zeros((test_labels.shape[0], n_labels))
        #One-hot vectors
        for i in range(0,train_labels.shape[0]):   
            y_train[i, train_labels[i].astype(int)]=1
            
        for i in range(0,test_labels.shape[0]):    
            y_test[i, test_labels[i].astype(int)]=1
    return x_train, y_train, x_test, y_test, n_samples, n_labels, img_size
