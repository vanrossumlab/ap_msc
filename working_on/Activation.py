import numpy as np

class Fn(): 
    def __init__(self, activation_function, ratio=0.0):
        
        self.fn_name = activation_function
        self.activation_function = self.name_to_number(activation_function.lower()) #string for desired 
        self.ratio = ratio
    def __call__(self, activation):
        if self.activation_function == 0:
            return self.relu(activation)
        elif self.activation_function == 1:
            return self.sigmoid(activation)
        elif self.activation_function == 2:
            return self.mixed_relu(activation)
        else:
            raise Exception("Invalid Activation Function")

    def name_to_number(self, activation_function_name):
        if activation_function_name == 'relu':
            return 0
        elif activation_function_name == 'sigmoid':
            return 1
        elif activation_function_name == 'mixed relu':
            return 2
        else:
            if type(activation_function_name) is int:
                return activation_function_name
            else:
                raise Exception("Invalid Activation Function")
    
    def prime(self, activation):
        if self.activation_function == 0:
            return self.relu_prime(activation)
        elif self.activation_function == 1:
            return self.sigmoid_prime(activation)
        elif self.activation_function == 2:
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