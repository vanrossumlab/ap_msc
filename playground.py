"""
    Used to test certain configurations.
"""
# profile code  "python3 -m cProfile  -s tottime file.py"
# possibly use parity problem for learning "Hertz  Krogh & Palmer book: parity problem."

#%% Imports
import sys
import keras
sys.path.append("..")
from SynapticCacheNetwork import SynapticCacheNetwork, Activation, load_mnist
import numpy as np
import matplotlib.pyplot as plt
import time

#%% Data loading
x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = load_mnist(True)

# (x_train, y_train_labels), (x_test, y_test_labels) = keras.datasets.cifar10.load_data()
# x_train = np.reshape(x_train, (50000, 3072))/255
# x_test = np.reshape(x_test, (10000, 3072))/255
# n_samples, img_size = x_train.shape
# n_labels = 10
# y_train = np.zeros((y_train_labels.shape[0], n_labels))
# y_test  = np.zeros((y_test_labels.shape[0], n_labels))
# for i in range(0,y_train_labels.shape[0]):   
#     y_train[i, y_train_labels[i].astype(int)]=1
    
# for i in range(0,y_test_labels.shape[0]):    
#     y_test[i, y_test_labels[i].astype(int)]=1  
#%% Network profile

np.random.seed(1)
connection_chance = 1
n_epochs = 5
n_samples_in_batch = 1

 # p = cProfile.Profile()
# p.enable()
start = time.time()
network = SynapticCacheNetwork([img_size, 100, n_labels], Activation("sigmoid"), 0.001, connection_chance, 0.001, 0, 1, 0.0007849108367626886)
#%% Training
accuracies, energies = network.train(x_train, y_train, n_epochs, n_samples_in_batch)
end = time.time()
print(end-start)
# p.disable()
# p.print_stats()

# np.random.seed(1)
# network = SynapticCacheNetwork([img_size, 188, 188, n_labels], Activation("relu"), 0.01, 0.001, 0, 1, 0.0007849108367626886)
# epochs, accuracies, energies = network.train_until_best(5, x_train, y_train, n_samples_in_batch)