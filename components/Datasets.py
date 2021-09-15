from tensorflow.keras.datasets import mnist
import numpy as np

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