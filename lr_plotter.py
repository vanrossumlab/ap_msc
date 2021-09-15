from components import DataManager as dm
import numpy as np
import matplotlib.pyplot as plt

data = dm.load_data('data/lr_vs_p')

for i in range(0, len(data[0])):
    if data[0][i]['network_parameters']['p_connect'][0] == 0.5:
        plt.plot(data[0][i]['results']['accuracy'], data[0][i]['results']['energy'])
    
plt.show()