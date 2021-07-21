%matplotlib qt5
import matplotlib.pyplot as plt
import numpy as np
import DataManager

m = DataManager.load_data("varying_lr_and_connection")

labels = []
for lr in range(1, len(m['simulation'])):
    for p in range(len(m['simulation'][lr])):
        prb = m['simulation'][lr][p]['network_parameters']['p_connect'][0]
        learn = m['simulation'][lr][p]['network_parameters']['learning_rate'][0]
        if prb == 1.0:
            c = [np.abs(np.log10(learn))*0.25, 0, 0]
        elif prb == 0.5:
            c = [0, np.abs(np.log10(learn))*0.25, 0]
        else:
            c = [0, 0, np.abs(np.log10(learn))*0.25]
        plt.plot(m['simulation'][lr][p]['results']['accuracy'], m['simulation'][lr][p]['results']['energy'], color=c)
        labels.append("lr: " + str(m['simulation'][lr][p]['network_parameters']['learning_rate']) + " | p: " + str(m['simulation'][lr][p]['network_parameters']['p_connect']))
plt.legend(labels)
plt.show()

# ax = plt.axes(projection='3d')
# ax.plot3D(m['results']['error'], m['results']['accuracy'], m['results']['energy'])
# ax.plot3D(m['results']['error'], m['results']['accuracy'], np.asarray(m['results']['samples_seen'])/5)
# plt.legend(['energy', 'samples'])
# plt.show()


