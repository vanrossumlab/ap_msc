%matplotlib qt5
import matplotlib.pyplot as plt
import numpy as np
import DataManager

m = DataManager.load_data("test")

ax = plt.axes(projection='3d')
ax.plot3D(m['results']['error'], m['results']['accuracy'], m['results']['energy'])
ax.plot3D(m['results']['error'], m['results']['accuracy'], np.asarray(m['results']['samples_seen'])/5)
plt.legend(['energy', 'samples'])
plt.show()


