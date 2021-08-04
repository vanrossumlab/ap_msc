
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import DataManager as dm


base_path = "data/n_hid_units_vs_p_connect/"
p1 = dm.load_data(base_path + "hidden_units_vs_connection_p1")
p2 = dm.load_data(base_path + "hidden_units_vs_connection_p2")
p3 = dm.load_data(base_path + "hidden_units_vs_connection_p3")
p4 = dm.load_data(base_path + "hidden_units_vs_connection_p4")
p5 = dm.load_data(base_path + "hidden_units_vs_connection_p5")
p6 = dm.load_data(base_path + "hidden_units_vs_connection_p6")
p7 = dm.load_data(base_path + "hidden_units_vs_connection_p7")
p8 = dm.load_data(base_path + "hidden_units_vs_connection_p8")

data = [p1, p2, p8, p3, p7, p4, p5, p6]

y = [[], [], [], [], [], []]
c = [[], [], [], [], [], []]
s = [[], [], [], [], [], []]

x = []
z = []

p_i = 0
acc = 94
cap = 20000
for n in range(0, len(p1['simulation'])):
    x.append(data[0]['simulation'][n]['network_parameters']['layers'][1])
    for p in range(0, len(data)):
        for i in range(0, len(data[p]['simulation'][n]['results']['accuracy'])):
            if data[p]['simulation'][n]['results']['accuracy'][i] > acc:
                y[n].append(np.asarray(data[p]['simulation'][n]['results']['energy'][i]))
                c[n].append(np.asarray(data[p]['simulation'][n]['results']['samples_seen'][i]))
                break
            else:
                if i == len(data[p]['simulation'][n]['results']['accuracy'])-1:
                    y[n].append(cap)
                    c[n].append(np.asarray(data[p]['simulation'][n]['results']['samples_seen'][-1]))
                #y[n].append([0])
                
        #y[n].append(np.asarray(data[p]['simulation'][n]['results']['energy'][-1]))
        #c[n].append(np.asarray(data[p]['simulation'][n]['results']['accuracy'][-1]))
        s[n].append(data[p]['simulation'][n]['results']['samples_seen'])
        if n == 0:
            z.append(data[p]['simulation'][0]['network_parameters']['p_connect'][0])

C = np.asarray(c)
minn, maxx = C.min(), C.max()
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='seismic')
m.set_array([])
fcolors = m.to_rgba(C.T)

Y = np.asarray(y)      
X, Z = np.meshgrid(x,z)


ax = plt.axes(projection='3d')
ax.plot_surface(X, Z, Y.T, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx)
ax.invert_xaxis()

cb = plt.colorbar(m)
plt.title("Energy @ Accuracy = " + str(acc) + "%\n(N/A values capped at " + str(cap) + " energy", fontsize=20)
cb.set_label("no. of samples", fontsize=16)
ax.set_xlabel("no. of hidden units", fontsize=16)
ax.set_ylabel("connection probability", fontsize=16)
ax.set_zlabel("Energy", fontsize=16)
ax.tick_params(labelsize=14)
plt.show()


