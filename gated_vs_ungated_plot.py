import numpy as np
import matplotlib.pyplot as plt
from components import DataManager as dm

def mov_mean(x, N):
    r = np.convolve(x, np.ones((N,))/N)[(N-1):]
    r[-1] = x[-1]
    for i in range(0, int(np.ceil(N))):
        r[-i] = x[-i]
    return r

def RGB(R, G, B):
    return (R/255, G/255, B/255)

base_path = 'data/'
file_path = 'ungated/250_units_various_p'

data = dm.load_data(base_path+file_path)
if not len(data) > 1:
    data = data[0]

energies = []
p_connects = []

fig, ax = plt.subplots()
fig.set_size_inches(16, 9)
marker_size = 25
line_size = 5
window_size = 3

acc = 92
for i in range(0, len(data)):
    p_connects.append(data[i]['network_parameters']['p_connect'][0])
    for j in range(0, len(data[i]['results']['energy'])):
        if data[i]['results']['accuracy'][j] > acc:
            energies.append(data[i]['results']['energy'][j])
            break
        else:
            if j == len(data[i]['results']['energy'])-1:
                energies.append(0)

order = np.argsort(p_connects)
energies = np.array(energies)[order]
p_connects = np.array(p_connects)[order]       
                
energies = np.where(energies == 0, np.max(energies), energies)
        
offset = -7


p1 = p_connects[np.where(energies == np.min(energies))]
e1 = energies[np.where(energies == np.min(energies))]
e_max = np.partition(energies, offset)[offset]
energy_max = np.max(energies)

plt.plot(p_connects, energies, linestyle='None', marker='.', markersize=marker_size, color=RGB(161, 159, 255))
plt.plot(p_connects, mov_mean(energies, window_size), linewidth=line_size, color=RGB(48, 37, 255))
plt.vlines(p1, e1, np.partition(energies, offset)[offset], linestyle='dashed')
ax.annotate(xy=(p_connects[np.where(energies == np.min(energies))], np.partition(energies, offset)[offset]), xytext=(-30,10), textcoords='offset points', text=str(np.around((1-np.min(energies)/np.max(energies))*100, 1))+"% Saving" , va='center', fontsize=16)

base_path = 'data/'
file_path = 'gated/250_units_gated_various_p'

data = dm.load_data(base_path+file_path)
if not len(data) > 1:
    data = data[0]

energies = []
p_connects = []

for i in range(0, len(data)):
    p_connects.append(data[i]['network_parameters']['p_connect'][0])
    for j in range(0, len(data[i]['results']['energy'])):
        if data[i]['results']['accuracy'][j] > acc:
            energies.append(data[i]['results']['energy'][j])
            break
        else:
            if j == len(data[i]['results']['energy'])-1:
                energies.append(0)

order = np.argsort(p_connects)
energies = np.array(energies)[order]
p_connects = np.array(p_connects)[order]       
                
energies = np.where(energies == 0, np.max(energies), energies)
        
offset = -7
        
plt.plot(p_connects, energies, linestyle='None', marker='.', markersize=marker_size, color=RGB(209, 159, 255))
plt.plot(p_connects, mov_mean(energies, window_size), linewidth=line_size, color=RGB(157, 37, 255))
plt.vlines(p_connects[np.where(energies == np.min(energies))], energies[np.where(energies == np.min(energies))], np.partition(energies, offset)[offset], linestyle='dashed')
ax.annotate(xy=(p_connects[np.where(energies == np.min(energies))], np.partition(energies, offset)[offset],), xytext=(-30,10), textcoords='offset points', text=str(np.around((1-np.min(energies)/energy_max)*100, 1))+"% Saving" , va='center', fontsize=16)
plt.legend(['ungated', 'ungated - mean', 'gated', 'gated - mean'], fontsize=18)
plt.title('Hidden Units: Synaptic Density vs. Energy @ ' + str(acc) + "% Accuracy", fontsize=32)
plt.xlabel('Probability of Connection', fontsize=24)
plt.xticks(fontsize=16)
plt.ylabel('Energy', fontsize=24)
plt.yticks(fontsize=16)
plt.show()


 