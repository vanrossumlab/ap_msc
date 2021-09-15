import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from components import DataManager as dm

def mov_mean(x, N):
    r = np.convolve(x, np.ones((N,))/N)[(N-1):]
    r[-1] = x[-1]
    for i in range(0, int(np.ceil(N))):
        r[-i] = x[-i]
    return r

def RGB(R, G, B, A):
    return (R/255, G/255, B/255, A/255)

def unitRGB(units, lineQ, opacity=255):
    if units == 100:
        if lineQ:
            c = RGB(255, 21, 255, opacity)
        else:
            c = RGB(255, 160, 240, opacity)
    elif units == 50:
        if lineQ:
            c = RGB(255, 149, 0, opacity)
        else:
            c = RGB(255, 195, 110, opacity)
    elif units == 150:
        if lineQ:
            c = RGB(255, 21, 0, opacity)
        else:
            c = RGB(255, 122, 110, opacity)
    elif units == 300:
        if lineQ:
            c = RGB(65, 97, 255, opacity)
        else:
            c = RGB(159, 183, 255, opacity)
    elif units == 250:
        if lineQ:
            c = RGB(144, 65, 255, opacity)
        else:
            c = RGB(196, 159, 255, opacity)              
    elif units == 200:
        if lineQ:
            c = RGB(46, 203, 255, opacity)
        else:
            c = RGB(157, 234, 255, opacity)
    else:
        if lineQ:
            c = RGB(31, 31, 31, opacity)
        else:
            c = RGB(158, 158, 158, opacity)   
    return c

n_units = 300
figure_path = 'data/figures/gated/'
filenames = []

accs = np.linspace(92.7, 97.5, 500) #np.flip(np.geomspace(97.5, 92.7, 500))


base_path = 'data/'
file_path = 'ungated/' + str(n_units) + '_units_various_p'
data = dm.load_data(base_path+file_path)
if not len(data) > 1:
    data = data[0]
marker_size = 25
line_size = 5
window_size = 2

acc = 95
energies = []
p_connects = []
subenergies = []

for i in range(0, len(data)):
    if data[i]['network_parameters']['p_connect'][0] == 1.0:
        full_a = data[i]['results']['accuracy']
        full_e = data[i]['results']['energy']
        full_min_e = data[i]['results']['min_energy']

fig, ax = plt.subplots()
fig.set_size_inches(16, 8)
all_units = [50, 100, 150, 200, 250, 300]
labs = []
for n_units in all_units:
    base_path = 'data/'
    file_path = 'gated/' + str(n_units) + '_units_gated_various_p'
    data = dm.load_data(base_path+file_path)
    if not len(data) > 1:
        data = data[0]

    for i in range(0, len(data)):
        if data[i]['network_parameters']['p_connect'][0] == 1.0:
            a = data[i]['results']['accuracy']
            e = data[i]['results']['energy']
            labs.append(str(n_units))
            plt.plot(a, e, linewidth=line_size, color=unitRGB(n_units, True, 255))

    # base_path = 'data/'
    # file_path = 'ungated/' + str(n_units) + '_units_various_p'
    # data = dm.load_data(base_path+file_path)
    # if not len(data) > 1:
    #     data = data[0]

    # for i in range(0, len(data)):
    #     if data[i]['network_parameters']['p_connect'][0] == 1.0:
    #         a = data[i]['results']['accuracy']
    #         e = data[i]['results']['energy']
    #         labs.append(str(n_units))
    #         plt.plot(a, e, linewidth=line_size, color=unitRGB(n_units, False, 175))
         
plt.legend(labs, fontsize=22, loc='lower right')
plt.title('Gated: Energy vs. Accuracy with All Networks', fontsize=36)
plt.xlabel('Accuracy /%', fontsize=28)
plt.xticks(fontsize=24)
plt.ylabel('Energy /A.U.', fontsize=28)
plt.yscale('log')
plt.yticks(fontsize=24)
plt.xlim([92, 97])
plt.ylim([10000, 100000])
plt.show()

# filename = 'gated_all_accuracy_vs_energy2.svg'
# plt.savefig(figure_path+filename)
# filename = 'gated_all_accuracy_vs_energy2.png'
# plt.savefig(figure_path+filename)
# plt.close()





 