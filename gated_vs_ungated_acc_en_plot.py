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

def RGB(R, G, B):
    return (R/255, G/255, B/255)

def unitRGB(units, lineQ):
    if units == 100:
        if lineQ:
            c = RGB(255, 85, 0)
        else:
            c = RGB(255, 158, 110)
    elif units == 50:
        if lineQ:
            c = RGB(255, 149, 0)
        else:
            c = RGB(255, 195, 110)
    elif units == 150:
        if lineQ:
            c = RGB(255, 21, 0)
        else:
            c = RGB(255, 122, 110)
    elif units == 300:
        if lineQ:
            c = RGB(65, 97, 255)
        else:
            c = RGB(159, 183, 255)
    elif units == 250:
        if lineQ:
            c = RGB(144, 65, 255)
        else:
            c = RGB(196, 159, 255)              
    elif units == 200:
        if lineQ:
            c = RGB(65, 255, 223)
        else:
            c = RGB(159, 255, 231)
    else:
        if lineQ:
            c = RGB(31, 31, 31)
        else:
            c = RGB(158, 158, 158)   
    return c

def swapUnits(u):
    if u == 300:
        u = 200
    elif u == 250:
        u = 150
    elif u == 200:
        u == 100
    elif u == 150:
        u = 50
    elif u == 100:
        u = 300
    elif u == 50:
        u = 250
    return u

n_units = 50
figure_path = 'data/figures/gated/' + str(n_units) + '/'
filenames = []

accs = np.linspace(92.7, 97.5, 500) #np.flip(np.geomspace(97.5, 92.7, 500))


base_path = 'data/'
file_path = 'ungated/' + str(n_units) + '_units_various_p'
data = dm.load_data(base_path+file_path)
if not len(data) > 1:
    data = data[0]



#plt.yscale('log')
marker_size = 25
line_size = 5
window_size = 2

acc = 96
energies = []
p_connects = []
subenergies = []

base_path = 'data/'
file_path = 'ungated/' + str(n_units) + '_units_various_p'
data = dm.load_data(base_path+file_path)
if not len(data) > 1:
    data = data[0]
for i in range(0, len(data)):
    if data[i]['network_parameters']['p_connect'][0] == 1.0:
        full_a = data[i]['results']['accuracy']
        full_e = data[i]['results']['energy']

base_path = 'data/'
file_path = 'gated/' + str(n_units) + '_units_gated_various_p'
data = dm.load_data(base_path+file_path)
if not len(data) > 1:
    data = data[0]
for i in range(0, len(data)):
    fig, ax = plt.subplots()
    fig.set_size_inches(16, 8)
    a = data[i]['results']['accuracy']
    e = data[i]['results']['energy']
    l = str(data[i]['network_parameters']['p_connect'][0])

        
    # p_connects.append(data[i]['network_parameters']['p_connect'][0])
    # for j in range(0, len(data[i]['results']['energy'])):
    #     if data[i]['results']['accuracy'][j] > acc:
    #         energies.append(data[i]['results']['energy'][j])
    #         subenergies.append(data[i]['results']['energy'][-1])
    #         break
    #     else:
    #         if j == len(data[i]['results']['energy'])-1:
    #             energies.append(0)#data[i]['results']['energy'][-1])
    #             subenergies.append(data[i]['results']['energy'][-1])
                
        #plt.plot(full_a, full_e, linestyle='None', marker='.', markersize=marker_size, color=unitRGB(0, False), label='_nolegend_')
    h = plt.plot(full_a, full_e, linewidth=line_size, color=unitRGB(0, True))
    
    #plt.plot(a, e, linestyle='None', marker='.', markersize=marker_size, color=unitRGB(200, False), label='_nolegend_')
    h = plt.plot(a, e, window_size, linewidth=line_size, color=unitRGB(swapUnits(n_units), True))
     
    filename = str(n_units) + '_gated_accuracy_vs_energy_' + l.replace('.', '_')
    plt.legend(['1.0', l], fontsize=22, loc='lower right')
    plt.title('Energy vs. Accuracy with ' + str(n_units) + ' units @ ' + l, fontsize=36)
    plt.xlabel('Accuracy /%', fontsize=28)
    plt.xticks(fontsize=24)
    plt.ylabel('Energy /A.U.', fontsize=28)
    plt.yscale('log')
    plt.yticks(fontsize=24)
    plt.xlim([80, 100])
    plt.ylim([2000, 200000])

    plt.savefig(figure_path+filename)
    plt.close()





 