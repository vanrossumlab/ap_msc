import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
            c = RGB(255, 21, 255)
        else:
            c = RGB(255, 160, 240)
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
            c = RGB(46, 203, 255)
        else:
            c = RGB(157, 234, 255)
    else:
        if lineQ:
            c = RGB(31, 31, 31)
        else:
            c = RGB(158, 158, 158)   
    return c

all_units = [50, 100, 150, 200, 250, 300]
u = 300
labs = []

fig, ax = plt.subplots()
ax2 = ax.twinx()
fig.set_size_inches(16, 8)
marker_size = 25
line_size = 4
for n_units in all_units:
    base_path = 'data/'
    file_path = 'ungated/' + str(n_units) + '_units_various_p'
    data = dm.load_data(base_path+file_path)
    if not len(data) > 1:
        data = data[0]
    
    p_s = []
    n = 0
    p = 1.0
    for j in range(0, len(data)):
        for i in range(0, len(data)):
            if n_units == u and int(p*100) == int(data[i]['network_parameters']['p_connect'][0]*100): #data[i]['network_parameters']['p_connect'][0] == 1.0: 
                a = data[i]['results']['accuracy']
                e = data[i]['results']['energy']
                err = data[i]['results']['error']
                s = data[i]['results']['samples_seen']
                s = np.asarray(s)/60000
                p_s.append(data[i]['network_parameters']['p_connect'][0])
                labs.append(data[i]['network_parameters']['p_connect'][0])
                ax.plot(s, err, color=unitRGB(n, False), linewidth=line_size)
                ax2.plot(s, a, color=unitRGB(n, True), linewidth=line_size)
                p = p - 0.2
                n = n + 50
                break

plt.rc('font', size=24)
ax.set_yscale('log')
ax2.set_ylim([84, 100])
ax.set_ylim([0.05, 2])
plt.xlim([0, 30])
plt.title('Ungated ' + str(u) + ' Units: Testing Accuracy & Error Curves\n@ Various Synaptic Densities', fontsize=36)
plt.xlabel('Samples', fontsize=28)
ax.set_ylabel('Cross Entropy Loss', fontsize=28)
ax2.set_ylabel('Accuracy /%', fontsize=28)
ax.set_xlabel('Samples (Epochs)', fontsize=28)
ax.xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ax.tick_params(labelsize=24)
ax.tick_params(which='minor', labelsize=16)
ax2.tick_params(labelsize=24)
ax.tick_params(which='major', labelsize=24)
plt.yticks(fontsize=24)
ax2.legend(labs, fontsize=22, loc='upper left')
t = ax.xaxis.get_offset_text()
t.set_size(20)
# figure_path = 'data/figures/ungated/'
# filename = 'ungated_' + str(u) + '_accuracy_error_curves.svg'
# fig.tight_layout()
# plt.savefig(figure_path+filename, bbox_inches='tight')
# filename = 'ungated_' + str(u) + '_accuracy_vs_energy.png'
# plt.savefig(figure_path+filename, bbox_inches='tight')
plt.show()
        




# plt.legend(['Ungated: 1.0', 'Ungated: '+l], fontsize=22, loc='lower right')
# plt.title('Energy vs. Accuracy with ' + str(n_units) + ' units @ ' + l, fontsize=36)
# plt.xlabel('Accuracy /%', fontsize=28)
# plt.xticks(fontsize=24)
# plt.ylabel('Energy /A.U.', fontsize=28)
# plt.yticks(fontsize=24)
# plt.xlim([84, 100])
# plt.ylim([2000, 250000])